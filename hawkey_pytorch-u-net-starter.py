import os

import gc

import sys

import cv2

import glob

import time

import signal

import numpy as np

import pandas as pd



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import Dataset, DataLoader



import matplotlib.pyplot as plt

import tensorboardX

from tqdm.notebook import tqdm



from collections import OrderedDict

from sklearn import model_selection



seed = 42

test_size = 0.2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

workspace = "./unet"



backbone = [1, 1, 1, 1]

encoder_channels = np.array([64, 128, 256, 512, 1024])*2

decoder_channels = np.array([512, 256, 128, 64])*2



fold = 0

time_step = 4000 # a continuous batch is 500000!

time_step_test = 10000

stride = 2 # for 4 times

batch_size = 8



TRAIN = True

PREDICT = True

ENSEMBLE = False



np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed_all(seed)

#torch.backends.cudnn.deterministic = True

#torch.backends.cudnn.benchmark = True





print(device)
# data loading 



df_train = pd.read_csv("../input/liverpool-ion-switching/train.csv")

df_test = pd.read_csv("../input/liverpool-ion-switching/test.csv")
# possible augmentations?

# https://github.com/iver56/audiomentations



def do_identity(image, label):

    return image, label



def do_flip(image, label):

    image = np.ascontiguousarray(np.flip(image, axis=1))

    label = np.ascontiguousarray(np.flip(label))

    return image, label



def train_augment(image, label):

    for op in np.random.choice([

        lambda image, label: do_identity(image, label),

        lambda image, label: do_flip(image, label),

    ], 1):

        image, label = op(image, label)

    

    return image, label



def valid_augment(image, label):

    return image, label
class IonDataset(Dataset):

    def __init__(self, data, labels=None, type='train', transform=None):

        self.data = data

        self.labels = labels

        self.type = type

        self.transform = transform

        

    def __getitem__(self, i):

        signal = self.data[i].astype(np.float32) # [1, time_step]

        if self.type == 'train':

            label = self.labels[i].astype(np.int64) # [time_step]

            if self.transform is not None:

                signal, label = self.transform(signal, label)

            return signal, label

        else:

            return signal

    

    def __len__(self):

        return len(self.data)
class SEModule(nn.Module):

    def __init__(self, in_channels, reduction=4):

        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, in_channels//reduction, kernel_size=1, padding=0)

        self.conv2 = nn.Conv1d(in_channels//reduction, in_channels, kernel_size=1, padding=0)

        

    def forward(self, x):

        # x: [B, C, H]

        s = F.adaptive_avg_pool1d(x, 1) # [B, C, 1]

        s = self.conv1(s) # [B, C//reduction, 1]

        s = F.relu(s, inplace=True)

        s = self.conv2(s) # [B, C, 1]

        x = x + torch.sigmoid(s)

        return x



class ConvBR1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, dilation=1, stride=1, groups=1, is_activation=True):

        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride, groups=groups, bias=False)

        self.bn = nn.BatchNorm1d(out_channels)

        self.is_activation = is_activation

        

        if is_activation:

            self.relu = nn.ReLU(inplace=True)

        

    def forward(self, x):

        x = self.bn(self.conv(x))

        if self.is_activation:

            x = self.relu(x)

        return x





class SENextBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=32, reduction=16, pool=None, is_shortcut=False):

        super().__init__()

        mid_channels = out_channels // 2

        self.conv1 = ConvBR1d(in_channels, mid_channels, 1, 0, 1, )

        self.conv2 = ConvBR1d(mid_channels, mid_channels, 3, 1, 1, groups=groups)

        self.conv3 = ConvBR1d(mid_channels, out_channels, 1, 0, 1, is_activation=False)

        self.se = SEModule(out_channels, reduction)

        self.stride = stride

        self.is_shortcut = is_shortcut

        

        if is_shortcut:

            self.shortcut = ConvBR1d(in_channels, out_channels, 1, 0, 1, is_activation=False)

        if stride > 1:

            if pool == 'max':

                self.pool = nn.MaxPool1d(stride, stride)

            elif pool == 'avg':

                self.pool = nn.AvgPool1d(stride, stride)

    

    def forward(self, x):

        s = self.conv1(x)

        s = self.conv2(s)

        if self.stride > 1:

            s = self.pool(s)

        s = self.conv3(s)

        s = self.se(s)

        

        if self.is_shortcut:

            if self.stride > 1:

                x = F.avg_pool1d(x, self.stride, self.stride) # avg

            x = self.shortcut(x)

        

        x = x + s

        x = F.relu(x, inplace=True)

        

        return x





class Encoder(nn.Module):

    def __init__(self, num_features=1):

        super().__init__()

        self.block0 = nn.Sequential(

            ConvBR1d(num_features, encoder_channels[0], kernel_size=5, stride=1, padding=2),

            ConvBR1d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),

            ConvBR1d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),

        )

        self.block1 = nn.Sequential(

            SENextBottleneck(encoder_channels[0], encoder_channels[1], stride=stride, is_shortcut=True, pool='max'),

          *[SENextBottleneck(encoder_channels[1], encoder_channels[1], stride=1, is_shortcut=False) for i in range(backbone[0])]

        )

        self.block2 = nn.Sequential(

            SENextBottleneck(encoder_channels[1], encoder_channels[2], stride=stride, is_shortcut=True, pool='max'),

          *[SENextBottleneck(encoder_channels[2], encoder_channels[2], stride=1, is_shortcut=False) for i in range(backbone[1])]

        )

        self.block3 = nn.Sequential(

            SENextBottleneck(encoder_channels[2], encoder_channels[3], stride=stride, is_shortcut=True, pool='max'),

          *[SENextBottleneck(encoder_channels[3], encoder_channels[3], stride=1, is_shortcut=False) for i in range(backbone[2])]

        )

        self.block4 = nn.Sequential(

            SENextBottleneck(encoder_channels[3], encoder_channels[4], stride=stride, is_shortcut=True, pool='avg'),

          *[SENextBottleneck(encoder_channels[4], encoder_channels[4], stride=1, is_shortcut=False) for i in range(backbone[3])]

        )  

        

    def forward(self, x):

        x0 = self.block0(x) # [B, 64, L]

        x1 = self.block1(x0) # [B, 256, L//2]

        x2 = self.block2(x1) # [B, 512, L//4]

        x3 = self.block3(x2) # [B, 1024, L//8]

        x4 = self.block4(x3) # [B, 2048, L//16]

        

        return [x0, x1, x2, x3, x4]



class SCSEModule(nn.Module):

    def __init__(self, in_channels, reduction=16):

        super().__init__()

        self.cSE = nn.Sequential(

            nn.AdaptiveAvgPool1d(1),

            nn.Conv1d(in_channels, in_channels // reduction, 1),

            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels // reduction, in_channels, 1),

            nn.Sigmoid(),

        )

        self.sSE = nn.Sequential(nn.Conv1d(in_channels, 1, 1), nn.Sigmoid())



    def forward(self, x):

        return x * self.cSE(x) + x * self.sSE(x)



class DecoderBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):

        super().__init__()

        self.conv1 = ConvBR1d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)

        self.conv2 = ConvBR1d(out_channels, out_channels, kernel_size=3, padding=1)

        # att

        #self.att1 = SCSEModule(in_channels + skip_channels)

        #self.att2 = SCSEModule(out_channels)

        

    def forward(self, x, skip=None):

        x = F.interpolate(x, scale_factor=stride, mode="linear", align_corners=True)

        if skip is not None:

            x = torch.cat([x, skip], dim=1)

            #x = self.att1(x)

        x = self.conv1(x)

        x = self.conv2(x)

        #x = self.att2(x)

        return x

    

class Decoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.block4 = DecoderBlock(encoder_channels[-1], encoder_channels[-2], decoder_channels[0])

        self.block3 = DecoderBlock(decoder_channels[0], encoder_channels[-3], decoder_channels[1])

        self.block2 = DecoderBlock(decoder_channels[1], encoder_channels[-4], decoder_channels[2])

        self.block1 = DecoderBlock(decoder_channels[2], encoder_channels[-5], decoder_channels[3])



    def forward(self, xs):



        x = self.block4(xs[4], xs[3])

        x = self.block3(x, xs[2])

        x = self.block2(x, xs[1])

        x = self.block1(x, xs[0])

        

        return x

        

    

class Unet(nn.Module):

    def __init__(self, num_features=1, num_classes=11):

        super().__init__()

        self.num_features = num_features

        self.num_classes = num_classes

        self.encoder = Encoder(num_features=num_features)

        self.decoder = Decoder()

        self.segmentation_head = nn.Conv1d(decoder_channels[-1], num_classes, kernel_size=1, padding=0, stride=1)

    

    def forward(self, x):

        features = self.encoder(x)

        x = self.decoder(features)

        x = self.segmentation_head(x)

        return x

# trainer

class Logger:

    def __init__(self, workspace=None, flush=True, mute=False):

        self.workspace = workspace

        self.flush = flush

        self.mute = mute

        if workspace is not None:

            os.makedirs(workspace, exist_ok=True)

            self.log_file = os.path.join(workspace, "log.txt")

            self.fp = open(self.log_file, "a+")

        else:

            self.fp = None



    def __del__(self):

        if self.fp: 

            self.fp.close()



    def _print(self, text, use_pprint=False):

        if not self.mute:

            print(text) if not use_pprint else pprint(text)

        if self.fp:

            print(text, file=self.fp)

        if self.flush:

            sys.stdout.flush()



    def log(self, text, level=0):

        text = "\t"*level + text

        text.replace("\n", "\n"+"\t"*level)

        self._print(text)



    def log1(self, text):

        self.log(text, level=1)



    def info(self, text):

        text = "[INFO] " + text

        text.replace("\n", "\n"+"[INFO] ")

        self._print(text)



    def error(self, text):

        text = "[ERROR] " + text

        text.replace("\n", "\n"+"[ERROR] ")

        self._print(text)





class DelayedKeyboardInterrupt(object):

    def __enter__(self):

        self.signal_received = False

        self.old_handler = signal.signal(signal.SIGINT, self.handler)



    def handler(self, sig, frame):

        self.signal_received = (sig, frame)

        print('SIGINT received. Delaying KeyboardInterrupt.')



    def __exit__(self, type, value, traceback):

        signal.signal(signal.SIGINT, self.old_handler)

        if self.signal_received: 

            self.old_handler(*self.signal_received)





def fix_random_seed(seed=42, cudnn=False):

    torch.manual_seed(seed)

    np.random.seed(seed)

    if cudnn:

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False





# from torch_summary

def summary(model, input_size, batch_size=-1, device="cuda", logger=None):

    # redirect to write in file

    if logger is not None:

        print = logger._print



    def register_hook(module):



        def hook(module, input, output):

            class_name = str(module.__class__).split(".")[-1].split("'")[0]

            module_idx = len(summary)



            m_key = "%s-%i" % (class_name, module_idx + 1)

            summary[m_key] = OrderedDict()

            summary[m_key]["input_shape"] = list(input[0].size())

            summary[m_key]["input_shape"][0] = batch_size

            if isinstance(output, (list, tuple)):

                summary[m_key]["output_shape"] = [

                    [-1] + list(o.size())[1:] for o in output

                ]

            else:

                summary[m_key]["output_shape"] = list(output.size())

                summary[m_key]["output_shape"][0] = batch_size



            params = 0

            if hasattr(module, "weight") and hasattr(module.weight, "size"):

                params += torch.prod(torch.LongTensor(list(module.weight.size())))

                summary[m_key]["trainable"] = module.weight.requires_grad

            if hasattr(module, "bias") and hasattr(module.bias, "size"):

                params += torch.prod(torch.LongTensor(list(module.bias.size())))

            summary[m_key]["nb_params"] = params



        if (

            not isinstance(module, nn.Sequential)

            and not isinstance(module, nn.ModuleList)

            and not (module == model)

        ):

            hooks.append(module.register_forward_hook(hook))



    device = device.lower()

    assert device in [

        "cuda",

        "cpu",

    ], "Input device is not valid, please specify 'cuda' or 'cpu'"



    if device == "cuda" and torch.cuda.is_available():

        dtype = torch.cuda.FloatTensor

    else:

        dtype = torch.FloatTensor



    # multiple inputs to the network

    if isinstance(input_size, tuple):

        input_size = [input_size]



    # batch_size of 2 for batchnorm

    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # print(type(x[0]))



    # create properties

    summary = OrderedDict()

    hooks = []



    # register hook

    model.apply(register_hook)



    # make a forward pass

    # print(x.shape)

    model(*x)



    # remove these hooks

    for h in hooks:

        h.remove()



    print("----------------------------------------------------------------")

    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")

    print(line_new)

    print("================================================================")

    total_params = 0

    total_output = 0

    trainable_params = 0

    for layer in summary:

        # input_shape, output_shape, trainable, nb_params

        line_new = "{:>20}  {:>25} {:>15}".format(

            layer,

            str(summary[layer]["output_shape"]),

            "{0:,}".format(summary[layer]["nb_params"]),

        )

        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])

        if "trainable" in summary[layer]:

            if summary[layer]["trainable"] == True:

                trainable_params += summary[layer]["nb_params"]

        print(line_new)



    # assume 4 bytes/number (float on cuda).

    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))

    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients

    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))

    total_size = total_params_size + total_output_size + total_input_size



    print("================================================================")

    print("Total params: {0:,}".format(total_params))

    print("Trainable params: {0:,}".format(trainable_params))

    print("Non-trainable params: {0:,}".format(total_params - trainable_params))

    print("----------------------------------------------------------------")

    print("Input size (MB): %0.2f" % total_input_size)

    print("Forward/backward pass size (MB): %0.2f" % total_output_size)

    print("Params size (MB): %0.2f" % total_params_size)

    print("Estimated Total Size (MB): %0.2f" % total_size)

    print("----------------------------------------------------------------")

    # return summary

        



class Trainer(object):

    """Base trainer class. 

    """

    def __init__(self,

                 device,

                 workspace,

                 model, 

                 optimizer, 

                 lr_scheduler, 

                 objective, 

                 dataloaders,

                 metrics=[],

                 model_name=None,

                 input_shape=None,

                 use_checkpoint="latest",

                 use_tensorboardX=True,

                 max_keep_ckpt=1,

                 eval_interval=1,

                 report_step_interval=300,

                 restart=False,

                 ):

        

        self.device = device

        self.workspace_path = workspace

        self.model = model

        self.model_name = model_name

        self.restart = restart

        self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler

        self.objective = objective

        self.dataloaders = dataloaders

        self.metrics = metrics

        self.log = Logger(workspace)

        self.use_checkpoint = use_checkpoint

        self.max_keep_ckpt = max_keep_ckpt

        self.eval_interval = eval_interval

        self.report_step_interval = report_step_interval

        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        self.log.info(f'Time stamp is {self.time_stamp}')

        self.use_tensorboardX = use_tensorboardX

        self.writer = None



        self.model.to(self.device)



        if input_shape is not None:

            summary(self.model, input_shape, logger=self.log)



        self.log.info(f'Number of model parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')



        self.epoch = 1

        self.global_step = 0

        self.local_step = 0

        self.stats = {

            "EvalResults": [],

            "Checkpoints": [],

            "BestResult": None,

            }



        if self.workspace_path is not None:

            os.makedirs(self.workspace_path, exist_ok=True)

            if self.use_checkpoint == "latest":

                self.log.info("Loading latest checkpoint ...")

                self.load_checkpoint()

            elif self.use_checkpoint == "scratch":

                self.log.info("Train from scratch")

            elif self.use_checkpoint == "best":

                self.log.info("Loading best checkpoint ...")

                model_name = type(self.model).__name__

                best_path = f"{self.workspace_path}/{model_name}_best.pth.tar"

                self.load_checkpoint(best_path)

            else: # path to ckpt

                self.log.info(f"Loading checkpoint {self.use_checkpoint} ...")

                self.load_checkpoint(self.use_checkpoint)

    

    ### ------------------------------

    

    def train_step(self, data):

        image, mask = data



        output = self.model(image)

        loss = self.objective(output, mask)

        pred = F.softmax(output, 1).detach().cpu().numpy().argmax(axis=1)



        return pred, mask, loss 



    def eval_step(self, data):

        image, mask = data

        

        # tta

        output = (self.model(image) + torch.flip(self.model(torch.flip(image, dims=[2])), dims=[2]))/2

        #output = self.model(image)



        loss = self.objective(output, mask)

        pred = F.softmax(output, 1).detach().cpu().numpy().argmax(axis=1)



        return pred, mask, loss 



    ### ------------------------------



    def train(self, max_epochs=None):

        """

        do the training process for max_epochs.

        """

        if max_epochs is None:

            max_epochs = self.conf.max_epochs

        

        

        if self.use_tensorboardX:

            logdir = os.path.join(self.workspace_path, "run")

            self.writer = tensorboardX.SummaryWriter(logdir)

        

        for epoch in range(self.epoch, max_epochs+1):

            self.epoch = epoch



            if self.optimizer.param_groups[0]['lr'] < 1e-8:

                self.log.info("Early stopping.")

            

            self.train_one_epoch()



            if self.epoch % self.eval_interval == 0:

                self.evaluate_one_epoch()



                if self.workspace_path is not None:

                    self.save_checkpoint()

                    

        if self.use_tensorboardX:

            self.writer.close()



        self.log.info("Finished Training.")



    def evaluate(self):

        self.log.info(f"Evaluate at best epoch...")



        # load model

        model_name = type(self.model).__name__

        best_path = f"{self.workspace_path}/{model_name}_best.pth.tar"

        if not os.path.exists(best_path):

            self.log.error(f"Best checkpoint not found! {best_path}, not loading anything.")

        else:

            self.load_checkpoint(best_path)



        self.use_tensorboardX = False

        self.evaluate_one_epoch()

        

    def get_time(self):

        if torch.cuda.is_available(): 

            torch.cuda.synchronize()

        return time.time()



    def prepare_data(self, data):

        """ ToTensor for various data format """

        if isinstance(data, list) or isinstance(data, tuple):

            for i, v in enumerate(data):

                if isinstance(v, np.ndarray):

                    data[i] = torch.from_numpy(v).to(self.device)

                if torch.is_tensor(v):

                    data[i] = v.to(self.device)

        elif isinstance(data, dict):

            for k, v in data.items():

                if isinstance(v, np.ndarray):

                    data[k] = torch.from_numpy(v).to(self.device)

                if torch.is_tensor(v):

                    data[k] = v.to(self.device)

        elif isinstance(data, np.ndarray):

            data = torch.from_numpy(data).to(self.device)

        else: # is_tensor

            data = data.to(self.device)



        return data



    

    def train_one_epoch(self):

        self.log.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']} ...")



        for metric in self.metrics:

            metric.clear()

        total_loss = []

        self.model.train()



        pbar = tqdm(self.dataloaders["train"])



        self.local_step = 0

        epoch_start_time = self.get_time()                     

                     

        for data in pbar:

            start_time = self.get_time()

            self.local_step += 1

            self.global_step += 1

            

            data = self.prepare_data(data)



            preds, truths, loss = self.train_step(data)

            

            loss.backward()

            

            self.optimizer.step()

            self.optimizer.zero_grad()



            for metric in self.metrics:

                metric.update(preds, truths)



            total_loss.append(loss.item())

            total_time = self.get_time() - start_time



            if self.report_step_interval > 0 and self.local_step % self.report_step_interval == 0:

                self.log.log1(f"step={self.epoch}/{self.local_step}, loss={loss.item():.4f}, time={total_time:.2f}")

                for metric in self.metrics:

                    self.log.log1(metric.report())



            if self.use_tensorboardX:

                self.writer.add_scalar(f"train{self.model_name}/loss", loss.item(), self.global_step)



        if self.report_step_interval < 0:

            for metric in self.metrics:

                self.log.log1(metric.report())

                metric.clear()



        epoch_end_time = self.get_time()

        average_loss = np.mean(total_loss)



        self.log.log(f"==> Finished Epoch {self.epoch}, average_loss={average_loss:.4f}, time={epoch_end_time-epoch_start_time:.4f}")





    def evaluate_one_epoch(self):

        self.log.log(f"++> Evaluate at epoch {self.epoch} ...")



        for metric in self.metrics:

            metric.clear()

        self.model.eval()



        pbar = tqdm(self.dataloaders['valid'])



        epoch_start_time = self.get_time()

        total_loss = []



        with torch.no_grad():

            self.local_step = 0

            start_time = self.get_time()

            

            for data in pbar:    

                self.local_step += 1

                

                data = self.prepare_data(data)

                preds, truths, loss = self.eval_step(data)

                total_loss.append(loss.item())

                

                for metric in self.metrics:

                    metric.update(preds, truths)



            total_time = self.get_time() - start_time

            self.log.log1(f"total_time={total_time:.2f}")



            average_loss = np.mean(total_loss)

            

            for metric in self.metrics:

                self.log.log1(metric.report())

                if self.use_tensorboardX:

                    metric.write(self.writer, self.epoch, prefix=f"evaluate{self.model_name}")

                metric.clear()

            

            if self.use_tensorboardX:

                self.writer.add_scalar(f"evaluate{self.model_name}/loss", average_loss, self.epoch)

            

            self.stats["EvalResults"].append(self.metrics[0].measure())



        # monitor val loss!!!

        self.lr_scheduler.step(average_loss)



        epoch_end_time = self.get_time()

        self.log.log(f"++> Evaluate Finished. time={epoch_end_time-epoch_start_time:.4f}, loss={average_loss:.4f}")



    def save_checkpoint(self):

        with DelayedKeyboardInterrupt():

            model_name = type(self.model).__name__ if self.model_name is None else self.model_name

            file_path = f"{self.workspace_path}/{model_name}_ep{self.epoch:04d}.pth.tar"

            best_path = f"{self.workspace_path}/{model_name}_best.pth.tar"

            os.makedirs(self.workspace_path, exist_ok=True)



            self.stats["Checkpoints"].append(file_path)



            if len(self.stats["Checkpoints"]) > self.max_keep_ckpt:

                old_ckpt = self.stats["Checkpoints"].pop(0)

                if os.path.exists(old_ckpt):

                    os.remove(old_ckpt)

                    self.log.info(f"Removed old checkpoint {old_ckpt}")



            state = {

                'epoch': self.epoch,

                'global_step': self.global_step,

                'model_name': model_name,

                'model': self.model.state_dict(),

                'optimizer' : self.optimizer.state_dict(),

                'lr_scheduler': self.lr_scheduler.state_dict(),

                'stats' : self.stats,

            }

            

            torch.save(state, file_path)

            self.log.info(f"Saved checkpoint {self.epoch} successfully.")

            

            if self.stats["EvalResults"] is not None:

                ### better function

                if self.stats["BestResult"] is None or self.stats["EvalResults"][-1] > self.stats["BestResult"]:

                    self.stats["BestResult"] = self.stats["EvalResults"][-1]

                    torch.save(state, best_path)

                    self.log.info(f"Saved Best checkpoint.")

            



    def load_checkpoint(self, checkpoint=None):



        model_name = self.model_name if self.model_name is not None else type(self.model).__name__

        

        if checkpoint is None:

            # Load most recent checkpoint            

            checkpoint_list = sorted(glob.glob(f'{self.workspace_path}/{model_name}_ep*.pth.tar'))

            if checkpoint_list:

                checkpoint_path = checkpoint_list[-1]

            else:

                self.log.info("No checkpoint found, model randomly initialized.")

                return False

        elif isinstance(checkpoint, int):

            # Checkpoint is the epoch number

            checkpoint_path = f'{self.workspace_path}/{model_name}_ep{checkpoint:04d}.pth.tar'

        elif isinstance(checkpoint, str):

            # checkpoint is the path

            checkpoint_path = os.path.expanduser(checkpoint)

        else:

            self.log.error("load_checkpoint: Invalid argument")

            raise TypeError



        checkpoint_dict = torch.load(checkpoint_path)



        self.model.load_state_dict(checkpoint_dict['model'])

        if not self.restart:

            self.log.info("Loading epoch and other status...")

            self.epoch = checkpoint_dict['epoch'] + 1

            self.global_step = checkpoint_dict['global_step']

            self.optimizer.load_state_dict(checkpoint_dict['optimizer'])

            self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])

            self.lr_scheduler.last_epoch = checkpoint_dict['epoch'] 

            self.stats = checkpoint_dict['stats']

        else:

            self.log.info("Only loading model parameters.")

        

        self.log.info("Checkpoint Loaded Successfully.")

        return True





class Predictor(object):

    def __init__(self, device, models):

        self.device = device

        self.models = models

        self.log = Logger()

        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        self.log.info(f'Time stamp is {self.time_stamp}')

        for model in self.models:

            model.to(self.device)

            

    def predict_step(self, data):

        image = data



        outputs = []

        for model in self.models:



            # tta: flip

            output = (model(image) + torch.flip(model(torch.flip(image, dims=[2])), dims=[2]))/2

            #output = model(image)



            outputs.append(F.softmax(output, 1))

            

        pred = torch.mean(torch.stack(outputs, 0), 0).detach().cpu().numpy()

        pred = pred.argmax(axis=1).astype(np.int64) # [B, L]

        

        return pred



    def predict(self, dataloader):

        self.log.log(f"++> Predict start")

        # predict

        for model in self.models:

            model.eval()

        pbar = tqdm(dataloader)

        res = []

        epoch_start_time = self.get_time()

        with torch.no_grad():

            self.local_step = 0

            start_time = self.get_time()

            for data in pbar:    

                self.local_step += 1

                data = self.prepare_data(data)

                pred = self.predict_step(data)

                res.extend(pred.reshape(-1))

                

            total_time = self.get_time() - start_time

            self.log.log1(f"total_time={total_time:.2f}")

        epoch_end_time = self.get_time()

        self.log.log(f"++> Predict Finished. time={epoch_end_time-epoch_start_time:.4f}")

        

        return res

        

    def get_time(self):

        if torch.cuda.is_available(): 

            torch.cuda.synchronize()

        return time.time()



    def prepare_data(self, data):

        """ ToTensor for various data format """

        if isinstance(data, list) or isinstance(data, tuple):

            for i, v in enumerate(data):

                if isinstance(v, np.ndarray):

                    data[i] = torch.from_numpy(v).to(self.device)

                if torch.is_tensor(v):

                    data[i] = v.to(self.device)

        elif isinstance(data, dict):

            for k, v in data.items():

                if isinstance(v, np.ndarray):

                    data[k] = torch.from_numpy(v).to(self.device)

                if torch.is_tensor(v):

                    data[k] = v.to(self.device)

        elif isinstance(data, np.ndarray):

            data = torch.from_numpy(data).to(self.device)

        else: # is_tensor

            data = data.to(self.device)

        return data



    def load_checkpoint(self, checkpoint_path, idx=0):

        checkpoint = torch.load(checkpoint_path)

        self.models[idx].load_state_dict(checkpoint['model'])

        self.log.info("Checkpoint Loaded Successfully.")

        return True
class ClassificationMeter:

    """ statistics for classification """

    def __init__(self, nCls, eps=1e-5, names=None):

        self.nCls = nCls

        self.names = names

        self.eps = eps

        self.N = 0

        self.table = np.zeros((self.nCls, 4), dtype=np.int32)

        self._measure = None



    def clear(self):

        self.N = 0

        self.table = np.zeros((self.nCls, 4), dtype=np.int32)



    def prepare_inputs(self, outputs, truths):

        """

        outputs and truths are pytorch tensors or numpy ndarrays.

        """

        if torch.is_tensor(outputs):

            outputs = outputs.detach().cpu().numpy()

        if torch.is_tensor(truths):

            truths = truths.detach().cpu().numpy()

        

        return outputs, truths



    def update(self, preds, truths):

        preds, truths = self.prepare_inputs(preds, truths)

        self.N += np.prod(truths.shape)

        for Cls in range(self.nCls):

            true_positive = np.count_nonzero(np.bitwise_and(preds == Cls, truths == Cls))

            true_negative = np.count_nonzero(np.bitwise_and(preds != Cls, truths != Cls))

            false_positive = np.count_nonzero(np.bitwise_and(preds == Cls, truths != Cls))

            false_negative = np.count_nonzero(np.bitwise_and(preds != Cls, truths == Cls))

            self.table[Cls] += [true_positive, true_negative, false_positive, false_negative]



    # call after report() !

    def measure(self):

        return self._measure



    def better(self, A, B):

        return A > B



    def write(self, writer, global_step, prefix=""):

        writer.add_scalar(os.path.join(prefix, "Accuracy"), self.measure(), global_step)



    def report(self, each_class=False):

        precisions = []

        recalls = []

        f1s = []

        for Cls in range(self.nCls):

            recall = self.table[Cls,0] / (self.table[Cls,0] + self.table[Cls,3] + self.eps) # TP / (TP + FN)

            precision = self.table[Cls,0] / (self.table[Cls,0] + self.table[Cls,2] + self.eps) # TP / (TP + FP)

            f1 = 2 * precision * recall / (precision + recall + self.eps)

            precisions.append(precision)

            recalls.append(recall)

            f1s.append(f1)

            

        total_TP = np.sum(self.table[:, 0]) # all true positives 

        

        accuracy = total_TP/self.N

        accuracy_mean_class = np.mean(precisions)

        

        macro_f1 = np.mean(f1s)

        

        # use macro_f1 to measure performance

        self._measure = macro_f1

        

        text =    f"Macro      F1       = {macro_f1:.4f}\n"

        text += f"\tOverall    Accuracy = {accuracy:.4f}({total_TP}/{self.N})\n"

        text += f"\tMean-class Accuracy = {accuracy_mean_class:.4f}\n"

        

        if each_class:

            for Cls in range(self.nCls):

                #if precisions[Cls] != 0 or recalls[Cls] != 0:

                text += f"\tClass {str(Cls)+'('+self.names[Cls]+')' if self.names is not None else Cls}: precision = {precisions[Cls]:.3f} recall = {recalls[Cls]:.3f}\n"



        return text
def five_type_remove_drift(train):

    # CLEAN TRAIN BATCH 2

    a=500000; b=600000 

    train.loc[train.index[a:b],'signal'] = train.signal[a:b].values - 3*(train.time.values[a:b] - 50)/10.

    def f(x,low,high,mid): 

        return -((-low+high)/625)*(x-mid)**2+high -low

    # CLEAN TRAIN BATCH 7

    batch = 7; a = 500000*(batch-1); b = 500000*batch

    train.loc[train.index[a:b],'signal'] = train.signal.values[a:b] - f(train.time[a:b].values,-1.817,3.186,325)

    # CLEAN TRAIN BATCH 8

    batch = 8; a = 500000*(batch-1); b = 500000*batch

    train.loc[train.index[a:b],'signal'] = train.signal.values[a:b] - f(train.time[a:b].values,-0.094,4.936,375)

    # CLEAN TRAIN BATCH 9

    batch = 9; a = 500000*(batch-1); b = 500000*batch

    train.loc[train.index[a:b],'signal'] = train.signal.values[a:b] - f(train.time[a:b].values,1.715,6.689,425)

    # CLEAN TRAIN BATCH 10

    batch = 10; a = 500000*(batch-1); b = 500000*batch

    train.loc[train.index[a:b],'signal'] = train.signal.values[a:b] - f(train.time[a:b].values,3.361,8.45,475)

    return train
if TRAIN:

    df_train = five_type_remove_drift(df_train)

    # 1 slow

    batch = 1; a = 500000*(batch-1); b = 500000*batch

    batch = 2; c = 500000*(batch-1); d = 500000*batch

    X_train_1s = np.concatenate([df_train.signal.values[a:b],df_train.signal.values[c:d]]).reshape((-1,1))

    y_train_1s = np.concatenate([df_train.open_channels.values[a:b],df_train.open_channels.values[c:d]]).reshape((-1,1))

    # 1 fast

    batch = 3; a = 500000*(batch-1); b = 500000*batch

    batch = 7; c = 500000*(batch-1); d = 500000*batch

    X_train_1f = np.concatenate([df_train.signal.values[a:b],df_train.signal.values[c:d]]).reshape((-1,1))

    y_train_1f = np.concatenate([df_train.open_channels.values[a:b],df_train.open_channels.values[c:d]]).reshape((-1,1))

    # 3

    batch = 4; a = 500000*(batch-1); b = 500000*batch

    batch = 8; c = 500000*(batch-1); d = 500000*batch

    X_train_3 = np.concatenate([df_train.signal.values[a:b],df_train.signal.values[c:d]]).reshape((-1,1))

    y_train_3 = np.concatenate([df_train.open_channels.values[a:b],df_train.open_channels.values[c:d]]).reshape((-1,1))

    # 5

    batch = 6; a = 500000*(batch-1); b = 500000*batch

    batch = 9; c = 500000*(batch-1); d = 500000*batch

    X_train_5 = np.concatenate([df_train.signal.values[a:b],df_train.signal.values[c:d]]).reshape((-1,1))

    y_train_5 = np.concatenate([df_train.open_channels.values[a:b],df_train.open_channels.values[c:d]]).reshape((-1,1))

    # 10

    batch = 5; a = 500000*(batch-1); b = 500000*batch

    batch = 10; c = 500000*(batch-1); d = 500000*batch

    X_train_10 = np.concatenate([df_train.signal.values[a:b],df_train.signal.values[c:d]]).reshape((-1,1))

    y_train_10 = np.concatenate([df_train.open_channels.values[a:b],df_train.open_channels.values[c:d]]).reshape((-1,1))



    for X, y, num_classes, model_name, max_epoch in zip(

            [X_train_1s, X_train_1f, X_train_3, X_train_5, X_train_10], 

            [y_train_1s, y_train_1f, y_train_3, y_train_5, y_train_10],

            np.array([1, 1, 3, 5, 10])+1,

            ['1s', '1f', '3', '5', '10'],

            [30, 30, 60, 60, 120],

        ):



        X = X.reshape(-1, 1, time_step)

        y = y.reshape(-1, time_step)



        # split

        idx = np.arange(X.shape[0])

        kf = model_selection.KFold(5, shuffle=True, random_state=seed)

        train_idx, val_idx = list(kf.split(idx))[fold]

        #train_idx, val_idx = model_selection.train_test_split(idx, random_state=seed, test_size=test_size)

        

        print("model_name:", model_name)

        print("train dataset shape:", X[train_idx].shape, y[train_idx].shape)



        train_dataset = IonDataset(X[train_idx], y[train_idx], 'train', train_augment)

        valid_dataset = IonDataset(X[val_idx], y[val_idx], 'train', valid_augment)



        loaders = {

            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),

            'valid': DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),

        }

        

        model = Unet(num_classes=num_classes)



        loss_function = nn.CrossEntropyLoss()

        

        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

        #optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-5)

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, min_lr=1e-8)

        

        metrics = [ClassificationMeter(num_classes),]

        

        trainer = Trainer(device, workspace, model, optimizer, scheduler, loss_function, loaders, metrics,

                          model_name=model_name,

                          report_step_interval=-1,

                          #input_shape=(1, time_step),

                         )

        

        trainer.train(max_epoch)



if PREDICT:

    ### preprocess

    # REMOVE BATCH 1 DRIFT

    start=500

    a = 0; b = 100000

    df_test.loc[df_test.index[a:b],'signal'] = df_test.signal.values[a:b] - 3*(df_test.time.values[a:b]-start)/10.

    start=510

    a = 100000; b = 200000

    df_test.loc[df_test.index[a:b],'signal'] = df_test.signal.values[a:b] - 3*(df_test.time.values[a:b]-start)/10.

    start=540

    a = 400000; b = 500000

    df_test.loc[df_test.index[a:b],'signal'] = df_test.signal.values[a:b] - 3*(df_test.time.values[a:b]-start)/10.

    # REMOVE BATCH 2 DRIFT

    start=560

    a = 600000; b = 700000

    df_test.loc[df_test.index[a:b],'signal'] = df_test.signal.values[a:b] - 3*(df_test.time.values[a:b]-start)/10.

    start=570

    a = 700000; b = 800000

    df_test.loc[df_test.index[a:b],'signal'] = df_test.signal.values[a:b] - 3*(df_test.time.values[a:b]-start)/10.

    start=580

    a = 800000; b = 900000

    df_test.loc[df_test.index[a:b],'signal'] = df_test.signal.values[a:b] - 3*(df_test.time.values[a:b]-start)/10.

    # REMOVE BATCH 3 DRIFT

    def f(x):

        return -(0.00788)*(x-625)**2+2.345 +2.58

    a = 1000000; b = 1500000

    df_test.loc[df_test.index[a:b],'signal'] = df_test.signal.values[a:b] - f(df_test.time[a:b].values)

    

    ## predict

    sub = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv", dtype={'time':str})



    predictors = {}

    for num_classes, model_name in zip(

            np.array([1, 1, 3, 5, 10])+1,

            ['1s', '1f', '3', '5', '10']

        ):

        

        if not ENSEMBLE:

            models = [Unet(num_classes=num_classes),]

            predictor = Predictor(device, models)

            predictor.load_checkpoint(os.path.join(workspace, f"{model_name}_best.pth.tar"), 0)

        else:

            # ensemble happens here

            models = [

                    Unet(num_classes=num_classes),

                    #Unet(num_classes=num_classes),

                    #Unet(num_classes=num_classes),

                    #Unet(num_classes=num_classes),

                    #Unet(num_classes=num_classes),

                ]

            predictor = Predictor(device, models)

            predictor.load_checkpoint(os.path.join("workspace/unet-wide-4k-f0", f"{model_name}_best.pth.tar"), 0)

            #predictor.load_checkpoint(os.path.join("workspace/unet-wide-4k-f1", f"{model_name}_best.pth.tar"), 1)

            #predictor.load_checkpoint(os.path.join("workspace/unet-4k-f2", f"{model_name}_best.pth.tar"), 2)

            #predictor.load_checkpoint(os.path.join("workspace/unet-4k-f3", f"{model_name}_best.pth.tar"), 3)

            #predictor.load_checkpoint(os.path.join("workspace/unet-4k-f4", f"{model_name}_best.pth.tar"), 4)



        predictors[model_name] = predictor



    for start, end, model_type in zip(

            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])*100000,

            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20])*100000,

            ['1s', '3', '5', '1s', '1f', '10', '5', '10', '1s', '3', '1s'],

        ):

        X = df_test.signal.values[start:end].reshape(-1, 1, time_step_test)

        test_dataset = IonDataset(X, type='test')

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        res = predictors[model_type].predict(test_loader)

        sub.iloc[start:end,1] = res





    #sub.to_csv(os.path.join(workspace, "pred.csv"), index=False)

    sub.to_csv("pred.csv", index=False)

    

    

    plt.figure(figsize=(20,5))

    res = 1000

    let = ['A','B','C','D','E','F','G','H','I','J']

    plt.plot(range(0,df_test.shape[0],res),sub.open_channels[0::res])

    for i in range(5): plt.plot([i*500000,i*500000],[-5,12.5],'r')

    for i in range(21): plt.plot([i*100000,i*100000],[-5,12.5],'r:')

    for k in range(4): plt.text(k*500000+250000,10,str(k+1),size=20)

    for k in range(10): plt.text(k*100000+40000,7.5,let[k],size=16)

    plt.title('Test Data Predictions',size=16)

    plt.show()

    