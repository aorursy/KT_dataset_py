!pip install prettyprinter -q
import os

import sys

import numpy as np

import pandas as pd

import random

from pathlib import Path

from typing import Union, List, Callable

import logging

from queue import Queue

from tqdm import tqdm

from sklearn.metrics import f1_score

from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO

from pprint import pformat

from prettyprinter import pprint, install_extras



import torch

import torch.nn as nn

from torch import Tensor

from dataclasses import dataclass, field

from torch.utils.data import Dataset, DataLoader

 

from PIL import Image

import torchvision

from torchvision import transforms, models



import matplotlib.pyplot as plt

%matplotlib inline



install_extras(

    include=[

        "dataclasses",

    ],

    warn_on_error=True

)
@dataclass

class Config:

    # paths

    base_dir: Path = Path(".").absolute().parent

    train_imgs_dir: Path = base_dir / "input/birdsong-log-mel-spectrograms"

    train_df: Path = train_imgs_dir / "train_all.csv"

    output_dir = base_dir / "working"

    checkpoint_dir = output_dir / "checkpoint_dir"

    

    def __post_init__(self):

        # create the directories if they don't exist

        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    

    # training

    seed: int = 100

    bs: int = 64

    num_epochs: int = 5

    lr: float = 0.003

    mixed_precision: bool = False

    opt_level: str = 'O1'

    step_scheduler: Callable = None

    device: str = 'cuda'

    # number of best models for saving

    num_best_models: int = 3

        

    # transforming

    mean: List[float] = field(default_factory=list)

    std: List[float] = field(default_factory=list)

    max_freqmask_width: int = 15

    max_timemask_width: int = 15

    timemask_p: float = 0.4

    freqmask_p: float = 0.4

        

        

config = Config()

config.mean += [0.485, 0.456, 0.406]

config.std += [0.229, 0.224, 0.225]
def create_logger():

    log_file = (config.checkpoint_dir / "logfile.txt").as_posix()



    # logger

    logger_ = getLogger(log_file)

    logger_.setLevel(INFO)



    # formatter

    fmr = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")



    # file handler

    fh = FileHandler(log_file)

    fh.setLevel(INFO)

    fh.setFormatter(fmr)



    # stream handler

    ch = StreamHandler()

    ch.setLevel(INFO)

    ch.setFormatter(fmr)



    logger_.addHandler(fh)

    logger_.addHandler(ch)

    

    return logger_



LOGGER = create_logger()
class AverageMeter(object):

    """Computes and stores the average and current value"""



    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count

        

def to_numpy(tensor: Union[Tensor, Image.Image, np.array]) -> np.ndarray:

    if type(tensor) == np.array or type(tensor) == np.ndarray:

        return np.array(tensor)

    elif type(tensor) == Image.Image:

        return np.array(tensor)

    elif type(tensor) == Tensor:

        return tensor.cpu().detach().numpy()

    else:

        raise ValueError(msg)

        

def seed_everything(seed):

        random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

        np.random.seed(seed)

        torch.manual_seed(seed)

        torch.cuda.manual_seed(seed)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = True

        

seed_everything(config.seed)
class FrequencyMask(object):

    def __init__(self, max_width=config.max_freqmask_width, 

                       use_mean=bool(random.randint(0, 1))):

        

        self.max_width = max_width

        self.use_mean = use_mean

        

    def __call__(self, tensor_obj):

        tensor = tensor_obj.detach().clone()

        start = random.randrange(0, tensor.shape[2])

        end = start + random.randrange(1, self.max_width)

       

        if self.use_mean:

            tensor[:, start:end, :] = tensor.mean()

        else:

            tensor[:, start:end, :] = 0

        return tensor #[C, H, W]

    

    def __repr__(self):

        format_string = self.__class__.__name__ + "(max_width="

        format_string += str(self.max_width) + ")"

        format_string += "use_mean=" + (str(self.use_mean) + ")")

        return format_string

    



class TimeMask(object):

    def __init__(self, max_width=config.max_timemask_width, 

                       use_mean=bool(random.randint(0, 1))):

        

        self.max_width = max_width

        self.use_mean = use_mean

        

    def __call__(self, tensor_obj):

        tensor = tensor_obj.detach().clone()

        start = random.randrange(0, tensor.shape[1])

        end = start + random.randrange(1, self.max_width)

        

        if self.use_mean:

            tensor[:, :, start:end] = tensor.mean()

        else:

            tensor[:, :, start:end] = 0

        return tensor

    

    def __repr__(self):

        format_string = self.__class__.__name__ + "(max_width="

        format_string += str(self.max_width) + ")"

        format_string += " use_mean=" + (str(self.use_mean) + ")")

        return format_string
class BSImageData(Dataset):

    def __init__(self, data, eval_fold=0, train=True):

        self.data = data

        

        if train:

            files = self.data[self.data.fold != eval_fold].reset_index(drop=True)

            # train transforms

            self.transforms = transforms.Compose([

                transforms.ToTensor(),

                transforms.RandomApply([FrequencyMask()], p=config.freqmask_p),

                transforms.RandomApply([TimeMask()], p=config.timemask_p),

                transforms.Normalize(

                    mean=config.mean,

                    std=config.std

                )

            ])

        else: 

            files = self.data[self.data.fold == eval_fold].reset_index(drop=True)

            # eval transforms

            self.transforms = transforms.Compose([

                transforms.ToTensor(),

                transforms.Normalize(

                    mean=config.mean,

                    std=config.std

                )

            ])

            

        self.items = files["im_path"].values

        self.labels = files["ebird_code"].values

        self.length = len(self.items)

        

        

    def __getitem__(self, index):

        fname = self.items[index]

        label = self.labels[index]

        img = Image.open(fname)

        img = (np.array(img.convert('RGB')) / 255.).astype(np.float32)

        return (self.transforms(img), label)

            

    def __len__(self):

        return self.length
sample_transforms = [

    ( "Original", [transforms.ToTensor()]), 

    ( FrequencyMask(), [transforms.ToTensor(), FrequencyMask(use_mean=True)] ),

    ( TimeMask(), [transforms.ToTensor(), TimeMask(use_mean=True)] ) 

]



def vis_imgs():

    plt.figure(figsize=[20,14])

    for i in range(3):

        img = random.choice(list((config.train_imgs_dir / "fold_0" / "fold_0" / "aldfly").glob('*')))

        img = Image.open(img)

        img = transforms.Compose( sample_transforms[i][1] )(img)

        img = np.array(transforms.ToPILImage()(img))

        ax = plt.subplot(1, 3, i+1)

        ax.set_title(sample_transforms[i][0], fontsize=12)

        ax.imshow(img)

        ax.set_xlabel("Time")

        ax.set_ylabel("Hz")

        

    plt.suptitle("Mel Spectrograms with different masks", y=0.78,  fontsize=16)   

    plt.tight_layout()
vis_imgs()
def build_model():

    resnet = models.resnet50(pretrained=True)

    for param in resnet.parameters():

        param.requires_grad = False



    resnet.fc = nn.Sequential(nn.Linear(resnet.fc.in_features, 500),

                         nn.ReLU(),

                         nn.Dropout(), nn.Linear(500, 264))



    return resnet
def get_learning_rate(optimizer):

    for param_group in optimizer.param_groups:

        return param_group["lr"]



def train_fn(data_loader, model, criterion, optimizer, epoch, config, device=config.device):

    

    model.train()

    loss_handler = AverageMeter()

    score_handler = AverageMeter()



    pbar = tqdm(total=len(data_loader) * config.bs)

    pbar.set_description(

        " Epoch {}, lr: {:.2e}".format(epoch + 1, get_learning_rate(optimizer))

    )



    for i, (inputs, target) in enumerate(data_loader):



        optimizer.zero_grad()

        inputs = inputs.to(device)

        target = target.to(device)

        out = model(inputs)



        loss = criterion(out, target)



        if config.mixed_precision:

            with amp.scale_loss(loss, optimizer) as scaled_loss:

                scaled_loss.backward()

        else:

            loss.backward()

        

        optimizer.step()



        if config.step_scheduler:

            scheduler.step()



        loss_handler.update(loss.item())                         

        score_handler.update( f1_score(to_numpy(target), to_numpy(torch.argmax(out, dim=1)), average='micro') )



        current_lr = get_learning_rate(optimizer)

        batch_size = len(inputs)

        pbar.update(batch_size)

        pbar.set_postfix(loss=f"{loss_handler.avg:.5f}")



    pbar.close()

    return loss_handler, score_handler.avg
def eval_fn(data_loader, model, criterion, device=config.device):

    model.eval()

    loss_handler = AverageMeter()

    score_handler = AverageMeter()

    

    with torch.no_grad():

        tk0 = tqdm(data_loader, total=len(data_loader))

        for inputs, target in tk0:



            inputs = inputs.to(device)

            target = target.to(device)

            out = model(inputs)

            loss = criterion(out, target)



            loss_handler.update(loss.item())

            score_handler.update( f1_score(to_numpy(target), to_numpy(torch.argmax(out, dim=1)), average="micro") )



            tk0.set_postfix(loss=f"{loss_handler.avg:.5f}")



    return loss_handler, score_handler.avg
def main(config, fold):

    

    LOGGER.info(f"\n{pformat(config.__dict__)}\n")

    LOGGER.info("\nHERE GOES! 🚀")

    LOGGER.info(f"eval fold: {fold}")

    df = pd.read_csv(config.train_df)

    ebird_dct = {}

    for i, label in enumerate(df.ebird_code.unique()):

        ebird_dct[label] = i



    train_data = df.loc[:, ["im_path", "ebird_code", "fold"]]

    train_data.ebird_code = train_data.ebird_code.map(ebird_dct)



    train_ds = BSImageData(train_data, eval_fold=fold)

    eval_ds = BSImageData(train_data, eval_fold=fold, train=False)



    train_dl = DataLoader(train_ds, batch_size=config.bs, num_workers=4, shuffle=True)

    eval_dl = DataLoader(eval_ds, batch_size=config.bs, num_workers=4, shuffle=False)



    device = config.device

    model = build_model()

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if config.mixed_precision:

        model, optimizer = amp.initialize(

                model, optimizer, opt_level=config.opt_level

        )



    criterion = nn.CrossEntropyLoss()

    model_queue = Queue()

    for epoch in range(config.num_epochs):

        train_loss, train_score = train_fn(

            train_dl, model, criterion, optimizer, config=config, epoch=epoch

        )



        valid_loss, valid_score = eval_fn(eval_dl, model, criterion)



        LOGGER.info(

            f"|EPOCH {epoch+1}| F1_train {train_score:.5f}| F1_valid {valid_score:.5f}|"

        )

        

        best_loss = float("inf")

        if valid_loss.avg < best_loss:

            best_loss = valid_loss.avg

            print(f"New best model in epoch {epoch+1}")

            mname = f"{epoch+1}_best_model_{best_loss:.5f}.pth"

            torch.save(model.state_dict(), config.checkpoint_dir / mname)

            

            model_queue.put(mname)

            if model_queue.qsize() > config.num_best_models:

                mname_to_del = model_queue.get()

                (config.checkpoint_dir / mname_to_del).unlink()

                LOGGER.info(f"{mname_to_del} deleted")

                

    LOGGER.info(f"saved models: {model_queue.queue}")

main(config, fold=0)
!cat ../working/checkpoint_dir/logfile.txt