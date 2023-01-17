import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import random



import librosa

import librosa.display

import cv2

import torch

import torch.nn as nn

from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader

import transformers

from transformers import AdamW, get_linear_schedule_with_warmup

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2 

from os import path, listdir



from sklearn.metrics import f1_score



SEED = 42



import warnings

warnings.filterwarnings("ignore")





def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(SEED)
root_dir = "../input/birdsong-recognition"

fake_test_dir = "../input/birdcall-check"



resampled_dirs = [f"birdsong-resampled-train-audio-0{i}" for i in range(5)]
test_audio_path = path.join(root_dir, "test_audio")

if path.exists(test_audio_path):

    test = pd.read_csv(path.join(root_dir, "test.csv"))

else:

    test_audio_path = path.join(fake_test_dir, "test_audio")

    test = pd.read_csv(path.join(fake_test_dir, "test.csv"))
# config

sr = 32000 # because test audios will be of 32k sr, gives better generalization

effective_length = 5 * 32000 # we take random 5 sec audio

TRAIN_BATCH_SIZE = 16

EPOCHS = 15
def set_range(img, max_, min_):

    img[img > max_] = max_

    img[img < min_] = min_

    return img

def min_max(img):

    img = (img - img.min())/(img.max() - img.min())

    img = img * 255

    

    return img.astype(int)
BIRD_CODE = {

    'aldfly': 0, 'ameavo': 1, 'amebit': 2, 'amecro': 3, 'amegfi': 4,

    'amekes': 5, 'amepip': 6, 'amered': 7, 'amerob': 8, 'amewig': 9,

    'amewoo': 10, 'amtspa': 11, 'annhum': 12, 'astfly': 13, 'baisan': 14,

    'baleag': 15, 'balori': 16, 'banswa': 17, 'barswa': 18, 'bawwar': 19,

    'belkin1': 20, 'belspa2': 21, 'bewwre': 22, 'bkbcuc': 23, 'bkbmag1': 24,

    'bkbwar': 25, 'bkcchi': 26, 'bkchum': 27, 'bkhgro': 28, 'bkpwar': 29,

    'bktspa': 30, 'blkpho': 31, 'blugrb1': 32, 'blujay': 33, 'bnhcow': 34,

    'boboli': 35, 'bongul': 36, 'brdowl': 37, 'brebla': 38, 'brespa': 39,

    'brncre': 40, 'brnthr': 41, 'brthum': 42, 'brwhaw': 43, 'btbwar': 44,

    'btnwar': 45, 'btywar': 46, 'buffle': 47, 'buggna': 48, 'buhvir': 49,

    'bulori': 50, 'bushti': 51, 'buwtea': 52, 'buwwar': 53, 'cacwre': 54,

    'calgul': 55, 'calqua': 56, 'camwar': 57, 'cangoo': 58, 'canwar': 59,

    'canwre': 60, 'carwre': 61, 'casfin': 62, 'caster1': 63, 'casvir': 64,

    'cedwax': 65, 'chispa': 66, 'chiswi': 67, 'chswar': 68, 'chukar': 69,

    'clanut': 70, 'cliswa': 71, 'comgol': 72, 'comgra': 73, 'comloo': 74,

    'commer': 75, 'comnig': 76, 'comrav': 77, 'comred': 78, 'comter': 79,

    'comyel': 80, 'coohaw': 81, 'coshum': 82, 'cowscj1': 83, 'daejun': 84,

    'doccor': 85, 'dowwoo': 86, 'dusfly': 87, 'eargre': 88, 'easblu': 89,

    'easkin': 90, 'easmea': 91, 'easpho': 92, 'eastow': 93, 'eawpew': 94,

    'eucdov': 95, 'eursta': 96, 'evegro': 97, 'fiespa': 98, 'fiscro': 99,

    'foxspa': 100, 'gadwal': 101, 'gcrfin': 102, 'gnttow': 103, 'gnwtea': 104,

    'gockin': 105, 'gocspa': 106, 'goleag': 107, 'grbher3': 108, 'grcfly': 109,

    'greegr': 110, 'greroa': 111, 'greyel': 112, 'grhowl': 113, 'grnher': 114,

    'grtgra': 115, 'grycat': 116, 'gryfly': 117, 'haiwoo': 118, 'hamfly': 119,

    'hergul': 120, 'herthr': 121, 'hoomer': 122, 'hoowar': 123, 'horgre': 124,

    'horlar': 125, 'houfin': 126, 'houspa': 127, 'houwre': 128, 'indbun': 129,

    'juntit1': 130, 'killde': 131, 'labwoo': 132, 'larspa': 133, 'lazbun': 134,

    'leabit': 135, 'leafly': 136, 'leasan': 137, 'lecthr': 138, 'lesgol': 139,

    'lesnig': 140, 'lesyel': 141, 'lewwoo': 142, 'linspa': 143, 'lobcur': 144,

    'lobdow': 145, 'logshr': 146, 'lotduc': 147, 'louwat': 148, 'macwar': 149,

    'magwar': 150, 'mallar3': 151, 'marwre': 152, 'merlin': 153, 'moublu': 154,

    'mouchi': 155, 'moudov': 156, 'norcar': 157, 'norfli': 158, 'norhar2': 159,

    'normoc': 160, 'norpar': 161, 'norpin': 162, 'norsho': 163, 'norwat': 164,

    'nrwswa': 165, 'nutwoo': 166, 'olsfly': 167, 'orcwar': 168, 'osprey': 169,

    'ovenbi1': 170, 'palwar': 171, 'pasfly': 172, 'pecsan': 173, 'perfal': 174,

    'phaino': 175, 'pibgre': 176, 'pilwoo': 177, 'pingro': 178, 'pinjay': 179,

    'pinsis': 180, 'pinwar': 181, 'plsvir': 182, 'prawar': 183, 'purfin': 184,

    'pygnut': 185, 'rebmer': 186, 'rebnut': 187, 'rebsap': 188, 'rebwoo': 189,

    'redcro': 190, 'redhea': 191, 'reevir1': 192, 'renpha': 193, 'reshaw': 194,

    'rethaw': 195, 'rewbla': 196, 'ribgul': 197, 'rinduc': 198, 'robgro': 199,

    'rocpig': 200, 'rocwre': 201, 'rthhum': 202, 'ruckin': 203, 'rudduc': 204,

    'rufgro': 205, 'rufhum': 206, 'rusbla': 207, 'sagspa1': 208, 'sagthr': 209,

    'savspa': 210, 'saypho': 211, 'scatan': 212, 'scoori': 213, 'semplo': 214,

    'semsan': 215, 'sheowl': 216, 'shshaw': 217, 'snobun': 218, 'snogoo': 219,

    'solsan': 220, 'sonspa': 221, 'sora': 222, 'sposan': 223, 'spotow': 224,

    'stejay': 225, 'swahaw': 226, 'swaspa': 227, 'swathr': 228, 'treswa': 229,

    'truswa': 230, 'tuftit': 231, 'tunswa': 232, 'veery': 233, 'vesspa': 234,

    'vigswa': 235, 'warvir': 236, 'wesblu': 237, 'wesgre': 238, 'weskin': 239,

    'wesmea': 240, 'wessan': 241, 'westan': 242, 'wewpew': 243, 'whbnut': 244,

    'whcspa': 245, 'whfibi': 246, 'whtspa': 247, 'whtswi': 248, 'wilfly': 249,

    'wilsni1': 250, 'wiltur': 251, 'winwre3': 252, 'wlswar': 253, 'wooduc': 254,

    'wooscj2': 255, 'woothr': 256, 'y00475': 257, 'yebfly': 258, 'yebsap': 259,

    'yehbla': 260, 'yelwar': 261, 'yerwar': 262, 'yetvir': 263

}



def one_hot(label):

    identity_matrix = np.eye(264)

    lbl_int = BIRD_CODE[label]

    lbl_enc = identity_matrix[lbl_int]

    return torch.from_numpy(lbl_enc)

def get_train_transforms():

    return A.Compose([

#         A.HorizontalFlip(p = 0.5),

#         A.VerticalFlip(p = 0.5),

#         A.Resize(height = 512, width = 512, p = 1.0),

        ToTensorV2(p = 1.0),

    ], p = 1.0)



# def get_valid_transforms():

#     return A.Compose([

#         A.Resize(height = 512, width = 512, p = 1.0),

#         ToTensorV2(p = 1.0)

#     ], p = 1.0)
class BirdcallDataset(Dataset):

    def __init__(self, data, resampled_dirs, effective_length, transforms = None):

        self.data = data

        self.audiofilename = self.data["resampled_filename"].values

        self.ebird_code = self.data["ebird_code"]

        self.resampled_dirs = resampled_dirs

        self.effective_length = effective_length

        self.transforms = transforms

        

        

    def __getitem__(self, index: int):

        audiofilename = self.audiofilename[index]

        ebird_code = self.ebird_code[index]

        for dir_ in self.resampled_dirs:

            if path.exists(path.join("../input", dir_, ebird_code, audiofilename)):

                x, sr = librosa.load(path.join("../input", dir_, ebird_code, audiofilename),

                                     sr = 32000)

                # Randomly taking 5 sec audio

                if len(x) > effective_length:

                    start_index = np.random.randint(len(x) - effective_length)

                    end_index = start_index + self.effective_length

                    x = x[start_index: end_index]

                elif self.effective_length > len(x):

                    temp_ = np.zeros(effective_length)

                    start_index = np.random.randint(effective_length - len(x))

                    end_index = start_index + len(x)

                    temp_[start_index: end_index] = x

                    x = temp_

                else:

                    x = x

                    

                # convert audio to image mel-spectrogram

                # https://github.com/librosa/librosa/blob/main/examples/LibROSA%20demo.ipynb

                S = librosa.feature.melspectrogram(x, sr=sr, n_mels= 128,

                        fmin = 20,

                        fmax = 16000)

                image = librosa.power_to_db(S)

                # https://medium.com/@manivannan_data/resize-image-using-opencv-python-d2cdbbc480f0

                image = cv2.resize(image,(360,360))

                

                

                

                # Scaling image between 0-255

                image = (image - image.mean())/image.std()

                image = set_range(image, max_ = 4.0, min_ = -4.0)

                imgray = min_max(image)

                

                

                # Changing 1 channel to three channel image (so that resnest can work)

                # https://stackoverflow.com/questions/14786179/how-to-convert-a-1-channel-image-into-a-3-channel-with-opencv2

                image = cv2.merge((imgray,imgray,imgray)).astype(np.float32)

                

                if self.transforms:

                    sample = {"image": image}

                    sample = self.transforms(**sample)

                    image = sample["image"]    

                ebird_code = one_hot(ebird_code)

                return (image, ebird_code)

                break

                

    def __len__(self):

        return len(self.audiofilename)

    
df = pd.read_csv(path.join("../input", resampled_dirs[0], "train_mod.csv"))

data = BirdcallDataset(df, resampled_dirs, effective_length, transforms = get_train_transforms())
data[0][0].shape

from collections import OrderedDict



import torch

from torch import nn

from torch.jit.annotations import Dict





class IntermediateLayerGetter(nn.ModuleDict):

    _version = 2

    __annotations__ = {

        "return_layers": Dict[str, str],

    }



    def __init__(self, model, return_layers):

        if not set(return_layers).issubset([name for name, _ in model.named_children()]):

            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers

        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        layers = OrderedDict()

        for name, module in model.named_children():

            layers[name] = module

            if name in return_layers:

                del return_layers[name]

            if not return_layers:

                break



        super(IntermediateLayerGetter, self).__init__(layers)

        self.return_layers = orig_return_layers



    def forward(self, x):

        out = OrderedDict()

        for name, module in self.items():

            x = module(x)

            if name in self.return_layers:

                out_name = self.return_layers[name]

                out[out_name] = x

        return out
try:

    from torch.hub import load_state_dict_from_url

except ImportError:

    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch

import torch.nn as nn





__all__ = ['resnet50']





model_urls = {

    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',

}





def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):

    """3x3 convolution with padding"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,

                     padding=dilation, groups=groups, bias=False, dilation=dilation)





def conv1x1(in_planes, out_planes, stride=1):

    """1x1 convolution"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)





class BasicBlock(nn.Module):

    expansion = 1



    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,

                 base_width=64, dilation=1, norm_layer=None):

        super(BasicBlock, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:

            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        if dilation > 1:

            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)

        self.bn2 = norm_layer(planes)

        self.downsample = downsample

        self.stride = stride



    def forward(self, x):

        identity = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)



        if self.downsample is not None:

            identity = self.downsample(x)



        out += identity

        out = self.relu(out)



        return out





class Bottleneck(nn.Module):



    expansion = 4



    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,

                 base_width=64, dilation=1, norm_layer=None):

        super(Bottleneck, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv1x1(inplanes, width)

        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)

        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)

        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

        self.stride = stride



    def forward(self, x):

        identity = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)



        out = self.conv3(out)

        out = self.bn3(out)



        if self.downsample is not None:

            identity = self.downsample(x)



        out += identity

        out = self.relu(out)



        return out





class ResNet(nn.Module):



    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,

                 groups=1, width_per_group=64, replace_stride_with_dilation=None,

                 norm_layer=None):

        super(ResNet, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer



        self.inplanes = 64

        self.dilation = 1

        if replace_stride_with_dilation is None:

            # each element in the tuple indicates if we should replace

            # the 2x2 stride with a dilated convolution instead

            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:

            raise ValueError("replace_stride_with_dilation should be None "

                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups

        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,

                               bias=False)

        self.bn1 = norm_layer(self.inplanes)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,

                                       dilate=replace_stride_with_dilation[0])

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,

                                       dilate=replace_stride_with_dilation[1])

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,

                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)



        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)



        # Zero-initialize the last BN in each residual branch,

        # so that the residual branch starts with zeros, and each residual block behaves like an identity.

        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        if zero_init_residual:

            for m in self.modules():

                if isinstance(m, Bottleneck):

                    nn.init.constant_(m.bn3.weight, 0)

                elif isinstance(m, BasicBlock):

                    nn.init.constant_(m.bn2.weight, 0)



    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):

        norm_layer = self._norm_layer

        downsample = None

        previous_dilation = self.dilation

        if dilate:

            self.dilation *= stride

            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(

                conv1x1(self.inplanes, planes * block.expansion, stride),

                norm_layer(planes * block.expansion),

            )



        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,

                            self.base_width, previous_dilation, norm_layer))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):

            layers.append(block(self.inplanes, planes, groups=self.groups,

                                base_width=self.base_width, dilation=self.dilation,

                                norm_layer=norm_layer))



        return nn.Sequential(*layers)



    def _forward_impl(self, x):

        # See note [TorchScript super()]

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)



        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)



        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)



        return x



    def forward(self, x):

        return self._forward_impl(x)





def _resnet(arch, block, layers, pretrained, progress, **kwargs):

    model = ResNet(block, layers, **kwargs)

    if pretrained:

        state_dict = load_state_dict_from_url(model_urls[arch],

                                              progress=progress)

        model.load_state_dict(state_dict)

    return model







def resnet50(pretrained=False, progress=True, **kwargs):

    

    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,

                   **kwargs)





net = resnet50(pretrained = True)

net.fc = nn.Linear(in_features=2048, out_features=264, bias=True)
def micro_precision(ytrue, ypred, micro_recall = None):

    """

    Function for calculating micro-precision:

    y_true: list of one hot

    y_pred: list of one hot

 

    """

    

    # Taking 0.5 as threshold

    m = nn.Sigmoid()

    ypred = m(ypred.detach()).cpu().numpy()

    ypred[ypred > 0.5] = 1    

    ypred[ypred < 0.5] = 0



    ytrue = ytrue.detach().cpu().numpy()

    score = f1_score(ytrue, ypred, average="samples")

    return score
train_data_loader = DataLoader(

            data,

            batch_size=TRAIN_BATCH_SIZE,

            drop_last=True,  # take care of last batch

            num_workers=2

    )
def _run():

    device = torch.device("cuda: 0")

    model = net.to(device)

    

    def training_loop(train_data_loader, model, scheduler = None):

        for batch, (data, target) in enumerate(train_data_loader):

            inputs = data.to(device, dtype = torch.float32)

            targets = target.to(device, dtype = torch.float32)

            # clear optimizer

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, targets)

            

            micro_prec = micro_precision(targets, outputs)

            if batch != 0 and batch % 300 == 0:

                print(f"Batch = {batch}, Loss = {loss}, Micro precision = {micro_prec}")

            # gradients calculations

            loss.backward()

            # Update weights

            optimizer.step()

            if scheduler is not None:

                scheduler.step()

            

            

    lr = 0.001

    num_train_steps = (len(data) / TRAIN_BATCH_SIZE) * EPOCHS

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=lr)

    scheduler = get_linear_schedule_with_warmup(

                optimizer,

                num_warmup_steps=0,

                num_training_steps=num_train_steps

            )

    for e in range(EPOCHS):

        print("#" * 25)

        print(f"Epoch no: {e}")

        print("#" * 25)



        training_loop(train_data_loader, model, scheduler = scheduler)

    torch.save(model.state_dict(), f"model{e}.bin")
_run()