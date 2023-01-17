!pip install git+https://github.com/fastai/fastai2 > /dev/null

!pip install efficientnet-pytorch > /dev/null
from pathlib import Path



import pandas as pd



import torch

from efficientnet_pytorch import EfficientNet

from torch.utils import model_zoo



from fastai2.basics import *

from fastai2.data.all import *

from fastai2.callback.all import *

from fastai2.vision.all import *
DATA_PATH = Path('/kaggle/input/bengaliai-cv19')

IMAGE_DATA_PATH = Path('/kaggle/input/grapheme-imgs-128x128')

OUTPUT_PATH = Path('/kaggle/working')



VALID_PCT = 0.2

SEED = 420

BATCH_SIZE = 64

CROP_SIZE = 32

IMG_SIZE = 128
train_df = pd.read_csv(DATA_PATH/'train.csv')
class ImageWithCenterRemoved(Transform):

    """Transform that removes the center part of an image."""

    

    order = 6



    def __init__(self, crop_size=CROP_SIZE):

        self.crop_size = crop_size



    def encodes(self, x:PILImageBW) -> PILImageBW:

        x = array(x)

    

        start_height = tuple(IMG_SIZE // 2 - (CROP_SIZE // 2))

        start_width = tuple(IMG_SIZE // 2 - (CROP_SIZE // 2)) 

        

        x[

            ...,

            start_height:start_height+self.crop_size,

            start_width:start_width+self.crop_size

        ] = 0

    

        return PILImageBW(Image.fromarray(x))

    

    def encodes(self, x:TensorImage):

        start_height = IMG_SIZE // 2 - (CROP_SIZE // 2)

        start_width = IMG_SIZE // 2 - (CROP_SIZE // 2)

        

        x[

            ...,

            start_height:start_height+self.crop_size,

            start_width:start_width+self.crop_size

        ] = 0

        

        return TensorImage(x)

    

    

class ImageWithOnlyCenter(Transform):

    """Transform that keeps only the center part of an image."""

    

    order = 6

    

    def __init__(self, crop_size=CROP_SIZE):

        self.crop_size = crop_size



    def encodes(self, x:TensorImage) -> PILImageBW:

        start_height = IMG_SIZE // 2 - (CROP_SIZE // 2)

        start_width = IMG_SIZE // 2 - (CROP_SIZE // 2)

        

        output = x[

            ...,

            start_height:start_height + self.crop_size,

            start_width:start_width + self.crop_size

        ]



        return TensorImage(output)
items = get_image_files(IMAGE_DATA_PATH)
x_tfms = [PILImageBW.create, ToTensor, ImageWithCenterRemoved()]

y_tfms = [PILImageBW.create, ToTensor, ImageWithOnlyCenter()]

tfms = [x_tfms, y_tfms]



splitter = RandomSplitter(VALID_PCT, seed=SEED)



tds = Datasets(items, tfms, splits=splitter(items))
imagenet_stats
dl_tfms = [IntToFloatTensor,  Normalize(mean=0.485, std=0.229)]



train_dl = TfmdDL(tds.train, bs=BATCH_SIZE, after_batch=dl_tfms)

valid_dl = TfmdDL(tds.valid, bs=BATCH_SIZE, after_batch=dl_tfms)
train_dl.show_batch()
data = DataLoaders(train_dl, valid_dl)
class EfficientNetEncoder(EfficientNet):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        

        # the initial layer to convolve into 3 channels

        # idea from https://www.kaggle.com/aleksandradeis/bengali-ai-efficientnet-pytorch-starter

        self.input_conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)



    def forward(self, inputs):

        x = self.input_conv(inputs)

        return self.extract_features(x)

    

    @classmethod

    def load_pretrained(cls):

        model_name = 'efficientnet-b0'

        model = cls.from_name(model_name, override_params={'num_classes': 1})

        model_dict = model.state_dict()



        state_dict = model_zoo.load_url('https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b0-355c32eb.pth')

        state_dict_no_fc = {k: v for k, v in state_dict.items() if not k.startswith('_fc')}

        model_dict.update(state_dict_no_fc)

        

        model.load_state_dict(model_dict)



        return model
def up_conv(in_channels, out_channels):

    return nn.Sequential(

        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),

        nn.BatchNorm2d(out_channels),

        nn.ReLU(inplace=True)

    )
class Decoder(nn.Module):



    def __init__(self, encoder, n_channels, out_channels=1):

        super().__init__()



        self.encoder = encoder



        self.up_conv1 = up_conv(n_channels, 256)    

        self.up_conv2 = up_conv(256, 128)    # 8x8

        self.up_conv3 = up_conv(128, 64)    # 16x16

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    

    def forward(self, x):

        x = self.encoder(x)     # input: 1x128x128, output: 1280x4x4

        x = self.up_conv1(x)    # input: 1280x4x4, output: 256x8x8

        x = self.up_conv2(x)    # input: 256x8x8, output: 128x16x16

        x = self.up_conv3(x)    # input: 128x16x16, output: 64x32x32

        x = self.final_conv(x)  # input: 64x32x32, output: 1x32x32

        

        return x
encoder = EfficientNetEncoder.load_pretrained()

model = Decoder(encoder, n_channels=1280)  # 1280: EfficientNet b0 output. To do: don't hardcode this.
if torch.cuda.is_available():

    print('Cuda available')

    model = model.cuda()

    data = data.cuda()
learner = Learner(data, model, loss_func=nn.MSELoss())
# learner.lr_find()
learner.fit_one_cycle(4, 1e-3)
learner.recorder.plot_loss()
learner.validate()
learner.show_results(ds_idx=1)
learner.save('model_cycle_1')