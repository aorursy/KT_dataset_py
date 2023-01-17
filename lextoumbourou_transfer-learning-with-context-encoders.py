!pip install git+https://github.com/fastai/fastai2 > /dev/null

!pip install efficientnet-pytorch > /dev/null

!pip install scikit-learn --upgrade > /dev/null
from pathlib import Path



import pandas as pd



import torch

from efficientnet_pytorch import EfficientNet

from efficientnet_pytorch.utils import get_same_padding_conv2d, round_filters, load_pretrained_weights

from sklearn.metrics import recall_score

from torch.utils import model_zoo



from fastai2.basics import *

from fastai2.data.all import *

from fastai2.callback.all import *

from fastai2.vision.all import *
VALID_PCT = 0.2

SEED = 420

BATCH_SIZE = 64

IMG_SIZE = 128



DATA_PATH = Path('/kaggle/input/bengaliai-cv19')

IMAGE_DATA_PATH = Path('/kaggle/input/grapheme-imgs-128x128')

OUTPUT_PATH = Path('/kaggle/working')

LABELS_PATH  = Path('/kaggle/input/iterative-stratification')
train_df = pd.read_csv(LABELS_PATH/'train_with_fold.csv')#.sample(n=50000).reset_index(drop=True)
imagenet_stats
datablock = DataBlock(

    blocks=(ImageBlock(cls=PILImageBW), CategoryBlock, CategoryBlock, CategoryBlock),

    getters=[

        ColReader('image_id', pref=IMAGE_DATA_PATH, suff='.png'),

        ColReader('grapheme_root'),

        ColReader('vowel_diacritic'),

        ColReader('consonant_diacritic')

    ],

    splitter=IndexSplitter(train_df.loc[train_df.fold==0].index))
tfms = aug_transforms(do_flip=False, size=IMG_SIZE) + [Normalize(mean=0.485, std=0.229)]
data = datablock.dataloaders(train_df, bs=BATCH_SIZE, batch_tfms=tfms)

data.n_inp = 1 
data.show_batch()
class loss_func(Module):

    def __init__(self, func=F.cross_entropy, weights=[2, 1, 1]):

        self.func, self.w = func, weights



    def forward(self, xs, *ys):

        for i, w, x, y in zip(range(len(xs)), self.w, xs, ys):

            if i == 0:

                loss = w*self.func(x, y) 

            else:

                loss += w*self.func(x, y) 



        return loss
recall_score
class RecallPartial(Metric):

    # based on AccumMetric

    """Stores predictions and targets on CPU in accumulate to perform final calculations with `func`."""

    def __init__(self, a=0, **kwargs):

        self.func = partial(recall_score, average='macro', zero_division=0)

        self.a = a



    def reset(self): self.targs,self.preds = [],[]



    def accumulate(self, learn):

        pred = learn.pred[self.a].argmax(dim=-1)

        targ = learn.y[self.a]

        pred,targ = to_detach(pred),to_detach(targ)

        pred,targ = flatten_check(pred,targ)

        self.preds.append(pred)

        self.targs.append(targ)



    @property

    def value(self):

        if len(self.preds) == 0: return

        preds,targs = torch.cat(self.preds),torch.cat(self.targs)

        return self.func(targs, preds)



    @property

    def name(self): return train_df.columns[self.a+1]

    



class RecallCombine(Metric):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.combine = 0



    def accumulate(self, learn):

        scores = [learn.metrics[i].value for i in range(3)]

        self.combine = np.average(scores, weights=[2,1,1])



    @property

    def value(self):

        return self.combine
class BengaliEfficientNet(EfficientNet):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        

        # the initial layer to convolve into 3 channels

        # idea from https://www.kaggle.com/aleksandradeis/bengali-ai-efficientnet-pytorch-starter

        self.input_conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)



        self.fc1 = nn.Linear(in_features=1280, out_features=168) # grapheme_root

        self.fc2 = nn.Linear(in_features=1280, out_features=11) # vowel_diacritic

        self.fc3 = nn.Linear(in_features=1280, out_features=7) # consonant_diacritic

    

    def forward(self, inputs):

        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        

        bs = inputs.size(0)

        

        # Convolve to 3 channels

        x = self.input_conv(inputs)



        # Convolution layers

        x = self.extract_features(x)

        

        # Pooling

        x = self._avg_pooling(x)

        

        # Final layers

        x = x.view(bs, -1)



        return [self.fc1(x), self.fc2(x), self.fc3(x)]

    

    @classmethod

    def load(cls, path=None):

        model = cls.from_name('efficientnet-b0')



        if path is not None:

            pretrained = torch.load(path, map_location=torch.device('cpu'))

            encoder_only = {k[len('encoder.'):]: v for (k, v) in pretrained['model'].items() if k.startswith('encoder.')}

            encoder_only_no_fc = {k: v for k, v in encoder_only.items() if not k.startswith('_fc')}

    

            model_dict = model.state_dict()

            model_dict.update(encoder_only_no_fc) 

            model.load_state_dict(model_dict)



        return model

    

    @classmethod

    def load_imagenet(cls, advprop=False):

        model_name = 'efficientnet-b0'

        model = cls.from_name(model_name, override_params={'num_classes': 1})

        model_dict = model.state_dict()



        state_dict = model_zoo.load_url('https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b0-355c32eb.pth')

        state_dict_no_fc = {k: v for k, v in state_dict.items() if not k.startswith('_fc')}

        model_dict.update(state_dict_no_fc)

        

        model.load_state_dict(model_dict)



        return model
model = BengaliEfficientNet.load_imagenet()



if torch.cuda.is_available():

    model = model.cuda()

    data = data.cuda()



learner = Learner(

    data,

    model,

    loss_func=loss_func(),

    cbs=CSVLogger('history_imagenet.csv'),

    metrics=[RecallPartial(a=i) for i in range(len(data.c))] + [RecallCombine()]

)



learner.unfreeze()
learner.fit_one_cycle(4, lr_max=slice(1e-3, 1e-2))
learner.recorder.plot_loss()
model = BengaliEfficientNet.load('/kaggle/input/self-supervised-pretraining-with-context-encoders/models/model_cycle_1.pth')



if torch.cuda.is_available():

    model = model.cuda()

    data = data.cuda()



learner = Learner(

    data,

    model,

    loss_func=loss_func(),

    cbs=CSVLogger('history_context_encoders.csv'),

    metrics=[RecallPartial(a=i) for i in range(len(data.c))] + [RecallCombine()]

)



learner.unfreeze()
learner.fit_one_cycle(4, lr_max=slice(1e-3, 1e-2))
learner.recorder.plot_loss()
model = BengaliEfficientNet.load()



if torch.cuda.is_available():

    model = model.cuda()

    data = data.cuda()



learner = Learner(

    data,

    model,

    loss_func=loss_func(),

    cbs=CSVLogger('history_no_pretrained.csv'),

    metrics=[RecallPartial(a=i) for i in range(len(data.c))] + [RecallCombine()]

).to_fp16()



learner.unfreeze()
learner.fit_one_cycle(4, lr_max=slice(1e-3, 1e-2))
learner.recorder.plot_loss()