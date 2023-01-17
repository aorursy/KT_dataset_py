!pip install pretrainedmodels
%reload_ext autoreload

%autoreload 2

%matplotlib inline

import os

from fastai.vision import *

from fastai.utils import mem

from fastai.callbacks import ReduceLROnPlateauCallback, SaveModelCallback

from sklearn.metrics import f1_score

from fastai.vision.learner import model_meta

import pretrainedmodels

print('Make sure cuda is installed:', torch.cuda.is_available())

print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)

mem.gpu_mem_get()
#!kaggle competitions download -c iwildcam-2019-fgvc6
train = pd.read_csv('../input/train.csv')

train = train[['file_name', 'category_id']]

train.head()
test = pd.read_csv('../input/test.csv')

test = test[['file_name']]
PATH = '../input/'
datatest = ImageList.from_df(test, path=PATH, cols=0, folder='test_images')
def get_data(bs, size):

    return (ImageList.from_df(train, path=PATH, cols=0, folder='train_images')

     .split_by_rand_pct(0.2, seed=47)

     .label_from_df(cols=1)

     .transform(get_transforms(xtra_tfms=[pad(mode='reflection')]), size=size)

     .add_test(datatest)

     .databunch(bs=bs)) 
data = get_data(128, 32)

#stats = data.batch_stats()

data.normalize(imagenet_stats)
data.show_batch()
class FocalLoss(nn.Module):

    def __init__(self, alpha=1., gamma=1.):

        super().__init__()

        self.alpha = alpha

        self.gamma = gamma



    def forward(self, inputs, targets, **kwargs):

        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)

        pt = torch.exp(-CE_loss)

        F_loss = self.alpha * ((1-pt)**self.gamma) * CE_loss

        return F_loss.mean()



loss_func = FocalLoss(gamma=1.)
def senet154(pretrained=False):

    pretrained = 'imagenet' if pretrained else None

    model = pretrainedmodels.senet154(pretrained=pretrained)

    return model
_se_resnet_meta = {'cut': -3, 'split': lambda m: (m[0][3], m[1]) }

model_meta[senet154] = _se_resnet_meta
learn = create_cnn(data, senet154,  ps=0.5, wd=1e-1, loss_func=loss_func, metrics=[FBeta()], pretrained=True).to_fp16().mixup()
lr = 4.79E-02
RLR = ReduceLROnPlateauCallback(learn, monitor='f_beta',patience = 2)

SAVEML = SaveModelCallback(learn, every='improvement', monitor='f_beta', name='best')
#learn.fit_one_cycle(5, lr, callbacks = [RLR, SAVEML])
learn.recorder.plot_losses() 
learn.save('se-1')
learn.load('best')
learn.unfreeze()
#learn.fit_one_cycle(5, slice(1e-5,1e-3), callbacks = [RLR, SAVEML])
learn.load('best')
learn.save('se-2')
learn.recorder.plot_losses() 
pred, y = learn.get_preds()

#pred, y = learn.TTA()
f1_score = f1_score(y, np.argmax(pred.numpy(), 1), average='macro')  

f1_score
#learn = learn.to_fp32()
#pred_t, _ = learn.TTA(ds_type=DatasetType.Test)
#import os

#test_ids = [os.path.basename(f)[:-4] for f in learn.data.test_ds.items]

#subm = pd.read_csv('sample_submission.csv')

#orig_ids = list(subm['Id'])
#pred_t2 = np.argmax(pred_t.numpy(), 1)
def create_submission(orig_ids, test_ids, preds):

    preds_dict = dict((k, v) for k, v in zip(test_ids, preds))

    pred_cor = [preds_dict[id] for id in orig_ids]

    df = pd.DataFrame({'id':orig_ids,'Predicted':pred_cor})

    df.to_csv(f'submission_{f1_score}.csv', header=True, index=False)

    return df
#sub = create_submission(orig_ids, test_ids, pred_t2)
#sub.tail()
#! kaggle competitions submit -c iwildcam-2019-fgvc6 -f submission_0.5804866967400385.csv -m densenet