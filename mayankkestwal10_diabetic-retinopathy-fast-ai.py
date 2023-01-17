%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import pandas as pd

import matplotlib.pyplot as plt
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything(42)
PATH = Path('../input/aptos2019-blindness-detection/')
df_train = pd.read_csv(PATH/'train.csv').assign(filename=lambda df:"train_images/"+df.id_code+".png")

df_test = pd.read_csv(PATH/'test.csv').assign(filename=lambda df:"test_images/"+df.id_code+".png")
df_train.diagnosis.hist()
transforms = get_transforms(do_flip=True,flip_vert=True,max_zoom=1.1,max_rotate=360,max_lighting=0.2,max_warp=0.2,p_lighting=0.5)
data = ImageDataBunch.from_df(path=PATH,df=df_train,fn_col="filename",label_col="diagnosis",ds_tfms=transforms,size=224).normalize(imagenet_stats)
data.show_batch(rows=3,fig_size=(5,5))
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)
!cp '../input/pytorch-pretrained-image-models/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'
kappa = KappaScore()

kappa.weights = "quadratic"

learn = cnn_learner(data, models.resnet50,

                    metrics=[error_rate, kappa],

                    model_dir="/tmp/model/")
learn.lr_find()
learn.recorder.plot(suggestion=True)
lr=9e-3

learn.fit_one_cycle(10,lr)
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sample_df.head()
learn.data.add_test(ImageList.from_df(

    sample_df, PATH,

    folder='test_images',

    suffix='.png'

))
preds,y = learn.get_preds(DatasetType.Test)
sample_df.diagnosis = preds.argmax(1)
sample_df
sample_df.to_csv('submission.csv',index=False)

_ = sample_df.hist()