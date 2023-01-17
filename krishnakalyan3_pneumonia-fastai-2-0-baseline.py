!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

!pip install fastai==2.0.9
import random, os

import numpy as np

import torch

from fastai.vision.all import *
def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



seed_everything()
path = Path('/kaggle/input/chest-xray-pneumonia/chest_xray/'); path.ls()
dls = ImageDataLoaders.from_folder(path, train='train',

                                   item_tfms=Resize(224),valid_pct=0.2,

                                   bs=64,seed=0)
dls.show_batch()
print(dls.vocab); print(dls.c)
learn = cnn_learner(dls, resnet34, metrics=[error_rate, accuracy], model_dir="/tmp/model/").to_fp16()
learn.lr_find()
learn.fit_one_cycle(1, lr_max=1e-2)
learn.unfreeze()
learn.fit_one_cycle(1, lr_max=1e-4)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(5,5), dpi=60)
learn.show_results()
preds, _ = learn.get_preds(); preds.shape
test_items = get_image_files(path/'test')

dl = learn.dls.test_dl(test_items, rm_type_tfms=1, bs=64)
y_pred, _ = learn.get_preds(dl=dl)

thresh = 0.5

yhat_test = [' '.join([learn.dls.vocab[i] for i,p in enumerate(pred) if p > thresh]) for pred in y_pred.numpy()]
y_test = list(map(lambda x:x.parents[0].name, test_items))
results = pd.DataFrame({'target': y_test, 'pred': yhat_test})
accuracy = results[results.target == results.pred].shape[0]/ results.shape[0]; accuracy