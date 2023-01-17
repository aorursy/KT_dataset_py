import os
print(os.listdir("../input/"))
%reload_ext autoreload
%autoreload 2
%matplotlib inline
import numpy as np
import pandas as pd
from fastai.vision import *
from fastai.metrics import error_rate
from fastai import *
print("PyTorch version - ", torch.__version__)
print("Cuda version - ", torch.version.cuda)
print("cuDNN version - ", torch.backends.cudnn.version())
print("Device - ", torch.device("cuda:0"))
print("python PIL version - ", PIL.PILLOW_VERSION)
!nvidia-smi
batch_size = 64
data_path = "../input/"
data_path_train = data_path + "train/train/"
data_path_test = data_path + "test/test/"
df_train = pd.read_csv(data_path + "train.csv")
df_test = pd.read_csv(data_path + "sample_submission.csv")
df_train.head()
data = ImageDataBunch.from_df(data_path_train, df_train, ds_tfms=get_transforms(), bs=batch_size).normalize(imagenet_stats)
data.add_test(ImageList.from_df(df_test, path=data_path_test))
data
data.show_batch(rows = 3, figsize = (10,8))
print(data.classes)
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/")
learn.model
learn.fit_one_cycle(4)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix()
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
predictions = learn.get_preds(ds_type=DatasetType.Test)[0]
predictions[0]
predictions[:10]
predicted_classes = np.argmax(predictions, axis=1)
predicted_classes[:10]
df_test['has_cactus'] = predicted_classes
df_test.head(10)
from datetime import datetime
time_format = "%Y%m%d-%H%M%S.%f"
time_stamp = datetime.now().strftime(time_format)
file_path = "{0}submission_{1}.csv".format(data_path, datetime.now().strftime(time_format))
                                        
print("Exporting Submission file with {0} rows at {1}".format(df_test.shape[0], file_path))

df_test.to_csv(file_path, index = False)

