import pandas as pd 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from fastai import *

from fastai.vision import *
out_path = Path('./')

print(out_path.ls())

in_path = Path('../input/')
df = pd.read_csv(in_path/'train.csv')

df.head(n=5)
class CustomImageList(ImageList):

    def open(self, fn):

        img = fn.reshape(28, 28)

        img = np.stack((img,)*3, axis=-1)  # convert to 3 channels

        return Image(pil2tensor(img, dtype=np.float32))



    @classmethod

    def from_csv_custom(cls, path:PathOrStr, csv_name:str,

                             imgIdx:int=1, header:str='infer', **kwargs) -> 'ItemList':

        """create imagelist from the csv

        

        note: can add annotations to python 3 functions (->, :type, etc)

        

        use the super method to set-up the result, which we then add items

        to using the open method.

        """

        

        # read dataset

        df = pd.read_csv(Path(path)/csv_name, header=header)

        res = cls.from_df(df, path=path, cols=1, **kwargs)

        

        # convert pixels to an ndarray

        res.items = df.iloc[:, imgIdx:].apply(

            lambda x: x.values, axis=1).values

        return res
tfms = get_transforms(do_flip=False,

                      flip_vert=False,

                      max_rotate=20.0,

                      max_zoom=1.1, max_lighting=0.,

                      max_warp=0.2, p_affine=0.75, p_lighting=0.)
test = CustomImageList.from_csv_custom(path=in_path, csv_name='test.csv', imgIdx=0)

data = (CustomImageList.from_csv_custom(path=in_path, csv_name='train.csv')

                       .split_by_rand_pct(.2)  # make validation

                       .label_from_df(cols='label')  # uses self.inner_df

                       .add_test(test)

                       .transform(tfms)

                       .databunch(bs=64, num_workers=0)

                       .normalize(imagenet_stats))

data.show_batch(rows=3, figsize=(12, 9))
# kaggle does not have internet by default (some competitions don't allow it)

learner = cnn_learner(data, models.resnet50, metrics=error_rate, pretrained=False, model_dir="./")
learner.fit_one_cycle(25)
learner.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learner)
losses, idxs = interp.top_losses()

interp.plot_top_losses(16, figsize=(15,11))
# now use these to save as the submission file. overcome bug in prediction api

res, _ = learner.get_preds(ds_type=DatasetType.Test)

labels = np.argmax(res, 1)
! head "../input/sample_submission.csv"
! wc -l "../input/sample_submission.csv"
submission = pd.DataFrame({'ImageId': list(range(1, len(test)+1)),

                           'Label': labels})
submission.to_csv(out_path/'submission.csv', sep=',', index=False)
! head {out_path/'submission.csv'}