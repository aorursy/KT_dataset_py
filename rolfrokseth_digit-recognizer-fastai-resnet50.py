from fastai import *

from fastai.vision import *

from fastai.metrics import accuracy,error_rate
class CustomImageItemList(ImageList):

    def open(self, fn):

        img = fn.reshape(28,28)

        img = np.stack((img,)*3, axis=-1) # convert to 3 channels

        return Image(pil2tensor(img, dtype=np.float32))



    @classmethod

    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList':

        df = pd.read_csv(Path(path)/csv_name, header=header)

        res = super().from_df(df, path=path, cols=0, **kwargs)

        # convert pixels to an ndarray

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values

        return res
path = '../input'
test = CustomImageItemList.from_csv_custom(path=path, csv_name='test.csv', imgIdx=0)

data = (CustomImageItemList.from_csv_custom(path=path, csv_name='train.csv')

                           .split_by_rand_pct(.2)

                           .label_from_df(cols='label')

                           .add_test(test, label=0)

                           .databunch(bs=64, num_workers=0)

                           .normalize(imagenet_stats))
learn = cnn_learner(data, models.resnet50, metrics=error_rate,model_dir="/tmp/model/")
learn.fit_one_cycle(4)
learn.save("model_1", return_path=True)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('model_1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(10, max_lr=slice(1e-5,1e-4))
# get the predictions

predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)

# output to a file

submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})

submission_df.to_csv(f'submission.csv', index=False)