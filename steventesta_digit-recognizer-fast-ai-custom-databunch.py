from fastai import *

from fastai.vision import *

from fastai.metrics import accuracy,error_rate
class CustomImageItemList(ImageItemList):

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
# note: there are no labels in a test set, so we set the imgIdx to begin at the 0 col

test = CustomImageItemList.from_csv_custom(path=path, csv_name='test.csv', imgIdx=0)
tfms = get_transforms(do_flip=False)

data = (CustomImageItemList.from_csv_custom(path=path, csv_name='train.csv')

                           .random_split_by_pct(.2)

                           .label_from_df(cols='label')

                           .add_test(test, label=0)

                           .transform(tfms)

                           .databunch(bs=64, num_workers=0)

                           .normalize(imagenet_stats))

                          
data.show_batch(rows=3, figsize=(5,5))
learn = create_cnn(data, arch=models.resnet50, metrics=[accuracy,error_rate], model_dir='/kaggle/working/models')
learn.lr_find()

learn.recorder.plot()
lr = 1e-2

learn.fit_one_cycle(8, lr)
interp = ClassificationInterpretation.from_learner(learn)

interp.most_confused(min_val=2)
learn.save('stage1-resnet50')
learn.load('stage1-resnet50')

learn.validate()
#learn.unfreeze()

#learn.fit_one_cycle(1, 5e-6)
# get the predictions

predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)

# output to a file

submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})

submission_df.to_csv(f'submission.csv', index=False)