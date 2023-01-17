from fastai.vision import *
class CustomImageItemList(ImageItemList):

    def open(self, fn):

        img = fn.reshape(28, 28)

        img = np.stack((img,)*3, axis=-1) # convert to 3 channels

        return Image(pil2tensor(img, dtype=np.float32))



    @classmethod

    def custom_csv(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs) -> 'ItemList':

        df = pd.read_csv(Path(path)/csv_name, header=header)

        res = super().from_df(df, path=path, cols=0, **kwargs)

        # convert pixels to an ndarray

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 783.0, axis=1).values

        return res
test = CustomImageItemList.custom_csv(path=Path('../input/'), csv_name='test.csv', imgIdx=0)

data = (CustomImageItemList.custom_csv(path=Path('../input/'), csv_name='train.csv')

                       .random_split_by_pct(.2)

                       .label_from_df(cols='label')

                       .add_test(test, label=0)

                       .databunch(bs=64, num_workers=0)

                       .normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(12,9))
learn = create_cnn(data, models.resnet50, metrics=accuracy, model_dir = Path('../working/'))
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-2,1e-1))
learn.save('50-stage-1')
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()
learn.fit_one_cycle(2,slice(7e-6,1e-5))
learn.save('50-stage-2')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2,slice(2e-5,1e-4))
learn.save('50-stage-3')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2,slice(1e-4,2e-4))
learn.save('50-stage-4')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2,slice(5e-6,2e-5))
learn.save('50-stage-5')
learn.load('50-stage-5')

preds, y, losses = learn.get_preds(ds_type=DatasetType.Test, with_loss=True)
y = torch.argmax(preds, dim=1)
preds.shape, len(y)
submission_df = pd.DataFrame({'ImageId': range(1, len(y) + 1), 'Label': y}, columns=['ImageId', 'Label'])

submission_df.head()
submission_df.to_csv('submission.csv', index=False)
!head submission.csv