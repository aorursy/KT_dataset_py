from fastai.vision import *



import fastai.utils.collect_env; fastai.utils.collect_env.show_install(1)

print('Working on "%s"' % Path('.').absolute())
class NumpyImageList(ImageList):

    def open(self, fn):

        img = fn.reshape(28,28,1)

        return Image(pil2tensor(img, dtype=np.float32))

    

    @classmethod

    def from_csv(cls, path:PathOrStr, csv:str, **kwargs)->'ItemList': 

        df = pd.read_csv(Path(path)/csv, header='infer')

        res = super().from_df(df, path=path, cols=0, **kwargs)



        if 'label' in df.columns:

            df = df.drop('label', axis=1)

        df = np.array(df)/255.

        res.items = (df-df.mean())/df.std()



        return res
test = NumpyImageList.from_csv('../input/', 'test.csv')

test
tfms = get_transforms(do_flip=False)

data = (NumpyImageList.from_csv('../input/', 'train.csv')

        .split_by_rand_pct(.1)

        .label_from_df(cols='label')

        .add_test(test, label=0)

        .transform(tfms)

        .databunch(bs=128, num_workers=0)

        .normalize(imagenet_stats))

data
data.show_batch(rows=5, figsize=(10,10))
dropout = 0.25

model = nn.Sequential(

    nn.Conv2d(in_channels=3,

              out_channels=32,

              kernel_size=5),

    nn.ReLU(),

    nn.BatchNorm2d(32),

    nn.MaxPool2d(kernel_size=2,

                 stride=2),

    nn.Dropout(dropout),

    

    nn.Conv2d(in_channels=32,

              out_channels=64,

              kernel_size=3),

    nn.ReLU(),

    nn.BatchNorm2d(64),

    nn.MaxPool2d(kernel_size=2,

                 stride=2),

    nn.Dropout(dropout),

    

    nn.Conv2d(in_channels=64,

              out_channels=64,

              kernel_size=3),

    nn.ReLU(),

    

    Flatten(),

    nn.Linear(64*3*3, 256),

    nn.ReLU(),

    nn.Dropout(dropout*1.5),

    nn.BatchNorm1d(256),

    nn.Linear(256, 10),

)

if torch.cuda.is_available():

    model = model.cuda()

learn = Learner(data, model, metrics=accuracy, model_dir='/kaggle/working/models')

learn.summary()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(20, max_lr=slice(1e-1))
learn.save('stage1')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-2))
learn.save('stage2')
predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)

# output to a file

submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})

submission_df.to_csv(f'submission.csv', index=False)