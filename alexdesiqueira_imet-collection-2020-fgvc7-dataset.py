from fastai.vision import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
%matplotlib inline
batch_size = 400
path_input = Path('/kaggle/input/imet-2020-fgvc7')

folder_train = 'train'

folder_test = 'test'



path_output = Path('/kaggle/working')
train_csv = pd.read_csv(path_input/'train.csv')

train_csv.head()
np.random.seed(42)  # the "Answer" is 42! 

source = (ImageList.from_csv(path=path_input,

                             csv_name='train.csv',

                             folder=folder_train,

                             suffix='.png')

                   .split_by_rand_pct(0.2)

                   .label_from_df(label_delim=' ')

         )
aug_transforms = get_transforms(max_lighting=0.1,

                                max_zoom=1.05,

                                max_warp=0.)
data = (source.transform(aug_transforms, size=128)

              .databunch(bs=batch_size)

              .normalize(imagenet_stats)

       )
data.show_batch(rows=3, figsize=(10, 10))
# copying the ResNet-50 structure...

!mkdir -p /root/.cache/torch/checkpoints

!cp /kaggle/input/resnet/resnet50-19c8e357.pth /root/.cache/torch/checkpoints/resnet50-19c8e357.pth



# ... and the previously trained models.

!cp /kaggle/input/pretrained-models/imet-stage1_size128.pth /kaggle/working

!cp /kaggle/input/pretrained-models/imet-stage2_size128.pth /kaggle/working
architecture = models.resnet50
acc_02 = partial(accuracy_thresh, thresh=0.2) 

f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner(data,

                    architecture,

                    metrics=[acc_02, f_score],

                    model_dir=path_output)
# learn.lr_find()

# learn.recorder.plot(suggestion=True)
learning_rate = 3.31E-02  # as suggested by fastai's lr_find()
# learn.fit_one_cycle(100, slice(learning_rate))

# learn.save('imet-stage1_size128')
learn.load('imet-stage1_size128')

learn.save('imet-stage1_size128')

learn.unfreeze()
# learn.lr_find()

# learn.recorder.plot(suggestion=True)



learning_rate = 1.74E-05  # fastai's lr_find() suggested 8.32E-06 this time. I was sleeping, so I'll try this on a next iteration :)
# learn.fit_one_cycle(100, slice(learning_rate, learning_rate/10), wd=0.1)

# learn.save('imet-stage2_size128')
learn.load('imet-stage2_size128')

learn.save('imet-stage2_size128')
learn.freeze()

learn.export(path_output/'export.pkl')
itemlist_test = ItemList.from_folder(path_input/folder_test)

learn = load_learner(path=path_output, test=itemlist_test)

predictions, _ = learn.get_preds(ds_type=DatasetType.Test)
threshold = 0.1

labelled_preds = [' '.join([learn.data.classes[idx] for idx, pred in enumerate(prediction) if pred > threshold]) for prediction in predictions]

filenames = [f.name[:-4] for f in learn.data.test_ds.items]
data_frame = pd.DataFrame({'id':filenames, 'attribute_ids':labelled_preds},

                           columns=['id', 'attribute_ids'])

data_frame.to_csv(path_output/'submission.csv',

                  index=False)