from fastai import *

from fastai.vision import *

from sklearn.metrics import f1_score



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything(43)
train = pd.read_csv('../input/clabscvcomp/data/train.csv')

test_df = pd.read_csv('../input/clabscvcomp/data/sample_submission.csv')



train.head() ## Shows the first five rows of data frame
sorted(train.genres.unique()) ## Shows all classes in the dataframe
train.genres.value_counts(normalize=True) ## Distribution of dataset
sz = (512, 512) ## Image size

bs = 32 ## Batch size

tfms = get_transforms( ## Transformation to apply on Train data

    do_flip=True, ## Horizontal flip

    flip_vert=False, ## Vertical flip

    max_rotate=20, ## Rotation

    max_zoom=1.5, ## Center zoom

    max_lighting=0.75 ## lighting

)
plt.imread('../input/clabscvcomp/data/train_data/1000.jpg').shape
data = (

    ImageList.from_df(df=train, path='', folder='../input/clabscvcomp/data/train_data/', cols='id', suffix = '.jpg') ## define data path

    .split_by_rand_pct(valid_pct=0.2) ## validation split

    .label_from_df(cols='genres') ## load labels from

    .transform(tfms, size=sz)

    .databunch(bs=bs, num_workers=4) 

    .normalize(imagenet_stats)

    )
test_data = ImageList.from_df(test_df, path='../input/clabscvcomp/data/test_data/', cols='id', suffix = '.jpg')

data.add_test(test_data)
data.show_batch(3)
def F1(y_pred, y):

    y_pred = y_pred.softmax(dim=1) 

    y_pred = y_pred.argmax(dim=1)

    return torch.tensor(f1_score(y.cpu(), y_pred.cpu(), labels=list(range(10)), average='weighted'),device='cuda:0')
learn = cnn_learner(

                    data, ## DataBunch

                    models.resnet18, ## Resnet50 

                    metrics=[F1, accuracy], ## Matrices

                    callback_fns=ShowGraph ## Allows us to visualize training

).mixup().to_fp16()
learn.freeze() 

learn.fit_one_cycle(3)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(20, max_lr=slice(1e-4, 1e-3))
preds = learn.get_preds(ds_type=DatasetType.Test) ## get prediction in test data

np.save("preds.npy", preds[0].numpy())

preds = np.argmax(preds[0].numpy(),axis = 1)

categories = sorted(train.genres.unique().astype('str'))

final_preds = []

for idx in preds:

    final_preds.append(categories[idx])

final_submit = pd.read_csv('../input/clabscvcomp/data/sample_submission.csv')

final_submit.genres = final_preds

final_submit.head()

final_submit.to_csv('submission.csv',index = False)