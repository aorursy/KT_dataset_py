#Allows you to save your models somewhere without

# copying all the data over.

# Trust me on this.



!mkdir input

!cp /kaggle/input/train_labels.csv input

!cp /kaggle/input/sample_submission.csv input

!ln -s /kaggle/input/train/ input/train

!ln -s /kaggle/input/test/ input/test
from fastai.vision import *
tfms = get_transforms(flip_vert=True, max_warp=0, max_zoom=0, p_affine=0, max_rotate=0)

tfms[0][0].kwargs = {}

tfms[0][1] = dihedral()

tfms
from sklearn.model_selection import KFold



def data_gen(n_folds):

    il = ImageItemList.from_csv(csv_name='train_labels.csv', path='input', folder='train', suffix='.tif')

    

    idxs = array(range(len(il)))

    np.random.shuffle(idxs)

    kfold = KFold(n_splits=n_folds, shuffle=True)

    

    for curr_fold in kfold.split(idxs):

        val_idx = idxs[curr_fold[1]]

        db_split = (il

                    .split_by_idx(val_idx)

                    .label_from_df()

                    .transform(tfms, size=96)

                    .add_test_folder('test')

                    .databunch(bs=64)

                    .normalize(imagenet_stats))

        yield db_split

        

def get_output(db):

    learn = create_cnn(db, models.resnet50, metrics=[error_rate])

    learn.fit_one_cycle(4, max_lr=1e-2)

    learn.unfreeze()

    learn.fit_one_cycle(2, max_lr=slice(1e-5,1e-4))

    probs, _ = learn.TTA(ds_type=DatasetType.Test)

    preds = probs[:,1]

    return preds
n_folds = 4



sum_preds = None

gen = data_gen(n_folds)

for db in gen:

    preds = get_output(db)

    if sum_preds is None:

        sum_preds = preds

    else:

        sum_preds += preds
preds = sum_preds / n_folds
test_df = pd.read_csv('./input/sample_submission.csv')

test_df['id'] = [i.stem for i in db.test_ds.items]

test_df['label'] = preds
test_df.to_csv('submission.csv', index=False)