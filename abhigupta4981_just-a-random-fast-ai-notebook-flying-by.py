from fastai import *

from fastai.vision import *

from sklearn.model_selection import KFold
!ls ../input
PATH = Path('../input')
def create_data(valid_idx):

    test = ImageList.from_df(sub_csv, path=PATH/'test', folder='test')

    data = (ImageList.from_df(df, path=PATH/'train', folder='train')

            .split_by_idx(valid_idx)

            .label_from_df()

            .add_test(test)

            .transform(get_transforms(flip_vert=True, max_rotate=25, max_zoom=1.2, max_lighting=0.3))

            .databunch(path='.', bs=64)

            .normalize(imagenet_stats)

           )

    return data    
df = pd.read_csv(PATH/'train.csv')

df.head()
sub_csv = pd.read_csv(PATH/'sample_submission.csv')

sub_csv.head()
kf = KFold(n_splits=5, random_state=5)

epochs = 6
lr = 1e-2
preds = []

for train_idx, valid_idx in kf.split(df):

    data = create_data(valid_idx)

    learn = create_cnn(data, models.densenet201, metrics=[accuracy])

    learn.fit_one_cycle(epochs, slice(lr))

    learn.unfreeze()

    learn.fit_one_cycle(epochs, slice(lr/400, lr/4))

    learn.fit_one_cycle(epochs, slice(lr/800, lr/8))

    preds.append(learn.get_preds(ds_type=DatasetType.Test))
ens = torch.cat([preds[i][0][:,1].view(-1, 1) for i in range(5)], dim=1)
ens1  = (ens.mean(1)>=0.5).long(); ens1[:10]
ens2 = (ens.mean(1)>0.5).long(); ens2[:10]
sub_csv['has_cactus'] = ens1
sub_csv.to_csv('submission.csv', index=False)
sub_csv['has_cactus'] = ens2
sub_csv.to_csv('submission2.csv', index=False)