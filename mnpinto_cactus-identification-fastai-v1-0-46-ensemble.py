import fastai

from fastai.vision import *

from sklearn.model_selection import KFold
# Copy pretrained model weights to the default path

!mkdir '/tmp/.torch'

!mkdir '/tmp/.torch/models/'

#!cp '../input/resnet18/resnet18.pth' '/tmp/.torch/models/resnet18-5c106cde.pth'

#!cp '../input/densenet121/densenet121.pth' '/tmp/.torch/models/densenet121-a639ec97.pth'

!cp '../input/densenet201/densenet201.pth' '/tmp/.torch/models/densenet201-c1103571.pth'
fastai.__version__
data_path = Path('../input/aerial-cactus-identification')

df = pd.read_csv(data_path/'train.csv')

df.head()
sub_csv = pd.read_csv(data_path/'sample_submission.csv')

sub_csv.head()
def create_databunch(valid_idx):

    test = ImageList.from_df(sub_csv, path=data_path/'test', folder='test')

    data = (ImageList.from_df(df, path=data_path/'train', folder='train')

            .split_by_idx(valid_idx)

            .label_from_df()

            .add_test(test)

            .transform(get_transforms(flip_vert=True, max_rotate=20.0), size=128)

            .databunch(path='.', bs=64)

            .normalize(imagenet_stats)

           )

    return data
kf = KFold(n_splits=5, random_state=379)

epochs = 6

lr = 1e-2

preds = []

for train_idx, valid_idx in kf.split(df):

    data = create_databunch(valid_idx)

    learn = create_cnn(data, models.densenet201, metrics=[accuracy])

    learn.fit_one_cycle(epochs, slice(lr))

    learn.unfreeze()

    learn.fit_one_cycle(epochs, slice(lr/400, lr/4))

    learn.fit_one_cycle(epochs, slice(lr/800, lr/8))

    preds.append(learn.get_preds(ds_type=DatasetType.Test))
ens = torch.cat([preds[i][0][:,1].view(-1, 1) for i in range(5)], dim=1)

ens  = (ens.mean(1)>0.5).long(); ens[:10]
sub_csv['has_cactus'] = ens
sub_csv.to_csv('submission.csv', index=False)