from fastai import *

from fastai.vision import *
path = Path('../input/aptos2019-blindness-detection/')
path.ls()
df = pd.read_csv(path/'train.csv')

df.head()
df_test = pd.read_csv(path/'test.csv')

df_test.head()
print(len(df))

print(len(df_test))
src = (

    ImageList.from_df(df,path,folder='train_images',suffix='.png')

        .split_by_rand_pct(0.1, seed=42)

        .label_from_df()

    )
tfms = get_transforms(max_warp=0, max_zoom=1.1, max_lighting=0.1, p_lighting=0.1)
data = (

    src.transform(tfms,size=128)

    .databunch()

    .normalize(imagenet_stats)

)
data.show_batch(rows=3, figsize=(5,5))
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

!cp '../input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'
kappa = KappaScore()

kappa.weights = "quadratic"
learn = cnn_learner(data, base_arch=models.resnet50, metrics=[error_rate, kappa], model_dir='/kaggle', pretrained=True)
learn.fit_one_cycle(4)
learn.save("/kaggle/working/stage-1")
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))
learn.save('/kaggle/working/stage-2')
sample_df = pd.read_csv(path/'sample_submission.csv')

sample_df.head()
learn.data.add_test(ImageList.from_df(sample_df,path,folder='test_images',suffix='.png'))
preds,y = learn.get_preds(DatasetType.Test)
sample_df.diagnosis = preds.argmax(1)

sample_df.head()
sample_df.to_csv('submission.csv',index=False)
from IPython.display import FileLink

FileLink('submission.csv')