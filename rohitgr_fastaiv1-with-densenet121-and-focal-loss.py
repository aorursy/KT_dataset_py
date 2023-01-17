from fastai.vision import *

from sklearn.metrics import roc_auc_score



import warnings

warnings.simplefilter('ignore', category=FutureWarning)

warnings.simplefilter('ignore', category=UserWarning)



%matplotlib inline
PATH = Path('../input')

PATH.ls()
train_df = pd.read_csv(PATH/'train_labels.csv')

print(train_df.shape)

train_df.head()
train_df['label'].value_counts(normalize=True)
src = (ImageItemList.from_csv(PATH, folder='train', csv_name='train_labels.csv', suffix='.tif')

      .random_split_by_pct(0.1, seed=77)

      .label_from_df()

      .add_test_folder())
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=0, max_zoom=1., max_lighting=0.05, max_warp=0)



data = (src.transform(tfms, size=96, resize_method=ResizeMethod.SQUISH)

       .databunch(bs=64, path='.'))



data.normalize(imagenet_stats);
data.show_batch(rows=5, figsize=(15, 15))
class FocalLoss(nn.Module):

    def __init__(self, alpha=1., gamma=1.):

        super().__init__()

        self.alpha = alpha

        self.gamma = gamma



    def forward(self, inputs, targets, **kwargs):

        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)

        pt = torch.exp(-CE_loss)

        F_loss = self.alpha * ((1-pt)**self.gamma) * CE_loss

        return F_loss.mean()



    

def roc_score(inp, target):

    _, indices = inp.max(1)

    return torch.Tensor([roc_auc_score(target, indices)])[0]
loss_func = FocalLoss(gamma=1.)

learn = create_cnn(data, models.densenet121, metrics=[accuracy, roc_score], loss_func=loss_func)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, 1e-3)
learn.recorder.plot_lr(show_moms=True)
learn.recorder.plot_losses()
learn.save('stage-1')
learn.load('stage-1');
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(title='Confusion matrix')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
max_lr = 1e-4

learn.fit_one_cycle(4, slice(1e-6, max_lr))
learn.save('stage-2')
learn.load('stage-2');
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(title='Confusion matrix')
auc_val = learn.validate()[2].item()
preds, y = learn.TTA(beta=0.4, ds_type=DatasetType.Test)

preds = torch.softmax(preds, dim=1)[:, 1].numpy()
test_ids = [f.stem for f in learn.data.test_ds.items]

subm = pd.read_csv(PATH/'sample_submission.csv')

orig_ids = list(subm['id'])
def create_submission(orig_ids, test_ids, preds):

    preds_dict = dict((k, v) for k, v in zip(test_ids, preds))

    pred_cor = [preds_dict[id] for id in orig_ids]

    df = pd.DataFrame({'id':orig_ids,'label':pred_cor})

    df.to_csv(f'submission_{auc_val}.csv', header=True, index=False)

    

    return df
test_df = create_submission(orig_ids, test_ids, preds)

test_df.head()