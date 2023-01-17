!pip install fastai --upgrade -q

from fastai.vision.all import *

# from fastai.callback.cutmix import *



!pip install wwf -q

!pip install timm -q

from wwf.vision.timm import *



# !pip install efficientnet_pytorch -q

# from efficientnet_pytorch import EfficientNet
path = Path('../input/planets-dataset/planet/planet')
train_df = pd.read_csv(path/'train_classes.csv')

train_df
def get_x(r):

    return path/'train-jpg'/(r['image_name']+'.jpg')



def get_y(r):

    return r['tags'].split()



def get_data(size=224,bs=64,data_df=train_df):

    dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),

                       splitter=RandomSplitter(seed=42),

                       get_x=get_x, 

                       get_y=get_y,

                       item_tfms = Resize(size),

                       batch_tfms = [*aug_transforms(flip_vert=True,max_warp=0),

                                     Normalize.from_stats(*imagenet_stats)]

                      )

    return dblock.dataloaders(data_df,bs=bs)
dls = get_data(300,40)
dls.show_batch(nrows=2, ncols=3)
# f2macro = FBetaMulti(beta=2,average='macro')

# f2micro = FBetaMulti(beta=2,average='micro')

f2samples = FBetaMulti(beta=2,average='samples',thresh=0.2)



# model = EfficientNet.from_pretrained('efficientnet-b7', get_c(dls))

metrics = [partial(accuracy_multi, thresh=0.2), f2samples]

cbs = [MixUp,

      # SaveModelCallback(monitor='fbeta_score')

      ] 
# learn = cnn_learner(dls, resnet50, metrics=metrics, cbs=cbs)

# learn = Learner(dls, model, metrics=metrics, f2samples], cbs=cbs)



learn = timm_learner(dls, 'efficientnet_b3', metrics=metrics, cbs=cbs)



# learn.lr_find()
learn.fine_tune(15, base_lr=3e-2, freeze_epochs=6)
# def f2_score(y_pred, y_true, threshold=0.5, beta=2, eps=1e-9):

#     y_pred = (y_pred > threshold).float()



#     true_positives  = (y_pred * y_true)

#     true_negatives  = ((y_pred + y_true) == 0.).float()

#     false_positives = ((y_pred - y_true) == 1.).float()

#     false_negatives = ((y_true - y_pred) == 1.).float()

    

#     precision = true_positives.sum(dim=1) / ((true_positives + false_positives).sum(dim=1) + eps)

#     recall    = true_positives.sum(dim=1) / ((true_positives + false_negatives).sum(dim=1) + eps)

    

#     score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + eps)



#     return torch.mean(score)
# preds,targs = learn.get_preds()



# xs = torch.linspace(0.05,0.95,29)

# accs = [f2_score(preds, targs, threshold=i) for i in xs]

# plt.plot(xs,accs);
file_path = Path('../input/planets-dataset/test-jpg-additional/test-jpg-additional')

test_path = Path('../input/planets-dataset/planet/planet/test-jpg')

submission_df = pd.read_csv(path/'sample_submission.csv')

testing_path = (submission_df['image_name'] + '.jpg').apply(lambda x: test_path/x if x.startswith('test') else file_path/x)



def prediction(filename='submission.csv', tta=False):

    tst_dl = learn.dls.test_dl(testing_path)

    if tta:

        predictions = learn.tta(dl = tst_dl)

    else:

        predictions = learn.get_preds(dl = tst_dl)

    predlist = [' '.join(learn.dls.vocab[i]) for i in (predictions[0] > 0.2)]



    df = submission_df

    df['tags'] = predlist



    df.to_csv(filename, index=False)

    return df
prediction('submission_tta.csv', tta=True)
# dls = get_data(size=448,bs=32)

# learn.dls = dls

# learn.freeze()

# learn.lr_find()
# learn.fine_tune(12, base_lr=3e-3, freeze_epochs=4)
# prediction('submission_tta_2.csv', tta=True)