from fastai import *

from fastai.vision import *

from fastai.callbacks import *



DATAPATH = Path('/kaggle/input/Kannada-MNIST/')



import os

for dirname, _, filenames in os.walk(DATAPATH):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv(DATAPATH/'train.csv')

df_train['fn'] = df_train.index
class PixelImageItemList(ImageList):

    def open(self,fn):

        img_pixel = self.inner_df.loc[self.inner_df['fn'] == int(fn[2:])].values[0,1:785]

        img_pixel = img_pixel.reshape(28,28)

        return vision.Image(pil2tensor(img_pixel,np.float32).div_(255))
src = (PixelImageItemList.from_df(df_train,'.',cols='fn')

      .split_by_rand_pct()

      .label_from_df(cols='label'))
tfms=get_transforms(do_flip=False)

with_tta = ([],tfms[0])
bs = 1024

data = (src.transform(tfms=with_tta)

       .databunch(num_workers=2,bs=bs)

       .normalize())
data.show_batch(rows=3,figsize=(4,4),cmap='bone')
best_architecture = nn.Sequential(

    conv_layer(1,32,stride=1,ks=3),

    conv_layer(32,32,stride=1,ks=3),

    conv_layer(32,32,stride=2,ks=5),

    nn.Dropout(0.4),

    

    conv_layer(32,64,stride=1,ks=3),

    conv_layer(64,64,stride=1,ks=3),

    conv_layer(64,64,stride=2,ks=5),

    nn.Dropout(0.4),

    

    Flatten(),

    nn.Linear(3136, 128),

    relu(inplace=True),

    nn.BatchNorm1d(128),

    nn.Dropout(0.4),

    nn.Linear(128,10),

    nn.Softmax(-1)

)
learn = Learner(data, best_architecture, loss_func = nn.CrossEntropyLoss(), metrics=[accuracy]).mixup()
callbacks = [

SaveModelCallback(learn, monitor='valid_loss', mode='min',name='bestweights'),

ShowGraph(learn),

EarlyStoppingCallback(learn, min_delta=1e-5, patience=3),

]

learn.callbacks = callbacks
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(50,1e-2)
df_test = pd.read_csv(DATAPATH/'test.csv')

df_test.rename(columns={'id':'label'}, inplace=True)

df_test['fn'] = df_test.index

test_set = PixelImageItemList.from_df(df_test,path='.',cols='fn')
learn.data.add_test(test_set)
preds, _ = learn.get_preds(DatasetType.Test)

preds.unsqueeze_(1);preds.shape
num_preds = 14

for x in range(num_preds):

    new_preds, _ = learn.get_preds(DatasetType.Test)

    new_preds.unsqueeze_(1);preds.shape

    preds = torch.cat((preds,new_preds),1)
preds.shape
indv_preds = torch.argmax(preds, dim=2);indv_preds.shape
winner = torch.mode(indv_preds, dim=1).values;winner.shape
submission = pd.DataFrame({ 'id': np.arange(0,len(winner)),'label': winner })
submission.to_csv(path_or_buf ="submission.csv", index=False)