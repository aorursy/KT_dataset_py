# Based on https://www.kaggle.com/abhisheksinghblr/kannada-mnist-using-fast-ai
from fastai.vision import *
from fastai.metrics import *
import imageio
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test  = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
valid = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
def to_img_shape(data_X, data_y=[]):
    data_X = np.array(data_X).reshape(-1,28,28)
    data_X = np.stack((data_X,)*3, axis=-1)
    data_y = np.array(data_y)
    return data_X,data_y

train_X, train_y = train.loc[:,'pixel0':'pixel783'], train['label']
valid_X, valid_y = valid.loc[:,'pixel0':'pixel783'], valid['label']

train_X, train_y = to_img_shape(train_X, train_y)
valid_X, valid_y = to_img_shape(valid_X, valid_y)
def save_imgs(path:Path, data, labels):
    path.mkdir(parents=True,exist_ok=True)
    for label in np.unique(labels):
        (path/str(label)).mkdir(parents=True,exist_ok=True)
    for i in range(len(data)):
        if(len(labels)!=0):
            imageio.imsave( str( path/str(labels[i])/(str(i)+'.jpg') ), data[i] )
        else:
            imageio.imsave( str( path/(str(i)+'.jpg') ), data[i] )

save_imgs(Path('/data/train'),train_X,train_y)
save_imgs(Path('/data/valid'),valid_X,valid_y)
path = Path('/data')
path.ls()
tfms = get_transforms(do_flip=False, max_rotate=30.0, max_zoom=1.25)
data = (ImageList.from_folder('/data/') 
        .split_by_folder()          
        .label_from_folder()            
        .transform(tfms, size=64)   
        .databunch())
data
data.show_batch(6,figsize=(8,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy])
learn.fit_one_cycle(5)
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-04))
learn.save('stage-2')
test.drop('id',axis = 'columns',inplace = True)
sub_df = pd.DataFrame(columns=['id','label'])
test = np.array(test)
def get_img(data):
    t1 = data.reshape(28,28)/255
    t1 = np.stack([t1]*3,axis=0)
    img = Image(FloatTensor(t1))
    return img
from fastprogress import progress_bar
mb=progress_bar(range(test.shape[0]))

for i in mb:
    timg = test[i]
    img = get_img(timg)
    sub_df.loc[i] = [i+1,int(learn.predict(img)[1])]
def decr(ido):
    return ido-1

sub_df['id'] = sub_df['id'].map(decr)
sub_df.head()
sub_df.to_csv('submission.csv',index=False)
