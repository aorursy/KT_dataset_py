%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import *

import imageio
path = Path('/kaggle/input/Kannada-MNIST')

path.ls()

train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

predict_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
train_data.describe()
train_data.head()
train_data.shape
predict_data.describe()
predict_data.head()
predict_data.shape
print(f'train_data shape : {train_data.shape}')

print(f'predict_data shape  : {predict_data.shape}')

def to_img_shape(data_X, data_y=[]):

    data_X = np.array(data_X).reshape(-1,28,28)

    data_X = np.stack((data_X,)*3, axis=-1)

    data_y = np.array(data_y)

    return data_X,data_y
train_data_X, train_data_y = train_data.loc[:,'pixel0':'pixel783'], train_data['label']
print(f'train_data shape : {train_data_X.shape}')

print(f'train_data_y shape : {train_data_y.shape}')
from sklearn.model_selection import train_test_split



train_X, validation_X, train_y, validation_y = train_test_split(train_data_X, train_data_y, test_size=0.20,random_state=7,stratify=train_data_y)

print(f'train_X shape : {train_X.shape}')

print(f'train_y shape : {train_y.shape}')

print(f'validation_X shape : {validation_X.shape}')

print(f'validation_y shape : {validation_y.shape}')
train_X,train_y = to_img_shape(train_X,train_y)

validation_X,validation_y = to_img_shape(validation_X,validation_y)
print(f'train_X shape : {train_X.shape}')

print(f'train_y shape : {train_y.shape}')

print(f'validation_X shape : {validation_X.shape}')

print(f'validation_y shape : {validation_y.shape}')
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

save_imgs(Path('/data/valid'),validation_X,validation_y)
path = Path('/data')

path.ls()
tfms = get_transforms(do_flip=False )
data = (ImageList.from_folder('/data/') 

        .split_by_folder()          

        .label_from_folder()        

        .add_test_folder()          

        .transform(tfms, size=64)   

        .databunch())
#Another way to create data bunch

#data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=64)
data
data.show_batch(5,figsize=(6,6))

data.classes, data.c, len(data.train_ds), len(data.valid_ds)
!mkdir -p /root/.cache/torch/checkpoints/

!cp /kaggle/input/fast-ai-models/resnet50-19c8e357.pth /root/.cache/torch/checkpoints/resnet50-19c8e357.pth
learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy], model_dir=Path('/kaggle/input/fast-ai-models'))
learn.fit_one_cycle(5)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(20, figsize=(20,20))
interp.plot_confusion_matrix(figsize=(20,20), dpi=100)
interp.most_confused(min_val=2)
learn.model_dir = '/kaggle/output/fast-ai-models/'
learn.lr_find()
learn.recorder.plot()
lr = slice(1e-04)
learn.save('stage-1')
learn.unfreeze()
learn.fit_one_cycle(3,lr)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(20, figsize=(20,20))
interp.plot_confusion_matrix(figsize=(20,20), dpi=100)
interp.most_confused(min_val=2)
learn.save('stage-2')
predict_data.drop('id',axis = 'columns',inplace = True)

sub_df = pd.DataFrame(columns=['id','label'])
my_predict_data = np.array(predict_data)
# Handy function to get the image from the tensor data

def get_img(data):

    t1 = data.reshape(28,28)/255

    t1 = np.stack([t1]*3,axis=0)

    img = Image(FloatTensor(t1))

    return img
from fastprogress import progress_bar

mb=progress_bar(range(my_predict_data.shape[0]))
for i in mb:

    timg=my_predict_data[i]

    img = get_img(timg)

    sub_df.loc[i]=[i+1,int(learn.predict(img)[1])]
def decr(ido):

    return ido-1



sub_df['id'] = sub_df['id'].map(decr)

sub_df.to_csv('submission.csv',index=False)
# Displaying the submission file

sub_df.head()