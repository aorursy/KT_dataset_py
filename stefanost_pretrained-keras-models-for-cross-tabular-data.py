import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/cicids2017/train.csv')
test = pd.read_csv('../input/cicids2017/test.csv')
df.head()
df.info()
#testset number of rows and columns
print('number of rows:',test.shape[0])
print('number of columns:',test.shape[1])
print('')
# function to identify which columns are numerical, which categorical, store in two variables
# print some results
#input:dataframe, preferably with the label column sliced off
#returns two lists with names of categorical and numeric features

def column_types(df):
    numeric_columns = []
    categorical_columns = []
    
    for column in df.columns:
        if df[column].dtype != 'object':
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    
    print('number of numeric columns:',len(numeric_columns))
    print('column names:')
    print( numeric_columns)
    print('')
    
    print ('number of categorical columns:', len(categorical_columns))
    print ('column names:', categorical_columns)
    print('')
    print('Number of Unique Values:')

    [print ('{}:'.format(column),df[column].nunique()) 
     for column in categorical_columns]

    [print('\n\ncolumn {}:\n'.format(column),df[column].value_counts()) 
     for column in categorical_columns]
    
    return numeric_columns, categorical_columns

numeric_columns, categorical_columns = column_types(df.iloc[:,:-1])
# report on labels 
#input:dataframe and name of label column
#displays class information

def label_report(df,label_column_name):
    
    print ('number of classes:', df[label_column_name].nunique())
    print('')
    print ('class names:', np.unique(df[label_column_name]))
    print('')
    print('Number of Unique Values:')
    print(df[label_column_name].value_counts())


print('Train Data')
print('__________')
label_report(df,'label')
print('')
print('__________')
print('')
print('Test Data')
print('__________')
print('Number of Unique Values:')
print(test['label'].value_counts())
#function to resample dataset in balanced form
#majority classes may be downsampled, minority classes will be upsampled
#input:pandas dataframe, output:pandas dataframe
#params:df, name of label column, number of samples each class will have

def balanced_sampling(df, label_column_name, n_samples):
    import numpy as np
    import pandas as pd
    
    #identify majority and minority classes
    minority_classes = []
    majority_classes = []
    for value,index in zip(df['label'].value_counts(),df['label'].value_counts().index):
        if value<n_samples:
            minority_classes.append(index)
        else:
            majority_classes.append(index)
    
    #sample each class
    #oversample minority classes (replace=True)
    balanced_df_min = df[df[label_column_name].isin(minority_classes)].groupby(by=df[label_column_name]).\
    apply(lambda x: x.sample(n_samples, replace=True))
    #downsample majority classes (replace=False)
    balanced_df_maj = df[df[label_column_name].isin(majority_classes)].groupby(by=df[label_column_name]).\
    apply(lambda x: x.sample(n_samples, replace=False))
    #combine dataframes
    balanced_df = pd.concat((balanced_df_min,balanced_df_maj),axis=0)
    
    #shuffle new dataframe
    #(because it's sorted by class)
    rand_state = np.random.RandomState(seed=33)
    indices=np.arange(balanced_df.shape[0])
    rand_state.shuffle(indices)
       
    return balanced_df.iloc[indices].reset_index(drop=True)



data=balanced_sampling(df,'label', 200000)

# function that displays information about the resampling results
#input: original dataframe, new dataframe, name of label column

def sampling_results(df_orig, df_final, label_column_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print('original dataset rows:', df_orig.shape[0])
    print('new dataset rows:     ', df_final.shape[0])
    print ('')
    print ('original dataset: samples per class')
    print(df_orig[label_column_name].value_counts())
    print('')
    print('new dataset contains:')
    print('{} values per class'.\
          format(df_final[label_column_name].value_counts()[0]))
    
    plt.figure(figsize=(12,4));
    sns.countplot(x=label_column_name, data=df_orig, color='Gray');
    plt.title('original dataset -- samples per class');
    plt.figure(figsize=(12,4));
    sns.countplot(x=label_column_name, data=df_final, color='Gray');
    plt.title('new dataset -- samples per class');
    
sampling_results(df,data,'label')
#function to prepare input data
#input:dataframe without labels
#output1: numpy array of values of dataframe with categorical one-hot encoded
#                                                        and all data scaled
#output2: sklearn objects encoder and scaler, to apply .transform on test data, and to use attributes for reference

def prepare_input_data(df, scaling='minmax'):
    
    
    #scaling
    if scaling == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df = scaler.fit_transform(df)
    elif scaling == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler == StandardScaler()
        data = scaler.fit_transform(df)

    return df, scaler


x_tr, scaler = prepare_input_data(data.iloc[:,:-1])

x_ts = scaler.transform(test.iloc[:,:-1])
# export scaled samplings
np.save('CICIDS2017_mydata_tr.npy',x_tr)
np.save('CICIDS2017_mydata_ts.npy',x_ts[:200000,:])
np.save('CICIDS2017_mylabels_tr.npy',data.iloc[:,-1])
np.save('CICIDS2017_mylabels_ts.npy',test.iloc[:200000,-1])
x_tr = np.load('./CICIDS2017_mydata_tr.npy')
x_ts = np.load('./CICIDS2017_mydata_ts.npy')
y_tr = np.load('./CICIDS2017_mylabels_tr.npy')
y_ts = np.load('./CICIDS2017_mylabels_ts.npy')

x_tr.shape,y_tr.shape, x_ts.shape,y_ts.shape
#convert input data to images
#converts to square matrices, pads with zeros

#the function converts:
#1d arrays into 2d images
#2d arrays into sequence of 2d images (an image for each row)

def to_image(array):
    
    if array.ndim == 1:
        t = array.shape[0]
        sqr = int(np.round(np.sqrt(t)))
        
        if sqr**2 == t:
            im=np.reshape(array.copy(),(sqr,sqr))

        else:
            im = np.zeros((sqr+1,sqr+1))
            dif = (sqr+1)**2-t
            im = np.ravel(im)
            im[:-dif] = array.copy()
            im = np.reshape(im,(sqr+1,sqr+1))
            
        return np.array(im)
     
        
    elif array.ndim ==2:
        t=array.shape[1]
        sqr=int(np.round(np.sqrt(t)))
        
        if sqr**2 ==t:
            ims = np.zeros((array.shape[0],sqr,sqr))
            for i, im in enumerate(ims):
                im = np.reshape(array[i].copy(),(sqr,sqr))
        else:
            ims = np.zeros((array.shape[0],sqr+1,sqr+1))
            dif = (sqr+1)**2 - t
            for i, im in enumerate(ims):
                temp = np.ravel(im)
                temp[:-dif] = array[i].copy()
                im = np.reshape(temp,(sqr+1,sqr+1))
        
        return np.array(ims)
      
        
    else:
        print('wrong dimensions, 1d or 2d only')



        

#convert to rgb, keras pretrained need 3 color channels
#we copy the first channel to the other two

def to_rgb(array):
    
    images = np.empty((array.shape[0],10,10,3))
    
    for i,image in enumerate(array):
        
        images[i,:,:,1] = np.squeeze(image)
        images[i,:,:,2] = np.squeeze(image)
        
    return images





#transform each image to 80x80x3, to make compatible with InceptionV3 pretrained model
#we repeat the 10x10x3 matrix horizontally and vertically to get 80x80x3

def expand(array):


    
    expanded = np.repeat(array,8,axis=2)
    expanded = np.repeat(expanded,8,axis=1)
        
    return expanded
#visualize two images for each class

from skimage.io import imshow

for class_ in np.unique(y_tr):
    indices = np.random.choice \
    (np.argwhere(y_tr==class_).flatten(),2)
    class_ims = to_image(x_tr[indices])
    class_ims = to_rgb(class_ims)
    class_ims = expand(class_ims)
    
    plt.figure(figsize=(16,4));
    plt.subplot(1,3,1);
    imshow(class_ims[0]);
    plt.title(class_)
    plt.subplot(1,3,2);
    imshow(class_ims[1]);
    plt.title(class_);

print('image shape:',class_ims[1].shape)
import tensorflow as tf
import keras
from keras import backend as K
from keras import models
from keras.models import Model, load_model
from keras import layers
from keras.layers import Dense,Conv2D,MaxPooling2D, Flatten, Input
from keras.layers.core import Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

#weight and notops from pretrained models
!mkdir ~/.keras
!mkdir ~/.keras/models
!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/
!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/
#we will use an InceptionV3 model
conv_base = keras.applications.InceptionV3(include_top=False,weights='imagenet',input_shape=(80,80,3))
conv_base.summary()
#complete pipeline
# coefficient 'pack_size' determines chunks that will be processed at each iteration
# too large 'pack_size' puts burden on memory, too small adds time complexity
def pipeline(array, conv_base):
    
    array = to_image (array)
    pack_size = 10000
    first_dim_fract = array.shape[0]//pack_size
    array = to_rgb(array)
    features = []

    for i in (np.arange(first_dim_fract)+1):
        
        first_index = (i-1)*pack_size
        second_index = i*pack_size
        print('chunk of index:{}-{}'.format(first_index,second_index))
        temp = expand(array[first_index:second_index])
        temp = conv_base.predict(temp)
        features.append(temp)
    
    return np.reshape(features,(first_dim_fract*pack_size,1, 1, 2048))
#extract features in chunks to manage memory
train_features = pipeline(x_tr[:100000],conv_base)
train_features = np.append(train_features,pipeline(x_tr[100000:200000],conv_base),axis=0)
train_features = np.append(train_features,pipeline(x_tr[200000:300000],conv_base),axis=0)
np.save('CICIDS2017_InceptionV3_train_features.npy',train_features)
#train_features = np.load('./CICIDS2017_InceptionV3_train_features.npy')
test_features = pipeline(x_ts[:100000],conv_base)
np.save('CICIDS2017_InceptionV3_test_features.npy',test_features)
#test_features = np.load('')
#reshape to fit classifier
train_features = np.reshape(train_features,(train_features.shape[0],2048))
test_features = np.reshape(test_features,(test_features.shape[0],2048))
#train a classifier with the feature maps of the pretrained keras model

model = models.Sequential()
model.add(layers.Dense(256,activation='relu',input_dim=train_features.shape[1]))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

stop= EarlyStopping(patience=4, verbose=1)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
model.fit(train_features,y_tr,epochs=30,batch_size=20,callbacks=stop,validation_split=0.1)
#evaluate performance

pred = model.predict(test_features)
pred = np.squeeze((pred>0.5).astype(np.uint8))
print('accuracy:',accuracy_score(y_ts[:100000],pred))
print('f1-score:',f1_score(y_ts[:100000],pred))
plt.figure(figsize=(9,7));
conf=confusion_matrix(y_ts[:100000],pred)
sns.heatmap(conf, annot=True, fmt='1d');