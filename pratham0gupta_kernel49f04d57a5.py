# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


!unzip ../input/facial-keypoints-detection/training.zip
df = pd.read_csv('../input/facial-keypoints-detection/IdLookupTable.csv')
train = pd.read_csv('./training.csv')
key_points = train.drop(['Image'],axis=1)
def extract_image(train):

    image_list = []

    for i in range(train.shape[0]):
        img = train['Image'][i].split(' ')
        
        img = np.array(list(map(int,img)))

        image_list.append(img)
    image_array = np.array(image_list).reshape((-1,96,96,1))
    return image_array
image_array = extract_image(train)
def show_img_and_points(df,index):
    '''A function to display image along with facial key points'''
    temp_point = df.iloc[index]

    x = []
    y = []
    points = df.iloc[index]
    for i in range(len(points)):
        if i%2 == 0:
            x.append(points[i])
        else:
            y.append(points[i])

    plt.imshow(image_array[index].reshape((96,96)),cmap='gray')
    plt.scatter(x,y,color='red')
    plt.show()
show_img_and_points(key_points,0)
print(key_points['left_eye_inner_corner_x'].isnull())
show_img_and_points(key_points,7048)
feature_names = key_points.columns
feature_x = [i for i in feature_names if 'x' in i]
feature_y = [i for i in feature_names if 'y' in i]
temp_point = key_points.iloc[7048]

x = []
y = []
points = key_points.iloc[7048]
for i in range(len(points)):
    if i%2 == 0:
        x.append(points[i])
    else:
        y.append(points[i])
        
plt.imshow(image_array[7045].reshape((96,96)),cmap='gray')
plt.scatter(x,y,color='red')
def fill_missing_data(key_points,feature):
    
    nose_mean_x = key_points['nose_tip_x'].mean()
    nose_mean_y = key_points['nose_tip_y'].mean()
    
    if feature in feature_x:
      
        feature_mean = key_points[feature].mean()
    
        row_index_with_missing_value = key_points[feature][key_points[feature].isnull()].index
        for i in row_index_with_missing_value:
            error_i = key_points['nose_tip_x'].iloc[i] - nose_mean_x 
            key_points[feature].iloc[i] = feature_mean + error_i
            if key_points[feature].iloc[i] > 96 or key_points[feature].iloc[i] < 0:
                key_points[feature].iloc[i] = feature_mean - 0.5
            
    else:
        
        feature_mean = key_points[feature].mean()
    
        row_index_with_missing_value = key_points[feature][key_points[feature].isnull()].index
        for i in row_index_with_missing_value:
            error_i = key_points['nose_tip_y'].iloc[i] - nose_mean_x 
            key_points[feature].iloc[i] = feature_mean + error_i
            if key_points[feature].iloc[i] > 96 or key_points[feature].iloc[i] < 0:
                key_points[feature].iloc[i] = feature_mean - 0.5
            
           
            
    
   
        
    
key_points_new = key_points.copy()
for i in feature_names:
    fill_missing_data(key_points_new,i)
    print(i)
    
    
print('key_points_new',key_points_new.isnull().sum())
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Activation, Dropout, MaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.models import load_model
#callback = EarlyStopping(monitor='val_loss',restore_best_weights=True)
model = Sequential()

model.add(Conv2D(32,(3,3),padding='same',use_bias=False, input_shape=(96,96,1)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())


model.add(Conv2D(32, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Conv2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Conv2D(128, (3,3),padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Conv2D(128, (3,3),padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Conv2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Conv2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30))
model.summary()
model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['mae'])
Y_train = key_points_new.values
X_train = image_array
model.fit(X_train,Y_train,epochs = 50,batch_size = 256,validation_split=0.2)
!unzip ../input/facial-keypoints-detection/test.zip

model.save('weigts3.h5')

model = load_model('weigts3.h5')
test = pd.read_csv('./test.csv')
test_img_array = extract_image(test)
test_img_array.shape
predictions = model.predict(test_img_array)

predictions[0]
points = predictions[24]
x = []
y = []
for i,p in enumerate(points):
    if i%2 == 0:
        x.append(p)
    else:
        y.append(p)
        
        
plt.imshow(test_img_array[24].reshape(96,96),cmap='gray')
plt.scatter(x,y,color='red')
        
#predictions_submit = predictions.reshape(-1,1)
submission = pd.DataFrame(predictions,columns=feature_names)
submission = pd.concat([test['ImageId'],submission],axis=1)
submission
lookid = pd.read_csv('../input/facial-keypoints-detection/IdLookupTable.csv')



for i in range(lookid.shape[0]):
    feature = lookid['FeatureName'].iloc[i]
    img_id = lookid['ImageId'].iloc[i]
    location = submission[feature][submission['ImageId']==img_id].values[0]
#     print(feature,img_id,location,sep='--')
    lookid['Location'].iloc[i] = location
   
def submission_prepare(lookid):
    for i in range(lookid.shape[0]):
        feature = lookid['FeatureName'].iloc[i]
        img_id = lookid['ImageId'].iloc[i]
        location = submission[feature][submission['ImageId']==img_id].values[0]
#     print(feature,img_id,location,sep='--')
        lookid['Location'].iloc[i] = location
    
    return lookid


lookid.iloc[i]['Location'] =0 
new = submission_prepare(lookid)
new.to_csv('submission_new2.csv',index=False)
predictions[predictions>96] = 96
predictions[predictions>96]
(new['Location']>96).sum()
from IPython.display import FileLink
FileLink(r'./weigts3.h5')
!cd
