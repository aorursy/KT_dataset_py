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
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import timeit, time


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.experimental import WideDeepModel as WDM

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,f1_score,precision_score,balanced_accuracy_score

from IPython.display import Image
from IPython.core.display import HTML 


from plotly.offline import iplot, init_notebook_mode, plot, download_plotlyjs
init_notebook_mode()
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.subplots import make_subplots
url = 'https://i.pinimg.com/originals/e3/f4/f1/e3f4f140bbd3716c76d18a00ea11f22e.jpg'
  
Image(url= url, width=600, height=600, unconfined=True)

df = pd.read_csv('../input/age-gender-and-ethnicity-face-data-csv/age_gender.csv')
df.drop(columns= ['img_name'], axis=1, inplace= True)

df.head()
# Copied from starter code. 
def plot(X,y):
    for i in range(3):
        plt.title(y[i],)
        plt.imshow(X[i].reshape(48,48))
        plt.show()
        
x_dis = df['pixels'][0:3].apply(lambda x:  np.array(x.split(), dtype="float32"))
y_dis= df['ethnicity'][0:3]

plot(x_dis, y_dis)
sns.barplot(y=list(df['ethnicity'].value_counts().values),x= list(df['ethnicity'].value_counts().index))
sns.barplot(y=list(df['gender'].value_counts().values),x= list(df['gender'].value_counts().index))
sns.distplot(df['age'])
# Find the average of all emotion counts
m = df.groupby('ethnicity').count().mean().values[0]
#print("Mean of all ethnicity counts: " + str(m))

ethnicity = list(df.ethnicity.unique())

oversampled = pd.DataFrame()
for n in ethnicity:
    #print('\n' + n)
    l = len(df[df.ethnicity==n])
    print('Before sampling: ' + str(l))
    
    if (l>=m):
        dft = df[df.ethnicity==n].sample(int(m))
        oversampled = oversampled.append(dft)
        #print('Ater sampling: ' + str(len(dft)))
    else:
        frac = int(m/l)
        dft = pd.DataFrame()
        for i in range(frac+1):
            dft = dft.append(df[df.ethnicity==n])
            
        dft = dft[dft.ethnicity==n].sample(int(m))
        oversampled = oversampled.append(dft)
        #print('Ater sampling: ' + str(len(dft)))
        
oversampled = oversampled.sample(frac=1).reset_index().drop(columns=['index'])

sns.barplot(y=list(oversampled['ethnicity'].value_counts().values),x= list(oversampled['ethnicity'].value_counts().index))

# split the data into train and test sets. 
X = oversampled.drop(['ethnicity'], axis=1)
y= np.array(oversampled['ethnicity'].values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True )

# spliting the image data and tabular data as input a and b. 
#Converting the pixel data into array. 

X_test_A = X_test['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32")) #converting data to numpy array
X_test_A = np.array(X_test_A)/255.0 #normalization

X_t = []
for i in range(X_test_A.shape[0]):
    X_t.append(X_test_A[i].reshape(48,48,1)) #reshaping the data to (n,48,48)
    
X_test_A_shaped = np.array(X_t)
print(len(X_test_A_shaped))

X_train_A = X_train['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32")) #converting data to numpy array
X_train_A = np.array(X_train_A)/255.0 #normalization

X_t_a = []
for i in range(X_train_A.shape[0]):
    X_t_a.append(X_train_A[i].reshape(48,48,1)) #reshaping the data to (n,48,48)
    
X_train_A_shaped = np.array(X_t_a)
print(len(X_train_A_shaped))
print(X_train_A_shaped.shape)

# remove the pixels from the x input data. 
X_test_B = X_test.drop(['pixels'], axis=1)
X_train_B = X_train.drop(['pixels'], axis=1)


train = pd.DataFrame(data=y_train, columns=['ethnicity'])
test = pd.DataFrame(data=y_test, columns=['ethnicity'])

f, axes = plt.subplots(1, 2, figsize=(12, 5), sharex= True)
axes[0].set_title('Test Data split')
axes[1].set_title('Train Data split')
sns.barplot(y=list(train['ethnicity'].value_counts().values),x= list(train['ethnicity'].value_counts().index),ax=axes[0])
sns.barplot(y=list(test['ethnicity'].value_counts().values),x= list(test['ethnicity'].value_counts().index),ax=axes[1])


def scaler_std(series):
    '''
    input= df['series']
    output= scaled series
   
    '''
    mean = series.values.mean()
    std = series.values.std()
    return series.apply(lambda x: (x-mean)/std)

X_test_B['age'] = scaler_std(X_test_B['age'])
X_train_B['age'] = scaler_std(X_train_B['age'])

# the gender column to be encoded. 
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X_test_B = ct.fit_transform(X_test_B)
X_train_B = ct.fit_transform(X_train_B)
url = 'https://www.researchgate.net/profile/Kaveh_Bastani/publication/328161216/figure/fig3/AS:679665219928064@1539056224036/Illustration-of-the-wide-and-deep-model-which-is-an-integration-of-wide-component-and.ppm'
    
Image(url= url, width=600, height=600, unconfined=True)
#Wide and Deep model with SGD optimizer/ activation = selu 

start_time = time.time()
# model using image information and batch normalization
model_image_selu = keras.models.Sequential(
[
    keras.layers.Flatten(input_shape=[48,48,1]),
    keras.layers.Dense(300, activation= 'selu', kernel_initializer= 'lecun_normal'),
    keras.layers.Dense(100, activation= 'selu', kernel_initializer= 'lecun_normal'),
    keras.layers.Dense(50, activation=  'selu', kernel_initializer= 'lecun_normal'),
    keras.layers.Dense(5,activation= 'softmax')
   
])

#model using tabular data
model_data_selu = keras.models.Sequential(
[
    keras.layers.Input(shape=[3]),
    keras.layers.Dense(5,activation= 'softmax')
])

# compile the NN
combined_model = WDM(model_data_selu,model_image_selu)
combined_model.compile(loss= 'sparse_categorical_crossentropy', optimizer='sgd',metrics= ['SparseCategoricalAccuracy'])



# fit the data
input_x = [X_train_B, X_train_A_shaped]
input_y = y_train

val_data = [X_test_B, X_test_A_shaped], y_test

combined_model.fit(input_x, input_y, validation_data=val_data, shuffle= True, epochs=250,verbose=0, batch_size= 50)

selu_data_image_time = (time.time() - start_time)
print("--- %s seconds ---" % (time.time() - start_time))
# data and image info using optimizer 'adam'
model_image = keras.models.Sequential(
[
    keras.layers.Flatten(input_shape=[48,48,1]),
    keras.layers.Dense(300, activation= 'relu'),
    keras.layers.Dense(100, activation= 'relu'),
    keras.layers.Dense(50, activation= 'relu'),
    keras.layers.Dense(5,activation= 'softmax')
   
])

#tried with activation = 'selu' didnt increase the accuracy. The value remained around 0.790 for test data and train data was about 0.9. However 'relu' yeilds better. 


model_data = keras.models.Sequential(
[
    keras.layers.Input(shape=[3]),
    keras.layers.Dense(3, activation= 'relu'),
    keras.layers.Dense(5, activation= 'softmax')
])

model = WDM(model_data,model_image)


#data and image with adam optimizer
start_time = time.time()

input_x = [X_train_B, X_train_A_shaped]
input_y = y_train

val_data = [X_test_B, X_test_A_shaped], y_test


model.compile(loss= 'sparse_categorical_crossentropy', optimizer='adam',metrics= ['SparseCategoricalAccuracy'])

model.fit(input_x, input_y, validation_data=val_data, shuffle= True, epochs=250,verbose=0, batch_size= 50)

adam_time_data_image = (time.time() - start_time)
print("--- %s seconds ---" % (time.time() - start_time))
#data and image with sgd optimizer

start_time = time.time()

input_x = [X_train_B, X_train_A_shaped]
input_y = y_train

val_data = [X_test_B, X_test_A_shaped], y_test


model_image_sgd = keras.models.Sequential(
[
    keras.layers.Flatten(input_shape=[48,48,1]),
    keras.layers.Dense(300, activation= 'relu'),
    keras.layers.Dense(100, activation= 'relu'),
    keras.layers.Dense(50, activation= 'relu'),
    keras.layers.Dense(5,activation= 'softmax')
   
])

model_data_sgd = keras.models.Sequential(
[
    keras.layers.Input(shape=[3]),
    keras.layers.Dense(3, activation= 'relu'),
    keras.layers.Dense(5, activation= 'softmax')
])

model_sgd = WDM(model_data_sgd,model_image_sgd)

model_sgd.compile(loss= 'sparse_categorical_crossentropy', optimizer='sgd',metrics= ['SparseCategoricalAccuracy'])

model_sgd.fit(input_x, input_y, validation_data=val_data, shuffle= True, epochs=250,verbose=0, batch_size= 50)

sgd_data_image_time = (time.time() - start_time)
print("--- %s seconds ---" % (time.time() - start_time))
#Model with image data using sgd optimizer / activation = selu

start_time = time.time()
# model using image information and batch normalization
selu_image = keras.models.Sequential(
[
    keras.layers.Flatten(input_shape=[48,48,1]),
    keras.layers.Dense(300, activation= 'selu', kernel_initializer= 'lecun_normal'),
    keras.layers.Dense(100, activation= 'selu', kernel_initializer= 'lecun_normal'),
    keras.layers.Dense(50, activation=  'selu', kernel_initializer= 'lecun_normal'),
    keras.layers.Dense(5,activation= 'softmax')
   
])

# compile the NN
selu_image.compile(loss= 'sparse_categorical_crossentropy', optimizer='sgd',metrics= ['SparseCategoricalAccuracy'])

# fit the data
input_x = X_train_A_shaped
input_y = y_train

val_data = X_test_A_shaped, y_test

selu_image.fit(input_x, input_y, validation_data=val_data, shuffle= True, epochs=250,verbose=0, batch_size= 50)

selu_image_time = (time.time() - start_time)
print("--- %s seconds ---" % (time.time() - start_time))
#image only with adam optimizer
start_time = time.time()

adam_image = keras.models.Sequential(
[
    keras.layers.Flatten(input_shape=[48,48,1]),
    keras.layers.Dense(300, activation= 'relu'),
    keras.layers.Dense(100, activation= 'relu'),
    keras.layers.Dense(50, activation= 'relu'),
    keras.layers.Dense(5,activation= 'softmax')
   
])

input_x = X_train_A_shaped
input_y = y_train

val_data = X_test_A_shaped, y_test


adam_image.compile(loss= 'sparse_categorical_crossentropy', optimizer='adam',metrics= ['SparseCategoricalAccuracy'])

adam_image.fit(input_x, input_y, validation_data=val_data, shuffle= True, epochs=250,verbose=0, batch_size= 50)

adam_image_time = (time.time() - start_time)
print("--- %s seconds ---" % (time.time() - start_time))

#image only with sgd optimizer

start_time = time.time()

sgd_image = keras.models.Sequential(
[
    keras.layers.Flatten(input_shape=[48,48,1]),
    keras.layers.Dense(300, activation= 'relu'),
    keras.layers.Dense(100, activation= 'relu'),
    keras.layers.Dense(50, activation= 'relu'),
    keras.layers.Dense(5,activation= 'softmax')
  
])

sgd_image.compile(loss= 'sparse_categorical_crossentropy', optimizer='sgd',metrics= ['SparseCategoricalAccuracy'])

input_x = X_train_A_shaped
input_y = y_train

val_data = X_test_A_shaped, y_test

sgd_image.fit(input_x, input_y, validation_data=val_data, shuffle= True, epochs=250,verbose=0, batch_size= 50)

sgd_image_time = (time.time() - start_time)
print("--- %s seconds ---" % (time.time() - start_time))
# function to capture the diagonal values of the matrix 

def true_pred(model, confusion_matrix, y_test):
    '''
    Input:- 
    confusion_matrix = is a np.ndarray
    model = string name of the model. 
    y_test = classification label series
    
    Output:- 
    diagonal values of confusion matrix (true predictions) in dataframe 
    
    '''
    test = np.matrix(confusion_matrix)
    n,m = test.shape
    
    # get label names from y_test column
    labels = sorted(np.unique(y_test))
    #print(labels)
    # list of values 
    values = []
    
    if n == len(labels):
        for i in range(m):
            values.append(test[i,i])
    else :
        print('The lengths of y_test does not match with confusion matrix shape')
    
    #print(values)
    data = { model: values}
    #print(data)
    df= pd.DataFrame(data=data, index= labels)
    
    return df

## Image with CNN / sgd optimizer
start_time = time.time()

CNN_image_sgd = keras.models.Sequential(
[
    keras.layers.Conv2D(64,7, activation= 'relu', padding= 'same', input_shape= [48,48,1]), 
    keras.layers.MaxPooling2D(2), 
    keras.layers.Conv2D(128,3, activation= 'relu', padding= 'same'),
    keras.layers.MaxPooling2D(2), 
    keras.layers.Conv2D(256,3, activation= 'relu', padding= 'same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(input_shape= [48,48,1]),
    keras.layers.Dense(128, activation= 'relu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(64, activation= 'relu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(32, activation= 'relu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(5,activation= 'softmax')
    
])

CNN_image_sgd.compile(loss= 'sparse_categorical_crossentropy', optimizer='sgd',metrics= ['SparseCategoricalAccuracy'])

input_x = X_train_A_shaped
input_y = y_train

val_data = X_test_A_shaped, y_test

CNN_image_sgd.fit(input_x, input_y, validation_data=val_data, shuffle= True, epochs=250,verbose=0, batch_size= 50)

CNN_image_time = (time.time() - start_time)
print("--- %s seconds ---" % (time.time() - start_time))
pd.DataFrame(CNN_image_sgd.history.history).plot(figsize=(8,5))
plt.grid(True)
#image with CNN/ sgd optimizer
y_pred_test_7 = np.argmax(CNN_image_sgd.predict(X_test_A_shaped), axis=1)
y_pred_train_7 = np.argmax(CNN_image_sgd.predict(X_train_A_shaped), axis=1)

#Gathering true predictions image CNN --> Test
df13 = true_pred('CNN_test_image_sgd', confusion_matrix(y_pred_test_7, y_test), y_test)

# Gathering true prediction image models --> Train
df14 = true_pred('CNN_train_image_sgd', confusion_matrix(y_pred_train_7, y_train), y_train)

print('CNN Train Classification report')
print(classification_report(y_pred_train_7, y_train))

print('CNN Test Classification report')
print(classification_report(y_pred_test_7, y_test))

# plot history for each model. Compare with data/image vs image only model performance.
df1_hist = pd.DataFrame(combined_model.history.history) #selu_data_image
df2_hist = pd.DataFrame(model.history.history) #adam_data_image
df3_hist = pd.DataFrame(model_sgd.history.history) #sgd_data_image
df4_hist = pd.DataFrame(selu_image.history.history)
df5_hist = pd.DataFrame(adam_image.history.history)
df6_hist = pd.DataFrame(sgd_image.history.history)
df7_hist = pd.DataFrame(CNN_image_sgd.history.history)


fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(25, 14), sharex= True, sharey=True, constrained_layout= False )
plt.ylim(0,1.75)
axs[0,0].plot(df1_hist.index, df1_hist[list(df1_hist.columns)])
axs[0,1].plot(df2_hist.index, df2_hist[list(df2_hist.columns)])
axs[0,2].plot(df3_hist.index, df3_hist[list(df3_hist.columns)])
axs[1,0].plot(df4_hist.index, df4_hist[list(df4_hist.columns)])
axs[1,1].plot(df5_hist.index, df5_hist[list(df5_hist.columns)])
axs[1,2].plot(df6_hist.index, df6_hist[list(df6_hist.columns)])
#axs[2,0].plot(df7_hist.index, df7_hist[list(df7_hist.columns)])


axs[0, 0].set_title("Selu_data_image")
axs[0, 1].set_title("adam_data_image")
axs[0, 2].set_title("sgd_data_image")

axs[1, 0].set_title("selu_image")
axs[1, 1].set_title("adam_image")
axs[1, 2].set_title("sgd_image")

# axs[2,0].set_title('CNN_image')
# fig.delaxes(axs[2,1]) #The indexing is zero-based here
# fig.delaxes(axs[2,2]) #The indexing is zero-based here

axs[0, 0].set_xlabel("Epochs")
axs[0, 0].set_ylabel("loss/accuracy")
axs[0, 1].set_xlabel("Epochs")
axs[0, 2].set_xlabel('Epochs')
#axs[0, 1].set_ylabel("loss/accuracy")
axs[1, 0].set_xlabel("Epochs")
axs[1, 0].set_ylabel("loss/accuracy")
axs[1, 1].set_xlabel("Epochs")
axs[1, 2].set_xlabel('Epochs')

# axs[2, 0].set_ylabel("loss/accuracy")
# axs[2, 0].set_xlabel('Epochs')

axs[0,0].grid(axis='both')
axs[0,1].grid(axis='both')
axs[0,2].grid(axis='both')
axs[1,0].grid(axis='both')
axs[1,1].grid(axis='both')
axs[1,2].grid(axis='both')
#axs[0,0].legend(df1['loss'], ["loss"], loc=1)
axs[0,0].legend(labels= ['loss', 'val_loss','SparseCategoricalAccuracy', 'val_SparseCategoricalAccuracy' ])
# data and image with SGD optimizer / activation= selu
y_pred_test_1 = np.argmax(combined_model.predict([X_test_B, X_test_A_shaped]), axis=1)
y_pred_train_1 = np.argmax(combined_model.predict([X_train_B, X_train_A_shaped]), axis=1)

# data and image with adam optimizer / activation = relu
y_pred_test_2 = np.argmax(model.predict([X_test_B, X_test_A_shaped]), axis=1)
y_pred_train_2 = np.argmax(model.predict([X_train_B, X_train_A_shaped]), axis=1)

# data and image with SGD optimizer / activation = relu
y_pred_test_3 = np.argmax(model_sgd.predict([X_test_B, X_test_A_shaped]), axis=1)
y_pred_train_3 = np.argmax(model_sgd.predict([X_train_B, X_train_A_shaped]), axis=1)


#image with SGD optimizer/ activation = selu
y_pred_test_4 = np.argmax(selu_image.predict(X_test_A_shaped), axis=1)
y_pred_train_4 = np.argmax(selu_image.predict(X_train_A_shaped), axis=1)

# image with adam optimizer
y_pred_test_5 = np.argmax(adam_image.predict(X_test_A_shaped), axis=1)
y_pred_train_5 = np.argmax(adam_image.predict(X_train_A_shaped), axis=1)

# image with sgd optimizer
y_pred_test_6 = np.argmax(sgd_image.predict(X_test_A_shaped), axis=1)
y_pred_train_6 = np.argmax(sgd_image.predict(X_train_A_shaped), axis=1)

# Getting raw input data for comparision. 
input_train = pd.DataFrame(train['ethnicity'].value_counts())
input_train.rename(columns={"ethnicity": "Input_train"},inplace= True)

input_test = pd.DataFrame(test['ethnicity'].value_counts())
input_test.rename(columns={"ethnicity": "Input_test"},inplace= True)

# Gathering true prediction data/image models -->Test. 
df1 = true_pred('sgd_test_data_image_selu', confusion_matrix(y_pred_test_1, y_test), y_test)
df2 = true_pred('adam_test_data_image_relu',confusion_matrix(y_pred_test_2, y_test), y_test)
df3 = true_pred('sgd_test_data_image_relu' ,confusion_matrix(y_pred_test_3, y_test), y_test)

# Gathering true prediction data/image models -->Train. 
df4 = true_pred('sgd_train_data_image_selu', confusion_matrix(y_pred_train_1, y_train), y_train)
df5 = true_pred('adam_train_data_image_relu',confusion_matrix(y_pred_train_2, y_train), y_train)
df6 = true_pred('sgd_train_data_image_relu' ,confusion_matrix(y_pred_train_3, y_train), y_train)


# Gathering true prediction image models --> Test
df7 = true_pred('sgd_test_image_selu', confusion_matrix(y_pred_test_4, y_test), y_test)
df8 = true_pred('adam_test_image_relu', confusion_matrix(y_pred_test_5, y_test), y_test)
df9 = true_pred('sgd_test_image_relu', confusion_matrix(y_pred_test_6, y_test), y_test)

# Gathering true prediction image models --> Train
df10 = true_pred('sgd_train_image_selu', confusion_matrix(y_pred_train_4, y_train), y_train)
df11 = true_pred('adam_train_image_relu', confusion_matrix(y_pred_train_5, y_train), y_train)
df12 = true_pred('sgd_train_image_relu', confusion_matrix(y_pred_train_6, y_train), y_train)


# test data efficiency
df_confusion_matrix= pd.concat([df1,df2,df3,df7,df8,df9,df13, input_test], axis=1, sort=True)
#print(df_confusion_matrix)

# train data efficiency
df_confusion_matrix_train= pd.concat([df4,df5,df6,df10,df11,df12,df14, input_train], axis=1, sort=True)
#print(df_confusion_matrix_train)


# Model performance vs test data input for each label. 
fig = go.Figure()
fig.add_trace(go.Bar(
    x=df_confusion_matrix.index,
    y=list(df_confusion_matrix['sgd_test_data_image_selu'].values),
    name='sgd_dat_img_selu',
    marker_color='indianred'))
fig.add_trace(go.Bar(
    x=df_confusion_matrix.index,
    y=list(df_confusion_matrix['adam_test_data_image_relu'].values),
    name='adam_dat_img_relu',
    marker_color='lightsalmon'
))
fig.add_trace(go.Bar(
    x=df_confusion_matrix.index,
    y=list(df_confusion_matrix['sgd_test_data_image_relu'].values),
    name='sgd_dat_img_relu',
    marker_color='DarkSlateGrey'
))
fig.add_trace(go.Bar(
    x=df_confusion_matrix.index,
    y=list(df_confusion_matrix['sgd_test_image_selu'].values),
    name='sgd_image_selu',
    marker_color='MediumPurple'
))
fig.add_trace(go.Bar(
    x=df_confusion_matrix.index,
    y=list(df_confusion_matrix['adam_test_image_relu'].values),
    name='adam_image_relu',
    marker_color='turquoise'
))
fig.add_trace(go.Bar(
    x=df_confusion_matrix.index,
    y=list(df_confusion_matrix['sgd_test_image_relu'].values),
    name='sgd_image_relu',
    marker_color='lightpink'
))
fig.add_trace(go.Bar(
    x=df_confusion_matrix.index,
    y=list(df_confusion_matrix['CNN_test_image_sgd'].values),
    name='CNN_image',
    marker_color='silver'
))
fig.add_trace(go.Bar(
    x=df_confusion_matrix.index,
    y=list(df_confusion_matrix['Input_test'].values),
    name='input_test',
    marker_color='LightSkyBlue'
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(barmode='group', xaxis_tickangle=-45, title="Model performance vs input test data, label wise")
fig.update_yaxes(title= "True Predictions")
fig.update_xaxes(title= 'labels')
fig.show()

# Model performance vs train data input for each label. 
fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=df_confusion_matrix.index,
    y=list(df_confusion_matrix_train['sgd_train_data_image_selu'].values),
    name='sgd_dat_img_selu',
    marker_color='indianred'))
fig1.add_trace(go.Bar(
    x=df_confusion_matrix.index,
    y=list(df_confusion_matrix_train['adam_train_data_image_relu'].values),
    name='adam_dat_img_relu',
    marker_color='lightsalmon'
))
fig1.add_trace(go.Bar(
    x=df_confusion_matrix.index,
    y=list(df_confusion_matrix_train['sgd_train_data_image_relu'].values),
    name='sgd_dat_img_relu',
    marker_color='DarkSlateGrey'
))
fig1.add_trace(go.Bar(
    x=df_confusion_matrix.index,
    y=list(df_confusion_matrix_train['sgd_train_image_selu'].values),
    name='sgd_image_selu',
    marker_color='MediumPurple'
))
fig1.add_trace(go.Bar(
    x=df_confusion_matrix.index,
    y=list(df_confusion_matrix_train['adam_train_image_relu'].values),
    name='adam_image_relu',
    marker_color='turquoise'
))
fig1.add_trace(go.Bar(
    x=df_confusion_matrix.index,
    y=list(df_confusion_matrix_train['sgd_train_image_relu'].values),
    name='sgd_image_relu',
    marker_color='lightpink'
))
fig1.add_trace(go.Bar(
    x=df_confusion_matrix.index,
    y=list(df_confusion_matrix_train['CNN_train_image_sgd'].values),
    name='CNN_image',
    marker_color='silver'
))
fig1.add_trace(go.Bar(
    x=df_confusion_matrix.index,
    y=list(df_confusion_matrix_train['Input_train'].values),
    name='input_test',
    marker_color='LightSkyBlue'
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig1.update_layout(barmode='group', xaxis_tickangle=-45, title="Model performance vs input train data, label wise")
fig1.update_yaxes(title= "True Predictions")
fig1.update_xaxes(title= 'labels')
fig1.show()
