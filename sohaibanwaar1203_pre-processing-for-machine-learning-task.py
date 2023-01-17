# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import numpy as np

from sklearn.model_selection import train_test_split

import keras

from keras.callbacks import LambdaCallback

from keras.layers import Conv1D, Flatten

from keras.layers import Dense ,Dropout,BatchNormalization

from keras.models import Sequential 

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical 

from keras import regularizers

from sklearn import preprocessing

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn import metrics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/csv_result-messidor_features.csv')

df=df.drop('id',axis=1)

df.head(11)
df.describe()
from matplotlib import pyplot as plt

df_x=df.drop('Class',axis=1)

img=np.array(df[6:7])

label=df['Class'][6:7]

plt.imshow(img, interpolation='nearest')

plt.show()

if label.values ==0:

    print("This eye has no dr")

else:

     print("This eye has dr")





img=np.array(df.iloc[7:8])

label=df['Class'].iloc[7:8]

plt.imshow(img, interpolation='nearest')

plt.show()

if label.values ==0:

    print("This eye has no dr")

else:

     print("This eye has dr")



import seaborn as sns # for Visualizing my data

df_y=df['Class']

df_y.head()

sns.distplot(df_y, kde=False); # Visualizing levels in dataset


df_noDr=df[df['Class']==0]

print("Unique Value in No DR",df_noDr['Class'].unique()) # For confirming



df_Dr=df[df['Class']==1]  

print("Unique Value in DR",df_Dr['Class'].unique())    # For confirming


i=1

while(i<18):

    plt.scatter( df['Class'],df[str(i)], alpha=0.5)

    plt.title('Scatter plot of Coloum and Class Label')

    plt.xlabel('Class')

    plt.ylabel('Coloum '+ str(i))

    plt.show()

    i=i+1

    

import pandas as pd

import matplotlib.pyplot as plt

corr = df_x.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(df_x.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(df_x.columns)

ax.set_yticklabels(df_x.columns)

plt.show()


corr = df_x.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.9:

            if columns[j]:

                columns[j] = False

selected_columns = df_x.columns[columns]

data = df_x[selected_columns]

data.describe()


corr = data.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
selected_columns = selected_columns[1:].values

import statsmodels.formula.api as sm

def backwardElimination(x, Y, sl, columns):

    numVars = len(x[0])

    for i in range(0, numVars):

        regressor_OLS = sm.OLS(Y, x).fit()

        maxVar = max(regressor_OLS.pvalues).astype(float)

        if maxVar > sl:

            for j in range(0, numVars - i):

                if (regressor_OLS.pvalues[j].astype(float) == maxVar):

                    x = np.delete(x, j, 1)

                    columns = np.delete(columns, j)

                    

    regressor_OLS.summary()

    return x, columns

SL = 0.05

data_modeled, selected_columns = backwardElimination(data.iloc[:,1:].values, data.iloc[:,0].values, SL, selected_columns)
selected_columns
result = pd.DataFrame()

result['diagnosis'] = df_y
data = pd.DataFrame(data = data_modeled, columns = selected_columns)
fig = plt.figure(figsize = (20, 25))

j = 0

for i in data.columns:

    plt.subplot(6, 4, j+1)

    j += 1

    sns.distplot(data[i][result['diagnosis']==0], color='g', label = 'NO DR')

    sns.distplot(data[i][result['diagnosis']==1], color='r', label = 'DR')

    plt.legend(loc='best')

fig.suptitle('Diabetic Rateinopathy ')

fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
for column in data:

    plt.figure()

    sns.boxplot(x=data[column])
for column_1st in data:

    for coloum_2nd in data:

        jet=plt.get_cmap('jet')

        plt.figure(figsize=(15,5))

        plt.scatter(data[column_1st], data[coloum_2nd], s=30, c=df_y, vmin=0, vmax=1, cmap=jet)

        plt.xlabel(column_1st,fontsize=40)

        plt.ylabel(coloum_2nd,fontsize=40)

        plt.colorbar()

        plt.show()

        
print(data.shape)

z_Scored_df=pd.DataFrame(data)

from scipy import stats

z_Scored_df=z_Scored_df[(np.abs(stats.zscore(data)) < 3).all(axis=1)]

z_Scored_df.shape
#merging Labels according to the Index selected from Outlier Detecteion (e.g 1, 4... are outliers so removed)

z_Scored_df=z_Scored_df.merge(df_y.to_frame(), left_index=True, right_index=True)

z_Scored_df.shape
z_Scored_df_labels=z_Scored_df['Class']

z_Scored_df=z_Scored_df.drop('Class',axis=1)

#labels and X features are seperated

print (z_Scored_df[1:6])

print(z_Scored_df_labels.head())
X_train, X_test, y_train, y_test = train_test_split(

     z_Scored_df,z_Scored_df_labels, test_size=0.1, random_state=0)





classifiers = [

    KNeighborsClassifier(2),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    GaussianProcessClassifier(1.0 * RBF(1.0)),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



size=classifiers.count



for i in classifiers:

    clf=i

    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)

    print("Accuracy of : ",i,"\n",metrics.accuracy_score(y_test, y_pred))

    print("\n\n")
#i am copying the same model I dont know why I have to compile this cell when ever i want to train this model

#for some other X. 

def model_Z(input1)    :

    model = Sequential () # we make a sequentail model

    

    model.add(Dense(128, input_dim=input1,activation  ='relu',activity_regularizer = regularizers.l2(1e-4)))

    model.add(Dropout(0.5))

    model.add(Dense(64, input_dim=input1,activation  ='relu',activity_regularizer = regularizers.l2(1e-6)))

    model.add(Dropout(0.5))

    model.add(Dense(32, activation  ='relu',activity_regularizer = regularizers.l2(1e-8)))

    model.add(Dropout(0.25))

    model.add(Dense(2, activation  ='softmax')) #softmax layer to compute the probability of

                                                #labels#softmax layer to compute the probability of

                                                #labels

    

    model.summary()

    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['accuracy'])

    return model

categorical_y_test=to_categorical(y_test)

categorical_y_train=to_categorical(y_train)



X_train = np.array(X_train)

X_test = np.array(X_test)



X_train=X_train.reshape(X_train.shape[0],X_train.shape[1])

X_test=X_test.reshape(X_test.shape[0],X_train.shape[1])



print(X_train.shape)

print(y_test.shape)
import matplotlib.pyplot as plt

model=model_Z(6)

history =model.fit(X_train, categorical_y_train, epochs=500, batch_size=101,validation_data=(X_test,categorical_y_test)

                  )



# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])



plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
Q1 = np.quantile(data,0.25)

Q3 = np.quantile(data,0.75)

IQR = Q3 - Q1

print(IQR)
data_o=pd.DataFrame()

data_o = data[((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]

print("DataFrame with outliers: ",data.shape )

print("DataFrame with_out outliers: ",data_o.shape )
#merging Labels according to the Index selected from Outlier Detecteion (e.g 1, 4... are outliers so removed)

data_o=data_o.merge(df_y.to_frame(), left_index=True, right_index=True)

data_o.shape
data_o_labels=data_o['Class']

data_o_x=data_o.drop('Class',axis=1)

#labels and X features are seperated

print (data_o_labels[1:6])

print(data_o_x.head())
X_train, X_test, y_train, y_test = train_test_split(

     data_o_x,data_o_labels, test_size=0.1, random_state=0)





classifiers = [

    KNeighborsClassifier(2),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    GaussianProcessClassifier(1.0 * RBF(1.0)),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



size=classifiers.count



for i in classifiers:

    clf=i

    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)

    print("Accuracy of : ",i,"\n",metrics.accuracy_score(y_test, y_pred))

    print("\n\n")
#i am copying the same model I dont know why I have to compile this cell when ever i want to train this model

#for some other X. 

def model_IQR(input1)    :

    model = Sequential () # we make a sequentail model

    

    model.add(Dense(128, input_dim=input1,activation  ='relu',activity_regularizer = regularizers.l2(1e-4)))

    model.add(Dropout(0.5))

    model.add(Dense(64, input_dim=input1,activation  ='relu',activity_regularizer = regularizers.l2(1e-6)))

    model.add(Dropout(0.25))

    model.add(Dense(32, activation  ='relu',activity_regularizer = regularizers.l2(1e-8)))

    model.add(Dropout(0.25))

    model.add(Dense(2, activation  ='softmax')) #softmax layer to compute the probability of

                                                #labels#softmax layer to compute the probability of

                                                #labels

    

    model.summary()

    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['accuracy'])

    return model

categorical_y_test=to_categorical(y_test)

categorical_y_train=to_categorical(y_train)



X_train = np.array(X_train)

X_test = np.array(X_test)



X_train=X_train.reshape(X_train.shape[0],X_train.shape[1])

X_test=X_test.reshape(X_test.shape[0],X_train.shape[1])



print(X_train.shape)

print(y_test.shape)
import matplotlib.pyplot as plt

model=model_IQR(6)

history =model.fit(X_train, categorical_y_train, epochs=500, batch_size=91,validation_data=(X_test,categorical_y_test)

                  )



# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])



plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
from sklearn import preprocessing

standardized_X = preprocessing.scale(z_Scored_df)

standardized_X[1]




X_train, X_test, y_train, y_test = train_test_split(

   standardized_X,z_Scored_df_labels, test_size=0.1, random_state=0)





classifiers = [

    KNeighborsClassifier(2),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    GaussianProcessClassifier(1.0 * RBF(1.0)),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



size=classifiers.count



for i in classifiers:

    clf=i

    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)

    print("Accuracy of : ",i,"\n",metrics.accuracy_score(y_test, y_pred))

    print("\n\n")
#i am copying the same model I dont know why I have to compile this cell when ever i want to train this model

#for some other X. 

def Z_Standardized_model(input1)    :

    model = Sequential () # we make a sequentail model

    model.add(Dense(256,input_dim=input1,activation  ='relu',))

    model.add(Dense(128,activation  ='relu',))

    model.add(Dense(64,activation  ='relu',))

    

    model.add(Dense(32, activation  ='relu',))

    model.add(Dropout(0.25))

    model.add(Dense(16, activation  ='relu',))

    

    model.add(Dense(8, activation  ='relu'))

    

    model.add(Dense(2, activation  ='softmax')) #softmax layer to compute the probability of

                                                #labels

    

    model.summary()

    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['accuracy'])

    return model

categorical_y_test=to_categorical(y_test)

categorical_y_train=to_categorical(y_train)



X_train = np.array(X_train)

X_test = np.array(X_test)



X_train=X_train.reshape(X_train.shape[0],X_train.shape[1])

X_test=X_test.reshape(X_test.shape[0],X_train.shape[1])



print(X_train.shape)

print(y_test.shape)
import matplotlib.pyplot as plt

model=Z_Standardized_model(6)

history =model.fit(X_train, categorical_y_train, epochs=500, batch_size=101,validation_data=(X_test,categorical_y_test)

                  )



# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])



plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
from sklearn import preprocessing

standardized_X = preprocessing.scale(data_o_x)

standardized_X[1]




X_train, X_test, y_train, y_test = train_test_split(

   data_o_x,data_o_labels, test_size=0.1, random_state=0)





classifiers = [

    KNeighborsClassifier(2),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    GaussianProcessClassifier(1.0 * RBF(1.0)),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



size=classifiers.count



for i in classifiers:

    clf=i

    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)

    print("Accuracy of : ",i,"\n",metrics.accuracy_score(y_test, y_pred))

    print("\n\n")
#i am copying the same model I dont know why I have to compile this cell when ever i want to train this model

#for some other X. 

def Standardized_model(input1)    :

    model = Sequential () # we make a sequentail model

    model.add(Dense(256,input_dim=input1,activation  ='relu',))

    model.add(Dense(128,activation  ='relu',))

    model.add(Dense(64,activation  ='relu',))

    

    model.add(Dense(32, activation  ='relu',))

    model.add(Dropout(0.25))

    model.add(Dense(16, activation  ='relu',))

    

    model.add(Dense(8, activation  ='relu'))

    

    model.add(Dense(2, activation  ='softmax')) #softmax layer to compute the probability of

                                                #labels

    

    model.summary()

    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['accuracy'])

    return model

categorical_y_test=to_categorical(y_test)

categorical_y_train=to_categorical(y_train)



X_train = np.array(X_train)

X_test = np.array(X_test)



X_train=X_train.reshape(X_train.shape[0],X_train.shape[1])

X_test=X_test.reshape(X_test.shape[0],X_train.shape[1])



print(X_train.shape)

print(y_test.shape)
import matplotlib.pyplot as plt

model=Standardized_model(6)

history =model.fit(X_train, categorical_y_train, epochs=500, batch_size=91,validation_data=(X_test,categorical_y_test)

                  )



# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])



plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
normalized_X = preprocessing.normalize(data_o_x)

normalized_X[1]
X_train, X_test, y_train, y_test = train_test_split(

     data_o_x,data_o_labels, test_size=0.1, random_state=0)





classifiers = [

    KNeighborsClassifier(2),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    GaussianProcessClassifier(1.0 * RBF(1.0)),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



size=classifiers.count



for i in classifiers:

    clf=i

    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)

    print("Accuracy of : ",i,"\n",metrics.accuracy_score(y_test, y_pred))

    print("\n\n")
#i am copying the same model I dont know why I have to compile this cell when ever i want to train this model

#for some other X. 

def Normalized_model(input1)    :

    model = Sequential () # we make a sequentail model

    model.add(Dense(256,input_dim=input1,activation  ='relu',))

    model.add(Dense(128,activation  ='relu',))

    model.add(Dense(64,activation  ='relu',))

    

    model.add(Dense(32, activation  ='relu',))

    model.add(Dropout(0.25))

    model.add(Dense(16, activation  ='relu',))

    

    model.add(Dense(8, activation  ='relu'))

    

    model.add(Dense(2, activation  ='softmax')) #softmax layer to compute the probability of

                                                #labels

    

    model.summary()

    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['accuracy'])

    return model

categorical_y_test=to_categorical(y_test)

categorical_y_train=to_categorical(y_train)



X_train = np.array(X_train)

X_test = np.array(X_test)



X_train=X_train.reshape(X_train.shape[0],X_train.shape[1])

X_test=X_test.reshape(X_test.shape[0],X_train.shape[1])



print(X_train.shape)

print(y_test.shape)
import matplotlib.pyplot as plt

model=Normalized_model(6)

history =model.fit(X_train, categorical_y_train, epochs=500, batch_size=728,validation_data=(X_test,categorical_y_test)

                  )



# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])



plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
X_train, X_test, y_train, y_test = train_test_split(

     data_o_x,data_o_labels, test_size=0.1)

def Seperate_xtest_x_Train_model(input1)    :

    model = Sequential () # we make a sequentail model

    

    model.add(Dense(128, input_dim=input1,activation  ='relu',activity_regularizer = regularizers.l2(1e-4)))

    model.add(Dropout(0.5))

    model.add(Dense(64, input_dim=input1,activation  ='relu',activity_regularizer = regularizers.l2(1e-6)))

    model.add(Dropout(0.25))

    model.add(Dense(32, activation  ='relu',activity_regularizer = regularizers.l2(1e-8)))

    model.add(Dropout(0.5))

    model.add(Dense(2, activation  ='softmax')) #softmax layer to compute the probability of

                                                #labels

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['accuracy'])

    return model




X_train = preprocessing.scale(X_train)

X_test = preprocessing.scale(X_test)



y_test=to_categorical(y_test)

y_train=to_categorical(y_train)



X_train = np.array(X_train)

X_test = np.array(X_test)



X_train=X_train.reshape(X_train.shape[0],6)

X_test=X_test.reshape(X_test.shape[0],6)



print(X_train.shape)

print(y_test.shape)
import matplotlib.pyplot as plt

Seperate_Normalized_model=Seperate_xtest_x_Train_model(6)

history =Seperate_Normalized_model.fit(X_train, y_train, epochs=500, batch_size=91,validation_data=(X_test,y_test))



# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])



plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
X_train, X_test, y_train, y_test = train_test_split(

     data_o_x,data_o_labels, test_size=0.1)





X_train = preprocessing.normalize(X_train)

X_test = preprocessing.normalize(X_test)



y_test=to_categorical(y_test)

y_train=to_categorical(y_train)



X_train = np.array(X_train)

X_test = np.array(X_test)



X_train=X_train.reshape(X_train.shape[0],6)

X_test=X_test.reshape(X_test.shape[0],6)



print(X_train.shape)

print(y_test.shape)
def Normalize_Seperate_xtest_x_Train_model(input1)    :

    model = Sequential () # we make a sequentail model

    

    model.add(Dense(128, input_dim=input1,activation  ='relu',activity_regularizer = regularizers.l2(1e-4)))

    model.add(Dropout(0.5))

    model.add(Dense(64, input_dim=input1,activation  ='relu',activity_regularizer = regularizers.l2(1e-6)))

    model.add(Dropout(0.25))

    model.add(Dense(32, activation  ='relu',activity_regularizer = regularizers.l2(1e-8)))

    model.add(Dropout(0.5))

    model.add(Dense(2, activation  ='softmax')) #softmax layer to compute the probability of

                                                #labels

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['accuracy'])

    return model
import matplotlib.pyplot as plt

Seperate_Normalized_model=Normalize_Seperate_xtest_x_Train_model(6)

history =Seperate_Normalized_model.fit(X_train, y_train, epochs=500, batch_size=91,validation_data=(X_test,y_test))



# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])



plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
X=data_o_x.drop('1',axis=1)

X.head(5)
def Experimental_Model(input1)    :

    model = Sequential () # we make a sequentail model

    

    

    model.add(Dense(128 , input_dim=input1,activation  ='relu',activity_regularizer = regularizers.l2(1e-4)))

    model.add(Dropout(0.5))

    model.add(BatchNormalization())

    model.add(Dense(64,activation  ='relu',activity_regularizer = regularizers.l2(1e-6)))

    model.add(Dropout(0.25))

    model.add(BatchNormalization())

    model.add(Dense(32, activation  ='relu',activity_regularizer = regularizers.l2(1e-8)))

    model.add(Dropout(0.5))

    model.add(BatchNormalization())

    model.add(Dense(2, activation  ='softmax')) #softmax layer to compute the probability of

                                                #labels

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['accuracy'])

    return model
X_train, X_test, y_train, y_test = train_test_split(

    X,data_o_labels, test_size=0.1)





X_train = preprocessing.normalize(X_train)

X_test = preprocessing.normalize(X_test)



y_test=to_categorical(y_test)

y_train=to_categorical(y_train)



X_train = np.array(X_train)

X_test = np.array(X_test)



X_train=X_train.reshape(X_train.shape[0],5)

X_test=X_test.reshape(X_test.shape[0],5)



print(X_train.shape)

print(y_test.shape)
import matplotlib.pyplot as plt

model=Experimental_Model(5)

history =model.fit(X_train, y_train, epochs=2000, batch_size=364,validation_data=(X_test,y_test))



# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])



plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()