# start with loading lib

#load packages

import numpy as np 

import pandas as pd 



# keras for NN

from keras.layers import Dense

from keras.models import Sequential



#Common Model Helpers

from sklearn import model_selection

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix,precision_score



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



#Configure Visualization Defaults

%matplotlib inline 

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
#import data

# train is used to optimize a model, as the column survived is present (so it will be split at some point in train and test)

# validation does not have survived and is used for kaggle competition



train_data = pd.read_csv('../input/train.csv')

validation_data  = pd.read_csv('../input/test.csv')



# can clean both datasets at once

data_cleaner = [train_data, validation_data]



train_data.sample(5)
train_data.describe(include = 'all')
print('Train columns with null values:\n', train_data.isnull().sum())

print("-"*10)



print('Test/Validation columns with null values:\n', validation_data.isnull().sum())

print("-"*10)
# complete or delete missing values in train and test/validation dataset

for dataset in data_cleaner:    

    #complete missing age with median

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)



    #complete embarked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



    #complete missing fare with median

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    

    #delete column Cabin and Ticket in dataset

    drop_column = ['Cabin', 'Ticket']

    dataset.drop(drop_column, axis=1, inplace = True)

    

#delete column PassengerId in train_data

drop_column = ['PassengerId']

train_data.drop(drop_column, axis=1, inplace = True)



print(train_data.isnull().sum())

print("-"*10)

print(validation_data.isnull().sum())
###CREATE: Feature Engineering for train and test/validation dataset

for dataset in data_cleaner:    

    #Discrete variables

    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1



    dataset['IsAlone'] = 1 #initialize to yes/1 is alone

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1



    #quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]





    #Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut

    #Fare Bins/Buckets 

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)



    #Age Bins/Buckets 

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)



    #cleanup rare title names

    stat_min = 10 # common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/

    title_names = (dataset['Title'].value_counts() < stat_min) #this will create a true false series with title name as index



    #apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/

    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)



train_data.sample(3)
#CONVERT: convert objects to category using Label Encoder for train and test/validation dataset



#code categorical data

label = LabelEncoder()

for dataset in data_cleaner:    

    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])

    dataset['Title_Code'] = label.fit_transform(dataset['Title'])

    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])

    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])



    

# Now we can drop some column 

#delete column Cabin and Ticket in test dataset

for dataset in data_cleaner:

    drop_column = ['Name', 'Sex','Embarked','Title','FareBin','AgeBin']

    dataset.drop(drop_column, axis=1, inplace = True)

train_data.sample(3)
validation_data.sample(3)
#split train to train and test data with function defaults (as validation is only for kaggle compet.. )

seed = 4



xt = train_data.drop('Survived', axis=1)

yt = train_data['Survived']

x_train, x_test, y_train, y_test = model_selection.train_test_split(xt, yt, random_state = seed)

print(x_train.shape)

x_train.sample(3)
#NN using keras

class NN_keras():

    def __init__(self,nbneuron, indim):

        # create model

        self.model = Sequential()

        self.model.add(Dense(nbneuron, input_dim=indim, kernel_initializer='normal', activation='relu'))

        self.model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

        # Compile model

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    def __call__(self):

        return self.model
model = NN_keras(nbneuron=24,indim=12).__call__()

model.summary()



model_output = model.fit(x_train,y_train,epochs = 200,batch_size = 10,validation_data=(x_test,y_test), verbose=0)
# show model accuracy

plt.subplot(211)

acc_data = pd.DataFrame({'train_acc': model_output.history["acc"], 'test_acc': model_output.history["val_acc"]})

sns.lineplot(data = acc_data, palette="tab10", linewidth=2.5)



# show model loss

plt.subplot(212)

loss_data = pd.DataFrame({'train_loss': model_output.history["loss"], 'test_loss': model_output.history["val_loss"]})

sns.lineplot(data = loss_data, palette="tab10", linewidth=2.5)



print('Training Accuracy : ', np.mean(model_output.history["acc"]))

print('Validation Accuracy : ', np.mean(model_output.history["val_acc"]))
y_pred = model.predict(x_test)

rounded = [round(x[0]) for x in y_pred]

y_pred1 = np.array(rounded,dtype='int64')

confusion_matrix(y_test,y_pred1)
precision_score(y_test,y_pred1)
x_val = validation_data.drop("PassengerId", axis=1)

y_val = model.predict(x_val)
#submission = pd.DataFrame({

#        "PassengerId": validation_data["PassengerId"],

#        "Survived": y_val.round().astype(int).flatten()})



#submission.to_csv('../output/titanic_keras.csv', index=False)

#submission.sample(10)