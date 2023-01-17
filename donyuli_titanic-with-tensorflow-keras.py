# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

print(tf.__version__)

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
raw_train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

raw_test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

example = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

print(example.head())
print('Columns:====>\n',list(raw_train_data.columns.unique()))

print()

print('NaN Data distributes as below:')

print(raw_train_data.isnull().sum())
target = raw_train_data.Survived

raw_train_data.describe()
raw_train_data.std()
if 0:

    plt.figure(figsize=(20,20))

    plt.subplot(331)

    sns.barplot(x=raw_train_data.Sex,y=target)

    plt.subplot(332)

    sns.barplot(x=raw_train_data.Pclass,y=target)

    plt.subplot(333)

    sns.lineplot(x=raw_train_data.Age,y=target)

    plt.subplot(334)

    sns.lineplot(x=raw_train_data.SibSp,y=target)

    plt.subplot(334)

    sns.lineplot(x=raw_train_data.SibSp,y=target)

    plt.subplot(335)

    sns.lineplot(x=raw_train_data.Parch,y=target)

    plt.subplot(336)

    sns.lineplot(x=raw_train_data.Cabin,y=target)

    plt.subplot(337)

    sns.barplot(x=raw_train_data.Embarked,y=target)

    plt.subplot(338)

    sns.lineplot(x=raw_train_data.Fare,y=target)
#Choose desired features based on observations of plots above

features = ['Sex','Pclass','Age','SibSp','Parch','Embarked','Fare']



X = raw_train_data[features]

X_test = raw_test_data[features]

X.head()

X.Embarked.unique()
print(X.dtypes == 'object')

# Encode categorical values

X.Sex[X.loc[X.Sex == 'female'].index] = 0

X.Sex[X.loc[X.Sex == 'male'].index] = 1

X.Embarked[X.loc[X.Embarked == 'S'].index] = 0

X.Embarked[X.loc[X.Embarked == 'C'].index] = 1

X.Embarked[X.loc[X.Embarked == 'Q'].index] = 2



X_test.Sex[X_test.loc[X_test.Sex == 'female'].index] = 0

X_test.Sex[X_test.loc[X_test.Sex == 'male'].index] = 1

X_test.Embarked[X_test.loc[X_test.Embarked == 'S'].index] = 0

X_test.Embarked[X_test.loc[X_test.Embarked == 'C'].index] = 1

X_test.Embarked[X_test.loc[X_test.Embarked == 'Q'].index] = 2



# For now let's just drop Embarked list for prototyping

#X.drop('Embarked',axis=1,inplace = True)

#X_test.drop('Embarked',axis=1,inplace = True)
print(X_test.columns)
# Process NaN data

my_imputer = SimpleImputer()

imputed_X = pd.DataFrame(my_imputer.fit_transform(X))

imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))



# 注意数据填充操作移除了DataFrame的列标签，因此需要把它们取回来

imputed_X.columns = X.columns

imputed_X_test.columns = X_test.columns
# Normalize input. Note that the same process must be used on val set/test set.

miu = imputed_X.values.mean()

sigma = imputed_X.values.std()

imputed_X = imputed_X.applymap(lambda x:(x-miu)/sigma)

imputed_X_test = imputed_X_test.applymap(lambda x:(x-miu)/sigma)

imputed_X_test.head()
processed_X_train,processed_X_val,processed_y_train,processed_y_val = train_test_split(imputed_X,target,test_size = 0.3)
processed_X_train.head()
# Convert pandas DataFrame to lists in order to utilize tensorflow 

m = len(processed_X_train.Sex)

processed_X_train_list = np.array(processed_X_train.values.tolist())#training data

processed_X_val_list = np.array(processed_X_val.values.tolist())#validation set



print(processed_X_train_list.shape)

print(imputed_X_test.shape)
model = tf.keras.Sequential([

    tf.keras.layers.Dense(10,input_dim=len(imputed_X_test.columns),use_bias=True,bias_initializer='zeros',activation=tf.nn.relu),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(10,activation=tf.nn.relu),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)

])
# Summary of hyperparameters

pretrain_BATCH_SIZE = 4

pretrain_EPOCHS = 100

BATCH_SIZE = 64

EPOCHS = 700

ALPHA = 0.003
model.compile(optimizer=tf.keras.optimizers.Adam(ALPHA),

             loss = 'binary_crossentropy',

             metrics=['accuracy'])
history_of_training = model.fit(processed_X_train_list,np.array(processed_y_train), 

                                    validation_data = [processed_X_val_list,processed_y_val],

                                    batch_size=pretrain_BATCH_SIZE,epochs=pretrain_EPOCHS,verbose=4)



history_of_training = model.fit(processed_X_train_list,np.array(processed_y_train), 

                                    validation_data = [processed_X_val_list,processed_y_val],

                                    batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=4)

# plot the statues of convergence

acc = history_of_training.history['accuracy']

val_acc = history_of_training.history['val_accuracy']



loss = history_of_training.history['loss']

val_loss = history_of_training.history['val_loss']



epochs_range = range(EPOCHS)



plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
#print(processed_X_val_list)

val_prediction = model.predict(processed_X_val_list)

val_prediction = (val_prediction>0.5).astype(np.int)



processed_y_val = np.array(processed_y_val.values).reshape(-1,1)



score = 0

for item in range(len(val_prediction)):

    if(val_prediction[item] == processed_y_val[item]):

        score = score + 1

score = score/len(val_prediction)

print(score)
print(imputed_X_test)
print(imputed_X_test)
imputed_X_test = np.array(imputed_X_test.values.tolist())

print(imputed_X_test)
# Make predictions on testset

prediction = model.predict(imputed_X_test)

prediction = (prediction>0.5).astype(np.int)

prediction = np.squeeze(prediction)

prediction = pd.Series(prediction,name='Survived')

print('There\'re',prediction.sum(),'people survived in the test set.')

Index = raw_test_data['PassengerId']

# Analyze the output

#survived = prediction.loc[prediction > 0.5]

#raw_test_data.loc[survived.index]
result = pd.DataFrame({'PassengerId':Index,'Survived':prediction})

result.to_csv('prediction_submission.csv',index=False)

prediction.head()
reload = pd.read_csv('prediction_submission.csv')

reload.head()
reload.describe()