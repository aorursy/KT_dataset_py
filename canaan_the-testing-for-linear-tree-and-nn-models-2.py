#import pkg_resources

#for dist in pkg_resources.working_set:
#    print(dist.project_name, dist.version)
import os

# for a local xgboost, the path is required to be added.
#mingw_path = 'W:\\langs\MinGW64\\mingw64\\bin;'
#os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import xgboost as xgb

# linear algebra
import numpy as np 

# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.cross_validation import *
from sklearn.linear_model import *
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

import matplotlib as mpl
from matplotlib import pyplot as plt

import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense,Dropout

import warnings
warnings.filterwarnings('ignore')
#print(os.listdir("../input"))

csv_train = pd.read_csv('../input/train.csv')
csv_final_test = pd.read_csv('../input/test.csv')
csv_gender_submit = pd.read_csv('../input/gender_submission.csv')

#The score of gender_submission.csv is 0.76555
csv_train.info()
csv_train_null_values = csv_train.isnull().sum()
csv_train_null_values
csv_train_null_values.plot.bar()
csv_final_test.info()
csv_final_test_null_values = csv_final_test.isnull().sum()
csv_final_test_null_values
csv_final_test_null_values.plot.bar()
csv_train.describe()
csv_train['Sex'].describe()
csv_train['Age'].describe()
sns.countplot (csv_train['Sex'], hue=csv_train['Survived'])
display(csv_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().round(3))
csv_train['Sex']
csv_train['Sex'] = csv_train['Sex'].map({'female':0, 'male':1}).astype(int)
csv_train['Sex']

csv_train.describe()
#The percentage of the male is 64%, the mean Survived 38%.
csv_final_test['Sex'] = csv_final_test['Sex'].map({'female':0, 'male':1}).astype(int)
csv_final_test['Sex']
age_mean = csv_train['Age'].mean()
print ("age_mean={}".format(age_mean))
csv_train['Age'] = csv_train['Age'].fillna(age_mean)
csv_train.describe()
csv_final_test['Age'] = csv_final_test['Age'].fillna(age_mean)
csv_final_test.describe()
csv_train_OneHot = pd.get_dummies(data=csv_train,columns=['Embarked'])
csv_train_OneHot[:5]
csv_final_test_OneHot = pd.get_dummies(data=csv_final_test,columns=['Embarked'])
csv_final_test_OneHot[:5]
csv_train_OneHot = csv_train_OneHot.fillna(0)
csv_train_OneHot[:5]
csv_final_test_OneHot = csv_final_test_OneHot.fillna(0)
csv_final_test_OneHot[:5]
plt.figure(figsize = (10, 10))
sns.heatmap(csv_train_OneHot.corr(),  vmax=0.9, square=True);
csv_train_corr_Survived = csv_train_OneHot.corr()['Survived']
csv_train_corr_Survived
train_data_X = csv_train_OneHot[['Sex','Pclass','Fare','Age','SibSp','Parch','Embarked_C','Embarked_Q','Embarked_S']]
train_data_X[:5] 
train_data_Y = csv_train_OneHot[['Survived']]
train_data_Y[:5] 
train_data,validation_data,train_labels,validation_labels=train_test_split(train_data_X,train_data_Y,random_state=7,train_size=0.8)
train_data.describe()
train_data[:5]

train_labels[:5]

validation_data.describe()
validation_data[:5]

validation_labels[:5]
test_data = csv_final_test_OneHot[['Sex','Pclass','Fare','Age','SibSp','Parch','Embarked_C','Embarked_Q','Embarked_S']]
test_data[:5]

LR = LinearRegression()
predictions = []

#train_predictors = (titanic[predictors].iloc[train, :])  # the features for training (x1, x2...xn)
#train_target = titanic["Survived"].iloc[train]  # the predictive target (y)

LR.fit(train_data, train_labels)  # finding the best fit for the target 
validation_predictions = LR.predict(validation_data)  # predict based on the best fit produced by alg.fit
predictions.append(validation_predictions)
predictions = np.concatenate(predictions)
predictions[predictions >= 0.5] = 1
predictions[predictions < 0.5] = 0
#print (predictions)

accuracy = np.count_nonzero(validation_labels == predictions)/validation_labels.count()
print ('accuracy: {}'.format(accuracy['Survived']))

predictions = []
#print (predictions)
test_predictions = LR.predict(test_data)  # predict based on the best fit produced by alg.fit
predictions.append(test_predictions)
#predictions = test_predictions
predictions = np.concatenate(predictions)
predictions[predictions >= 0.5] = 1
predictions[predictions < 0.5] = 0
#csv_gender_submit["PassengerId"]
#print (predictions)
#predictions_list = predictions.tolist()
#print (predictions_list)
submission = pd.DataFrame()
#submission = pd.DataFrame({"PassengerId": csv_gender_submit["PassengerId"],"Survived": predictions})
submission['PassengerId'] = csv_gender_submit["PassengerId"]
submission['Survived'] = predictions.astype(int)
#submission.info()
#submission

submission.to_csv("lr_submission.csv", index=False)

# the score of "lr_submission.csv" is 0.76076
DT = tree.DecisionTreeClassifier()
predictions = []

DT.fit(train_data, train_labels) 
validation_predictions = DT.predict(validation_data)  # predict based on the best fit produced by alg.fit
predictions.append(validation_predictions)
#predictions = np.concatenate(predictions)
#predictions[predictions > 0.5] = 1
#predictions[predictions < 0.5] = 0
#print (predictions)
#validation_labels
accuracy = np.count_nonzero(validation_labels == predictions)/validation_labels.count()
print ('accuracy: {}'.format(accuracy['Survived']))

print('DT.score: {}'.format(DT.score(train_data, train_labels)))
predictions = []
#print (predictions)
test_predictions = DT.predict(test_data)  # predict based on the best fit produced by alg.fit
#predictions.append(test_predictions)
predictions = test_predictions
#predictions = np.concatenate(predictions)
#predictions[predictions > 0.5] = 1
#predictions[predictions < 0.5] = 0

#print (predictions)
#predictions_list = predictions.tolist()
#print (predictions_list)
submission = pd.DataFrame()
#submission = pd.DataFrame({"PassengerId": csv_gender_submit["PassengerId"],"Survived": predictions})
submission['PassengerId'] = csv_gender_submit["PassengerId"]
submission['Survived'] = predictions
#submission.info()
#print (submission)

submission.to_csv("dt_submission.csv", index=False)

# the score of "dt_submission.csv" is 0.72248
train_data[:5]
ndarray = train_data.values
ndarray.shape
train_data_array = ndarray
ndarray[:2]
ndarray = train_labels.values
ndarray.shape
train_labels_array = ndarray
ndarray[:5]
ndarray = validation_data.values
ndarray.shape
validation_data_array = ndarray
ndarray[:2]
ndarray = validation_labels.values
ndarray.shape
validation_labels_array = ndarray
ndarray[:5]

ndarray = test_data.values
ndarray.shape
test_data_array = ndarray
ndarray[:2]
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
train_data_array_scaled = minmax_scale.fit_transform(train_data_array)
validation_data_array_scaled = minmax_scale.fit_transform(validation_data_array)
test_data_array_scaled = minmax_scale.fit_transform(test_data_array)
train_data_array_scaled[:5]
validation_data_array_scaled[:5]
test_data_array_scaled[:5]
model = Sequential()

# input layer and hidden layer 1
model.add(Dense(units=40, input_dim=9, 
                kernel_initializer='uniform', 
                activation='relu'))

# hidden layer 2
model.add(Dense(units=30, 
                kernel_initializer='uniform', 
                activation='relu'))

# output layer
model.add(Dense(units=1, 
                kernel_initializer='uniform',
                activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['accuracy'])
train_history =model.fit(x=train_data_array_scaled, 
                         y=train_labels_array, 
                         validation_split=0.1, 
                         epochs=30, 
                         batch_size=30,verbose=2)
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')
scores = model.evaluate(x=validation_data_array_scaled, 
                        y=validation_labels_array)
scores[1]
test_probability=model.predict(test_data_array_scaled)
test_probability[test_probability >= 0.5] = 1
test_probability[test_probability < 0.5] = 0
test_probability[:10]
submission = pd.DataFrame()
#submission = pd.DataFrame({"PassengerId": csv_gender_submit["PassengerId"],"Survived": predictions})
submission['PassengerId'] = csv_gender_submit["PassengerId"]
submission['Survived'] = test_probability.astype(int)
#submission.info()
print (submission)

submission.to_csv("nn_submission.csv", index=False)

# the score of "nn_submission.csv" is 0.77033

