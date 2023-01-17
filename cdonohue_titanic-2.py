# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')
data.tail(5)
data.drop('Cabin',axis=1,inplace=True)

data.tail(5)
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_mean.fit(data['Age'].values.reshape(-1,1))

data['Age'] = imp_mean.transform(data['Age'].values.reshape(-1,1))

data.tail(10)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler() 

scaled_values = scaler.fit_transform(data['Age'].values.reshape(-1, 1)) 

data['Age'] = scaled_values
scaler = MinMaxScaler() 

scaled_values = scaler.fit_transform(data['Fare'].values.reshape(-1, 1)) 

data['Fare'] = scaled_values
data.isnull().sum()
data.dropna(how='any', inplace=True)

data.reset_index(drop=True, inplace=True)
data.info()
data.drop('PassengerId',axis=1,inplace=True)

data.head()
#data['Level'] = data.apply(lambda x : str(x)[0])

#data.head(5)

#data['Level'] = data['Cabin'].apply(lambda x : x[0])

#data.head(3)
'''

from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()

#df[['Ticket', 'Fare']]

dict_values = data[['Level']].T.to_dict().values()

one_hot_arr = vec.fit_transform(dict_values).toarray()

one_hot_df = pd.DataFrame(one_hot_arr, columns=vec.get_feature_names())

one_hot_df



level_features = one_hot_df.columns

'''
#data[one_hot_df.columns] = one_hot_df

#data
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()

#df[['Ticket', 'Fare']]

dict_values = data[['Sex']].T.to_dict().values()

one_hot_arr = vec.fit_transform(dict_values).toarray()

one_hot_df = pd.DataFrame(one_hot_arr, columns=vec.get_feature_names())

one_hot_df.head()
#one_hot_df.set_index(data.index)

data[one_hot_df.columns] = one_hot_df

data.head()
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()

#df[['Ticket', 'Fare']]

dict_values = data[['Embarked']].T.to_dict().values()

one_hot_arr = vec.fit_transform(dict_values).toarray()

one_hot_df = pd.DataFrame(one_hot_arr, columns=vec.get_feature_names())

one_hot_df.head()
#one_hot_df.set_index(data.index)

data[one_hot_df.columns] = one_hot_df

data.head()
data.describe()
import matplotlib.pyplot as plt

fig = data[data.Survived==0].plot(kind='scatter',x='Age',y='Fare',color='orange', label='Dead')

data[data.Survived==1].plot(kind='scatter',x='Age',y='Fare',color='blue', label='Survived',ax=fig)

fig.set_xlabel("Age")

fig.set_ylabel("Fare")

fig.set_title("Age vs Fare")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
data.hist(edgecolor='black', linewidth=1.2)

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
import seaborn as sns

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Survived',y='Age',data=data)

plt.subplot(2,2,2)

sns.violinplot(x='Survived',y='Fare',data=data)

plt.subplot(2,2,3)

sns.violinplot(x='Survived',y='SibSp',data=data)

plt.subplot(2,2,4)

sns.violinplot(x='Survived',y='Pclass',data=data)
plt.figure(figsize=(15,15)) 

sns.heatmap(data.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())

plt.show()
from sklearn.model_selection import train_test_split #to split the dataset for training and testing

train, test = train_test_split(data, test_size = 0.3)# in this our main data is split into train and test
data.columns
features = ['Fare', 'Sex=female', 'Sex=male', 'Embarked=C', 'Embarked=Q', 'Embarked=S', 'Pclass', 'Age', 'SibSp']

#features.extend(level_features)

pd.DataFrame(data[features], index=data.index).head(2)
sum(n < 0 for n in X.values.flatten())
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



X = data[features]  #independent columns

y = data['Survived']    #target column i.e price range

#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k='all')

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features
features = featureScores[featureScores.Score > .2]['Specs'].tolist()

features
train_X = train[features]

train_y = train['Survived']

test_X = test[features]

test_y = test['Survived']
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn import metrics #for checking the model accuracy



model = svm.SVC() #select the algorithm

model.fit(train_X,train_y) # we train the algorithm with the training data and the training output

prediction=model.predict(test_X) #now we pass the testing data to the trained algorithm

print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y))#now we check the accuracy of the algorithm. 

#we pass the predicted output by the model and the actual output
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm



model = LogisticRegression()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_y))
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm



model=DecisionTreeClassifier()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_y))

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours



model=KNeighborsClassifier(n_neighbors=10) #this examines 3 neighbours for putting the new data into a class

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_y))
a_index=list(range(1,11))

a=pd.Series()

x=[1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(train_X,train_y)

    prediction=model.predict(test_X)

    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_y)))

plt.plot(a_index, a)

plt.xticks(x)
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv1D, Dropout

'''

train_X = train_X.values.reshape(train_X.shape[0], len(features),1)

test_X = test_X.values.reshape(test_X.shape[0], len(features), 1)



model.add(Conv1D(12, kernel_size=1, activation="relu", input_shape=(len(features), 1)))

model.add(Conv1D(20, activation='relu', kernel_size=1, strides=2))

model.add(Conv1D(20, activation='relu', kernel_size=1))

model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()

'''
train_X = train_X.values

test_X = test_X.values



model = Sequential()

model.add(Dense(64, input_dim=len(features), activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
callbacks_list = [keras.callbacks.EarlyStopping(monitor='acc', patience=100)]

history = model.fit(train_X, train_y, batch_size=32, epochs = 500, validation_split = 0.2, callbacks = callbacks_list, verbose=1)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
print(model.metrics_names)

model.evaluate(test_X, test_y, verbose=1)
pred_y = pd.DataFrame(model.predict_classes(test_X, verbose=1))

pred_y.head(5)
from sklearn.metrics import confusion_matrix

confusion_matrix(test_y, pred_y)
labels = [0,1]

cm = confusion_matrix(test_y, pred_y, labels)

print(cm)

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

plt.title('Confusion matrix of the classifier')

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()
test_y = test_y.values

pred_y = pred_y.iloc[:,0].values

(test_y != pred_y).sum()/len(test_y)
data = pd.read_csv('../input/test.csv',sep=',',skipinitialspace=True,quotechar='"',engine='python')


from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_mean.fit(data['Age'].values.reshape(-1,1))

data['Age'] = imp_mean.transform(data['Age'].values.reshape(-1,1))



from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler() 

scaled_values = scaler.fit_transform(data['Age'].values.reshape(-1, 1)) 

data['Age'] = scaled_values



scaler = MinMaxScaler() 

scaled_values = scaler.fit_transform(data['Fare'].values.reshape(-1, 1)) 

data['Fare'] = scaled_values



from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()

dict_values = data[['Sex']].T.to_dict().values()

one_hot_df = vec.fit_transform(dict_values).toarray()

one_hot_df = pd.DataFrame(one_hot_df, columns=vec.get_feature_names())

one_hot_df.head(5)

data[one_hot_df.columns] = one_hot_df



vec = DictVectorizer()

dict_values = data[['Embarked']].T.to_dict().values()

one_hot_df = vec.fit_transform(dict_values).toarray()

one_hot_df = pd.DataFrame(one_hot_df, columns=vec.get_feature_names())

one_hot_df.head(5)

data[one_hot_df.columns] = one_hot_df



test_X = data[features]

#test_X = test_X.values.reshape(test_X.shape[0], len(features), 1)

test_X = test_X.values



pred_y = model.predict_classes(test_X, verbose=1)

output_df = data[['PassengerId']]

output_df['Survived'] = pd.DataFrame(pred_y)

output_df.head(10)
output_df.PassengerId.size
output_df.to_csv('output.csv', index=False)