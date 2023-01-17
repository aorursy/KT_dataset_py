import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)
import tensorflow as tf
from tensorflow.python.framework import ops
import math


from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras import optimizers
# Read CSV train data file into DataFrame
train_df = pd.read_csv("../input/train.csv")

# Read CSV test data file into DataFrame
test_df = pd.read_csv("../input/test.csv")

# preview train data
train_df.head()
print('The number of samples into the train data is {}.'.format(train_df.shape[0]))
# preview test data
test_df.head()
print('The number of samples into the test data is {}.'.format(test_df.shape[0]))
# check missing values in train data
train_df.isnull().sum()
# percent of missing "Age" 
print('Percent of missing "Age" records is %.2f%%' %((train_df['Age'].isnull().sum()/train_df.shape[0])*100))
ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_df["Age"].plot(kind='density', color='teal')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()
# mean age
print('The mean of "Age" is %.2f' %(train_df["Age"].mean(skipna=True)))
# median age
print('The median of "Age" is %.2f' %(train_df["Age"].median(skipna=True)))
# percent of missing "Cabin" 
print('Percent of missing "Cabin" records is %.2f%%' %((train_df['Cabin'].isnull().sum()/train_df.shape[0])*100))
# percent of missing "Embarked" 
print('Percent of missing "Embarked" records is %.2f%%' %((train_df['Embarked'].isnull().sum()/train_df.shape[0])*100))
print('Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):')
print(train_df['Embarked'].value_counts())
sns.countplot(x='Embarked', data=train_df, palette='Set2')
plt.show()
print('The most common boarding port of embarkation is %s.' %train_df['Embarked'].value_counts().idxmax())
train_data = train_df.copy()
train_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
train_data["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)
# check missing values in adjusted train data
train_data.isnull().sum()
# preview adjusted train data
train_data.head()
plt.figure(figsize=(15,8))
ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_df["Age"].plot(kind='density', color='teal')
ax = train_data["Age"].hist(bins=15, density=True, stacked=True, color='orange', alpha=0.5)
train_data["Age"].plot(kind='density', color='orange')
ax.legend(['Raw Age', 'Adjusted Age'])
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()
## Create categorical variable for traveling alone
train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)
#create categorical variables and drop some variables
training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])
training.drop('Sex_female', axis=1, inplace=True)
# training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)

final_train = training
final_train.head()
test_df.isnull().sum()
test_data = test_df.copy()
test_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
test_data["Fare"].fillna(train_df["Fare"].median(skipna=True), inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)

test_data.drop('SibSp', axis=1, inplace=True)
test_data.drop('Parch', axis=1, inplace=True)

testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])
testing.drop('Sex_female', axis=1, inplace=True)
# testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)

final_test = testing
final_test.head()
plt.figure(figsize=(15,8))
ax = sns.kdeplot(final_train["Age"][final_train.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(final_train["Age"][final_train.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()
plt.figure(figsize=(20,8))
avg_survival_byage = final_train[["Age", "Survived"]].groupby(['Age'], as_index=False).mean()
g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="LightSeaGreen")
plt.show()
final_train['IsMinor']=np.where(final_train['Age']<=16, 1, 0)

final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)

sns.barplot('Sex', 'Survived', data=train_df, color="aquamarine")
plt.show()
final_train.head(10)
final_test.head()
def one_hot_encoder(labels, C):
    C = tf.constant(C, dtype = tf.int32)
    one_hot_matrix = tf.one_hot(labels, depth = C, axis = 0)
    session = tf.Session()
    one_hot_mat = session.run(one_hot_matrix)
    session.close()
    return one_hot_mat
survived = final_train['Survived'].tolist()
test_labels = survived[:]
test_labels = np.array(test_labels)
test_labels = one_hot_encoder(test_labels, 2)
print(test_labels.shape)
cols = ["Age","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 
# X = final_train[cols]
# y = final_train['Survived']
# Get labels

# y = test_labels.T
y = final_train[['Survived']].values
# Get inputs; we define our x and y here.
X = final_train[cols]
print(X.shape, y.shape) # Print shapes just to check


#Split the dataset into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=0)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
def get_model(n_x, n_h1, n_h2):
#     np.random.seed(10)
#     K.clear_session()
    model = Sequential()
    model.add(Dense(n_h1, input_dim=n_x, activation='relu'))
    model.add(Dense(n_h2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    adam = optimizers.Adam(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    return model
(n_x, m) = X.T.shape
n_h1 = 8
n_h2 = 12

batch_size = 128
epochs = 200

print(n_x)
model = get_model(n_x, n_h1, n_h2)
model.fit(X, y,epochs=epochs, batch_size=batch_size, verbose=2)
# predictions = model.predict_classes(final_test[cols])

final_test['Survived'] = model.predict_classes(final_test[cols])
final_test.head(10)
submit = final_test[['PassengerId','Survived']]
submit.to_csv("../working/submit.csv", index=False)
