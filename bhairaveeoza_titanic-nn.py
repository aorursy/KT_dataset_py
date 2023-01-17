import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import preprocessing



import keras 

from keras.models import Sequential 

from keras.layers import Dense      

 

from sklearn.metrics import accuracy_score,confusion_matrix,precision_score, recall_score, classification_report

train_orig = pd.read_csv('../input/titanic/train.csv')
def data_kyc(data):

    print("**INFO**")

    print(data.info())

    print("**DESCRIBE**")

    print(data.describe())

    print("**MISSING VALUES**")

    print(data.isna().sum())
data_kyc(train_orig)
sns.heatmap(train_orig.isna(),cmap="YlGnBu")
sns.boxplot(x='Parch',y='Age', data=train_orig)
parch_medians = dict(train_orig[['Age']].groupby(train_orig['Parch']).mean().apply((list)))['Age']
def map_age_parch(frame):

    age = frame[0]

    parch = frame[1]

    if pd.isnull(age):

        return parch_medians[parch]

    else:

        return age
train_orig['Age'] = train_orig[['Age','Parch']].apply(map_age_parch,axis=1)
train_orig.isna().sum() #All NA values in Age are removed
#what is the correlation between Cabin present with Survival

train_orig[["Cabin","Survived"]].groupby(train_orig["Cabin"].isnull()).mean()
#67% on an average survived when cabin was missing so correlation can be established

train_orig['Cabin_derived'] = np.where(train_orig['Cabin'].isnull(),0,1)
train_orig.isna().sum()
part = train_orig[['Parch','SibSp','Survived']]

d=part.corr()
sns.heatmap(d,cmap="YlGnBu") #SibSp and Parch have good correlation can be combined
sns.barplot(x='SibSp',y='Survived', data=train_orig) 

plt.show()

sns.barplot(x='Parch',y='Survived', data=train_orig) 

plt.show()
train_orig['family_count'] = train_orig['SibSp'] + train_orig['Parch']
part = train_orig[['Parch','SibSp','family_count','Survived']]

d=part.corr()
sns.heatmap(d,cmap="YlGnBu")
train_orig['Sex_int'] = train_orig['Sex'].map({'male':1, 'female':0})
train_orig.columns
train = train_orig.drop(['PassengerId','Name','Sex','Ticket','Embarked','Cabin','Parch','SibSp'], axis=1)
train.columns
train.head()
sns.heatmap(train.corr(),cmap="YlGnBu")
#Fare is strong correlation with Cabin_derived

train[["Fare"]].groupby(train["Cabin_derived"]).mean()

#if cabin derived is 1 then fare 76.14 else 19.15
train.corr() #Pclass has strong correlation with Fare, Cabin_derived and Family count so better to remove it
train=train.drop('Pclass',axis=1)
def pre_processing(data):

    data['Age'] = data[['Age','Parch']].apply(map_age_parch,axis=1)

    data['Cabin_derived'] = np.where(data['Cabin'].isnull(),0,1)

    data['family_count'] = data['SibSp'] + data['Parch']

    data['Sex_int'] = data['Sex'].map({'male':1, 'female':0})

    data=data.drop(['PassengerId','Name','Sex','Ticket','Embarked','Cabin','Parch','SibSp','Pclass'], axis=1)

    return data
x = train.drop("Survived",axis=1)

y = train["Survived"]
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2, random_state=7)
sns.countplot(Y_train) #target variable is not normalized so need to use standard scaler
scaler = preprocessing.StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
print(X_train.shape)

print(Y_test.shape)
X_train #all are converted into (x - u) / s
model = Sequential()

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5))

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, Y_train, batch_size = 32, epochs = 200)
y_pred = model.predict(X_test)
y_pred_final =  (y_pred > 0.5).astype(int).reshape(X_test.shape[0])
accuracy_score(y_pred_final, Y_test)
con_mat = confusion_matrix(y_pred_final, Y_test)
plt.imshow(con_mat, interpolation='nearest', cmap="YlGnBu")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.ylabel('True label')

plt.xlabel('Predicted label')
precision_score(y_pred_final, Y_test)
print(classification_report(y_pred_final, Y_test))
test_orig = pd.read_csv("../input/titanic/test.csv")
test_orig.columns
test_orig.isna().sum()
sns.countplot(test_orig['Parch'])
parch_medians[9]=parch_medians.mean()
parch_medians
test = pre_processing(test_orig)
test.isna().sum()
test[test['Fare'].isnull()]
# as seen in the training data if cabin derived is 1 then fare 76.14 else 19.15

test.iloc[152, test.columns.get_loc('Fare')] = 19.15
test.isna().sum()
scaler = preprocessing.StandardScaler()

test = scaler.fit_transform(test)
Y_pred_test = model.predict(test)
Y_pred_test_final =  (Y_pred_test > 0.5).astype(int).reshape(test.shape[0])
print(test_orig.shape)

print(Y_pred_test_final.shape)
submission_frame = pd.DataFrame({'PassengerId':test_orig['PassengerId'],'Survived':Y_pred_test_final})

submission_frame.to_csv("NN_submit.csv")
sns.countplot(submission_frame['Survived'])