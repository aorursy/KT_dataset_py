import numpy as np 

import pandas as pd 

file_dirs = []

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file_dirs.append((os.path.join(dirname, filename)))

train_data = pd.read_csv('/kaggle/input/titanic/train.csv',index_col = 'PassengerId')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv',index_col = 'PassengerId')

cmp_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

train_data.describe()
train_data.head()
import matplotlib.pyplot as plt

import seaborn as sns
train_data.isna().sum()
train_data.drop('Cabin',inplace=True,axis=1)

test_data.drop('Cabin',inplace=True,axis=1)
sns.set_style('darkgrid')

plt.figure(figsize=(15,9))

ax = sns.distplot(train_data.Age)

ax.set_title("Age Distribution",fontsize=15)

plt.legend(['Skew: {:.2} , Kurt:{:.2} '.format(train_data.Age.skew(),train_data.Age.kurt())])

plt.show()
plt.figure(figsize=(15,9))

sns.regplot(train_data['Age'],train_data['Fare'])

plt.show()
plt.figure(figsize=(15,9))

sns.scatterplot(train_data['Age'],train_data['Fare'],hue=train_data['Parch'])

plt.show()
plt.figure(figsize=(15,9))

sns.barplot(y=train_data['Age'],x=train_data['Parch'],hue=train_data['Sex'])

plt.show()
plt.figure(figsize=(15,9))

ax = sns.scatterplot(y=train_data['Fare'],x=train_data['Age'],hue=train_data['Survived'])

ax.set_title('Survival Depending On Fare Amount And Age')

plt.show()
from sklearn.linear_model import LinearRegression
#for our train_data

missing_age_samepls = train_data[train_data.Age.isna()].copy()

not_missing_age = train_data[~train_data.Age.isna()].copy()



not_missing_age.Sex.replace({'female':0,'male':1},inplace=True)

missing_age_samepls.Sex.replace({'female':0,'male':1},inplace=True)



model = LinearRegression()



model.fit(not_missing_age[['Sex','Fare','Parch']],not_missing_age['Age'])

age_predictions = model.predict(missing_age_samepls[['Sex','Fare','Parch']])

missing_age_samepls['Age'] = age_predictions



train_data.loc[train_data.Age.isna(),'Age'] = age_predictions

train_data.Embarked.fillna('S',inplace=True)
#for out test data 

test_data.Fare.fillna(test_data.Fare.mean(),inplace=True)

missing_age_samepls = test_data[test_data.Age.isna()].copy()

not_missing_age = test_data[~test_data.Age.isna()].copy()



not_missing_age.Sex.replace({'female':0,'male':1},inplace=True)

missing_age_samepls.Sex.replace({'female':0,'male':1},inplace=True)



model = LinearRegression()



model.fit(not_missing_age[['Sex','Fare','Parch']],not_missing_age['Age'])

age_predictions = model.predict(missing_age_samepls[['Sex','Fare','Parch']])

missing_age_samepls['Age'] = age_predictions



test_data.loc[test_data.Age.isna(),'Age'] = age_predictions

train_data.isna().sum()
test_data.isna().sum()
train_data.replace({'male':1,'female':0},inplace=True)

test_data.replace({'male':1,'female':0},inplace=True)
correlation = train_data.corr()

plt.figure(figsize=(15,9))

ax = sns.heatmap(correlation,cmap='PiYG_r',annot=True)
plt.figure(figsize=(14,8))

ax =sns.countplot(x=train_data['Sex'],hue=train_data['Survived'])

ax.set_title('Survival Rate Depending On Gendder',fontsize=15)

plt.show()
plt.figure(figsize=(14,8))

ax =sns.countplot(train_data['Parch'],palette='GnBu',alpha=1)

ax2 =sns.countplot(train_data['SibSp'],alpha=1)

plt.show()
plt.figure(figsize=(14,8))

ax =sns.countplot(train_data['Embarked'],palette='GnBu',alpha=1)
train_data.replace({'S':2,'C':1,'Q':0},inplace=True)

test_data.replace({'S':2,'C':1,'Q':0},inplace=True)
age_groups = pd.cut(train_data.Age,6,labels=['0-13','14-26','27-40','41-53','54-66','67-80'])

train_data['Age Group'] = age_groups

age_groups_2 = pd.cut(test_data.Age,6,labels=['0-13','14-26','27-40','41-53','54-66','67-80'])

test_data['Age Group'] = age_groups_2

test_data
plt.figure(figsize=(14,8))

ax = sns.countplot(x=train_data['Survived'],hue=train_data['Age Group'])

ax.set_title('Amount Of Survivers And Not Survivers Per Age Group',fontsize=15)

ax.set_xticklabels(['Did Not Survive','Survived'])

plt.show()
train_data['Age Group'] = train_data['Age Group'].astype('category').cat.codes

test_data['Age Group'] = test_data['Age Group'].astype('category').cat.codes

train_data
plt.figure(figsize=(20,11))

ax = sns.boxplot(x=train_data['Pclass'],y=train_data['Fare'],hue = train_data['Survived'])
train_data = train_data[train_data['Fare'] < 120]

train_data = train_data[train_data['Age'] < 55]

pclass_d = pd.get_dummies(train_data.Pclass,prefix='Pclass_')

age_g_d = pd.get_dummies(train_data['Age Group'],prefix='Age_Group_')

parch_d = pd.get_dummies(train_data['Parch'],prefix='Parch_')



pclass_d = pclass_d[pclass_d.columns[:2]]

age_g_d = age_g_d[age_g_d.columns[:4]]

parch_d = parch_d[parch_d.columns[:5]]



train_data = pd.concat([train_data,pclass_d,age_g_d,parch_d],axis=1)



pclass_d = pd.get_dummies(test_data.Pclass,prefix='Pclass_')

age_g_d = pd.get_dummies(test_data['Age Group'],prefix='Age_Group_')

parch_d = pd.get_dummies(test_data['Parch'],prefix='Parch_')



pclass_d = pclass_d[pclass_d.columns[:2]]

age_g_d = age_g_d[age_g_d.columns[:4]]

parch_d = parch_d[parch_d.columns[:5]]



test_data = pd.concat([test_data,pclass_d,age_g_d,parch_d],axis=1)

test_data



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score as f1

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
correlation = train_data.corr()

plt.figure(figsize=(30,14))

ax = sns.heatmap(correlation,cmap='PiYG_r',annot=True)

#The Features will use for prediction 

from sklearn.feature_selection import SelectKBest,chi2

n_fets = 5

fets = [fet for fet in train_data.columns[3:] if fet  != 'Ticket']

#features = ['Fare','Age Group','SibSp','Pclass','Parch','Sex','Age']

selector = SelectKBest(chi2,k=n_fets)

X = selector.fit_transform(train_data[fets],train_data['Survived'])

y = train_data['Survived']



selected_feaures = train_data[fets].iloc[:,list(selector.get_support(indices=True))].columns.to_list()

selected_feaures
train_x,test_x,train_y,test_y = train_test_split(X,y)
#scaler = StandardScaler()

#train_x = scaler.fit_transform(train_x)

#test_x = scaler.fit_transform(test_x)
def RandomForest_Optimal_n(train_x,test_x,train_y,test_y,n_list):

    result = []

    for n in n_list:

        model = RandomForestClassifier(random_state=42,n_estimators=n,max_leaf_nodes=25)

        model.fit(train_x,train_y)

        pred = model.predict(test_x)

        result.append(f1(pred,test_y))

    return result
RF_results = RandomForest_Optimal_n(train_x,test_x,train_y,test_y,[5,10,30,50,100,130,150,200,350])

RF_results

plt.figure(figsize=(15,8))

ax = sns.lineplot(x=np.arange(9),y=RF_results)

ax.set_xlabel('Number Of Estimators',fontsize=13)

ax.set_ylabel('Accuracy Score',fontsize=13)

y_pos = RF_results.index(max(RF_results))

x_pos = np.arange(9)[y_pos]

ax.plot(np.arange(9)[x_pos],RF_results[x_pos],'r*',label='Best Score',ms=15)

ax.set_xticklabels([-1,5,10,30,50,100,130,150,200,350])

ax.legend()

plt.show()
def KNN_Optimal_n(train_x,test_x,train_y,test_y,n_list):

    result = []

    for n in n_list:

        model = KNeighborsClassifier(n_neighbors=n)

        model.fit(train_x,train_y)

        pred = model.predict(test_x)

        result.append(f1(pred,test_y))

    return result
KNN_results = KNN_Optimal_n(train_x,test_x,train_y,test_y,[1,2,5,8,10,20,35,42,50])

KNN_results

plt.figure(figsize=(15,8))

ax = sns.lineplot(x=np.arange(9),y=KNN_results)

ax.set_xlabel('Number Of Estimators',fontsize=13)

ax.set_ylabel('Accuracy Score',fontsize=13)

y_pos = KNN_results.index(max(KNN_results))

x_pos = np.arange(9)[y_pos]

ax.plot(np.arange(9)[x_pos],KNN_results[x_pos],'r*',label='Best Score',ms=15)

ax.set_xticklabels([-1,1,2,5,8,10,20,35,42,50])

ax.legend()

plt.show()
def SVM_Optimal_C(train_x,test_x,train_y,test_y,n_list):

    result = []

    for n in n_list:

        model = SVC(C=n,kernel='poly')

        model.fit(train_x,train_y)

        pred = model.predict(test_x)

        result.append(f1(pred,test_y))

    return result
SVM_results = SVM_Optimal_C(train_x,test_x,train_y,test_y,[1,2,5,20,35,42,50])

SVM_results

plt.figure(figsize=(15,8))

ax = sns.lineplot(x=np.arange(7),y=SVM_results)

ax.set_xlabel('Number Of Estimators',fontsize=13)

ax.set_ylabel('Accuracy Score',fontsize=13)

y_pos = SVM_results.index(max(SVM_results))

x_pos = np.arange(7)[y_pos]

ax.plot(np.arange(7)[x_pos],SVM_results[x_pos],'r*',label='Best Score',ms=15)

ax.set_xticklabels([-1,1,2,5,20,35,42,50])

ax.legend()

plt.show()
def ADA_Optimal_n(train_x,test_x,train_y,test_y,n_list):

    result = []

    for n in n_list:

        model = AdaBoostClassifier(n_estimators=n,learning_rate=0.6)

        model.fit(train_x,train_y)

        pred = model.predict(test_x)

        result.append(f1(pred,test_y))

    return result
ADA_results = ADA_Optimal_n(train_x,test_x,train_y,test_y,[1,2,5,20,35,42,50])

ADA_results

plt.figure(figsize=(15,8))

ax = sns.lineplot(x=np.arange(7),y=ADA_results)

ax.set_xlabel('Number Of Estimators',fontsize=13)

ax.set_ylabel('Accuracy Score',fontsize=13)

y_pos = ADA_results.index(max(ADA_results))

x_pos = np.arange(7)[y_pos]

ax.plot(np.arange(7)[x_pos],ADA_results[x_pos],'r*',label='Best Score',ms=15)

ax.set_xticklabels([-1,1,2,5,20,35,42,50])

ax.legend()

plt.show()
def DT_Optimal_n(train_x,test_x,train_y,test_y,n_list):

    result = []

    for n in n_list:

        model = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=n,splitter='random',

                                      random_state=42)

        model.fit(train_x,train_y)

        pred = model.predict(test_x)

        result.append(model.score(test_x,test_y))

    return result
DT_results = DT_Optimal_n(train_x,test_x,train_y,test_y,[2,3,5,7,20,35,42,50])

DT_results

plt.figure(figsize=(15,8))

ax = sns.lineplot(x=np.arange(8),y=DT_results)

ax.set_xlabel('Number Of Estimators',fontsize=13)

ax.set_ylabel('Accuracy Score',fontsize=13)

y_pos = DT_results.index(max(DT_results))

x_pos = np.arange(8)[y_pos]

ax.plot(np.arange(8)[x_pos],DT_results[x_pos],'r*',label='Best Score',ms=15)

ax.set_xticklabels([-1,2,3,5,7,20,35,42,50])

ax.legend()

plt.show()
from keras.layers import Dense

from keras.models import Sequential

model = Sequential()

model.add(Dense(64,activation='sigmoid',input_dim=n_fets))

model.add(Dense(36,activation='sigmoid'))

model.add(Dense(64,activation='sigmoid'))

model.add(Dense(8,activation='sigmoid'))

model.add(Dense(1,activation='sigmoid'))



model.compile(optimizer='adam',loss='binary_crossentropy',metrics='accuracy')
pred = model.predict(test_x)

print(classification_report(np.round(pred),test_y))
plt.figure(figsize=(15,8))

ax = sns.lineplot(x=np.arange(350/15),y=history.history['accuracy'][::15],color='g',label='Accuracy Growth')

ax.set_title('Neural Network Learning Process On Train/Test Data',fontsize=15)

ax.set_xlabel=('# Epoch / 15')

ax.set_ylabel=('Accuracy')

ax.legend(loc =2 ,prop={'size':16})

plt.show()
all_data_fit = model.fit(X,y,epochs=350)
plt.figure(figsize=(15,8))

ax = sns.lineplot(x=np.arange(350/15),y=all_data_fit.history['accuracy'][::15],color='g',label='Accuracy Growth')

ax.set_title('Neural Network Learning Process On The Entire DataSet',fontsize=15)

ax.set_xlabel=('# Epoch / 15')

ax.set_ylabel=('Accuracy')

ax.legend(loc =2 ,prop={'size':16})

plt.show()
from sklearn.ensemble import VotingClassifier

final_prediction = model.predict(test_data[selected_feaures])

dt_model = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=20,splitter='random',random_state=42)

ada_model = AdaBoostClassifier(n_estimators=5,learning_rate=0.6)

RF_model = RandomForestClassifier(random_state=42,n_estimators=300,max_leaf_nodes=25)



fcnn_prediction = np.round(final_prediction)

vc = VotingClassifier(estimators = [('RF',RF_model),('ada',ada_model),('dt',dt_model)],voting='hard',weights=[0.2,0.1,0.7])

vc.fit(X,y)



final_prediction = vc.predict(test_data[selected_feaures])



final_prediction = final_prediction.astype(np.int16)







submission = pd.read_csv('../input/titanic/gender_submission.csv')

submission['Survived'] = final_prediction

submission.to_csv('submission.csv', index=False)


