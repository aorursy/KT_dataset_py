# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Reading Data
train_df=pd.read_csv('/kaggle/input/titanic/train.csv')
test_df=pd.read_csv('/kaggle/input/titanic/test.csv')
train_df.head()


sns.set(rc={'figure.figsize':(9.7,8.27)})
sns.heatmap(train_df.isnull(), cbar=False).set_title('Train Data');


sns.set(rc={'figure.figsize':(9.7,8.27)})
sns.heatmap(test_df.isnull(), cbar=False).set_title('Test Data');

print("TRAIN MISING FEATURES STATS")
print("age missing: "+ str(sum(train_df['Age'].isnull())))
print("cabin missing:"+str(sum(train_df['Cabin'].isnull())))
print("embark missing:"+str(sum(train_df['Embarked'].isnull())));
sns.set(rc={'figure.figsize':(9.7,8.27)})
sns.heatmap(test_df.isnull(), cbar=False).set_title('Test Data');
print("TEST MISSING FEATURES STATS")
print("age missing: "+ str(sum(test_df['Age'].isnull())))
print("cabin missing:"+str(sum(test_df['Cabin'].isnull())))

train_df.columns
train_df['Pclass'].describe()
sns.set(rc={'figure.figsize':(7,5)})
sns.countplot(train_df['Pclass']);
train_df['Parch'].unique()
train_df['Age'].describe()
#For distribution plot let's convert the NaNs with -1 for plot
sns.distplot(train_df['Age'].fillna(-20));
sns.set(rc={'figure.figsize':(15,5)})
plt.subplot(121)
sns.countplot(train_df['SibSp']);
plt.subplot(122)
sns.countplot(train_df['Parch']);
print(train_df[['SibSp','Parch']].describe())
train_df['Fare'].describe()
sns.set(rc={'figure.figsize':(7,5)})
sns.distplot(train_df['Fare']);
train_df['Name'].describe()
x=train_df['Name'][~train_df['Name'].str.contains('Mr.',regex=False)] #891(total) - 374(NotMr.) = 517 (Mr.)
y=x[~x.str.contains('Mrs.',regex=False)] #517-249=268 Mrs.
z=y[~y.str.contains('Master.',regex=False)] #268-209=59 Master.
a=z[~z.str.contains('Miss.',regex=False)] #209-27=182 Miss.
b=a[~a.str.contains('Rev.',regex=False)] #27-21=6 Rev.
c=b[~b.str.contains('Dr.',regex=False)] #21-14=7 Dr.
# 14 other titles

#Manipulation: Changing the name into category variable
#RUN ONCE - as inplace changes
encod_df=train_df.copy()
encod_df.loc[train_df['Name'].str.contains('Mr.',regex=False,na=False),'Name'] = 1
encod_df.loc[train_df['Name'].str.contains('Master.',regex=False,na=False),'Name'] = 2
encod_df.loc[train_df['Name'].str.contains('Miss.',regex=False,na=False),'Name'] = 3
encod_df.loc[train_df['Name'].str.contains('Mrs.',regex=False,na=False),'Name'] = 4
encod_df.loc[train_df['Name'].str.contains('Dr.',regex=False,na=False),'Name'] = 5
encod_df.loc[train_df['Name'].str.contains('Rev.',regex=False,na=False),'Name'] = 6
encod_df.loc[(encod_df['Name']!=1)&(encod_df['Name']!=2)&(encod_df['Name']!=3)&(encod_df['Name']!=4)&(encod_df['Name']!=5)&
        (encod_df['Name']!=6),'Name']=7

sns.countplot(encod_df['Name']);
print(encod_df['Name'].unique())
train_df['Sex'].describe()
sns.set(rc={'figure.figsize':(7,5)})
sns.countplot(train_df['Sex']);
#Manipulation
encod_df.loc[train_df['Sex']=='male','Sex']=0
encod_df.loc[train_df['Sex']=='female','Sex']=1
sns.countplot(encod_df['Sex']);
encod_df.head()
train_df['Ticket'].describe()
train_df['Ticket'].head(10)
train_df['Cabin'].isna().sum()
train_df['Cabin'].describe()
train_df['Cabin'].head()
sns.set(rc={'figure.figsize':(7,5)})
print(train_df.groupby('Cabin').count()['Ticket'])
sns.countplot(train_df.groupby('Cabin').count()['Ticket']);
train_df['Embarked'].describe()
sns.set(rc={'figure.figsize':(7,5)})
sns.countplot(train_df['Embarked']);
#Manipulation
encod_df.loc[encod_df['Embarked']=='S','Embarked']=1
encod_df.loc[encod_df['Embarked']=='C','Embarked']=2
encod_df.loc[encod_df['Embarked']=='Q','Embarked']=3
sns.countplot(encod_df['Embarked']);
#Manipulation
encod_df.dropna(inplace=True)
#Dropping few columns 
#encod_df.drop(columns=['PassengerId','Ticket','Cabin'],inplace=True)
encod_df.reset_index()
encod_df['Embarked']=encod_df['Embarked'].astype(int)
print(encod_df.columns)

#Correlation heatmaps
sns.set(rc={'figure.figsize':(12,7)})
sns.heatmap(encod_df.corr().astype(float),vmin=-1,vmax=1,center=0,annot=True);
df2=train_df.groupby(['Sex','Survived'])['Sex'].count()
df2
total_passengers=train_df.shape[0]

# How many Males vs. Females were aboard?
fig,ax=plt.subplots(1,2,figsize=(17,5))
df2=train_df.groupby(['Sex','Survived'])['Sex'].count()
gender_survival=np.array([[df2['male'][1],df2['female'][1]],[df2['male'][0],df2['female'][0]]])
temp=pd.DataFrame(gender_survival,columns=['Male','Female'],index=['Survived','Dead'])
ax[0].set_title('Gender Survival Rate')
temp.plot.bar(stacked=True,ax=ax[0]);

#Plot the perecent of saved and died passengers gender wise
labels=['Males-Survived','Males-Died','Females-Survived','Females-Died']
colors=['lightblue', 'lightblue','pink','pink']
explode=[0.1,0,0,0.1]
ax[1].pie([gender_survival[0,0],gender_survival[1,0],gender_survival[0,1],gender_survival[1,1]],startangle=90,labels=labels,colors=colors,autopct='%1.1f%%',explode=explode);
ax[1].set_title('Gender Wise passenger survival rate');
encod_df.groupby(['Name'])['Survived'].sum().plot.bar().set_title("Survived vs. Names");
fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(18,5))
encod_df.groupby('SibSp')['Survived'].sum().plot.bar(ax=axes[0]).set_title("SibSp vs. Survived");
encod_df[['Fare','Survived']].plot.line(ax=axes[1]).set_title("Fares vs. Survived");
encod_df.groupby('Parch')['Survived'].sum().plot.bar(ax=axes[2]).set_title("Parch vs. Survived");
#Plot pclass against survived passengers to view numbe of saved passengers of each class.
fig,ax=plt.subplots(1,2,figsize=(17,5))
plt.style.use('seaborn-whitegrid')
ax[0].set_title('Number of Passengers survived vs. dies Pclass wise')
sns.countplot(y='Pclass',hue='Survived',data=train_df,ax=ax[0]);

#Plot the perecent of saved and died passengers for each class w.r.t total saved and died passengers
labels=['Pc1-Died','Pc1-Survived','Pc2-Died','Pc2-Survived','Pc3-Died','Pc3-Survived']
colors=['lightgreen', 'lightgreen', 'coral', 'coral','lightblue','lightblue']
pclass_survival=train_df.groupby(['Pclass','Survived'])['Survived'].count()
pclass_survival=np.array([pclass_survival[1,0],pclass_survival[1,1],pclass_survival[2,0],pclass_survival[2,1],pclass_survival[3,0],pclass_survival[3,1]])
explode=[0.1,0,0,0,0,0.1]
ax[1].pie(pclass_survival,startangle=90,labels=labels,colors=colors,autopct='%1d%%',explode=explode);
ax[1].set_title('Class Wise passenger survival rate');
fig,ax = plt.subplots(1,2,figsize=(19,5))
encod_df[encod_df['Survived']==0]['Age'].plot.hist(ax=ax[0]).set_title("Age Histogram of Died Passengers")
encod_df[encod_df['Survived']==1]['Age'].plot.hist(ax=ax[1]).set_title("Age Histogram of Survived Passengers");

encod_df.columns
# Feature engineering on train dataframe
encod_df['Family']=encod_df['Parch']+encod_df['SibSp']
X_train=encod_df[['Pclass', 'Name', 'Sex', 'Age', 'Family', 'Fare','Embarked', 'Family']]
Y_train=encod_df['Survived']
#Feature engineering on test dataframe
#we cannot drop nans rows form test dataframe for submission purposes
#name feature
test_df.loc[test_df['Name'].str.contains('Mr.',regex=False,na=False),'Name'] = 1
test_df.loc[test_df['Name'].str.contains('Master.',regex=False,na=False),'Name'] = 2
test_df.loc[test_df['Name'].str.contains('Miss.',regex=False,na=False),'Name'] = 3
test_df.loc[test_df['Name'].str.contains('Mrs.',regex=False,na=False),'Name'] = 4
test_df.loc[test_df['Name'].str.contains('Dr.',regex=False,na=False),'Name'] = 5
test_df.loc[test_df['Name'].str.contains('Rev.',regex=False,na=False),'Name'] = 6
test_df.loc[(test_df['Name']!=1)&(test_df['Name']!=2)&(test_df['Name']!=3)&(test_df['Name']!=4)&(test_df['Name']!=5)&
        (test_df['Name']!=6),'Name']=7
#Sex feature
test_df.loc[test_df['Sex']=='male','Sex']=0
test_df.loc[test_df['Sex']=='female','Sex']=1
#Family=SibSp+Parch
test_df['Family']=test_df['Parch']+test_df['SibSp']
#Embarked
test_df.loc[test_df['Embarked']=='S','Embarked']=1
test_df.loc[test_df['Embarked']=='C','Embarked']=2
test_df.loc[test_df['Embarked']=='Q','Embarked']=3

X_test=test_df[['Pclass', 'Name', 'Sex', 'Age', 'Family', 'Fare','Embarked', 'Family']]
#using last 50 samples of the train set to acess model's performace
x_test=X_train[-50:]
y_test=Y_train[-50:]
X_train=X_train[:-50]
Y_train=Y_train[:-50]
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,f1_score
gnb=GaussianNB()
y_pred = gnb.fit(X_train, Y_train).predict(x_test)
print("Naive Bayes accuracy score: "+str(accuracy_score(y_test,y_pred)*100)+'%')
print("Naive Bayes F1 score: "+str(np.round(f1_score(y_test,y_pred),2)))
logr=LogisticRegression(random_state=0,solver='liblinear')
y_pred = logr.fit(X_train, Y_train).predict(x_test)
print("Logistic Regression accuracy score: "+str(accuracy_score(y_test,y_pred)*100)+'%')
print("Logistic Regression F1 score: "+str(np.round(f1_score(y_test,y_pred),2)))
dtree=DecisionTreeClassifier(random_state=0,criterion='entropy')
y_pred = dtree.fit(X_train, Y_train).predict(x_test)
print("Decesion Tree accuracy score: "+str(accuracy_score(y_test,y_pred)*100)+'%')
print("Decesion TreeF1 score: "+str(np.round(f1_score(y_test,y_pred),2)))
rtree=RandomForestClassifier(random_state=0,n_estimators=15,criterion='entropy')
y_pred = rtree.fit(X_train, Y_train).predict(x_test)
print("Random Forest accuracy score: "+str(accuracy_score(y_test,y_pred)*100)+'%')
print("Random Forest F1 score: "+str(np.round(f1_score(y_test,y_pred),2)))
knbrs=KNeighborsClassifier(n_neighbors=15)
y_pred = knbrs.fit(X_train, Y_train).predict(x_test)
print("KNeighborsClassifier accuracy score: "+str(accuracy_score(y_test,y_pred)*100)+'%')
print("KNeighborsClassifier F1 score: "+str(np.round(f1_score(y_test,y_pred),2)))
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(9,input_dim=8,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=15, batch_size=20)
_, accuracy = model.evaluate(x_test, y_test)
print('Simple NN Accuracy: %.2f' % (accuracy*100))
X_test.fillna(0.,inplace=True)
Y_test=rtree.predict(X_test)
temp=np.ones((test_df.shape[0],2),dtype=int)
temp[:,0]=test_df['PassengerId']
temp[:,1]=Y_test
pd.DataFrame(temp,columns=['PassengerId','Survived']).to_csv("MomalTitanicSubmission.csv")

