import numpy as np

import pandas as pd 

import scipy.stats as sci

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

import re
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')
def examine(data):

    print("Data Info \n")

    data.info()

    print(data.describe())

    print("\nNumber Of duplicate Values \n")

    temp = data.isnull().sum()

    print(data.duplicated().value_counts(),"\n")

    per = (temp /data.shape[0])*100

    print("Print Null Value Count\n")

    print("Number(0) and percentage(1) of null values\n")

    print(pd.concat([temp,per],axis = 1))
examine(df_train)
examine(df_test)
df_train.Cabin.unique(),df_test.Cabin.unique()
df_train.Age.hist(bins = 20)
df_train.Embarked.fillna(sci.mode(df_train.Embarked)[0][0],inplace = True) 
df_test.Age.hist(bins = 20)
df_test.Fare.fillna(sci.mode(df_train.Fare.dropna())[0][0],inplace = True) 
df_test.head()
sns.countplot(x = 'Survived' , data = df_train)
x = df_train.groupby(['Sex','Survived'])['Survived'].count()

sns.countplot(x = 'Sex', hue = 'Survived',data = df_train)

print(x)
print('Percentage of Females Survived of Pclass = 1 is',df_train[(df_train.Pclass == 1)&(df_train.Sex == 'female')]['Survived'].mean()*100)

print('Percentage of Females Survived of Pclass = 2 is',df_train[(df_train.Pclass == 2)&(df_train.Sex == 'female')]['Survived'].mean()*100)

print('Percentage of Females Survived of Pclass = 3 is',df_train[(df_train.Pclass == 3)&(df_train.Sex == 'female')]['Survived'].mean()*100)
print('Percentage of Males Survived of Pclass = 1 is',df_train[(df_train.Pclass == 1)&(df_train.Sex == 'male')]['Survived'].mean()*100)

print('Percentage of Males Survived of Pclass = 2 is',df_train[(df_train.Pclass == 2)&(df_train.Sex == 'male')]['Survived'].mean()*100)

print('Percentage of Males Survived of Pclass = 3 is',df_train[(df_train.Pclass == 3)&(df_train.Sex == 'male')]['Survived'].mean()*100)
plt.figure(figsize = [7,8])

sns.violinplot(x = 'Sex', y = 'Age',hue = 'Survived',data = df_train)
print('Percentage of Males survived age less than 15 are :',df_train[(df_train.Age <= 15) & (df_train.Sex == 'male')]['Survived'].mean()*100)
print(df_train[(df_train.Age < 15)&(df_train.Sex == 'male')].groupby(['Pclass'])['Survived'].mean())

df_train[(df_train.Age <= 15) & (df_train.Sex == 'male')].groupby(['Pclass','Survived'])['Survived'].mean()
print('Percentage of females survived age less than 15 are :',df_train[(df_train.Age <= 15) & (df_train.Sex == 'female')]['Survived'].mean()*100)
for dataset in [df_train,df_test]:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
df_train[['FamilySize', 'Survived']].groupby('FamilySize')['Survived'].mean()
plt.figure(figsize = [10,8])

sns.countplot(x = 'FamilySize', hue = 'Survived',data = df_train)
for dataset in [df_train, df_test]:

    dataset['IsAlone'] = 1

    dataset.loc[(dataset.FamilySize > 1), 'IsAlone'] = 0



df_train.groupby('IsAlone')['Survived'].mean()
df_train[(df_train.FamilySize == 1) & (df_train.Age < 16)].groupby(['Pclass','Sex'])['Survived'].mean()
df_train[(df_train.FamilySize == 1) & (df_train.Age.isnull())].groupby(['Pclass','Sex'])['Survived'].mean()
for dataset in [df_train, df_test]:

    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)



pd.crosstab(df_train['Title'], df_train['Sex'])

#used code from the discussion section of this competiton
for dataset in [df_train, df_test]:

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'Don', 'Rev'], 'High' )

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
print(pd.crosstab(df_train['Title'], df_train['Sex']))

print("*******************************************************")

print(pd.crosstab(df_test['Title'], df_test['Sex']))
df_test['Title'] = df_test['Title'].replace('Dona', 'High')
df_train[(df_train.Title == 'Master') | (df_train.Title == 'Miss')].groupby(['Title'])['Age'].mean()
df_train[df_train['Title'] == 'Miss'].describe()
df_train[(df_train['Title'] == 'Miss')&(df_train['FamilySize'] != 1)&(df_train['Parch'] != 0)]['Age'].mean()
#we have to create a new Title for Female Child which will be a subset of Title Miss

df_train.loc[(df_train.Title == 'Miss') & (df_train.Parch != 0) & (df_train.FamilySize > 1), 'Title'] = 'FemaleChild'

df_train[(df_train.Title == 'FemaleChild') & (df_train.Age.isnull())]
df_train.groupby(['Title'])['Survived'].mean()
df_test.loc[(df_test.Title == 'Miss') & (df_test.Parch != 0) & (df_test.FamilySize > 1), 'Title'] = 'FemaleChild'
df_train['Ticket'].describe()
df_train[(df_train.Title == 'FemaleChild') & (df_train.Age.isnull())]
combined = df_train.append(df_test,sort = False)

combined.shape
combined.groupby(['Pclass', 'Sex', 'Title'])['Age'].mean()
x = combined.groupby(['Pclass', 'Sex', 'Title'])['Age'].mean().reset_index()
x
def imputeage(row):

    return x[(x.Pclass == row.Pclass) & (x.Sex == row.Sex) & (x.Title == row.Title)]['Age'].values[0]
df_train['Age'], df_test['Age'] = [dataset.apply(lambda x: imputeage(x) if np.isnan(x['Age']) else x['Age'], axis = 1)for dataset in [df_train,df_test]]
df_train.head()
df_train.Cabin = df_train.Cabin.astype('str').apply(lambda x : re.findall("[a-zA-Z]",x)[0] if x !='nan' else 'T')
df_test.Cabin = df_test.Cabin.astype('str').apply(lambda x : re.findall("[a-zA-Z]",x)[0] if x !='nan' else 'T')
combined.Cabin = combined.Cabin.astype('str').apply(lambda x : re.findall("[a-zA-Z]",x)[0] if x !='nan' else 'T')
y = combined.groupby(['Pclass','Embarked']).apply(lambda x: x.Cabin.value_counts()).reset_index()
y.drop(['Cabin'],axis = 1,inplace = True)
y = y[~y.level_2.str.contains("T")]
y
def imputecabin(row):

    return y[(y.Pclass == row.Pclass) & (y.Embarked == row.Embarked)]['level_2'].values[0] 
df_train['Cabin'], df_test['Cabin'] = [dataset.apply(lambda x: imputecabin(x) if x['Cabin'] == 'T' else x['Cabin'], axis = 1)for dataset in [df_train,df_test]]
df_train.head()
df_train.Age.plot(kind = 'hist',bins = 50 )
df_train.Fare.plot(kind = 'hist')
df_train.FamilySize.plot(kind = 'hist')
df_train.Title.value_counts()
df_train.Cabin.value_counts()
df_train.Embarked.value_counts()
df_train.IsAlone.value_counts()
df_train.SibSp.value_counts()
df_train.Parch.value_counts()
df_train.Pclass.value_counts()
df_train.Pclass.unique()
df_train.SibSp.unique()
df_train.Parch.unique()
df_train.Title.unique()
df_train.head()
for dataset in [df_train,df_test]: 

    dataset['Age_bin'] = pd.cut(dataset['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])
for dataset in [df_train,df_test]:    

    dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','median_fare',

                                                                                      'Average_fare','high_fare'])
df_train.head()
df_train.Pclass.value_counts()
df_train['Age'] = df_train['Age'].astype('int')

df_train['Fare'] = df_train['Fare'].astype('int')

df_test['Age'] = df_test['Age'].astype('int')

df_test['Fare'] = df_test['Fare'].astype('int')

df_train['Age*Class'] = df_train['Age'] * df_train['Pclass']

df_test['Age*Class'] = df_test['Age'] * df_test['Pclass']
df_train.info()
def checkcomb(data,data2):

    print(df_train.groupby(data)['Survived'].mean())

    print(df_train.groupby([data,data2])['Survived'].mean())

    print(df_train.groupby([data,data2])['Survived'].count())
checkcomb('FamilySize','Sex')

checkcomb('Cabin','Sex')

checkcomb('Title','Sex')

checkcomb('Embarked','Sex')

checkcomb('Fare_bin','Sex')
df_train['ClassGen'] = df_train['Pclass'].astype('str') + df_train['Sex']

df_test['ClassGen'] = df_test['Pclass'].astype('str') + df_test['Sex']
df_train['FamGen'] = df_train['FamilySize'].astype('str') + df_train['Sex']

df_test['FamGen'] = df_test['FamilySize'].astype('str') + df_test['Sex']
df_train['AgeSex'] = df_train['Age_bin'].astype('str') + df_train['Sex']

df_test['AgeSex'] = df_test['Age_bin'].astype('str') + df_test['Sex']
df_train['EmbarkedSex'] = df_train['Embarked'].astype('str') + df_train['Sex']

df_test['EmbarkedSex'] = df_test['Embarked'].astype('str') + df_test['Sex']
df_train['Fare_binSex'] = df_train['Fare_bin'].astype('str') + df_train['Sex']

df_test['Fare_binSex'] = df_test['Fare_bin'].astype('str') + df_test['Sex']
df_train['TitleSex'] = df_train['Title'].astype('str') + df_train['Sex']

df_test['TitleSex'] = df_test['Title'].astype('str') + df_test['Sex']
df_train['CabinSex'] = df_train['Cabin'].astype('str') + df_train['Sex']

df_test['CabinSex'] = df_test['Cabin'].astype('str') + df_test['Sex']
df_train.head()
for dataset in [df_train,df_test]:

    drop_column = ['Age','Fare','Name','Ticket','Age*Class','Pclass','Age_bin','Embarked','Fare_bin','Title','Cabin','Sex','FamilySize']

    dataset.drop(drop_column, axis=1, inplace = True)
sns.heatmap(df_train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(20,12)

plt.show()
df_train = pd.get_dummies(data = df_train,columns = ["TitleSex","AgeSex","EmbarkedSex","Fare_binSex",'ClassGen','CabinSex','FamGen','IsAlone'])

df_test = pd.get_dummies(data =  df_test,columns = ["TitleSex","AgeSex","EmbarkedSex","Fare_binSex",'ClassGen','CabinSex','FamGen','IsAlone'])
from sklearn.model_selection import train_test_split #for split the data

from sklearn.metrics import accuracy_score  #for accuracy_score

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.metrics import confusion_matrix #for confusion matrix
df_test.info()
Y = df_train.Survived

df_train.drop(['Survived','PassengerId'],axis = 1,inplace = True)
from sklearn.model_selection import train_test_split

x_train,y_train = np.array(df_train),np.array(Y)

x_val,y_val = np.array(df_train),np.array(Y)

#x_train,x_val,y_train,y_val = train_test_split(np.array(df_train),np.array(Y),test_size = 0.3,random_state = 0,shuffle = False)
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

sc=StandardScaler()
x_train=sc.fit_transform(x_train)

x_val = sc.fit_transform(x_val)
reg = LogisticRegression().fit(x_train, y_train)

print(reg.score(x_train, y_train)) #0.14

print(reg.score(x_val, y_val))
from sklearn.feature_selection import SelectFromModel,RFE

smf = SelectFromModel(reg,threshold = -np.inf,max_features = 5)

smf.fit(x_train,y_train)

feature_idx = smf.get_support()

feature_name = df_train.columns[feature_idx]

feature_name
coeff_df = pd.DataFrame(df_train.columns)

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(reg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
coeff_df[abs(coeff_df['Correlation']) > 0.2]['Feature']
print(np.round(abs(reg.coef_),decimals = 2) > 0.2)
predictors = x_train

selector = RFE(reg,n_features_to_select = 1)

selector = selector.fit(predictors,y_train)
order = selector.ranking_

order
feature_ranks = []

for i in order:

    feature_ranks.append(f"{i}. {df_train.columns[i-1]}")

feature_ranks
from sklearn.model_selection import train_test_split

#x_train,y_train = np.array(df_train),np.array(Y)

#x_val,y_val = np.array(df_train),np.array(Y)

x_train,x_val,y_train,y_val = train_test_split(np.array(df_train),np.array(Y),test_size = 0.3,random_state = 0,shuffle = False)
x_train=sc.fit_transform(x_train)

x_val = sc.fit_transform(x_val)
reg = LogisticRegression(dual = False,random_state = 5,C= 0.14).fit(x_train, y_train)

print(reg.score(x_train, y_train)) #0.14

print(reg.score(x_val, y_val))
from sklearn.svm import LinearSVC

regr = LinearSVC(penalty = 'l1',dual = False,loss = 'squared_hinge')

regr.fit(x_train, y_train)

print(regr.score(x_train, y_train))

print(regr.score(x_val, y_val))
from keras.models import Sequential

from keras.layers import Dense

from tensorflow.keras.optimizers import Adam

import tensorflow as tf
model = Sequential()



model.add(Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l1(0.01)))



# The Output Layer :

model.add(Dense(1,activation='sigmoid',kernel_regularizer='l1'))



model.compile(optimizer=Adam(learning_rate=0.1, epsilon=1e-07, decay=0.01),loss='binary_crossentropy',metrics=['accuracy'])



model.fit(x_train,y_train,batch_size = 64,validation_data=(x_val,y_val),epochs=500,use_multiprocessing=True)
df_train.shape
x_test = df_test
x_test = sc.transform(x_test.drop(['PassengerId'],axis = 1))
svm_out = regr.predict(x_test)

print(svm_out)
nn_out = model.predict(x_test)
for i,j in enumerate(nn_out):

    if j < 0.5 :

        nn_out[i] = 0

    else :

        nn_out[i] = 1 
print(nn_out.flatten().astype('int'))
log_out = reg.predict(x_test)

print(log_out)
x_PassengerId = np.array(df_test['PassengerId'])
dict = {'PassengerId': x_PassengerId, 'Survived': nn_out.flatten().astype('int') }  

     

df = pd.DataFrame(dict) 

  

# saving the dataframe 

df.to_csv('21thsub.csv',index=False)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)

Y_pred = random_forest.predict(x_test)

random_forest.score(x_train, y_train)

acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)

acc_random_forest
Y_pred
x_PassengerId = np.array(df_test['PassengerId'])
dict = {'PassengerId': x_PassengerId, 'Survived': Y_pred_rf}  

     

df = pd.DataFrame(dict) 

  

# saving the dataframe 

df.to_csv('21thsub_rand.csv',index=False)