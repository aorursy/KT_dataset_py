import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

import re

import seaborn as sns

warnings.simplefilter(action='ignore')





data_train=pd.read_csv("../input/titanic/train.csv")  ## importing files train and test

data_test=pd.read_csv("../input/titanic/test.csv")



data_train.info()
data_train.head()
sns.countplot(x = 'Survived', data=data_train )
x=data_train.groupby(['Sex','Survived'])['Survived'].count()

sns.countplot(x='Sex',hue='Survived',data=data_train)



print(x)
print("percentage of female of pclass1 is =",data_train[(data_train.Pclass == 1) & (data_train.Sex=='female')]['Survived'].mean()*100)

print("percentage of female of pclass2 is =",data_train[(data_train.Pclass == 2) & (data_train.Sex=='female')]['Survived'].mean()*100) 

print("percentage of female of pclass3 is =",data_train[(data_train.Pclass == 3) & (data_train.Sex=='female')]['Survived'].mean()*100)      
print("percentage of male of pclass1 is =",data_train[(data_train.Pclass == 1) & (data_train.Sex=='male')]['Survived'].mean()*100)

print("percentage of male of pclass2 is =",data_train[(data_train.Pclass == 2) & (data_train.Sex=='male')]['Survived'].mean()*100)

print("percentage of male of pclass3 is =",data_train[(data_train.Pclass == 1) & (data_train.Sex=='male')]['Survived'].mean()*100)
plt.figure(figsize=[8,8])

sns.violinplot(x= "Sex", y = 'Age', hue = 'Survived', data =data_train)
print("percentage of male survived less than 15 of age",data_train[(data_train.Age<=15) & (data_train.Sex =='male')]['Survived'].mean()*100)

print("percentage of female survived less than 15 of age",data_train[(data_train.Age<=15) & (data_train.Sex =='female')]['Survived'].mean()*100)
print(data_train[(data_train.Age < 15) & (data_train.Sex == 'male')].groupby(['Pclass'])['Survived'].mean())



data_train[(data_train.Age < 15) & (data_train.Sex == 'male')].groupby(['Pclass','Survived'])['Survived'].count()
sns.kdeplot(data_train[data_train.Sex == 'male']['Age'], shade = True, color = 'r')

sns.kdeplot(data_train[data_train.Sex == 'female']['Age'], shade = True, color = 'g')
# now making changes in both data together



for dataset in [data_train,data_test]:

    

    dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1
data_train[['FamilySize','Survived']].groupby('FamilySize')['Survived'].mean()
plt.figure(figsize = [10,8])

sns.countplot(x = 'FamilySize', hue = 'Survived', data = data_train)
data_train[(data_train.Age.isnull()) & (data_train.FamilySize == 1)].groupby(['Pclass', 'Sex'])['Survived'].mean()
for dataset in [data_train,data_test]:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    

pd.crosstab(data_train['Title'], data_train['Sex'])    
for dataset in [data_train,data_test]:

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Countess', 'Jonkheer', 'Lady', 'Major', 'Sir','Dona','Don','Dr'], 'High' )

    dataset['Title']=dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title']=dataset['Title'].replace('Ms', 'Miss')

    dataset['Title']=dataset['Title'].replace('Mme', 'Mrs')

print(pd.crosstab(data_train['Title'], data_train['Sex']))

print("*******************************************************")

print(pd.crosstab(data_test['Title'], data_test['Sex']))

data_test[data_test['Title'] == 'Miss'].describe()
#we have to create a new Title for Female Child which will be a subset of Title Miss

for dataset in [data_train,data_test]:



    dataset.loc[(data_train.Title == 'Miss') & (dataset.Parch != 0) & (dataset.FamilySize > 1), 'Title'] = 'FemaleChild'

    dataset[(dataset.Title == 'FemaleChild') & (dataset.Age.isnull())]
data_train.groupby(['Pclass', 'Sex', 'Title'])['Age'].mean()
x = data_train.groupby(['Pclass', 'Sex', 'Title'])['Age'].mean().reset_index()

type(x)
def imputeage(row):

    return x[(x.Pclass == row.Pclass) & (x.Sex == row.Sex) & (x.Title == row.Title)]['Age'].values[0]
data_train['Age'], data_test['Age'] = [dataset.apply(lambda x: imputeage(x) if np.isnan(x['Age']) else x['Age'], axis = 1)\

                          for dataset in [data_train, data_test]]

combined = data_train.append(data_test)

combined.shape



data_train['PersonPerTicket'] = data_train['Ticket'].map(combined['Ticket'].value_counts())

data_train['PricePerTicket'] = data_train['Fare']/data_train['PersonPerTicket']





data_train.isnull().sum()
xt=data_train['Age'].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)

data_train["Age"].plot(kind='density', color='teal')





plt.xlim(-10,85)

plt.show()



###as we see that the graph is left skewed so we take median



# BUT THERE IS ONE PROBLEM WITH THIS IS THAT GIRL CHILD LOST HER CHANCE OF SURVIVAL
print("Percent of missing Embarked records is %.2f%%'" %((data_train['Embarked'].isnull().sum()/data_train.shape[0])*100))



print(data_train['Embarked'].value_counts())





## so that the value of s is max so the nan is replace by s
data_train.info()
  # create a copy of the original data

train_df=data_train.copy()           

train_df['Fare']=train_df['Fare'].astype('float')

train_df['Age_Group'] = pd.cut(train_df.Age,bins=[0,2,17,65,99],labels=['Toddler/Baby','Child','Adult','Elderly'])

train_df['Embarked'].fillna(train_df['Embarked'].value_counts().idxmax(),inplace=True)



train_df.Cabin=train_df.Cabin.astype('str').apply(lambda x : re.findall("[a-zA-Z]",x)[0] if x !='nan' else 'T')



## combining our data 

train_df['travle_alone']=np.where(train_df['SibSp']+train_df['Parch']>0,0,1) ## for false value 0 and for true value 1

train_df.drop('SibSp', axis=1, inplace=True)

train_df.drop('Parch', axis=1, inplace=True)







## i use to dummy my data ...... to make it understandable value



train_df=pd.get_dummies(train_df, columns=["Pclass","Embarked","Sex","Age_Group","Cabin","Title"])



train_df.drop('Name', axis=1, inplace=True)

train_df.drop('Ticket', axis=1, inplace=True)

train_df.drop('PassengerId', axis=1, inplace=True)





train_df
train_df.isnull().sum()

data_t=train_df.copy()



y=data_t['Survived']



data_t.drop('Survived',axis=1,inplace=True)

x=data_t

x.shape
x.info()
x=x[['Title_Miss','Sex_male','Title_Rev','Pclass_2','Cabin_F','travle_alone','Cabin_E','Embarked_Q','Pclass_3','Cabin_G','Age_Group_Elderly','FamilySize','Cabin_A','Title_Master','Embarked_S','Title_High']]
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(np.array(x),np.array(y),train_size=0.7,random_state=0)

y_test.shape,x_test.shape
x_train.shape,y_test.shape


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.transform(x_test)


from sklearn.linear_model import LogisticRegression

clf=LogisticRegression()

clf.fit(x_train,y_train)



clf.score(x_train,y_train),clf.score(x_test,y_test)

coeff_df = pd.DataFrame(x.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(clf.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
from sklearn.feature_selection import SelectFromModel,RFE

smf = SelectFromModel(clf,threshold = -np.inf,max_features = 15)

smf.fit(x_train,y_train)

feature_idx = smf.get_support()

feature_name = x.columns[feature_idx]

feature_name
predictors = x_train

selector = RFE(clf,n_features_to_select = 1)

selector = selector.fit(predictors,y_train)
order = selector.ranking_

order
feature_ranks = []

for i in order:

    feature_ranks.append(f"{i}. {x.columns[i-1]}")

feature_ranks


from sklearn.svm import LinearSVC

regr = LinearSVC(penalty = 'l1',dual = False,loss = 'squared_hinge')

regr.fit(x_train, y_train)

print(regr.score(x_train, y_train))

print(regr.score(x_test, y_test))



from sklearn.ensemble import RandomForestClassifier

rfr = RandomForestClassifier(max_depth=11,random_state=0)

rfr.fit(x_train, y_train)

y_pred=rfr.predict(x_test)

print(rfr.score(x_test, y_test))

print(rfr.score(x_train, y_train))
import scipy as sci

test_data=data_test.copy()    

test_data['Fare']=test_data['Fare'].astype('float')

test_data.Fare.fillna(sci.median(test_data.Fare.dropna()),inplace = True) 

test_data['Age_Group'] = pd.cut(test_data.Age,bins=[0,2,17,65,99],labels=['Toddler/Baby','Child','Adult','Elderly'])

test_data['Embarked'].fillna(test_data['Embarked'].value_counts().idxmax(),inplace=True)

test_data.Cabin=test_data.Cabin.astype('str').apply(lambda x : re.findall("[a-zA-Z]",x)[0] if x !='nan' else 'T')



test_data['PersonPerTicket'] = test_data['Ticket'].map(combined['Ticket'].value_counts())

test_data['PricePerTicket'] = test_data['Fare']/test_data['PersonPerTicket']



## combining our data 

test_data['travle_alone']=np.where(test_data['SibSp']+test_data['Parch']>0,0,1) ## for false value 0 and for true value 1

test_data.drop('SibSp', axis=1, inplace=True)

test_data.drop('Parch', axis=1, inplace=True)





## i use to dummy my data ...... to make it understandable value



test_data=pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex","Age_Group","Cabin","Title"])



test_data.drop('Name', axis=1, inplace=True)

test_data.drop('Ticket',axis=1, inplace=True)

test_data.drop('PassengerId',axis=1, inplace=True)



test_data.head()
test_data.isnull().sum()

data_te=test_data.copy()

x.shape

data_te.shape

data_te=data_te[['Title_Miss','Sex_male','Title_Rev','Pclass_2','Cabin_F','travle_alone','Cabin_E','Embarked_Q','Pclass_3','Cabin_G','Age_Group_Elderly','FamilySize','Cabin_A','Title_Master','Embarked_S','Title_High']]


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit_transform(x)

x_test1=sc.transform(data_te)



clf.fit(x,y)

clf.score(x,y)
regr.fit(x,y)




y_predict=clf.predict(x_test1)
regr.score(x,y)
y_predict
y_predict_s=regr.predict(x_test1)

y_predict_s


from sklearn.ensemble import RandomForestClassifier

rfr = RandomForestClassifier(random_state=0)

rfr.fit(x, y)

y_pred=rfr.predict(x_test1)

print(rfr.score(x, y))

y_pred
from keras.models import Sequential

from keras.layers import Dense

from tensorflow.keras.optimizers import Adam

import tensorflow as tf





model = Sequential()



model.add(Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l1(0.01)))



# The Output Layer :

model.add(Dense(1,activation='sigmoid',kernel_regularizer='l1'))



model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



model.fit(x_train,y_train,batch_size = 16,epochs=200,validation_data=(x_test,y_test),use_multiprocessing=True,shuffle = False)
x=model.predict(x_test1)

for i,j in enumerate(x):

    if j < 0.5 :

        x[i] = 0

    else :

        x[i] = 1 

predict_nn=x.flatten().astype('int')
x_PassengerId=np.array(data_test['PassengerId'])

dict = {'PassengerId': x_PassengerId, 'Survived': predict_nn }  

     

df = pd.DataFrame(dict) 

  

# saving the dataframe 

df.to_csv('servived_r_15.csv',index=False) 