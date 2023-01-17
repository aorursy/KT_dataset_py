import pandas as pd

import numpy as np

import datetime

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from sklearn.metrics import r2_score

import warnings

warnings.warn = False



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



print("Training set = ",train.shape)

print("Testing set = ",test.shape)

print("Sum of Missing Values (Train/Test)= ",train.isna().sum().sum(),"(",test.isna().sum().sum(),")")

print("Survival Rate (in Training Data) =",round(train.Survived.sum()/train.shape[0]*100,2),"%")
train.describe(include='all')
print("Missing Values in Training Dataset:\n",round(train.isna().sum()/train.shape[0]*100,2))

print("Missing Values in Testing Dataset:\n",round(test.isna().sum()/test.shape[0]*100,2))
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score,mean_squared_error





def get_accuracy_score(data,col):

  X_train, X_test, y_train, y_test = train_test_split(data[col], data['Age'], test_size=0.3, random_state=7)



  lr1=LinearRegression()

  lr1.fit(X_train,y_train)

  y_pred = lr1.predict(X_test)

  return mean_squared_error(y_test, y_pred)
def get_missing_age(data):

  index = np.where(data.Age.isnull())

  

  # Find Prediction based on Present Elements

  data1=data[~data.Age.isnull()]

  acc_score0 = get_accuracy_score(data1,['SibSp','Parch','Fare'])

  acc_score1 = get_accuracy_score(data1,['SibSp','Parch'])

  acc_score2 = get_accuracy_score(data1,['SibSp','Fare'])

  acc_score3 = get_accuracy_score(data1,['Parch','Fare'])

  min_score = min(acc_score0,acc_score1,acc_score2,acc_score3) # Error to be minimized



  # Fit Model by Best Feature Selection

  data2 = data[data.Age.isnull()] 

  if min_score == acc_score0:

    col = ['SibSp','Parch','Fare']  

  else:

    if min_score == acc_score1:

      col = ['SibSp','Parch']

    else:

      if min_score == acc_score2:

        col = ['SibSp','Fare']                

      else:

        col = ['Parch','Fare']        

  X_train, y_train, X_test = data1[col], data1['Age'], data2[col]    



  # Do Prediction on Absent Elements

  lr = LinearRegression()

  lr.fit(X_train,y_train)

  



  data2 = data2.drop(columns=['Age'],axis=1)

  data2['Age']=[max(0,min(100,i)) for i in lr.predict(X_test)]

  

  temp=data1['Age']

  data1 = data1.drop(columns=['Age'],axis=1)

  data1['Age']=temp



  return data1.append(data2)
def impute_fare(dataset):

  if dataset.Fare.isnull().sum()<=0:

    return dataset

  else:

    ok_set = dataset[~dataset.Fare.isnull()]

    to_set = dataset[dataset.Fare.isnull()]



    l=list()

    for i in range(to_set.shape[0]):  

      l.append(ok_set.Fare[(ok_set.Pclass==to_set.iloc[i]['Pclass']) & (ok_set.Parch==to_set.iloc[i]['Parch'])].mean())



    # Structural Re-format (Faster compared to For Loop)

    temp = ok_set['Fare']

    ok_set = ok_set.drop(columns='Fare',axis=1)

    ok_set['Fare'] = temp

    

    to_set=to_set.drop(columns='Fare',axis=1)

    to_set['Fare']=l



    return ok_set.append(to_set)
def impute_cabin(data):

  if data.Cabin.isnull().sum()<=0:

    return data

  else:

    ok_set = data[~data.Cabin.isnull()]

    to_set = data[data.Cabin.isnull()]



    #l=list()

    #for i in range(ok_set.shape[0]):

    # l.append(ok_set.iloc[i]['Cabin'][0])



    l = [ok_set.iloc[i]['Cabin'][0] for i in range(ok_set.shape[0]) ] # Cannot put in next assignment, drop used

    ok_set = ok_set.drop(columns=['Cabin'],axis=1)

    ok_set['Cabin'] = l



    to_set = to_set.drop(columns='Cabin',axis=1)

    to_set['Cabin'] = 'U'



    return ok_set.append(to_set)
def impute_embarked(data):

  data.Embarked.replace('S',1,inplace=True)

  data.Embarked.replace('C',2,inplace=True)

  data.Embarked.replace('Q',3,inplace=True)

  if data.Embarked.isnull().sum()<=0:

    return data

  else:

    ok_data = data[~data.Embarked.isnull()]

    to_data = data[data.Embarked.isnull()]



    lr = LinearRegression() # Using Regression, the mode of Embarked is captured

    lr.fit(ok_data[['Pclass','Fare','Parch']],ok_data['Embarked'])

    

    temp = ok_data['Embarked']

    ok_data = ok_data.drop(columns='Embarked',axis=1)

    ok_data['Embarked'] = temp

    

    to_data = to_data.drop(columns='Embarked',axis=1)

    temp = [int(i) for i in lr.predict(to_data[['Pclass','Fare','Parch']])]

    to_data['Embarked'] = temp



    return ok_data.append(to_data)
# Impute Missing Value 

# ~ Fare

test = impute_fare(test)



# ~ Age

train = get_missing_age(train)

test = get_missing_age(test)



# ~ Cabin

train=impute_cabin(train)

test=impute_cabin(test)



# ~ Embarked

train = impute_embarked(train)

test = impute_embarked(test)
# Age Grouping

age_group = 3

max_age = max(train.Age)

min_age = min(train.Age)



train['Age_Group'] = [int(i) for i in round((train.Age-min_age)/((max_age-min_age)/age_group),0)+1]

test['Age_Group']  = [int(i) for i in round((test.Age-min_age)/((max_age-min_age)/age_group),0)+1]

train = train.drop(columns='Age')

test = test.drop(columns='Age')
# Fare: Based on Pclass and Parch

# Depending on Passenger class and number of family members, Fare is provided. 

# Thus, Fare is more like a dependent feature (not correlated) when compared to other features.

pd.pivot_table(train, values='Fare', index=['Pclass'],columns=['Parch'], aggfunc=np.mean)
def get_feature_count(data,col_names):

  df_all=pd.DataFrame()

  for i in col_names:

    u = data[i].unique()

    temp=pd.DataFrame()

    for j in u:

      m = (data[i]==j).sum()

      temp = temp.append([[j,m]])

    temp['col_name'] = i    

    df_all = df_all.append(temp)



  df_all.columns = ['X','Y','Feature']

  return df_all
df = get_feature_count(train,['Pclass','Sex','Cabin','Embarked','Age_Group','SibSp','Parch'])



fig=px.bar(data_frame=df, x='X',y='Y',color='Y',facet_col='Feature',facet_col_wrap=7,width=1000,height=350)

fig.update_xaxes(matches=None)

fig.update_yaxes(matches=None)

fig.show()
# How people paid Fair based on Pclass

fig=px.histogram(train,x='Fare',color='Pclass',height=300)

fig.show()
# Who survided and who didn't get to?

temp_table = train

temp_table['Survive_Copy'] = 1



cols=['Pclass','Sex','SibSp','Parch','Cabin','Embarked','Age_Group']

for i in cols:

  print("\nSurvival per",i,"\n",'_'*70,"\n",

        pd.pivot_table(temp_table, values='Survive_Copy', index=['Survived'],columns=[i], aggfunc=np.sum))
# Survivor Selection 



cols=['Pclass','Sex','SibSp','Parch','Cabin','Embarked','Age_Group']

for i in cols:

  for j in cols:

    if i==j:

      # do nothing

      d=0

    else:

      print("\n",i,"vs",j,"\n",'_'*70,"\n",

            pd.pivot_table(train, values='Survived', index=[i],columns=[j], aggfunc=np.sum))
result = pd.DataFrame()



# Rearrange Columns (Before Modeling)

X_train_ID = train[['PassengerId']]

X_train = train[['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked', 'Fare', 'Age_Group']]

y_train = train[['Survived']]

X_test_ID = test[['PassengerId']]

X_test  = test[['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked', 'Fare', 'Age_Group']]

print("Original Dimension\n",X_train.shape,y_train.shape,X_test.shape)



result['PassengerId']=X_test_ID.PassengerId



# One hot Encoding for Sex, Cabin, Embarked

X_train = pd.get_dummies(X_train, columns=['Sex','Cabin','Embarked'])

X_test  = pd.get_dummies(X_test , columns=['Sex','Cabin','Embarked'])

print("Post OHE Dimension\n",X_train.shape,X_test.shape)





# Post OHE: Remove irrelevant column (remove multi-collinearity) & rename column (for consistency)

X_train = X_train.drop(columns=['Sex_female','Cabin_U','Cabin_T','Embarked_1.0'],axis=1)

X_test  = X_test.drop(columns=['Sex_female','Cabin_U','Embarked_1'],axis=1)

X_train.columns = X_test.columns

print("Post Dimension Process\n",X_train.shape,X_test.shape)



print("List of columns:\n",*X_train.columns)
from sklearn.model_selection import train_test_split



temp_train = X_train

cols=temp_train.columns

temp_train['target'] = y_train



a_train, a_test, b_train, b_test = train_test_split(temp_train[cols],temp_train['target'], test_size=0.3, random_state=7)

print("Re-Split dimenstions\n",a_train.shape,a_test.shape,b_train.shape,b_test.shape)
from sklearn.metrics import accuracy_score



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='lbfgs',max_iter=1000,tol=0.001,random_state=7)

lr.fit(a_train,b_train)

print("Accuracy on Training Set=",round(accuracy_score(b_test,lr.predict(a_test))*100,2))



result['Log_Reg'] = lr.predict(X_test)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='gini',max_features=.75,max_depth=3,random_state=7)

dt.fit(a_train,b_train)

print("Accuracy on Training Set=",round(accuracy_score(b_test,dt.predict(a_test))*100,2))



result['DT'] = dt.predict(X_test)
from sklearn.linear_model import Perceptron

p = Perceptron(max_iter=50,shuffle=True, tol=0.001,random_state=7)

p.fit(a_train,b_train)

print("Accuracy on Training Set=",round(accuracy_score(b_test,p.predict(a_test))*100,2))



result['Perceptron'] = p.predict(X_test)
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(10,5), max_iter=1000,

                    n_iter_no_change=5,learning_rate='constant',shuffle=True,

                    validation_fraction=0.1,tol=0.001, random_state=7)

mlp.fit(a_train,b_train)

print("Accuracy on Training Set=",round(accuracy_score(b_test,mlp.predict(a_test))*100,2))



result['MLP'] = mlp.predict(X_test)
print("Survival Rate from Logistic Regression    =",round(result.Log_Reg.sum()/result.shape[0]*100,2)   ,"%")

print("Survival Rate from Decision Tree          =",round(result.DT.sum()/result.shape[0]*100,2)        ,"%")

print("Survival Rate from Perceptron             =",round(result.Perceptron.sum()/result.shape[0]*100,2),"%")

print("Survival Rate from Multi-layer Perceptron =",round(result.MLP.sum()/result.shape[0]*100,2)       ,"%")
result['Survived'] = [int(result.loc[i][[1,2,4]].mode()) for i in range(result.shape[0]) ]                  

print("Survival Response (from each method)\n",result.sum()[1:])    
submission = result[['PassengerId','Survived']]

submission.to_csv("submission.csv", index=False)