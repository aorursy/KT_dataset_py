# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#data visualization



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#for reproducible results



random_state=42



#for ignoring warnings while executing the code cell



import warnings

warnings.filterwarnings(action='ignore')
#loading the data



train=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')
train.head()
test.head()
#checking the submission format

gender=pd.read_csv('../input/titanic/gender_submission.csv')

gender
train.shape
#Features in training set including the target variable- Survived

train.columns
train.info()
#Separating the target variable from predictors



X=train.drop('Survived',axis=1)

y=train['Survived']
X.shape,y.shape
#making our training set and validation set

from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=random_state)
#a={col:[train[col].isnull().sum(),len(train)-train[col].isnull().sum()] for col in train.columns if train[col].isnull().sum()>0}

plt.figure(figsize=(14,5))

sns.set_style('dark')

f=[col for col in train.columns if train[col].isnull().any()]

for i in range(3):

    plt.subplot(1,3,i+1)

    sns.countplot(x=f[i],data=train.isnull())

    plt.xticks(np.arange(2),['Non-Null','Null'],rotation=20)

plt.tight_layout()
#Lets visualize the distribution of some continuous numerical variable(float64) to check their skewness

plt.figure(figsize=(14,5))

col=['Age','Fare']

for i in range(len(col)):

    plt.subplot(1,2,i+1)

    sns.distplot(a=train[col[i]],kde_kws={'color':'green'},color='y')

plt.tight_layout()

# sns.kdeplot(data=train[col[0]],shade=True)

# sns.kdeplot(data=train[col[1]],shade=True)
col1=['Pclass','SibSp','Parch']

sns.catplot(x=col1[0],data=train,hue='Sex',kind='count',col='Survived')

sns.catplot(x=col1[1],data=train,hue='Sex',kind='count',col='Survived')

sns.catplot(x=col1[2],data=train,hue='Sex',kind='count',col='Survived')
#This also verifies our observation of female passengers lives getting more priority.

tab=pd.pivot_table(train,index='Sex',values='Survived',aggfunc=np.sum)

df=pd.DataFrame({'% of survived':[tab.loc['female','Survived']/np.sum(tab)['Survived']*100,tab.loc['male','Survived']/np.sum(tab)['Survived']*100]})

df.index=tab.index

tab=pd.concat([tab,df],axis=1)

tab
#This plot is not as significant but if we carefully see the high fare (around 500) their survival percentage is 100% which clearly indicates that upper class is being given the priority.

sns.swarmplot(x=train['Survived'],y=train['Fare'])
categorical={col:train[col].nunique() for col in train.columns if str(train[col].dtype)=='object'}

plt.title('Cardinality of Different Categorical Features')

df1=pd.DataFrame({'Categorical Features':list(categorical.keys()),'Cardinality':list(categorical.values())})

sns.barplot(x=df1['Categorical Features'],y=df1['Cardinality'],palette='Blues_d',order=sorted(categorical,key=categorical.get))
#dropping Name,Ticket,Cabin columns as discussed above.

x_train.drop(columns=['Name','Ticket','Cabin'],inplace=True,axis=1)

x_val.drop(columns=['Name','Ticket','Cabin'],inplace=True,axis=1)
#imputation for Embarked Column,since it is categorical we are treating it individually from rest of the column

from sklearn.impute import SimpleImputer



impute=SimpleImputer(strategy='most_frequent')



x_train_1=pd.DataFrame(impute.fit_transform(x_train[['Embarked']]))

x_val_1=pd.DataFrame(impute.transform(x_val[['Embarked']]))



# x_train.columns=cols

# x_val.columns=cols

x_train_1.columns=['Embarked']

x_val_1.columns=['Embarked']

x_train_1.index=x_train.index

x_val_1.index=x_val.index



x_train.drop('Embarked',axis=1,inplace=True)

x_val.drop('Embarked',axis=1,inplace=True)

x_train=pd.concat([x_train,x_train_1],axis=1)

x_val=pd.concat([x_val,x_val_1],axis=1)



x_train
#Age is nearly uniformly distributed so ,we will impute its value with mean

impute_1=SimpleImputer(strategy='mean')



x_train_2=pd.DataFrame(impute_1.fit_transform(x_train[['Age']]))

x_val_2=pd.DataFrame(impute_1.transform(x_val[['Age']]))





x_train_2.columns=['Age']

x_val_2.columns=['Age']

x_train_2.index=x_train.index

x_val_2.index=x_val.index



x_train.drop('Age',axis=1,inplace=True)

x_val.drop('Age',axis=1,inplace=True)

x_train=pd.concat([x_train,x_train_2],axis=1)

x_val=pd.concat([x_val,x_val_2],axis=1)



x_train
print("{} null-values left in training-set".format(x_train.isnull().any().sum()),end='\n')

print("{} null-values left in validation-set".format(x_val.isnull().any().sum()),end='\n')
#fare is left-skewed ,so we need to scale it

x_train.select_dtypes(include=['object'])
#one hot encoding the Sex and Embarked columns

from sklearn.preprocessing import OneHotEncoder

encode=OneHotEncoder(handle_unknown='ignore',sparse=False)



categorical_cols=list(x_train.select_dtypes(include=['object']).columns)

categorical_cols



x_train_oh=pd.DataFrame(encode.fit_transform(x_train[categorical_cols]))

x_val_oh=pd.DataFrame(encode.transform(x_val[categorical_cols]))



x_train_oh.index=x_train.index

x_val_oh.index=x_val.index



x_train.drop(columns=categorical_cols,axis=1,inplace=True)

x_val.drop(columns=categorical_cols,axis=1,inplace=True)



x_train=pd.concat([x_train,x_train_oh],axis=1)

x_val=pd.concat([x_val,x_val_oh],axis=1)



x_train
#Centering and Scaling happens independently on each feature,so we will pass all features in a list together which we need to transform

from sklearn.preprocessing import StandardScaler



col_scale=['Age','Fare']



scale=StandardScaler()



x_train_scale=pd.DataFrame(scale.fit_transform(x_train[col_scale]))

x_val_scale=pd.DataFrame(scale.transform(x_val[col_scale]))



x_train_scale.index=x_train.index

x_val_scale.index=x_val.index



x_train_scale.columns=col_scale

x_val_scale.columns=col_scale



x_train.drop(columns=col_scale,axis=1,inplace=True)

x_val.drop(columns=col_scale,axis=1,inplace=True)



x_train=pd.concat([x_train,x_train_scale],axis=1)

x_val=pd.concat([x_val,x_val_scale],axis=1)



x_train
#Effect of scaling the Age and Fare attribute

sns.kdeplot(data=x_train.Age,shade=True)

sns.kdeplot(data=x_train.Fare,shade=True)
#So,now we have dealt with categorical columns too.

x_train.select_dtypes(include=['object']).columns
#Test set preprocessing

test.drop(columns=['Ticket','Cabin','Name'],axis=1,inplace=True)
test_1=pd.DataFrame(impute_1.transform(test[['Age']]))



test_1.columns=['Age']

test_1.index=test.index



test.drop('Age',axis=1,inplace=True)



test=pd.concat([test,test_1],axis=1)

test







#On further examining the test set i found that Fare column is also missing some values .So,I applied imputation here too.

impute_3=SimpleImputer(strategy='most_frequent')



test_2=pd.DataFrame(impute_3.fit_transform(test[['Fare']]))



test_2.columns=['Fare']

test_2.index=test.index



test.drop('Fare',axis=1,inplace=True)



test=pd.concat([test,test_2],axis=1)

test





test_oh=pd.DataFrame(encode.transform(test[categorical_cols]))



test_oh.index=test.index



test.drop(columns=categorical_cols,axis=1,inplace=True)



test=pd.concat([test,test_oh],axis=1)

test
test_scale=pd.DataFrame(scale.transform(test[col_scale]))



test_scale.columns=col_scale

test_scale.index=test.index



test.drop(columns=col_scale,axis=1,inplace=True)



test=pd.concat([test,test_scale],axis=1)



test
#We have just tried default models without any hyperparameter tweaking,i will try hyperparameter tuning in the next version of the notebook.

from sklearn.linear_model import LogisticRegression as lr

from sklearn.linear_model import SGDClassifier as sg

from sklearn.ensemble import RandomForestClassifier as rf

from sklearn.tree import DecisionTreeClassifier as dt

from sklearn.svm import SVC

from xgboost import XGBClassifier



log_reg=lr(random_state=random_state)

sgd=sg(random_state=random_state)

random_forest=rf(random_state=random_state)

tree_model=dt(random_state=random_state)

svm_model=SVC(random_state=random_state)

xgb=XGBClassifier(random_state=random_state)



models={'Logistic Regression':log_reg,'SGD':sgd,'Random Forest':random_forest,'Decision Tree':tree_model,'SVM':svm_model,'XGBoost':xgb}



for model_name,model in models.items():

    model.fit(x_train,y_train)

    rslt=model.predict(x_val)

    print('{}:{}'.format(model_name,np.sum(rslt==y_val)/len(y_val)),end='\n')
result=random_forest.predict(test)
output=pd.DataFrame({'PassengerId':test.PassengerId,

                      'Survived':result})

output.to_csv('submission.csv',index=False)