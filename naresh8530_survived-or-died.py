# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import train and test to play with it

titanic_data_train = pd.read_csv('../input/train.csv')

titanic_data_test = pd.read_csv('../input/test.csv')
titanic_data_train.head()
titanic_data_test.head()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re
# Add the first bar plot which represents the count of people who survived vs not survived.

f,ax=plt.subplots(1,2,figsize=(15,5))

ax[0]=titanic_data_train.Survived.value_counts().plot(kind='bar', alpha=0.8,ax=ax[0])

ax[0].set_title("Distribution of Survival, (0=Died,1 = Survived)")

ax[1]=titanic_data_train.Sex.value_counts().plot.pie(autopct='%1.1f%%',shadow=True,ax=ax[1])

ax[1].set_title('Male Vs Female')
titanic_data_train.Pclass.value_counts().plot(kind="barh", alpha=0.8)

plt.title("Class Distribution")

plt.grid(b=True, which='major', axis='y')

plt.ylabel('Pclass')
f,ax=plt.subplots(1,2,figsize=(15,8))



pd.crosstab(titanic_data_train.Pclass,titanic_data_train.Survived).plot.bar(ax=ax[0],stacked = False,)

ax[0].set_title('Pclass:Survived vs Dead')



sns.countplot('Survived',hue='Pclass',data=titanic_data_train,ax=ax[1])

ax[1].set_title('Survived vs Pclass')
#Kernel Density Plot

titanic_data_train.Age[titanic_data_train.Pclass == 1].plot(kind='kde')

titanic_data_train.Age[titanic_data_train.Pclass == 2].plot(kind='kde')

titanic_data_train.Age[titanic_data_train.Pclass == 3].plot(kind='kde')

# Adding x label (age) to the plot

plt.xlabel("Age")

plt.title("Age Distribution within classes")

plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best')
titanic_data_train.Embarked.value_counts().plot(kind='bar', alpha=0.8)

plt.xlim(-1, len(titanic_data_train.Embarked.value_counts()))

plt.title("Passengers per boarding location")
sns.factorplot('Pclass','Survived',hue='Sex',data=titanic_data_train)
#Data Wrangling

titanic_data_train.Embarked[titanic_data_train.Embarked.isnull()] = titanic_data_train.Embarked.dropna().mode().values

from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

titanic_data_train['Embarked_encode']=label.fit_transform(titanic_data_train['Embarked'])
# handling the missing values by replacing it with the median fare value

titanic_data_train['Fare'][np.isnan(titanic_data_train['Fare'])] = titanic_data_train.Fare.dropna().median()
titanic_data_train['Title']=titanic_data_train.Name.str.extract('([A-Za-z]+)\.')

    # handling the low occuring titles

titanic_data_train['Title'][titanic_data_train.Title == 'Jonkheer'] = 'Master'

titanic_data_train['Title'][titanic_data_train.Title.isin(['Ms', 'Mlle'])] = 'Miss'

titanic_data_train['Title'][titanic_data_train.Title == 'Mme'] = 'Mrs'

titanic_data_train['Title'][titanic_data_train.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'

titanic_data_train['Title'][titanic_data_train.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'



titanic_data_train['Title_id']=LabelEncoder().fit_transform(titanic_data_train.Title)

titanic_data_test['Title']=titanic_data_test['Name'].str.extract('([A-Za-z]+)\.')

titanic_data_test['Title'][titanic_data_test.Title == 'Jonkheer'] = 'Master'

titanic_data_test['Title'][titanic_data_test.Title.isin(['Ms', 'Mlle'])] = 'Miss'

titanic_data_test['Title'][titanic_data_test.Title == 'Mme'] = 'Mrs'

titanic_data_test['Title'][titanic_data_test.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'

titanic_data_test['Title'][titanic_data_test.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'



titanic_data_test['Title_id']=LabelEncoder().fit_transform(titanic_data_test.Title)
#Missing Values imputaion in Embarked

titanic_data_train['Embarked'][titanic_data_train.Embarked.isnull()]=titanic_data_train.Embarked.dropna().mode().values

titanic_data_test['Embarked'][titanic_data_test.Embarked.isnull()]=titanic_data_test.Embarked.dropna().mode().values

#Label Encode

titanic_data_train['Embarked_encode']=label.fit_transform(titanic_data_train['Embarked'].values)

titanic_data_test['Embarked_encode']=label.fit_transform(titanic_data_test['Embarked'])

#process Pclass

titanic_data_train=pd.concat([titanic_data_train,pd.get_dummies(titanic_data_train['Pclass']).rename(columns=lambda y:'Pclass'+str(y))],axis=1)

titanic_data_test=pd.concat([titanic_data_test,pd.get_dummies(titanic_data_test['Pclass']).rename(columns=lambda y:'Pclass'+str(y))],axis=1)
#Process Cabin



#replacing missing values with Standard cabin value as 'U0'

titanic_data_train['Cabin']=titanic_data_train.Cabin.replace(np.NaN,'U0')

titanic_data_test['Cabin']=titanic_data_test.Cabin.replace(np.NaN,'U0')



#Text Mining

#Process Cabin data

 

import re

def get_cabin_letter(cabin_value):

    # searching for the letters in the cabin alphanumerical value

    letter_match = re.compile("([a-zA-Z]+)").search(cabin_value)



    if letter_match:

        return letter_match.group()

    else:

        return 'U'
titanic_data_train['Cabin_letter']=titanic_data_train['Cabin'].map(lambda y:get_cabin_letter(y))

titanic_data_test['Cabin_letter']=titanic_data_test['Cabin'].map(lambda y:get_cabin_letter(y))
#Label encoding in Cabin Letter

titanic_data_train['Cabin_letter']=label.fit_transform(titanic_data_train['Cabin_letter'])

titanic_data_test['Cabin_letter']=label.fit_transform(titanic_data_test['Cabin_letter'])
#Getting Cabin Number :

def cabin_number(cabin):

    get_number=re.compile('([0-9]+)').search(cabin)

    if get_number:

        return get_number.group()

    else:

        return 0

titanic_data_train['Cabin_num']=titanic_data_train['Cabin'].map(lambda i:cabin_number(i)).astype(int)

titanic_data_test['Cabin_num']=titanic_data_test['Cabin'].map(lambda i:cabin_number(i)).astype(int)
#Process ticket

def get_ticket_prefix(ticket):

    get_letter=re.compile('([A-Za-z\.\/]+)\.').search(ticket)

    if get_letter:

        return get_letter.group()

    else:

        return'U'

titanic_data_train['Ticket_letter']=titanic_data_train['Ticket'].map(lambda x:get_ticket_prefix(x))

titanic_data_test['Ticket_letter']=titanic_data_test['Ticket'].map(lambda x:get_ticket_prefix(x))

#Getting Ticket Letter Prefix



titanic_data_train['Ticket_letter_prefix']=titanic_data_train['Ticket_letter'].map(lambda x:re.sub('[\.?\/?]',"",x))

titanic_data_test['Ticket_letter_prefix']=titanic_data_test['Ticket_letter'].map(lambda x:re.sub('[\.?\/?]',"",x))
#getting TIcket Number :

def get_ticket_num(ticket):

    ticket_num=re.compile('([\d]+$)').search(ticket)

    if ticket_num:

        return ticket_num.group()

    else :

        return 0

titanic_data_train['Ticket_number']=titanic_data_train['Ticket'].map(lambda x:get_ticket_num(x))

titanic_data_test['Ticket_number']=titanic_data_test['Ticket'].map(lambda x:get_ticket_num(x))
#Converting Ticket_number Object to Numeric 

titanic_data_train['Ticket_number']=titanic_data_train.Ticket_number.convert_objects(convert_numeric=True)
#Labeling of Object Datatypes :

titanic_data_train_obj=titanic_data_train.select_dtypes(['object'])

titanic_data_test_obj=titanic_data_test.select_dtypes(['object'])

def label_encode(columns):

    titanic_data_train[columns]=label.fit_transform(titanic_data_train[columns])

for i in range(0,len(titanic_data_train_obj.columns)):

    label_encode(titanic_data_train_obj.columns[i])
def label_encode(columns):

    titanic_data_test[columns]=label.fit_transform(titanic_data_test[columns])

for i in range(0,len(titanic_data_test_obj.columns)):

    label_encode(titanic_data_test_obj.columns[i])
#Imputation of Missing Age Data by Random Forest Regressor

age_data=titanic_data_train[['Age','Pclass','Sex','Parch','Fare', 'Cabin_num','Cabin_letter','Embarked_encode', 'Title_id', 'Cabin_num', 'Ticket_number','Title_id']]

age_data.info()
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=200,n_jobs=-1)



#Missing Age Processing for Train data set



input_data=age_data.loc[(titanic_data_train.Age.notnull())].values[:,1::]

out_data=age_data.loc[(titanic_data_train.Age.notnull())].values[:,0]

rf.fit(input_data,out_data)

predict_age=rf.predict(age_data.loc[titanic_data_train.Age.isnull()].values[:,1::])

titanic_data_train.loc[(titanic_data_train.Age.isnull()),'Age']=predict_age
#Replacing missing Fare value with the mean of total value

titanic_data_test[titanic_data_test.Fare.isna()]=np.mean(titanic_data_test.Fare.notnull())
#Missing Age Processing for Test data set

age_data_test=titanic_data_test[['Age','Pclass','Sex','Parch','Fare', 'Cabin_num','Cabin_letter','Embarked_encode', 'Title_id', 'Cabin_num', 'Ticket_number','Title_id']]

input_test=age_data_test.loc[(titanic_data_test.Age.notna())].values[:,1::]

out_test=age_data_test.loc[titanic_data_test.Age.notna()].values[:,0]

rf.fit(input_test,out_test)
predict_age_test=rf.predict(age_data_test.loc[titanic_data_test.Age.isnull()].values[:,1::])

titanic_data_test.loc[(titanic_data_test.Age.isnull()),'Age']=predict_age_test
# Drop cloumns which are not relevant 

drop_columns_train=['PassengerId','Survived','Name','Pclass','Cabin', 'Embarked', 'Title','Ticket','Ticket_letter']

drop_columns_test=['PassengerId','Name','Pclass','Cabin', 'Embarked', 'Title','Ticket','Ticket_letter']
x_train=titanic_data_train.drop(drop_columns_train,axis=1).values

x_test=titanic_data_test.drop(drop_columns_test,axis=1).values

y_train=titanic_data_train['Survived'].values.reshape(-1,1)
titanic_data_train.columns
titanic_data_test.columns
x_train=titanic_data_train.drop(drop_columns_train,axis=1)

x_test=titanic_data_test.drop(drop_columns_test,axis=1)
#feature Scaling

from sklearn.preprocessing import StandardScaler

scale=StandardScaler()

x_train_scale=scale.fit_transform(x_train)

x_test_scale=scale.fit_transform(x_test)
#Import Required packages

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import perceptron

from xgboost import XGBClassifier

from sklearn.naive_bayes import  GaussianNB

from sklearn.model_selection import KFold,cross_val_score



import warnings

warnings.filterwarnings("ignore")
seed=12

models=[]

models.append(('Decisson Tree',DecisionTreeClassifier(random_state=seed)))

models.append(('Random Forest',RandomForestClassifier(random_state=seed)))

models.append(('Logistic Regression',LogisticRegression(random_state=seed)))

models.append(('SVM',SVC(random_state=seed)))

models.append(('KNN',KNeighborsClassifier()))

models.append(('Perceptron',perceptron.Perceptron()))

models.append(('Adaboost',AdaBoostClassifier(random_state=seed)))

models.append(('Naive Bayes',GaussianNB()))

models.append(('XGBoost',XGBClassifier()))



#Evaluate 

results=[]

names=[]

scorring='accuracy'

for name,model in models:

    kfold=KFold(n_splits=10,random_state=seed)

    cv_results=cross_val_score(model,x_train_scale,y_train,cv=kfold,scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    result="%s: %f (%f)"% (name,cv_results.mean(),cv_results.std())

    print(result)
fig = plt.figure(figsize=(15,8))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
model_svm=SVC(random_state=seed)

model_svm.fit(x_train_scale,y_train)
y_pre=model_svm.predict(x_test_scale)
submission=pd.read_csv('../input/gender_submission.csv')

submission=pd.DataFrame({



        "PassengerId": gender_submission["PassengerId"],

        "Survived":y_pre})
submission.to_csv('titanic.csv', index=False)