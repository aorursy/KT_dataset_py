# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier

sns.set_style('whitegrid')



%matplotlib inline
###Loading the data

titanic_df = pd.read_csv('../input/train.csv')

titanic_df.head()
titanic_df.info()
###Who were the passengers on the titanic? (What age, gender, class etc)



###Gender Plot

sns.factorplot('Sex',data=titanic_df,kind='count')



### Shows more male passengers than female 
### Class plot

sns.factorplot('Pclass',data=titanic_df,kind='count')
###Interesting! More passengers are from class Three. Now lets find the gender ration among the classes



sns.factorplot('Pclass',data=titanic_df,hue='Sex',kind='count')

##This gives us an insight that there are quite a few males than females in 3rd class. Now lets dig deeper and find the children among the passengers.



def titanic_children(passenger):

    

    age , sex = passenger

    if age <16:

        return 'child'

    else:

        return sex



titanic_df['person'] = titanic_df[['Age','Sex']].apply(titanic_children,axis=1)

        
titanic_df.head(10)
### Plotting a graph to check the ratio of male,female and children in each category of class



sns.factorplot('Pclass',data=titanic_df,hue='person',kind='count')
###Now let us look at the ages of the passengers



titanic_df['Age'].hist(bins=70)
as_fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=5)



as_fig.map(sns.kdeplot,'Age',shade=True)



oldest = titanic_df['Age'].max()



as_fig.set(xlim=(0,oldest))



as_fig.add_legend()
as_fig = sns.FacetGrid(titanic_df,hue='person',aspect=5)



as_fig.map(sns.kdeplot,'Age',shade=True)



oldest = titanic_df['Age'].max()



as_fig.set(xlim=(0,oldest))



as_fig.add_legend()
as_fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=5)



as_fig.map(sns.kdeplot,'Age',shade=True)



oldest = titanic_df['Age'].max()



as_fig.set(xlim=(0,oldest))



as_fig.add_legend()
###Mean age of the passengers

titanic_df['Age'].mean()
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
#### Drop the Cabin column as there are many null values and it does not help in making prediction



titanic_df.drop('Cabin',axis=1,inplace=True)
## Filling the null values in the Embarked column with S as there are more number of passengers boarded from Southhampton

titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')



## To check if there are still any null values in the dataset

titanic_df.isnull().values.any()
sns.factorplot('Embarked',data=titanic_df,kind='count')
sns.factorplot('Embarked',data=titanic_df,hue='Pclass',kind='count')
## Let's check who are with family and who are alone

## This can be found by adding Parch and Sibsp columns

titanic_df['Alone'] = titanic_df.Parch + titanic_df.SibSp

## if Alone value is >0 then they are with family else they are Alone



titanic_df['Alone'].loc[titanic_df['Alone']>0] = 'With Family'

titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Without Family'

#Let us visualise the Alone column



sns.factorplot('Alone',kind='count',data=titanic_df)
# let us see who are alone according to class

sns.factorplot('Alone',kind='count',data=titanic_df,hue='Pclass')
sns.factorplot('Survived',data=titanic_df,kind='count')
## checking of the class had any effect in the survival rate

sns.factorplot('Survived',data=titanic_df,kind='count',hue='Pclass')
sns.factorplot('Pclass','Survived',data=titanic_df,hue='person')
sns.factorplot('Pclass','Survived',data=titanic_df,hue='Alone')
sns.lmplot('Age','Survived',data=titanic_df)




sns.lmplot('Age','Survived',data=titanic_df,hue='Pclass')
sns.lmplot('Age','Survived',data=titanic_df,hue='Sex')
sns.lmplot('Age','Survived',data=titanic_df,hue='Alone')
sns.lmplot('Age','Survived',data=titanic_df,hue='Embarked')
person_dummies = pd.get_dummies(titanic_df['person'])

alone_dummies = pd.get_dummies(titanic_df['Alone'])



embarked_dummies = pd.get_dummies(titanic_df['Embarked'])



embarked_dummies.drop('Q',axis=1,inplace=True)
pclass_dummies = pd.get_dummies(titanic_df['Pclass'])



pclass_dummies.columns=['class_1','class_2','class_3']
import math



titanic_df['Age'] = titanic_df['Age'].apply(math.ceil)

titanic_df['Fare'] = titanic_df['Fare'].apply(math.ceil)
titanic_df = pd.concat([titanic_df,pclass_dummies,person_dummies,alone_dummies,embarked_dummies],axis=1)
titanic_df.drop(['PassengerId','Name','Sex','SibSp','Parch','Ticket','Embarked'],axis=1,inplace=True)

titanic_df.drop(['Alone','person','Pclass','Without Family','male','class_3'],axis=1,inplace=True)
titanic_df.head()
titanic_train = titanic_df.drop('Survived',axis=1)

titanic_survived = titanic_df.Survived

x_train, x_test, y_train, y_test = train_test_split(titanic_train,titanic_survived,test_size=0.2)
x_train.head()
x_train.head()
x_test.head()
log_model = LogisticRegression()



log_model.fit(x_train,y_train)



train_survival = log_model.predict(x_test)

print("Accuracy Score of logistic model is",metrics.accuracy_score(y_true=y_test,y_pred=train_survival))

corr_coeff = list(zip(x_train.columns,np.transpose(log_model.coef_)))
print('Correlation coefficients are ',corr_coeff)
rand_model = RandomForestClassifier()

rand_model.fit(x_train,y_train)



rand_predict = rand_model.predict(x_test)

#rand_model.score(y_test,rand_predict)
print("Accuracy Score of Random Forest model is",metrics.accuracy_score(y_true=y_test,y_pred=rand_predict))
## Null error rate



y_train.mean()



## The accuarcy is greater than the 1-y_train.mean() = x < accuracy which means the model is not just guessing the output.

## Laoding the test data

titanic_df_test = pd.read_csv('../input/test.csv')
titanic_df_test.head()
## Storing the PassengerId column for the submission purpose

passenger_id = titanic_df_test.PassengerId
embarked_test_dummies = pd.get_dummies(titanic_df_test['Embarked'])



embarked_test_dummies.drop('Q',axis=1,inplace=True)
titanic_df_test['Alone'] = titanic_df_test.SibSp + titanic_df_test.Parch



titanic_df_test['Alone'].loc[titanic_df_test['Alone']>0] = 'With Family'

titanic_df_test['Alone'].loc[titanic_df_test['Alone'] == 0] = 'Without Family'



#titanic_df_test.head()
alone_test_dummies = pd.get_dummies(titanic_df_test['Alone'])



pclass_test_dummies = pd.get_dummies(titanic_df_test['Pclass'])



pclass_test_dummies.columns = ['class_1','class_2','class_3']



titanic_df_test['person'] = titanic_df_test[['Age','Sex']].apply(titanic_children,axis=1)



person_test_dummies = pd.get_dummies(titanic_df_test['person'])
titanic_df_test = pd.concat([titanic_df_test,embarked_test_dummies,alone_test_dummies,person_test_dummies,pclass_test_dummies],axis=1)
titanic_df_test.drop(['PassengerId','Name','Sex','SibSp','Parch','Ticket','Cabin','Embarked','Alone','person','Pclass','Without Family','male','class_3'],axis=1,inplace=True)
titanic_df_test.head()
titanic_df_test['Age'] = titanic_df_test['Age'].fillna(titanic_df_test['Age'].mean())
titanic_df_test['Fare'] = titanic_df_test['Fare'].fillna(titanic_df_test['Fare'].mean())
titanic_df_test['Age'] = titanic_df_test['Age'].apply(math.ceil)

titanic_df_test['Fare'] = titanic_df_test['Fare'].apply(math.ceil)
survival_prediction = log_model.predict(titanic_df_test)

rand_survival_predictions = rand_model.predict(titanic_df_test)
Final_predictions = DataFrame({'passenger_id':passenger_id,'survived':survival_prediction})



Final_predictions.to_csv('sample_submission.csv',index=False)
Final_predictions.head()
### Let's see if our intuitions were correctly predicted by the model



check_model = pd.read_csv('../input/test.csv')
check_model['survived'] = rand_survival_predictions

check_model
check_model['Age'] = check_model['Age'].fillna(check_model['Age'].mean())
sns.factorplot('survived',data=check_model,kind='count',hue='Pclass')
sns.factorplot('survived',data=check_model,kind='count',hue='Sex')