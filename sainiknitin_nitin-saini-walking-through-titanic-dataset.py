# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#checking out how to submit the results

submission_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

print(submission_data.shape)

submission_data.head()
#reading the training data for this challenge

#It has 891 rows and 12 columns, 1 being the target variable i.e. Survived

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

print(train_data.shape)

train_data.head()
#checking how many null values are there in the training dataset

train_data.isnull().sum()

#out of 891 observations, we don't have Cabin details for 687
#reading the test data to confirm if all the data is similar to training data

#It has 418 rows and 11 columns. That's good as it should not have the target variable.

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

print(test_data.shape)

test_data.head()
#checking how many null values are there in the test dataset

test_data.isnull().sum()

#Even in the test data, out of  418 observations, we don't have Cabin details for 327. 

#To me, it means that I won't probably be able to use this columns in my regression analysis

#AGE, FARE and EMBARKED can be imputed based on some understanding of the data.
#Now I will start doing EDA for my training data and then whatever changes I make in the training data, 

#I will make on the Test data later, before predicting the results.



#checking the datatype of all the columns here

#data with 'object' datatype cannot be fed into the models directly, 

#hence they are required to converted to 'int64', but with careful analysis



#Survived: looks fine

#Pclass: though it already 'int64', it's an ordinal variable. We will discuss about it later. We might want to convert it into dummy variables

#Name: this is not going to be used in the modeling as it is similar to an ID variable. But this might help us impute age values

#Sex: Since it an 'object', we need to convert it into 0 and 1 i.e. 'int' to be able to use it in the modelling

#Age: This is OK

#SibSp: This is again ordinal, we will discuss later.

#Parch: Ordinal

train_data.dtypes
#Ticket: This is again going to be similar to the ID as each one of the passenger will have a unique ticket.

#Fare: This is continuous data. Hence we can keep it and let the model decide if it is useful or not.

#Cabin: As discussed, I will remove this column, since I don't have almost 80% data

#Embarked: This is a nominal variable and should be converted to 2 dummy variables as there are 3 ports.

train_data[train_data.Pclass == 1]
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

train_data.Survived.value_counts().plot.pie(autopct='%1.1f%%', textprops=dict(color="black"))

#This shows that only 38.4% people survived on Titanic
#Lets start by making scatter plots between continuous variables.

plt.scatter(train_data.Age, train_data.Fare)

#They don't seem to have any correlation between them
#Let's plot boxplots between target and continuous variables

fig, ax =plt.subplots(1,2)

sns.boxplot(x = train_data.Survived, y = train_data.Age, ax=ax[0])

sns.boxplot(x = train_data.Survived, y = train_data.Fare, ax=ax[1])

plt.tight_layout()

plt.show()

#Thought the Age vs Survived plot doesn't show much, 

#the one with Fare tells that those who paid higher fare, were more likely to be saved

#This can be confirmed when we check the cross table between Pclass and Survived
#Let's make tables to see the relations between categorical data and target

#Let's see how is Sex related with Survived

pd.crosstab(train_data.Sex, train_data.Survived, margins = True)



#This shows that out of 342 people who survived, 233 were female. This is almost the double number than the males who survived.

#Sex seem to be an important parameter for modelling and predicting the survivors
pd.crosstab(train_data.Pclass, train_data.Survived, margins=True)

#Again, Class 1 passengers have the highest rate of survival, followed by Class 2 and then Class 3. PClass too is an important parameter for modelling, but needs to be recoded for correct usage.
pd.crosstab(train_data.Embarked, train_data.Survived, margins=True)

#This shows that people who boarded from C has the highest survivors, followed by Q and then S. This may just be

#be a coincidence, but can also be important as it's able to directly help us predict the survivors to an extent

#Again, this needs to be recoded to make it useful in modelling
#Let's check SibSp

pd.crosstab(train_data.SibSp, train_data.Survived, margins=True)

#Interestingly, people with 1 or 2 siblings have higher survival rate (~50%) than those who did not have any siblings

#Higher the number of siblings, lower the number of survivors
#Let's see how these people were travelling

pd.crosstab([train_data.SibSp, train_data.Survived], train_data.Pclass, margins=True)

#This confirms the above understanding as the highest % of survived travellers with 1 sibling were in Class 1.
#Let's check the same with Parch column

pd.crosstab(train_data.Parch, train_data.Survived, margins=True)

#This is similar to SibSp columns except that someone with 0 to 3 parents+children survived than those with more.

#This makes sense to me as if you are going to be saved, you'd want to save all your relations, 

#but higher the number of dependents, lower the possibility to save them
#Let's check the same based on Pclass

pd.crosstab([train_data.Parch, train_data.Survived], train_data.Pclass, margins=True)

#Not particularly useful
#Let's start worrying about the missing data

#We have Age, Embarked and Cabin with missing values

#I will eliminate Cabin as I don't think it's a good variable.

#There's only 2 values missing in that column, I can either remove those rows or impute them based on some understanding

#There are a lot of values missing in Age. I will impute those values based on some understanding.

#Let's go one by one



#Let's think about imputing Age.

#I'll make a histogram of that to see the distribution

plt.hist(train_data.Age, bins = 30)

#This shows that there's high number of people from 18 to 35 years of age. 

#Also, there are good number of children from 0 to 5 years of age.
#Let's see the descriptive statistics of Age

train_data.Age.describe()

#You can see that 67% of the people lie within 16 to 43 years age, which makes sense with the histogram above.
#Now let's see how Age is affecting Survival based Sex and Pclass

fig,axes=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Pclass","Age", hue="Survived", data=train_data, split=True, ax=axes[0])

axes[0].set_title('Pclass and Age vs Survived')

axes[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=train_data, split=True, ax=axes[1])

axes[1].set_title('Sex and Age vs Survived')

axes[1].set_yticks(range(0,110,10))

plt.show()



#Observations

#1. Survival rate of children, below the age of 10 is quite high irrespective of what Pclass or Sex

#2. The men with more Age did not survive

#3. Survival rate of people in 20 to 50 age group is higher than others



# This makes it clear that if we can impute age better than the mean of the whole dataset, we'd have higher efficiency.
#Let's see if we can use name column to understand the age of the people

train_data['Sal'] = None

for i in train_data:

    train_data['Sal']=train_data.Name.str.extract('([A-Za-z]+)\.')

train_data.Sal.unique()
#Let's check these Salutations based on Sex

pd.crosstab(train_data.Sal,train_data.Sex).T.style.background_gradient(cmap='Accent_r')

#This is fine as all the Salutations that should be female are female and same with the males.

#Now we need to reduce them to a few categories to be able to impute the ages
#Let's reduce the categories

#######Please uncomment the below line when running this code for the first time

train_data.Sal.replace(train_data.Sal.unique(),['Mr','Mrs','Miss','Master','Mr','Others','Mr','Miss','Miss','Mr','Mrs','Mr','Miss','Mr','Mr','Miss','Others'],inplace=True)

train_data.Sal.unique()
#Let's find out the mean age in each Salutation group

train_data.groupby(train_data.Sal).Age.mean()
#This should give better imputation values to us

#Let's impute the values

#train_data[train_data.Age.isnull() & train_data.Sal == 'Master'].Age = train_data[train_data.Sal == 'Master'].Age.mean()

for sal in train_data.Sal:

    train_data.loc[(train_data.Age.isnull()) & (train_data.Sal == sal), 'Age'] = train_data[train_data.Sal == sal].Age.mean()

#The for loop above replaced all the missing age values with the mean of age in that Sal category
#As shown below, there's no missing value in the Age column now.

train_data.Age.isnull().sum()
train_data.Embarked.value_counts()
#Let's see how to impute Embarked

train_data.Embarked.value_counts().plot.pie(autopct='%1.1f%%', textprops=dict(color="black"))
#Let's also check what are two rows where embarked is missing

train_data[train_data.Embarked.isnull()]

#Now this puts me in confusion since they have the same ticket number. I was assuming that each person will have unique ticket number
#by checking the unique counts of Ticket, I found there's only 681 unique tickets

print('Unique Tickets: ', train_data.Ticket.nunique())

print('Total Passengers: ', train_data.PassengerId.nunique())
train_data.Ticket.value_counts()
#Let's also see how many survived from each Port

pd.crosstab(train_data.Embarked, train_data.Survived, margins = True)

# I can see that people who boarded from Port C had the highest survivors. So I'll impute the values with C and both of them Survived.
train_data.loc[(train_data.Embarked.isnull()), 'Embarked'] = 'C'

#Rechecking if the values are imputed or not

train_data.Embarked.isnull().any()
#Now let's check again, the datatype of each column before we start working on NOMINAL and ORDINAL columns

train_data.dtypes



#Colums not to be used in modelling

#PASSENGERID, NAME, TICKET, CABIN, SAL



#Columns to be recoded as dummy variables

#SEX, CLASS, EMBARKED
#Let's start by recoding SEX

if str(train_data.Sex.dtypes) != 'int64':

    train_data.Sex.replace(['male','female'],[0,1],inplace=True)

train_data.head(2)

#you can see that the Sex is changed to a binary variable
#Let's convert Pclass into dummy variables

Pclass_dummies = pd.get_dummies(train_data.Pclass, prefix='Pclass', prefix_sep='_', drop_first=True)
train_data = train_data.merge(Pclass_dummies, left_index=True, right_index=True)

train_data.head()

#Now you have 3 - 1 = 2 Pclass columns as dummy variables in the data
#Let's convert Embarked into dummy variables

Embarked_dummies = pd.get_dummies(train_data.Embarked, prefix='Embarked', prefix_sep='_', drop_first=True)

train_data = train_data.merge(Embarked_dummies, left_index=True, right_index=True)

train_data.head()

#Now you have 3 - 1 = 2 Embarked columns as dummy variables in the data
#Let's drop the columns we don't need for modelling anymore, except PassengerId for now

if set(['Pclass','Name','Ticket','Cabin','Embarked','Sal']).issubset(train_data.columns):

    train_data = train_data.drop(['Pclass','Name','Ticket','Cabin','Embarked','Sal'], axis=1)

train_data.head()
train_data.dtypes

#You can see that all our columns are of number datatype. Hence, we are now ready for modelling
#First thing is to split the data into training and testing, to make sure we don't overfit the model

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix

train, test = train_test_split(train_data, test_size=0.25, random_state=0, stratify = train_data.Survived)

#stratify here keeps the proportion of Survived same in both train and test sample

#Let's divide the data in X (independent) and y(dependent) variables

X = train_data.drop('Survived', axis = 1)

y = train_data.Survived

train_X = train.drop('Survived', axis = 1)

train_y = train.Survived

test_X = test.drop('Survived', axis = 1)

test_y = test.Survived

print('Shape of training data: ', train_X.shape, train_y.shape)

print('Shape of test data: ', test_X.shape, test_y.shape)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process



models = []

models.append(('LR', LogisticRegression(max_iter=1000000)))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier(n_neighbors=20)))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('XGB', XGBClassifier()))

models.append(('ADA', ensemble.AdaBoostClassifier()))

models.append(('BAG', ensemble.BaggingClassifier()))

models.append(('TRE', ensemble.ExtraTreesClassifier()))

models.append(('GRA', ensemble.GradientBoostingClassifier()))

models.append(('RAN', ensemble.RandomForestClassifier()))

models.append(('PAS', linear_model.PassiveAggressiveClassifier()))

models.append(('RID', linear_model.RidgeClassifierCV()))

models.append(('SDG', linear_model.SGDClassifier()))

models.append(('PER', linear_model.Perceptron()))

#Navies Bayes

models.append(('BER', naive_bayes.BernoulliNB()))

models.append(('GAU', naive_bayes.GaussianNB())) 

#SVM

models.append(('NUS', svm.NuSVC(probability=True)))

models.append(('LIN', svm.LinearSVC()))

#Trees    

models.append(('EXT', tree.ExtraTreeClassifier()))

models.append(('QDA', discriminant_analysis.QuadraticDiscriminantAnalysis()))



seed = 3

results = []

names = []



# store preds

from sklearn.model_selection import cross_val_predict

dwPreds = []

for name, model in models:

    kfold = KFold(n_splits=13, random_state=seed, shuffle=True)

    # store the metrics

    cv_results = cross_val_score(model, train_X, train_y, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %s= %f %s= (%f)" % (name, 'Mean', cv_results.mean(), 'Std', cv_results.std())

    print(msg)
from matplotlib import pyplot

fig = pyplot.figure(figsize=(20,5))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

model = LogisticRegression()

model.fit(train_X, train_y)

predicted_y = model.predict(test_X)

print(accuracy_score(test_y, predicted_y))

print(confusion_matrix(test_y, predicted_y))

print(classification_report(test_y, predicted_y))
model = ensemble.GradientBoostingClassifier(learning_rate=0.1, n_estimators=120, max_depth=5)

model.fit(train_X, train_y)



predicted_train_y = model.predict(train_X)

print(accuracy_score(train_y, predicted_train_y))

print(confusion_matrix(train_y, predicted_train_y))

print(classification_report(train_y, predicted_train_y))



predicted_test_y = model.predict(test_X)

print(accuracy_score(test_y, predicted_test_y))

print(confusion_matrix(test_y, predicted_test_y))

print(classification_report(test_y, predicted_test_y))
#Let's prepare the submission dataset for pedicitons and then run Logistic Regression on it

test_data.head()
print('Shape of test data: ', test_data.shape)

test_data.isnull().sum()
#Here we have missing values in AGE, FARE and CABIN

#Let's impute the AGE values as we did before

test_data['Sal'] = None

for i in test_data:

    test_data['Sal']=test_data.Name.str.extract('([A-Za-z]+)\.')

test_data.Sal.unique()
pd.crosstab(test_data.Sal,train_data.Sex).T.style.background_gradient(cmap='Accent_r')
test_data.Sal.replace(test_data.Sal.unique(),['Mr','Mrs','Miss','Master','Miss','Mr','Mr','Mr','Others'],inplace=True)

test_data.Sal.unique()
test_data.groupby(test_data.Sal).Age.mean()
for sal in test_data.Sal:

    test_data.loc[(test_data.Age.isnull()) & (test_data.Sal == sal), 'Age'] = test_data[test_data.Sal == sal].Age.mean()
test_data.Age.isnull().any()
#Let's just inpute the missing fare value with the mean of the total fare

test_data.loc[(test_data.Fare.isnull()), 'Fare'] = test_data.Fare.mean()
test_data.Fare.isnull().any()
#Let's create dummy variables for SEX, PCLASS and EMBARKED

if str(test_data.Sex.dtypes) != 'int64':

    test_data.Sex.replace(['male','female'],[0,1],inplace=True)

test_data.head(2)
Pclass_dummies = pd.get_dummies(test_data.Pclass, prefix='Pclass', prefix_sep='_', drop_first=True)

test_data = test_data.merge(Pclass_dummies, left_index=True, right_index=True)

test_data.head(2)
Embarked_dummies = pd.get_dummies(test_data.Embarked, prefix='Embarked', prefix_sep='_', drop_first=True)

test_data = test_data.merge(Embarked_dummies, left_index=True, right_index=True)

test_data.head()
if set(['Pclass','Name','Ticket','Cabin','Embarked','Sal']).issubset(test_data.columns):

    test_data = test_data.drop(['Pclass','Name','Ticket','Cabin','Embarked','Sal'], axis=1)

test_data.head()
print('Shape of test data: ', test_data.shape)

test_data.dtypes
#Let's predict on the Test Dataset based on Logistic Regression model

predicted_test_y = model.predict(test_data)

predicted_test_y = pd.DataFrame(predicted_test_y)

print('Shape of predicted y: ', predicted_test_y.shape)

predicted_test_y.head()
submission = test_data.merge(predicted_test_y, left_index=True, right_index=True)

if set(['Sex','Age','SibSp','Parch','Fare','Pclass_2','Pclass_3','Embarked_Q','Embarked_S']).issubset(submission.columns):

    submission = submission.drop(['Sex','Age','SibSp','Parch','Fare','Pclass_2','Pclass_3','Embarked_Q','Embarked_S'], axis=1)

submission = submission.rename(columns={0: 'Survived'})

submission.head()
submission.Survived.value_counts()
from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Submission", filename = "submission.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe

create_download_link(submission)
