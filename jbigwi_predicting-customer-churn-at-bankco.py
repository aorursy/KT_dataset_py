# Import libraries

import pandas as pd

import numpy as np



#Data visualisaton and images

import matplotlib.pyplot as plt

#reading data of churn for bank customers dataset

data = pd.read_csv('../input/predicting-churn-for-bank-customers/Churn_Modelling.csv')
#verifying if there was Nan data in the given dataset

data.info()

#Droping the Row number as it was the index of dataset in Excel version

data = data.drop('RowNumber', axis = 1)

data.head()
'''This function is for helping us for better visualization of our dataset '''

def bar_chart(feature):

    exited = data[data['Exited']==1][feature].value_counts()

    stayed = data[data['Exited']==0][feature].value_counts()

    df = pd.DataFrame([exited,stayed])

    df.index = ['Exited','Stayed']

    df.plot(kind = 'bar', stacked = True, figsize=(10,5))
bar_chart('Exited')
# Now we have the data in dataframe, we can begin an advanced analysis

#of data. Lets examine the overall chance to Exited from the bank



data['Exited'].mean()
bar_chart('NumOfProducts')
nberOfProcucts_grouping = data.groupby('NumOfProducts').mean()

nberOfProcucts_grouping
nberOfProcucts_grouping['Exited'].plot.bar()
bar_chart('Gender')
'''We can continue to extend the statIcal breakdown by 

using the grouping function for both number of products  and Gender'''



nberOfProcucts_genger_grouping = data.groupby(['NumOfProducts', 'Gender']).mean()

nberOfProcucts_genger_grouping
nberOfProcucts_genger_grouping['Exited'].plot.bar()
'''Function of categorizing the age of customers'''

data_age = data

def age_cat(x):

    if x <= 20:

        return 0

    elif x > 20 and x <=30:

        return 1

    elif x >30 and x <= 40:

        return 2

    elif x >40 and x <= 50:

        return 3

    elif x > 50 and x <= 60:

        return 4

    elif x >60 and x <= 70:

        return 5

    else:

        return 6

data_age['Age_categ'] = data_age['Age'].apply(age_cat)

data_age.head(10)
bar_chart('Age_categ')
#Grouping the cutomers' age and make barchart for the Exited customers

group_by_age = pd.cut(data["Age"], np.arange(0,100,10))

age_grouping = data.groupby(group_by_age).mean()

age_grouping['Exited'].plot.bar()
#Bar chart for Geograhical countries location of the customers 

bar_chart('Geography')
nberOfProcucts_Geo_grouping = data.groupby(['NumOfProducts', 'Geography']).mean()

nberOfProcucts_Geo_grouping
nberOfProcucts_Geo_grouping['Exited'].plot.bar()
data_credit = data

def credit_cat(x):

    if x <= 350:

        return 0

    elif x > 350 and x <=450:

        return 1

    elif x >450 and x <= 550:

        return 2

    elif x >550 and x <= 650:

        return 3

    elif x > 650 and x <= 750:

        return 4

    else:

        return 5

data_credit['credit_categ'] = data_credit['CreditScore'].apply(credit_cat)

data_credit.head(10)
bar_chart('credit_categ')
Credit_grouping = data_credit.groupby(['credit_categ']).mean()

Credit_grouping['Exited'].plot.bar()
bar_chart('HasCrCard')
bar_chart('IsActiveMember')
bar_chart('Tenure')
data_salary = data

def salary_cat(x):

    if x <= 50000:

        return ("Normal Customer")

    elif x > 50000 and x <=100000:

        return ("Intermediate Customer")

    elif x >100000 and x <= 150000:

        return ("Class customer")

    else:

        return ("VIP Customer")

data_salary['salary_categ'] = data_salary['EstimatedSalary'].apply(salary_cat)

data_salary.head(10)
bar_chart('salary_categ')
data.head()
#Dropping the added columns

data_final = data.drop(['Age_categ', 'credit_categ', 'salary_categ'], axis=1)

data_final.head()
#create a dict file to convert string variable into numerical one

# for Gender column

gender = {'Male':0, 'Female':1}

data_final.Gender = [gender[item] for item in data_final.Gender]

data_final.head()
data_final.info()
#create a dict file to convert string variable into numerical one

#For contries

geo = {'France':1, 'Spain':2, 'Germany':3}

data_final.Geography = [geo[item] for item in data_final.Geography]

data_final.head()
def age_cat(x):

    if x <= 20:

        return 0

    elif x > 20 and x <=30:

        return 1

    elif x >30 and x <= 40:

        return 2

    elif x >40 and x <= 50:

        return 3

    elif x > 50 and x <= 60:

        return 4

    elif x >60 and x <= 70:

        return 5

    else:

        return 6

data_final['Age'] = data_final['Age'].apply(age_cat)
def salary_cat(x):

    if x <= 50000:

        return 0

    elif x > 50000 and x <=100000:

        return 1

    elif x >100000 and x <= 150000:

        return 2

    else:

        return 3

data_final['EstimatedSalary'] = data_final['EstimatedSalary'].apply(salary_cat)
def credit_cat(x):

    if x <= 350:

        return 0

    elif x > 350 and x <=450:

        return 1

    elif x >450 and x <= 550:

        return 2

    elif x >550 and x <= 650:

        return 3

    elif x > 650 and x <= 750:

        return 4

    else:

        return 5

data_final['CreditScore'] = data_final['CreditScore'].apply(credit_cat)
def balance_cat(x):

    if x <= 50000:

        return 0

    elif x > 50000 and x <=100000:

        return 1

    elif x >100000 and x <= 150000:

        return 2

    elif x >150000 and x <= 200000:

        return 3

    elif x > 200000 and x <= 250000:

        return 4

    else:

        return 5

data_final['Balance'] = data_final['Balance'].apply(balance_cat)
def Tenure_cat(x):

    if x <= 3:

        return 0

    elif x > 3 and x <=6:

        return 1

    elif x >6 and x <= 9:

        return 2

    else:

        return 3

data_final['Tenure'] = data_final['Tenure'].apply(Tenure_cat)
# delete the unnecessary features from dataset



df = data_final.drop(['Surname'], axis=1)

df.head()
df.head()
# Splitting the dataset

from sklearn.model_selection import train_test_split

target = df['Exited']

features = df.drop(['Exited'],  axis=1)

features.shape, target.shape
X_train, X_test, y_train, y_test = train_test_split(

     features, target, test_size=0.4, random_state=0)



X_train.shape, y_train.shape
X_train = X_train.drop(['CustomerId'], axis=1)
X_train.head(10)
# Import classifier Modules



from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



import numpy as np
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = KNeighborsClassifier(n_neighbors = 13)

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# kNN Score

round(np.mean(score)*100, 2)
clf = DecisionTreeClassifier()

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# decision tree Score

round(np.mean(score)*100, 2)
scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# Random Forest Score

round(np.mean(score)*100, 2)
clf = GaussianNB()

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# Naive Bayes Score

round(np.mean(score)*100, 2)
clf = SVC()

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100,2)
clf = SVC()

clf.fit(X_train, y_train)



test_data = X_test.drop("CustomerId", axis=1).copy()

prediction = clf.predict(test_data)
#Calculating the accuracy

print("%s: %.2f%%" % ('Accuracy: ', (clf.score(test_data,y_test))*100))