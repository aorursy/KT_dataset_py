# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# importing "random" for random value assignments 

import random 



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



#importing the required ML libraries

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path = os.path.join(dirname, filename)

        print(path)





# Any results you write to the current directory are saved as output.



# Import Data

test_df = pd.read_csv("/kaggle/input/ucfai-dsg-quals-fa19/test.csv")

train_df = pd.read_csv("/kaggle/input/ucfai-dsg-quals-fa19/train.csv")
# Display fields and their data types

# Most fields are categorical, we will have to convert these to numerical

# dummy values so that our model understands them.

train_df.info()

test_df.info()
# Since the DOB attribute has mostly unqiue values, it will be tricky to convert them into useful categories. We will drop it from the dataset.

train_df["DOB"].describe()

train_df = train_df.drop(['DOB'], axis=1)

test_df = test_df.drop(['DOB'], axis=1)
# First, I will drop ‘ID’ from the train set, because it does not contribute to a persons survival probability. 

# I will not drop it from the test set, since it is required there for the submission.

train_df = train_df.drop(['ID'], axis=1)
print(train_df.isnull().sum()) 

print(test_df.isnull().sum()) 
# The last row in the training data seems to be the culprit for many of our missing null values. Lets get rid of it.

train_df.tail(5)

train_df.drop(train_df.tail(1).index, inplace=True) # drop last row

print(train_df.isnull().sum()) 

print(test_df.isnull().sum()) 

# Describe the Martial Status feature.

# Because this feature has 12 null values, we will assign

# 6 Married, 5 Unmarried, 1 Divorced, based on the valuepercentage breakdown.

# There are the same number of nulls in the test and training for this feature.

print(train_df["M_STATUS"].describe())

print()

print(train_df.groupby(['M_STATUS', 'Class']).size())

print()

print(train_df['M_STATUS'].value_counts(normalize=True) * 100)

print()

print(train_df['M_STATUS'].isnull().sum())

print()



# Convert M_STATUS feature from string into numeric

m_status = {"Married": 0, "Unmarried": 1, "Divorced": 2}

data = [train_df, test_df]



for dataset in data:

    dataset['M_STATUS'] = dataset['M_STATUS'].map(m_status)

    

    value = random.randint(0,100)

    if value <= 53:

        value = 0

    elif value > 53 <= 95:

        value = 1

    else:

        value = 2

        

    dataset['M_STATUS'] = dataset['M_STATUS'].fillna(value)
# Convert EMP_DATA feature from string into numeric

# Fill the one null value with the most commost value

emp_status = {"Employed": 0, "Unemployed": 1, "Self-Employed": 2}

data = [train_df, test_df]



for dataset in data:

    dataset['EMP_DATA'] = dataset['EMP_DATA'].map(emp_status)

    dataset['EMP_DATA'] = dataset['EMP_DATA'].fillna(0)
# Convert REL_ORIEN feature from string into numeric

# Fill the one null value with the most commost value

rel_status = {"Believer": 0, "Agnostic": 1, "Atheist": 2}

data = [train_df, test_df]



for dataset in data:

    dataset['REL_ORIEN'] = dataset['REL_ORIEN'].map(rel_status)

    dataset['REL_ORIEN'] = dataset['REL_ORIEN'].fillna(0)
# Convert GEN_MOVIES feature from string into numeric

movie_status = {"Horror": 0, "Thriller": 1, "Comedy": 2, "Action":3, "Drama":4, "Historical":5, "Romantic":6}

data = [train_df, test_df]



for dataset in data:

    dataset['GEN_MOVIES'] = dataset['GEN_MOVIES'].map(movie_status)

    dataset['GEN_MOVIES'] = dataset['GEN_MOVIES'].fillna(0)
# Convert EDU-Data feature from string into numeric

# There is are 4 Null values. We will assign one to each of the values of this feature, an even scaling.

education_status = {"Graduate": 0, "High-School": 1, "Post-Graduate": 2, "Uneducated":3}

data = [train_df, test_df]

assignment = 0



for dataset in data:

    dataset['EDU_DATA'].map(education_status)

    dataset["EDU_DATA"].fillna(assignment, inplace = True) 

    assignment += 1
# Convert GENDER feature from string into numeric

# There is one NaN value in the Gender column so I will fill this with the more common value: Male

gender_status = {"Male": 0, "Female": 1}

data = [train_df, test_df]



for dataset in data:

    dataset['GENDER'] = dataset['GENDER'].map(gender_status)

    dataset["GENDER"].fillna(0, inplace = True) 
# Convert Alcohol feature from string into numeric

# There is one NaN value in the Gender column so I will fill this with the more common value: Male

alcohol_status = {"Beer": 0, "Gin": 1, "Non":2, "Rum":3, "Vodka":4, "Whiskey":5, "wine":6}

data = [train_df, test_df]



for dataset in data:

    dataset['ALCOHOL'] = dataset['ALCOHOL'].map(alcohol_status)

    dataset["ALCOHOL"].fillna(0, inplace = True) 
# Convert SALARY feature from string into numeric

# Missing SALARY fields will be filled randomly

data = [train_df, test_df]



for dataset in data:

    dataset["SALARY"].fillna(random.randint(0, 5), inplace = True) 

    dataset['SALARY'] = dataset['SALARY'].astype('category').cat.codes
# Convert PREF_CAR feature from string into numeric

data = [train_df, test_df]



for dataset in data:

    dataset['PREF_CAR'] = dataset['PREF_CAR'].astype('category').cat.codes
# Convert FAV_CUIS feature from string into numeric

data = [train_df, test_df]



for dataset in data:

    dataset['FAV_CUIS'] = dataset['FAV_CUIS'].astype('category').cat.codes
# Convert FAV_COLR feature from string into numeric

data = [train_df, test_df]



for dataset in data:

    dataset['FAV_COLR'] = dataset['FAV_COLR'].astype('category').cat.codes
# Convert NEWS_SOURCE feature from string into numeric

data = [train_df, test_df]



for dataset in data:

    dataset['NEWS_SOURCE'] = dataset['NEWS_SOURCE'].astype('category').cat.codes
# Convert FAV_MUSIC feature from string into numeric

data = [train_df, test_df]



for dataset in data:

    dataset['FAV_MUSIC'] = dataset['FAV_MUSIC'].astype('category').cat.codes
# Convert MNTLY_TRAVEL feature from string into numeric

data = [train_df, test_df]



for dataset in data:

    dataset['MNTLY_TRAVEL'] = dataset['MNTLY_TRAVEL'].astype('category').cat.codes
# Convert DIST_FRM_COAST feature from string into numeric

data = [train_df, test_df]



for dataset in data:

    dataset['DIST_FRM_COAST'] = dataset['DIST_FRM_COAST'].astype('category').cat.codes
# Convert FAV_SUBJ feature from string into numeric

data = [train_df, test_df]



for dataset in data:

    dataset['FAV_SUBJ'] = dataset['FAV_SUBJ'].astype('category').cat.codes
# Convert FAV_SUPERHERO feature from string into numeric

data = [train_df, test_df]



for dataset in data:

    dataset['FAV_SUPERHERO'] = dataset['FAV_SUPERHERO'].astype('category').cat.codes
# Convert FAV_SUPERHERO feature from string into numeric

data = [train_df, test_df]



for dataset in data:

    dataset['FAV_SUPERHERO'] = dataset['FAV_SUPERHERO'].astype('category').cat.codes
# Convert EDU_DATA feature from string into numeric

data = [train_df, test_df]



for dataset in data:

    dataset['FAV_SPORT'] = dataset['FAV_SPORT'].astype('category').cat.codes
# Convert EDU_DATA feature from string into numeric

data = [train_df, test_df]



for dataset in data:

    dataset['EDU_DATA'] = dataset['EDU_DATA'].astype('category').cat.codes
# Convert EDU_DATA feature from string into numeric

data = [train_df, test_df]



for dataset in data:

    dataset['ENDU_LEVEL'] = dataset['ENDU_LEVEL'].astype('category').cat.codes
train_df = train_df.drop(['FAV_TV'], axis=1)

test_df = test_df.drop(['FAV_TV'], axis=1)
# Convert Class feature from string into numeric

survived_status = {'x': 0, 'y': 1}

train_df['Class'] = train_df['Class'].map(survived_status)
train_df.head(25)
# Dist_Coast column has continuos data and there might be fluctuations that do not reflect patterns in the data, which might be noise. 

# That's why we WILL put people that are within a certain range IN the same bin. This can be achieved using qcut method in pandas.

print(train_df["Dist_Coast"].describe())



train_df["Dist_Coast"] = pd.qcut(train_df.Dist_Coast, int(dist_coast.max() / dist_coast.min()), labels=False)

test_df["Dist_Coast"] = pd.qcut(test_df.Dist_Coast, int(dist_coast.max() / dist_coast.min()), labels=False)
train_df.head()
sns.set_style('whitegrid')

sns.countplot(x='Class',data=train_df,palette='RdBu_r')
sns.countplot('GENDER',data=train_df)

train_df['GENDER'].value_counts()
# Comparing the Sex feature against Survived

# mALE : 0; Female : 1

print(train_df.groupby(['GENDER', 'Class']).size())

sns.set_style('whitegrid')

sns.countplot(x='Class',hue='GENDER',data=train_df,palette='RdBu_r')
#

print(train_df.groupby(['EMP_DATA', 'Class']).size())

sns.set_style('whitegrid')

sns.countplot(x='Class',hue='EMP_DATA',data=train_df,palette='rainbow')
print(train_df.groupby(['SALARY', 'Class']).size())

sns.set_style('whitegrid')

sns.countplot(x='Class',hue='SALARY',data=train_df,palette='rainbow')
print(train_df.groupby(['M_STATUS', 'Class']).size())

sns.set_style('whitegrid')

sns.countplot(x='Class',hue='M_STATUS',data=train_df,palette='rainbow')
print(train_df.groupby(['REL_ORIEN', 'Class']).size())

sns.set_style('whitegrid')

sns.countplot(x='Class',hue='REL_ORIEN',data=train_df,palette='rainbow')
print(train_df.groupby(['EDU_DATA', 'Class']).size())

sns.set_style('whitegrid')

sns.countplot(x='Class',hue='EDU_DATA',data=train_df,palette='rainbow')
sns.set_style('whitegrid')

sns.countplot(x='Class',hue='FAV_TV',data=train_df,palette='rainbow')
sns.lmplot(x='SALARY',y='Class',data=train_df,hue='GENDER',palette='Set1')
#Splitting out training data into X: features and y: target

y = train_df['Class']

X = train_df.drop("Class", axis=1)



#splitting our training data again in train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)



print(X_train)

print(y_train)
#Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)

acc_logreg
#let's perform some K-fold cross validation for logistic Regression

cv_scores = cross_val_score(logreg,X,y,cv=5)

 

np.mean(cv_scores)*100
#Decision Tree Classifier



decisiontree = DecisionTreeClassifier()

dep = np.arange(1,10)

param_grid = {'max_depth' : dep}



clf_cv = GridSearchCV(decisiontree, param_grid=param_grid, cv=5)



clf_cv.fit(X, y)

clf_cv.best_params_,clf_cv.best_score_*100

print('Best value of max_depth:',clf_cv.best_params_)

print('Best score:',clf_cv.best_score_*100)
gbk = GradientBoostingClassifier()

ne = np.arange(1,20)

dep = np.arange(1,10)

param_grid = {'n_estimators' : ne,'max_depth' : dep}



gbk_cv = GridSearchCV(gbk, param_grid=param_grid, cv=5)



gbk_cv.fit(X, y)

print('Best value of parameters:',gbk_cv.best_params_)

print('Best score:',gbk_cv.best_score_*100)
y_final = clf_cv.predict(test_df.drop("ID", axis=1))



submission = pd.DataFrame({

        "ID": test_df["ID"],

        "Class": y_final

    })

submission.head()

submission.to_csv('hurricane.csv', index=False)