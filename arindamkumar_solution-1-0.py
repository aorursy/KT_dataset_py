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

import matplotlib as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
path='../input/'
train=pd.read_csv(path+'train.csv')
test=pd.read_csv(path+'test.csv')
# save the passenger id for the final submission
passengerId=test.PassengerId

# merge train and test
titanic = train.append(test, ignore_index=True)

## we use the ignore_index as in test data we have the labels columns which is not present in the train data.
train_id=len(train)
test_id=len(titanic)-len(test)
train_id
test_id
len(titanic)
len(test)
titanic.head()
titanic.info()
titanic.drop(['PassengerId'],1,inplace=True)
titanic.head()
titanic['Title']=titanic.Name.apply(lambda name:name.split(',')[1].split('.')[0].strip() )
titanic.head()
## title counts
#print("There are {} unique title.".format(titanic.Title.nunique))
print("There are {} unique titles.".format(titanic.Title.nunique()))
print("\n", titanic.Title.unique())
titanic.head()
# normalize the titles
normalized_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}

def convert(val):
    return normalized_titles[val]
titanic.head()
type(titanic.Title.values[0])
# view value counts for the normalized titles
print(titanic.Title.value_counts())

titanic.Title = titanic.Title.map(normalized_titles)

titanic.head()
# view value counts for the normalized titles
print(titanic.Title.value_counts())
#groupby sex,Pclass and Title
grouped=titanic.groupby(['Sex','Pclass','Title'])
grouped.Age.median()
## applying the grouped median age value
titanic.Age=grouped.Age.apply(lambda x:x.fillna(x.median()))

titanic.info()
titanic.head(10)
titanic.Cabin=titanic.Cabin.fillna('NA')     ## NA-not available
titanic.head()
titanic.Embarked.value_counts()
most_embarked=titanic.Embarked.value_counts().index[0]
most_embarked
titanic.Embarked=titanic.Embarked.fillna(most_embarked)
titanic.head()
titanic.info()
##only fare is left incomplete
titanic.Fare=titanic.Fare.fillna(titanic.Fare.median())

titanic.info()
##percentage of death vs percentage of survival
titanic.Survived.value_counts()
titanic.Survived.value_counts(normalize=True)
## lets dig deeper and determine the survival rates based on the gender
groupbysex=titanic.groupby(['Sex'])
groupbysex.Survived.value_counts(normalize=True)
##survival rates based on their sex
groupbysex.Survived.mean()
## group by passenge Pclass and sex
group_class_sex=titanic.groupby(['Pclass','Sex'])
group_class_sex.Survived.mean()
##get stats on all other metrics
titanic.describe()
## size of the family including the passenger.
titanic['FamilySize']=titanic['Parch']+titanic['SibSp']+1
## map the first letter of the cabin to the cabin.
titanic.Cabin=titanic.Cabin.map(lambda x:x[0])

## view the normalized count
titanic.Cabin.value_counts(normalize=True)
titanic.head()
def handle_non_numeric_data(df):
	columns=df.columns.values
	for column in columns:
		text_digit_vals={}
		def convert_to_int(val):
			return text_digit_vals[val] 

		if df[column].dtype!= np.int64 and df[column].dtype!= np.float64:
			column_contents=df[column].values.tolist()		#.values is used to get the values of a function
			unique_elements=set(column_contents)	#converting to a set
			x=0
					
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique]=x
					x+=1

			df[column]=list(map(convert_to_int,df[column]))		#we are resetting the df column by mapping the function here to the value in the column

	return df
titanic=handle_non_numeric_data(titanic)
titanic.head()
train=titanic[:train_id]
test=titanic[test_id:]
## convert the survived back to int
train.Suvived=train.Survived.astype(int)
train.head()
# create X and y for data and target values
X = train.drop('Survived', axis=1).values
y = train.Survived.values
test.head()
X_test=test.drop('Survived',1).values
# The parameters that we are going to optimise
parameters = dict(
    C = np.logspace(-5, 10, 15),
    penalty = ['l1', 'l2']
    #solver =[‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’]
    
)
## instantiate the logistic regression
clf=LogisticRegression()

# Perform grid search using the parameters and f1_scorer as the scoring method
grid_search=GridSearchCV(estimator=clf,param_grid=parameters,cv=6,n_jobs=-1)
# here cv is used for the cross-validation strategy.

grid_search.fit(X,y)
clf1=grid_search.best_estimator_        # get the best estimator(classifier)
print(clf1)
# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(grid_search.best_params_)) 
print("Best score is {}".format(grid_search.best_score_))
## prediction on test set
pred=grid_search.predict(X_test)
print(pred)
# create param grid object
forrest_params = dict(
    max_depth = [n for n in range(7, 14)],
    min_samples_split = [n for n in range(4, 12)],
    min_samples_leaf = [n for n in range(2, 6)],
    n_estimators = [n for n in range(10, 60, 10)],
)
forest=RandomForestClassifier()
# build and fit model
forest_cv = GridSearchCV(estimator=forest, param_grid=forrest_params, cv=5)
forest_cv.fit(X, y)
print("Best score: {}".format(forest_cv.best_score_))
print("Optimal params: {}".format(forest_cv.best_estimator_))
# random forrest prediction on test set
forrest_pred = forest_cv.predict(X_test)
sub=pd.DataFrame({'PassengerId':passengerId,'Survived':forrest_pred})
sub.head()
sub.to_csv('prediction.csv',index=False)   
## we initialise the index as false as we donot need the index


