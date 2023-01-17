import pandas as pd
import numpy as np
import matplotlib 

#Read the data and input them in Dataframes
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
#print(train_df.describe(), test_df.describe())
#print(train_df.head(10),test_df.head(10))

#Handle missing values in age
train_df['Age']= train_df['Age'].fillna(-1)
test_df['Age']= train_df['Age'].fillna(-1)

#Feature Extrtaction
#Making "Age brackets" instead of age values
age_bins = [-1, 0, 10, 20, 30, 40, 50, 60, 70, 100]
age_labels = ['Missing', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', 'Over 70']

train_df['Age_bins'] = pd.cut(train_df['Age'],bins = age_bins, labels = age_labels, include_lowest = True)
test_df['Age_bins'] = pd.cut(train_df['Age'],bins = age_bins, labels = age_labels, include_lowest = True)

#for fare bins, we need to find the max value to partition it appropriately
max = train_df.loc[train_df['Fare'].idxmax()]
test_max = test_df.loc[test_df['Fare'].idxmax()]
#print(test_max)

fare_bins = [0, 30, 50, 100, 200, 513]
fare_labels = ['Under 30', 'Upto 50', 'Upto 100', 'Upto 200', 'Over 200']

train_df['Fare_bins'] = pd.cut(train_df['Fare'], bins=fare_bins, labels=fare_labels, include_lowest=True)
test_df['Fare_bins'] = pd.cut(test_df['Fare'], bins=fare_bins, labels=fare_labels, include_lowest=True)

#Extracting salutations from names
split_salutation = train_df['Name'].str.split()
train_df['Salutation'] = split_salutation.str[1]
test_split_salutation = test_df['Name'].str.split()
test_df['Salutation'] = test_split_salutation.str[1]

#print(train_df['Salutation'].unique())
#Bucketing Misc as 'other'
train_df['Salutation'] = train_df['Salutation'].replace(['Planke,','Don.','Rev.','Billiard,','der','Walle,','Dr.','Pelsmaeker,','Mulder,','Steen,','Carlo,','Mme.','Impe,','Ms.','Major','Gordon,','Messemaeker,','Mlle.','Col.','Capt.','Velde,','the','Shawah,','Jonkheer.','Melkebeke,','Cruyssen,', 'Khalil,', 'y'], 'Other')
test_df['Salutation'] = test_df['Salutation'].replace(['Planke,','Don.','Rev.','Billiard,','der','Walle,','Dr.','Pelsmaeker,','Mulder,','Steen,','Carlo,','Mme.','Impe,','Ms.','Major','Gordon,','Messemaeker,','Mlle.','Col.','Capt.','Velde,','the','Shawah,','Jonkheer.','Melkebeke,','Cruyssen,', 'Khalil,', 'y'], 'Other')

#Is Cabin a worthwhile feature?
train_df['Cabin_wing'] = train_df['Cabin'].astype(str).str[0]
test_df['Cabin_wing'] = test_df['Cabin'].astype(str).str[0]
#print(train_df['Cabin_wing'].head(20))
#Maybe next version?

#Encoding - Changing categorical or groups and values to integers 
#Ex : Female - 0, Male 1, Age groups get their own integers from 0 to 8
#Doing this for sex, age, fare, cabin, salutation

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
train_df['Sex_code'] = enc.fit_transform(train_df['Sex'])
train_df['Age_code'] = enc.fit_transform(train_df['Age_bins'])
train_df['Fare_code'] = enc.fit_transform(train_df['Fare_bins'].astype(str))
train_df['Salutation'] = enc.fit_transform(train_df['Salutation'].astype(str))

test_df['Sex_code'] = enc.fit_transform(test_df['Sex'])
test_df['Age_code'] = enc.fit_transform(test_df['Age_bins'])
test_df['Fare_code'] = enc.fit_transform(test_df['Fare_bins'].astype(str))
test_df['Salutation'] = enc.fit_transform(test_df['Salutation'].astype(str))
test_df['Salutation'] = enc.fit_transform(test_df['Salutation'].astype(str))
#print(train_df, test_df)

#Handle missing values
#print(train_df.isna(), test_df.isna())
train_df.dropna()
test_df.dropna()

#Model testing
#gathering all imports from sklearn
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

#trying logistic regression 
lr = LogisticRegression()

#columns to test it on
columns = ['Pclass', 'Age_code', 'Sex_code', 'Parch', 'Salutation']
all_X = train_df[columns]
all_y = train_df['Survived']
#Dividing train and test data to test accuracy of models before putting it on test data
train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2,random_state=0)

#testing logistic regression
lr.fit(train_X, train_y)
lr_predictions = lr.predict(test_X)
lr_accuracy = accuracy_score(test_y, lr_predictions)
lr_conf_matrix = confusion_matrix(test_y, lr_predictions)
lr_scores = cross_val_score(lr, all_X, all_y, cv=10)
lr_mean_scores = np.mean(lr_scores)
#print('Linear Regression', lr_accuracy, lr_mean_scores)

modelTest = pd.DataFrame([['Linear Regression', lr_accuracy, lr_mean_scores]])

#Model Selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

#Random Forest Classification
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(train_X, train_y)
#print(clf.feature_importances_)

clf_predictions = clf.predict(test_X)
clf_accuracy = accuracy_score(test_y, clf_predictions)
clf_scores = cross_val_score(clf, all_X, all_y, cv=10)
clf_mean_scores = np.mean(clf_scores)

randomforest = pd.DataFrame([['Random Forest', clf_accuracy, clf_mean_scores]])
modelTest = modelTest.append(randomforest)

#Perceptron
perc = Perceptron(max_iter=1)
perc.fit(train_X, train_y)
perc_predictions = perc.predict(test_X)
perc_accuracy = accuracy_score(test_y, perc_predictions)
perc_scores = cross_val_score(perc, all_X, all_y, cv=10)
perc_mean_scores = np.mean(perc_scores)
perceptron = pd.DataFrame([['Perceptron', perc_accuracy, perc_mean_scores]])
modelTest = modelTest.append(perceptron)

#Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc.fit(train_X, train_y)
dtc_predictions = dtc.predict(test_X)
dtc_accuracy = accuracy_score(test_y, dtc_predictions)
dtc_scores = cross_val_score(dtc, all_X, all_y, cv=10)
dtc_mean_scores = np.mean(dtc_scores)
decisiontree = pd.DataFrame([['Decision Trees', dtc_accuracy, dtc_mean_scores]])
modelTest = modelTest.append(decisiontree)

#K Neighbor
knc = KNeighborsClassifier()
knc.fit(train_X, train_y)
knc_predictions = knc.predict(test_X)
knc_accuracy = accuracy_score(test_y, knc_predictions)
knc_scores = cross_val_score(knc, all_X, all_y, cv=10)
knc_mean_scores = np.mean(knc_scores)
knclass = pd.DataFrame([['K neighbour', knc_accuracy, knc_mean_scores]])
modelTest = modelTest.append(knclass)

##Stochastic Gradient Descent classifier
sgdc = SGDClassifier(max_iter = 5)
sgdc.fit(train_X, train_y)
sgdc_predictions = sgdc.predict(test_X)
sgdc_accuracy = accuracy_score(test_y, sgdc_predictions)
sgdc_scores = cross_val_score(sgdc, all_X, all_y, cv=10)
sgdc_mean_scores = np.mean(sgdc_scores)
sgdclass = pd.DataFrame([['SGDC', sgdc_accuracy, sgdc_mean_scores]])
modelTest = modelTest.append(sgdclass)

modelTest.columns = ['Model Name', 'Accuracy', 'Mean Scores']
print(modelTest)

##Applying test data
new_X = test_df[columns]
clf.fit(train_X, train_y)
new_clf_predictions = clf.predict(new_X)

pass_id = test_df['PassengerId']
result = pd.DataFrame({ 'PassengerId' : pass_id, 'Survived': new_clf_predictions })
result.to_csv('titanic-results.csv')

print(result.head(20))


