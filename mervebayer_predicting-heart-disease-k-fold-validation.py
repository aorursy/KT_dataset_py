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
# main libraries
import numpy as np
import pandas as pd

# visual libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
data= pd.read_csv('../input/heart-disease-uci/heart.csv')
data.info()
"""
age -> age in years 
sex(1 = male; 0 = female) 
cp -> chest pain type 
trestbps -> resting blood pressure (in mm Hg on admission to the hospital) 
chol -> serum cholestoral in mg/dl 
fbs -> (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) 
restecg -> resting electrocardiographic results 
thalach -> maximum heart rate achieved 
exang -> exercise induced angina (1 = yes; 0 = no) 
oldpeak -> ST depression induced by exercise relative to rest 
slope -> the slope of the peak exercise ST segment 
ca -> number of major vessels (0-3) colored by flourosopy 
thal -> 3 = normal; 6 = fixed defect; 7 = reversable defect 
target -> 1 or 0 

"""
data.head()
data.isnull().sum() #checking null values, there is no null value.
data.describe() #show basic statistical details 
sns.pairplot(data) #Plot pairwise relationships in a dataset.
#visualize the correlation
plt.figure(figsize=(15,10))
sns.heatmap(data.corr(), annot=True,fmt=".0%")
plt.show() 
#thalach, cp and slope are the most correlated features with target. However, these are only individual correlation. 
# create figure and axis
fig, ax = plt.subplots()
# plot histogram
ax.hist(data['age'])
# set title and labels
ax.set_title('Age Distribution')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')

#When we look at the age disturbution in the dataset, there are too many sample which ages are between 55-65 and least between 
# 30-40 and 70+ no sample which has less than 29 and bigger than 77  
data[data["age"].min() == data["age"]]["age"].iloc[0] #age is between 29 and 77
data[data["age"].max() == data["age"]]["age"].iloc[0]
sns.violinplot(x='target', y='age', data=data) #in dataset samples which are not sick is crowded in age between 55-75
sns.stripplot(x="sex",y="age",data=data,jitter=True,hue='target',palette='Set1')
#in dataset samples which are male is bigger but number of people have disease ratio is bigger in female class.
data['sex'].value_counts().plot.pie(autopct="%1.1f%%")

#sex 0 is female and 1 is male there are 96 females and 207 males in dataset
data['sex'].value_counts()
sns.violinplot(x='sex', y='age', data=data)
data['target'].value_counts().plot.pie(autopct="%1.1f%%") #data has more samples which are having disease
data['target'].value_counts()
pd.crosstab(data.sex,data.target).plot(kind="bar")
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()

#Almost 75% of females are sick in this dataset.
sns.barplot(x="target", y='exang', hue="target", data=data,palette='Set1')
#As seen in the figure number of not exercise angina samples are bigger.
sns.stripplot(x="exang",y="age",data=data,jitter=True,hue='target',palette='Set1')
#in dataset samples which exang is 0 much bigger than 1, and if the sample experinced the angina, possibility of having disease is less than 0
pd.crosstab(data.ca,data.target).plot(kind="bar")
plt.title('Heart Disease Frequency for Ca')
plt.xlabel('Ca')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()

#if you have less vessels then probability of having disease is much bigger.
pd.crosstab(data.thal,data.target).plot(kind="bar")
plt.title('Heart Disease Frequency for thal')
plt.xlabel('thal')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()

#it has a big risk if sample's thal value is 2
sns.stripplot(x="cp",y="oldpeak",data=data,jitter=True,hue='target',palette='Set1')
#data is crowded in cp 0. if oldpeak is bigger than 2.5, there is no big risk
pd.crosstab(data.cp,data.target).plot(kind="bar")
plt.title('Heart Disease Frequency for cp')
plt.xlabel('Cp')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()

#chance of having disease is
#less if sample has typical angina. Chance is getting higher when sample has non-anginal
#pain. However, since there are 87 values, it cannot be claimed that possibility of
#having disease on non-anginal has much bigger than atypical angina or asymptomatic.
X = data.drop(['target'], axis=1) #Aim is to find whether have disease or not.label=target
y = data.target
X.head()
y.head()
#apply SelectKBest class to extract top 10 best features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(13,'Score'))

# According to Chi-squared thalach is the best feature. But in every model, it will use the best feature according to the models
#Chi-squared looks at the correlation between all features.
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier() 
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(13).plot(kind='barh')
plt.show()

#For random forest classifier cp, ca, oldpeak, thalach and age are the most important features.
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
best_features = SelectFromModel(RandomForestClassifier())
best_features.fit(X, y)

transformedX = best_features.transform(X)
print(f"Old Shape: {X.shape}, New shape: {transformedX.shape}")
#selected 7 best features
transformedX 
# you may get same split every time by random state
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


cv = KFold(n_splits=10, shuffle=True, random_state=42) 

accuracies = []
for train, test in cv.split(transformedX):
  model = RandomForestClassifier(criterion='entropy', n_estimators=1000, max_depth=4)
  model.fit(transformedX[train], y[train])

  scr = model.score(transformedX[test], y[test])
  accuracies.append(scr)

print(f"Mean of experiment scores: {np.mean(accuracies)}")
from sklearn.ensemble import RandomForestClassifier

parameters = [{'max_depth' : [1,2,3,4,5,6,7,8,9,10]}]
clf = RandomForestClassifier()
Grid1 = GridSearchCV(clf, parameters, cv=10)
Grid1.fit(transformedX, y)

Grid1.best_estimator_

scores = Grid1.cv_results_
scores['mean_test_score']

#best depth is 5
from sklearn.ensemble import RandomForestClassifier

parameters = [{'n_estimators': [1,10,100,1000],'criterion': ['gini', 'entropy']}]
clf = RandomForestClassifier()
Grid1 = GridSearchCV(clf, parameters, cv=10)
Grid1.fit(transformedX, y)

Grid1.best_estimator_

scores = Grid1.cv_results_
scores['mean_test_score']

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

#For decision tree classifier cp, ca, chol, thal and oldpeak are the most important features.
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
best_features = SelectFromModel(DecisionTreeClassifier())
best_features.fit(X, y)

transformedX = best_features.transform(X)
print(f"Old Shape: {X.shape}, New shape: {transformedX.shape}")
#new features size is 5
# you may get same split every time by random state
from sklearn.model_selection import KFold


cv = KFold(n_splits=10, shuffle=True, random_state=42) 

accuracies = []
for train, test in cv.split(transformedX):
  model = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
  model.fit(transformedX[train], y[train])

  scr = model.score(transformedX[test], y[test])
  accuracies.append(scr)

print(f"Mean of experiment scores: {np.mean(accuracies)}")
parameters = {'max_depth': range(1,10), 
              'min_samples_split': range(2,8), 
              'min_samples_leaf': range(2, 8)}

gcv = GridSearchCV(DecisionTreeClassifier(), parameters, cv=10).fit(transformedX, y)
print(f"Best Estimator: {gcv.best_estimator_}")
print(f"Best Parameter: {gcv.best_params_}")
print(f"Best Score: {gcv.best_score_}")
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
best_features = SelectFromModel(LinearSVC())
best_features.fit(X, y)

transformedX = best_features.transform(X)
print(f"Old Shape: {X.shape}, New shape: {transformedX.shape}")
cv = KFold(n_splits=10, shuffle=True, random_state=42) 

accuracies = []
for train, test in cv.split(transformedX):
  model =  LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
  model.fit(transformedX[train], y[train])

  scr = model.score(transformedX[test], y[test])
  accuracies.append(scr)

print(f"Mean of experiment scores: {np.mean(accuracies)}")
from sklearn.linear_model import LogisticRegression
best_features = SelectFromModel(LogisticRegression())
best_features.fit(X, y)

transformedX = best_features.transform(X)
print(f"Old Shape: {X.shape}, New shape: {transformedX.shape}")
#new size is 7
cv = KFold(n_splits=10, shuffle=True, random_state=42) 

accuracies = []
for train, test in cv.split(transformedX):
  model = LogisticRegression(max_iter=100)
  model.fit(transformedX[train], y[train])

  scr = model.score(transformedX[test], y[test])
  accuracies.append(scr)

print(f"Mean of experiment scores: {np.mean(accuracies)}")
from sklearn.neighbors import KNeighborsClassifier

parameters = [{'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}]
clf = KNeighborsClassifier()
Grid1 = GridSearchCV(clf, parameters, cv=4)
Grid1.fit(transformedX, y)

Grid1.best_estimator_

scores = Grid1.cv_results_
scores['mean_test_score']
#best is 10 with auto
cv = KFold(n_splits=10, shuffle=True, random_state=42) 

accuracies = []
for train, test in cv.split(transformedX):
  model =  KNeighborsClassifier(n_neighbors=10, algorithm="auto")
  model.fit(transformedX[train], y[train])

  scr = model.score(transformedX[test], y[test])
  accuracies.append(scr)

print(f"Mean of experiment scores: {np.mean(accuracies)}")
#Logistic Regression's features are used. Because it is the only one that gave the highest accuracy. 10 is the best choice for 
#n_neighbors
from sklearn.naive_bayes import GaussianNB

cv = KFold(n_splits=10, shuffle=True, random_state=42) 

accuracies = []
for train, test in cv.split(transformedX):
  model =  GaussianNB()
  model.fit(transformedX[train], y[train])

  scr = model.score(transformedX[test], y[test])
  accuracies.append(scr)

print(f"Mean of experiment scores: {np.mean(accuracies)}")