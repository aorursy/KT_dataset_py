import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns

sns.set_style(style='darkgrid')

from sklearn.tree import DecisionTreeClassifier 

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import plot_tree 

from sklearn import svm

from sklearn.metrics import log_loss

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import cross_val_score 

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import plot_confusion_matrix 

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import jaccard_score

from sklearn.model_selection import GridSearchCV

from math import sqrt
heart = pd.read_csv('../input/heart-disease-uci/heart.csv')
heart.head()
heart.shape # To see now of rows and columns
heart.describe() # StatisticalSummary
heart.isnull().sum() # Checking Missing Values
plt.figure(dpi=80)

sns.heatmap(heart.isnull())

plt.show()
heart.dtypes
#Age

plt.figure(dpi=125)

sns.distplot(a=heart['age'],kde=False,bins=20)

plt.axvline(x=np.mean(heart['age']),c='green',label='Mean Age of all People')

plt.legend()

plt.xlabel('Age')

plt.ylabel('Count')

plt.title('Distribution of Age')

plt.show()
plt.figure(dpi=125)

male =len(heart[heart['sex'] == 1])

female = len(heart[heart['sex']== 0])

sns.countplot('sex',data = heart,)

plt.xlabel('Sex Female-0, Male-1')

plt.ylabel('Count')

plt.title('Count of Sex')

Male, Female =heart.sex.value_counts()

print('Female -',Female)

print('Male -',Male)

plt.show()
plt.figure(dpi=125)

sns.countplot('cp',data = heart,)

plt.xlabel('Chest Pain - 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic')

plt.ylabel('Count')

plt.title('Count of Chest Pain')

A,B,C,D =heart.cp.value_counts()



print('Typical Angina -',A)

print('Atypical Angina -',C)

print('Non-Anginal Pain -',B)

print('Asymptomatic -',D)

 

plt.show()
plt.figure(dpi=125)

sns.countplot('fbs',data = heart,)

plt.xlabel('Fasting Blood Pressure -  0 = >=120 mg/dl,1 = <120 mg/dl')

plt.ylabel('Count')

plt.title('Count of Fasting Blood Pressure')

A,B =heart.fbs.value_counts()



print('Greater than 120 mg/dl -',A)

print('Less than 120 mg/dl-',B)



 

plt.show()
plt.figure(dpi=125)

sns.countplot('exang',data = heart,)

plt.xlabel(' Exercise Induced Angina - 0 = no, 1 = yes')

plt.ylabel('Count ')

plt.title('Count of Exercise Induced Angina')

A,B =heart.exang.value_counts()



print('No -',A)

print('Yes -',B)



 

plt.show()
plt.figure(figsize=(14,7),dpi=100)

sns.heatmap(np.round(heart.corr(),2), annot = True)

plt.show()
plt.figure(figsize=(12,6),dpi=100)

sns.regplot(x='age',y='chol',data=heart,color='Green')

plt.xlabel('Age')

plt.ylabel('Cholesterol in mg/dl')

plt.title('Age vs Cholesterol')

plt.show()
plt.figure(figsize=(12,6),dpi=100)

sns.regplot(x='age',y='thalach',data=heart,color='Blue')

plt.xlabel('Age')

plt.ylabel('Maximum heart rate achieved')

plt.title('Age vs Max Heart Rate')

plt.show()
plt.figure(figsize=(12,6),dpi=100)

sns.regplot(x='age',y='trestbps',data=heart,color='Red')

plt.xlabel('Age')

plt.ylabel('Resting blood pressure (in mm Hg)')

plt.title('Age vs Resting Blood Pressure')

plt.show()
plt.figure(figsize=(12,6),dpi=100)

sns.scatterplot(x='chol',y='thalach',data=heart,hue='target')

plt.xlabel('Cholesterol (in mg/dl)')

plt.ylabel('Resting blood pressure (in mm Hg)')

plt.title('Scatter Plot for Cholesterol and Resting blood pressure')

plt.show()
plt.figure(figsize=(12,6),dpi=100)

sns.swarmplot(x='sex',y='chol',data=heart,hue='target')

plt.xlabel('Sex Female-0, Male-1')

plt.ylabel('Cholesterol in mg/dl')

plt.show()
plt.figure(figsize=(12,6),dpi=100)

sns.swarmplot(x='sex',y='thalach',data=heart,hue='target',dodge=False)

plt.xlabel('Sex Female-0, Male-1')

plt.ylabel('Maximum heart rate achieved')

plt.show()
plt.figure(figsize=(12,6),dpi=100)

sns.swarmplot(x='sex',y='trestbps',data=heart,hue='target',dodge=False)

plt.xlabel('Sex Female-0, Male-1')

plt.ylabel('Resting blood pressure (in mm Hg)')

plt.show()
X= heart.drop('target',axis=1)

y= heart['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTreeClassifier(random_state=42)

clf_dt = clf_dt.fit(X_train, y_train)
plot_confusion_matrix(clf_dt, X_test, y_test, display_labels=["Does not have HD", "Has HD"])
path = clf_dt.cost_complexity_pruning_path(X_train, y_train)

ccp_alphas, impurities = path.ccp_alphas, path.impurities

ccp_alphas = ccp_alphas[:-1]



clf_dts = []

for ccp_alpha in ccp_alphas:

    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)

    clf_dt.fit(X_train, y_train)

    clf_dts.append(clf_dt)
train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]

test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]



fig, ax = plt.subplots()

ax.set_xlabel("alpha")

ax.set_ylabel("accuracy")

ax.set_title("Accuracy vs alpha for training and testing sets")

ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")

ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")

ax.legend()

plt.show()
clf_dt_pruned = DecisionTreeClassifier(random_state=42, 

                                       ccp_alpha=0.02)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train) 
plot_confusion_matrix(clf_dt_pruned, 

                      X_test, 

                      y_test, 

                      display_labels=["Does not have HD", "Has HD"])
DT_score = clf_dt_pruned.score(X_test, y_test)

print("Decision Tree Accuracy:" , DT_score)
plt.figure(figsize=(15,7.5))

plot_tree(clf_dt_pruned, 

          filled=True, 

          rounded=True, 

          class_names=["No HD", "Yes HD"], 

          feature_names=X.columns)
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

LR
LR_score = LR.score(X_test, y_test)

print("Logistic Regression Accuracy:" ,LR_score)
ylr = LR.predict(X_test)
jaccard_score(y_test, ylr)
plot_confusion_matrix(LR, 

                      X_test, 

                      y_test, 

                      display_labels=["Does not have HD", "Has HD"])
print (classification_report(y_test, ylr))
rfc =  RandomForestClassifier(random_state=42).fit(X_train, y_train)
print("Random Forest Accuracy: ", rfc.score(X_test,y_test))
print('Parameters currently in use')

print(rfc.get_params())
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rfc = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train,y_train)
rf_random.best_params_
rfc_n =  RandomForestClassifier(n_estimators= 1000,min_samples_split = 5,min_samples_leaf = 2,max_depth = 20,bootstrap =True,random_state=42).fit(X_train, y_train)
RF_score = rfc_n.score(X_test,y_test)

print("Random Forest Accuracy: ", RF_score)
data = [['Decision Tree', DT_score], ['Logistic Regression', LR_score], ['Random Forest', RF_score]] 

accuracy = pd.DataFrame(data,columns = ['Model', 'Accuracy',])

accuracy.head()
fig= plt.figure(dpi=100)

sns.barplot(x=accuracy['Model'],y=accuracy['Accuracy'])

plt.title('Model Accuracy Comparison')

plt.show()