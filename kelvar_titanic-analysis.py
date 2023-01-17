import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pandas.plotting import scatter_matrix
warnings.filterwarnings('ignore')
%matplotlib inline
import os
dataset = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
dataset.info()
dataset.describe()
dataset.head()
def dropColumns(dataset, columns):
    return dataset.drop(columns,axis=1)
dataset = dropColumns(dataset=dataset, columns=["PassengerId", "Name"])
dataset.head()
plt.figure(figsize=(20,20))
p=sns.heatmap(dataset.corr(), annot=True,cmap='RdYlGn',square=True)  
p=sns.pairplot(dataset)
cat_features = dataset.select_dtypes(include=['object']).copy()
cat_features.head()
cat_features.isnull().sum()
cat_features['Cabin'] = cat_features['Cabin'].fillna(cat_features['Cabin'].value_counts().index[0])
cat_features['Embarked'] = cat_features['Embarked'].fillna(cat_features['Embarked'].value_counts().index[0])
cat_features.head()
dataset.dtypes
non_cat_features = dataset.select_dtypes(include=['int64', 'float64']).copy()
non_cat_features.head()
non_cat_features.isnull().sum()
non_cat_features['Age'].fillna(non_cat_features['Age'].mean(), inplace=True)
non_cat_features.isnull().sum()
clean_dataset = pd.concat([non_cat_features, cat_features], axis=1)
clean_dataset.head()
# for the submission
passengerId = test_data.PassengerId
test_data = dropColumns(dataset=test_data, columns=["PassengerId", "Name"])
test_data.head()
cat_features_test = test_data.select_dtypes(include=['object']).copy()
cat_features_test.head()
cat_features_test.isnull().sum()
cat_features_test['Cabin'] = cat_features_test['Cabin'].fillna(cat_features_test['Cabin'].value_counts().index[0])
non_cat_features_test = test_data.select_dtypes(include=['int64', 'float64']).copy()
non_cat_features_test.head()
non_cat_features_test.isnull().sum()
non_cat_features_test['Age'].fillna(non_cat_features_test['Age'].mean(), inplace=True)
non_cat_features_test['Fare'].fillna(non_cat_features_test['Fare'].mean(), inplace=True)
clean_dataset_test = pd.concat([non_cat_features_test, cat_features_test], axis=1)
clean_dataset_test.head()
# histogram
clean_dataset.hist(figsize=(20, 20))
plt.show()
sm = scatter_matrix(clean_dataset, alpha=0.2, figsize=(20, 20), diagonal='kde')

#Change label rotation
[s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
[s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]

#May need to offset label when rotating to prevent overlap of figure
[s.get_yaxis().set_label_coords(-0.3,0.5) for s in sm.reshape(-1)]

#Hide all ticks
[s.set_xticks(()) for s in sm.reshape(-1)]
[s.set_yticks(()) for s in sm.reshape(-1)]

plt.show()
def displayBarPlot(feature):
    sex_count = clean_dataset[feature].value_counts()
    sns.set(style="darkgrid")
    sns.barplot(sex_count.index, sex_count.values, alpha=0.9)
    plt.title('Frequency Distribution of %s' %feature)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    
def displayPieChart(feature):
    labels = clean_dataset[feature].astype('category').cat.categories.tolist()
    counts = clean_dataset[feature].value_counts()
    sizes = [counts[var_cat] for var_cat in labels]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
    ax1.axis('equal')
    plt.show()
    
def barPlotFeatureTarget(feature):
    f,ax=plt.subplots(1,2,figsize=(18,8))
    clean_dataset[[feature,'Survived']].groupby([feature]).mean().plot.bar(ax=ax[0])
    ax[0].set_title('Survived vs %s' %feature)
    sns.countplot(feature,hue='Survived',data=clean_dataset,ax=ax[1])
    ax[1].set_title('%s:Survived vs Dead' %feature)
    plt.show()
    
def displayKdePlot(feature, hue):
    sns.FacetGrid(clean_dataset, hue=hue, size=5).map(sns.kdeplot, feature).add_legend()
    plt.show()
    
def displayJointPlot(x, y):
    sns.jointplot(x=x,y=y, data=clean_dataset)

displayBarPlot('Sex')
displayPieChart('Sex')
p = sns.factorplot(x='Sex', y='Survived', data=clean_dataset, kind='box' ,aspect=2.5 )
barPlotFeatureTarget('Sex')
p = sns.factorplot(x='Embarked', data=clean_dataset , kind='count',aspect=2.5 )
displayBarPlot(feature='Embarked')
displayPieChart(feature='Embarked')
barPlotFeatureTarget(feature='Embarked')
displayKdePlot(feature='Age',hue='Survived')
clean_dataset.head()
# Number of distinct Ticket
clean_dataset['Ticket'].value_counts().count()
# Number of distinct Cabin
clean_dataset['Cabin'].value_counts().count()
# pd.options.display.max_rows = 4000
clean_dataset.groupby("Ticket")['Fare'].mean().sort_values()
"""
0 - 50 = T1
50 - 100 = T2
100 - 150 = T3
150 - 200 = T4
200 - 250 = T5
250 - 300 = T6
300 - ... = T7
"""
def ticket_group(i):
    group = 0
    if i<100:
        group = "T1"
    elif i>=100 and i<150:
        group = "T2"
    elif i>=150 and i<200:
        group = "T3"
    elif i>=200 and i<250:
        group = "T4"
    elif i>= 250 and i<300:
        group = "T5"
    else:
        group = "T6"
    return group
    
clean_dataset['Ticket'] = clean_dataset.Fare.apply(lambda x: ticket_group(x))
clean_dataset.head()
clean_dataset.groupby("Ticket")['Fare'].mean().sort_values()
clean_dataset['Ticket'].value_counts()
clean_dataset['Cabin'] = [i[0] for i in clean_dataset.Cabin]
clean_dataset.head()
dataset[dataset.Cabin == 'T'].count()
test_data[test_data.Cabin == 'T'].count()
clean_dataset.columns
i = clean_dataset[clean_dataset.Cabin == 'T'].index
clean_dataset = clean_dataset.drop(i)
#clean_dataset = clean_dataset.drop("Cabin_T", axis=1)
clean_dataset.head()
clean_dataset_test['Ticket'].value_counts().count()
clean_dataset_test['Cabin'].value_counts().count()
clean_dataset_test['Ticket'] = clean_dataset_test.Fare.apply(lambda x: ticket_group(x))
clean_dataset_test.groupby("Ticket")['Fare'].mean().sort_values()
clean_dataset_test['Cabin'] = [i[0] for i in clean_dataset_test.Cabin]
def oneHotEncode(dataset, feature):
    return pd.get_dummies(dataset[feature])
encoded_embarked = oneHotEncode(dataset=clean_dataset, feature='Embarked')
encoded_embarked.head()
encoded_sex =  oneHotEncode(dataset=clean_dataset, feature='Sex')
encoded_sex.head()
encoded_ticket =  oneHotEncode(dataset=clean_dataset, feature='Ticket')
encoded_ticket.head()
encoded_cabin =  oneHotEncode(dataset=clean_dataset, feature='Cabin')
# Rename the columns to distinguish from the "C" cabin from the "C" value in Embarked
encoded_cabin = encoded_cabin.rename(columns={'A':'Cabin_A', 'B': 'Cabin_B', 
                                                         'C': 'Cabin_C', 'D': 'Cabin_D', 
                                                         'E': 'Cabin_E', 'F': 'Cabin_F', 
                                                         'G': 'Cabin_G', 'T': 'Cabin_T'})
#encoded_cabin = encoded_cabin.rename_axis({'A': 'Cabin_A'})
encoded_cabin.head()
clean_dataset = pd.concat([clean_dataset, encoded_sex, encoded_ticket, encoded_cabin, encoded_embarked], axis=1)
# Drop column
clean_dataset = clean_dataset.drop(['Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
clean_dataset.head()
clean_dataset.columns
# Embarked Feature
encoded_embarked_test = oneHotEncode(dataset=clean_dataset_test, feature='Embarked')
# Sex feature
encoded_sex_test =  oneHotEncode(dataset=clean_dataset_test, feature='Sex')
# Ticket
encoded_ticket_test =  oneHotEncode(dataset=clean_dataset_test, feature='Ticket')
# Cabin
encoded_cabin_test =  oneHotEncode(dataset=clean_dataset_test, feature='Cabin')
# Rename the columns to distinguish from the "C" cabin from the "C" value in Embarked
encoded_cabin_test = encoded_cabin_test.rename(columns={'A':'Cabin_A', 'B': 'Cabin_B', 
                                                         'C': 'Cabin_C', 'D': 'Cabin_D', 
                                                         'E': 'Cabin_E', 'F': 'Cabin_F', 
                                                         'G': 'Cabin_G', 'T': 'Cabin_T'})

clean_dataset_test = pd.concat([clean_dataset_test, encoded_sex_test, encoded_ticket_test, encoded_cabin_test, encoded_embarked_test], axis=1)
# Drop column
clean_dataset_test = clean_dataset_test.drop(['Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
clean_dataset_test.head()

clean_dataset_test.columns
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

columns = clean_dataset.columns
columns = columns.drop(['Survived'])
survived = clean_dataset.Survived
data = clean_dataset.drop(['Survived'], axis=1)


clean_dataset_scaled = pd.DataFrame(sc.fit_transform(data),columns=columns,index=clean_dataset.index)
clean_dataset_scaled = pd.concat([clean_dataset_scaled, survived], axis=1)
clean_dataset_scaled.head()
# No 'Survived' column in test set
columns_test = clean_dataset_test.columns

clean_dataset_test_scaled = pd.DataFrame(sc.fit_transform(clean_dataset_test),columns=columns_test,index=clean_dataset_test.index)
clean_dataset_test_scaled.head()
X = clean_dataset_scaled.drop(["Survived"],axis=1)
y = clean_dataset_scaled.Survived
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 2,test_size=0.3)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
def checkPerformances(classifier, best_clf):
    # Make predictions using the unoptimized and optimized and model
    predictions = (classifier.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)


    # Report the before-and-afterscores
    print("Unoptimized model\n------")
    print(classifier)
    print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5, average="micro")))
    print("\nOptimized Model\n------")
    print(best_clf)
    print("\nFinal accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
    print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5,  average="micro")))
scorer = make_scorer(fbeta_score, beta=0.5, average="micro")
DT_clf = DecisionTreeClassifier()
DT_parameters = {
    'criterion': ['gini', 'entropy'],
    'min_samples_leaf': [1, 2, 3, 4]
    }

# Run the grid search
grid_obj = GridSearchCV(DT_clf, DT_parameters, scoring=scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the regressor to the best combination of parameters
best_DT_clf = grid_obj.best_estimator_

checkPerformances(DT_clf, best_DT_clf)
from sklearn.ensemble import RandomForestClassifier
RF_clf = RandomForestClassifier(max_depth=None, random_state=None)

parameters = {'n_estimators': [10, 20, 30, 100], 'max_features':[3,4,5, None], 'max_depth': [5,6,7, None], 'criterion': ['gini', 'entropy']}
grid_obj = GridSearchCV(RF_clf, parameters, scoring=scorer)
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_RF_clf = grid_fit.best_estimator_

checkPerformances(classifier=RF_clf, best_clf=best_RF_clf)
LR_clf = LogisticRegression(random_state = 0)
LR_parameters = {
    'penalty': ['l1', 'l2'],
    'C': np.logspace(0, 4, 10)
    }

# Run the grid search
grid_obj = GridSearchCV(LR_clf, LR_parameters, scoring=scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the regressor to the best combination of parameters
best_LR_clf = grid_obj.best_estimator_

best_LR_clf
checkPerformances(LR_clf, best_LR_clf)
NB_clf = GaussianNB()
NB_clf.fit(X_train, y_train)
predictions = (NB_clf.fit(X_train, y_train)).predict(X_test)

print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5, average="micro")))
SVM_clf = SVC()
SVM_parameters = {
    'kernel': ['linear', 'poly', 'rbf'],
    'degree': [1, 2, 3, 4],
    'shrinking' : [True, False]
    }

# Run the grid search
grid_obj = GridSearchCV(SVM_clf, SVM_parameters, scoring='accuracy')
grid_obj = grid_obj.fit(X_train, y_train)

# Set the regressor to the best combination of parameters
best_SVM_clf = grid_obj.best_estimator_

checkPerformances(SVM_clf, best_SVM_clf)
from sklearn import model_selection
# 10-fold cross validation
# Test options and evaluation metric
seed = 7
# Using metric accuracy to measure performance
scoring = 'accuracy' 
# Spot Check Algorithms
models = []
models.append(('LR', best_LR_clf))
models.append(('DT', best_DT_clf))
models.append(('RF', best_RF_clf))
#models.append(('KNN', best_KNN_clf))
models.append(('NB', NB_clf))
models.append(('SVM', best_SVM_clf))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
y_pred = best_RF_clf.predict(X_test)
# Confusion matrix
labels = [0, 1]
cm = confusion_matrix(y_test, y_pred, labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the RF classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
'''
submission = pd.DataFrame({
        "PassengerId": dataset["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('submission.csv', index=False)
'''

X_train.columns
clean_dataset_test_scaled.columns
best_RF_clf.fit(X_train, y_train)
pred = best_RF_clf.predict(clean_dataset_test_scaled)
output=pd.DataFrame({'PassengerId':passengerId,'Survived':pred})
output.to_csv('Submission.csv', index=False)