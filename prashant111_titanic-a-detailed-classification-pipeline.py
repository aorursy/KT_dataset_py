# Ignore warnings 
import warnings
warnings.filterwarnings('ignore')


# Data processing and analysis
import numpy as np
import pandas as pd
import math 
import re


# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


# Configure visualisations
%matplotlib inline
plt.style.use('fivethirtyeight')
sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)


# Classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb


# Data preprocessing :
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, scale, LabelEncoder, OneHotEncoder


# Modeling helper functions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV , KFold , cross_val_score



# Classification metrices
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report, precision_score,recall_score,f1_score 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Load train and Test set

%time

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
IDtest = test['PassengerId']
print('The shape of the training set : {} '.format(train.shape))
import pandas_profiling as pp
pp.ProfileReport(train)
train.head()
test.head()
train.info()
var1 = [col for col in train.columns if train[col].isnull().sum() != 0]

print(train[var1].isnull().sum())
train.describe()
# find categorical variables

categorical = [var for var in train.columns if train[var].dtype =='O']

print('There are {} categorical variables in training set.\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
# find numerical variables

numerical = [var for var in train.columns if train[var].dtype !='O']

print('There are {} numerical variables in training set.\n'.format(len(numerical)))

print('The numerical variables are :', numerical)
print('The shape of the test set : {} '.format(test.shape))
pp.ProfileReport(test)
test.head()
test.info()
var2 = [col for col in test.columns if test[col].isnull().sum() != 0]

print(test[var2].isnull().sum())
test.describe()
# find categorical variables

categorical = [var for var in test.columns if test[var].dtype =='O']

print('There are {} categorical variables in test set.\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
# find numerical variables

numerical = [var for var in test.columns if test[var].dtype !='O']

print('There are {} numerical variables in test set.\n'.format(len(numerical)))

print('The numerical variables are :', numerical)
# view missing values in training set
msno.matrix(train, figsize = (30,10))
train['Survived'].value_counts()
fig, ax = plt.subplots(figsize=(6,6))
graph = sns.countplot(ax=ax,x=train['Survived'], data = train, palette = 'PuBuGn_d')
graph.set_title('Distribution of people who survived', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
train.groupby('Survived')['Sex'].value_counts()
fig, ax = plt.subplots(figsize=(6,6))
graph = sns.countplot(ax=ax,x=train['Survived'], data = train, hue='Sex', palette = 'PuBuGn_d')
graph.set_title('Distribution of people who survived', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
females = train[train['Sex'] == 'female']
females.head()
females['Survived'].value_counts()/len(females)
males = train[train['Sex'] == 'male']
males.head()
males['Survived'].value_counts()/len(males)
# create the first of two pie-charts and set current axis
plt.figure(figsize=(8,6))
plt.subplot(1, 2, 1)   # (rows, columns, panel number)
labels1 = females['Survived'].value_counts().index
size1 = females['Survived'].value_counts()
colors1=['cyan','pink']
plt.pie(size1, labels = labels1, colors = colors1, shadow = True, autopct='%1.1f%%',startangle = 90)
plt.title('Percentage of females who survived', fontsize = 20)
plt.legend(['1:Survived', '0:Not Survived'], loc=0)
plt.show()

# create the second of two pie-charts and set current axis
plt.figure(figsize=(8,6))
plt.subplot(1, 2, 2)   # (rows, columns, panel number)
labels2 = males['Survived'].value_counts().index
size2 = males['Survived'].value_counts()
colors2=['pink','cyan']
plt.pie(size2, labels = labels2, colors = colors2, shadow = True, autopct='%1.1f%%',startangle = 90)
plt.title('Percentage of males who survived', fontsize = 20)
plt.legend(['0:Not Survived','1:Survived'])
plt.show()

train['Sex'].value_counts()
fig, ax = plt.subplots(figsize=(6,6))
graph = sns.countplot(ax=ax,x=train['Sex'], data=train, palette = 'bone')
graph.set_title('Distribution of sex among passengers', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
train['Sex'].value_counts()/len(train)
plt.figure(figsize=(8,6))
labels = train['Sex'].value_counts().index
size = train['Sex'].value_counts()
colors=['cyan','pink']
plt.pie(size, labels = labels, shadow = True, colors=colors, autopct='%1.1f%%',startangle = 90)
plt.title('Percentage distribution of sex among passengers', fontsize = 20)
plt.legend()
plt.show()
train.groupby('Pclass')['Sex'].value_counts()
fig, ax = plt.subplots(figsize=(8,6))
graph = sns.countplot(ax=ax,x=train['Pclass'], data=train, palette = 'bone')
graph.set_title('Number of people in different classes', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
fig, ax = plt.subplots(figsize=(8,6))
graph = sns.countplot(ax=ax,x=train['Pclass'], data=train, hue='Survived', palette = 'bone')
graph.set_title('Distribution of people segregated by survival', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
# percentage of survivors per class
sns.factorplot('Pclass', 'Survived', data = train)
fig, ax = plt.subplots(figsize=(8,6))
graph = sns.countplot(ax=ax,x=train['Embarked'], data=train, palette = 'bone')
graph.set_title('Number of people across different embarkment', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
fig, ax = plt.subplots(figsize=(8,6))
graph = sns.countplot(ax=ax,x=train['Embarked'], data=train, hue='Survived', palette = 'bone')
graph.set_title('Number of people across different embarkment', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
x = train['Age']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='g')
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.title('Age distribution of passengers', fontsize = 20)
plt.show()
train.drop(['Ticket', 'PassengerId'], axis = 1, inplace = True)
test.drop(['Ticket','PassengerId'], axis = 1, inplace = True)
# function to extract title from Name feature
def passenger_title(passenger):
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'
# extract title  
train['Title'] = train['Name'].apply(passenger_title)
test['Title'] = test['Name'].apply(passenger_title)
# fill missing age, with median from title segregation: funtion
def fill_age(passenger):
    
    # determine age by group 
    temp = train.groupby(train.Title).median()
    
    age, title = passenger
    
    if age == age:
        return age
    else:
        if title == 'Mr':
            return temp.Age['Mr']
        elif title == 'Miss':
            return temp.Age['Miss']
        elif title == ['Mrs']:
            return temp.Age['Mrs']
        elif title == 'Master':
            return temp.Age['Master']
        else:
            return temp.Age['Other']
# fill age according to title
train['Age'] = train[['Age', 'Title']].apply(fill_age, axis = 1)
test['Age'] = test[['Age', 'Title']].apply(fill_age, axis = 1)
# Remove column Name, it is not useful for predictions and we extracted the title already
train.drop('Name', axis = 1, inplace = True)
test.drop('Name', axis = 1, inplace = True)
# Remove column Title, it is not useful for predictions and we imputed the age already
train.drop('Title', axis = 1, inplace = True)
test.drop('Title', axis = 1, inplace = True)
def isNaN(num):
    return num != num # checks if cell is NaN
# get the first letter of cabin 
def first_letter_of_cabin(cabin):
    if not isNaN(cabin):
        return cabin[0]
    else:
        return 'Unknown'
train['Deck'] = train['Cabin'].apply(first_letter_of_cabin)
test['Deck'] = test['Cabin'].apply(first_letter_of_cabin)
# drop old variable Cabin
train.drop('Cabin', axis = 1, inplace = True)
test.drop('Cabin', axis = 1, inplace = True)
train["Embarked"].fillna("S", inplace = True)
test['Embarked'].fillna("S", inplace = True)
train.isnull().sum()
test.isnull().sum()
#we can replace missing value in fare by taking median of all fares of those passengers 
#who share 3rd Passenger class and Embarked from 'S' 
test['Fare'].fillna(test['Fare'].median(), inplace = True)

test.isnull().sum()
# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))


plt.subplot(1, 2, 1)
fig = train.boxplot(column='Age')
fig.set_title('')
fig.set_ylabel('Age')


plt.subplot(1, 2, 2)
fig = train.boxplot(column='Fare')
fig.set_title('')
fig.set_ylabel('Fare')

# find outliers in Age variable

IQR = train.Age.quantile(0.75) - train.Age.quantile(0.25)
Lower_fence = train.Age.quantile(0.25) - (IQR * 3)
Upper_fence = train.Age.quantile(0.75) + (IQR * 3)
print('Age outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=max(0, Lower_fence), upperboundary=Upper_fence))
# find outliers in Fare variable

IQR = train.Fare.quantile(0.75) - train.Fare.quantile(0.25)
Lower_fence = train.Fare.quantile(0.25) - (IQR * 3)
Upper_fence = train.Fare.quantile(0.75) + (IQR * 3)
print('Fare outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=max(0, Lower_fence), upperboundary=Upper_fence))
def max_value(df, variable, top):
    return np.where(df[variable]>top, top, df[variable])

for df in [train, test]:
    df['Age'] = max_value(df, 'Age', 81.0)
    df['Fare'] = max_value(df, 'Fare', 100.2688)
    
train.Age.max(), test.Age.max()
train.Fare.max(), test.Fare.max()
# label minors as child, and remaining people as female or male
def male_female_child(passenger):
    # take the age and sex
    age, sex = passenger
    
    # compare age, return child if under 16, otherwise leave sex
    if age < 16:
        return 'child'
    else:
        return sex
# new columns called person specifying if the person was female, male or child
train['Person'] = train[['Age', 'Sex']].apply(male_female_child, axis = 1)
test['Person'] = test[['Age', 'Sex']].apply(male_female_child, axis = 1)

# Number of male, female and children on board
train['Person'].value_counts()
# age segregated by class
fig = sns.FacetGrid(train, hue = 'Person', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)
fig.add_legend()
# age segregated by class
fig = sns.FacetGrid(train, hue = 'Pclass', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)
fig.add_legend()
sns.factorplot('Pclass', 'Survived', hue = 'Person', data = train)
def travel_alone(df):
    df['Alone'] = df.Parch + df.SibSp
    df['Alone'].loc[df['Alone'] > 0] = 'With Family'
    df['Alone'].loc[df['Alone'] == 0] = 'Alone'
    
    return df
train = travel_alone(train)
test = travel_alone(test)
# check how many passengers are travelling with family and alone
train['Alone'].value_counts()
fig, ax = plt.subplots(figsize=(6,6))
graph = sns.countplot(ax=ax,x=train['Alone'], data = train, palette = 'PuBuGn_d')
graph.set_title('Distribution of people travelling alone or with family', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
fig, ax = plt.subplots(figsize=(6,6))
graph = sns.countplot(ax=ax,x=train['Alone'], data = train, hue = 'Survived', palette = 'PuBuGn_d')
graph.set_title('Distribution of people travelling alone or with family', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
# percentage of survivors depending on traveling alone or with family
sns.factorplot('Alone', 'Survived', hue = 'Person', data = train)
train.head()
fig, ax = plt.subplots(figsize=(6,6))
graph = sns.countplot(ax=ax,x=train['Deck'], data = train[train.Deck != 'Unknown'], hue = 'Survived', palette = 'PuBuGn_d')
graph.set_title('Distribution of people on each deck', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
train.corr()['Survived']
corr=train.corr()#["Survived"]
plt.figure(figsize=(10, 10))
sns.heatmap(corr, vmax=.8, linewidths=0.01, square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features')
plt.show()
train.head()
train.drop('Sex', axis=1, inplace=True)
test.drop('Sex', axis=1, inplace=True)
train['Alone'] = pd.get_dummies(train['Alone'])
test['Alone'] = pd.get_dummies(test['Alone'])

labelenc=LabelEncoder()

categorical=['Embarked','Deck','Person']
for col in categorical:
    train[col]=labelenc.fit_transform(train[col])
    test[col]=labelenc.fit_transform(test[col])

train.head()
test.head()
train_cols = train.columns
test_cols = test.columns
scaler = StandardScaler()
train[['Age', 'Fare']] = scaler.fit_transform(train[['Age', 'Fare']])
test[['Age', 'Fare']] = scaler.transform(test[['Age', 'Fare']])
# Declare feature vector and target variable
X = train.drop(labels = ['Survived'],axis = 1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
names = ["Logistic Regression", "Nearest Neighbors", "Naive Bayes", "Linear SVM", "RBF SVM", 
         "Gaussian Process", "Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting", 
         "LDA", "QDA", "Neural Net", "LightGBM", "XGBoost" ]    


classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(5),
    GaussianNB(),
    SVC(kernel="linear", C=0.025),
    SVC(kernel = "rbf", gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    MLPClassifier(alpha=1, max_iter=1000),
    lgb.LGBMClassifier(),    
    xgb.XGBClassifier()
   ]

accuracy_scores = []

# iterate over classifiers and predict accuracy
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    score = round(score, 4)
    accuracy_scores.append(score)
    print(name ,' : ' , score)
classifiers_performance = pd.DataFrame({"Classifiers": names, "Accuracy Scores": accuracy_scores})
classifiers_performance
classifiers_performance.sort_values(by = 'Accuracy Scores' , ascending = False)[['Classifiers', 'Accuracy Scores']]
fig, ax = plt.subplots(figsize=(8,6))
x = classifiers_performance['Accuracy Scores']
y = classifiers_performance['Classifiers']
ax.barh(y, x, align='center', color='green')
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Accuracy Scores')
ax.set_ylabel('Classifiers', rotation=0)
ax.set_title('Classifier Accuracy Scores')
plt.show()
# instantiate the classifier with n_estimators = 100
clf = RandomForestClassifier(n_estimators=100, random_state=0)


# fit the classifier to the training set
clf.fit(X_train, y_train)
# view the feature scores
feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_scores
feature_scores.values
feature_scores.index
# Creating a seaborn bar plot to visualize feature scores
f, ax = plt.subplots(figsize=(8,6))
ax = sns.barplot(x=feature_scores.values, y=feature_scores.index, palette='spring')
ax.set_title("Visualize feature scores of the features")
ax.set_yticklabels(feature_scores.index)
ax.set_xlabel("Feature importance score")
ax.set_ylabel("Features")
plt.show()
# drop the least important feature from X_train, X_test and test set for further analysis
X1_train = X_train.drop(['Alone'], axis=1)
X1_test = X_test.drop(['Alone'], axis=1)
test = test.drop(['Alone'], axis=1)
accuracy_scores1 = []

# iterate over classifiers and predict accuracy
for name, clf in zip(names, classifiers):
    clf.fit(X1_train, y_train)
    score = clf.score(X1_test, y_test)
    score = round(score, 4)
    accuracy_scores1.append(score)
    print(name ,' : ' , score)
classifiers_performance1 = pd.DataFrame({"Classifiers": names, "Accuracy Scores": accuracy_scores, 
                                         "Accuracy Scores1": accuracy_scores1})
classifiers_performance1
# instantiate the XGBoost classifier
gpc_clf = GaussianProcessClassifier(1.0 * RBF(1.0))


# fit the classifier to the modified training set
gpc_clf.fit(X1_train, y_train)
# predict on the test set
y1_pred = gpc_clf.predict(X1_test)

# print the accuracy
print('Gaussian Process Classifier model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y1_pred)))
# print confusion-matrix

cm = confusion_matrix(y_test, y1_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
print(classification_report(y_test, y1_pred))
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))
# print precision score

precision = TP / float(TP + FP)

print('Precision : {0:0.4f}'.format(precision))
recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))
true_positive_rate = TP / float(TP + FN)

print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
false_positive_rate = FP / float(FP + TN)

print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))
# iterate over classifiers and calculate cross-validation score
for name, clf in zip(names, classifiers):
    scores = cross_val_score(clf, X1_train, y_train, cv = 10, scoring='accuracy')
    print(name , ':{:.4f}'.format(scores.mean()))
   
abc_params = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2]
             }

dtc_clf = DecisionTreeClassifier(random_state = 0, max_features = "auto", class_weight = "balanced", max_depth = None)

abc_clf = AdaBoostClassifier(base_estimator = dtc_clf)


abc_grid_search = GridSearchCV(estimator = abc_clf,  
                               param_grid = abc_params,
                               scoring = 'accuracy',
                               cv = 5,
                               verbose=0)


abc_grid_search.fit(X1_train, y_train)

# examine the best model

# best score achieved during the GridSearchCV
print('AdaBoost GridSearch CV best score : {:.4f}\n\n'.format(abc_grid_search.best_score_))

# print parameters that give the best results
print('AdaBoost Parameters that give the best results :','\n\n', (abc_grid_search.best_params_))

# print estimator that was chosen by the GridSearch
abc_best = abc_grid_search.best_estimator_
print('\n\nXGBoost Estimator that was chosen by the search :','\n\n', (abc_best))
lgb_clf = lgb.LGBMClassifier()


lgb_params={'learning_rate': [0.005],
    'num_leaves': [6,8,12,16],
    'objective' : ['binary'],
    'colsample_bytree' : [0.5, 0.6],
    'subsample' : [0.65,0.66],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.2,1.4],
    }


lgb_grid_search = GridSearchCV(estimator = lgb_clf,  
                               param_grid = lgb_params,
                               scoring = 'accuracy',
                               cv = 5,
                               verbose=0)


lgb_grid_search.fit(X1_train, y_train)

# examine the best model

# best score achieved during the GridSearchCV
print('LightGBM GridSearch CV best score : {:.4f}\n\n'.format(lgb_grid_search.best_score_))

# print parameters that give the best results
print('LightGBM Parameters that give the best results :','\n\n', (lgb_grid_search.best_params_))

# print estimator that was chosen by the GridSearch
lgb_best = lgb_grid_search.best_estimator_
print('\n\nLightGBM Estimator that was chosen by the search :','\n\n', (lgb_best))
gbc_clf = GradientBoostingClassifier()

gbc_params = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gbc_grid_search = GridSearchCV(estimator = gbc_clf, 
                               param_grid = gbc_params, 
                               scoring = "accuracy", 
                               cv = 5,
                               verbose = 0)

gbc_grid_search.fit(X1_train,y_train)

# examine the best model

# best score achieved during the GridSearchCV
print('Gradient Boosting GridSearch CV best score : {:.4f}\n\n'.format(gbc_grid_search.best_score_))

# print parameters that give the best results
print('Gradient Boosting Parameters that give the best results :','\n\n', (gbc_grid_search.best_params_))

# print estimator that was chosen by the GridSearch
gbc_best = gbc_grid_search.best_estimator_
print('\n\nGradient Boosting Estimator that was chosen by the search :','\n\n', (gbc_best))
votingC = VotingClassifier(estimators=[('abc', abc_best), ('lgb',lgb_best), ('gbc',gbc_best)], voting='soft')

votingC = votingC.fit(X1_train, y_train)
test_Survived = pd.Series(votingC.predict(test), name="Survived")

submission = pd.concat([IDtest,test_Survived],axis=1)


submission.to_csv("titanic_submission.csv", index=False)