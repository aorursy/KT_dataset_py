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
# import all the needed Libraries 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv('/kaggle/input/titanic-machine-learning-from-disaster/train.csv')
# looking into the training Dataset
# print first 5 rows of the train dataset
df.head(5)
df.shape
df.describe()
## big 'O' (not zero) 
df.describe(include=['O']) 
df.info()
df.isnull().sum()
df1= pd.read_csv('/kaggle/input/titanic-machine-learning-from-disaster/test.csv')
df1.shape
df1.head(5)
df1.info()
df1.isnull().sum()
sns.countplot(x='Survived',data=df,palette='RdBu_r');
survived = df[df['Survived']==0]
notsurvived =df[df['Survived']==1]
print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(df)*100.0))
print ("Not Survived: %i (%.1f%%)"%(len(notsurvived), float(len(notsurvived))/len(df)*100.0))
print ("Total: %i"%len(df))
sns.countplot(x='Sex',data=df,palette='RdBu_r')
sns.countplot(x='Survived', hue='Sex',data=df,palette='RdBu_r')
pd.crosstab(df.Survived,df.Sex,margins=True)
pd.crosstab(df.Survived,df.Sex,normalize='index')
sns.countplot(x='Survived',hue='Pclass',data=df,palette='RdBu_r')
pd.crosstab([df.Pclass, df.Sex], [df.Survived], margins=True)
df[(df.Sex == 'female') & (df.Pclass == 1) & (df.Survived == 0)]
df[(df.Sex == 'female') & (df.Pclass == 1) & (df.Survived == 1)].head(20)
df[(df.Cabin == 'C49') | (df.Cabin == 'C22 C26')]
df_baby = df[np.isfinite(df['Age'])]
df_baby.Age[(df_baby.Survived == 0)].min()
df_baby[(df_baby.Age < 1.0) & (df_baby.Survived == 0)]
df_baby[(df_baby.Age < 1.0) & (df_baby.Survived == 1)]
df[(df.Cabin == 'C49')]
df[(df.SibSp == 0) & (df.Parch == 0) & (df.Sex == 'female') ].head()
df[(df.Sex == 'female') & (df.Pclass == 2) & (df.Survived == 0)]
df[(df.Sex == 'female') & (df.Pclass == 2) & (df.Survived == 1)].head()
df[(df.Sex == 'female') & (df.Pclass == 2) & (df.Survived == 1)].head()
df[(df.Sex == 'female') & (df.Pclass == 3) & (df.Survived == 0)].head(20)
df[(df.Sex == 'female') & (df.Pclass == 3) & (df.Survived == 1)].head(20)
grid = sns.FacetGrid(df, col='Survived',size=4, aspect=1.8)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
#The cutoff
df[df['Survived']==0].max()
# Parch & Sibsp
pd.crosstab([df.Parch, df.SibSp], [df.Survived], margins=True)
# For visual
sns.countplot(x='SibSp',hue = 'Parch', data=df)
grid = sns.FacetGrid(df, col='Survived',size=4.2, aspect=1.6)
grid.map(plt.hist, 'Fare', alpha=.5, bins=40)
grid.add_legend()
df.isnull().sum().plot(kind='bar');
df.Embarked.unique()
df[~((df.Embarked == 'S') | (df.Embarked == 'C') | (df.Embarked == 'Q'))]
df[(df.Sex == 'female') & (df.Pclass == 1) & (df.Survived == 1)].Embarked.value_counts()
df[df['Ticket'] == '113572']
df.loc[61,'Embarked'] = 'C'
df.loc[829,'Embarked'] = 'C'
df.isnull().sum()
df1.isnull().sum() 
df1[~np.isfinite(df1['Fare'])]
#df is a large dataset and we search ther
df[(df.Pclass == 3) & (df.Embarked == 'S') & (df.SibSp == 0) & (df.Parch == 0) & (df.Sex == 'male') & (df.Age > 50.0)]
df1.loc[152, 'Fare'] = 7.0
df1.loc[152, :]
df2 = pd.concat([df,df1],axis = 0)
df2.shape
def cabin_travelled(cabin):
    if not cabin or pd.isnull(cabin):
        return 'Unknown'
    else:
        return cabin[0]
# Lets create a new column for the Cabins
df2['Cabin'] = df2['Cabin'].map(lambda x: cabin_travelled(x))
df2.Cabin.unique()
# Plot the result
df2.Cabin.value_counts().plot(kind='bar')
# Functions that returns the title from a name. All the name in the dataset has the format "Surname, Title. Name"
def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'
# Lets create a new column for the titles
df2['Title'] = df2['Name'].map(lambda x: get_title(x))
df2.Title.unique()
df2.Title.value_counts()
# Plot the result
df2.Title.value_counts().plot(kind='bar');
# Mr is widely distributed
df2.Age[df2.Title == 'Mr'].plot(kind = 'hist', bins = 40)
df2[df2.Title == 'Mr'].min()
# Mrs is widely distributed
df2.Age[df2.Title == 'Mrs'].plot(kind = 'hist', bins = 40)
df2[df2.Title == 'Mrs'].min()
# Miss is widely distributed
df2.Age[df2.Title == 'Miss'].plot(kind = 'hist', bins = 40)
# Master refered to kids below 12 yrs old  
df2.Age[df2.Title == 'Master'].plot(kind = 'hist', bins = 40);
df2[~((df2.Title == 'Mr') | (df2.Title == 'Mrs') | (df2.Title == 'Master') | (df2.Title == 'Miss'))] 
df2.drop(df2[df2.PassengerId == 767].index, inplace = True)
df2[df2.Title == 'Dr']
# Let us look at the dataset once again
df2.head()
df2.columns
# All can be done in one single step
sex = pd.get_dummies(df2['Sex'],drop_first=True)
# And so with Embarked
embarked = pd.get_dummies(df2['Embarked'],drop_first=True)
# And so with Title
title = pd.get_dummies(df2['Title'],drop_first=True)
# And so with Cabin
cabin = pd.get_dummies(df2['Cabin'], prefix='Cabin')
# And so with Class 
pclass = pd.get_dummies(df2['Pclass'], prefix='Pclass')
df2 = pd.concat([df2,sex,embarked,title,cabin, pclass],axis=1)
df2.drop(['PassengerId', 'Sex','Embarked','Name','Ticket', 'Title', 'Cabin', 'Pclass'],axis=1,inplace=True)
# Age available
df_train = df2[np.isfinite(df2['Age'])]
# Age not available
df_test = df2[~np.isfinite(df2['Age'])]
# Splitting labels
X_train_age_survived = df_train.Survived
X_train_age = df_train.drop(['Age','Survived'], axis=1)
y_train_age = df_train.Age

X_test_age_survived = df_test.Survived
X_test_age = df_test.drop(['Age','Survived'], axis=1)
y_test_age = df_test.Age
df_test
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train_age, y_train_age.ravel())
y_test_age = regressor.predict(X_test_age)
y_test_age = pd.Series(y_test_age)
y_test_age.index = X_test_age.index
# Train data with Age, Survived
df_train1 = pd.concat([X_train_age,y_train_age, X_train_age_survived],axis=1)
 # Test data with Age, Survived
df_test1 = pd.concat([X_test_age,y_test_age, X_test_age_survived],axis=1)
df_test1.columns
# Rename '0' to 'Age' column
df_test1.rename(columns={0: 'Age'}, inplace=True)
df_test1.columns
df_train2=pd.concat([df_train1[np.isfinite(df_train1['Survived'])],
                               df_test1[np.isfinite(df_test1['Survived'])]],axis=0) 
df_test2=pd.concat([df_train1[~np.isfinite(df_train1['Survived'])],
                               df_test1[~np.isfinite(df_test1['Survived'])]],axis=0) 
# And sort index
df_train2 = df_train2.sort_index()
df_test2 = df_test2.sort_index()
# Training data
X_train = df_train2.iloc[:,:-1]
y_train = df_train2.iloc[:,-1]

# Testing data
X_test = df_test2.iloc[:,:-1]
y_test = df_test2.iloc[:,-1]
# To store execution time of each algorithm fitting method. We will make a comparison later.
import time
from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression()
time_in = time.time()
log_clf.fit(X_train,y_train)
time_out = time.time()
# Execution time in secs
time_clf = (time_out - time_in)
# To make predicion we use it like
y_predict_log_clf = log_clf.predict(X_train)
# The logistic classification works on classiying the '0' & '1' based on probablility of max occurance of target variables
# How do I calculate the probability for each of the target labels 
# The predicted variables are classified on max of all probablility calculated
pred = log_clf.predict_proba(X_train)
# How do we know or model is better at prediction
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(accuracy_score(y_train, log_clf.predict(X_train)))
# One single parameter is good. 
# But if we want to know more on how did we arrive at this we have one more called 'classification report'
print(classification_report(y_train,log_clf.predict(X_train)))
# Confusion Matrix
print(confusion_matrix(y_train,log_clf.predict(X_train)))
# The result shown in diagonal are where actual and predicted values match and in offdiagonal the misclassification
# Cross-validation
# The idea now it to split the train data itself into multiple folds (K-fold) 
# and make themselves act like a train/test and we do this in a roundrobin system
# Then finally we average the accuracy (mean, variance) to understand its performance
from sklearn.model_selection import cross_val_score
print(cross_val_score(log_clf, X_train, y_train, cv=10, scoring='accuracy').mean())
print(cross_val_score(log_clf, X_train, y_train, cv=10, scoring='accuracy').std()) 
# Let us store the cross_val_score for future comparison
cols = ['Classifier','AccuracyScore','cross_val_score_mean','cross_val_score_std','Time_clf(sec)']
df_comparison = pd.DataFrame(columns=cols)

df_comparison.loc[len(df_comparison), :] = ['log_clf',
                  accuracy_score(y_train, log_clf.predict(X_train)),
                  cross_val_score(log_clf, X_train, y_train, cv=10, scoring='accuracy').mean(),
                  cross_val_score(log_clf, X_train, y_train, cv=10, scoring='accuracy').std(),
                  time_clf]
df_comparison.head()
# Ref: http://www.prolekare.cz/en/journal-of-czech-physicians-article/roc-analysis-and-the-use-of-cost-benefit-analysis-for-determination-of-the-optimal-cut-point-5403?confirm_rules=1
from sklearn.metrics import roc_curve, auc
#import matplotlib.pyplot as plt
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, log_clf.predict(X_train))
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
### Fitting K-NN to the Training set
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_knn = sc.fit_transform(X_train)
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
time_in = time.time()
knn_clf.fit(X_train_knn, y_train)  
time_out = time.time()
time_clf = (time_out - time_in)
y_predict_knn_clf = knn_clf.predict(X_train_knn)
# We define a function for evaluation and call it whenever necessary
def print_score(classifier_name, classifier, X_train, y_train, time_clf):
    print(accuracy_score(y_train, classifier.predict(X_train)))
    print(classification_report(y_train, classifier.predict(X_train)))
    print(confusion_matrix(y_train, classifier.predict(X_train)))
    print(cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy').mean())
    print(cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy').std()) 
    df_comparison.loc[len(df_comparison), :] = [classifier_name,
                  accuracy_score(y_train, classifier.predict(X_train)),
                  cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy').mean(),
                  cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy').std(),
                  time_clf]
# Again evaluating the parameters
print_score('knn_clf',knn_clf, X_train_knn, y_train, time_clf)
# Parameter Tuning
# This is done through GridSearchCV
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Enter the space to be searched
k_range = list(range(1, 31))
print(k_range)

# Create a parameter grid
param_grid = dict(n_neighbors=k_range)
print(param_grid) 
# Instantiate the grid
knn_grid = GridSearchCV(knn_clf, param_grid, cv=10, scoring='accuracy')

# Fit the grid with data
time_in = time.time()
knn_grid.fit(X_train_knn, y_train)
time_out = time.time()
time_clf = (time_out - time_in)
# Now the grid contains the list of scores for all the grid searched
# By default the grid picks up the best estimator on the space and we can 
# continue with the prediction
print(knn_grid.best_score_)
print(knn_grid.best_params_)
print(knn_grid.best_estimator_)
# We may again evaluate the performance  
print_score('knn_grid', knn_grid, X_train, y_train, time_clf)
# Scaling of the input data is necessary for the model
X_train_svc = sc.fit_transform(X_train)
from sklearn.svm import SVC
svc_clf = SVC(kernel = 'rbf', random_state = 0)
time_in = time.time()
svc_clf.fit(X_train_svc, y_train)
time_out = time.time()
time_clf = (time_out - time_in)
print_score('svc_clf', svc_clf, X_train_svc, y_train, time_clf)
# One can again choose between various options for parameter tuning like
# linear, polynomial and sigmoid
# SVM are good for higher dimensional problem. Although effective in higher dimension
# avoid using them in cases where no of samples is less than features describing them
# Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' 
# theorem with strong (naive) independence assumptions between the features.
# Naive Bayes classifiers are highly scalable
# Further it requires a small number of training data to estimate the parameters necessary for classification
# In simple terms find the probability of occurance of an event with given data
# Model probability of occurance of events and if >0.5 take it as '1' and <0.5 otherwise
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_naive = sc.fit_transform(X_train)
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
naive_clf = GaussianNB()
time_in = time.time()
naive_clf.fit(X_train_naive, y_train)
time_out = time.time()
time_clf = (time_out - time_in)
print_score('naive_clf', naive_clf, X_train_svc, y_train, time_clf)
# We move down the tree like graph of input parameters and split
# until 0/1 decision (target category) is clearly demarkable at the leaf nodes
# The condition of choosing the tree top input parameters and splitting is based on the theory
# of maximum entropy means '0/1' can occur 50/50% at best on splitting for better prediction.
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
dc_clf = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
time_in = time.time()
dc_clf.fit(X_train, y_train) 
time_out = time.time()
time_clf = (time_out - time_in)
print_score('dc_clf',dc_clf, X_train, y_train, time_clf)
#Much better than anything seen before
# The disadvantage includes overfitting with its train set and also when it meets 
# a new data of test set where variance with test model may be high.
# In order to overcome an ensemble of trees with reduced feature and sample 
# data are used to train the dataset which helps in increasing the model performance
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
time_in = time.time()
rf_clf.fit(X_train, y_train.ravel())
time_out = time.time()
time_clf = (time_out - time_in)
print_score('rf_clf',rf_clf, X_train, y_train, time_clf)
# We have an added advantage here. The feature importances
pd.Series(rf_clf.feature_importances_, index=X_train.columns).sort_values(ascending=False).plot(kind='bar', figsize=(12,6));
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier()
time_in = time.time()
ada_clf.fit(X_train, y_train.ravel())          
time_out = time.time()
time_clf = (time_out - time_in)
print_score('ada_clf', ada_clf, X_train, y_train, time_clf)
ada_rf_clf = AdaBoostClassifier(RandomForestClassifier())
time_in = time.time()
ada_rf_clf.fit(X_train, y_train.ravel()) 
time_out = time.time()
time_clf = (time_out - time_in)
print_score('ada_rf_clf',ada_rf_clf, X_train, y_train, time_clf)
from sklearn.ensemble import GradientBoostingClassifier
gbc_clf = GradientBoostingClassifier()
time_in = time.time()
gbc_clf.fit(X_train, y_train.ravel())
time_out = time.time()
time_clf = (time_out - time_in)
print_score('gbc_clf',gbc_clf, X_train, y_train, time_clf)
import xgboost as xgb
xgb_clf = xgb.XGBClassifier()
time_in = time.time()
xgb_clf.fit(X_train, y_train.ravel())
time_out = time.time()
time_clf = (time_out - time_in)
print_score('xgb_clf',xgb_clf, X_train, y_train, time_clf)
from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(estimators=[('dc_clf', dc_clf), ('rf_clf', rf_clf), ('ada_clf', ada_clf),
                                     ('ada_rf_clf', ada_rf_clf), ('gbc_clf', gbc_clf), ('xgb_clf', xgb_clf)], voting='soft')
time_in = time.time()
voting_clf = eclf.fit(X_train, y_train.ravel())
time_out = time.time()
time_clf = (time_out - time_in)
print_score('voting_clf',voting_clf, X_train, y_train, time_clf)
df_comparison
dims = (10, 8)
fig, ax = plt.subplots(figsize=dims)
sns.barplot(ax=ax, y='AccuracyScore',x='Classifier',data=df_comparison);
fig, ax = plt.subplots(figsize=dims)
sns.barplot(ax=ax, y='cross_val_score_mean',x='Classifier',data=df_comparison);
fig, ax = plt.subplots(figsize=dims)
sns.barplot(ax=ax, y='cross_val_score_std',x='Classifier',data=df_comparison);
fig, ax = plt.subplots(figsize=dims)
sns.barplot(ax=ax, y='Time_clf(sec)',x='Classifier',data=df_comparison)
y_predict = voting_clf.predict(X_test) 
# Concatenate our predicted value with test data
df_predict = pd.concat([df1, pd.Series(y_predict)], axis = 1)
df_predict.rename(columns={0: 'Survived'}, inplace=True)
df_predict[(df_predict.Sex == 'female') & (df_predict.Pclass == 1) & (df_predict.Survived == 0)]
df_predict[(df_predict.Sex == 'female') & (df_predict.Pclass == 1) & (df_predict.Survived == 1)].shape
df_predict[(df_predict.Age < 1.0) & (df_predict.Survived == 0)]
df_predict[(df_predict.Age > 75.0) & (df_predict.Survived == 0)]
df_predict[(df_predict.Age > 75.0) & (df_predict.Survived == 1)]
# Submit to Kaggle
Titanic_Kaggle_Submit = pd.DataFrame({
        "PassengerId": df_predict.PassengerId,
        "Survived": df_predict.Survived
    })

Titanic_Kaggle_Submit.PassengerId = Titanic_Kaggle_Submit.PassengerId.astype(int)
Titanic_Kaggle_Submit.Survived = Titanic_Kaggle_Submit.Survived.astype(int)

Titanic_Kaggle_Submit.to_csv("Titanic_Kaggle_Submit_voting_clf.csv", index=False)
