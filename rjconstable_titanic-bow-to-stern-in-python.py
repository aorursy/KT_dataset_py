# Load dependencies

import pandas as pd    # data cleaning and preparation
import numpy as np     # arrays, linear algebra 
import matplotlib.pyplot as plt  #visualisation libraries
import seaborn as sns    #pretty plots by default
import re              # regular expressions for pattern matching in strings

sns.set(style='whitegrid', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=2)
%matplotlib inline
#Read in the titanic survival dataset CSV files

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
#Take a look at the training data frame 
df_train.head()
#Use df.info() to get a count of the non-null rows in each column. We can also see the data type of each column
df_train.info()
#Use df.describe() to see a summary statistics table for the dataframe

df_train.describe()
# More women surived than men
sns.factorplot(x='Sex', col='Survived', kind='count', data=df_train);
# First class passengers > second class passengers > third class passengers, in percentage survival
sns.factorplot('Pclass','Survived', data=df_train)
# The bulk of the dead were men in Pclass 3
pd.crosstab([df_train.Sex, df_train.Survived], df_train.Pclass, margins=True).style.background_gradient(cmap='winter_r')
# More people embarked at S than the other ports and the majority of those embarking at S died
sns.factorplot(x='Survived', col='Embarked', kind='count', data=df_train);
# Most of the men who died were in their prime, between 18-40
grid = sns.FacetGrid(df_train, col='Survived', row='Sex', size=3.5, aspect=1.6, hue = 'Survived')
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
# df.corr() calculates Pearsons correlation coefficient by default for each feature pair and target-feature pair
corr = df_train.corr()
sns.heatmap(corr, cmap = 'RdBu_r', center = 0)
# Drop the passenger ID and Ticket columns, they're unlikely to have any predictive utility

df_train = df_train.drop(['PassengerId', 'Ticket'], axis = 1)
df_test = df_test.drop(['Ticket'], axis = 1)
df_train = df_train.fillna(method='ffill')
df_test = df_test.fillna(method='ffill')
df_train.Sex = df_train.Sex.map({'male':0, 'female':1}).astype(int)
df_test.Sex = df_test.Sex.map({'male':0, 'female':1}).astype(int)
df_train.Sex.value_counts()
data = [df_train, df_test]

for dataset in data:
    dataset.Embarked = dataset.Embarked.map({'S':0, 'C':1, 'Q':2}).astype(int)
plt.subplots(figsize=(10,7))
plt.xticks(np.arange(min(df_train.Age), max(df_train.Age)+1, 10.0))
sns.distplot(df_train.Age)
data = [df_train, df_test]

for df in data:
    df['Age_bin']=np.nan
    for i in range(8,0,-1):
        df.loc[df['Age'] <= i*10, 'Age_bin'] = i
data = [df_train, df_test]

for df in data:
    df_train.Age_bin = df_train.Age_bin.astype(int)
    df_test.Age_bin = df_test.Age_bin.astype(int)
sns.distplot(df_train.Age_bin)
# Plot the Fares to take a look at their distributions
plt.subplots(figsize=(30,10))
plt.xticks(np.arange(min(df_train.Fare), max(df_train.Fare)+1, 10.0))
sns.distplot(df_train.Fare)
# Based on the distribution of the Fares, assign them to the following bins corresponding to roughly each peak in 
# the distplot kde profile (gaussian kernel density estimate) in an attempt to capture the distribution
data = [df_train, df_test]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 5, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 5) & (dataset['Fare'] <= 10), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10) & (dataset['Fare'] <= 20), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 20) & (dataset['Fare'] <= 40), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 40) & (dataset['Fare'] <= 99), 'Fare']   = 4
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 180), 'Fare']   = 5
    dataset.loc[(dataset['Fare'] > 180) & (dataset['Fare'] <= 280), 'Fare']   = 6
    dataset.loc[ dataset['Fare'] > 280, 'Fare'] = 7
    dataset['Fare'] = dataset['Fare'].astype(int)
data = [df_train, df_test]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # use the regex ' ([A-Za-z]+)\.' to extract the titles from the names
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Assign rare titles as Rare and convert Mme and Ms to the more common Mrs and Miss 
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # fill NaN with 0 to keep the dimensions the same as the other features
    dataset['Title'] = dataset['Title'].fillna(0)

# Extract the Deck feature from the Cabin column    
    
deck = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5,'F':6,'G':7,'U':8}    
    
for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna('U0')
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile('([a-zA-Z]+)').search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)
data = [df_train, df_test]
for dataset in data:
    dataset['Relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['Relatives']>0, 'not_alone'] = 0
    dataset.loc[dataset['Relatives']== 0 , 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
data = [df_train, df_test]

for dataset in data:
    dataset['Male_P3'] = (dataset['Sex'] == 0) & (dataset['Pclass'] == 3).astype(int)

data = [df_train, df_test]
for dataset in data:
    dataset['Age_Class']= dataset['Age_bin']* dataset['Pclass']
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['Relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
# Now drop the old continuous or text features which aren't required
data = [df_train, df_test]

for dataset in data:
    dataset.drop(['Age','Name', 'Cabin'],axis=1,inplace=True)
plt.subplots(figsize=(20,10))
corr = df_train.corr()
sns.heatmap(corr, cmap = 'RdBu_r', center = 0)
# Prepare dataframes for sklearn, so that the feature columns and target label are in separate dataframes
X_training = df_train.drop('Survived', axis = 1).copy()
y_training = df_train.Survived
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, StratifiedShuffleSplit

from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(df_train.drop('Survived',axis=1),
                                                    df_train['Survived'], test_size=0.2, random_state=2)
rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)
rfc_prediction = rfc.predict(X_test) 
accuracy_score(rfc_prediction, y_test)
# stratified shuffle split of the data into 10 folds, with 20% of the data used in testing
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=2)
random_state=2

# store all the classifiers in a list
classifiers=[]
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(XGBClassifier(random_state=random_state))

# store the cross validation accuracy results in a list
cv_results=[]
for classifier in classifiers:
    cv_results.append(cross_val_score(estimator=classifier,X=X_training,y=y_training,
                                      cv=sss,scoring='accuracy', n_jobs=-1))

# store the mean accuracy and standard deviation of the accuracy for each cross validation fold in lists    
cv_means=[]
cv_std=[]
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

# Combine the results lists in a dataframe
cv_results=pd.DataFrame({'CV_mean':cv_means , 'CV_std':cv_std , 'Algorithm':['SVC','DTC','RFC','KNN','LR',
                                                                             'ADA','XT','GBC','XGB']})
# sort the results by score
cv_results = cv_results.sort_values('CV_mean',ascending=False)

# plot the values for swift visual assessment of classifier performance
plt.subplots(figsize=(8,5))
sns.barplot('Algorithm','CV_mean',data=cv_results)
cv_results
xgbc = XGBClassifier(random_state=random_state)
xgbc.fit(X_training, y_training)
scores = cross_val_score(xgbc, X_training, y_training, cv=sss, scoring = "accuracy")

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

importances = pd.DataFrame({'feature':X_training.columns,'importance':np.round(xgbc.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.plot.bar(figsize=(10,5))
X_training = X_training.drop(['not_alone', 'Male_P3','Parch'], axis=1)
xgbc = XGBClassifier(random_state=random_state)
xgbc.fit(X_training, y_training)
scores = cross_val_score(xgbc, X_training, y_training, cv=sss, scoring = "accuracy")

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

# GridSearch CV automatically tests a series of model parameters through the cross validation specified and outputs
# the parameters which generated the best cross validation score

params = {
          'max_depth' : range(3,10,1),       # maximum depth of the trees - deeper = more likely to overfit
          'min_child_weight' : range(1,5,1), # minimum child weight - The lower the more likely to overfit
          'gamma' : [1,2,3,4],               # gamma - regularisation parameter
         }

xgbc = XGBClassifier()

clf = GridSearchCV(estimator=xgbc, param_grid=params, cv = sss, n_jobs=-1)

clf.fit(X_training, y_training)

clf.best_params_
xgbc = XGBClassifier(random_state=random_state, gamma = 2, max_depth = 8, min_child_weight = 2)
xgbc.fit(X_training, y_training)
scores = cross_val_score(xgbc, X_training, y_training, cv=sss, scoring = "accuracy")

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
train_sizes, train_scores, test_scores = learning_curve(xgbc, X_training, y_training, n_jobs=-1, cv=sss, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

# calculate mean and standard deviation for training and testing scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# setup the plot
plt.figure(figsize = (8,5))
plt.title('XGBoost classifier')
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.plot([1,760],[1,1], linewidth=2, color = 'r')

plt.grid()
    
# shade the area +/- one standard deviation of the mean scores
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="b")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
# plot the mean training and test scores vs training set size
plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.ylim(-.1,1.1)
from sklearn.model_selection import cross_val_predict
# from pandas_ml import ConfusionMatrix
predictions = cross_val_predict(xgbc, X_training, y_training, cv=10)
# cm = ConfusionMatrix(y_training, predictions)
# cm.plot()
from sklearn.metrics import f1_score
f1_score(y_training, predictions)
from sklearn.metrics import precision_recall_curve

# generate predictions from the training data
y_scores = xgbc.predict_proba(X_training)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(y_training, y_scores)
def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g", linewidth=2)
    plt.ylabel("Recall")
    plt.xlabel("Precision")
    plt.axis([0, 1.5, 0, 1.5])
    plt.xlim(0,1)
    plt.ylim(0,1.1)
    
plt.figure(figsize=(8, 5))
plot_precision_vs_recall(precision, recall)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="Precision", linewidth=2)
    plt.plot(threshold, recall[:-1], "b", label="Recall", linewidth=2)
    plt.xlabel("Threshold")
    plt.legend(loc="upper right")
    plt.ylim([0, 1])

plt.figure(figsize=(8, 5))
plot_precision_and_recall(precision, recall, threshold)
from sklearn.metrics import roc_curve

# calculate rates of false +ve and true +ve predictions
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_training, y_scores)

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=2)
    plt.plot([0, 1], [1, 1], 'g', linewidth=1.5)
    plt.plot([0, 0], [1, 0], 'g', linewidth=1.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim(0,1.02)
    plt.xlim(-0.005,1)

plt.figure(figsize=(8, 5))
plot_roc_curve(false_positive_rate, true_positive_rate)
from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(y_training, y_scores)
print("ROC-AUC Score:", r_a_score)
submission_testing = df_test.drop(['PassengerId','not_alone', 'Male_P3','Parch'], axis = 1).copy()
Y_prediction = xgbc.predict(submission_testing)
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_prediction
    })
submission.to_csv('submission.csv', index=False)