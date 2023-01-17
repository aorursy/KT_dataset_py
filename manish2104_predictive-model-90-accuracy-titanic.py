import numpy as np
import pandas as pd
import nltk
import plotly
import re
          
plotly.offline.init_notebook_mode() # run at the start of every notebook
import cufflinks as cf

cf.go_offline()
cf.getThemes()
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
%matplotlib inline
import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from IPython.display import display

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
full_data = [df_train,df_test]
df_train.info()
# Function to calculate no. of null values with percentage in the dataframe
def null_values(DataFrame_Name):
    
    sum_null = DataFrame_Name.isnull().sum()
    total_count = DataFrame_Name.isnull().count()
    percent_nullvalues = sum_null/total_count * 100
    df_null = pd.DataFrame()
    df_null['Total_values'] = total_count
    df_null['Null_Count'] = sum_null
    df_null['Percent'] = percent_nullvalues
    df_null = df_null.sort_values(by='Null_Count',ascending = False)

    return(df_null)
null_values(df_train)
null_values(df_test)
df_train.describe()
df_train.head(5)
## get the most important variables. 
corr = df_train.corr()**2
corr.Survived.sort_values(ascending=False)
## heatmeap to see the correlation between features. 
# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(df_train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize = (15,12))
sns.heatmap(df_train.corr(), 
            annot=True,
            mask = mask,
            cmap = 'RdBu_r',
            linewidths=0.1, 
            linecolor='white',
            vmax = .9,
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20);
# Lets start with Pclass column - Already an integer - good
# Lets check the impact of this column on the survived column in the train dataset.
# We will calculate mean of survived people in each class - This will tell us how many survived out of total for each class
df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean()
sns.barplot('Pclass','Survived', data=df_train)
# Kernel Density Plot
fig = plt.figure(figsize=(15,8),)
## I have included to different ways to code a plot below, choose the one that suites you. 
ax=sns.kdeplot(df_train.Pclass[df_train.Survived == 0] , 
               color='gray',
               shade=True,
               label='not survived')
ax=sns.kdeplot(df_train.loc[(df_train['Survived'] == 1),'Pclass'] , 
               color='g',
               shade=True, 
               label='survived')
plt.title('Passenger Class Distribution - Survived vs Non-Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Passenger Class", fontsize = 15)
## Converting xticks into words for better understanding
labels = ['Upper', 'Middle', 'Lower']
plt.xticks(sorted(df_train.Pclass.unique()), labels);
females=df_train['Sex'].apply(lambda x: x.count('female')).sum()
print('Total males=',891-females)
print('Total females=',females)
# Now lets focus on the Sex column and evaluate its impact on the survived column
df_train[['Sex','Survived']].groupby(['Sex'],as_index = False).mean()
sns.barplot(x='Sex', y='Survived', data=df_train)
df_train[['Embarked','Survived']].groupby(['Embarked'],as_index = False).mean()
sns.barplot(x='Embarked', y='Survived', data=df_train)
df_train['Family_members'] = df_train['SibSp'] + df_train['Parch']
df_test['Family_members'] = df_test['SibSp'] + df_test['Parch']
df_train[['Family_members','Survived']].groupby(['Family_members'],as_index=False).mean()

sns.barplot(x='Family_members', y='Survived', data=df_train)
df_train = df_train.drop(['PassengerId'],axis=1)
#df_test = df_test.drop(['PassengerId'],axis=1)
full_data = [df_train,df_test]

df_train = df_train.drop(['Cabin','Ticket'],axis=1)

df_test = df_test.drop(['Cabin','Ticket'],axis=1)
full_data = [df_train,df_test]
for dataset in full_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(df_train['Title'], df_train['Sex'])
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
sns.barplot(x='Title', y='Survived', data=df_train)
df_train
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,LabelBinarizer
#cat_features = df_train['Title']
#encoder = LabelBinarizer()
#new_cat_features = encoder.fit_transform(cat_features)
#new_cat_features

#pd.get_dummies(df_train, columns=['Title'], prefix=['Title'])
df_train = df_train.drop(['Name'],axis = 1)
df_test = df_test.drop(['Name'],axis = 1)


df_train[['Title','Age']].groupby(['Title'],as_index = False).mean().sort_values(by='Age')
Mean_Age = df_train[['Title','Age']].groupby(['Title'],as_index = False).mean().sort_values(by='Age')
sns.barplot(x='Title', y='Age', data=Mean_Age)
df_train['Age'] = df_train['Age'].fillna(-1)
df_test['Age'] = df_test['Age'].fillna(-1)  
full_data = [df_train,df_test]

for dataset in full_data:
    
    dataset.loc[(dataset['Age'] == -1) &(dataset['Title'] == 'Master'), 'Age'] = 4.57
    dataset.loc[(dataset['Age'] == -1) &(dataset['Title'] == 'Miss'), 'Age'] = 21.84
    dataset.loc[(dataset['Age'] == -1) &(dataset['Title'] == 'Mr'), 'Age'] = 32.36
    dataset.loc[(dataset['Age'] == -1) &(dataset['Title'] == 'Mrs'), 'Age'] = 35.78
    dataset.loc[(dataset['Age'] == -1) &(dataset['Title'] == 'Rare'), 'Age'] = 45.54
    dataset['Age'] = dataset['Age'].astype(int)   
    
full_data = [df_train, df_test]
for dataset in full_data:
    
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 7
df_train[['Sex','Age','Survived']].groupby(['Sex','Age'],as_index=False).mean()
agesexsurv = df_train[['Sex','Age','Survived']].groupby(['Sex','Age'],as_index=False).mean()
sns.factorplot('Age','Survived','Sex', data=agesexsurv
                ,aspect=3,kind='bar')
plt.suptitle('AgeBand,Sex vs Survived')
full_data = [df_train, df_test]
for dataset in full_data:
    
    dataset.loc[ dataset['Family_members'] == 0, 'Family_members_Band'] = 0
    dataset.loc[(dataset['Family_members'] == 1)|(dataset['Family_members'] == 2),'Family_members_Band'] = 1
    dataset.loc[ dataset['Family_members'] == 3, 'Family_members_Band'] = 2
    dataset.loc[(dataset['Family_members'] == 4)|(dataset['Family_members'] == 5),'Family_members_Band'] = 3
    dataset.loc[ dataset['Family_members'] == 6, 'Family_members_Band'] = 4
    dataset.loc[(dataset['Family_members'] == 7)|(dataset['Family_members'] == 10),'Family_members_Band'] = 5
    dataset['Family_members_Band'] = dataset['Family_members_Band'].astype(int)
df_train['FareBand'] = pd.qcut(df_train['Fare'], 4)
df_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
FarePlot = df_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand')
sns.barplot(x='FareBand', y='Survived', data=FarePlot)
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].dropna().mean()) # df_test has one null value
full_data = [df_train,df_test]
for dataset in full_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare_Band'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare_Band'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare_Band'] = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare_Band'] = 3
    dataset['Fare_Band'] = dataset['Fare_Band'].astype(int)
sns.factorplot('Fare_Band','Survived','Sex', data=df_train
                ,aspect=3,kind='bar')
plt.suptitle('FareBand,Sex vs Survived')
most_frequent = df_train['Embarked'].mode()[0]
most_frequent
full_data = [df_train,df_test]
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna(most_frequent)
    
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
embarkedgraph = df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Embarked',y='Survived',data=embarkedgraph)

df_train = df_train.drop(['SibSp','Parch','Fare','Family_members','FareBand'],axis = 1)

df_test = df_test.drop(['SibSp','Parch','Fare','Family_members'],axis = 1)

X_train = pd.get_dummies(df_train, columns=['Pclass','Sex','Age','Embarked','Title','Family_members_Band','Fare_Band'], prefix=['Pclass','Sex'
                                                                ,'Age','Embarked','Title','Family_members_Band','Fare_Band'])
Y_train = X_train['Survived']
X_train = X_train.drop('Survived', axis=1)
X_train.shape
X_test = pd.get_dummies(df_test, columns=['Pclass','Sex','Age','Embarked','Title','Family_members_Band','Fare_Band'], prefix=['Pclass','Sex','Age','Embarked','Title','Family_members_Band','Fare_Band'])
X_test.shape
X_test=X_test.drop(['PassengerId'],axis=1)
# stochastic gradient descent (SGD) learning
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)

sgd.score(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)


print(round(acc_sgd,2,), "%")
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(round(acc_log,2,), "%")
# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(round(acc_knn,2,), "%")
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print(round(acc_gaussian,2,), "%")
# Perceptron
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print(round(acc_perceptron,2,), "%")
# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(round(acc_linear_svc,2,), "%")
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(round(acc_decision_tree,2,), "%")
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)

result_df.head(9)
bestmodelgraph = result_df.head(9)
ax = sns.factorplot("Model", y="Score", data=bestmodelgraph,
                palette='Blues_d',aspect=3.5,kind='bar')
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")

    plt.legend(loc="best")
    return plt
# Plot learning curves
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

title = "Learning Curves (Random Forest)"
cv = 10
plot_learning_curve(rf, title, X_train, Y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=1);
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False)
importances_most = importances.head(10) # 10 most important features
axes = sns.factorplot('feature','importance', 
                      data=importances_most, aspect = 4, )
importances_least = importances.tail(10) # least 10 important features
axes = sns.factorplot('feature','importance', 
                      data=importances_least, aspect = 4,)
# Random Forest , Testing with oob score

random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")
print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
# Random Forest
random_forest = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)
conf_mat = confusion_matrix(Y_train, predictions)
TP = conf_mat[0][0]
FP = conf_mat[0][1]
FN = conf_mat[1][0]
TN = conf_mat[1][1]

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print('Sensitivity, hit rate, recall, or true positive rate=',TPR)

# Specificity or true negative rate
TNR = TN/(TN+FP) 
print('Specificity or true negative rate=',TNR)

# Precision or positive predictive value
PPV = TP/(TP+FP)
print('Precision or positive predictive value=',PPV)

# Negative predictive value
NPV = TN/(TN+FN)
print('Negative predictive value=',NPV)

# Fall out or false positive rate
FPR = FP/(FP+TN)
print('Fall out or false positive rate=',FPR)

# False negative rate
FNR = FN/(TP+FN)
print('False negative rate=',FNR)

# False discovery rate
FDR = FP/(TP+FP)
print('False discovery rate=',FDR)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print('Overall accuracy=',ACC)
positives = pd.DataFrame({
    'Factor': ['True Positives', 'False Positives', ],
    'Score': [TP, FP]})

sns.barplot(x='Factor',y='Score',data=positives)
negatives = pd.DataFrame({
    'Factor':['True Negative', 'False Negative'],
    'Score':[TN, FN]
})

sns.barplot(x='Factor',y='Score',data=negatives)
from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(Y_train, predictions))
print("Recall:",recall_score(Y_train, predictions))
from sklearn.metrics import f1_score
print('F1score',f1_score(Y_train, predictions))

from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = random_forest.predict_proba(X_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(Y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()
def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g--", linewidth=2.5)
    plt.ylabel("recall", fontsize=19)
    plt.xlabel("precision", fontsize=19)
    plt.axis([0, 1.5, 0, 1.5])

plt.figure(figsize=(14, 7))
plot_precision_vs_recall(precision, recall)
plt.show()
from sklearn.metrics import roc_curve
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()
from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(Y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_prediction
    })
submission.to_csv('submission.csv', index=False)