#importing the necessary libraries


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import RFE


#Algorithms

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
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import graphviz 
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
import graphviz
import pydot

#Accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#import xgboost as xgb
#from xgboost import XGBClassifier
#from imblearn.over_sampling import SMOTE
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.metrics import confusion_matrix
bank_df = pd.read_csv('../input/bank-additional-full.csv', sep = ';')
bank_df.head()


# checking the shape of the dataframe

bank_df.shape
bank_df.info()
sns.countplot(x='marital',hue='y',data=bank_df)
# Kernel Density Plot
fig = plt.figure(figsize=(15,4),)
ax=sns.kdeplot(bank_df.loc[(bank_df['y'] == 'yes'),'duration'] , color='b',shade=True,label='Yes')
ax=sns.kdeplot(bank_df.loc[(bank_df['y'] == 'no'),'duration'] , color='r',shade=True, label='No')
ax.set(xlabel='Duration Distribution', ylabel='Duration of Call')
plt.title('Duration V.S. Term Deposit')

sns.countplot(x='month',hue='y',data=bank_df)
sns.countplot(x='day_of_week',hue='y',data=bank_df)
sns.countplot(x='poutcome',hue='y',data=bank_df)
client_df = bank_df.iloc[: , 0:7]
client_df.head()

fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'job', data = client_df)
ax.set_xlabel('Job', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Job Count Distribution', fontsize=15)
ax.tick_params(labelsize=15)


fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'marital', data = client_df)
ax.set_xlabel('Marital status', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Marriage Count Distribution', fontsize=15)
ax.tick_params(labelsize=15)


fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'education', data = client_df)
ax.set_xlabel('Education', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Education Count Distribution', fontsize=15)
ax.tick_params(labelsize=15)

# Is credit in default ?
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (20,8))
sns.countplot(x = 'default', data = client_df, ax = ax1, order = ['no', 'unknown', 'yes'])
ax1.set_title('Any Defaults ?', fontsize=15)
ax1.set_xlabel('')
ax1.set_ylabel('Count', fontsize=15)
ax1.tick_params(labelsize=15)

# Has housing loan ?
sns.countplot(x = 'housing', data = client_df, ax = ax2, order = ['no', 'unknown', 'yes'])
ax2.set_title('Has Housing Loan ?', fontsize=15)
ax2.set_xlabel('')
ax2.set_ylabel('Count', fontsize=15)
ax2.tick_params(labelsize=15)

# Has Personal loan ?
sns.countplot(x = 'loan', data = client_df, ax = ax3, order = ['no', 'unknown', 'yes'])
ax3.set_title('Any Personal Loan ?', fontsize=15)
ax3.set_xlabel('')
ax3.set_ylabel('Count', fontsize=15)
ax3.tick_params(labelsize=15)

plt.subplots_adjust(wspace=0.5)

fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'age', data = client_df)
ax.set_xlabel('Age', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Age Wise Distribution', fontsize=15)
sns.despine()

fig, (ax1) = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
sns.boxplot(x = 'age', data = client_df, orient = 'v', ax = ax1)
ax1.set_xlabel('People Age', fontsize=15)
ax1.set_ylabel('Age', fontsize=15)
ax1.set_title('Age Distribution', fontsize=15)
ax1.tick_params(labelsize=15)

# Calculate the outliers using:

  # Interquartile range, IQR = Q3 - Q1
  # lower 1.5*IQR whisker = Q1 - 1.5 * IQR 
  # Upper 1.5*IQR whisker = Q3 + 1.5 * IQR
 
Q1=client_df['age'].quantile(q = 0.25)
Q2=client_df['age'].quantile(q = 0.50)
Q3=client_df['age'].quantile(q = 0.75)
Q4=client_df['age'].quantile(q = 1.00)                        

IQR= Q3-Q1

print('1st Quartile: ', Q1)
print('2nd Quartile: ', Q2)
print('3rd Quartile: ', Q3)
print('4th Quartile: ', Q4)
print('IQR: ',IQR)


print('Ages above: ', Q3 + 1.5*(IQR), 'are outliers')
print('Ages below: ', Q1 - 1.5*(IQR), 'are outliers')


# checking other details of age
client_df['age'].describe()
# functions to create bucketing in age

def age(df):
    df.loc[df['age'] <= 32, 'age'] = 1
    df.loc[(df['age'] > 32) & (df['age'] <= 47), 'age'] = 2
    df.loc[(df['age'] > 47) & (df['age'] <= 70), 'age'] = 3
    df.loc[(df['age'] > 70), 'age']=4
           
    return df

age(client_df);
# converting categorical columns to numerical values
"""
client_df['job'].replace(['housemaid' , 'services' , 'admin.' , 'blue-collar' , 'technician', 'retired' , 'management', 'unemployed', 'self-employed', 'unknown' , 'entrepreneur', 'student'] , [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace=True)


client_df['education'].replace(['basic.4y' , 'high.school', 'basic.6y', 'basic.9y', 'professional.course', 'unknown' , 'university.degree' , 'illiterate'], [1, 2, 3, 4, 5, 6, 7, 8], inplace=True)

client_df['marital'].replace(['married', 'single', 'divorced', 'unknown'], [1, 2, 3, 4], inplace=True)

client_df['default'].replace(['yes', 'no', 'unknown'],[1, 2, 3], inplace=True)

client_df['housing'].replace(['yes', 'no', 'unknown'],[1, 2, 3], inplace=True)

client_df['loan'].replace(['yes', 'no', 'unknown'],[1, 2, 3], inplace=True)
"""

client_df=pd.get_dummies(client_df)
client_df.head()
# Creating seperate datasets for marketing related data
marketing_df = bank_df.iloc[: , 7:15]
marketing_df.head()


fig, ax = plt.subplots()
fig.set_size_inches(4, 4)
sns.countplot(x = 'contact', data = marketing_df)
ax.set_xlabel('Contact Mode', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Contact Count Distribution', fontsize=15)
ax.tick_params(labelsize=15)


fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
sns.countplot(x = 'month', data = marketing_df)
ax.set_xlabel('Month wise campaigns', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Month Count Distribution', fontsize=15)
ax.tick_params(labelsize=15)


fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
sns.countplot(x = 'day_of_week', data = marketing_df)
ax.set_xlabel('Week wise campaigns', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Week Count Distribution', fontsize=15)
ax.tick_params(labelsize=15)

# checking the duration column

fig, (ax1) = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
sns.boxplot(x = 'duration', data = marketing_df, orient = 'h', ax = ax1)
ax1.set_xlabel('Call duration', fontsize=15)
ax1.set_ylabel('Duration', fontsize=15)
ax1.set_title('Call duration Distribution', fontsize=15)
ax1.tick_params(labelsize=15)

# finding the outliers

Q1=marketing_df['duration'].quantile(q = 0.25)
Q2=marketing_df['duration'].quantile(q = 0.50)
Q3=marketing_df['duration'].quantile(q = 0.75)
Q4=marketing_df['duration'].quantile(q = 1.00)                        

IQR= Q3-Q1

print('1st Quartile: ', Q1)
print('2nd Quartile: ', Q2)
print('3rd Quartile: ', Q3)
print('4th Quartile: ', Q4)
print('IQR: ',IQR)


print('Duration above: ', Q3 + 1.5*(IQR), 'are outliers')
print('Duration below: ', Q1 - 1.5*(IQR), 'are outliers')


def duration(df):

    df.loc[df['duration'] <= 102, 'duration'] = 1
    df.loc[(df['duration'] > 102) & (df['duration'] <= 180)  , 'duration']    = 2
    df.loc[(df['duration'] > 180) & (df['duration'] <= 319)  , 'duration']   = 3
    df.loc[(df['duration'] > 319) & (df['duration'] <= 644.5), 'duration'] = 4
    df.loc[df['duration']  > 644.5, 'duration'] = 5

    return df

duration(marketing_df).head()

d_mons = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 
    'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10,
    'nov':11, 'dec':12}

marketing_df.month=marketing_df.month.map(d_mons)

# converting datatype to int
marketing_df['month'] =marketing_df['month'].astype(str).astype(int)
corr=marketing_df.corr()
corr
sns.set_context("notebook",font_scale = 1.0, rc = {"lines.linewidth":2.5})
plt.figure(figsize = (13,7))
a = sns.heatmap(corr, annot = True, fmt = ".2f")
#marketing_df['contact']=pd.get_dummies(marketing_df['contact'])
#marketing_df['poutcome']=pd.get_dummies(marketing_df['poutcome'])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
marketing_df['contact'] = le.fit_transform(marketing_df['contact'])
marketing_df['poutcome'] = le.fit_transform(marketing_df['poutcome'])
#marketing_df= marketing_df.loc[:, marketing_df.columns != 'day_of_week']
#marketing_df= marketing_df.loc[:, marketing_df.columns != 'previous']
d_week = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5, 
    'sat':6, 'sun':7}

marketing_df.day_of_week=marketing_df.day_of_week.map(d_week)
# converting datatype to int

marketing_df['contact'] =marketing_df['contact'].astype(str).astype(int)
marketing_df['month'] =marketing_df['month'].astype(str).astype(int)
marketing_df['day_of_week'] =marketing_df['day_of_week'].astype(str).astype(int)
marketing_df['poutcome'] =marketing_df['poutcome'].astype(str).astype(int)
marketing_df.head()
# Slicing market economic index data 
index_df = bank_df.iloc[: , 15:21]
index_df.head()
idx_corr=index_df.corr()
idx_corr
sns.set_context("notebook",font_scale = 1.0, rc = {"lines.linewidth":2.5})
plt.figure(figsize = (10,7))
a = sns.heatmap(idx_corr, annot = True, fmt = ".2f")
index_df= index_df.loc[:, index_df.columns != 'emp.var.rate']
index_df= index_df.loc[:, index_df.columns != 'cons.price.idx']
index_df= index_df.loc[:, index_df.columns != 'nr.employed']
bank_final_df= pd.concat([client_df, marketing_df, index_df], axis = 1)
bank_final_df.shape
bank_final_df.info()
#from sklearn.model_selection import train_test_split
X = bank_final_df.loc[:, bank_final_df.columns != 'y']
Y = bank_final_df.loc[:, bank_final_df.columns == 'y']
## Import the random forest model.
from sklearn.ensemble import RandomForestClassifier 
## This line instantiates the model. 
rf = RandomForestClassifier() 
## Fit the model on your training data.
rf.fit(X, Y) 
## And score it on your testing data.
rf.score(X, Y)

feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances.to_csv('feature_importance_4.csv')
bank_final_df['y'].replace(['yes', 'no'],[1,0 ], inplace=True)
bank_final_df.columns
bank_final_df = bank_final_df[['y','duration',
'euribor3m',
'cons.conf.idx',
'campaign',
'day_of_week',
'month',
'age',
'previous',
'pdays',
'housing_yes',
'housing_no',
'contact',
'marital_married',
'job_admin.']]
X = bank_final_df.loc[:, bank_final_df.columns != 'y']
Y = bank_final_df.loc[:, bank_final_df.columns == 'y']
X.columns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
columns = X_train.columns
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))
X=os_data_X
Y=os_data_y
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
columns = X_train.columns
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")

cm_rf = confusion_matrix(y_test, y_pred)
print("Report Entropy: ", classification_report(y_test, y_pred))
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)
print(round(acc_log,2,), "%")

cm_lr = confusion_matrix(y_test, y_pred)
print("Report Entropy: ", classification_report(y_test, y_pred))
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)

y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
print(round(acc_gaussian,2,), "%")

cm_nv = confusion_matrix(y_test, y_pred)
print("Report Entropy: ", classification_report(y_test, y_pred))
# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)

y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
print(round(acc_linear_svc,2,), "%")

cm_svm = confusion_matrix(y_test, y_pred)
print("Report Entropy: ", classification_report(y_test, y_pred))
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
print(round(acc_decision_tree,2,), "%")

cm_dt = confusion_matrix(y_test, y_pred)
print("Report Entropy: ", classification_report(y_test, y_pred))
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 
              'Logistic Regression', 
              'Random Forest', 
              'Naive Bayes', 
              'Decision Tree'],
    'Score': [acc_linear_svc, 
              acc_log, 
              acc_random_forest, 
              acc_gaussian, 
              acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)
from sklearn.metrics import roc_auc_score
param_grid = {'max_depth': np.arange(1, 10)}

grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid)
grid_tree.fit(X_train, y_train)

tree_preds = grid_tree.predict_proba(X_test)[:, 1]
tree_performance = roc_auc_score(y_test, tree_preds)
print(grid_tree.best_estimator_)
print('DecisionTree: Area under the ROC curve = {}'.format(tree_performance))
print('Best CV Score:')
print(grid_tree.best_score_)
from sklearn.metrics import roc_curve, auc
max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)
    train_pred = dt.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # Add auc score to previous train results
    train_results.append(roc_auc)
    y_pred = dt.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # Add auc score to previous test results
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()
# Decision tree with entropy 
clf_gini =DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')


clf_gini.fit(X, Y) 
scores = cross_val_score(clf_gini, X_train, y_train, cv=10, scoring = "accuracy")
print("Gini Scores:", scores)
print("Gini Mean:", scores.mean())
print("Gini Standard Deviation:", scores.std())
# Predicton on test with entropy 
y_pred_gini = clf_gini.predict(X_test) 
print("Predicted values:") 
print(y_pred_gini) 
print("Confusion Matrix Entropy: ", confusion_matrix(y_test, y_pred_gini)) 
print ("Accuracy Entropy: ", accuracy_score(y_test,y_pred_gini)*100) 
print("Report Entropy: ", classification_report(y_test, y_pred_gini))
feature_names = X.columns
class_names = str(Y.columns)

dot_data = export_graphviz(clf_gini, out_file=None, filled=True, rounded=True,
                                feature_names=feature_names,  
                                class_names=class_names)
graph = graphviz.Source(dot_data)  
graph
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, clf_gini.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf_gini.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Gini (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Bank Marketing Prediction')
plt.legend(loc="lower right")
plt.savefig('DecisionGini_ROC')
plt.show()











