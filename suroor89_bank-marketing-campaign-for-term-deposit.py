# Current workspace
!pwd
import os
os.environ['KMP_DUPLICATE_LIB_OK']= 'True'
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import plotly.graph_objs as go
from sklearn.utils import shuffle
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import warnings
warnings.simplefilter('ignore')
sns.set_style("darkgrid")
pd.pandas.set_option('display.max_columns', None)
%matplotlib inline
#Load csv file to pd dataframe
bank_data = pd.read_csv("../input/bank-additional-full.csv",sep=';')
# Columns information
bank_data.columns
#change column names
bank_data.rename(columns={'default': 'has_credit','housing':'housing_loan','loan':'personal_loan','y':'subscribed'}, inplace=True)
bank_data.columns
# print first five rows of bank_data
bank_data.head()
# display total number of rows and columns
bank_data.shape
bank_data.isnull().sum()
# verify data types of bank_data information 
bank_data.info()
# Describe numeric bank_data
bank_data.describe()
# check the occurrence of each housing_loan in bank_data
house_count = pd.DataFrame(bank_data['housing_loan'].value_counts())
house_count
# check the occurrence of each education in bank_data
ed_count = pd.DataFrame(bank_data['education'].value_counts())
ed_count
!pip install researchpy
from scipy.stats import chisquare
import researchpy as rp
table, results = rp.crosstab(bank_data['education'], bank_data['subscribed'], prop= 'col', test= 'chi-square')
results
table
table, results = rp.crosstab(bank_data['emp.var.rate'], bank_data['cons.price.idx'], prop= 'col', test= 'chi-square')
results
fig, ax = plt.subplots()
fig.set_size_inches(12, 8)
sns.boxplot(x='marital', y='duration', data=bank_data, palette="colorblind")
plt.title('Martial data by duration')
plt.show()
bank_data.duplicated().sum()
'''remove duplicated rows'''
def clean_data(data):
    clean_data = data.drop_duplicates()
    return clean_data
clean_bank_data = clean_data(bank_data)
clean_bank_data.shape
clean_bank_data.info()
clean_bank_data['age'] = clean_bank_data['age'].astype(float)
clean_bank_data.info()
'''Divide varibles in categorical and numerical'''
categorical_vars = [col for col in clean_bank_data.columns if (clean_bank_data[col].dtype == 'object') & (col != 'subscribed')]
numeric_vars = ['age', 'duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
target_var = ['subscribed']
# Look for outliners for all numerical features
fig, ax = plt.subplots(2, 5, figsize=(35, 15))
for i, subplot in zip(numeric_vars, ax.flatten()):
    plt.title('Box plot of: '+i)
    sns.boxplot(y = clean_bank_data[i], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
plt.subplots_adjust(wspace=0.45, hspace=0.8)

# explore target variable data
fig, ax = plt.subplots()
fig.set_size_inches(10, 8)
sns.countplot(x = 'subscribed', data = clean_bank_data)
plt.title('Target Variable')
plt.show()
fig, ax = plt.subplots()
fig.set_size_inches(10, 8)
sns.countplot(x ='marital', hue = 'subscribed', data = clean_bank_data)
plt.title('Marital Status Distribution')
plt.show()
clean_bank_data.describe(include=[np.number])
clean_bank_data.describe(include = ['O'])
def get_correlation(data):
    return data.corr()

corr_data = get_correlation(clean_bank_data)
corr_data
sns.FacetGrid(clean_bank_data,hue = 'subscribed', height = 7).map(plt.scatter, "emp.var.rate","euribor3m").add_legend()
plt.show()
sns.FacetGrid(clean_bank_data,hue = 'subscribed', height = 7).map(plt.scatter, "nr.employed", "emp.var.rate").add_legend()
plt.show()
sns.FacetGrid(clean_bank_data,hue = 'subscribed', height = 7).map(plt.scatter, "euribor3m", "nr.employed").add_legend()
plt.show()
# Visualize all numerical variables
sns.set(style="darkgrid")
fig, ax = plt.subplots(2, 5, figsize=(35, 15))
for variable, subplot in zip(numeric_vars, ax.flatten()):
    sns.distplot(clean_bank_data[variable], ax=subplot, kde=False, hist=True,color='purple')
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
plt.subplots_adjust(wspace=0.45, hspace=0.8)
# Visualize all categorical variables
sns.set(style="darkgrid")
fig, ax = plt.subplots(2, 5, figsize=(35, 15))
for variable, subplot in zip(categorical_vars, ax.flatten()):
    sns.countplot(clean_bank_data[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
plt.subplots_adjust(wspace=0.45, hspace=0.8)

fig, ax = plt.subplots(2, 5, figsize=(35, 15))
for variable, subplot in zip(categorical_vars, ax.flatten()):
    sns.countplot(x=variable,hue='subscribed',data=clean_bank_data,ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
plt.subplots_adjust(wspace=0.45, hspace=0.8)
vals = clean_bank_data['marital'].value_counts().tolist()
labels = ['married', 'divorced', 'single']

data = [go.Bar(x=labels, y=vals, marker=dict(color="#76D7C4"))]
layout = go.Layout(title="Count by Marital Status",)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='marital bar')
clean_bank_data.info()
# apply pairplot to see how data is distributed 
#sns.pairplot(clean_bank_data, hue=y1, height =5)
#plt.show()
fig, ax = plt.subplots()
fig.set_size_inches(15, 10)
sns.set_context("notebook",font_scale = 1.0, rc = {"lines.linewidth":2.5})
ax = sns.heatmap(corr_data, annot = True, fmt = ".2f")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
# seperate feature and traget variables
X = clean_bank_data.iloc[:,:-1]
y = clean_bank_data.iloc[:,-1]
X.columns
#One hot coding with pandas get_dummies method
one_hot = pd.get_dummies(X)
one_hot.shape
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1,2,3,4,5,6,7,8,9,14])], remainder='passthrough')
X = np.array(columnTransformer.fit_transform(X), dtype = np.str)
X.shape
columnTransformer.get_feature_names
y, unique = pd.factorize(y)
y
# change numpy array to dataframe 
X = pd.DataFrame(X)
y = pd.DataFrame(y)
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 10)
# Import the model RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 20 decision trees
rf = RandomForestClassifier(n_estimators = 20,n_jobs=1)

# Train the model on training data
rf.fit(X_train, y_train);

# Use the forest's predict method on the test data
y_predict = rf.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_absolute_error

# Model Accuracy, how often is the classifier correct?
training_accuracy = rf.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_predict)

print(f'Training set accuracy: {round(training_accuracy*100,2)}%')
print(f'Test set accuracy: {round(test_accuracy*100,2)}%')
# Confusion Matrix Heatmap
from sklearn.metrics import confusion_matrix

cm = confusion_matrix( y_test, y_predict)
ax = sns.heatmap(cm, annot = True, fmt = ".2f",cmap=plt.cm.Blues)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Confusion matrix of the classifier',fontsize=16)
plt.show()
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0)
columns = X_train.columns
smote_X, smote_y = smote.fit_sample(X_train, y_train)
#smote_X = pd.DataFrame(data=smote_X,columns=columns )
#smote_y= pd.DataFrame(data=smote_y,columns=[clean_bank_data['subscribed']])
# we can Check the numbers of our data
print("length of oversampled data is ",len(smote_X))
print("length of oversampled data is ",len(smote_y))
#print("Number of no subscription in oversampled data",len(smote_y[smote_y['subscribed']==0]))
#print("Number of subscription",len(smote_y[smote_y['subscribed']==1]))
#print("Proportion of no subscription data in oversampled data is ",len(smote_y[smote_y['subscribed']==0])/len(smote_X))
#print("Proportion of subscription data in oversampled data is ",len(smote_y[smote_y['subscribed']==1])/len(smote_X))
# explore target variable data after applying SMOTE
fig, ax = plt.subplots()
fig.set_size_inches(10, 8)
sns.countplot(x = "subscribed", data = smote_y)
plt.title('Target Variable')
plt.show()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X= smote_X
y= smote_y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
columns = X_train.columns
print('Training Features Shape:', X_train.shape)
print('Training test Shape:', X_test.shape)
print('Testing Features Shape:', y_train.shape)
print('Testing test Shape:', y_test.shape)
from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussiaNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
models = [
     ('LR',LogisticRegression()),
     ('NB',GaussianNB()),
     ('DT',DecisionTreeClassifier())
     ]

cv_model_score = {}
for name, model in models:
    clf = model
    score = cross_val_score(clf, X, y, cv=10).mean()
    clf.fit(X_train,y_train)
    accuracy = clf.score(X_test,y_test)
    print(name,': ',accuracy)
    cv_model_score[name] = score
print('CV scores: ', cv_model_score)
# Import the model RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 20 decision trees
rf = RandomForestClassifier(n_estimators = 20,n_jobs=1)

# Train the model on training data
rf.fit(X_train, y_train);

# Use the forest's predict method on the test data
y_predict = rf.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_absolute_error

# Model Accuracy, how often is the classifier correct?
training_accuracy = rf.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_predict)

print(f'Training set accuracy: {round(training_accuracy*100,2)}%')
print(f'Test set accuracy: {round(test_accuracy*100,2)}%')
# import RandomizedSearchCV from model_selection for hyper parameter tuning
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

rd_est = RandomForestClassifier(n_jobs=-1)
# apply parameters which are importatnt
rf_param={'max_depth':[80, 90, 100, 110],
              'n_estimators':[100, 150, 180, 200],
              'max_features':randint(1,3),
               'criterion':['gini','entropy'],
               'bootstrap':[True,False],
               'min_samples_split': [8, 10, 12],
               'min_samples_leaf':randint(1,5)
              }

def hypertuning_rscv(rd_est, p_distr, nbr_iter,X,y):
    rdmsearch = RandomizedSearchCV(rd_est, param_distributions=p_distr,
                                  n_jobs=-1, n_iter=nbr_iter, cv=10) # Cross-Validation - using Stratified KFold CV
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score

rf_parameters, rf_ht_score = hypertuning_rscv(rd_est, rf_param, 20, X, y)
rf_parameters
rf_ht_score
classifier=RandomForestClassifier(n_estimators=200,bootstrap= True,criterion='entropy',max_depth=110,max_features=1,min_samples_leaf= 1,min_samples_split= 8)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_predict = classifier.predict(X_test)

## Cross Validation good for selecting models
from sklearn.model_selection import cross_val_score

cross_val=cross_val_score(classifier,X,y,cv=10,scoring='accuracy').mean()
print(cross_val)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_absolute_error

# Model Accuracy, how often is the classifier correct?
training_accuracy = classifier.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_predict)

print(f'Training set accuracy: {round(training_accuracy*100,2)}%')
print(f'Test set accuracy: {round(test_accuracy*100,2)}%')

# Print out the mean absolute error (mae)
print('Mean Absolute Error {:0.3f}\n'.format(mean_absolute_error(y_test, y_predict)))

print("Confusion Matrix:\n", confusion_matrix(y_test,y_predict),'\n')

print("Classification Report:\n", classification_report(y_test, y_predict))
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(classifier.get_params())
feature_importances = pd.DataFrame(classifier.feature_importances_,
                                   index = one_hot.columns,
                    columns=['Feature_Importance_score']).sort_values('Feature_Importance_score',ascending=False)
feature_importances.head()
# Confusion Matrix Heatmap
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predict)
ax = sns.heatmap(cm, annot = True, fmt = ".2f",cmap=plt.cm.Blues)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Confusion matrix of the classifier',fontsize=16)
plt.show()
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_predict)
from sklearn.model_selection import KFold, cross_val_score
K_fold = KFold (n_splits=10, shuffle=True,random_state=0)
clf = RandomForestClassifier(n_estimators = 10)
score = cross_val_score(clf,X_test,y_test,cv=5,n_jobs=1,scoring='accuracy')
score
round(np.mean(score)*100,2)
y_pred_prob = classifier.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test,y_pred_prob)
# creating plot of ROC Curve
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
_ = plt.xlabel('False Positive Rate')
_ = plt.ylabel('True Positive Rate')
_ = plt.title('ROC Curve')
_ = plt.xlim([-0.02, 1])
_ = plt.ylim([0, 1.02])
_ = plt.legend(loc="lower right")
from sklearn.metrics import roc_curve, auc
max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
    rf = RandomForestClassifier(max_depth=max_depth)
    rf.fit(X_train, y_train)
    train_pred = rf.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # Add auc score to previous train results
    train_results.append(roc_auc)
    y_pred = rf.predict(X_test)
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