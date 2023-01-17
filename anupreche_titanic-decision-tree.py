#Install Packages
!pip -q install plotly-express
!pip -q install shap
!pip -q install eli5
!pip -q install lime

import warnings
warnings.filterwarnings("ignore")
import time
import pandas as pd               
import numpy as np
import pickle

from sklearn.model_selection import train_test_split   #splitting data
from pylab import rcParams
from sklearn.linear_model import LinearRegression         #linear regression
from sklearn.metrics.regression import mean_squared_error #error metrics
from sklearn.metrics import mean_absolute_error

import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation

%matplotlib inline     
sns.set(color_codes=True)

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14

import pandas_profiling
from pandas_profiling import ProfileReport
from IPython.display import Image

from IPython.display import Image
from sklearn.preprocessing import StandardScaler 
from sklearn.feature_selection import RFE
import statsmodels.api as sm 


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

TIT =pd.read_csv('../input/titanic/train.csv')
print(TIT.shape) 
pd.set_option('display.max_columns', 50)                          # For displaying all the columns
TIT.head()
TIT.iloc[:,1].value_counts()
TIT.info
TIT.columns
TIT.dtypes
TIT.set_index('PassengerId', inplace = True)
# Lets separate the Categorical columns

cat_cols = ['Pclass','Sex','Embarked']
# Using Pandas profiling package to perform EDA

reports1=pandas_profiling.ProfileReport(TIT)
reports1.to_file('TIT_merge_EDA.html')                                   # Creating HTML file of pandas-profiling report
# count the number of NaN values in each column
print(TIT.isnull().sum())
TIT.describe().T
# Lets plot the correlation matrix 

corr_matrix = TIT.corr()
corr_matrix
# Lets take random 15 columns and plot the correlation matrix

small_df = TIT.sample(11, axis=1)
small_corr_matrix = small_df.corr()

plt.figure(figsize=(11,11))
sns.heatmap(small_corr_matrix, annot=True)
%matplotlib inline
# Scatter plot of Fare and Pclass 

plt.figure(figsize=(6,6))
sns.scatterplot('Fare','Pclass', data=TIT, hue= 'Survived')
# Scatter plot of SibSp and Age

plt.figure(figsize=(6,6))
sns.scatterplot('SibSp', 'Age', data=TIT, hue='Survived')
# Scatter plot of Pclass and Age

plt.figure(figsize=(6,6))
sns.scatterplot('Pclass', 'Age', data=TIT, hue='Survived')
# Scatter plot of Pclass and Age

plt.figure(figsize=(6,30))
sns.scatterplot('Pclass', 'Cabin', data=TIT, hue='Survived')
TIT['Age'] = TIT['Age'].interpolate(method='linear', order=1)
TIT['Age'].isnull().sum()
pd.set_option('display.max_rows', None)
TIT.Cabin.value_counts(ascending=False)
TIT['Cabin'] = TIT['Cabin'].replace(np.nan,'X')
TIT['Cabin'].isnull().sum()
TIT.Cabin.value_counts(ascending=False)
# Lets one hot encode all the categorical columns

encoded_cols = pd.get_dummies(TIT[cat_cols], drop_first=True)
encoded_cols.shape
# We will concat the one hot encoded dataframe with train dataframe

TIT = pd.concat([TIT, encoded_cols], axis=1)
TIT.shape
TIT = TIT.drop(columns = ['Name','Ticket','Cabin'],axis=1)
TIT.shape
TIT.drop(cat_cols,axis=1)
# Lets drop the categorical column as we have already included one hot columns

X_train = TIT.drop(cat_cols, axis = 1)

# We will also drop the target column "attack" and "Num_outbound_cmds"
X_train = X_train.drop(['Survived'], axis = 1)
# The number of rows and columns in the features
X_train.shape 
# The target label
y_train = TIT['Survived']                                            
y_train.unique()
X_train
from sklearn.feature_selection import chi2
chi_scores = chi2(X_train ,y_train)
chi_scores
p_values = pd.Series(chi_scores[1],index = X_train.columns)
p_values.sort_values(ascending = False , inplace = True)
p_values.plot.bar()
X_train = X_train.drop(columns=['Embarked_Q','SibSp'],axis = 1)
TES = pd.read_csv('../input/titanic/test.csv')
print(TES.shape) 
# count the number of NaN values in each column
print(TES.isnull().sum())
print(TES.shape) 
TES.head()
TES.columns
TES['Age'] =TES['Age'].interpolate(method='linear', order=1)
TES['Age'].isnull().sum()
'?'# Lets one hot encode all the categorical columns

encoded_cols_Test = pd.get_dummies(TES[cat_cols], drop_first=True)
encoded_cols_Test.shape
# We will concat the one hot encoded dataframe with train dataframe

TES = pd.concat([TES, encoded_cols_Test], axis=1)

TES.drop(encoded_cols_Test,axis=1)
TES = TES.drop(columns = ['Name', 'Ticket','Cabin'], axis =1 )
# metrics

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, recall_score
def model_train(model, name):
    model.fit(X_train, y_train)                                          # Fitting the model
    y_pred = model.predict(X_test)                                       # Making prediction from the trained model
    cm = confusion_matrix(y_test, y_pred)                               
    print("Grid Search Confusion Matrix " +" Validation Data")                # Displaying the Confusion Matrix
    print(cm)
    print('-----------------------')
    print('-----------------------')
    cr = classification_report(y_test, y_pred)
    print(name +" Classification Report " +" Validation Data")           # Displaying the Classification Report
    print(cr)
    print('------------------------')
    print(name +" AUC Score " +" Validation Data")
    auc = roc_auc_score(y_test, y_pred)       
    print("AUC Score " + str(auc))                                       # Displaying the AUC score
    print(name +" Recall " +" Validation Data")
    rec = recall_score(y_test, y_pred)
    print("Recall "+ str(rec))                                           # Displaying the Recall score
    print('_________________________')
    print(name + " Bias")                                                 # Calculating bias
    bias = y_pred - y_test.mean()
    print("Bias "+ str(bias.mean()))
    
    print(name + " Variance")                                             # Calculate Variance
    var = np.var([y_test, y_pred], axis=0)
    print("Variance " + str(var.mean()) )
    return auc, rec, model


lr = LogisticRegression(random_state=101)
lr_auc, lr_rec, lr_model = model_train(lr, "Logistic Regression")
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'entropy',max_depth = 10, min_samples_leaf =3, random_state=101)
dt_auc, dt_rec, dt_model = model_train(dt, "Decision Tree")
# Random Forest Algorithm

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=40, max_features='sqrt', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=18,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
rf_auc, rf_rec, rf_model = model_train(rf_clf, "Random Forest Classifier")
from sklearn.model_selection import GridSearchCV

random_grid = {'n_estimators': range(5,20),
              'max_features' : ['auto', 'sqrt'],
              'max_depth' : [10,20,30,40],
              'min_samples_split':[2,5,10],
              'min_samples_leaf':[1,2,4]}

rf = RandomForestClassifier()

rf_gs = GridSearchCV(rf, random_grid, cv = 3, n_jobs=-1, verbose=2)

rf_gs.fit(X_train, y_train)
y_pred = rf_gs.predict(X_test)

rf_gs.best_estimator_
print("Grid Search Validation Data")
cm = confusion_matrix(y_test, y_pred)                               
print("Grid Search Confusion Matrix " +" Validation Data")                # Displaying the Confusion Matrix
print(cm)
print('-----------------------')
cr = classification_report(y_test, y_pred)
print("Grid Search Classification Report " +" Validation Data")           # Displaying the Classification Report
print(cr)
print('------------------------')
print("Grid Search AUC Score " +" Validation Data")
auc = roc_auc_score(y_test, y_pred)       
print("AUC Score " + str(auc))                                       # Displaying the AUC score
print("Grid Search Recall " +" Validation Data")
rec = recall_score(y_test, y_pred)
print("Recall "+ str(rec))                                           # Displaying the Recall score
print('_________________________')
print("Grid Search Bias")                                                 # Calculating bias
bias = y_pred - y_test.mean()
print("Bias "+ str(bias.mean()))
    
print("Grid Search Variance")                                             # Calculate Variance
var = np.var([y_test, y_pred], axis=0)
print("Variance " + str(var.mean()) )
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestClassifier()

rf_random = RandomizedSearchCV(rf, random_grid, cv = 3, n_jobs=-1, verbose=2)

rf_random.fit(X_train, y_train)
y_pred = rf_random.predict(X_test)
print("Randomized Grid Search Validation Data")
cm = confusion_matrix(y_test, y_pred)                               
print("Randomized Grid Search Confusion Matrix " +" Validation Data")                # Displaying the Confusion Matrix
print(cm)
print('-----------------------')
cr = classification_report(y_test, y_pred)
print("Randomized Grid Search Classification Report " +" Validation Data")           # Displaying the Classification Report
print(cr)
print('------------------------')
print("Randomized Grid Search AUC Score " +" Validation Data")
auc = roc_auc_score(y_test, y_pred)       
print("AUC Score " + str(auc))                                       # Displaying the AUC score
print("Randomized Grid Search Recall " +" Validation Data")
rec = recall_score(y_test, y_pred)
print("Recall "+ str(rec))                                           # Displaying the Recall score
print('_________________________')
print("Randomized Grid Search Bias")                                                 # Calculating bias
bias = y_pred - y_test.mean()
print("Bias "+ str(bias.mean()))
    
print("Randomized Grid Search Variance")                                             # Calculate Variance
var = np.var([y_test, y_pred], axis=0)
print("Variance " + str(var.mean()) )
rf_random.best_estimator_
import eli5

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(dt_model, random_state=101).fit(X_test, y_test)      # Evaluate the permutation importance 
eli5.show_weights(perm, feature_names = X_test.columns.values)                    # Display the weights of each features
perm = PermutationImportance(rf_model, random_state=101).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.values)