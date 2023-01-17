
#Import all the libraries required for analysis and building predcitive modeling
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visullizong missing vlaues in data set 
import missingno as msno 

# visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#setting display options
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 500)
np.set_printoptions(linewidth =400)

# ignore warnings
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

# imuters 

from sklearn.impute import KNNImputer 

# machine learning computation 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# Split data
from sklearn.model_selection import train_test_split

print("All necessary files read")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
#combine = [train_df, test_df]
#combine
train_df.head()
train_df.dtypes

print(train_df .shape)
print(test_df.shape)
#full_data = train_df.append( test_df , ignore_index = True )
#titanic_train = full_data[ :891 ]

#del train , test

#print ('Datasets:' , 'full_data:' , full_data.shape , 'titanic:' , titanic_train.shape)
#full_data.tail()
#titanic_test= full_data[891:]
#titanic_test.head()
#Visualization
# For categorical variable
def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

# For numerical variable
def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()
 
# Plot corelation
def plot_correlation_map( df ):
    corr = train.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )
 

# Total % of missing vlaues
def missing_percentage(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])
# missing values in Train data: data quality
missing_percentage(train_df)
# missing values in Test data: data quality
missing_percentage(test_df)
# Count categorical feature instances and %
def percent_value_counts(df, feature):
    """This function takes in a dataframe and a column and finds the percentage of the value_counts"""
    percent = pd.DataFrame(round(df.loc[:,feature].value_counts(dropna=False, normalize=True)*100,2))
    ## creating a df with th
    total = pd.DataFrame(df.loc[:,feature].value_counts(dropna=False))
    ## concating percent and total dataframe

    total.columns = ["Total"]
    percent.columns = ['Percent']
    return pd.concat([total, percent], axis = 1)
percent_value_counts(train_df, 'Embarked')
# what are those 2 Null vlaues in Embarked
train_df[train_df.Embarked.isnull()]
#import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(10,7),ncols=2)
ax1 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train_df, ax = ax[0]);
ax2 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=test_df, ax = ax[1]);
ax1.set_title("Training Set", fontsize = 15)
ax2.set_title('Test Set',  fontsize = 15)
# Replace NA in Embarked column with 'C'
train_df.Embarked.fillna('C', inplace = True)
#percent_value_counts(train_df, 'Embarked')
# See value count of Cabin
percent_value_counts(train_df, 'Cabin')
# See value count of Cabin
percent_value_counts(test_df, 'Cabin')
#combine train and test to tackle Cabin featute
## Concat train and test into a variable "all_data"
survivers = train_df.Survived

train_df.drop(["Survived"],axis=1, inplace=True)

all_data = pd.concat([train_df,test_df], ignore_index=False)

## Assign all the null values to N
all_data.Cabin.fillna("N", inplace=True)
all_data.Cabin = [i[0] for i in all_data.Cabin]
percent_value_counts(all_data, 'Cabin')
all_data.groupby('Cabin')['Fare'].mean().sort_values()
with_N = all_data[all_data.Cabin == "N"]
without_N = all_data[all_data.Cabin != "N"]
def cabin_estimator(i):
    """Grouping cabin feature by the first letter"""
    a = 0
    if i<16:
        a = "G"
    elif i>=16 and i<27:
        a = "F"
    elif i>=27 and i<38:
        a = "T"
    elif i>=38 and i<47:
        a = "A"
    elif i>= 47 and i<53:
        a = "E"
    elif i>= 53 and i<54:
        a = "D"
    elif i>=54 and i<116:
        a = 'C'
    else:
        a = "B"
    return a
##applying cabin estimator function. 
with_N['Cabin'] = with_N.Fare.apply(lambda x: cabin_estimator(x))

## getting back train. 
all_data = pd.concat([with_N, without_N], axis=0)

## PassengerId helps us separate train and test. 
all_data.sort_values(by = 'PassengerId', inplace=True)

## Separating train and test from all_data. 
train = all_data[:891]

test = all_data[891:]

# adding saved target variable with train. 
train['Survived'] = survivers
test[test.Fare.isnull()]
missing_Fare = test[(test.Pclass == 3) & (test.Sex == "male") & (test.Embarked == "S")].Fare.mean()
missing_Fare
test.Fare.fillna(missing_Fare, inplace = True)
print ("Train age missing value: " + str((train.Age.isnull().sum()/len(train))*100)+str("%"))
print ("Test age missing value: " + str((test.Age.isnull().sum()/len(test))*100)+str("%"))
# Transform Sex into binary values 0 and 1
 
train['Sex'] = train.Sex.apply(lambda x: 0 if x == "female" else 1)
test['Sex'] = test.Sex.apply(lambda x: 0 if x == "female" else 1)
# display the statistial overview of data
train.describe()
# display survival summary
Survival_summary = train.groupby('Survived')
Survival_summary.mean()
#Survival_summary.head()
#Survival_summary.std()
 # Plot correlation to see what features correate most with survived.

#plot_correlation_map(train_df )
#Type of variables in  Full dataset
#titanic_train.dtypes
#titanic_train.Survived.value_counts(1)*100#
plt.figure(figsize=(8,4))
Survived = train.Survived.value_counts()
sns.barplot(y=Survived.values, x=Survived.index, alpha=0.6)
plt.title('Distribution of Passenger survival')
plt.xlabel('Passenger survival', fontsize=10)
plt.ylabel('Count', fontsize=10)
# Survival rate and relationship with Parch
plot_categories(train, cat ='Parch', target= 'Survived')  # Keywork argument
# plot Embarked feature and realtion with survival
#plot_categories(titanic_train, cat ='Embarked', target= 'Survived')
# plot sex feature and realtion with survival
plot_categories(train, cat ='Sex', target= 'Survived')
# Survival rate and relationship with Pclass
plot_categories(train, cat ='Pclass', target= 'Survived')
# Survival rate and relationship with SibSp
plot_categories(train, cat ='SibSp', target= 'Survived')
# Plot distribution of Age of passenger with respect to Sex
plot_distribution( train , var = 'Age' , target = 'Survived' , row = 'Sex' )
# Plot distribution of Fare of passenger with respect to Survival rate
plot_distribution(train , var = 'Fare' , target = 'Survived')
plot_correlation_map(train)
train.corr()
pd.DataFrame(abs(train.corr()['Survived']).sort_values(ascending = False))
## Family_size seems like a good feature to create
train['family_size'] = train.SibSp + train.Parch+1  # extra 1 is for a passenger
test['family_size'] = test.SibSp + test.Parch+1
def family_group(size):
    a= ''
    if (size <=1):
        a = 'alone'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a
train['family_group'] = train['family_size'].map(family_group)
test['family_group'] = test['family_size'].map(family_group)
#train.head()
## Calculating fare based on family size. 
#train['calculated_fare'] = train.Fare/train.family_size
#test['calculated_fare'] = test.Fare/test.family_size
#train['Fare'].dtypes
def fare_group(fare):
    a= ''
    if fare <= 4:
        a = 'Very_low'
    elif fare <= 10:
        a = 'low'
    elif fare <= 30:
        a = 'mid'
    elif fare <= 50:
        a = 'high'
    else:
        a = "very_high"
    return a
train['fare_group'] = train['Fare'].map(fare_group)
test['fare_group'] = test['Fare'].map(fare_group)
train.drop(['PassengerId', 'Name', 'Ticket'], axis =1, inplace = True)
test.drop(['PassengerId', 'Name', 'Ticket'], axis =1, inplace = True)
train.head()
#percent_value_counts(train ,'Fare')


# Separate caegorocal variables and numerical variables to do feature encoding and combine later
cat_var = ['Embarked', 'Pclass', 'Cabin', 'fare_group', 'family_group']
num_var = ['Age', 'Fare']
rest_var = ['Parch', 'SibSp' , 'Survived', 'Pclass']
#The drop_first=True drops one column from the resulted dummy features. The purpose is to avoid multicollinearity. Here is the results:

train = pd.get_dummies(train, columns=cat_var, drop_first=True)
test = pd.get_dummies(test, columns=cat_var, drop_first=True)
# drop variable, no longer use
train.drop(['family_size', 'Parch', 'SibSp'], axis =1, inplace = True)
test.drop(['family_size',  'Parch', 'SibSp'], axis =1, inplace = True)
train.head()
train[train.Age.isnull()].head()
train.dtypes
test[test.Age.isnull()].head()
train_new = train
test_new = test
train_new.shape
test_new.shape

# from sklearn.impute import KNNImputer
# age1 = train_new[['Age']]
# age1.head()
# imputer = KNNImputer(n_neighbors=2)
# agetrain_imp =imputer.fit_transform(age1)
# import sklearn.preprocessing as preprocessing
# scaler = preprocessing.StandardScaler()
# col_names = ['Age']
# #age_scale_param = scaler.fit(df['Age'].values)
# features = train_new[col_names]
# scaler = scaler.fit(features.values)
# features = scaler.transform(features.values)
# train_new[col_names] = features
# train_new.head()
# from sklearn.ensemble import RandomForestRegressor
# def predict_missing_ages(train_new):
    
#     known_age = train_new[train_new.Age.notnull()]
#     unknown_age = train_new[train_new.Age.isnull()]
#     unknown_age = unknown_age.drop(['Age'], axis =1)
#     y = known_age['Age']
#     X = known_age.drop(['Age'], axis =1)
    
#     print(X.shape, y.shape, unknown_age.shape)
    

#     rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
#     rfr.fit(X, y)

#     predictedAges = rfr.predict(unknown_age)
    
#     print(predictedAges)
    
 
#     df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
#     return df, rfr
print ("Train age missing value: " + str((known_age_trn.Age.isnull().sum()/len(train))*100)+str("%"))
print ("Test age missing value: " + str((unknown_age_trn.Age.isnull().sum()/len(test))*100)+str("%"))
from sklearn.ensemble import RandomForestRegressor
known_age_trn = train_new[train_new.Age.notnull()]
unknown_age_trn = train_new[train_new.Age.isnull()]
unknown_age_trn_rfr = unknown_age_trn.drop('Age', axis =1).values
# unknown_age_trn = unknown_age_trn .values
y = known_age_trn['Age'].values
X = known_age_trn.drop('Age', axis =1).values

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# print(X.shape, y.shape, unknown_age.shape)
    
RFReg = RandomForestRegressor(n_estimators = 1000, random_state = 0)
RFReg.fit(X_train, y_train)

# predictedAges = RFReg.predict(unknown_age)
    
# print(predictedAges)

# unknown_age
#Predicted Height from test dataset w.r.t Random Forest Regression
y_predict_trn= RFReg.predict((X_test))

#Model Evaluation using R-Square for Random Forest Regression
from sklearn import metrics
r_square = metrics.r2_score(y_test, y_predict_trn)
print('R-Square Error associated with Random Forest Regression is:', r_square)
print ("Train age missing value: " + str((known_age_trn.Age.isnull().sum()/len(train))*100)+str("%"))
print ("Test age missing value: " + str((unknown_age_trn.Age.isnull().sum()/len(test))*100)+str("%"))
y_predict_trn = RFReg.predict((unknown_age_trn_rfr ))
# y_predict_trn = pd.DataFrame(y_predict_trn)
unknown_age_trn['Age'] = y_predict_trn 
# unknown_age_trn.head(100)
# unknown_age_trn['Age']
print(unknown_age_trn.shape)
print(y_predict_trn.shape)
print(unknown_age_trn_rfr.shape)
print ("Train age missing value: " + str((known_age_trn.Age.isnull().sum()/len(train))*100)+str("%"))
# Combine tow dataframes to make it orignal train data
train_new = pd.concat([known_age_trn, unknown_age_trn],ignore_index=True)
print ("Train age missing value: " + str((train_new.Age.isnull().sum()/len(train))*100)+str("%"))
# unknown_age1['Age'] = y_predict_rfr 
# unknown_age.head()
from sklearn.ensemble import RandomForestRegressor
known_age_tst = test_new[test_new.Age.notnull()]
unknown_age_tst = test_new[test_new.Age.isnull()]
unknown_age_tst_rfr = unknown_age_tst.drop('Age', axis =1).values
# unknown_age_trn = unknown_age_trn .values
y = known_age_tst['Age'].values
X = known_age_tst.drop('Age', axis =1).values

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# print(X.shape, y.shape, unknown_age.shape)
    
RFReg = RandomForestRegressor(n_estimators = 1000, random_state = 0)
RFReg.fit(X_train, y_train)

# predictedAges = RFReg.predict(unknown_age)
    
# print(predictedAges)

# unknown_age
#Predicted Height from test dataset w.r.t Random Forest Regression
y_predict_tst= RFReg.predict((X_test))

#Model Evaluation using R-Square for Random Forest Regression
from sklearn import metrics
r_square = metrics.r2_score(y_test, y_predict_tst)
print('R-Square Error associated with Random Forest Regression is:', r_square)
y_predict_tst = RFReg.predict((unknown_age_tst_rfr ))
# y_predict_tst = pd.DataFrame(y_predict_tst)
unknown_age_tst['Age'] = y_predict_tst
unknown_age_tst.head()
# Combine tow dataframes to make it orignal test data
test_new = pd.concat([known_age_tst, unknown_age_tst],ignore_index=True)
test_new.shape
print ("Test age missing value: " + str((test_new.Age.isnull().sum()/len(train))*100)+str("%"))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

## transforming "train_x"
#df = sc.fit_transform(train_new[''])
#categorical_df = pd.get_dummies(categorical_df) #
#categorical_df.head()
#numerical_df = numerical_df .fillna(numerical_df .mean() )
#numerical_df.head()
#full_data_FE['Survived']=full_data_FE['Survived'].astype(int)
# Define X and Y in train_new
y = train_new['Survived']
X = train_new.drop(['Survived'], axis =1)
print("done")
#X.head()
# Define train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X ,y,test_size=0.20, random_state=123, stratify=y)
# from sklearn.utils import resample
# from imblearn.over_sampling import SMOTE 

# # Upsample minority class
# X_train_u, y_train_u = resample(X_train[y_train == 1],
#                                 y_train[y_train == 1],
#                                 replace=True,
#                                 n_samples=X_train[y_train == 0].shape[0],
#                                 random_state=1)

# X_train_u = np.concatenate((X_train[y_train == 0], X_train_u))
# y_train_u = np.concatenate((y_train[y_train == 0], y_train_u))

# # Upsample using SMOTE
# #sm = SMOTE(random_state=12)
# #x_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)


# # Downsample majority class
# X_train_d, y_train_d = resample(X_train[y_train == 0],
#                                 y_train[y_train == 0],
#                                 replace=True,
#                                 n_samples=X_train[y_train == 1].shape[0],
#                                 random_state=1)
# X_train_d = np.concatenate((X_train[y_train == 1], X_train_d))
# y_train_d = np.concatenate((y_train[y_train == 1], y_train_d))


# print("Original shape:", X_train.shape, y_train.shape)
# print("Upsampled shape:", X_train_u.shape, y_train_u.shape)
# #print ("SMOTE sample shape:", x_train_sm.shape, y_train_sm.shape)
# print("Downsampled shape:", X_train_d.shape, y_train_d.shape)
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_score

# # Create the Original, Upsampled, and Downsampled training sets
# methods_data = {"Original": (X_train, y_train),
#                 "Upsampled": (X_train_u, y_train_u),
#                 "Downsampled": (X_train_d, y_train_d)}

# # Loop through each type of training sets and apply 5-Fold CV using Logistic Regression
# # By default in cross_val_score StratifiedCV is used
# for method in methods_data.keys():
#     lr_results = cross_val_score(LogisticRegression(), methods_data[method][0], methods_data[method][1], cv=5, scoring='f1')
#     print(f"The best F1 Score for {method} data:")
#     print (lr_results.mean())
 
# cross_val_score(LogisticRegression(class_weight='balanced'), X_test, y_test, cv=5, scoring='f1').mean()
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

lr = LogisticRegression()

# Fit the model to the Upsampling data
lr = lr.fit(X_train, y_train)

print ("\n\n ---Logistic Regression Model---")
lr_auc = roc_auc_score(y_test, lr.predict(X_test))
print ("Logistic Regression AUC = %2.2f" % lr_auc)

lr_accuracy = accuracy_score(y_test, lr.predict(X_test))
print ("Logistic Regression  = %2.2f" % lr_auc)

#lr2 = lr.fit(X_train, y_train)
print(classification_report(y_test, lr.predict(X_test)))
# lr_ = LogisticRegression()

# # Fit the model to the Upsampling data
# lr_u = lr.fit(X_train_u, y_train_u)

# print ("\n\n ---Logistic Regression Model---")
# lr_auc = roc_auc_score(y_test, lr.predict(X_test))
# print ("Logistic Regression AUC = %2.2f" % lr_auc)

# lr_accuracy = accuracy_score(y_test, lr.predict(X_test))
# print ("Logistic Regression  = %2.2f" % lr_auc)

# #lr2 = lr.fit(X_train_d, y_train_d)
# print(classification_report(y_test, lr_u.predict(X_test)))
# Crosss validation logictic regression: original data
from sklearn.model_selection import cross_val_score
lr_result = cross_val_score(lr, X_train, y_train, cv=5, scoring='f1')
lr_result.mean()
# lr_result = cross_val_score(lr, X_train_d, y_train_d, cv=5, scoring='f1')
# lr_result.mean()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Random Forest Model
rf = RandomForestClassifier()

rf_result = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')

rf_result.mean()
from sklearn.metrics import roc_auc_score

rf = rf.fit(X_train, y_train)

print ("\n\n ---Random Forest Model---")
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
print ("Random Forest AUC = %2.2f" % rf_roc_auc)
print(classification_report(y_test, rf.predict(X_test)))
#import sklearn
#print(sklearn.__version__)

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()  

gbc = gbc.fit(X_train, y_train)

#gbc
# # original sample
# gbc_orignalsample = GradientBoostingClassifier()  

# gbc_orignalsample = gbc_orignalsample.fit(X_train, y_train)

# gbc_orignalsample
gbc_result = cross_val_score(gbc, X_train, y_train, cv=5, scoring='f1')
gbc_result.mean()
# gbc_result = cross_val_score(gbc, X_train_d, y_train_d, cv=5, scoring='f1')
# gbc_result.mean()
# gbc_u = GradientBoostingClassifier()  

# gbc_u = gbc_u.fit(X_train_u, y_train_u)

# gbc_u
# gbc_result_u = cross_val_score(gbc, X_train_u, y_train_u, cv=5, scoring='f1')
# gbc_result_u.mean()
from sklearn.metrics import roc_auc_score

print ("\n\n ---Gradient Boosting Model---")
gbc_auc = roc_auc_score(y_test, gbc.predict(X_test))
print ("Gradient Boosting Classifier AUC = %2.2f" % gbc_auc)
print(classification_report(y_test, gbc.predict(X_test)))
# Create ROC Graph
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
gbc_fpr, gbc_tpr, gbc_thresholds = roc_curve(y_test, gbc.predict_proba(X_test)[:,1])


plt.figure()

# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % lr_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest Classifier (area = %0.2f)' % rf_roc_auc)

# Plot Decision Tree ROC
plt.plot(gbc_fpr, gbc_tpr, label='Gradient Boosting Classifier (area = %0.2f)' % gbc_auc)

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()
# # logistic regression prediction
# predictions = lr.predict(X_test_f)

# output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
# output.to_csv('my_submission_lr.csv', index=False)
# print("Your submission was successfully saved!")
# #Gradient Boost Classifier
# predictions = gbc.predict(X_test_f)
# output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
# output.to_csv('my_submission_gbc.csv', index=False)
# print("Your submission was successfully saved!")
# ##Gradient Boost Classifier with original Sample
# predictions = gbc_orignalsample.predict(X_test_f)
# output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
# output.to_csv('my_submission_gbcOS.csv', index=False)
# print("Your submission was successfully saved!")
# Random Forest: orignal data result saving
predictions = rf.predict(test_new)
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
output.to_csv('my_submission_rf_orignal.csv', index=False)
print("Your submission was successfully saved!")
# gbu , no scaling
predictions = gbc.predict(test_new)
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
output.to_csv('my_submission_gbc.csv', index=False)
print("Your submission was successfully saved!")
# lr
predictions = lr.predict(test_new)
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
output.to_csv('my_submission_lru.csv', index=False)
print("Your submission was successfully saved!")
# gbc_u
predictions = gbc_u.predict(X_test_f)
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
output.to_csv('my_submission_gbcu.csv', index=False)
print("Your submission was successfully saved!")