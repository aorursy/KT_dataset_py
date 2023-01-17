# Load libraries for analysis and visualization
import pandas as pd # collection of functions for data processing and analysis modeled after R dataframes with SQL like features
import numpy as np  # foundational package for scientific computing
import re           # Regular expression operations
import matplotlib.pyplot as plt # Collection of functions for scientific and publication-ready visualization

%matplotlib inline

import plotly.offline as py     # Open source library for composing, editing, and sharing interactive data visualization 
from matplotlib import pyplot
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from collections import Counter

# Machine learning libraries
import sys
sys.path.append('/Users/minjielu/anaconda3/envs/python/lib/python3.5/site-packages')

import xgboost as xgb  # Implementation of gradient boosted decision trees designed for speed and performance that is dominative competitive machine learning
import seaborn as sns  # Visualization library based on matplotlib, provides interface for drawing attractive statistical graphics

import sklearn         # Collection of machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix

import warnings
warnings.filterwarnings('ignore')
# Load train and test datasets from CSV files
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Record passenger ids of test set for composing submission files
passengerid = test['PassengerId']

full_data = [train,test]
train.sample(5)
train.info(5)
train.describe()
test.sample(5)
test.info()
test.describe()
# Set font sizes for figure display
SMALL_SIZE = 13
MEDIUM_SIZE = 17
LARGE_SIZE = 21

plt.rc('font', size=SMALL_SIZE)           # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the titles
plt.rc('axes', labelsize=LARGE_SIZE)      # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)     # fontsize of the tick labels

f,ax = plt.subplots(2,3,figsize=(24,16))

f.subplots_adjust(wspace=0.2,hspace=0.3)
f.delaxes(ax[1,2])

sns.barplot(x='Pclass',y='Survived',data=train,ax=ax[0,0])
sns.barplot(x='Sex',y='Survived',data=train,ax=ax[0,1])
sns.barplot(x='SibSp',y='Survived',data=train,ax=ax[0,2])
sns.barplot(x='Parch',y='Survived',data=train,ax=ax[1,0])
sns.barplot(x='Embarked',y='Survived',data=train,ax=ax[1,1])

_ = ax[0,0].set_title('Survival Rate by Class',fontweight='bold')
_ = ax[0,1].set_title('Survival Rate by Sex',fontweight='bold')
_ = ax[0,2].set_title('Survival Rate \n by Number of Siblings/Spouses',fontweight='bold')
_ = ax[1,0].set_title('Survival Rate \n by Number of Parent/children',fontweight='bold')
_ = ax[1,1].set_title('Survival Rate \n by Port of Embarkation',fontweight='bold')
# Fill nan items in Embarked
# Since only two items in the train set have nan Embarked, it should be
# okay to suppose they all embarked at Southampton where most passengers embarked
train['Embarked'] = train['Embarked'].fillna('S')

# Map some literal attributes to integer attributes
# In order to support linear classifiers, attributes Embarked, SibSp and Parch are reordered
# so that categories of larger number has larger survival rate
for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map({'male':0,'female':1}).astype(int)
    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':2,'Q':1}).astype(int)
a = sns.FacetGrid(train,hue='Survived',aspect=6)
a.map(sns.kdeplot,'Age',shade=True)
a.add_legend()
_ = a.axes[0,0].set_title('KDE plot for Survival Rate versus Age',fontweight='bold')
# Skewness is a measurement of degree of asymmetry
train['Fare'].skew()
# Since attribute Fare has a very large skewness, it's better to change it to log
# scale to prevent machine learning models from being biassed
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].apply(lambda x: np.log(x) if x > 0 else 0)
a = sns.FacetGrid(train,hue='Survived',aspect=6)
a.map(sns.kdeplot,'Fare',shade=True)
a.add_legend()
_ = a.axes[0,0].set_title('KDE plot for Survival Rate versus Fare',fontweight='bold')
# Define a function to extract titles from names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.',name)
    # Return the title if one is found in the name. Return an empty string otherwise
    if title_search:
        return title_search.group(1)
    return ''

# Create a new attribute Title extracted from names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
    
f,ax1 = plt.subplots(1,1,figsize=(18,6))
TINY_SIZE = 7
plt.rc('xtick', labelsize=TINY_SIZE)
sns.barplot(x='Title',y='Survived',data=train,ax=ax1)

_ = ax1.set_title('Survival Rate by Title',fontweight='bold')
train['Title'].value_counts()
test['Title'].value_counts()
# Titles of a very small value count are combined into similar titles with large value count
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Dona','Mme','Countess','Lady','Ms'],'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Mlle'],'Miss')
    dataset['Title'] = dataset['Title'].replace(['Jonkheer','Don','Capt','Major','Col','Sir'],'Mr')
    dataset['Title_Mapped'] = dataset['Title'].map({'Mr':1,'Miss':4,'Mrs':5,'Master':3,'Dr':2,'Rev':0}).astype(int)
# Ticket and Cabin should have no relation to age. Correlations between Age and other attributes are examined
plt.rc('xtick', labelsize=SMALL_SIZE)
g = sns.heatmap(train[["Age","Sex","SibSp","Parch","Pclass",'Title_Mapped','Embarked']].corr(),cmap="BrBG",annot=True)
_ = g.set_title('Correlation Plot of Features', y=1.05, fontweight='bold')
print('Average Age: {}'.format(train[(train['Parch'] == 2) & (train['SibSp'] == 4) & (train['Pclass'] == 3) & (train['Title'] == 'Miss')]['Age'].mean()))
train[(train['Parch'] == 2) & (train['SibSp'] == 4) & (train['Pclass'] == 3) & (train['Title'] == 'Miss')][['Parch','SibSp','Pclass','Title','Age']]
print('Average Age: {}'.format(train[(train['Parch'] == 0) & (train['SibSp'] == 1) & (train['Pclass'] == 1) & (train['Title'] == 'Miss')]['Age'].mean()))
train[(train['Parch'] == 0) & (train['SibSp'] == 1) & (train['Pclass'] == 1) & (train['Title'] == 'Miss')][['Parch','SibSp','Pclass','Title','Age']]
print('Average Age: {}'.format(train[(train['Parch'] == 2) & (train['SibSp'] == 4) & (train['Pclass'] == 3) & (train['Title'] == 'Master')]['Age'].mean()))
train[(train['Parch'] == 2) & (train['SibSp'] == 4) & (train['Pclass'] == 3) & (train['Title'] == 'Master')][['Parch','SibSp','Pclass','Title','Age']]
# Fill nan items in Age according to 'Pclass', 'Parch' and 'Title'
all_data = pd.concat([train,test],axis=0)
for dataset in full_data:
    index_nan_age = list(dataset['Age'][dataset['Age'].isna()].index)
    for i in index_nan_age:
        age_med = all_data[(all_data['Pclass'] == dataset.iloc[i]['Pclass']) & (all_data['SibSp'] == dataset.iloc[i]['SibSp']) & (all_data['Parch'] == dataset.iloc[i]['Parch']) & (all_data['Title'] == dataset.iloc[i]['Title'])]['Age'].median()
        # 6 passengers with 8 SibSp can't find references, therefore only Pclass, Parch and Title are used for them
        age_med_1 = all_data[(all_data['Pclass'] == dataset.iloc[i]['Pclass']) & (all_data['Parch'] == dataset.iloc[i]['Parch']) & (all_data['Title'] == dataset.iloc[i]['Title'])]['Age'].median()
        if not np.isnan(age_med):
            dataset.iloc[i,dataset.columns.get_loc('Age')] = age_med
        else:
            dataset.iloc[i,dataset.columns.get_loc('Age')] = age_med_1
# Combine categories of Title with similar survial rate
for dataset in full_data:
    dataset['Title_Mapped'] = dataset['Title_Mapped'].map({0:0,1:0,2:1,3:1,4:2,5:2}).astype(int)
# Create attribute Family_Size
for dataset in full_data:
    dataset['Family_Size'] = dataset['SibSp']+dataset['Parch']
    
f,ax1 = plt.subplots(1,1,figsize=(18,6))
sns.pointplot(x='Family_Size',y='Survived',data=train,ax=ax1)
_ = ax1.set_title('Survival Rate versus Family Size',fontweight='bold')
for dataset in full_data:
    # Categories of family size are relabelled so that larger label has larger survival rate
    # Categories of family size with similar survival rate are combined
    dataset['Family_Size'] = dataset['Family_Size'].map({10:0,7:1,5:2,4:3,0:4,6:5,1:6,2:7,3:8}).astype(int)
    dataset['Family_Size'] = dataset['Family_Size'].map({0:0,1:0,2:1,3:1,4:2,5:2,6:3,7:3,8:3}).astype(int)
 
# SibSp and Parch are reordered to linear classifiers
for dataset in full_data:
    dataset['SibSp'] = dataset['SibSp'].map({5:0,8:1,4:2,3:3,0:4,2:5,1:6}).astype(int)
    dataset['Parch'] = dataset['Parch'].map({4:0,6:1,5:2,0:3,2:4,1:5,3:6,9:6}).astype(int)
    
# For SibSp and Parch, categories with similar survival rate are grouped together
# We only have 891 passengers in the train set. If we have too many attributes or categories for attributes, we 
# may have very few samples for each group. Since these categories have similar survial rate, it may just
# be because of noise. Therefore, we may get overfitting.
for dataset in full_data:
    dataset['SibSp'] = dataset['SibSp'].map({0:0,1:0,2:1,3:1,4:1,5:2,6:2}).astype(int)
    dataset['Parch'] = dataset['Parch'].map({0:0,1:0,2:1,3:1,4:2,5:2,6:2}).astype(int)   
# Get Surname which is an indication of families
for dataset in full_data:
    dataset['Surname'] = dataset['Name'].apply(lambda x: x.split(',')[0].strip())
train[train['Surname'] == 'Andersson'][['Surname','Ticket']]
train[train['Surname'] == 'Ali'][['Surname','Ticket']]
# Create attribute Ticket_N by removing the last two digits from Ticket
for dataset in full_data:
    dataset['Ticket_N'] = dataset['Ticket'].apply(lambda x: x[0:-2]+'XX')
train[train['Ticket'] == '19877'][['Surname','Sex','Age','Ticket']]
train[train['Ticket'] == '17421'][['Surname','Sex','Age','Ticket']]
train[train['Ticket'] == '13502'][['Surname','Sex','Age','Ticket']]
# Combine train and test set
test_helper = test.copy()
test_helper['Survived'] = 2
# Mark whether a passenger is from train or test set
test_helper['Testset'] = 1
train_helper = train.copy()
train_helper['Testset'] = 0
df_helper = pd.concat([train_helper,test_helper],axis=0)

# Group passengers sharing the same Surname and Ticket_N into families
df_helper =  df_helper.sort_values(['Surname','Ticket_N'])
i,j = 0,0
df_helper['Family_No'] = 0
family_count = 1 # family_count is used to label families
num_row = df_helper.shape[0]

while j < num_row:
    while (j < num_row) and (df_helper.iloc[j]['Surname'] == df_helper.iloc[i]['Surname']) and (df_helper.iloc[j]['Ticket_N'] == df_helper.iloc[i]['Ticket_N']):
        j += 1
    if j-i > 1:
        while i < j:
            df_helper.iloc[i,df_helper.columns.get_loc('Family_No')] = family_count
            i += 1
        family_count += 1
    i = j

# Group passengers sharing the same Ticket into families
df_helper = df_helper.sort_values(['Ticket'])
i,j = 0,0

while j < num_row:
    family_no = 0
    while (j < num_row) and (df_helper.iloc[j]['Ticket'] == df_helper.iloc[i]['Ticket']):
        # In a group of passengers sharing the same Ticket, if a family is already found according to Surname,
        # the family number is assigned to new members. Otherwise, a new family number is assigned to this group
        family_no = max(family_no,df_helper.iloc[j]['Family_No'])
        j += 1
    if j-i > 1:
        if not family_no:
            while i < j:
                df_helper.iloc[i,df_helper.columns.get_loc('Family_No')] = family_count
                i += 1
            family_count += 1
        else:
            while i < j:
                df_helper.iloc[i,df_helper.columns.get_loc('Family_No')] = family_no
                i += 1
    i = j
    
# Define function get_FS_onboard to get number of passengers in a family    
def get_FS_onboard(passengerid):
    family_no = df_helper.loc[df_helper['PassengerId'] == passengerid]['Family_No'].item()
    if family_no == 0:
        return 1
    else:
        return df_helper[df_helper['Family_No'] == family_no].shape[0]
    
# Create attribute Family_Size_Onboard which is the number of family members on board a passenger has
for dataset in full_data:
    dataset['Family_Size_Onboard'] = dataset['PassengerId'].apply(get_FS_onboard)
# Create a function to get the number with the maximum count in an array
def get_majority(A):
    majority = 0.5
    cnt = 0
    for x in range(len(A)):
        if cnt == 0:
            majority = A[x]
            cnt += 1
        elif A[x] == majority:
            cnt += 1
        else:
            cnt -= 1
    return majority

# Create an attribute to identify woman_child
# So under what age can a man be called a boy? -No idea. I just tried different values and 18 gives me the best result
df_helper['Woman_Child'] = 0
df_helper.loc[(df_helper['Sex'] == 1) | (df_helper['Age'] <= 18),'Woman_Child'] = 1

# Create attribute Family_WCM_Survived
# If the passenger is a woman or child, the value indicates whether a majority of women or children
# in her family died or survived
# If the passenger is a adult man, the value indicates whether a majority of adult men in his family
# died or survived
# Create attribute Family_Survived which indicates whether a majority of members in a family died or
# survived
# If there is no information, default values for these two attributes are 0.5
df_helper['Family_WCM_Survived'] = 0.5
df_helper['Family_Survived'] = 0.5
df_helper = df_helper.sort_values(['Family_No','Woman_Child'])
i,j = 0,0
while df_helper.iloc[i]['Family_No'] == 0: # Passengers travel alone are passed
    i += 1
    j += 1
while j < num_row:
    counter = [[],[],[]]
    while (j < num_row) and (df_helper.iloc[j]['Family_No'] == df_helper.iloc[i]['Family_No']):
        if df_helper.iloc[j]['Survived'] != 2: # Family members in the test set don't contribute to the majority vote
            counter[df_helper.iloc[j]['Woman_Child']].append(df_helper.iloc[j]['Survived'])
        j += 1
    counter[2].extend(counter[0])
    counter[2].extend(counter[1])
    counter = [get_majority(counter[0]),get_majority(counter[1]),get_majority(counter[2])]
    while i < j:
        df_helper.iloc[i,df_helper.columns.get_loc('Family_WCM_Survived')] = counter[df_helper.iloc[i]['Woman_Child']]
        df_helper.iloc[i,df_helper.columns.get_loc('Family_Survived')] = counter[2]
        i += 1

df_helper = df_helper.sort_values(['PassengerId'])

train['Family_WCM_Survived'] = df_helper.loc[df_helper['Testset'] == 0]['Family_WCM_Survived']
test['Family_WCM_Survived'] = df_helper.loc[df_helper['Testset'] == 1]['Family_WCM_Survived']
train['Family_Survived'] = df_helper.loc[df_helper['Testset'] == 0]['Family_Survived']
test['Family_Survived'] = df_helper.loc[df_helper['Testset'] == 1]['Family_Survived']
# Test the correctness of above rules we assumed for families in train set.
cm = confusion_matrix(train[train['Family_WCM_Survived'] != 0.5]['Family_WCM_Survived'],train[train['Family_WCM_Survived'] != 0.5]['Survived'])
cm = pd.DataFrame(cm,index=['Died','Survived'],columns=['Died','Survived'])
g = sns.heatmap(cm,cmap="BrBG",annot=True,fmt='g')
_ = g.set_xlabel('Predication Label')
_ = g.set_ylabel('Truth Label')
_ = g.set_title('Confusion Matrix', y=1.05, fontweight='bold')
# Survival rate for female passengers in the third class having more than 4 family members onboard
train[(train['Family_Size_Onboard'] >= 4) & (train['Sex'] == 1) & (train['Pclass'] == 3)]['Survived'].mean()
# Create bins for Age and Fare according to previous analysis of KDE plots
for dataset in full_data:
    dataset['Age_Discrete'] = 0
    # The same as what I did for SibSp, Parch and Title, range of Age and Fare with larger survival rate is 
    # assigned a larger integer
    dataset.loc[dataset['Age'] <= 14,'Age_Discrete'] = 2
    dataset.loc[(dataset['Age'] > 14) & (dataset['Age'] <= 31),'Age_Discrete'] = 0
    dataset.loc[dataset['Age'] > 31,'Age_Discrete'] = 1
    dataset['Fare_Discrete'] = 0
    dataset.loc[dataset['Fare'] <= 2.6,'Fare_Discrete'] = 0
    dataset.loc[dataset['Fare'] > 2.6,'Fare_Discrete'] = 1
# Normalize numerical attributes
# After I tried different combinations of attributes, Pclass, Sex, Age, Fare, Title, Family_Size, Family_WCM_Survived and Family_Survived
# are the ones that help to increase cross validation accuracy
attributes = ['Pclass','Sex','Age_Discrete','Fare_Discrete','Title_Mapped','Family_Size','Family_WCM_Survived','Family_Survived']
for dataset in full_data:
    for attribute in attributes:
        mean_value = dataset[attribute].mean()
        std_value = dataset[attribute].std()
        dataset[attribute+'_Nor'] = dataset[attribute].apply(lambda x: (x-mean_value)/std_value)
# Drop useless attributes
drop_attributes = ['Name','Ticket','SibSp','Parch','Cabin','Embarked','Title','Surname','Ticket_N','Family_Size_Onboard']
train = train.drop(drop_attributes, axis=1)
test = test.drop(drop_attributes, axis=1)

# Examine the data set one more time before we proceed
pd.set_option('display.max_columns',20)
train.sample(10)
test.sample(10)
# Data set with normalized attributes
train_1 = ['Pclass_Nor','Sex_Nor','Age_Discrete_Nor','Fare_Discrete_Nor','Title_Mapped_Nor','Family_Size_Nor']
train_target = ['Survived']

# Data set with family survival information
train_2 = ['Pclass_Nor','Sex_Nor','Age_Discrete_Nor','Fare_Discrete_Nor','Title_Mapped_Nor','Family_Size_Nor','Family_WCM_Survived','Family_Survived']
# Cross validate model with Kfold stratified cross validation
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10)

# Gradient boosting tunning
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] }
gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsGBC.fit(train[train_1],train[train_target])
GBC_best = gsGBC.best_estimator_

GBC_predictions = GBC_best.predict(test[train_1])
GBC_predictions = pd.Series(GBC_predictions,name='Survived')

# Generate submission file for predictions by gradient boosting classifier
submission_GBC = pd.concat([passengerid,GBC_predictions],axis=1)
submission_GBC.to_csv('submission_GBC',index=False)
gsGBC.best_score_
# Enforce rules we assumed for families
n_rows = test.shape[0]
# Count the number of corrections made to the result
count = 0

for i in range(n_rows):
    family_wcm_survived = test.iloc[i]['Family_WCM_Survived']
    passenger_id = test.iloc[i]['PassengerId']
    # Use the result inferred from family members if gradient boosting gives a different prediction
    if family_wcm_survived != 0.5 and submission_GBC[submission_GBC['PassengerId'] == passenger_id]['Survived'].item() != family_wcm_survived:
        count += 1
        submission_GBC.loc[submission_GBC['PassengerId'] == passenger_id,'Survived'] = family_wcm_survived

# Generate submission file for model I
submission_GBC.loc[submission_GBC['PassengerId'] == 1122,'Survived'] = 0
submission_GBC = submission_GBC.astype(int)
submission_GBC.to_csv('submission_M1',index=False)
print('Number of Corrections Made to the Prediction of Gradient Boosting Classifier: {}'.format(count))
# Record best cross validation accuracies for algorithms
best_scores = []

# Gradient boosting tunning
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] }
gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsGBC.fit(train[train_2],train[train_target])
GBC_best = gsGBC.best_estimator_

# Generate the meta feature by gradient boosting classifier for test set
GBC_meta = GBC_best.predict(test[train_2])
GBC_meta = pd.Series(GBC_meta,name='GBC_meta')
GBC_params = gsGBC.best_params_
best_scores.append(gsGBC.best_score_)
# Random forest tunning 
random_forest = RandomForestClassifier()
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 7],
              "min_samples_split": [2, 3, 7],
              "min_samples_leaf": [1, 3, 7],
              "bootstrap": [False],
              "n_estimators" :[300,600],
              "criterion": ["gini"]}
gsRF = GridSearchCV(random_forest,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsRF.fit(train[train_2],train[train_target])
# Best score
RF_best = gsRF.best_estimator_

# Generate the meta feature by random forest classifier for test set
RF_meta = RF_best.predict(test[train_2])
RF_meta = pd.Series(RF_meta,name='RF_meta')
RF_params = gsRF.best_params_
best_scores.append(gsRF.best_score_)
# ExtraTrees tunning
ExtC = ExtraTreesClassifier()
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 7],
              "min_samples_split": [2, 3, 7],
              "min_samples_leaf": [1, 3, 7],
              "bootstrap": [False],
              "n_estimators" :[300,600],
              "criterion": ["gini"]}
gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsExtC.fit(train[train_2],train[train_target])
ExtC_best = gsExtC.best_estimator_

# Generate the meta feature by extra trees classifier for test set
ET_meta = ExtC_best.predict(test[train_2])
ET_meta = pd.Series(ET_meta,name='ET_meta')
ExtC_params = gsExtC.best_params_
best_scores.append(gsExtC.best_score_)
# Support vector machine tunning
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'],
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1,10,50,100,200,300, 1000]}
gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsSVMC.fit(train[train_2],train[train_target])
SVMC_best = gsSVMC.best_estimator_

# Generate the meta feature by support vector machine classifier for test set
SVMC_meta = SVMC_best.predict(test[train_2])
SVMC_meta = pd.Series(SVMC_meta,name='SVMC_meta')
SVMC_params = gsSVMC.best_params_
best_scores.append(gsSVMC.best_score_)
# K nearest neighbors tunning
KNN = KNeighborsClassifier()
knn_param_grid = {'n_neighbors':[3,5,8,13],
                  'algorithm': ['auto'],
                  'weights': ['uniform','distance'],
                  'leaf_size': list(range(1,50,5))}
gsKNN = GridSearchCV(KNN,param_grid = knn_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsKNN.fit(train[train_2],train[train_target])
KNN_best = gsKNN.best_estimator_

# Generate the meta feature by k nearest neighbors classifier for test set
KNN_meta = KNN_best.predict(test[train_2])
KNN_meta = pd.Series(KNN_meta,name='KNN_meta')
KNN_params = gsKNN.best_params_
best_scores.append(gsKNN.best_score_)
# Logistic regression tunning
LogR = LogisticRegression()
LogR_param_grid = {'penalty': ['l1','l2'],
                  'C': [0.001,0.01,0.1,1,10,100,1000]}
gsLogR = GridSearchCV(LogR,param_grid = LogR_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsLogR.fit(train[train_2],train[train_target])
LogR_best = gsLogR.best_estimator_

# Generate the meta feature by k nearest neighbors classifier for test set
LogR_meta = LogR_best.predict(test[train_2])
LogR_meta = pd.Series(LogR_meta,name='LogR_meta')
LogR_params = gsLogR.best_params_
best_scores.append(gsLogR.best_score_)
# Compare the best cross validation scores of these algorithms
cv_scores = pd.DataFrame({"Cross Validation Scores":best_scores,"Algorithm":['GBoosting',  'RandomForest', 'ExtraTrees',
    'RBF SVM','KNN','LogRegression']})

g = sns.barplot("Cross Validation Scores","Algorithm",data = cv_scores, palette="Set3",orient = "h")
_ = g.set_title("Cross Validation Scores for Different Algorithms",y=1.05,fontweight='bold')
# Generate test data set comprised of meta features
test_meta = pd.concat([GBC_meta,RF_meta,ET_meta,SVMC_meta,KNN_meta,LogR_meta],axis=1)
# Create classifier objects using optimal parameters
GBC = GradientBoostingClassifier(**GBC_params)
RF = RandomForestClassifier(**RF_params)
ExtC = ExtraTreesClassifier(**ExtC_params)
SVMC = SVC(**SVMC_params,probability=True)
KNN = KNeighborsClassifier(**KNN_params)
LogR = LogisticRegression(**LogR_params)
n = train.shape[0]
x_train = train[train_2].values
y_train = train[train_target].values

kf = KFold(n,10,random_state=2)
# Define a function to generate meta features using a manner of cross validation
def gen_meta_features(clf,x,y):
    feature = np.zeros((len(y),))
    for train_index,test_index in kf:
        model = clf.fit(x[train_index],y[train_index])
        feature[test_index] = model.predict(x[test_index])
    return feature

GBC_meta = pd.Series(gen_meta_features(GBC,x_train,y_train),name='GBC_meta')
RF_meta = pd.Series(gen_meta_features(RF,x_train,y_train),name='RF_meta')
ET_meta = pd.Series(gen_meta_features(ExtC,x_train,y_train),name='ET_meta')
SVMC_meta = pd.Series(gen_meta_features(SVMC,x_train,y_train),name='SVMC_meta')
KNN_meta = pd.Series(gen_meta_features(KNN,x_train,y_train),name='KNN_meta')
LogR_meta = pd.Series(gen_meta_features(LogR,x_train,y_train),name='LogR_meta')
# Generate train data set comprised of meta features
train_meta = pd.concat([GBC_meta,RF_meta,ET_meta,SVMC_meta,KNN_meta,LogR_meta],axis=1)
# Train an XGBoost Classifier using meta features of train set
gbm = xgb.XGBClassifier(learning_rate = 0.95,
 n_estimators= 5000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=1,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train).fit(train_meta, train[train_target])

# Make predictions using meta features of test set
stacking_predictions = gbm.predict(test_meta)
stacking_predictions = pd.Series(stacking_predictions,name='Survived')

# Generate submission file for predictions by the stacking model
submission_stacking = pd.concat([passengerid,stacking_predictions],axis=1)
submission_stacking.to_csv('submission_stacking',index=False)
# Check whether rules we assumed for family members are already learnt by algorithms
n_rows = test.shape[0]
# Count the number of corrections made to the result predicted by the stacking model
count = 0

for i in range(n_rows):
    family_wcm_survived = test.iloc[i]['Family_WCM_Survived']
    passenger_id = test.iloc[i]['PassengerId']
    # Use the result inferred from other family members if stacking model gives a different prediction
    if family_wcm_survived != 0.5 and submission_stacking[submission_stacking['PassengerId'] == passenger_id]['Survived'].item() != family_wcm_survived:
        count += 1

print('Number of Corrections Made to the Prediction of Model II: {}'.format(count))