# This notebook demonstrates the attempt to predict whether the passenger survived or not in 
# the Titanic event using Kaggle Titanic dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
# Check the shape of train and test files
train.shape, test.shape
train.head()
test.head()
# Let us combine both train and test for data cleaning & preparation for modeling
data = pd.concat([train,test],axis=0,ignore_index=True,sort=False)
data.shape
# Check for proper concatenation is done
data.shape[0] == train.shape[0]+ test.shape[0]
data['dtype'] = np.where((data['Survived']== 0.0) | (data['Survived']== 1.0) ,'train','test')
data.tail()
data['dtype'].value_counts()
# Adding the column names to the list
columns = data.columns.to_list()
# Checking the number of cells without values and datatype of each column
data.info()
# Create Correlation matrix for all variables
corr = data.corr()

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr,annot=True,cmap='RdYlGn',linewidth=0.1)
fig=plt.gcf()
fig.set_size_inches(12,10)
plt.show()
# Check for total no. of cells with missing values
data.isnull().sum()
#Filling the missing value for Fare & Embarked

sns.countplot(x='Embarked',data=data)
plt.show()
table = pd.crosstab(data['Survived'].dropna(),data['Embarked'])
table
import statsmodels.api as sm
import scipy as sp

value, p ,dof ,expected = sp.stats.chi2_contingency(table)

if p <= 0.05:
    print('Reject H0: Dependant')
else:
    print('Fail to reject H0: Independant')
data[data['Embarked'].isna()]
data['Embarked'].fillna('Q',inplace=True)
# Fare values
sns.distplot(data['Fare'].dropna())
plt.show()
# Filling NaN with median values
data['Fare'].fillna(np.median(data['Fare'].dropna()),inplace=True)
# Filling zero values with median values as fare price cannot zero
data['Fare'].replace(0,np.median(data['Fare']),inplace=True)
import re
data['Name'] = data['Name'].apply(lambda x: re.sub('[^A-Za-z. ]+', '',x))
data['Name'] = data['Name'].apply(lambda x: x.lower())
words = pd.DataFrame(data['Name'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0))
words = words[0].sort_values(ascending=False)
words.head(5)
# Encoding the gender sex variable
data['Sex'] = np.where(data['Sex']=='male',1,0)
# Get dummy codes for Embarked variables
dummy = pd.get_dummies(data['Embarked'])
dummy.head()
dummy.drop(columns='Q',inplace=True)
dummy.rename(columns={'C':'Embarked_C','S':'Embarked_S'},inplace=True)
dummy.head()
# Adding the dummy variables and dropping the original column from data

data = pd.merge(data,dummy,left_index=True,right_index=True)
data.drop(columns='Embarked',inplace=True)
data.head()
# Impute Age group for missing values using Iterative Imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10,random_state=0,verbose=1)
imp.fit(data.drop(columns=['PassengerId','Survived','Name','Parch','Ticket','Cabin','dtype']))
drop_col = ['PassengerId','Survived','Name','Parch','Ticket','Cabin','dtype']
data_imp = pd.DataFrame(imp.transform(data.drop(columns=drop_col)))
data_imp
data['Age_imp'] = abs(data_imp[2])
age_imp = np.median(data.loc[(data['Age'].dropna()>18.0)&(data['Sex']==1),'Age'])
# Creating user defined function to impute age value wherever it is missing
def a(x):
    if (np.isnan(x['Age'])) & (x['Age_imp'] < 18.0):
        return age_imp
    elif (np.isnan(x['Age'])) & (x['Age_imp'] >= 18.0):
        return x['Age_imp']
    else:
        return x['Age']
    
data['Age_x'] = data.apply(a,axis=1)
data.drop(columns=['Age','Age_imp'],inplace=True)
data.rename(columns={'Age_x':'Age'},inplace= True)
data.head()
# Create variable IsAlone
data['IsAlone'] = np.where(data['SibSp']+data['Parch'] == 0,1,0)
data['Fam_size'] = np.where(data['IsAlone'] == 0,data['SibSp']+data['Parch']+1,1)
data['Fam_size'].value_counts()
# Creating user defined function to assign Gender based on the age group
def f(x):
    if x['Age'] <= 15.0:
        return 'Children'
    elif (x['Age'] > 15.0) & (x['Sex'] == 1):
        return 'Adult-Male'
    else:
        return 'Adult-Female'

data['Gender'] = data.apply(f,axis=1)

# Lookup the frequency distribution table between gender and passenger class 
pd.crosstab(data['Gender'],data['Pclass'])
# 2 more user defined functions to assign each passenger with time of boarding
# based on which the passenger class got filled
def g(x):
    if (x['Embarked_S'] == 1) & (x['Embarked_C'] == 0):
        return 'first'
    elif (x['Embarked_S'] == 0) & (x['Embarked_C'] == 1):
        return 'second'
    else:
        return 'third'

def h(x):
    if x['Pclass'] == 1:
        return 'upper'
    elif x['Pclass'] == 2:
        return 'middle'
    else:
        return 'lower'
    
data['Boarding'] = data.apply(g,axis=1) +'-' + data.apply(h,axis=1)
levels = pd.DataFrame(data['Boarding'].value_counts())
levels
level_list = levels.index.to_list()
level_dict = {'lower':['Deck_G','Deck_F','Deck_E'],
              'middle':['Deck_E','Deck_D'],
              'upper':['Deck_C','Deck_B','Deck_A']}
level_list
level_dict.get(level_list[0].split('-')[1])

Fare1 = pd.DataFrame(columns=None)

for level in level_list:
    Fare = pd.DataFrame(data[data['Boarding'] == level]['Fare'])
    deck = level_dict.get(level.split('-')[1])
    Fare['Deck'] = pd.cut(np.array(Fare['Fare']),len(deck), labels=deck)
    Fare1 = pd.concat([Fare1,Fare],axis=0,sort=False)
Fare1['Deck'].value_counts()
# Deck class is assigned for each passenger based on the fair price,embarked station,Pclass
Fare1.head()
# Merging this data with original dataframe
data = pd.merge(data,Fare1['Deck'],left_index=True,right_index=True)
data.drop(columns=['Name','Ticket','Cabin','Boarding'],inplace=True) # remove unwanted columns
data.head()
# Create dummy variables for categorical values
dummy1 = pd.get_dummies(data['Gender'])
dummy1.drop(columns='Children',inplace=True)

dummy2 = pd.get_dummies(data['Deck'])
dummy2.drop(columns='Deck_A',inplace=True)

dummies = pd.merge(dummy1,dummy2,left_index=True,right_index=True)
data = pd.merge(data,dummies,left_index=True,right_index=True)

data.drop(columns=['Gender','Deck'],inplace=True)
data.head()
# Box Cox transformation for Fare value, since it is highly skewed
from sklearn.preprocessing import power_transform

data['Fare'] = power_transform(np.array(data['Fare']).reshape(-1,1),method='box-cox')
data.head()
# Split the train and test to build ML models
train_new = data[data['dtype']=='train'].drop(columns='dtype')
test_new = data[data['dtype']=='test'].drop(columns=['dtype','Survived'])
test_new.head()
corr_n = data.corr()

sns.heatmap(corr_n,annot=True,cmap='RdYlGn',linewidth=0.1)
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()
# Split the dataset

X = train_new.loc[:,train_new.columns != 'Survived'].copy()
y = train_new.loc[:,train_new.columns == 'Survived'].copy()

X.drop(columns='PassengerId',inplace=True)
# Run the logistic regression model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score,plot_roc_curve

logit_model = sm.Logit(y,X)
result = logit_model.fit()
print(result.summary())
from sklearn.model_selection import GridSearchCV

penalty = ['l1','l2']
C = np.logspace(0,4,10)
hyperparameters = dict(C= C,penalty = penalty)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3,random_state=0)

lr = LogisticRegression(max_iter=200,solver='liblinear')
clf = GridSearchCV(lr, hyperparameters, cv=5, verbose=0,scoring='recall')

model_lr = clf.fit(X_train,y_train.values.ravel())
print('Best penalty:',model_lr.best_estimator_.get_params()['penalty'])
print('Best C:', round(model_lr.best_estimator_.get_params()['C'],2))
y_pred_lr = model_lr.predict(X_test)

confusion1 = pd.DataFrame(pd.crosstab(y_test['Survived'],y_pred_lr))
confusion1
plot_roc_curve(model_lr,X_test,y_test)
plt.show()
# Model Evaluation metrics 
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
print('Accuracy Score : {}'.format(round(accuracy_score(y_test,y_pred_lr),2)))
print('Precision Score : {}'.format(round(precision_score(y_test,y_pred_lr),2)))
print('Recall Score : {}'.format(round(recall_score(y_test,y_pred_lr),2)))
print('F1 Score : {}'.format(round(f1_score(y_test,y_pred_lr),2)))
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3,random_state=0)

rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train.values.ravel())
# Getting the best parameters from the randomised search
rf_random.best_params_
# Build Random forest model
best_rf = rf_random.best_estimator_

y_pred_rfs = best_rf.predict(X_test)

confusion2 = pd.DataFrame(pd.crosstab(y_test['Survived'],y_pred_rfs))
confusion2
print('Accuracy Score : {}'.format(round(accuracy_score(y_test,y_pred_rfs),2)))
print('Precision Score : {}'.format(round(precision_score(y_test,y_pred_rfs),2)))
print('Recall Score : {}'.format(round(recall_score(y_test,y_pred_rfs),2)))
print('F1 Score : {}'.format(round(f1_score(y_test,y_pred_rfs),2)))
plot_roc_curve(best_rf,X_test,y_test)
plt.show()
# Variable importance table to look for most important variables

var_imp1 = pd.DataFrame({'Variable': X.columns,
                        'Importance':best_rf.feature_importances_}).sort_values('Importance', ascending=False)
var_imp1.head(10)
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3,random_state=0)

# Create the model with 100 trees
model_rf = RandomForestClassifier(n_estimators=100, 
                               random_state= 0, 
                               max_features = 'sqrt',
                               n_jobs=-1, verbose = 1)

# Fit on training data
model_rf.fit(X_train, y_train.values.ravel())

n_nodes = []
max_depths = []

for ind_tree in model_rf.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
    
print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')
ytrain_rf_pred = model_rf.predict(X_train)
ytrain_rf_prob = model_rf.predict_proba(X_train)[:, 1]

ytest_rf_pred = model_rf.predict(X_test)
ytest_rf_prob = model_rf.predict_proba(X_test)[:, 1]
confusion3 = pd.crosstab(y_test['Survived'],ytest_rf_pred)
confusion3
print('Accuracy Score : {}'.format(round(accuracy_score(y_test,ytest_rf_pred),2)))
print('Precision Score : {}'.format(round(precision_score(y_test,ytest_rf_pred),2)))
print('Recall Score : {}'.format(round(recall_score(y_test,ytest_rf_pred),2)))
print('F1 Score : {}'.format(round(f1_score(y_test,ytest_rf_pred),2)))
plot_roc_curve(model_rf,X_test,y_test)
plt.show()
model_prob1 = model_lr.predict_proba(test_new.drop(columns='PassengerId'))[:,1]
model_prob2 = best_rf.predict_proba(test_new.drop(columns='PassengerId'))[:,1]
model_prob3 = model_rf.predict_proba(test_new.drop(columns=['PassengerId']))[:,1]
ensemble_prob = pd.DataFrame({'PassengerId': test['PassengerId'],
                              'LogisticR': model_prob1,
                              'RF_search': model_prob2,
                              'RandomF': model_prob3})
ensemble_prob.head()
ensemble_prob['Prob'] = ensemble_prob.apply(lambda x: (x['LogisticR'] + x['RF_search'] + x['RandomF'] )/3,axis=1)
ensemble_prob['Survived'] = np.where(ensemble_prob['Prob']>= 0.5,1,0)
ensemble_prob.head()
# Preparing the submission file
submission = pd.DataFrame({'PassengerId': ensemble_prob['PassengerId'],
                           'Survived': ensemble_prob['Survived']})
submission['Survived'].value_counts()
submission