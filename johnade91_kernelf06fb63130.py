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
train_df  = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
total_df = pd.concat([train_df, test_df], ignore_index=True, sort  = False)

print('Train:',train_df.shape)
print('Test:',test_df.shape)
print('Total:',total_df.shape)
print(total_df.isna().sum())

#Separate titles out from the names

total_df['Title'] = total_df.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
total_df['Title']

#create a mapping of titles 

normalized_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Doctor",
    "Rev":        "Clergy",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}

#Map normalised titles to data to create Title variables

total_df['Title'] = total_df['Title'].map(normalized_titles)

total_df.head(15)

TSP_group = total_df.groupby(['Title','Sex','Pclass'])
TSP_group.Age.agg(lambda x: pd.Series.mode(x)[0])
TSP_group.Embarked.agg(lambda x: pd.Series.mode(x)[0])
TSP_group.Fare.agg(lambda x: pd.Series.mode(x)[0])
total_df['Age'] = TSP_group['Age'].apply(lambda x: x.fillna(pd.Series.mode(x)[0]))
total_df['Embarked'] = TSP_group['Embarked'].apply(lambda x: x.fillna(pd.Series.mode(x)[0]))
total_df['Fare'] = TSP_group['Fare'].apply(lambda x: x.fillna(pd.Series.mode(x)[0]))
total_df['Cabin'].fillna('Unknown', inplace = True)

print(print(total_df.isna().sum()))
print(total_df.dtypes)

total_df['Cabin'] = total_df['Cabin'].apply(lambda x: x[0][0])
# required imports
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder

# separate out training data
df_traintable = total_df[total_df['Survived'].notnull()]

#feature = ['Sex','Embarked','Title','Pclass']


# create contingency table using cross tab
def chisquare(feature,dependent):
        table = pd.crosstab(df_traintable[feature], df_traintable[dependent])
        # Get chi-square value , p-value, degrees of freedom, expected frequencies using the function chi2_contingency
        stat, p, dof, expected = chi2_contingency(table)
        # select significance value
        alpha = 0.05
        # Determine whether to reject or keep your null hypothesis
        print('significance=%.3f, p=%.3f' % (alpha, p))
        if p <= alpha:
            return print(feature,dependent,'chi^2',stat,'Variables are associated (reject H0)')
        else:
            return print(feature,dependent,'chi^2',stat,'Variables are not associated(fail to reject H0)')
        


chisquare('Sex','Survived')
chisquare('Embarked','Survived')
chisquare('Cabin','Survived')
chisquare('Title','Survived')  
chisquare('Pclass','Survived') 


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
dummylist = ['Sex','Cabin','Embarked','Title', 'Name', 'Ticket','Pclass','Fare','Parch','SibSp','Age'] 
standard_df = total_df[['Fare','Parch','SibSp','Age']]

# generate binary values using get_dummies
def dum_df(feature,total_df):
    dummy_feature = pd.get_dummies(total_df[feature],prefix=feature)
    return dummy_feature

#standardise continous variables
Scaled_Features = scaler.fit_transform(standard_df)
dummiestotal_df = pd.DataFrame(Scaled_Features, columns = ['Fare','Parch','SibSp','Age'])

#Concatinate dummy and standard variables 
dummiestotal_df2 = pd.concat([dum_df('Pclass', total_df),dum_df('Sex', total_df),dum_df('Embarked', total_df),dum_df('Title', total_df),dum_df('Cabin', total_df), total_df.drop(columns = dummylist),dummiestotal_df], axis=1)




from sklearn.model_selection import train_test_split

titanictrain_df = dummiestotal_df2[dummiestotal_df2['Survived'].notnull()]
titanictest_df = dummiestotal_df2[dummiestotal_df2['Survived'].isnull()]

X = titanictrain_df.drop(columns = ['Survived','PassengerId'])
y = titanictrain_df['Survived']

# test size automatically set to 25% of overall data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Check size of train and test split data
print('X train',X_train.shape)
print('X_test',X_test.shape)
print('y_train', y_train.shape)
print('y_test' ,y_test.shape)

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

clf = LogisticRegression()
grid_values = {'solver':['liblinear'],'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,2,5,10,25,50]}
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,scoring = 'roc_auc')
grid_clf_acc.fit(X_train, y_train)

#Predict values based on new parameters
y_pred_acc = grid_clf_acc.predict(X_test)

# New Model Evaluation metrics 
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred_acc)))
print('Precision Score : ' + str(precision_score(y_test,y_pred_acc)))
print('Recall Score : ' + str(recall_score(y_test,y_pred_acc)))
print('F1 Score : ' + str(f1_score(y_test,y_pred_acc)))
print ('ROC AUC Score : ' + str(roc_auc_score(y_test,y_pred_acc)))
print('Cross Validation Score:' ,cross_val_score(grid_clf_acc,X_train,y_train,cv=4))

#Logistic Regression (Grid Search) Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test,y_pred_acc).ravel()

print('tn:',tn,'fp:',fp, 'fn:',fn, 'tp:',tp)
#print('parameter:', grid_clf_acc.get_params)

grid_clf_acc.best_params_
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(random_state = 42, criterion = 'entropy', bootstrap=True, max_features='auto');

grid_values = {'n_estimators':[1000]}
grid_rf = GridSearchCV(rf, param_grid = grid_values,scoring = 'roc_auc')
grid_rf.fit(X_train, y_train)

#Predict values based on new parameters
y_pred_acc = grid_rf.predict(X_test)

# New Model Evaluation metrics 
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred_acc)))
print('Precision Score : ' + str(precision_score(y_test,y_pred_acc)))
print('Recall Score : ' + str(recall_score(y_test,y_pred_acc)))
print('F1 Score : ' + str(f1_score(y_test,y_pred_acc)))
print ('ROC AUC Score : ' + str(roc_auc_score(y_test,y_pred_acc)))
print('Cross Validation Score:' ,cross_val_score(grid_rf,X_train,y_train,cv=4))

#Logistic Regression (Grid Search) Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test,y_pred_acc).ravel()

print('tn:',tn,'fp:',fp, 'fn:',fn, 'tp:',tp)
#print('parameter:', grid_clf_acc.get_params)


grid_rf.best_params_
Final_LRModel = LogisticRegression(C= 2, penalty='l1', solver= 'liblinear')
Final_LRModel.fit(X,y)


# New Model Evaluation metrics 
print('Cross Validation Score:' ,cross_val_score(Final_LRModel,X,y,cv=5))
print('Cross Validation Score:' ,cross_val_score(Final_LRModel,X,y,cv=5).mean())


X_test = titanictest_df.drop(columns=['Survived','PassengerId'])

titanictest_df['Survived']= Final_LRModel.predict(X_test)

titanictest_df.shape

Final_Submission = titanictest_df[['PassengerId','Survived']]

Final_Submission.head()




