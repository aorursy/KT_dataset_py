import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
data.head()
# Checking the shape of the data
data.shape
# Checking null values
data.isnull().sum().sum()
# checking data types of the columns
data.info()
#checking for quasi constants
data.nunique()
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
X=data.drop(['blueWins', 'gameId'], axis=1)
y=data['blueWins']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
sel_rf=SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=1))

sel_rf.fit(X_train, y_train)
sel_rf.get_support()
print("Total number of features in the database: ", len(X_train.columns))
print("Total number of features after removing according to RF feature importances: ", sel_rf.get_support().sum())
print("Total features removed: ", int(len(X_train.columns)-sel_rf.get_support().sum()))
X_train_rfc=sel_rf.transform(X_train)
X_test_rfc=sel_rf.transform(X_test)
# Let's check the shape of the data now to confirm that they have 16 features now
X_train_rfc.shape, X_test_rfc.shape
def classifier_model(X_train, X_test, y_train, y_test, method, data):
    rf_clf=RandomForestClassifier(n_estimators=1000, random_state=1)
    rf_clf.fit(X_train, y_train)
    y_pred_rf=rf_clf.predict(X_test)
    score_rlf=accuracy_score(y_test, y_pred_rf)
    print("---Feature Selection method: {}---". format(method))
    print("---Checking Accuracy with {}---".format(data))
    print("The accuracy score of Random Forest:", score_rlf)
    
    
    gb_clf=GradientBoostingClassifier(n_estimators=1000, random_state=1)
    gb_clf.fit(X_train, y_train)
    y_pred_gb=gb_clf.predict(X_test)
    score_gb=accuracy_score(y_test, y_pred_gb)
    print("The accuracy score of Gradient Boosting:", score_rlf)
classifier_model(X_train_rfc, X_test_rfc, y_train, y_test, "Random Forest Feature importance", "Reduced Features")
classifier_model(X_train, X_test, y_train, y_test, "Random Forest Feature importance", "All Features")
sel_rfe=RFE(RandomForestClassifier(n_estimators=100, random_state=1),n_features_to_select=20)
sel_rfe.fit(X_train, y_train)

# Total features selected:
sel_rfe.get_support().sum()
#### Let's transform the data now;
X_train_rfe=sel_rfe.transform(X_train)
X_test_rfe=sel_rfe.transform(X_test)
classifier_model(X_train_rfe, X_test_rfe, y_train, y_test, "Recursive feature extraction with RF", "Reduced Features")
classifier_model(X_train, X_test, y_train, y_test, "Recursive feature extraction with RF", "All Features")
sel_rfe_gb=RFE(GradientBoostingClassifier(n_estimators=100, random_state=1), n_features_to_select=22)
sel_rfe_gb.fit(X_train, y_train)

X_train_rfe_gb=sel_rfe_gb.transform(X_train)
X_test_rfe_gb=sel_rfe_gb.transform(X_test)

    
classifier_model(X_train_rfe_gb, X_test_rfe_gb, y_train, y_test, "Recursive feature extraction with GB", "Reduced Features")
for index in range(14,39):
    sel_rfe_gb=RFE(GradientBoostingClassifier(n_estimators=100, random_state=1), n_features_to_select=index)
    sel_rfe_gb.fit(X_train, y_train)

    X_train_rfe_gb=sel_rfe_gb.transform(X_train)
    X_test_rfe_gb=sel_rfe_gb.transform(X_test)
    
    clf_gb=GradientBoostingClassifier(n_estimators=200, random_state=1)
    clf_gb.fit(X_train_rfe_gb, y_train)
    y_pred_gb=clf_gb.predict(X_test_rfe_gb)
    score_gb=accuracy_score(y_test, y_pred_gb)
    print("Number of features: ", index)
    print("Accuracy: ", score_gb)
    print()
sel_rfe_gb_new=RFE(GradientBoostingClassifier(n_estimators=1000, random_state=1), n_features_to_select=16)
sel_rfe_gb_new.fit(X_train, y_train)

X_train_final=sel_rfe_gb_new.transform(X_train)
X_test_final=sel_rfe_gb_new.transform(X_test)
gb_clf_1=GradientBoostingClassifier(n_estimators=400, random_state=1)

gb_clf_1.fit(X_train_final, y_train)
y_pred_gb_1=gb_clf_1.predict(X_test_final)

score_gb_1=accuracy_score(y_test, y_pred_gb_1)

print("Accuracy:" ,score_gb_1)
params_grid_gb={'n_estimators' : [100,200,400,600,1000,1200],
                'min_samples_split': [100,200,300,400],
                'min_samples_leaf' : [10,20,30,40,60,100],
                'max_depth' : [2,4,6,8],
                'learning_rate' : [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
               }
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

gridsearch_gb=RandomizedSearchCV(estimator=GradientBoostingClassifier(), param_distributions=params_grid_gb, cv=5, scoring='accuracy')

gridsearch_gb.fit(X_train_final, y_train)
gridsearch_gb.best_score_
gridsearch_gb.best_params_
#### Checking accuracy on Test set
y_pred_final_gb=gridsearch_gb.predict(X_test_final)

print("Accuracy of GBM with accuracy_scoreced features on test set", accuracy_score(y_test, y_pred_final_gb))
gridsearch_gb.fit(X_train, y_train)
gridsearch_gb.best_score_
gridsearch_gb.best_params_
#### Checking accuracy on Test set
y_pred_final_gb_all=gridsearch_gb.predict(X_test)

print("Accuracy of GBM with all features on test set", accuracy_score(y_test, y_pred_final_gb_all))
data.head()
# Let's calculate the difference of values b/w Blue and red teams in all the columns
cols=[x[4:] for x in data.columns if "blue" in x and x[4:]!= 'Wins']
cols
# Below columns to be dropped  because they are already the difference of blue and red
cols_to_drop=['GoldDiff', 'ExperienceDiff']
final_cols=[x for x in cols if x not in cols_to_drop]
final_cols
data_new=pd.DataFrame()

for col in final_cols:
    data_new[f'Diff_{col}'] =data[f'blue{col}']-data[f'red{col}']

    
# Keeping values corresponding to only Red in ['GoldDiff', 'ExperienceDiff'] i.e redGoldDiff and redExperienceDiff
for col_ in cols_to_drop:
    data_new[col_]=data[f'red{col_}']
data_new.head()
# Now split the dataset into train and test
X_new=data_new
y_new=data['blueWins']
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=1, stratify=y)   
    
### Create 2 datasets tuples in order to run the model easily on new dataset ( feature engineered) and old dataset( original)

#Originaldata
dataset_1=(X_train, X_test, y_train, y_test, 'dataset_1')

#Featureengineered data
dataset_2=(X_train_new, X_test_new, y_train_new, y_test_new, 'dataset_2')
def run_classifier(model, dataset):
    model.fit(dataset[0], dataset[2])
    y_pred=model.predict(dataset[1])
    score_=accuracy_score(dataset[3], y_pred)
    return f'{round(score_, 4)*100}%'
model_dict={ 'Decision Tree' : DecisionTreeClassifier(max_depth=6,random_state=1),
            'Random Forest' : RandomForestClassifier(n_estimators=100, random_state=1),
           'Support Vector Classification': SVC(random_state=1), 
           'Gaussian Naive Bayes': GaussianNB(),
           'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=1),
           'XG Boost Classifier': XGBClassifier()
                 
          }
for model in model_dict:
    print(f'model:{model} -accuracy: {run_classifier(model_dict[model],dataset_1)}')
for model in model_dict:
    print(f'model:{model} -accuracy: {run_classifier(model_dict[model],dataset_2)}')