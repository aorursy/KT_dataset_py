import math
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.drop(['Name', 'Ticket'], axis=1, inplace=True)
test.head()
data.drop(['Name', 'Ticket'], axis=1, inplace=True)
data.head()
data.shape
data.describe()
data.isnull().sum()
bool_data = data.isnull()

sns.countplot(bool_data[data.Survived == 0].Cabin)
def fill_cabin(columns):
    cabin = columns[0]
    
    try:
        cabin = float(cabin)
        return 0
    except:
        return 1
    
data['cabin_details'] = data[['Cabin']].apply(fill_cabin, axis=1)
data.head()
test['cabin_details'] = test[['Cabin']].apply(fill_cabin, axis=1)
test.head()
test.drop('Cabin', axis=1, inplace=True)
data.drop('Cabin', axis=1, inplace=True)
data.head()
sns.boxplot(data.Sex, data.Age)
#fill the null values of the Age by mean values as the ages are not varying a lot.
backup_data = pd.concat([data, test], axis=0)
male_value = round(backup_data[backup_data.Sex == 'male'].Age.mean())
female_value = round(backup_data[backup_data.Sex == 'female'].Age.mean())

def fill_age(columns):
    age = columns[0]
    gender = columns[1]
    
    if math.isnan(age):
        if gender == 'male':
            return male_value
        else:
            return female_value
    else:
        return age
    
data['Age'] = data[['Age', 'Sex']].apply(fill_age, axis=1)
test['Age'] = test[['Age', 'Sex']].apply(fill_age, axis=1)
sns.countplot(data.Embarked)
data.Embarked.fillna('S', inplace=True)
data.head()
test.fillna(test.Fare.mean(), inplace=True)
def alone(cols):
    sib = cols[0]
    par = cols[0]
    
    if (sib + par) == 0:
        return 1
    else:
        return 0
    
data['alone'] = data[['SibSp', 'Parch']].apply(alone, axis=1)
test['alone'] = test[['SibSp', 'Parch']].apply(alone, axis=1)
def encode_sex(col):
    sex = col[0]
    if sex == 'male':
        return 1
    else:
        return 0
    
#1 for male and 0 for female.
data['Sex'] = data[['Sex']].apply(encode_sex, axis=True)
test['Sex'] = test[['Sex']].apply(encode_sex, axis=True)
label = LabelEncoder()
x = data.copy()
y = test.copy()
x['Embarked']=label.fit_transform(x['Embarked'])
y['Embarked'] = label.fit_transform(y['Embarked'])
data = x.copy()
test = y.copy()
sns.pairplot(data)
#distribution of Fare zoomed in.
sns.distplot(data['Fare'])
#add a constant c to avoid logging zeros.
C = 2
data['Fare'] = data['Fare'] + C

def log_transform(col):
    return np.log(col[0])

data['Fare'] = data[['Fare']].apply(log_transform, axis=1)
corr = data.corr()

plt.figure(figsize=(10, 10))
sns.heatmap(corr, cmap='YlGnBu', annot=True)
sns.boxplot(data['Survived'], data['PassengerId'])
data.drop('PassengerId', axis=1, inplace=True)
test.drop('PassengerId', axis=1, inplace=True)
test.head()
labels = pd.DataFrame(data['Survived'], columns=['Survived'])
data.drop('Survived', axis=1, inplace=True)
extra_tree_forest = ExtraTreesClassifier() 
  
extra_tree_forest.fit(data, labels) 
  
feature_importance = extra_tree_forest.feature_importances_ 

plt.figure(figsize=(10, 10))
plt.bar(data.columns, feature_importance) 
plt.xlabel('Feature Labels') 
plt.ylabel('Feature Importances') 
plt.title('Comparison of different Feature Importances') 
plt.show() 
sns.countplot(labels.Survived)
sns.countplot(data.Pclass, hue=labels.Survived)
sns.boxplot(labels.Survived, data.Fare)
sns.countplot(data.Sex, hue=labels.Survived)
sns.countplot(data.SibSp, hue=labels.Survived)
sns.countplot(data.Parch, hue=labels.Survived)
kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

#cross validation.
def cross_validate(model, train, labels):
    models = []
    print(train.shape)
    fold_count = 1

    for train_ind, val_ind in kfold.split(train, labels):

        t_data = train.loc[train_ind]
        t_labels = labels.loc[train_ind]
        v_data = train.loc[val_ind]
        v_labels = labels.loc[val_ind]
    
        model.fit(t_data, t_labels)
        
        preds = model.predict(v_data)
        
        accuracy = accuracy_score(v_labels, preds)
        precision = precision_score(v_labels, preds)
        recall = recall_score(v_labels, preds)
        f1score = f1_score(v_labels, preds)
    
        print(
            ' Fold : ' + str(fold_count) +
            ' Accuracy : ' + str(round(accuracy, 2)) +
            ' Precision : ' + str(round(precision, 2)) +
            ' Recall : ' + str(round(recall, 2)) + 
            ' f1score : ' + str(round(f1score, 2))
        )
        
        models.append(model)
        
        fold_count += 1
        
    return models
    
#parameter tuning.
def tune(model, params, train, labels):
    search = RandomizedSearchCV(model, params, n_iter=50, cv=5, random_state=21)
    best_model = search.fit(train, labels)
    pprint(best_model.best_estimator_.get_params())
    return best_model
model = LogisticRegression()
logistic_models = cross_validate(model, data, labels)
model = RandomForestClassifier()
rf_models = cross_validate(model, data, labels)
model = xgb.XGBClassifier()
xgb_models = cross_validate(model, data, labels)
model = lgb.LGBMClassifier()
lgb_models = cross_validate(model, data, labels)
params = {
    'max_iter' : [80, 90, 100, 110, 120],
    'random_state' : [0, 1, 42],
    'penalty' : ['l1', 'l2']
}

model = LogisticRegression()

best_model = tune(model, params, data, labels)
model = LogisticRegression(**best_model.best_estimator_.get_params())
tuned_logistic_models = cross_validate(model, data, labels)
params = {
    'n_estimators' : [110, 120, 130, 140],
    'max_depth' : [6, 7, 8],
    'max_features' : [6, 7, 8],
    'bootstrap' : [True],
    'min_samples_leaf' : [2, 3]
}

model = RandomForestClassifier()

best_model = tune(model, params, data, labels)
model = RandomForestClassifier(**best_model.best_estimator_.get_params())
tuned_rf_models = cross_validate(model, data, labels)
params = {
    'eta' : [0.05, 0.1],
    'max_depth' : [6, 7],
    'verbosity' : [1],
    'subsample' : [0.75],
    'n_estimators' : [110, 120, 130]
}

model = xgb.XGBClassifier()
best_model = tune(model, params, data, labels)
model = xgb.XGBClassifier(**best_model.best_estimator_.get_params())
tuned_xgb_models = cross_validate(model, data, labels)
feature_imp = pd.DataFrame(sorted(zip(tuned_xgb_models[0].feature_importances_, data.columns)), columns=['Value','Feature'])

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('XGB Features (avg over folds)')
plt.tight_layout()
plt.show()
params = {
    'n_estimators' : [155, 160, 165],
    'learning_rate' : [0.05],
    'max_depth' : [7, 8],
    'num_leaves' : [25],
    'min_data_in_leaf' : [15, 18, 20],
    'bagging_fraction' : [0.5],
    'feature_fraction' : [0.7],
    'lambda_l2' : [0.75],
    'subsample' : [0.5],
}

model = lgb.LGBMClassifier()
best_model = tune(model, params, data, labels)
model = lgb.LGBMClassifier(**best_model.best_estimator_.get_params())
tuned_lgb_models = cross_validate(model, data, labels)
feature_imp = pd.DataFrame(sorted(zip(tuned_lgb_models[1].feature_importances_,data.columns)), columns=['Value','Feature'])

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()
test.head()
final_model = tuned_lgb_models[1]
print(final_model)
preds = final_model.predict(test)
print(preds)
temp = pd.read_csv('/kaggle/input/titanic/test.csv')
submission = pd.DataFrame({'PassengerId': temp.PassengerId, 'Survived': preds})
submission.head()
submission.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
