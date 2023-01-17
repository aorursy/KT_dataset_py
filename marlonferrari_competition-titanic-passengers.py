!pip install xgboost
import pandas as pd
import xgboost as xgb
dataset = pd.read_csv('../input/titanic/train.csv')

null_columns=dataset.columns[dataset.isnull().any()]
dataset[null_columns].isnull().sum()
#drop unimportant features
df_dataset = dataset.drop('Cabin', axis=1).drop('Name', axis=1).drop('Ticket', axis=1)
df_dataset['Age'].fillna((df_dataset['Age'].mean()), inplace=True)
X_dataset = df_dataset.drop('Survived', axis=1).drop('PassengerId', axis=1)
y_dataset = df_dataset[['Survived']]
#gets test columns based on train featured columns
columns = X_dataset.columns
validation=pd.read_csv('../input/titanic/test.csv')
df_validation = validation[columns]
df_validation.head()
df_validation['Age'].fillna((df_dataset['Age'].mean()), inplace=True)
X_dataset =  pd.get_dummies(X_dataset)

X_validation= df_validation
X_validation =  pd.get_dummies(X_validation)
from sklearn import preprocessing
#min-Max Scaler
min_max_scaler = preprocessing.MinMaxScaler()

#transform main set
X_dataset_values = X_dataset.values 
X_dataset_values_scaled = min_max_scaler.fit_transform(X_dataset_values)
X_dataset_values_scaled = pd.DataFrame(X_dataset_values_scaled)

#transform validation set
X_validation_values = X_validation.values
X_validation_values_scaled = min_max_scaler.transform(X_validation_values)
X_validation_values_scaled = pd.DataFrame(X_validation_values_scaled)
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

#X_train, X_test, y_train, y_test = train_test_split(X_dataset_values_scaled, y_dataset, test_size=0.1)

# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
xgb_model = xgb.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)
folds = 3
param_comb = 5

X = X_dataset_values_scaled
Y = y_dataset

skf = StratifiedKFold(n_splits=folds, shuffle = True)

random_search = RandomizedSearchCV(xgb_model, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3)
random_search.fit(X, Y)
print('\n Best estimator:')
print(random_search.best_estimator_)
xgb_model_final = random_search.best_estimator_
#xgb_model.fit(X_train, y_train)
#TEST_SPLIT_CASE
#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

#y_hat = xgb_model.predict(X_test)
#print("Roc AUC: ", roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1],
#              average='macro'))
#print(confusion_matrix(y_test,y_hat))
#print(classification_report(y_test,y_hat))
y_hat_sub = xgb_model_final.predict(X_validation_values_scaled)
y_hat_sub = pd.DataFrame(y_hat_sub)
y_hat_sub.shape
validation['Survived'] = y_hat_sub
validation[['PassengerId','Survived']].to_csv('submission_final.csv',index=False)