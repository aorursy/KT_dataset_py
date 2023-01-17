import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from IPython.display import display

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, recall_score

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier

import warnings



warnings.filterwarnings('ignore')
data = pd.read_csv('../input/diabetes.csv')



display(data.head())

display(data.describe())

sns.countplot(data['Outcome'],label="Count")

display(data.info())
#Feature Preprocessing

for col in ['BloodPressure', 'Glucose','SkinThickness','Insulin','BMI','Age']:

    for target in data.Outcome.unique():

        mask = (data[col] != 0) & (data['Outcome'] == target)

        data[col][(data[col] == 0) & (data['Outcome'] == target)] = data[col][(data[col] == 0) & (data['Outcome'] == target)].replace(0,data[mask][col].mean())

        

#Extract data

X = data.iloc[:,0:8]

y = data.iloc[:,-1]



#X2 = X.copy()

#X2['Glucose<125'] = X2['Glucose']<125

#X2['Glucose<125'] = X2['Glucose<125'].astype(int)



#X_train,X_test,y_train,y_test = train_test_split(X2, y, random_state=0,stratify=y)



X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0,stratify=y)



scaler = StandardScaler()

scaler.fit(X_train,y_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
corr_mat=sns.heatmap(data.corr(method='spearman'),annot=True,cbar=True,

            cmap='viridis', vmax=1,vmin=-1,

            xticklabels=X.columns,yticklabels=X.columns)

corr_mat.set_xticklabels(corr_mat.get_xticklabels(),rotation=90)
sns.pairplot(data, hue = "Outcome", diag_kind='kde')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

param_grid = {'alpha': np.arange(1,40)}

mlp = MLPClassifier(solver='lbfgs', activation='relu', random_state=9, learning_rate='adaptive',

                        hidden_layer_sizes=[10,10,10])

grid_mlp = GridSearchCV(mlp, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)

grid_mlp.fit(X_train_scaled, y_train)

print("Best cross-validation accuracy: {:.3f}".format(grid_mlp.best_score_))

print("Test set score: {:.3f}".format(grid_mlp.score(X_test_scaled,y_test)))

print("Best parameters: {}".format(grid_mlp.best_params_))
mlp = MLPClassifier(solver='lbfgs', activation='relu', random_state=9, learning_rate='adaptive',

                        hidden_layer_sizes=[10,10,10], alpha=grid_mlp.best_params_['alpha'])

mlp.fit(X_train_scaled,y_train)

conf_mat_mlp = confusion_matrix(y_test, mlp.predict(X_test_scaled))

sns.heatmap(conf_mat_mlp, annot=True, cbar=False, cmap="viridis_r",

            yticklabels=['Healthy', 'Diabetes'], xticklabels=['Healthy', 'Diabetes'])

print("Neural Net Recall Score: {:.2f}".format(recall_score(y_test, mlp.predict(X_test_scaled))))
param_grid = {'n_estimators': [100,200,300,400,500]}

forest = RandomForestClassifier(random_state=79)

grid_forest = GridSearchCV(forest, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)

grid_forest.fit(X_train_scaled, y_train)

print("Best cross-validation accuracy: {:.3f}".format(grid_forest.best_score_))

print("Test set score: {:.3f}".format(grid_forest.score(X_test_scaled,y_test)))

print("Best parameters: {}".format(grid_forest.best_params_))

#1
forest = RandomForestClassifier(n_estimators=grid_forest.best_params_['n_estimators'])

forest.fit(X_train_scaled,y_train)

conf_mat_forest = confusion_matrix(y_test, forest.predict(X_test_scaled))

sns.heatmap(conf_mat_forest, annot=True, cbar=False, cmap="viridis_r",

            yticklabels=['Healthy', 'Diabetes'], xticklabels=['Healthy', 'Diabetes'])

param_grid = {'n_estimators': [100,200,300,400,500], 'max_depth': [2,3,4,5]}

gbrt = GradientBoostingClassifier(random_state=0)

grid_gbrt = GridSearchCV(gbrt, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)

grid_gbrt.fit(X_train_scaled, y_train)

print("Best cross-validation accuracy: {:.3f}".format(grid_gbrt.best_score_))

print("Test set score: {:.3f}".format(grid_gbrt.score(X_test_scaled,y_test)))

print("Best parameters: {}".format(grid_gbrt.best_params_))
gbrt = GradientBoostingClassifier(random_state=0, max_depth=grid_gbrt.best_params_['max_depth'],

                                 n_estimators=grid_gbrt.best_params_['n_estimators'])

gbrt.fit(X_train_scaled,y_train)

conf_mat_gbrt = confusion_matrix(y_test, gbrt.predict(X_test_scaled))

sns.heatmap(conf_mat_gbrt, annot=True, cbar=False, cmap="viridis_r",

            yticklabels=['Healthy', 'Diabetes'], xticklabels=['Healthy', 'Diabetes'])

print("GBRT Recall Score: {:.2f}".format(recall_score(y_test, gbrt.predict(X_test_scaled))))
#gbrt

fpr,tpr,thresholds = roc_curve(y_test, gbrt.decision_function(X_test_scaled))

plt.plot(fpr,tpr, label='GBRT')

print("AUC GBRT: {:.3f}".format(roc_auc_score(y_test, gbrt.decision_function(X_test_scaled))))



#Forest

fpr,tpr,thresholds = roc_curve(y_test, forest.predict_proba(X_test_scaled)[:,1])

plt.plot(fpr,tpr, label='Forest')

print("AUC Forest: {:.3f}".format(roc_auc_score(y_test, forest.predict_proba(X_test_scaled)[:,1])))





#nn

fpr,tpr,thresholds = roc_curve(y_test, mlp.predict_proba(X_test_scaled)[:,1])

plt.plot(fpr,tpr, label='NN')

print("AUC NN: {:.3f}".format(roc_auc_score(y_test, mlp.predict_proba(X_test_scaled)[:,1])))

plt.legend(loc='best')

plt.xlim(0,1)

plt.ylim(0,1)
summary = {'Metrics': ["Accuracy", "Recall", "AUC"],

           'Neural Network': [mlp.score(X_test_scaled,y_test), recall_score(y_test, mlp.predict(X_test_scaled)), roc_auc_score(y_test, mlp.predict_proba(X_test_scaled)[:,1])], 

           'Random Forest': [forest.score(X_test_scaled,y_test), recall_score(y_test, forest.predict(X_test_scaled)), roc_auc_score(y_test, forest.predict_proba(X_test_scaled)[:,1])],

           'Gradient Boosted Classifier': [gbrt.score(X_test_scaled,y_test), recall_score(y_test, gbrt.predict(X_test_scaled)), roc_auc_score(y_test, gbrt.predict_proba(X_test_scaled)[:,1])]}

summary_df = pd.DataFrame(data=summary)

display(summary_df)