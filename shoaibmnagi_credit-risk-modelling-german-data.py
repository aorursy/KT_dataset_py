import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("/kaggle/input/german-credit-data-with-risk/german_credit_data.csv", index_col=0)

df.head()
df.isna().sum()
for col in ['Saving accounts', 'Checking account']:

    df[col].fillna('none', inplace=True)

j = {0: 'unskilled and non-res', 1: 'unskilled and res', 2: 'skilled', 3: 'highly skilled'}

df['Job'] = df['Job'].map(j)



df.head()
g = sns.FacetGrid(df, col="Sex",  row="Job")

g = g.map(sns.countplot, "Risk")
sns.countplot(x="Housing", hue="Risk", data=df)

plt.show()

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,8))

sns.countplot(x="Saving accounts", hue="Risk", data=df, ax=ax1)

sns.countplot(x="Checking account", hue="Risk", data=df, ax=ax2)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)





fig.show()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize=(15,25))



sns.boxplot(x="Purpose", y="Credit amount", data=df, hue="Risk", palette="coolwarm", ax=ax1)

sns.boxplot(x="Saving accounts", y="Credit amount", data=df, hue="Risk", palette="coolwarm", ax=ax2)

sns.boxplot(x="Checking account", y="Credit amount", data=df, hue="Risk", palette="coolwarm", ax=ax3)

sns.boxplot(x="Job", y="Credit amount", data=df, hue="Risk", palette="coolwarm", ax=ax4)



ax1.set_title("Credit Amount Distribution by Purpose", fontsize=16)

ax2.set_title("Credit Amount Distribution by Saving Account Status", fontsize=16)

ax3.set_title("Credit Amount Distribution by Checking Account Status", fontsize=16)

ax4.set_title("Credit Amount Distribution by Job Type", fontsize=16)



plt.subplots_adjust(hspace = 0.6, top = 0.6)

plt.show()

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,14))



sns.pointplot(x="Age", y="Credit amount", data=df, hue="Risk", palette="coolwarm", ax=ax1)

sns.pointplot(x="Duration", y="Credit amount", data=df, hue="Risk", palette="coolwarm", ax=ax2)



ax1.set_title("Credit Amount Distribution by Age", fontsize=16)

ax2.set_title("Credit Amount Distribution by Loan Duration", fontsize=16)



plt.subplots_adjust(wspace = 0.4, hspace = 0.4,top = 0.9)

plt.show()
# getting dummies for all the categorical variables

dummies_columns = ['Job', 'Purpose', 'Sex', 'Housing', 'Saving accounts', 'Checking account']

for col in dummies_columns:

    df = df.merge(pd.get_dummies(df[col], drop_first=True, prefix=str(col)), left_index=True, right_index=True) 



# encoding risk as binary

r = {"good":0, "bad": 1}

df['Risk'] = df['Risk'].map(r)



# drop redundant variables

columns_to_drop = ['Job', 'Purpose','Sex','Housing','Saving accounts','Checking account']

df.drop(columns_to_drop, axis=1, inplace=True)



df.head()



    

df['Log_CA'] = np.log(df['Credit amount'])
plt.figure(figsize=(14,12))

sns.heatmap(df.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True,  linecolor='white', cmap='coolwarm')

plt.show()
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.naive_bayes import GaussianNB



X = df.drop(['Risk', 'Credit amount'], axis=1).values

y = df['Risk'].values



# train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)



log = LogisticRegression()

log.fit(X_train, y_train)

y_pred_log = log.predict(X_test)

print(accuracy_score(y_pred_log, y_test))

print(confusion_matrix(y_test, y_pred_log))

print(classification_report(y_test, y_pred_log))



"""knn = KNeighborsClassifier()

grid = GridSearchCV(knn, param_grid={'n_neighbors':range(1,20)}, scoring='recall')

grid.fit(X_train, y_train)



grid.best_params_



for i in range(0, len(grid.cv_results_['mean_test_score'])):

    print('N_Neighbors {}: {} '.format(i+1, grid.cv_results_['mean_test_score'][i]*100))"""

    

# recall peaks at k = 1



knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print(accuracy_score(y_pred_knn, y_test))

print(confusion_matrix(y_test, y_pred_knn))

print(classification_report(y_test, y_pred_knn))



    
"""param_grid_svc = {"gamma": [0.1,0.5,1,5,10,50,100],

                  "C": [0.1,0.5,1,5,10,50,100]}



svc = SVC(kernel='linear')



gs_svc = GridSearchCV(svc, param_grid = param_grid_svc, cv=5, scoring='recall', verbose=4)

gs_svc.fit(X_train, y_train)



gs_svc.best_params_ # gamma = , C = """



svc = SVC(kernel='linear', gamma=10, C=0.8)

svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)

print(accuracy_score(y_pred_svc, y_test))

print(confusion_matrix(y_test, y_pred_svc))

print(classification_report(y_test, y_pred_svc))

    

"""param_grid_rf = {"max_depth": range(3,10),

                  "max_features": [3,5,7,9,11,13,15,17,20],

                  "min_samples_leaf": [5,10,15,20,25,30],

                  "n_estimators": [3,5,10,25,50,150]}



rf = RandomForestClassifier()

gs_rf = GridSearchCV(rf, param_grid=param_grid_rf, cv=5, scoring="recall", verbose=4)

gs_rf.fit(X_train, y_train)



gs_rf.best_params_

"""

# {'max_depth': 9, 'max_features': 15, 'min_samples_leaf': 5, 'n_estimators': 25}





rf = RandomForestClassifier(max_depth=9, max_features=15, min_samples_leaf=5, n_estimators=25)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print(accuracy_score(y_pred_rf, y_test))

print(confusion_matrix(y_test, y_pred_rf))

print(classification_report(y_test, y_pred_rf))

nb = GaussianNB()

nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)

print(accuracy_score(y_pred_nb, y_test))

print(confusion_matrix(y_test, y_pred_nb))

print(classification_report(y_test, y_pred_nb))

    

"""param_grid_xgb = {"max_depth": range(3,10),

                  "subsample": [0.5,0.6,0.7,0.8,0.9,1],

                  "eta": [0.01,0.03,0.05,0.07,0.09,0.14,0.19],

                  "colsample_bytree": [0.5,0.6,0.7,0.8,0.9,1],

                  "n_estimators": [3,5,10,25,50,150]}



xgb = XGBClassifier()

gs_xgb = GridSearchCV(xgb, param_grid=param_grid_xgb, cv=5, scoring="recall", verbose=4)

gs_xgb.fit(X_train, y_train)



gs_xgb.best_params_ """



"""{'colsample_bytree': 1,

 'eta': 0.19,

 'max_depth': 8,

 'n_estimators': 150,

 'subsample': 0.8}"""



xgb = XGBClassifier(eta=0.19, max_depth=8, n_estimators=150, subsample=0.8, colsample_bytree=1)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print(accuracy_score(y_pred_xgb, y_test))

print(confusion_matrix(y_test, y_pred_xgb))

print(classification_report(y_test, y_pred_xgb))

    

x_training, x_valid, y_training, y_valid = train_test_split(X_train, y_train,

                                                            test_size=0.5,

                                                            random_state=42)

#specify models

model1 = LogisticRegression()

model2 = SVC(kernel='linear', gamma=10, C=0.8)

model3 = GaussianNB()

model4 = XGBClassifier(eta=0.19, max_depth=8, n_estimators=150, subsample=0.8, colsample_bytree=1)

#fit models

model1.fit(x_training, y_training)

model2.fit(x_training, y_training)

model3.fit(x_training, y_training)

model4.fit(x_training, y_training)

#make pred on validation

preds1 = model1.predict(x_valid)

preds2 = model2.predict(x_valid)

preds3 = model3.predict(x_valid)

preds4 = model4.predict(x_valid)

#make pred on test

testpreds1 = model1.predict(X_test)

testpreds2 = model2.predict(X_test)

testpreds3 = model3.predict(X_test)

testpreds4 = model4.predict(X_test)

#form new dataset from valid and test

stackedpredictions = np.column_stack((preds1, preds2, preds3, preds4))

stackedtestpredictions = np.column_stack((testpreds1, testpreds2,

                                              testpreds3, testpreds4))

#make meta model

metamodel = LogisticRegression()

metamodel.fit(stackedpredictions, y_valid)

final_predictions = metamodel.predict(stackedtestpredictions)

    

print(accuracy_score(final_predictions, y_test))

print(confusion_matrix(y_test, final_predictions))

print(classification_report(y_test, final_predictions))

    

from sklearn.metrics import roc_curve, roc_auc_score



results_table = pd.DataFrame(columns = ['models', 'fpr','tpr','auc'])



predictions = {'LR': y_pred_log, 'SVC': y_pred_svc, 'NB': y_pred_nb, 'XGB': y_pred_xgb, 'Stacked': final_predictions}



for key in predictions:

    fpr, tpr, _ = roc_curve(y_test, predictions[key])

    auc = roc_auc_score(y_test, predictions[key])

    

    results_table = results_table.append({'models': key,

                                         'fpr' : fpr,

                                         'tpr' : tpr,

                                         'auc' : auc}, ignore_index=True)

    

results_table.set_index('models', inplace=True)



print(results_table)



fig = plt.figure(figsize = (8,6))



for i in results_table.index:

    plt.plot(results_table.loc[i]['fpr'], 

             results_table.loc[i]['tpr'], 

             label = "{}, AUC={:.3f}".format(i, results_table.loc[i]['auc']))

    

plt.plot([0,1], [0,1], color = 'black', linestyle = '--')



plt.xticks(np.arange(0.0, 1.1, step=0.1))

plt.xlabel("False Positive Rate", fontsize=15)



plt.yticks(np.arange(0.0, 1.1, step=0.1))

plt.ylabel("True Positive Rate", fontsize=15)



plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)

plt.legend(prop = {'size':13}, loc = 'lower right')



plt.show()