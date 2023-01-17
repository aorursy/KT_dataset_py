import numpy as np

import pandas as pd 



import os

print(os.listdir("../input"))



%matplotlib inline



import matplotlib.pyplot as plt

import missingno

import seaborn as sbrn



from sklearn.preprocessing import OneHotEncoder, label_binarize



import catboost

from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, datasets, linear_model

from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix

from catboost import CatBoostClassifier, Pool, cv



from IPython.display import HTML

import base64
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

submission = pd.read_csv('../input/gender_submission.csv')
train.head(15)
len(train)
train.describe()
train.Age.plot.hist()
train.Pclass.plot.hist()
train.Fare.plot.hist()
missingno.matrix(train)
train.isnull().sum()
train.dtypes
sbrn.countplot(y='Survived', data=train);

print(train.Survived.value_counts())
#Separating our variables into categorical and continuous variables within dataframes, makes exploration easier



df_cat = pd.DataFrame()

df_con = pd.DataFrame()
#To explore the variables as related to the outcome of survival, we'll add the outcome to the empty dataframes



df_cat['Survived'] = train['Survived']

df_con['Survived'] = train['Survived']



#Adding Pclass to the dataframes since no data is missing for that variable



df_cat['Pclass'] = train['Pclass']

df_con['Pclass'] = train['Pclass']



#Adding Sex to the dataframes since no data is missing for that variable



df_cat['Sex'] = train['Sex']

#This next line of code will recode the variables into 0/1; female = 1, male = 0. Male = 0 was decided since the idea of "women and children first" would probably lead to higher survival for 'female'

df_cat['Sex'] = np.where(df_cat['Sex'] == 'male', 1, 0)

df_con['Sex'] = train['Sex']
train.groupby(['Sex', 'Survived']).count()
train.SibSp.value_counts()
train.groupby(['SibSp', 'Survived']).count()
df_cat['SibSp'] = train['SibSp']

# We will rewrite them as categories where 0=0, 1=1, 2-8=2

df_cat['SibSp'] = pd.cut(train['SibSp'], [0,1,2,8], labels=['0','1','2+'], right=False)



df_con['SibSp'] = train['SibSp']
train.Parch.value_counts()
train.groupby(['Parch', 'Survived']).count()
train.groupby(['Parch', 'SibSp']).count()
df_cat['Parch'] = train['Parch']

# We will rewrite them as categories where 0=0, 1=1, 2=2, 3=3-6

df_cat['Parch'] = pd.cut(train['Parch'], [0,1,2,3,6], labels=['0','1','2','3+'], right=False)



df_con['Parch'] = train['Parch']
train.Ticket.value_counts()
from scipy.stats import boxcox



df_cat['Fare'] = train['Fare'] + 0.1

df_cat['Fare'] = boxcox(df_cat.Fare)[0]

df_cat['Fare'] = pd.cut(df_cat['Fare'], bins=5)



df_con['Fare'] = train['Fare']
train['Cabin'].head(15)
df_cat['Cabin'] = train['Cabin']

df_cat['Cabin'] = df_cat['Cabin'].replace(np.nan, '0')



df_cat['Cabin'] = df_cat['Cabin'].str.replace('.*A+.*', '1')

df_cat['Cabin'] = df_cat['Cabin'].str.replace('.*B+.*', '1')

df_cat['Cabin'] = df_cat['Cabin'].str.replace('.*C+.*', '1')

df_cat['Cabin'] = df_cat['Cabin'].str.replace('.*D+.*', '1')

df_cat['Cabin'] = df_cat['Cabin'].str.replace('.*E+.*', '1')

df_cat['Cabin'] = df_cat['Cabin'].str.replace('.*F+.*', '1')

df_cat['Cabin'] = df_cat['Cabin'].str.replace('.*G+.*', '1')

train[train.Embarked.isna()]
train.groupby(['Embarked',train['Cabin'].str.contains("B", na=False)]).count()
df_cat['Embarked'] = train['Embarked']

df_cat.set_value(61, 'Embarked', 'S')

df_cat.set_value(829, 'Embarked', 'S')



df_con['Embarked'] = df_cat['Embarked']
df_cat.head()
# One-Hot Encoding employed

onehot_cols = df_cat.columns.tolist()

onehot_cols.remove('Survived')

df_cat_enc = pd.get_dummies(df_cat, columns=onehot_cols)



df_cat_enc.head(8)
df_con.head(15)
df_embarked_onehot = pd.get_dummies(df_con['Embarked'], 

                                     prefix='embarked')



df_sex_onehot = pd.get_dummies(df_con['Sex'], 

                                prefix='sex')



df_plcass_onehot = pd.get_dummies(df_con['Pclass'], 

                                   prefix='pclass')
# Now lets remove the original variables

df_con_enc = pd.concat([df_con, 

                        df_embarked_onehot, 

                        df_sex_onehot, 

                        df_plcass_onehot], axis=1)



# Drop the original categorical columns (because now they've been one hot encoded)

df_con_enc = df_con_enc.drop(['Pclass', 'Sex', 'Embarked'], axis=1)
df_con_enc.head(9)
chosen_df = df_con_enc
X_train = chosen_df.drop('Survived', axis=1)

y_train = chosen_df.Survived
def fit_ml_algo(algo, X_train, y_train, cv):

    

    # One Pass

    model = algo.fit(X_train, y_train)

    acc = round(model.score(X_train, y_train) * 100, 2)

    

    # Cross Validation 

    train_pred = model_selection.cross_val_predict(algo, 

                                                  X_train, 

                                                  y_train, 

                                                  cv=cv, 

                                                  n_jobs = -1)

    

    

    # Cross-validation accuracy metric

    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)

    confus_matr = metrics.confusion_matrix(y_train, train_pred)

    precision = precision_score(y_train, train_pred)

    recall = recall_score(y_train, train_pred)

    F1 = f1_score(y_train, train_pred)

    auc = roc_auc_score(y_train, train_pred)

    

    return train_pred, acc, acc_cv, confus_matr, precision, recall, F1, auc
train_pred_log, acc_log, acc_cv_log, cfmt_log, prec_log, rec_log, F1_log, auc_log = fit_ml_algo(LogisticRegression(), 

                                                                                               X_train, 

                                                                                               y_train, 

                                                                                                10)

print("Accuracy: %s" % acc_log)

print("Accuracy CV 10-Fold: %s" % acc_cv_log)

print("Confusion Matrix: %s" % cfmt_log)

print("Precision: ", float("{0:.3f}".format(round(prec_log, 3))))

print("Recall: ", float("{0:.3f}".format(round(rec_log, 3))))

print("F1 Score: ", float("{0:.3f}".format(round(F1_log, 3))))

print("AUC: ", float("{0:.3f}".format(round(auc_log, 3))))
train_pred_knn, acc_knn, acc_cv_knn, cfmt_knn, prec_knn, rec_knn, F1_knn, auc_knn = fit_ml_algo(KNeighborsClassifier(), 

                                                                                                X_train, 

                                                                                                y_train, 

                                                                                                10)

print("Accuracy: %s" % acc_knn)

print("Accuracy CV 10-Fold: %s" % acc_cv_knn)

print("Confusion Matrix: %s" % cfmt_knn)

print("Precision: ", float("{0:.3f}".format(round(prec_knn, 3))))

print("Recall: ", float("{0:.3f}".format(round(rec_knn, 3))))

print("F1 Score: ", float("{0:.3f}".format(round(F1_knn, 3))))

print("AUC: ", float("{0:.3f}".format(round(auc_knn, 3))))
train_pred_svc, acc_linear_svc, acc_cv_linear_svc, cfmt_svc, prec_svc, rec_svc, F1_svc, auc_svc = fit_ml_algo(LinearSVC(),

                                                                                                                X_train, 

                                                                                                                y_train, 

                                                                                                                10)

print("Accuracy: %s" % acc_linear_svc)

print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)

print("Confusion Matrix: %s" % cfmt_svc)

print("Precision: ", float("{0:.3f}".format(round(prec_svc, 3))))

print("Recall: ", float("{0:.3f}".format(round(rec_svc, 3))))

print("F1 Score: ", float("{0:.3f}".format(round(F1_svc, 3))))

print("AUC: ", float("{0:.3f}".format(round(auc_svc, 3))))
train_pred_sgd, acc_sgd, acc_cv_sgd, cfmt_sgd, prec_sgd, rec_sgd, F1_sgd, auc_sgd = fit_ml_algo(SGDClassifier(), 

                                                                                                  X_train, 

                                                                                                  y_train,

                                                                                                  10)

print("Accuracy: %s" % acc_sgd)

print("Accuracy CV 10-Fold: %s" % acc_cv_sgd)

print("Confusion Matrix: %s" % cfmt_sgd)

print("Precision: ", float("{0:.3f}".format(round(prec_sgd, 3))))

print("Recall: ", float("{0:.3f}".format(round(rec_sgd, 3))))

print("F1 Score: ", float("{0:.3f}".format(round(F1_sgd, 3))))

print("AUC: ", float("{0:.3f}".format(round(auc_sgd, 3))))
train_pred_gaussian, acc_gaussian, acc_cv_gaussian, cfmt_gaussian, prec_gaussian, rec_gaussian, F1_gaussian, auc_gaussian = fit_ml_algo(GaussianNB(), 

                                                                                                              X_train, 

                                                                                                              y_train, 

                                                                                                               10)

print("Accuracy: %s" % acc_gaussian)

print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)

print("Confusion Matrix: %s" % cfmt_gaussian)

print("Precision: ", float("{0:.3f}".format(round(prec_gaussian, 3))))

print("Recall: ", float("{0:.3f}".format(round(rec_gaussian, 3))))

print("F1 Score: ", float("{0:.3f}".format(round(F1_gaussian, 3))))

print("AUC: ", float("{0:.3f}".format(round(auc_gaussian, 3))))
train_pred_dt, acc_dt, acc_cv_dt, cfmt_dt, prec_dt, rec_dt, F1_dt, auc_dt = fit_ml_algo(DecisionTreeClassifier(), 

                                                                                        X_train, 

                                                                                        y_train,

                                                                                        10)

print("Accuracy: %s" % acc_dt)

print("Accuracy CV 10-Fold: %s" % acc_cv_dt)

print("Confusion Matrix: %s" % cfmt_dt)

print("Precision: ", float("{0:.3f}".format(round(prec_dt, 3))))

print("Recall: ", float("{0:.3f}".format(round(rec_dt, 3))))

print("F1 Score: ", float("{0:.3f}".format(round(F1_dt, 3))))

print("AUC: ", float("{0:.3f}".format(round(auc_dt, 3))))
train_pred_gbt, acc_gbt, acc_cv_gbt, cfmt_gbt, prec_gbt, rec_gbt, F1_gbt, auc_gbt = fit_ml_algo(GradientBoostingClassifier(), 

                                                                                               X_train, 

                                                                                               y_train,

                                                                                               10)

print("Accuracy: %s" % acc_gbt)

print("Accuracy CV 10-Fold: %s" % acc_cv_gbt)

print("Confusion Matrix: %s" % cfmt_gbt)

print("Precision: ", float("{0:.3f}".format(round(prec_gbt, 3))))

print("Recall: ", float("{0:.3f}".format(round(rec_gbt, 3))))

print("F1 Score: ", float("{0:.3f}".format(round(F1_gbt, 3))))

print("AUC: ", float("{0:.3f}".format(round(auc_gbt, 3))))
## Remove non-categorical variables

cat_features = np.where(X_train.dtypes != np.float)[0]
train_pool = Pool(X_train, 

                  y_train,

                  cat_features)
# CatBoost model 

catboost_model = CatBoostClassifier(iterations=1000,

                                    custom_loss=['Accuracy'],

                                    loss_function='Logloss')



# Fit CatBoost model

catboost_model.fit(train_pool,

                   plot=False)



# CatBoost accuracy

acc_catboost = round(catboost_model.score(X_train, y_train) * 100, 2)
# Set params for cross-validation

cv_params = catboost_model.get_params()



# Run the cross-validation for 10-folds

cv_data = cv(train_pool,

             cv_params,

             fold_count=10,

             plot=False)





# CatBoost CV results save into a dataframe (cv_data), let's withdraw the maximum accuracy score

acc_cv_catboost = round(np.max(cv_data['test-Accuracy-mean']) * 100, 2)
# Print out the CatBoost model metrics

print("---CatBoost Metrics---")

print("Accuracy: {}".format(acc_catboost))

print("Accuracy cross-validation 10-Fold: {}".format(acc_cv_catboost))
models = pd.DataFrame({

    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree', 'Gradient Boosting Trees',

              'CatBoost'],

    'Score': [

        acc_knn, 

        acc_log,  

        acc_gaussian, 

        acc_sgd, 

        acc_linear_svc, 

        acc_dt,

        acc_gbt,

        acc_catboost

    ]})

print("---Reuglar Accuracy Scores---")

models.sort_values(by='Score', ascending=False)
cv_models = pd.DataFrame({

    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree', 'Gradient Boosting Trees',

              'CatBoost'],

    'Score': [

        acc_cv_knn, 

        acc_cv_log,      

        acc_cv_gaussian, 

        acc_cv_sgd, 

        acc_cv_linear_svc, 

        acc_cv_dt,

        acc_cv_gbt,

        acc_cv_catboost

    ]})

print('---Cross-validation Accuracy Scores---')

cv_models.sort_values(by='Score', ascending=False)
model_gbt = GradientBoostingClassifier()

model_gbt.fit(X_train, y_train)

# plot feature importance

print(model_gbt.feature_importances_)

plt.bar(range(len(model_gbt.feature_importances_)), model_gbt.feature_importances_)

plt.show()
def feature_importance(model, data):

    """

    Function to show which features are most important in the model.

    ::param_model:: Which model to use?

    ::param_data:: What data to use?

    """

    fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': data.columns})

    fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]

    _ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))

    return fea_imp
feature_importance(catboost_model, X_train)


metrics = ['Precision', 'Recall', 'F1', 'AUC']



eval_metrics = catboost_model.eval_metrics(train_pool,

                                           metrics=metrics,

                                           plot=False)



for metric in metrics:

    print(str(metric)+": {}".format(np.mean(eval_metrics[metric])))
test_embarked_onehot = pd.get_dummies(test['Embarked'], 

                                     prefix='embarked')



test_sex_onehot = pd.get_dummies(test['Sex'], 

                                prefix='sex')



test_plcass_onehot = pd.get_dummies(test['Pclass'], 

                                   prefix='pclass')
# Now lets remove the original variables

test = pd.concat([test, 

                        test_embarked_onehot, 

                        test_sex_onehot, 

                        test_plcass_onehot], axis=1)
test.head(3)
want_test_colum = X_train.columns

want_test_colum
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
predict_test_cb = catboost_model.predict(test[want_test_colum])
submission_catboost = pd.DataFrame()

submission_catboost['PassengerId'] = test['PassengerId']

submission_catboost['Survived'] = predict_test_cb

submission_catboost['Survived'] = submission_catboost['Survived'].astype(int)

submission_catboost.head()
create_download_link(submission_catboost)
test[test.Fare.isna()]
df_con.groupby(['Pclass', 'Fare']).count()
test.set_value(152, 'Fare', '55.7875')

test.iloc[152]
test[want_test_colum].isna().sum()
predict_test_gbt = model_gbt.predict(test[want_test_colum])
submission_gbt = pd.DataFrame()

submission_gbt['PassengerId'] = test['PassengerId']

submission_gbt['Survived'] = predict_test_gbt

submission_gbt['Survived'] = submission_gbt['Survived'].astype(int)

submission_gbt.head()
create_download_link(submission_gbt)