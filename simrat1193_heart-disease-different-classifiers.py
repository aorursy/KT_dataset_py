import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve, auc, confusion_matrix, recall_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import *

from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/heart.csv')

df.shape
df.head()
df['target'].value_counts()
def histogram(col_name, title, xlabel):

    fig, ax = plt.subplots()

    df[col_name].hist(color='#A9C5D3', edgecolor='black', grid=False)

    ax.set_title(title, fontsize=12)

    ax.set_xlabel(xlabel, fontsize=12)

    ax.set_ylabel('Frequency', fontsize=12)
histogram('age', 'Patient Age Histogram', 'Age')
def cut_quantile(quant_col_name, col_to_cut):

    quantile_list = [0, 0.25, 0.5, 0.75, 1.0]

    quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']

    df[quant_col_name] = pd.qcut(df[col_to_cut], q=quantile_list, labels=quantile_labels)
cut_quantile('age_quantile_label', 'age')

df.head(5)
sns.countplot('age_quantile_label', data=df);

sns.catplot(x="age_quantile_label", col="target", data=df, kind="count", height=4, aspect=.9);
histogram('trestbps', 'Patient trestbps', 'trestbps')
cut_quantile('trestbps_quantile_label', 'trestbps')

df.head(5)
sns.countplot('trestbps_quantile_label', data=df);

sns.catplot(x="trestbps_quantile_label", col="target", data=df, kind="count", height=4, aspect=.9);
histogram('chol', 'Patient Cholestrol', 'Cholestrol')
cut_quantile('chol_quantile_label', 'chol')

sns.countplot('chol_quantile_label', data=df);

sns.catplot(x="chol_quantile_label", col="target", data=df, kind="count", height=4, aspect=.9);
refined_df = df

target = refined_df['target']



refined_df = refined_df.drop(['age','chol', 'trestbps', 'target'], axis=1)

refined_df = pd.get_dummies(refined_df)

refined_df.head(5)
X_train, X_test, y_train, y_test = train_test_split(refined_df, target, test_size=0.20, random_state=42)
def plot_roc_(false_positive_rate,true_positive_rate,roc_auc):

    plt.figure(figsize=(5,5))

    plt.title('Receiver Operating Characteristic')

    plt.plot(false_positive_rate, true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],linestyle='--')

    plt.axis('tight')

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()
models_data = dict()

def analyse_model(model, name):

    y_preds = model.predict(X_test)

    y_proba = model.predict_proba(X_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba[:,1])

    roc_auc = round(auc(false_positive_rate, true_positive_rate), 3)

    accuracy = round(accuracy_score(y_test, y_preds), 3)

    f1 = round(f1_score(y_test, y_preds), 3)

    recall = round(recall_score(y_test, y_preds), 3)

    plot_roc_(false_positive_rate, true_positive_rate, roc_auc)

    

    print("Accuracy {}".format(accuracy))

    print("F1 Score {}".format(f1))

    print("AUC Score {}".format(roc_auc))

    print("Recall {}".format(recall))

    models_data[name] = [roc_auc, accuracy, f1, recall] 
logs_regr_model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000).fit(X_train, y_train)

analyse_model(logs_regr_model, "Logistic Regression")
parameters = {'C': [0.01, 0.05, 0.1, 0.5, 1]}

logs_grid_model = GridSearchCV(logs_regr_model, parameters, scoring='roc_auc')

logs_grid_model.fit(X_train, y_train)

print(logs_grid_model.best_params_)

analyse_model(logs_grid_model, "Logistic Regression GS")
dec_tree_model = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)

analyse_model(dec_tree_model, "Decision Tree")
parameters = {'max_depth': [2, 3, 4, 5, 6, 7], 'criterion': ('gini', 'entropy'), 

              'min_samples_split': [2, 3, 4, 5], 'min_samples_leaf': [1, 2, 3, 4]}

dec_grid_model = GridSearchCV(dec_tree_model, parameters, scoring='roc_auc')

dec_grid_model.fit(X_train, y_train)



print(dec_grid_model.best_params_)

analyse_model(dec_grid_model, "Decision Tree GS")
rf_model = RandomForestClassifier(random_state=0).fit(X_train, y_train)

analyse_model(rf_model, "Random Forest")
parameters = {'n_estimators': [10, 50, 100], 'max_depth': [2, 3, 4, 5, 6, 7]}

rf_grid_model = GridSearchCV(rf_model, parameters, scoring='roc_auc').fit(X_train, y_train)

print(rf_grid_model.best_params_)

analyse_model(rf_grid_model, "Random Forest GS")
gbm_model = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)

analyse_model(gbm_model, "Gradient Boost")
parameters = {'n_estimators': [10, 100], 'max_depth': [2, 3, 4, 5, 6, 7]}

gbm_grid_model = GridSearchCV(gbm_model, parameters, scoring='roc_auc').fit(X_train, y_train)

print(gbm_grid_model.best_params_)

analyse_model(gbm_grid_model, "Gradient Boost GS")
xgb_model = xgb.XGBClassifier(random_state=0).fit(X_train, y_train)

analyse_model(xgb_model, "XGBoost")
parameters = {'n_estimators': [10, 100], 'max_depth': [2, 3, 4, 5, 6, 7]}

xgb_grid_model = GridSearchCV(xgb_model, parameters, scoring='roc_auc').fit(X_train, y_train)

print(xgb_grid_model.best_params_)

print(xgb_grid_model.best_score_)

analyse_model(xgb_grid_model, "XGBoost GS")
score_df = pd.DataFrame.from_dict(models_data, orient='index', columns=['AUC Score', 'Accuracy', 'F1 Score', 'Recall'])

score_df.sort_values(by=['AUC Score'], ascending=False)