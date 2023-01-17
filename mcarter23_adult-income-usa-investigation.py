import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split 

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from matplotlib import pyplot

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

import seaborn as sns

import eli5 

from eli5.sklearn import PermutationImportance 

from eli5 import show_weights 

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
features=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain","capital-loss", "hours-per-week", "native-country", "class"]

df = pd.read_csv("/kaggle/input/adult-incomes-in-the-united-states/adult.data", names=features)

y = df["class"]

X = df.drop(["class"], axis=1)



train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)
train_X.isna().sum()
train_X = train_X.drop(["fnlwgt"],axis=1)

val_X = val_X.drop(["fnlwgt"],axis=1)
high_cardinality = [col for col in train_X.columns if train_X[col].nunique() > 25]

high_cardinality
train_X = train_X.drop(["native-country"],axis=1)

val_X = val_X.drop(["native-country"],axis=1)
train_X = pd.get_dummies(train_X, prefix_sep='_', drop_first=True)

val_X = pd.get_dummies(val_X, prefix_sep='_', drop_first=True)

train_X.columns
train_y = pd.Series(np.where(train_y.values == ' >50K', 1, 0), train_y.index)

val_y = pd.Series(np.where(val_y.values == ' >50K', 1, 0), val_y.index)
train_y.value_counts()
my_model = XGBClassifier()



my_model.fit(train_X, train_y,  

             early_stopping_rounds=5,  

             eval_set=[(val_X, val_y)], 

             eval_metric="auc",

             verbose=False)
lr_probs = my_model.predict_proba(val_X)

lr_probs = lr_probs[:, 1]

yhat = my_model.predict(val_X)

lr_precision, lr_recall, _ = precision_recall_curve(val_y, lr_probs)

lr_f1, lr_auc = f1_score(val_y, yhat), auc(lr_recall, lr_precision)

print('XGBoost: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))

no_skill = len(val_y[val_y==1]) / len(val_y)

pyplot.plot([0, 1], [no_skill, no_skill], linestyle='-', label='No Skill')

pyplot.plot(lr_recall, lr_precision, marker='.', label='GXBoost')

pyplot.xlabel('Recall')

pyplot.ylabel('Precision')

pyplot.legend()

pyplot.show()
ns_probs = [0 for _ in range(len(val_y))]

ns_auc = roc_auc_score(val_y, ns_probs)

lr_auc = roc_auc_score(val_y, lr_probs)

print('No Skill: ROC AUC=%.3f' % (ns_auc))

print('XGBoost: ROC AUC=%.3f' % (lr_auc))

ns_fpr, ns_tpr, _ = roc_curve(val_y, ns_probs)

lr_fpr, lr_tpr, _ = roc_curve(val_y, lr_probs)

pyplot.plot(ns_fpr, ns_tpr, linestyle='-', label='No Skill')

pyplot.plot(lr_fpr, lr_tpr, marker='.', label='XGBoost')

pyplot.xlabel('False Positive Rate')

pyplot.ylabel('True Positive Rate')

pyplot.legend()

pyplot.show()
perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y) 

show_weights(perm, feature_names = val_X.columns.tolist()) 
import shap

explainer = shap.TreeExplainer(my_model) 

shap_values = explainer.shap_values(val_X) 

shap.summary_plot(shap_values, val_X) 
shap.dependence_plot('capital-gain', shap_values, val_X, interaction_index='capital-loss') 
shap.dependence_plot('capital-gain', shap_values, val_X, interaction_index='marital-status_ Married-civ-spouse') 
#for col in train_X.columns:

#    shap.dependence_plot('capital-gain', shap_values, val_X, interaction_index=col) 
shap.dependence_plot('capital-loss', shap_values, val_X, interaction_index="occupation_ Exec-managerial")
shap.dependence_plot('capital-gain', shap_values, val_X, interaction_index="occupation_ Exec-managerial")
train = train_X[["age","capital-gain","capital-loss","education-num", "hours-per-week","occupation_ Exec-managerial", "marital-status_ Married-civ-spouse"]]

correlations = train.corr()

fig, ax = pyplot.subplots(figsize=(10,10))

sns.heatmap(correlations,vmax=1.0, center=0, fmt='.2f',square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})

pyplot.show();
"""

occup_cols= ['occupation_ Adm-clerical',

       'occupation_ Armed-Forces', 'occupation_ Craft-repair',

       'occupation_ Exec-managerial', 'occupation_ Farming-fishing',

       'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct',

       'occupation_ Other-service', 'occupation_ Priv-house-serv',

       'occupation_ Prof-specialty', 'occupation_ Protective-serv',

       'occupation_ Sales', 'occupation_ Tech-support',

       'occupation_ Transport-moving']

exe_mang_income = pd.concat([train_y, train_X[occup_cols]], axis=1)

exe_mang_income.columns = ['Income >50K', 'occupation_ Adm-clerical',

       'occupation_ Armed-Forces', 'occupation_ Craft-repair',

       'occupation_ Exec-managerial', 'occupation_ Farming-fishing',

       'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct',

       'occupation_ Other-service', 'occupation_ Priv-house-serv',

       'occupation_ Prof-specialty', 'occupation_ Protective-serv',

       'occupation_ Sales', 'occupation_ Tech-support',

       'occupation_ Transport-moving']



for col in occup_cols:

    sns.countplot(x = col, hue="Income >50K", data=exe_mang_income)

    pyplot.show()

"""

cols_of_interest=["occupation_ Exec-managerial","marital-status_ Married-civ-spouse"]

exe_mang_income = pd.concat([train_y, train_X[cols_of_interest]], axis=1)

exe_mang_income.columns = ['Income >50K','occupation_ Exec-managerial', "marital-status_ Married-civ-spouse"]

for col in cols_of_interest:

    sns.countplot(x=col, hue="Income >50K", data=exe_mang_income)

    pyplot.show()

data_for_prediction = val_X.iloc[:100, :] 

shap_values = explainer.shap_values(data_for_prediction)

shap.initjs()

shap.force_plot(explainer.expected_value, shap_values, data_for_prediction)
data_for_prediction = val_X.iloc[34, :] 

shap_values = explainer.shap_values(data_for_prediction)

shap.initjs()

shap.force_plot(explainer.expected_value, shap_values, data_for_prediction)