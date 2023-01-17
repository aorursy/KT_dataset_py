import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



sns.set()



df_train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

df_test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')



print('Train dataset rows:', df_train.shape[0])

display(df_train.head())

print('Test dataset rows:', df_test.shape[0])

display(df_test.head())
print('Train dataset\n')

print(100*df_train.count().sort_values()/df_train.shape[0])



print('\n\nTest dataset\n')

print(100*df_test.count().sort_values()/df_test.shape[0])
fig, axes = plt.subplots(1, 2, figsize=(18, 4))

g1 = sns.barplot(x="sex", y="age_approx", data=df_train, estimator=lambda x: len(x) / len(df_train) * 100.0, ax = axes[0]).set(title = 'Train dataset', xlabel = 'Sex', ylabel = 'Frequency (%)', ylim = [0,60])

g2 = sns.barplot(x="sex", y="age_approx", data=df_test, estimator=lambda x: len(x) / len(df_test) * 100.0, ax = axes[1]).set(title = 'Test dataset', xlabel = 'Sex', ylabel = 'Frequency (%)', ylim = [0,60])



fig, axes = plt.subplots(figsize=(7.3, 4))

g3 = sns.barplot(x='sex', y='target', data = df_train, estimator = lambda x: np.mean(x) * 100.0, ci = None).set(title = 'Target distribution by Sex', xlabel = 'Sex', ylabel = 'Malignant prob (%)' )
fig, axes = plt.subplots(1, 2, figsize=(18, 4))

g1 = sns.barplot(x="age_approx", y="age_approx", data=df_train, estimator=lambda x: len(x) / len(df_train) * 100.0, ax = axes[0], color = 'orange')

g1.set(title = 'Train dataset', xlabel = 'Age', ylabel = 'Frequency (%)', ylim = [0,16])

plt.setp(g1.get_xticklabels(), rotation=90)

g2 = sns.barplot(x="age_approx", y="age_approx", data=df_test, estimator=lambda x: len(x) / len(df_test) * 100.0, ax = axes[1], color = 'orange')

g2.set(title = 'Test dataset', xlabel = 'Age', ylabel = 'Frequency (%)', ylim = [0,16])

plt.setp(g2.get_xticklabels(), rotation=90)



fig, axes = plt.subplots(figsize=(7.3, 4))

g3 = sns.barplot(x='age_approx', y='target', data = df_train, estimator = lambda x: np.mean(x) * 100.0, ci = None, color = 'orange')

g3.set(title = 'Target distribution by Age', xlabel = 'Age', ylabel = 'Malignant prob (%)' )

plt.setp(g3.get_xticklabels(), rotation=90);
fig, axes = plt.subplots(1, 2, figsize=(18, 4))

g1 = sns.barplot(x="anatom_site_general_challenge", y="age_approx", data=df_train, estimator=lambda x: len(x) / len(df_train) * 100.0, ax = axes[0], color = 'orange', order=df_train.anatom_site_general_challenge.sort_values().unique())

g1.set(title = 'Train dataset', xlabel = 'Location', ylabel = 'Frequency (%)', ylim = [0,60])

plt.setp(g1.get_xticklabels(), rotation=30)

g2 = sns.barplot(x="anatom_site_general_challenge", y="age_approx", data=df_test, estimator=lambda x: len(x) / len(df_test) * 100.0, ax = axes[1], color = 'orange', order=df_test.anatom_site_general_challenge.sort_values().unique())

g2.set(title = 'Test dataset', xlabel = 'Location', ylabel = 'Frequency (%)', ylim = [0,60])

plt.setp(g2.get_xticklabels(), rotation=30)



fig, axes = plt.subplots(figsize=(7.3, 4))

g3 = sns.barplot(x='anatom_site_general_challenge', y='target', data = df_train, estimator = lambda x: np.mean(x) * 100.0, ci = None, color = 'orange', order=df_train.anatom_site_general_challenge.unique())

g3.set(title = 'Target distribution by Location', xlabel = 'Location', ylabel = 'Malignant prob (%)' )

plt.setp(g3.get_xticklabels(), rotation=30);
from catboost import CatBoostClassifier, Pool, cv



df_train_naive = df_train[['sex', 'age_approx','anatom_site_general_challenge','target']]

df_train_naive.fillna({'sex':'Nan', 'age_approx':df_train['age_approx'].median(), 'anatom_site_general_challenge':'Nan'}, inplace = True)



df_test_naive = df_test[['sex', 'age_approx','anatom_site_general_challenge']]

df_test_naive.fillna({'sex':'Nan', 'age_approx':df_train['age_approx'].median(), 'anatom_site_general_challenge':'Nan'}, inplace = True)



cat_train = Pool(

    data = df_train_naive.drop(columns = ['target']),

    label = df_train_naive['target'],

    cat_features = ['sex', 'anatom_site_general_challenge'])



cat_test = Pool(

    data = df_test_naive,

    cat_features = ['sex', 'anatom_site_general_challenge'])



params = {

    "iterations": 2000,

    "random_seed": 0,

    "od_wait": 20,

    "learning_rate": 0.001,

    "loss_function": 'Logloss',

    "eval_metric": 'AUC'

    }



scores = cv(cat_train, params, fold_count = 5)
model_cat = CatBoostClassifier(

    iterations = 9,

    random_seed = 0,

    learning_rate = 0.005,

    loss_function = 'Logloss'

)



model_cat.fit(cat_train)



df_submit = df_test.copy()

df_submit['target'] = model_cat.predict_proba(cat_test)[:,1]

df_submit[['image_name', 'target']].to_csv('submission.csv', index=False )