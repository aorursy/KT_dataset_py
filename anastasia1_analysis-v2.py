import time

import lightgbm

import numpy as np

import pandas as pd



import matplotlib



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



from sklearn.model_selection import GridSearchCV



from sklearn.metrics import f1_score

from sklearn.manifold import TSNE



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier





from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix



from sklearn.metrics import classification_report



from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier



from sklearn.metrics import roc_auc_score

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.feature_selection import chi2

from sklearn.preprocessing import MinMaxScaler
# matplotlib

plt.style.use('seaborn-whitegrid')

%config InlineBackend.figure_format = 'retina'

matplotlib.rcParams.update({'font.size': 14})



# pandas

pd.set_option('float_format', '{:f}'.format)
data = pd.read_csv('/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')

data.head(10)
data.describe().T
print('\033[1m Check none values \033[0m')

# check Nones

data.isna().sum()
# change target column name

data = data.rename(

    columns={'default.payment.next.month': 'Target'}

)
fig, axis = plt.subplots(1, 2, figsize=(18, 8))

sns.countplot(data['MARRIAGE'], ax=axis[0])

sns.countplot(data['EDUCATION'], ax=axis[1])
fig, axis = plt.subplots(2, 3, figsize=(18, 9))

sns.countplot('PAY_0', ax=axis[0, 0], data=data)

sns.countplot('PAY_2', ax=axis[0, 1], data=data)

sns.countplot('PAY_3', ax=axis[0, 2], data=data)

sns.countplot('PAY_4', ax=axis[1, 0], data=data)

sns.countplot('PAY_5', ax=axis[1, 1], data=data)

sns.countplot('PAY_6', ax=axis[1, 2], data=data)
fig, axis = plt.subplots(1, 2, figsize=(18, 8))

sns.distplot(data['AGE'], ax=axis[0], norm_hist=False, kde=False)

sns.countplot(data['SEX'], ax=axis[1])
# fix issues of labeling 



# assign 0 class to 3rd class

data['MARRIAGE'] = data['MARRIAGE'].replace({0: 3})



# assign 6 and 0 class to 5th class

data['EDUCATION'] = data['EDUCATION'].replace({6: 5, 0: 5})



# iterate over columns and assign -1, and -2 class to 0

for column in ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:

    data[column] = data[column].replace({-1: 0, -2: 0})
fig, axis = plt.subplots(2, 3, figsize=(24, 15))



sns.distplot(data['BILL_AMT1'], ax=axis[0, 0], kde=False, norm_hist=False, bins=20)

sns.distplot(data['BILL_AMT2'], ax=axis[0, 1], kde=False, norm_hist=False, bins=20)

sns.distplot(data['BILL_AMT3'], ax=axis[0, 2], kde=False, norm_hist=False, bins=20)



sns.boxplot(data['BILL_AMT1'], ax=axis[1, 0])

sns.boxplot(data['BILL_AMT2'], ax=axis[1, 1])

sns.boxplot(data['BILL_AMT3'], ax=axis[1, 2])
fig, axis = plt.subplots(2, 3, figsize=(24, 15))



sns.distplot(data['BILL_AMT4'], ax=axis[0, 0], kde=False, norm_hist=False, bins=20)

sns.distplot(data['BILL_AMT5'], ax=axis[0, 1], kde=False, norm_hist=False, bins=20)

sns.distplot(data['BILL_AMT6'], ax=axis[0, 2], kde=False, norm_hist=False, bins=20)



sns.boxplot(data['BILL_AMT4'], ax=axis[1, 0])

sns.boxplot(data['BILL_AMT5'], ax=axis[1, 1])

sns.boxplot(data['BILL_AMT6'], ax=axis[1, 2])
fig, axis = plt.subplots(2, 3, figsize=(24, 15))

print("\033[1m Distribution of PAY_AMT 1-3 \033[0m")

sns.distplot(data['PAY_AMT1'], ax=axis[0, 0], kde=False, norm_hist=False)

sns.distplot(data['PAY_AMT2'], ax=axis[0, 1], kde=False, norm_hist=False)

sns.distplot(data['PAY_AMT3'], ax=axis[0, 2], kde=False, norm_hist=False)



sns.boxplot(data['PAY_AMT1'], ax=axis[1, 0])

sns.boxplot(data['PAY_AMT2'], ax=axis[1, 1])

sns.boxplot(data['PAY_AMT3'], ax=axis[1, 2])
fig, axis = plt.subplots(2, 3, figsize=(24, 15))

print("\033[1m Distribution of PAY_AMT 4-6 \033[0m")



sns.distplot(data['PAY_AMT4'], ax=axis[0, 0], kde=False, norm_hist=False)

sns.distplot(data['PAY_AMT5'], ax=axis[0, 1], kde=False, norm_hist=False)

sns.distplot(data['PAY_AMT6'], ax=axis[0, 2], kde=False, norm_hist=False)



sns.boxplot(data['PAY_AMT4'], ax=axis[1, 0])

sns.boxplot(data['PAY_AMT5'], ax=axis[1, 1])

sns.boxplot(data['PAY_AMT6'], ax=axis[1, 2])
filtered_data = list(

    data[column][data[column] <= data[column].quantile(0.75)]

    for column in ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

)



fig, axis = plt.subplots(2, 3, figsize=(24, 15))

print("\033[1m Розподіл значень PAY_AMT 1-3 \033[0m")



sns.distplot(filtered_data[0], ax=axis[0, 0], kde=False, norm_hist=False)

sns.distplot(filtered_data[1], ax=axis[0, 1], kde=False, norm_hist=False)

sns.distplot(filtered_data[2], ax=axis[0, 2], kde=False, norm_hist=False)



sns.boxplot(filtered_data[0], ax=axis[1, 0])

sns.boxplot(filtered_data[1], ax=axis[1, 1])

sns.boxplot(filtered_data[2], ax=axis[1, 2])
fig, axis = plt.subplots(2, 3, figsize=(24, 15))

print("\033[1m Розподіл значень PAY_AMT 4-6 \033[0m")



sns.distplot(filtered_data[3], ax=axis[0, 0], kde=False, norm_hist=False)

sns.distplot(filtered_data[4], ax=axis[0, 1], kde=False, norm_hist=False)

sns.distplot(filtered_data[5], ax=axis[0, 2], kde=False, norm_hist=False)



sns.boxplot(filtered_data[3], ax=axis[1, 0])

sns.boxplot(filtered_data[4], ax=axis[1, 1])

sns.boxplot(filtered_data[5], ax=axis[1, 2])
correlation = data.corr()



mask = np.zeros_like(

    correlation,

    dtype=np.bool

)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(25, 25))



sns.heatmap(

    correlation,

    xticklabels=correlation.columns,

    yticklabels=correlation.columns,

    linewidths=.1,

    vmin=-1,

    vmax=1,

    annot=True,

    mask=mask

)
# encode

data = pd.get_dummies(data, columns=['SEX', 'MARRIAGE', 'EDUCATION'])
# drop junk columns

data = data.drop(columns=['SEX_2', 'ID'])
pd.set_option('display.max_columns', 50)

data.head(10)
# select columns for visualization

data_tmp_1 = data[['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]

data_tmp_2 = data[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'AGE']]
sns.pairplot(data_tmp_1)
sns.pairplot(data_tmp_2)
plt.figure(figsize=(18, 8))

sns.countplot(data['Target'])
Y = data['Target']  # select target 

X = data[data.columns.difference(['Target'])]  # select features
scaler = MinMaxScaler()



for column in X.columns:

    X[column] = scaler.fit_transform(np.array(X[column]).reshape(-1,1))
X.head(10)
selector_chi = SelectKBest(chi2, k=10)  # using Chi2 select 10 best features

selector_chi.fit(X, Y)

features_1_scores = selector_chi.scores_

features_1 = list(X.columns[selector_chi.get_support(indices=True)])



plt.figure(figsize=(18, 8))

features_scores = pd.DataFrame({'features': X.columns.tolist(), 'scores': features_1_scores})

sns.barplot('features', 'scores', data=features_scores, order=features_scores.sort_values('scores')['features'])

plt.title('Best features - Chi2')

plt.ylabel('Score')

plt.xticks(list(range(len(features_1_scores))), X.columns.tolist(), rotation=45)

plt.show()
selector = SelectKBest(f_classif, k=10)  # using f_classif select 10 best features

selector.fit(X, Y)

features_2_scores = selector.scores_

features_2 = list(X.columns[selector.get_support(indices=True)])



plt.figure(figsize=(18, 8))

features_scores = pd.DataFrame({'features': X.columns.tolist(), 'scores': features_2_scores})

sns.barplot('features', 'scores', data=features_scores, order=features_scores.sort_values('scores')['features'])

plt.title('Best features - ANOVA F-value')

plt.ylabel('Score')

plt.xticks(list(range(len(features_2_scores))), X.columns.tolist(), rotation=45)

plt.show()
features = set(features_1 + features_2)

features
# select only best features

X = X[features]
X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.3, stratify=Y)

X_dev, X_test, y_dev, y_test = train_test_split(X_dev, y_dev, test_size=0.5, stratify=y_dev)
fig, axis = plt.subplots(1, 3, figsize=(25, 8))



sns.countplot(y_train, ax=axis[0], label='train')

sns.countplot(y_dev, ax=axis[1], label='dev')

sns.countplot(y_test, ax=axis[2], label='test')

axis[0].set_title('train')

axis[1].set_title('dev')

axis[2].set_title('test')

plt.show()
# list of models

models = [

    LogisticRegression(solver="liblinear"),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators=10),

    XGBClassifier(),

    GradientBoostingClassifier(),

    LGBMClassifier(),

]
auc_scores = []

cv_scores = []

acc_scores = []



# iterate over models

for model in models:

    

    # fit model

    model.fit(X_train, y_train)

    

    # predict on dev

    prediction = model.predict(X_dev)

    

    # predict probability on dev

    probability = model.predict_proba(X_dev)

    

    # get AUC

    auc = roc_auc_score(y_dev, probability[:,1])

    

    # get CV score on train

    cv_score = cross_val_score(model, X_train, y_train, cv=10).mean()

    

    # get acc

    score = accuracy_score(y_dev, prediction)

    

    # get report

    report = classification_report(y_dev, prediction, zero_division=1)

    

    # print report

    name = str(model)

    print(name[0:name.find("(")])

    

    print("Accuracy :", score)

    print("CV Score :", cv_score)

    print("AUC Score : ", auc)

    print(report)

    print(confusion_matrix(y_dev, prediction))

    print(" \033[1m ------------------------------------------------------------ \033[0m ")

    

    auc_scores.append(auc)

    cv_scores.append(cv_score)

    acc_scores.append(score)
metrics = pd.DataFrame({

    'AUC': auc_scores,

    'CV Score': cv_scores,

    'Accuracy': acc_scores,

    'name': ['LogisticRegression',

             'DecisionTreeClassifier',

             'RandomForestClassifier',

             'XGBClassifier',

             'GradientBoostingClassifier',

             'LGBMClassifier']

})
fig, axis = plt.subplots(3, 1, figsize=(25, 25))



sns.barplot(x='name', y='AUC', data=metrics, order=metrics.sort_values('AUC')['name'], ax=axis[0])

sns.barplot(x='name', y='CV Score', data=metrics, order=metrics.sort_values('CV Score')['name'], ax=axis[1])

sns.barplot(x='name', y='Accuracy', data=metrics, order=metrics.sort_values('Accuracy')['name'], ax=axis[2])

plt.show()
lgbm_params = {"n_estimators" : [100, 500, 1000],

               "num_leaf": [5, 15, 25, 30],

               "subsample" : [0.6, 0.8, 1.0],

               "learning_rate" : [0.1, 0.01, 0.02],

               "min_child_samples" : [5, 10, 20]}
lgbm_model = LGBMClassifier()



lgbm_cv_model = GridSearchCV(

    lgbm_model, 

    lgbm_params, 

    cv=5,

    verbose=1,

    n_jobs=-1)
lgbm_cv_model.fit(X_train, y_train)
print('Best params LightGBM')

best_params = lgbm_cv_model.best_params_

best_params
auc_scores = []

f1_scores = []

cv_scores = []

acc_scores = []
# train LightGBM with best params and predict on test



model = LGBMClassifier(

**best_params

)



# fit model

model.fit(X_train, y_train)



# predict on dev

prediction = model.predict(X_test)



# predict probability on dev

probability = model.predict_proba(X_test)



# get AUC

auc = roc_auc_score(y_test, probability[:,1])



# get CV score on train

cv_score = cross_val_score(model, X_train, y_train, cv=10).mean()



# get acc

score = accuracy_score(y_test, prediction)



# get report

report = classification_report(y_test, prediction, zero_division=1)





print("Accuracy :", score)

print("CV Score :", cv_score)

print("AUC Score : ", auc)

print(report)

print(confusion_matrix(y_test, prediction))

print(" \033[1m ------------------------------------------------------------ \033[0m ")





auc_scores.append(auc)

acc_scores.append(score)

cv_scores.append(cv_score)

f1_scores.append(f1_score(y_test, prediction))
gb_params = parameters = {

    "learning_rate": [0.01, 0.05, 0.075, 0.1],

    "min_samples_leaf": np.linspace(0.1, 0.5, 4),

    "max_depth":[3,5,8],

    "max_features":["log2","sqrt"],

    "subsample":[0.5, 0.6, 0.8],

    "n_estimators":[10, 30, 60, 90]

    }
gb_model = GradientBoostingClassifier()



gb_cv_model = GridSearchCV(

    gb_model, 

    gb_params, 

    cv=5,

    verbose=1,

    n_jobs=-1)
gb_cv_model.fit(X_train, y_train)

print(' ')
print('Best params Gradient Boosting')

best_params = gb_cv_model.best_params_

best_params
# train GradientBoosting with best params and predict on test



model = GradientBoostingClassifier(

**best_params

)



# fit model

model.fit(X_train, y_train)



# predict on dev

prediction = model.predict(X_test)



# predict probability on dev

probability = model.predict_proba(X_test)



# get AUC

auc = roc_auc_score(y_test, probability[:,1])



# get CV score on train

cv_score = cross_val_score(model, X_train, y_train, cv=10).mean()



# get acc

score = accuracy_score(y_test, prediction)



# get report

report = classification_report(y_test, prediction, zero_division=1)





print("Accuracy :", score)

print("CV Score :", cv_score)

print("AUC Score : ", auc)

print(report)

print(confusion_matrix(y_test, prediction))

print(" \033[1m ------------------------------------------------------------ \033[0m ")





auc_scores.append(auc)

acc_scores.append(score)

cv_scores.append(cv_score)

f1_scores.append(f1_score(y_test, prediction))
metrics = pd.DataFrame({

    'AUC': auc_scores,

    'Accuracy': acc_scores,

    'F1': f1_scores,

    'CV Score': cv_scores,

    'name': ['LGBMClassifier',

             'GradientBoostingClassifier']

})
fig, axis = plt.subplots(4, 1, figsize=(25, 25))



sns.barplot(x='name', y='AUC', data=metrics, order=metrics.sort_values('AUC')['name'], ax=axis[0])

sns.barplot(x='name', y='CV Score', data=metrics, order=metrics.sort_values('CV Score')['name'], ax=axis[1])

sns.barplot(x='name', y='Accuracy', data=metrics, order=metrics.sort_values('Accuracy')['name'], ax=axis[2])

sns.barplot(x='name', y='F1', data=metrics, order=metrics.sort_values('Accuracy')['name'], ax=axis[3])

plt.show()