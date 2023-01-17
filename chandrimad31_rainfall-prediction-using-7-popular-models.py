import pandas as pd

full_data = pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')

full_data.head()
full_data.shape
full_data.info()
full_data['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)

full_data['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8,5))

full_data.RainTomorrow.value_counts(normalize = True).plot(kind='bar', color= ['skyblue','navy'], alpha = 0.9, rot=0)

plt.title('RainTomorrow Indicator No(0) and Yes(1) in the Imbalanced Dataset')

plt.show()
from sklearn.utils import resample



no = full_data[full_data.RainTomorrow == 0]

yes = full_data[full_data.RainTomorrow == 1]

yes_oversampled = resample(yes, replace=True, n_samples=len(no), random_state=123)

oversampled = pd.concat([no, yes_oversampled])



fig = plt.figure(figsize = (8,5))

oversampled.RainTomorrow.value_counts(normalize = True).plot(kind='bar', color= ['skyblue','navy'], alpha = 0.9, rot=0)

plt.title('RainTomorrow Indicator No(0) and Yes(1) after Oversampling (Balanced Dataset)')

plt.show()
# Missing Data Pattern in Training Data

import seaborn as sns

sns.heatmap(oversampled.isnull(), cbar=False, cmap='PuBu')
total = oversampled.isnull().sum().sort_values(ascending=False)

percent = (oversampled.isnull().sum()/oversampled.isnull().count()).sort_values(ascending=False)

missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing.head(4)
oversampled.select_dtypes(include=['object']).columns
# Impute categorical var with Mode

oversampled['Date'] = oversampled['Date'].fillna(oversampled['Date'].mode()[0])

oversampled['Location'] = oversampled['Location'].fillna(oversampled['Location'].mode()[0])

oversampled['WindGustDir'] = oversampled['WindGustDir'].fillna(oversampled['WindGustDir'].mode()[0])

oversampled['WindDir9am'] = oversampled['WindDir9am'].fillna(oversampled['WindDir9am'].mode()[0])

oversampled['WindDir3pm'] = oversampled['WindDir3pm'].fillna(oversampled['WindDir3pm'].mode()[0])
# Convert categorical features to continuous features with Label Encoding

from sklearn.preprocessing import LabelEncoder

lencoders = {}

for col in oversampled.select_dtypes(include=['object']).columns:

    lencoders[col] = LabelEncoder()

    oversampled[col] = lencoders[col].fit_transform(oversampled[col])
import warnings

warnings.filterwarnings("ignore")
# Multiple Imputation by Chained Equations

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

MiceImputed = oversampled.copy(deep=True) 

mice_imputer = IterativeImputer()

MiceImputed.iloc[:, :] = mice_imputer.fit_transform(oversampled)
MiceImputed.head()
MiceImputed.isna()
# Detecting outliers with IQR

Q1 = MiceImputed.quantile(0.25)

Q3 = MiceImputed.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
# Removing outliers from dataset

MiceImputed = MiceImputed[~((MiceImputed < (Q1 - 1.5 * IQR)) |(MiceImputed > (Q3 + 1.5 * IQR))).any(axis=1)]

MiceImputed.shape
# Correlation Heatmap

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

corr = MiceImputed.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))

f, ax = plt.subplots(figsize=(20, 20))

cmap = sns.diverging_palette(250, 25, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .9})
sns.pairplot( data=MiceImputed, vars=('MaxTemp','MinTemp','Pressure9am','Pressure3pm', 'Temp9am', 'Temp3pm', 'Evaporation'), hue='RainTomorrow' )

# Standardizing data

from sklearn import preprocessing

r_scaler = preprocessing.MinMaxScaler()

r_scaler.fit(MiceImputed)

modified_data = pd.DataFrame(r_scaler.transform(MiceImputed), index=MiceImputed.index, columns=MiceImputed.columns)

modified_data.head()
# Feature Importance using Filter Method (Chi-Square)

from sklearn.feature_selection import SelectKBest, chi2

X = modified_data.loc[:,modified_data.columns!='RainTomorrow']

y = modified_data[['RainTomorrow']]

selector = SelectKBest(chi2, k=10)

selector.fit(X, y)

X_new = selector.transform(X)

print(X.columns[selector.get_support(indices=True)])
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier as rf



X = MiceImputed.drop('RainTomorrow', axis=1)

y = MiceImputed['RainTomorrow']

selector = SelectFromModel(rf(n_estimators=100, random_state=0))

selector.fit(X, y)

support = selector.get_support()

features = X.loc[:,support].columns.tolist()

print(features)

print(rf(n_estimators=100, random_state=0).fit(X,y).feature_importances_)
import warnings

warnings.filterwarnings("ignore")
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(rf(n_estimators=100, random_state=0).fit(X,y),random_state=1).fit(X,y)

eli5.show_weights(perm, feature_names = X.columns.tolist())
features = MiceImputed[['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 

                       'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 

                       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 

                       'RainToday']]

target = MiceImputed['RainTomorrow']



# Split into test and train

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=12345)



# Normalize Features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
def plot_roc_cur(fper, tper):  

    plt.plot(fper, tper, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
import time

from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, plot_confusion_matrix, roc_curve, classification_report

def run_model(model, X_train, y_train, X_test, y_test, verbose=True):

    t0=time.time()

    if verbose == False:

        model.fit(X_train,y_train, verbose=0)

    else:

        model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_pred) 

    coh_kap = cohen_kappa_score(y_test, y_pred)

    time_taken = time.time()-t0

    print("Accuracy = {}".format(accuracy))

    print("ROC Area under Curve = {}".format(roc_auc))

    print("Cohen's Kappa = {}".format(coh_kap))

    print("Time taken = {}".format(time_taken))

    print(classification_report(y_test,y_pred,digits=5))

    

    probs = model.predict_proba(X_test)  

    probs = probs[:, 1]  

    fper, tper, thresholds = roc_curve(y_test, probs) 

    plot_roc_cur(fper, tper)

    

    plot_confusion_matrix(model, X_test, y_test,cmap=plt.cm.Blues, normalize = 'all')

    

    return model, accuracy, roc_auc, coh_kap, time_taken
from sklearn.linear_model import LogisticRegression



params_lr = {'penalty': 'l1', 'solver':'liblinear'}



model_lr = LogisticRegression(**params_lr)

model_lr, accuracy_lr, roc_auc_lr, coh_kap_lr, tt_lr = run_model(model_lr, X_train, y_train, X_test, y_test)
from sklearn.tree import DecisionTreeClassifier



params_dt = {'max_depth': 16,

             'max_features': "sqrt"}



model_dt = DecisionTreeClassifier(**params_dt)

model_dt, accuracy_dt, roc_auc_dt, coh_kap_dt, tt_dt = run_model(model_dt, X_train, y_train, X_test, y_test)
from sklearn.neural_network import MLPClassifier



params_nn = {'hidden_layer_sizes': (30,30,30),

             'activation': 'logistic',

             'solver': 'lbfgs',

             'max_iter': 500}



model_nn = MLPClassifier(**params_nn)

model_nn, accuracy_nn, roc_auc_nn, coh_kap_nn, tt_nn = run_model(model_nn, X_train, y_train, X_test, y_test)
from sklearn.ensemble import RandomForestClassifier



params_rf = {'max_depth': 16,

             'min_samples_leaf': 1,

             'min_samples_split': 2,

             'n_estimators': 100,

             'random_state': 12345}



model_rf = RandomForestClassifier(**params_rf)

model_rf, accuracy_rf, roc_auc_rf, coh_kap_rf, tt_rf = run_model(model_rf, X_train, y_train, X_test, y_test)
import lightgbm as lgb

params_lgb ={'colsample_bytree': 0.95, 

         'max_depth': 16, 

         'min_split_gain': 0.1, 

         'n_estimators': 200, 

         'num_leaves': 50, 

         'reg_alpha': 1.2, 

         'reg_lambda': 1.2, 

         'subsample': 0.95, 

         'subsample_freq': 20}



model_lgb = lgb.LGBMClassifier(**params_lgb)

model_lgb, accuracy_lgb, roc_auc_lgb, coh_kap_lgb, tt_lgb = run_model(model_lgb, X_train, y_train, X_test, y_test)
import catboost as cb

params_cb ={'iterations': 50,

            'max_depth': 16}



model_cb = cb.CatBoostClassifier(**params_cb)

model_cb, accuracy_cb, roc_auc_cb, coh_kap_cb, tt_cb = run_model(model_cb, X_train, y_train, X_test, y_test, verbose=False)
import xgboost as xgb

params_xgb ={'n_estimators': 500,

            'max_depth': 16}



model_xgb = xgb.XGBClassifier(**params_xgb)

model_xgb, accuracy_xgb, roc_auc_xgb, coh_kap_xgb, tt_xgb = run_model(model_xgb, X_train, y_train, X_test, y_test)
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import itertools

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

import catboost as cb

import xgboost as xgb

from mlxtend.classifier import EnsembleVoteClassifier

from mlxtend.plotting import plot_decision_regions



value = 1.80

width = 0.90



clf1 = LogisticRegression(random_state=12345)

clf2 = DecisionTreeClassifier(random_state=12345) 

clf3 = MLPClassifier(random_state=12345, verbose = 0)

clf4 = RandomForestClassifier(random_state=12345)

clf5 = lgb.LGBMClassifier(random_state=12345, verbose = 0)

clf6 = cb.CatBoostClassifier(random_state=12345, verbose = 0)

clf7 = xgb.XGBClassifier(random_state=12345)

eclf = EnsembleVoteClassifier(clfs=[clf4, clf5, clf6, clf7], weights=[1, 1, 1, 1], voting='soft')



X_list = MiceImputed[["Sunshine", "Humidity9am", "Cloud3pm"]] #took only really important features

X = np.asarray(X_list, dtype=np.float32)

y_list = MiceImputed["RainTomorrow"]

y = np.asarray(y_list, dtype=np.int32)



# Plotting Decision Regions

gs = gridspec.GridSpec(3,3)

fig = plt.figure(figsize=(18, 14))



labels = ['Logistic Regression',

          'Decision Tree',

          'Neural Network',

          'Random Forest',

          'LightGBM',

          'CatBoost',

          'XGBoost',

          'Ensemble']



for clf, lab, grd in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, eclf],

                         labels,

                         itertools.product([0, 1, 2],

                         repeat=2)):

    clf.fit(X, y)

    ax = plt.subplot(gs[grd[0], grd[1]])

    fig = plot_decision_regions(X=X, y=y, clf=clf, 

                                filler_feature_values={2: value}, 

                                filler_feature_ranges={2: width}, 

                                legend=2)

    plt.title(lab)



plt.show()
accuracy_scores = [accuracy_lr, accuracy_dt, accuracy_nn, accuracy_rf, accuracy_lgb, accuracy_cb, accuracy_xgb]

roc_auc_scores = [roc_auc_lr, roc_auc_dt, roc_auc_nn, roc_auc_rf, roc_auc_lgb, roc_auc_cb, roc_auc_xgb]

coh_kap_scores = [coh_kap_lr, coh_kap_dt, coh_kap_nn, coh_kap_rf, coh_kap_lgb, coh_kap_cb, coh_kap_xgb]

tt = [tt_lr, tt_dt, tt_nn, tt_rf, tt_lgb, tt_cb, tt_xgb]



model_data = {'Model': ['Logistic Regression','Decision Tree','Neural Network','Random Forest','LightGBM','Catboost','XGBoost'],

              'Accuracy': accuracy_scores,

              'ROC_AUC': roc_auc_scores,

              'Cohen_Kappa': coh_kap_scores,

              'Time taken': tt}

data = pd.DataFrame(model_data)



fig, ax1 = plt.subplots(figsize=(12,10))

ax1.set_title('Model Comparison: Accuracy and Time taken for execution', fontsize=13)

color = 'tab:green'

ax1.set_xlabel('Model', fontsize=13)

ax1.set_ylabel('Time taken', fontsize=13, color=color)

ax2 = sns.barplot(x='Model', y='Time taken', data = data, palette='summer')

ax1.tick_params(axis='y')

ax2 = ax1.twinx()

color = 'tab:red'

ax2.set_ylabel('Accuracy', fontsize=13, color=color)

ax2 = sns.lineplot(x='Model', y='Accuracy', data = data, sort=False, color=color)

ax2.tick_params(axis='y', color=color)
fig, ax3 = plt.subplots(figsize=(12,10))

ax3.set_title('Model Comparison: Area under ROC and Cohens Kappa', fontsize=13)

color = 'tab:blue'

ax3.set_xlabel('Model', fontsize=13)

ax3.set_ylabel('ROC_AUC', fontsize=13, color=color)

ax4 = sns.barplot(x='Model', y='ROC_AUC', data = data, palette='winter')

ax3.tick_params(axis='y')

ax4 = ax3.twinx()

color = 'tab:red'

ax4.set_ylabel('Cohen_Kappa', fontsize=13, color=color)

ax4 = sns.lineplot(x='Model', y='Cohen_Kappa', data = data, sort=False, color=color)

ax4.tick_params(axis='y', color=color)

plt.show()