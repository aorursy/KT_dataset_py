import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings("ignore")
data=pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
data.head(10)
data.info()
data.describe()
f,ax = plt.subplots(figsize=(20, 20))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
sns.countplot(data['OverTime'])

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('OverTime')
sns.countplot(data['MaritalStatus'])

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('Marital Status')
sns.countplot(data['JobRole'])

fig = plt.gcf()

fig.set_size_inches(20,14)

plt.title('Job Role')
sns.countplot(data['Gender'])

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('Gender')
sns.countplot(data['EducationField'])

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('Education Field')
sns.countplot(data['Department'])

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('eEpartment')
sns.countplot(data['BusinessTravel'])

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('Business travel')
sns.catplot(x="OverTime", y="Age", kind="swarm", data=data);
plt.subplots(figsize=(15,5))

sns.countplot(data.TotalWorkingYears)
plt.subplots(figsize=(15,5))

sns.countplot(data.Education)
sns.countplot(data.NumCompaniesWorked)
plt.subplots(figsize=(18,5))

sns.countplot(data.DistanceFromHome)
import time

import math

import seaborn as sns

import pandas as pd

import numpy as np

import scipy as sci

import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)



from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from hyperopt import hp, tpe, Trials, STATUS_OK

from hyperopt import fmin



from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.model_selection import train_test_split



from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight') 

%matplotlib inline
ibm_df = data

description = pd.DataFrame(index=['observations(rows)', 'percent missing', 'dtype', 'range'])

numerical = []

categorical = []

for col in ibm_df.columns:

    obs = ibm_df[col].size

    p_nan = round(ibm_df[col].isna().sum()/obs, 2)

    num_nan = f'{p_nan}% ({ibm_df[col].isna().sum()}/{obs})'

    dtype = 'categorical' if ibm_df[col].dtype == object else 'numerical'

    numerical.append(col) if dtype == 'numerical' else categorical.append(col)

    rng = f'{len(ibm_df[col].unique())} labels' if dtype == 'categorical' else f'{ibm_df[col].min()}-{ibm_df[col].max()}'

    description[col] = [obs, num_nan, dtype, rng]



numerical.remove('EmployeeCount')

numerical.remove('StandardHours')

pd.set_option('display.max_columns', 100)

display(description)

display(ibm_df.head())
def org_results(trials, hyperparams, model_name):

    fit_idx = -1

    for idx, fit  in enumerate(trials):

        hyp = fit['misc']['vals']

        xgb_hyp = {key:[val] for key, val in hyperparams.items()}

        if hyp == xgb_hyp:

            fit_idx = idx

            break

            

    train_time = str(trials[-1]['refresh_time'] - trials[0]['book_time'])

    acc = round(trials[fit_idx]['result']['accuracy'], 3)

    train_auc = round(trials[fit_idx]['result']['train auc'], 3)

    test_auc = round(trials[fit_idx]['result']['test auc'], 3)



    results = {

        'model': model_name,

        'parameter search time': train_time,

        'accuracy': acc,

        'test auc score': test_auc,

        'training auc score': train_auc,

        'parameters': hyperparams

    }

    return results
lgb_data = ibm_df.copy()

lgb_dummy = pd.get_dummies(lgb_data[categorical], drop_first=True)

lgb_data = pd.concat([lgb_dummy, lgb_data], axis=1)

lgb_data.drop(columns = categorical, inplace=True)

lgb_data.rename(columns={'Attrition_Yes': 'Attrition'}, inplace=True)



y_df = lgb_data['Attrition'].reset_index(drop=True)

x_df = lgb_data.drop(columns='Attrition')

train_x, test_x, train_y, test_y = train_test_split(x_df, y_df, test_size=0.20)



def lgb_objective(space, early_stopping_rounds=50):

    

    lgbm = LGBMClassifier(

        learning_rate = space['learning_rate'],

        n_estimators= int(space['n_estimators']), 

        max_depth = int(space['max_depth']),

        num_leaves = int(space['num_leaves']),

        colsample_bytree = space['colsample_bytree'],

        feature_fraction = space['feature_fraction'],

        reg_lambda = space['reg_lambda'],

        reg_alpha = space['reg_alpha'],

        min_split_gain = space['min_split_gain']

    )

    

    lgbm.fit(train_x, train_y, 

            eval_set = [(train_x, train_y), (test_x, test_y)],

            early_stopping_rounds = early_stopping_rounds,

            eval_metric = 'auc',

            verbose = False)

    

    predictions = lgbm.predict(test_x)

    test_preds = lgbm.predict_proba(test_x)[:,1]

    train_preds = lgbm.predict_proba(train_x)[:,1]

    

    train_auc = roc_auc_score(train_y, train_preds)

    test_auc = roc_auc_score(test_y, test_preds)

    accuracy = accuracy_score(test_y, predictions)  



    return {'status': STATUS_OK, 'loss': 1-test_auc, 'accuracy': accuracy,

            'test auc': test_auc, 'train auc': train_auc

           }



trials = Trials()

space = {

    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),

    'n_estimators': hp.quniform('n_estimators', 50, 1200, 25),

    'max_depth': hp.quniform('max_depth', 1, 15, 1),

    'num_leaves': hp.quniform('num_leaves', 10, 150, 1),

    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0), 

    'feature_fraction': hp.uniform('feature_fraction', .3, 1.0),

    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),

    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),

    'min_split_gain': hp.uniform('min_split_gain', 0.0001, 0.1)

}



lgb_hyperparams = fmin(fn = lgb_objective, 

                 max_evals = 150, 

                 trials = trials,

                 algo = tpe.suggest,

                 space = space

                 )



lgb_results = org_results(trials.trials, lgb_hyperparams, 'LightGBM')

display(lgb_results)
age=pd.DataFrame(data.groupby("Age")[["MonthlyIncome","Education","JobLevel","JobInvolvement","PerformanceRating","JobSatisfaction","EnvironmentSatisfaction","RelationshipSatisfaction","WorkLifeBalance","DailyRate","MonthlyRate"]].mean())

age["Count"]=data.Age.value_counts(dropna=False)

age.reset_index(level=0, inplace=True)

age.head()
plt.figure(figsize=(15,10))

ax=sns.barplot(x=age.Age,y=age.Count)

plt.xticks(rotation=180)

plt.xlabel("Age")

plt.ylabel("Counts")

plt.title("Age Counts")

plt.show()
plt.figure(figsize=(15,10))

ax=sns.barplot(x=age.Age,y=age.MonthlyIncome,palette = sns.cubehelix_palette(len(age.index)))

plt.xticks(rotation=180)

plt.xlabel("Age")

plt.ylabel("Monthly Income")

plt.title("Monthly Income According to Age")

plt.show()
income=pd.DataFrame(data.groupby("JobRole").MonthlyIncome.mean().sort_values(ascending=False))
plt.figure(figsize=(15,10))

ax=sns.barplot(x=income.index,y=income.MonthlyIncome)

plt.xticks(rotation=90)

plt.xlabel("Job Roles")

plt.ylabel("Monthly Income")

plt.title("Job Roles with Monthly Income")

plt.show()
jobrole=pd.DataFrame(data.groupby("JobRole")["PercentSalaryHike","YearsAtCompany","TotalWorkingYears","YearsInCurrentRole","WorkLifeBalance"].mean())

jobrole
labels=data.EducationField.value_counts().index

colors=["cyan","orange","hotpink","green","navy","#9b59b6"]

#explode=[0,0,0,0,0,0]

sizes=data.EducationField.value_counts().values

plt.figure(figsize=(7,7))

plt.pie(sizes,labels=labels,colors=colors,autopct="%1.1f%%")

plt.title("Education Field Counts",color="saddlebrown",fontsize=15)