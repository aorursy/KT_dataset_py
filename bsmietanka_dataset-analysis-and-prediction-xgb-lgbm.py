import pandas as pd

from matplotlib import pyplot as plt

import numpy as np

import seaborn as sns

from ipywidgets import interact

import warnings



from sklearn.model_selection import train_test_split



warnings.filterwarnings("ignore")

%matplotlib inline
df = pd.read_csv(r'../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()
df.tail()
print(df.columns.values)
df.info()
df.isnull().any()
df.isnull().values.any()
cat_df = df.select_dtypes(include = 'object')

num_types = [t for t in df.dtypes.unique() if t not in cat_df.dtypes.unique()]

num_df = df.select_dtypes(include = num_types)
num_df.describe()
drop_labels = num_df.columns[num_df.std() == 0]

num_df.drop(columns = drop_labels, inplace = True)
potential_cat_df = num_df[num_df.columns[num_df.nunique() <= 5]].astype('str')

reduced_num_df = num_df.drop(columns=potential_cat_df.columns)

ext_cat_df = pd.concat([cat_df, potential_cat_df], axis=1)
cat_df.describe()
drop_labels = cat_df.columns[cat_df.nunique() == 1]

cat_df.drop(columns = drop_labels, inplace = True)
def num_dist_plot(feature):

    sns.distplot(df[feature])

interact(num_dist_plot, feature=reduced_num_df.columns);
num_df["EmployeeNumber"].nunique()
num_df.drop(columns = "EmployeeNumber", inplace = True)
corr = num_df.corr()

mask = np.array(corr)

mask[np.tril_indices_from(mask)] = False

fig=plt.gcf()

fig.set_size_inches(30,12)

sns.heatmap(data = corr, mask = mask, square = True, annot = True, cbar = True);
temp_df = pd.concat([cat_df["Attrition"], reduced_num_df], axis=1)

def boxplot_numerical_target(feature):

    sns.boxplot(x="Attrition", y=feature, data=temp_df)



interact(boxplot_numerical_target, feature=reduced_num_df.columns);
num_df.drop(columns=["DailyRate", "HourlyRate", "MonthlyRate", "TrainingTimesLastYear"], inplace=True)
def relation_to_attrition(feature):

    grouped = ext_cat_df.groupby([feature, "Attrition"])["Attrition"].count().unstack()

    grouped.plot(kind="bar", stacked=True)

    xtab = pd.crosstab(columns=ext_cat_df.Attrition, index=ext_cat_df[feature], margins=True, normalize='index')

    table = plt.table(cellText=np.round(xtab.values, 3), rowLabels=xtab.index,

            colLabels=xtab.columns, loc='top', cellLoc='center')

    table.auto_set_column_width(range(xtab.columns.size))

    fig=plt.gcf()

    fig.set_size_inches(8,6)



interact(relation_to_attrition, feature=ext_cat_df.columns.drop("Attrition"));
selected = pd.concat([cat_df, num_df], axis=1)

selected.info()
hyp_df = selected[['Gender', 'JobRole', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']]

hyp_df.info()

female = hyp_df[hyp_df.Gender == 'Female']

male = hyp_df[hyp_df.Gender == 'Male']
from scipy import stats

from IPython.display import display



alpha = 0.1

df = len(hyp_df.index) - 2

# alpha/2, because it's a two-tailed test

crit_val = np.abs(stats.t.ppf(alpha/2, df))
sns.boxplot(x='Gender', y='MonthlyIncome', data=hyp_df);

t, p = stats.ttest_ind(female['MonthlyIncome'], male['MonthlyIncome'], equal_var=False)

if np.abs(t) < crit_val:

    display(f"Can't reject null hypothesis (t_val : {t}, p_val : {p})")

else:

    display(f"Hypothesis rejected (t_val : {t}, p_val : {p})")
sns.boxplot(x="JobRole", y="MonthlyIncome", hue="Gender", data=hyp_df);

fig=plt.gcf()

fig.set_size_inches(8, 8)



def test_for_position(position):

    dof = len(hyp_df[hyp_df.JobRole == position].index) - 2

    cv = np.abs(stats.t.ppf(alpha/2, dof))

    t, p = stats.ttest_ind(female[female.JobRole == position]['MonthlyIncome'], male[male.JobRole == position]['MonthlyIncome'], equal_var=False)

    if np.abs(t) < cv:

        display(f"Can't reject null hypothesis (t_val : {t}, p_val : {p})")

    else:

        display(f"Hypothesis rejected (t_val : {t}, p_val : {p})")



interact(test_for_position, position=hyp_df["JobRole"].unique());
fig, ax = plt.subplots(1, 2)

fig.set_size_inches(12, 5)

sns.boxplot(y="YearsAtCompany", x="Gender", data=hyp_df[hyp_df.JobRole == 'Research Director'], ax=ax[0]);

sns.boxplot(y="TotalWorkingYears", x="Gender", data=hyp_df[hyp_df.JobRole == 'Research Director'], ax=ax[1]);



hyp_df[hyp_df.JobRole == 'Research Director'].groupby('Gender')[['YearsAtCompany', 'TotalWorkingYears']].describe()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



cat_mask = selected.dtypes==object

cat_cols = selected.columns[cat_mask].tolist()



selected[cat_cols] = selected[cat_cols].apply(lambda col: le.fit_transform(col))
# could be also done with sklearn.model_selection.train_test_split

mask = np.random.rand(len(selected)) < 0.8

train = selected[mask]

test = selected[~mask]



y_train = train['Attrition']

x_train = train.drop(columns='Attrition')



y_test = test['Attrition']

x_test = test.drop(columns='Attrition')
from imblearn.over_sampling import SMOTE



oversampler=SMOTE(random_state=1234)

x_train_smote,  y_train_smote = oversampler.fit_resample(x_train,y_train)

x_train_smote = pd.DataFrame(data=x_train_smote, columns=x_train.columns)

y_train_smote = pd.Series(data=y_train_smote)
import lightgbm as lgb

import xgboost as xgb

from sklearn.model_selection import GridSearchCV



params = {

        'num_iterations' : [50, 200, 500, 1000],

        'learning_rate' : [0.05, 0.1, 0.25],

        'subsample': [0.2, 0.4, 0.6, 0.8],

        'num_leaves': [4, 6, 10, 20, 50]

        }



gsearch_LGBM = GridSearchCV(estimator=lgb.LGBMClassifier(), param_grid=params,

                            scoring='recall', n_jobs=-1, cv=5)



gsearch_XGB  = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=params,

                            scoring='recall', n_jobs=-1, cv=5)



%time gsearch_LGBM.fit(x_train_smote, y_train_smote);

%time gsearch_XGB.fit(x_train_smote, y_train_smote);

print("Training finished")
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import json



pred_LGBM = gsearch_LGBM.predict(x_test) > 0.5

pred_XGB = gsearch_XGB.predict(x_test) > 0.5

preds = {"XGB" : pred_XGB, "LGBM" : pred_LGBM}

metrics = {}

for k, v in preds.items():

    metrics[k] = {'acc' : accuracy_score(y_test, v), 'prec' : precision_score(y_test, v),

                  'rec' : recall_score(y_test, v),   'roc' : roc_auc_score(y_test, v)}



print(f'XGB params: {gsearch_XGB.best_params_}')

print(f'XGB score : {gsearch_XGB.best_score_}')

print('XGB scores: {}'.format(json.dumps(metrics['XGB'], indent=4)))

print(f'LGBM params : {gsearch_LGBM.best_params_}')

print(f'LGBM score : {gsearch_LGBM.best_score_}')

print('LGBM scores: {}'.format(json.dumps(metrics['LGBM'], indent=4)))
lgb.plot_importance(gsearch_LGBM.best_estimator_, figsize=(6, 6), title='LGBM');

ax = xgb.plot_importance(gsearch_XGB.best_estimator_, title='XGB');

fig = ax.figure

fig.set_size_inches(6, 6)