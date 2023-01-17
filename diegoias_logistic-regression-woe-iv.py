import numpy as np

import pandas as pd

import statsmodels.api as sm

from scipy.stats import zscore



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import KBinsDiscretizer



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
df = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

df.shape
df.head()
df['y'] = (df.Attrition == 'Yes').astype(int)

df.drop('Attrition', axis = 1, inplace = True)
StandardHours = 80

drop_fea = [

  'Over18', 

  'EmployeeCount', 

  'StandardHours', 

  'EmployeeNumber', 

]

df.drop(drop_fea, axis=1, inplace=True)
num_fea = [

  'Age',  

  'DailyRate',

  'DistanceFromHome',

  'HourlyRate',

  'MonthlyIncome', 

  'MonthlyRate',

  'NumCompaniesWorked',

  'PercentSalaryHike',

  'TotalWorkingYears', 

  'TrainingTimesLastYear',

  'YearsAtCompany', 

  'YearsInCurrentRole',

  'YearsSinceLastPromotion', 

  'YearsWithCurrManager',

]



cat_fea = [

  'BusinessTravel',

  'Department',

  'Education',

  'EducationField',

  'EnvironmentSatisfaction', # rating scale from 1 to 5

  'Gender',

  'JobInvolvement', # rating scale from 1 to 5

  'JobLevel', # rating scale from 1 to 5

  'JobRole',

  'JobSatisfaction', # rating scale from 1 to 5

  'MaritalStatus',

  'OverTime',

  'PerformanceRating', # rating scale from 1 to 5

  'RelationshipSatisfaction', # rating scale from 1 to 5

  'StockOptionLevel', # rating scale from 0 to 3

  'WorkLifeBalance', # rating scale from 1 to 5

]
df['PercentWorkingAtCompany'] = df['YearsAtCompany'] / df['TotalWorkingYears'] * 100

df['PercentCurrentRoleAtCompany'] = df['YearsInCurrentRole'] / df['YearsAtCompany'] * 100

df['PercentHourlyRate'] = df['HourlyRate'] / StandardHours * 100

num_fea += ['PercentWorkingAtCompany', 'PercentCurrentRoleAtCompany', 'PercentHourlyRate']

df.dropna(inplace=True)
attrittion = df.y.value_counts()

go.Figure([go.Pie(values = attrittion.values, labels=attrittion.index)])
df[num_fea] = df[num_fea].astype(float)
df[num_fea].describe(percentiles=[0.01,0.05,0.95,0.99])
df[num_fea].hist(figsize=(15, 15))

plt.show()
def normalize_discrete_values(df, feature, umbral = 0.05):

  aux = df[feature].value_counts(True).to_frame()

  aux['x'] = np.where(aux[feature] < umbral, 'Other', aux.index)

  if aux[aux['x'] == 'Other'][feature].sum() < umbral:

    aux['x'].replace({'Other' : aux.index[0]}, inplace=True)

  df[feature].replace(dict(zip(aux.index, aux.x)), inplace=True)

  return df
for fea in cat_fea:

  df = normalize_discrete_values(df, fea)
aux = df[num_fea].describe().T['max'] - df[num_fea].describe().T['min']

cut_features = aux[aux >= 6]

cut_features
range_to_int = dict()

int_to_range = dict()

for fea in cut_features.index:

  kb = KBinsDiscretizer(encode = 'ordinal', strategy = 'uniform')

  if cut_features[fea] < 10:

    kb.n_bins = 3

  elif cut_features[fea] < 50:

    kb.n_bins = 4

  else:

    kb.n_bins = 5

  df[fea] = kb.fit_transform(df[[fea]])

  ranges = ['%.2f|%.2f' % (a, b) for a, b in zip(kb.bin_edges_[0], kb.bin_edges_[0][1:])]

  int_to_range[fea] = dict(zip(range(len(ranges)), ranges))

  range_to_int[fea] = dict(zip(ranges, range(len(ranges))))

  df[fea].replace(int_to_range[fea], inplace=True)
def get_woe_iv(df, feature, target):

  lst = []

  for val in df[feature].unique():

      lst.append({

          'Value': val,

          'All': len(df[df[feature] == val]),

          'Good': len(df[(df[feature] == val) & (df[target] == 0)]),

          'Bad': len(df[(df[feature] == val) & (df[target] == 1)])

      })

      

  dset = pd.DataFrame(lst)

  dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()

  dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()

  dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])

  dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})

  dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']

  dset = dset.sort_values(by='WoE')

  

  return dset
iv_val = dict()

woe_val = dict()

for fea in cat_fea + num_fea:

  woe = get_woe_iv(df, fea, 'y')

  iv_val[fea] = woe['IV'].sum()

  woe_val[fea] = woe['WoE'].sum()

  print(fea)

  print(woe)

  print('WOE score: {:.2f}'.format(woe_val[fea]))

  print('IV score: {:.2f}'.format(iv_val[fea]))

  print()

iv_val = sorted(iv_val.items(), key=lambda item:item[1], reverse=True)
iv = pd.DataFrame(iv_val, columns=['Feature', 'IV'])

iv = iv[iv['IV'] > 0.1]

iv_fea = iv['Feature']

iv
for fea in num_fea:

  df[fea].replace(range_to_int[fea], inplace=True)



cat_str_fea = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

df = pd.concat([df, pd.get_dummies(df[cat_str_fea], drop_first=True)], axis=1, sort=False)

reverse_onehotencoding = df[cat_str_fea].copy()

df.drop(cat_str_fea, axis=1, inplace=True)

aux = [(x.split('_')[0] in list(iv_fea)) for x in df.columns]

iv_ohe_fea = df.columns[aux]



df = df.astype(float)

df.head()
X = df[iv_ohe_fea].copy()

y = df['y'].copy()



random_state = 760110

Xt, Xv, yt, yv = train_test_split(X, y, random_state = random_state)
model = LogisticRegression(C = 0.3, max_iter=len(Xt))

model.fit(Xt, yt)

print('roc_auc_score\t', roc_auc_score(yt, model.predict_proba(Xt)[:, 1]), roc_auc_score(yv, model.predict_proba(Xv)[:, 1]))

print('accuracy\t', accuracy_score(yt, model.predict(Xt)), accuracy_score(yv, model.predict(Xv)))

print('precision\t', precision_score(yt, model.predict(Xt)), precision_score(yv, model.predict(Xv)))

print('recall\t\t', recall_score(yt, model.predict(Xt)), recall_score(yv, model.predict(Xv)))

print('f1_score\t', f1_score(yt, model.predict(Xt)), f1_score(yv, model.predict(Xv)))
sns.heatmap(confusion_matrix(yt, model.predict(Xt)) / len(Xt), annot=True, fmt=".2f")

plt.show()

sns.heatmap(confusion_matrix(yv, model.predict(Xv)) / len(Xv), annot=True, fmt=".2f")

plt.show()
pd.DataFrame({'feature': iv_ohe_fea, 'coef': model.coef_[0]})
def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
fpr, tpr, thresholds = roc_curve(yt, model.predict_proba(Xt)[:, 1])

plot_roc_curve(fpr, tpr)

fpr, tpr, thresholds = roc_curve(yv, model.predict_proba(Xv)[:, 1])

plot_roc_curve(fpr, tpr)
aux = Xt.copy()

aux['proba'] = model.predict_proba(aux)[:,1]

aux['proba'] = pd.cut(aux['proba'],include_lowest=True,bins=np.arange(0,1.1,0.1))

aux.proba.value_counts(True).sort_index().plot(kind='bar')
pdo = 11.5  

base_score = 326 

base_odds = 1 

factor = float(pdo) / np.log(2)

offset = base_score - factor * np.log(base_odds)



n = len(iv_ohe_fea)

betas = model.coef_[0]

alpha = model.intercept_[0]

for feat, beta in zip(iv_ohe_fea, betas):

    aux['P_' + feat] = np.ceil((-aux[feat] * beta + alpha / n) * factor + offset / n).astype(int)

aux['score'] = aux[[f for f in aux.columns if f[:2] == 'P_']].sum(axis=1)

aux.score.describe()
aux.score.hist()
aux['r_score'] = pd.cut(aux.score, bins=range(min(aux.score) - 15,max(aux.score) + 15,15),include_lowest=True)

aux.r_score.value_counts().sort_index().plot(kind='bar')
aux['target']  = yt

aux['n']= 1.0

aux['r_score'] = aux['r_score'].astype(str)

aux['proba'] = aux['proba'].astype(str)

aux[['r_score','proba','target','n']].groupby(['r_score', 'proba', 'target']).sum()
for fea in [x for x in range_to_int.keys() if x in iv_ohe_fea]:

  aux[fea].replace(int_to_range[fea], inplace=True)



l_sc = []

for fea in iv_ohe_fea:

  aux2 = aux[[fea, 'P_%s' % fea]].copy().drop_duplicates().reset_index(drop=True)

  aux2.rename(columns={fea:'value','P_%s' % fea:'points'},inplace=True)

  aux2['feature'] = fea

  l_sc.append(aux2)



sc = pd.concat(l_sc,ignore_index=True)

sc = sc.groupby(['feature','value']).sum()

sc.reset_index()