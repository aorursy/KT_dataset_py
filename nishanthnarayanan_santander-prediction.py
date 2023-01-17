# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.max_columns = 9999 #Making sure all columns appear

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
FILEPATH_TRAIN = '/kaggle/input/santander-value-prediction-challenge/train.csv'
FILEPATH_TEST = '/kaggle/input/santander-value-prediction-challenge/test.csv'
train = pd.read_csv(FILEPATH_TRAIN, sep=',', engine='c') #Specify sep when using C engine
test = pd.read_csv(FILEPATH_TEST, sep=',', engine='c')
print("Shape of train:",train.shape, '\n',"Shape of test:",test.shape)
train.head()
train.describe()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(9, 7))
plt.scatter(range(train.shape[0]), np.sort(train['target'].values))
plt.xlabel('Index', fontsize=14)
plt.ylabel('Target', fontsize=14)
plt.title("Target Distribution", fontsize=14)
plt.show()
plt.figure(figsize=(10,8))
sns.distplot(train['target'].values, bins=50, kde=False)
plt.xlabel('Target', fontsize=11)
plt.title('Target Histogram', fontsize=11)
plt.show()
plt.figure(figsize=(10,8))
sns.distplot(np.log1p(train['target'].values), bins=50, kde=False)
plt.xlabel('Target', fontsize=11)
plt.title('Target Histogram', fontsize=11)
plt.show()
train.isnull().sum()
train.isnull().sum().any()
unique_vals = train.nunique().reset_index() #This drops NaN values by default
unique_vals.columns = ["Name", "Uniqueness"]
const_d = unique_vals[unique_vals["Uniqueness"]==1]
const_d.shape
str(const_d.Name.tolist())
#Ignore any warnings that arise
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
labels = []
values = []

for col in train.columns:
    if col not in ["ID", "target"]:
        labels.append(col)
        values.append(spearmanr(train[col].values, train['target'].values)[0])

correlation_df = pd.DataFrame({'column_label':labels, 'correlation_val':values})        
correlation_df = correlation_df.sort_values(by='correlation_val')

correlation_df = correlation_df[(correlation_df['correlation_val']>0.1) | (correlation_df['correlation_val']<-0.1)]
index = np.arange(correlation_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(10,25))
rec = ax.barh(index, np.array(correlation_df.correlation_val.values), color='r')
ax.set_yticks(index) #Set Y to index value of the df
ax.set_yticklabels(correlation_df.column_label.values, rotation='horizontal') #Define horizontal bar graph
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()
import seaborn as sns

columns = correlation_df[(correlation_df['correlation_val']>0.11) | (correlation_df['correlation_val']<-0.11)].column_label.tolist()

tmp = train[columns]
comat = tmp.corr(method='spearman') #Since we used spearman coefficient
fig, ax = plt.subplots(figsize=(30,30))

sns.heatmap(comat, square=True, cmap="RdYlGn", annot=True)
plt.title("Correlation Heatmap", fontsize=18)
plt.show()
tr_x = train.drop(const_d.Name.tolist()+ ["ID", "target"], axis=1)
te_x = test.drop(const_d.Name.tolist()+["ID"], axis=1)
tr_y = np.log1p(train['target'].values)
from sklearn import ensemble
model = ensemble.ExtraTreesRegressor(n_estimators=200, max_depth=20, max_features=0.5, n_jobs=-1, random_state=0)
model.fit(tr_x, tr_y)
#Plot Importance factor
features = tr_x.columns.values
importance = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importance)[::-1][:20]

plt.figure(figsize=(14,14))
plt.title("Feature Importances")
plt.bar(range(len(indices)), importance[indices], color="b", yerr=std[indices])
plt.xticks(range(len(indices)), features[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()
import lightgbm as lgb
def run_lgb(train_x, train_y, val_x, val_y, test_x):
    parameters = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 30,
        'learning_rate': 0.01,
        'bagging_fraction': 0.7,
        'feature_fraction': 0.7,
        'bagging_frequency': 5,
        'bagging_seed': 2018,
        'verbosity': -1
    }
    
    lgtrain = lgb.Dataset(train_x, label=train_y)
    lgval = lgb.Dataset(val_x, label=val_y)
    evals_result = {}
    model = lgb.train(parameters, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=200, evals_result=evals_result)
    
    pred_test_y = model.predict(test_x, num_iteration=model.best_iteration)
    
    return pred_test_y, model, evals_result
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=5, shuffle=True, random_state=2020)
pred_test_final = 0

for d_ind, v_ind in k_fold.split(tr_x):
    
    d_x, v_x = tr_x.loc[d_ind, :], tr_x.loc[v_ind, :]
    d_y, v_y = tr_y[d_ind], tr_y[v_ind]
    pred_test, model, evals_result = run_lgb(d_x, d_y, v_x, v_y, te_x)
    pred_test_final += pred_test
    
pred_test_final /= 5
pred_test_final = np.expm1(pred_test_final)
final_df = pd.DataFrame({"ID":test["ID"].values, "target":pred_test_final})
final_df.to_csv("submission.csv", index=False)
#Feature importance for LightGBM
fig, ax = plt.subplots(figsize=(14,20))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
#ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=16)
plt.show()