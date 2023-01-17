#Use this regressors only
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import AdaBoostRegressor


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split



%matplotlib inline
tables = '../input/intertaskds/tables.xlsx'
tablescsv = '../input/intrtasksdscsv/tables.csv'
# pip install deepchem
tax=pd.read_excel(tables)
ta = pd.read_csv(tablescsv,error_bad_lines=False)
data_ta = pd.DataFrame(data=ta)
data_ta.head()
data_ta.describe()
data_ta.count()
#data_TA.info
data_ta.info
pd.isnull(data_ta).any()
mask = np.zeros_like(data_ta.corr())

triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True
print(mask)
plt.figure(figsize=(16,10))
sns.heatmap(data_ta.corr(), mask=mask, annot = True, annot_kws = {"size": 14})
plt.xticks(size=14)
plt.yticks(size=14)
sns.set_style('white')
plt.show()
data_ta.head()
cols_to_drop   = ['reference','3day', '7day', '28day']
cols_to_encode = ['alkaline ratio','molarity', 'timecure', 'temp cure']
import seaborn as sns

sns.countplot(data_ta['alkaline ratio'].values)
plt.xlabel('alkaline ratio')
plt.ylabel('count')
plt.show()
sns.countplot(data_ta['molarity'].values)
plt.xlabel('molarity')
plt.ylabel('count')
plt.show()
sns.countplot(data_ta['temp cure'].values)
plt.xlabel('temp cure')
plt.ylabel('count')
plt.show()
sns.countplot(data_ta['timecure'].values)
plt.xlabel('timecure')
plt.ylabel('count')
plt.show()
categorical_data=data_ta[cols_to_encode]

data=data_ta.drop(cols_to_encode+cols_to_drop,axis=1)
data.head()
for i in data.columns:
    sns.scatterplot(x=data[i], y=data_ta['28day'])
    plt.xlabel(i)
    plt.ylabel('target')
    plt.show()
sns.pairplot(data)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

cate_data =categorical_data.apply(le.fit_transform)
cate_data.head()
enc = preprocessing.OneHotEncoder()

enc.fit(cate_data)

cate_onehot = enc.transform(cate_data).toarray()
cate_onehot.shape
cate_preprocess=pd.DataFrame(cate_onehot)
cate_preprocess.head()
preprocess_data = preprocessing.StandardScaler().fit_transform(data)
final_data=np.hstack((cate_onehot,preprocess_data))
final_data.shape
x = final_data
y = data_ta['28day']
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2,random_state = 1)
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import AdaBoostRegressor

final_model=final_layer = StackingRegressor(
     estimators=[('etr', ExtraTreesRegressor(random_state=1)),
                 ('rfr',  RandomForestRegressor(random_state=1)),
                 ('gbr', GradientBoostingRegressor(random_state = 1)),
                 ('abr',  AdaBoostRegressor(random_state = 1))],
     final_estimator=AdaBoostRegressor())
final_model.fit(X_train, y_train)
print('R2 score: {:.2f}'.format(final_model.score(X_test, y_test)))
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
s=np.linspace(1,400,400,dtype=int)
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm 

mse_train_loss=[]
mae_train_loss=[]
r2s_train_loss=[]
me_train_loss=[]
mse_test_loss=[]
mae_test_loss=[]
r2s_test_loss=[]
me_test_loss=[]
for i in tqdm(range(1,400)):
    rf_reg=RandomForestRegressor(random_state=1,n_estimators=i)
    rf_reg.fit(X_train, y_train)
    y_test_pred=rf_reg.predict(X_test)
    y_train_pred=rf_reg.predict(X_train)
    mse_test_loss.append(mean_squared_error(y_test,y_test_pred))
    mae_test_loss.append(mean_absolute_error(y_test,y_test_pred))
    r2s_test_loss.append(r2_score(y_test,y_test_pred))
    me_test_loss.append(max_error(y_test,y_test_pred))  
    mse_train_loss.append(mean_squared_error(y_train,y_train_pred))
    mae_train_loss.append(mean_absolute_error(y_train,y_train_pred))
    r2s_train_loss.append(r2_score(y_train,y_train_pred))
    me_train_loss.append(max_error(y_train,y_train_pred))  
plt.scatter(range(1,400),mse_train_loss,label='train_mse')
plt.scatter(range(1,400),mse_test_loss,label='test_mse')
plt.xlabel('n_estmators')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.scatter(range(1,400),r2s_train_loss,label='train_r2s')
plt.scatter(range(1,400),r2s_test_loss,label='test_r2s')
plt.xlabel('n_estmators')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.scatter(range(1,400),me_train_loss,label='train_me')
plt.scatter(range(1,400),me_test_loss,label='test_me')
plt.xlabel('n_estmators')
plt.ylabel('loss')
plt.legend()
plt.show()

def plot_regression_results(ax,y_true, y_pred, title, scores):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,edgecolor='none', linewidth=0)
    ax.legend([extra],  loc='upper left')


def plot_regression_results(ax, y_true, y_pred, title):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2,cmap='r')
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')


rf_reg=RandomForestRegressor(random_state=1,n_estimators=30)
rf_reg.fit(X_train, y_train)
y_pred=rf_reg.predict(X_test)
fig, axs = plt.subplots(figsize=(5,6))

plot_regression_results(axs, y_test, y_pred, 'random forest Refressor')