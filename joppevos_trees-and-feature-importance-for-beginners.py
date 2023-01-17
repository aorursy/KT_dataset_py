!pip install fastai==0.7.0
from fastai.imports import *

from fastai.structured import *



from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display

from sklearn.model_selection import train_test_split



from sklearn import metrics

import seaborn as sns

sns.set()



from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)
df_raw = pd.read_csv('../input/train.csv')
df_raw.head()
# check for missing values

null = df_raw.isnull().sum().sort_values(ascending=False)[:15]

pd.DataFrame(data=null, columns=['Missing'])
df_raw.SalePrice = np.log(df_raw.SalePrice)
corrmat = df_raw.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True);
train_cats(df_raw)
df, y, nas = proc_df(df_raw, 'SalePrice')
def split_vals(a,n): return a[:n], a[n:]

n_valid = 300

n_trn = len(df)-n_valid

X_train, X_valid = split_vals(df, n_trn)

y_train, y_valid = split_vals(y, n_trn)
def print_score(m):

    res = [m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
m = RandomForestRegressor(n_jobs=-1, n_estimators=40, oob_score=True)

m.fit(X_train, y_train);

print_score(m)
# get some insight in feature importance really important.

# todo, increase in size. change color palette

fi = pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}

                       ).sort_values('imp', ascending=False)[:20]



sns.set(rc={'figure.figsize':(10,9)})

sns.barplot(data=fi, y='cols', x='imp', palette="GnBu_d");
sns.set(rc={'figure.figsize':(5,5)})

sns.kdeplot(df_raw['OverallQual'] );
# take the whole df and copy it 

df_temp = df.copy()

# make an new column on the copy dataframe [is_valid]

df_temp['is_valid'] = 1

# every row in the validation set we will give a zero

df_temp.is_valid[:n_trn] = 0

# we now have a dataset with at target variable if the row is training or a validation set

x, y, nas = proc_df(df_temp, 'is_valid')
m = RandomForestClassifier(n_estimators=40, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(x, y);

m.oob_score_
def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}

                       ).sort_values('imp', ascending=False)

rf_feat_importance(m, x)[:5]
df.drop('Id', axis=1, inplace=True)
X_train, X_valid = split_vals(df, n_trn)

m = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)