import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression,Ridge, RidgeCV

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb

from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

import seaborn as sns

from scipy import stats

from scipy.stats import norm

import scipy.stats as st



#data import

file_path = '../input/train.csv'

df = pd.read_csv(file_path)

df.head()
#confirm NaN

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/ df.isnull().count()).sort_values(ascending=False)



missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Precent"])

missing_data.head(20)
#Check feartures and drop them (if there are important items then think about that)



df = df.drop((missing_data[missing_data["Total"]>1]).index, 1)

df.head()
#remaining NaN in the Electorical just drop 'cos only 1.



df = df.dropna()

df.shape
#drop outliers

df.sort_values(by = "GrLivArea", ascending = False)[:2]



df = df.drop(df[df['Id'] == 1299].index)

df = df.drop(df[df['Id'] == 524].index)

df.shape
#Check the relations on numerical features

corrmat = df.corr()

f, ax = plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=.8, square=True)
#check infulenced features against SalePrice

# drop similar features ex. GarageArea and GarageCars



k = 10

cols = corrmat.nlargest(k, "SalePrice")["SalePrice"].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square = True, fmt=".2f",

                 annot_kws={"size": 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#going to check relations btw SalePrice and features (numeric features)

sns.set()

cols_1 = ["SalePrice", 'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond','YearBuilt','YearRemodAdd']

sns.pairplot(df[cols_1], height=2.5)

plt.show()
sns.set()

cols_2 = ["SalePrice", 'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF']

sns.pairplot(df[cols_2], height=2.5)

plt.show()
sns.set()

cols_3 = ["SalePrice", 'LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']

sns.pairplot(df[cols_3], height=2.5)

plt.show()
sns.set()

cols_4 = ["SalePrice", 'BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea']

sns.pairplot(df[cols_4], height=2.5)

plt.show()
sns.set()

cols_5 = ["SalePrice", 'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea']

sns.pairplot(df[cols_5], height=2.5)

plt.show()
sns.set()

cols_6 = ["SalePrice", 'MiscVal','MoSold','YrSold']

sns.pairplot(df[cols_6], height=2.5)

plt.show()
categorical_fea = [word for word in df if df[word].dtype == "object"]

categorical_fea
#data setting

y = df.loc[:,["SalePrice"]]



X = pd.DataFrame(df[categorical_fea])
#onehot_encoding

X_ohe = pd.get_dummies(X, dummy_na=True, columns=categorical_fea)

X_ohe.shape

X_ohe.head()
X_ohe_columns = X_ohe.columns.values
#set to confirm feature importances 

def get_feature_importances(pipeline, columns):



    imp = pipeline.named_steps['est'].feature_importances_

    return pd.DataFrame([imp], columns=columns)



def get_top_N_features(pipeline, columns, N):

    

    df_imp=pd.DataFrame(get_feature_importances(pipeline,columns),

                        columns=columns).T.rank(method='min',ascending=False)

    

    return df_imp.where(df_imp[0]<=N).reset_index().dropna().sort_values(by=0)
from sklearn.ensemble.partial_dependence import plot_partial_dependence

import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (20,20)



pipe_gbr = Pipeline([('scl',StandardScaler()),

                     ('est',GradientBoostingRegressor(random_state=1))])

pipe_gbr.fit(X_ohe,y)



#Fearture importances

df_imp = get_feature_importances(pipe_gbr,

                                 X_ohe_columns)

df_imp.T.plot(kind='bar',

              figsize=(17,4),

              legend=False)

display(df_imp.T)





# PDP



est = pipe_gbr.named_steps['est']

scl = pipe_gbr.named_steps['scl']

N=15

df_top_N = get_top_N_features(pipe_gbr,

                              X_ohe_columns,

                              N)

fig,axs= plot_partial_dependence(est,

                                 scl.transform(X_ohe),

                                 features=df_top_N.index.tolist(),

                                 feature_names=X_ohe_columns)

plt.show()



#select meaningfull features
