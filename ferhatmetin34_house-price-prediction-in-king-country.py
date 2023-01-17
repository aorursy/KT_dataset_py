# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import pylab

import missingno as msn

import folium

from folium import plugins

import branca.colormap as cm

from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.linear_model import LinearRegression

from sklearn.decomposition import PCA

from sklearn.preprocessing import scale 

from sklearn import model_selection

from sklearn.cross_decomposition import PLSRegression, PLSSVD

from sklearn.linear_model import Ridge

from sklearn.linear_model import RidgeCV

from sklearn.linear_model import Lasso

from sklearn.linear_model import LassoCV

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import ElasticNetCV

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

import statsmodels.api as sm

from sklearn import model_selection

from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score

from lightgbm import LGBMRegressor

from sklearn.experimental import enable_hist_gradient_boosting 

from sklearn.ensemble import HistGradientBoostingRegressor

from catboost import CatBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor
data=pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")

df=data.copy()

df.head()
print("Row : {}  \nColumn: {}".format(df.shape[0],df.shape[1]))
df.columns.values
df.info()
df['date'] = df.date.apply(pd.to_datetime)
df.date.head().to_frame()
df.date.min(),df.date.max()
df["date_year"]=df.date.dt.year

df["date_month"]=df.date.dt.month

#df["day_of_week"]=df.date.dt.dayofweek

#df["is_weekend"]=df.day_of_week.apply(lambda x:1 if x>4 else 0)
df=df.drop("date",axis=1)
df.head()
df[df.duplicated(subset=['id',"price"],keep=False)].sort_values(by="id")
df[(df.price==0) & (df.price<0)]
df.eq(0).sum().to_frame()
plt.figure(figsize=(22,6))



plt.subplot(141)

sns.distplot(df.price);



plt.subplot(142)

stats.probplot(df.price, dist="norm", plot=pylab) ;



plt.subplot(143)

sns.distplot(np.log(df.price),color="magenta");



plt.subplot(144)

stats.probplot(np.log(df.price), dist="norm", plot=pylab) ;
df=df.drop("id",axis=1)

df.head()
m = folium.Map([47 ,-122], zoom_start=5,width="%100",height="%100")

locations = list(zip(df.lat, df.long))

cluster = plugins.MarkerCluster(locations=locations,popups=df["price"].tolist())

m.add_child(cluster)

m
m = folium.Map(location=[47,-122],width="%100",height="%100")

for i in range(len(locations)):

    folium.CircleMarker(location=locations[i],radius=1).add_to(m)

m
price=df[["lat","long","price"]]

min_price=df["price"].min()

max_price=df["price"].max()

min_price,max_price
pd.options.display.float_format = '{:.4f}'.format

df.price.describe().to_frame()
m = folium.Map(location=[47,-122],width="%100",height="%100")

colormap = cm.StepColormap(colors=['green','yellow','orange','red'] ,index=[min_price,321950,450000,645000,max_price],vmin= min_price,vmax=max_price)

for loc, p in zip(zip(price["lat"],price["long"]),price["price"]):

    folium.Circle(

        location=loc,

        radius=2,

        fill=True,

        color=colormap(p)).add_to(m)

m
plt.figure(figsize=(20,8))

corr=df.corr().abs()

mask=np.zeros_like(corr,dtype=np.bool)

mask[np.triu_indices_from(mask)]=True

sns.heatmap(corr.abs(),annot=True,cmap="coolwarm",mask=mask);
pairplot=sns.pairplot(df[["price","sqft_living","sqft_above","grade","sqft_living15","bathrooms","bedrooms"]],kind="reg",corner=True,diag_kind="kde");



def corrfunc(x,y, ax=None, **kws):

    r, _ = pearsonr(x, y)

    ax = ax or plt.gca()

    rho = '\u03C1'

    ax.annotate(f'{rho} = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)

pairplot.map_lower(corrfunc)

plt.show()
df.columns
to_plot_list=["bedrooms","bathrooms","floors","waterfront","view","condition","grade"]

  



for i in to_plot_list:

    plt.figure(figsize=(10,5))

    df.groupby(i)["price"].mean().plot.bar(color="orangered");

    plt.legend()

    plt.title(i + " BarPlot")

    plt.show()
df.corr().abs()["price"].nlargest(15)
to_plot_list=["bedrooms","bathrooms","floors","waterfront","view","condition","grade"]

  



for i in to_plot_list:

    plt.figure(figsize=(10,5))

    sns.countplot(x=i,data=df);

    plt.legend()

    plt.title(i + " CountPlot")

    plt.show()
to_plot_list=["bedrooms","bathrooms","floors","waterfront","view","condition","grade"]

  



for i in to_plot_list:

    plt.figure(figsize=(10,5))

    sns.boxplot(x=i,y="price",data=df);

    plt.legend()

    plt.title(i + " BoxPlot")

    plt.show()
to_plot_list=["sqft_living","sqft_living15","sqft_lot15"]

  



for i in to_plot_list:

    plt.figure(figsize=(10,5));

    sns.jointplot(x=i,y="price",data=df,color="darkred");

    plt.legend();

    plt.show();
df[df["bedrooms"]>10]
df=df.drop(df[df["bedrooms"]>10].index)
plt.figure(figsize=(12,6))

sns.boxplot(x="bedrooms",y="price",data=df);
df[df.sqft_lot15==df.sqft_lot].sample(10)
df[df["bathrooms"]>=7][["price","bathrooms","bedrooms","sqft_living"]]
plt.figure(figsize=(12,6))

sns.boxplot(x="bathrooms",y="price",data=df);
df["bathrooms"] = df['bathrooms'].round(0).astype(int)  #float bathroom?????
plt.figure(figsize=(15,6))

df.groupby("yr_built")["price"].mean().nlargest(30).plot.bar(color="darkblue");
sns.catplot(x="yr_built", y = "price", data=df,size= 7, aspect = 3, kind="box" );

plt.xticks(rotation=90);


sns.catplot(x="yr_built", y = "price", data=df[df.price<1000000],size= 7, aspect = 3, kind="box" );

plt.xticks(rotation=90);
sns.catplot(x="date_year", y = "price", data=df,size= 4, aspect = 2, kind="box" );

plt.xticks(rotation=90);
sns.catplot(x="date_month", y = "price", data=df,size= 5, aspect = 3, kind="box" );

plt.xticks(rotation=90);
plt.figure(figsize=(12,5))

plt.subplot(121)

sns.boxplot(df[df["yr_renovated"]>0]["price"]);

plt.title("renovated")

plt.subplot(122)

sns.boxplot(df[df["yr_renovated"]==0]["price"]);

plt.title("not-renovated");
df[(df["date_year"]-df["yr_built"])<0][["date_year","yr_built"]] # they could be sold before being build or there is a mistake,I dont know
df=df.drop(df[(df["date_year"]-df["yr_built"])<0].index)
df_new=df.copy()
df_new["is_renovated"]=df_new["yr_renovated"].apply( lambda x:1 if x>0 else 0)
df_new.head()
df.columns
df[["sqft_living","sqft_lot","sqft_above","sqft_basement","sqft_living15","sqft_lot15"]].sample(7)

# sqft_living - sqft_above=sqft_basement

df[df["sqft_living"]==df["sqft_above"]][["sqft_living","sqft_above","sqft_lot","sqft_basement","sqft_living15","sqft_lot15"]]
df_new["total_room"]=df_new["bedrooms"]+df_new["bathrooms"]
#df_new["sqft_per_room"]=pd.Series(df_new["sqft_living"]/df_new["total_room"],index=df_new.index)
df_new.corr()["price"].nlargest(15)
df_new.head()
X=df_new.drop(["lat","long","price","is_renovated","sqft_above","sqft_basement","bedrooms","bathrooms"],axis=1)



y=np.log(df_new["price"])

#y=df_new["price"]
X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y, 

                                                    test_size=0.30, 

                                                    random_state=42)
def adjustedR2(r2,n,k):

    return r2-(k-1)/(n-k)*(1-r2)
gbm_model = GradientBoostingRegressor(max_depth=7,random_state=42)

gbm_model.fit(X_train, y_train)

y_pred = gbm_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
r2_score(y_test,y_pred)
y_tr_pred=gbm_model.predict(X_train)
r2_score(y_train,y_tr_pred)
adjustedR2(r2_score(y_test,y_pred),df_new.shape[0],df_new.shape[1])
plt.figure(figsize=(12,6))

ax1=sns.distplot(y_test,hist=False);

sns.distplot(y_pred,ax=ax1,hist=False);
Importance = pd.DataFrame({"Importance": gbm_model.feature_importances_*100},

                         index = X_train.columns)

Importance.sort_values(by = "Importance", 

                       axis = 0, 

                       ascending = True).plot(kind ="barh", color = "r")



plt.xlabel("İmportance LEevels")
DM_train = xgb.DMatrix(data = X_train, label = y_train)

DM_test = xgb.DMatrix(data = X_test, label = y_test)

xgb_model = XGBRegressor(max_depth=4,random_state=42).fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
y_tr_pred=xgb_model.predict(X_train)
r2_score(y_train,y_tr_pred)
adjustedR2(r2_score(y_test,y_pred),df_new.shape[0],df_new.shape[1])
plt.figure(figsize=(12,6))

ax1=sns.distplot(y_test,hist=False);

sns.distplot(y_pred,ax=ax1,hist=False);
Importance = pd.DataFrame({"Importance": xgb_model.feature_importances_*100},

                         index = X_train.columns)

Importance.sort_values(by = "Importance", 

                       axis = 0, 

                       ascending = True).plot(kind ="barh", color = "r")



plt.xlabel("İmportance LEevels")
lgbm = LGBMRegressor()

lgbm_model = lgbm.fit(X_train, y_train)

y_pred = lgbm_model.predict(X_test, 

                            num_iteration = lgbm_model.best_iteration_)

np.sqrt(mean_squared_error(y_test, y_pred))
y_tr_pred=xgb_model.predict(X_train)
r2_score(y_train,y_tr_pred)
r2_score(y_test,y_pred)
adjustedR2(r2_score(y_test,y_pred),df_new.shape[0],df_new.shape[1])
plt.figure(figsize=(12,6))

ax1=sns.distplot(y_test,hist=False);

sns.distplot(y_pred,ax=ax1,hist=False);
Importance = pd.DataFrame({"Importance":lgbm_model.feature_importances_*100},

                         index = X_train.columns)

Importance.sort_values(by = "Importance", 

                       axis = 0, 

                       ascending = True).plot(kind ="barh", color = "r")



plt.xlabel("İmportance LEevels")
est = HistGradientBoostingRegressor(random_state=42)

hist_model=est.fit(X_train, y_train)

y_pred=hist_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
y_tr_pred=xgb_model.predict(X_train)

r2_score(y_train,y_tr_pred)
r2_score(y_test,y_pred)
adjustedR2(r2_score(y_test,y_pred),df_new.shape[0],df_new.shape[1])
plt.figure(figsize=(12,6))

ax1=sns.distplot(y_test,hist=False);

sns.distplot(y_pred,ax=ax1,hist=False);
catb = CatBoostRegressor(random_state=42)

catb_model = catb.fit(X_train, y_train)

y_pred = catb_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
y_tr_pred=xgb_model.predict(X_train)

r2_score(y_train,y_tr_pred)
r2_score(y_test,y_pred)
adjustedR2(r2_score(y_test,y_pred),df_new.shape[0],df_new.shape[1])
plt.figure(figsize=(12,6))

ax1=sns.distplot(y_test,hist=False);

sns.distplot(y_pred,ax=ax1,hist=False);