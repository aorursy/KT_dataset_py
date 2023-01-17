# normal libraries

import pandas as pd

import numpy as np



# importing graph libraries 

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

from matplotlib import pyplot as plt

import plotly.express as px

import plotly.graph_objs as go

from pylab import rcParams

rcParams['figure.figsize'] = 14, 8



# With Gridspec you can make static dashboards

from matplotlib.gridspec import GridSpec





# Machine learning libraries

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from scipy.stats import norm

from scipy.stats import probplot

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.linear_model import Ridge, Lasso



# warning libraries

import warnings

warnings.filterwarnings("ignore")





# Maps

import folium



# Very powerfull plugin for maps

from folium.plugins import FastMarkerCluster, HeatMap





# Deep leerning imports

import tensorflow as tf

from tensorflow.keras.models import Sequential # for creating the model

from tensorflow.keras.layers import Dense # for creating the layers





#  Metrics

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score



from sklearn.feature_selection import RFE

from sklearn.decomposition import PCA



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv(r"../input/house-prices-advanced-regression-techniques/train.csv", index_col=["Id"])

shape = train.shape

print(f"there are {shape[0]} rows and {shape[1]} columns in the train-dataframe")

Sum = train.isnull().sum()

Percentage = (train.isnull().sum()/train.isnull().count())

df_null= pd.concat([Sum,Percentage], axis =1 , keys = ["Sum","Percentage"])

df_null=df_null.sort_values(by = "Sum",ascending=False)

df_null= df_null.style.background_gradient("Reds")

df_null
def devide(train):

    numbers = train.select_dtypes(include= "number")

    objects = train.select_dtypes(include= "object")

    return numbers, objects

numbers, objects = devide(train)
numbers.isnull().sum().sort_values(ascending=False)
numbers["LotFrontage"]= numbers["LotFrontage"].fillna(numbers["LotFrontage"].mean())
numbers["MasVnrArea"]= numbers["MasVnrArea"].fillna(0)
numbers["GarageYrBlt"] = numbers["GarageYrBlt"].dropna()
plt.figure(figsize=(25,25))

drop = np.zeros_like(train.corr())

drop[np.triu_indices_from(drop)] = True

sns.heatmap(train.corr(), annot = True, fmt = ".1f", cmap = "Blues", linewidth = 1, mask = drop);



plt.title("Correlation");

sns.set_style("white")
objects.isnull().sum().sort_values(ascending=False)
for x in ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu","GarageType","GarageCond","GarageQual","GarageFinish"]:

    objects[x] = objects[x].fillna("None")

    
objects = objects.dropna()

All_objects = pd.DataFrame({})



for y in objects:

    var = pd.get_dummies(objects[y], drop_first=True, prefix=y+"_")

    objects = pd.concat([objects, var], axis=1)

    objects.drop(y, axis=1, inplace=True)

    

    

    
train = pd.concat([numbers,objects], axis = 1)
train.isnull().sum().sort_values(ascending=False)
# after cleaning 

after_clean = train.shape

print(f" After cleaning there are {after_clean[0]} rows and {after_clean[1]} columns in this dataframe")

train = train.dropna()
scores = pd.DataFrame({"Model":[],

                       "Cross_vall_score":[], 

                       "Mean_squared_error":[],

                       "R2":[]})


sns.set_style("dark")





fig = plt.figure(constrained_layout= True);



gs = GridSpec(3,3, fig)



ax = fig.add_subplot(gs[0,:])

ax2 = fig.add_subplot(gs[1,:])

ax3 = fig.add_subplot(gs[2,:])





sumis = round(train["SalePrice"].mean(),2)



skew = round(train["SalePrice"].skew(),2)

kurt = round(train["SalePrice"].kurt(),2)





ax.set_title("Price averange", fontsize=40, pad = 10, color='dimgrey' )

ax.text(0.25, 0.43, f' $ {sumis}', fontsize=30, color='mediumseagreen', ha='center',        

        bbox=dict(facecolor='navy', alpha=0.1, pad=10, boxstyle='round, pad=.7'))

ax.text(0.25, 0.81, 'Total sum of price',color='darkslateblue', fontsize=20, ha='center')





ax.text(0.50, 0.43, f'{skew}', fontsize=40, color='mediumseagreen', ha='center',

          bbox=dict(facecolor='navy', alpha=0.1, pad=10, boxstyle='round, pad=.4'))



ax.text(0.50, 0.81, 'skew:',color='darkslateblue', fontsize=20, ha='center')





ax.text(0.75, 0.43, f'{kurt}', fontsize=40, color='mediumseagreen', ha='center',

          bbox=dict(facecolor='navy', alpha=0.1, pad=1, boxstyle='round, pad=.4'))

ax.text(0.75, 0.81, 'kurtosis:',color='darkslateblue', fontsize=20, ha='center')



ax2.set_title("Boxplot of price", fontsize=20, pad = 10, color='dimgrey' )



sns.boxplot(train["GrLivArea"],ax = ax3)

sns.boxplot(train["SalePrice"],ax = ax2)





ax.axis("off");
log = np.log1p(train["SalePrice"])





fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1);



sns.distplot(train["SalePrice"], fit = norm, ax = ax1);

sns.distplot(log, fit = norm, ax = ax2);

probplot(train["SalePrice"], plot=ax3);

probplot(log, plot=ax4);
train.isnull().sum().sort_values(ascending=False)
train["SalePrice"] = np.log1p(train["SalePrice"])
X = train.drop(["SalePrice"], axis = 1)

y = train["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state = 42)
lin = LinearRegression()

lin.fit(X_train,y_train)

predict = lin.predict(X_test)

score_bass = round(np.sqrt(mean_squared_error(predict,y_test)),2)

base_r2=round(r2_score(predict,y_test),2)

Base_cross = round(cross_val_score(lin,X_train,y_train,cv=5).mean(),2)

scores.loc[0] = ["Simple_linear", Base_cross,score_bass,base_r2]

print(scores)

las = Lasso(alpha=0.01)

las.fit(X_train,y_train)

predict3 = las.predict(X_test)

cross_vall_lass1 = round(cross_val_score(las,X_train,y_train,cv=5).mean(),2)

score_lass1 = round(np.sqrt(mean_squared_error(predict3,y_test)),2)

r2_score_lass1 = round(r2_score(predict3,y_test),2)



n = scores.shape[0]

scores.loc[n] = ["Lasso_Regression1",cross_vall_lass1,score_lass1,r2_score_lass1]

print(scores)
las = Lasso(alpha=0.5)

las.fit(X_train,y_train)

predict3 = las.predict(X_test)

cross_vall_lass2 = round(cross_val_score(las,X_train,y_train,cv=5).mean(),2)

score_lass2 = round(np.sqrt(mean_squared_error(predict3,y_test)),2)

r2_score_lass2 = round(r2_score(predict3,y_test),2)



n = scores.shape[0]

scores.loc[n] = ["Lasso_Regression2",cross_vall_lass2,score_lass2,r2_score_lass2]

print(scores)
las = Lasso(alpha=0.0001)

las.fit(X_train,y_train)

predict3 = las.predict(X_test)

cross_vall_lass3 = round(cross_val_score(las,X_train,y_train,cv=5).mean(),2)

score_lass3 = round(np.sqrt(mean_squared_error(predict3,y_test)),2)

r2_score_lass3 = round(r2_score(predict3,y_test),2)



n = scores.shape[0]

scores.loc[n] = ["Lasso_Regression3",cross_vall_lass3,score_lass3,r2_score_lass3]

print(scores)
rid = Ridge(alpha=0.001)

rid.fit(X_train,y_train)

predict4 = las.predict(X_test)

cross_vall_rid1 = round(cross_val_score(las,X_train,y_train,cv=5).mean(),2)

score_rid1 = round(np.sqrt(mean_squared_error(predict4,y_test)),2)

r2_score_rid1 = round(r2_score(predict4,y_test),2)



n = scores.shape[0]

scores.loc[n] = ["Ridge1",cross_vall_rid1,score_rid1,r2_score_rid1]

print(scores)

tree = DecisionTreeRegressor()

tree.fit(X_train,y_train)

predict5 = tree.predict(X_test)

cross_vall_tree = round(cross_val_score(tree,X_train,y_train,cv=5).mean(),2)

score_tree = round(np.sqrt(mean_squared_error(predict5,y_test)),2)

r2_score_tree = round(r2_score(predict5,y_test),2)



n = scores.shape[0]

scores.loc[n] = ["Tree",cross_vall_tree,score_tree,r2_score_tree]

print(scores)
forest = RandomForestRegressor(n_estimators=100)

forest.fit(X_train,y_train)

predict6 = forest.predict(X_test)

cross_vall_forest = round(cross_val_score(forest,X_train,y_train,cv=5).mean(),2)

score_forest = round(np.sqrt(mean_squared_error(predict6,y_test)),2)

r2_score_forest = round(r2_score(predict6,y_test),2)



n = scores.shape[0]

scores.loc[n] = ["forest",cross_vall_forest,score_forest,r2_score_forest]

print(scores)
xgb = XGBRegressor()

xgb.fit(X_train,y_train)

predict7 = xgb.predict(X_test)

cross_vall_xgb = round(cross_val_score(xgb,X_train,y_train,cv=5).mean(),2)

score_xgb = round(np.sqrt(mean_squared_error(predict7,y_test)),2)

r2_score_xgb = round(r2_score(predict7,y_test),2)



n = scores.shape[0]

scores.loc[n] = ["xgb",cross_vall_xgb,score_xgb,r2_score_xgb]

print(scores)


fig =  go.Figure(go.Bar(x= scores["Model"], y = round(scores["Cross_vall_score"],2),

                       marker={'color': scores['Cross_vall_score'],

                              'colorscale': 'Viridis'},

                        text=round(scores['Cross_vall_score'],2),

                        textposition =  "outside"

                       ))





fig.update_layout(title_text = "Models")