import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt 

import matplotlib.gridspec as gridspec

import seaborn as sns 

import math 

import re

from IPython.display import display

from PIL import Image





from scipy import stats

from scipy.stats import norm,skew

import folium 



sns.set_style("whitegrid")

%matplotlib inline
path=('../input/images/House.jpg')

display(Image.open(path))
path=('../input/images-2/data.png')

display(Image.open(path))
df = pd.read_csv("../input/california-housing-prices/housing.csv")

df.tail(10)
fig,(ax1) = plt.subplots(1, figsize=(10,5))



sns.heatmap(df.isnull(), yticklabels = False , cmap = 'plasma', ax = ax1).set_title("Missing Values")

print("Mssing Values")
# --->  Finding Missing Values 



Missing_values=df[df.isna().any(axis=1)].sort_values(by='total_rooms')['total_rooms'].values

                                                                                         

#--> iterating to get the mean Values 



TB = [] #< -- Here total Bedroom

MV = [] #< -- Here Mean values 



for i in Missing_values:

    values = df[df['total_rooms'] == i]['total_bedrooms'].mean()

    values= round(values,1)

    TB.append(i)

    MV.append(values)

    

#--> Creating Dicctionaty to Group the final Values



Key = TB

VAL = MV

dic = dict(zip(Key,VAL)) # In this dictionaty we have Nan Values 



#--> Eliminating Nan Values from Dicctionaty

new_dic = {k : v for k,v in dic.items() if pd.Series(v).notna().all()}

T_nan_values =len(dic)-len(new_dic)



# Total Nan Values 

print ("Total Nan Values in dict =",T_nan_values)
#--> Replacing Values 



for i, j in new_dic.items():

    df.loc[(df['total_rooms'] == i) & (df['total_bedrooms']!= i), 'total_bedrooms'] = j 

    #find Values in Total roms that = i and total bedrooms == nan and repace them by J.value

    

df[df.isnull().any(axis = 1)] # Excatly the 15 Nan Values 
from sklearn.preprocessing import OneHotEncoder



ohc= OneHotEncoder()

ohe=ohc.fit_transform(df.ocean_proximity.values.reshape(-1,1)).toarray()

dfOneHot = pd.DataFrame(ohe ,columns=["Ocean_"+str(ohc.categories_[0][i])

                                     for i in range(len(ohc.categories_[0]))])



data =pd.concat([df,dfOneHot],axis=1)





data.tail(3)
#Creating Map 

USA = folium.Map(location = [37.880,-122.230],tiles='OpenStreetMap',

                   min_zoom = 6 , max_zoom = 13 , zoom_start = 7)



# Adding Position 

for (index,row) in data[0:5000].iterrows():

    folium.Circle(

        radius = int(row.loc['median_house_value'])/10000,

        location = [row.loc['latitude'], row.loc['longitude']],

        popup = 'House Age ' + str(row.loc['housing_median_age']), color = 'crimson',

        tooltip =  '<li><bold>Price :' + str(row.loc['median_house_value']) + str('K'),

        fill = True, fill_color ='#ccfa00').add_to(USA) 

    

display(USA)
# Correlation 

correlation = data.corr()

f,ax =plt.subplots(figsize =(15,10))

mask = np.triu(correlation)

sns.heatmap(correlation, annot=True, mask=mask , ax=ax, 

            linewidths = 4, cmap = 'viridis', square=True).set_title("Correlation")

bottom,top = ax.get_ylim()

ax.set_ylim (bottom + 0.5 , top - 0.5)

print("Heatmap - Correlation")
def mul_plot (df, feature):

    fig=plt.figure(constrained_layout = True , figsize = (12,8))

    grid= gridspec.GridSpec(ncols = 3 , nrows = 2 , figure=fig)



    ax1= fig.add_subplot(grid[0,1:3])

    ax1.set_title("Histogram")

    sns.distplot(df.loc[:,feature], norm_hist = True, ax= ax1)



    ax2= fig.add_subplot(grid[1,1:3])

    ax2.set_title("QQ_plot")

    stats.probplot(df.loc[:,feature] , plot=ax2)



    ax3= fig.add_subplot(grid[:2,0])

    ax3.set_title("Box Plot")

    sns.boxplot(df.loc[:,feature], orient = "v" , ax= ax3)

    

    print("Skewness: "+ str(df['median_house_value'].skew().round(3))) 

    print("Kurtosis: " + str(df['median_house_value'].kurt().round(3)))



mul_plot (data,'median_house_value')
from sklearn.neighbors import LocalOutlierFactor



def outliers (x,y, top = 5 , plot = True):

    lof = LocalOutlierFactor(n_neighbors=40, contamination=0.1)

    x_ =np.array(x).reshape(-1,1)

    preds = lof.fit_predict(x_)

    lof_scr = lof.negative_outlier_factor_

    out_idx = pd.Series(lof_scr).sort_values()[:top].index

    

    if plot:

        f, ax = plt.subplots(figsize=(9, 6))

        plt.scatter(x=x, y=y, c=np.exp(lof_scr), cmap='RdBu')

    return out_idx



outs = outliers(data['median_house_value'], data['median_income'],top=5)

print("Outliers detected:",outs)

plt.show()
# Dropping Values

data.dropna(axis = 0 , inplace=True)

data.drop(['longitude','latitude','ocean_proximity'], axis = 1,   inplace=True)

data.drop(data[data['median_house_value'] > 500000].index, inplace = True)

data.tail(5)
''' Normalizing '''



#--- Appliying Log10  = np.log1p()

data['median_house_value'] = np.log1p(data['median_house_value'])



#Creating new plot 

mul_plot (data,'median_house_value')
path=('../input/images/machine.gif')

display(Image.open(path))
# Applying Machine Learning 

from sklearn import preprocessing 

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV

from sklearn.metrics import mean_squared_error, mean_absolute_error



#Defining Target Values 

y = data['median_house_value'] 

X = data.drop('median_house_value', axis = 1)



#Splitting Data 

y = y

X = X.values



#MinMaxScaler 

MX = MinMaxScaler()

X = MX.fit_transform(X)



#Splitting Train 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
#Applying Machine Learning 

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score



    

# Model initilization



Lnr = LinearRegression()

SVR_rbf = SVR()

DT = DecisionTreeRegressor()

RDF = RandomForestRegressor()

XR = XGBRegressor()



# Model 



model_list = [Lnr,DT,RDF,XR,SVR_rbf]



final_score = []



#Chossong the best model 



def Evaluating (model,X,Y, CV ):

    score = cross_val_score (model,X, Y, cv=CV ,scoring='neg_mean_squared_error')

    

    final = -score.mean()

    final = np.sqrt(final)

    final_score.append(final) #Adding the Final Score the final List
#Running models



for i in model_list:

    Evaluating(i,X,y,4)

    

#Best_model

print(final_score)

    

Best_model = pd.DataFrame(final_score,

                          index =['Lnr','DT','RDF','XR','SVR'], 

                          columns =['Error'])

Best_model.sort_values(by = 'Error' , ascending=True)

# Tunnig XGBRegressor and SVR



param_grid = {'C' :[0.1,1,10,100,1000], 'gamma' : [1,0.1,0.01,0.001,0.0001]}

grid = GridSearchCV(SVR_rbf , param_grid , verbose = 3)

#grid.fit(X_train,y_train)  # if you download the code please remove the # to find the best Parameter but it will take arround 20 min to finish the computation

#grid.best_parameter_





#Hyper Parameter Optimization

n_estimators = [100,500,900,1100,1500]

max_depth = [2,3,5,10,15]

booster = ['gbtree', 'gblinear']

learning_rate = [0.05,0.1,0.15,0.20]

min_child_weight = [1,2,3,4]

base_score = [0.25,0.5,0.75,1]



#Define the grid of Hyperparameters to search

hyperparameter_grid = { 'n_estimators': n_estimators,'max_depth': max_depth,'booster': booster,

                       'learning_rate': learning_rate,'min_child_weight': min_child_weight,

                       'base_score' : base_score}

random_cv = RandomizedSearchCV(estimator = XR,

                              param_distributions=hyperparameter_grid,

                              cv=5,n_iter = 50,

                              scoring = 'neg_mean_absolute_error',n_jobs = 3,

                              verbose = 5,

                              return_train_score=True,

                              random_state=42)



#random_cv.fit(X_train,y_train)  # if you download the code please remove the # to find the best Parameter but it will take arround 20 min to finish the computation

#random_cv.best_parameter_
# Model initilization After Tunning



Lnr = LinearRegression()

SVR_rbf = SVR(C=100, gamma=1)

DT = DecisionTreeRegressor()

RDF = RandomForestRegressor()

XR = XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints='',

             learning_rate=0.1, max_delta_step=0, max_depth=10,

             min_child_weight=3, missing=None, monotone_constraints='()',

             n_estimators=100, n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',

             validate_parameters=1, verbosity=None)



# Model 



model_list = [Lnr,DT,RDF,XR,SVR_rbf]



#Running Models



FS = []



def running_model (model, X_train,y_train,X_test,y_test):

    Algo = model.fit(X_train,y_train)

    Algo_pred = Algo.predict(X_test)

    

    MER = mean_squared_error(y_test,Algo_pred)

    FS.append(MER)



for i in model_list:

    running_model(i,X_train,y_train,X_test,y_test)

BM= pd.DataFrame(FS, index =['Lnr','DT','RDF','XR','SVR'], columns =['Error'])

BM.sort_values(by = 'Error' , ascending=True)