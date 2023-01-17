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



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# LOADING MAIN LIBRARIES



import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns
plt.rcParams['figure.figsize'] = (15, 15)

%matplotlib inline
!pip install chart-studio
# VISUALIZATION LIBRARIES



import chart_studio.plotly as py 

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# LOADING THE DATA



df_2015 = pd.read_csv('../input/world-happiness/2015.csv')

df_2016 = pd.read_csv('../input/world-happiness/2016.csv')

df_2017 = pd.read_csv('../input/world-happiness/2017.csv')

df_2018 = pd.read_csv('../input/world-happiness/2018.csv')

df_2019 = pd.read_csv('../input/world-happiness/2019.csv')
df_2019.info()
df_2019.columns = ["rank","region","score", "gdp","support","life_exp","freedom","generosity","corruption"]
df_2018.info()
df_2018.columns = ["rank","region","score", "gdp","support","life_exp","freedom","generosity","corruption"]
df_2017.info()
df_2017.drop(["Whisker.high","Whisker.low","Dystopia.Residual"],axis=1,inplace=True)

df_2017.columns =  ["region","rank","score","gdp","support", "life_exp","freedom","generosity","corruption"]
df_2016.info()
df_2016.drop(['Region','Lower Confidence Interval','Upper Confidence Interval','Dystopia Residual'],axis=1,inplace=True)

df_2016.columns = ["region","rank","score","gdp","support","life_exp","freedom","generosity","corruption"]
df_2015.info()
df_2015.drop(["Region",'Standard Error', 'Dystopia Residual'],axis=1,inplace=True)

df_2015.columns = ["region", "rank", "score", "gdp", "support","life_exp", "freedom", "corruption", "generosity"]
# ADDING COLUMN FOR YEAR



df_2015["year"] = 2015

df_2016["year"] = 2016

df_2017["year"] = 2017

df_2018["year"] = 2018

df_2019["year"] = 2019
quarter = ['Top','Top-Mid', 'Low-Mid', 'Low' ]

quarter_n = [4, 3, 2, 1]





df_2015["quarter"] = pd.qcut(df_2015['rank'], len(quarter), labels=quarter)

df_2015["quarter_n"] = pd.qcut(df_2015['rank'], len(quarter), labels=quarter_n)



df_2016["quarter"] = pd.qcut(df_2016['rank'], len(quarter), labels=quarter)

df_2016["quarter_n"] = pd.qcut(df_2016['rank'], len(quarter), labels=quarter_n)



df_2017["quarter"] = pd.qcut(df_2017['rank'], len(quarter), labels=quarter)

df_2017["quarter_n"] = pd.qcut(df_2017['rank'], len(quarter), labels=quarter_n)



df_2018["quarter"] = pd.qcut(df_2018['rank'], len(quarter), labels=quarter)

df_2018["quarter_n"] = pd.qcut(df_2018['rank'], len(quarter), labels=quarter_n)



df_2019["quarter"] = pd.qcut(df_2019['rank'], len(quarter), labels=quarter)

df_2019["quarter_n"] = pd.qcut(df_2019['rank'], len(quarter), labels=quarter_n)
# APPENDING ALL TOGUETHER

df = df_2015.copy()

df = df.append([df_2016,df_2017,df_2018,df_2019])
#CHECKING FOR MISSING DATA

df.isnull().any()



#filling missing values of corruption with its mean

df.corruption.fillna((df.corruption.mean()), inplace = True)
df
df.columns
# CHECKING NUMERICAL DATA

df.describe()
# DISTRIBUTION OF ALL NUMERIC DATA

plt.rcParams['figure.figsize'] = (15, 15)

df.hist();
plt.rcParams['figure.figsize'] = (10, 10)

df[['score', 'gdp', 'support', 'life_exp', 'freedom',

       'corruption', 'generosity']].boxplot();
# CHECK FOR TOP COUNTRIES IN EACH FEATURE



fig, axes = plt.subplots(nrows=3, ncols=2,constrained_layout=True,figsize=(10,10));



sns.barplot(x='gdp',y='region',data=df.nlargest(10,'gdp'),ax=axes[0,0],palette="RdYlGn_r")

sns.barplot(x='support' ,y='region',data=df.nlargest(10,'support'),ax=axes[0,1],palette="RdYlGn_r")

sns.barplot(x='life_exp' ,y='region',data=df.nlargest(10,'life_exp'),ax=axes[1,0],palette='RdYlGn_r')

sns.barplot(x='freedom' ,y='region',data=df.nlargest(10,'freedom'),ax=axes[1,1],palette='RdYlGn_r')

sns.barplot(x='generosity' ,y='region',data=df.nlargest(10,'generosity'),ax=axes[2,0],palette='RdYlGn_r')

sns.barplot(x='corruption' ,y='region',data=df.nlargest(10,'corruption'),ax=axes[2,1],palette='RdYlGn_r')
df.loc[df['region']=='Brazil']
# COMPARING BIGGEST ECONOMIES IN THE WORLD



top_econ = ['Brazil','India','China', 'United States', 'Japan', 'Germany', 'United Kingdon', 'France', 'Italy','Canada']



df_top = df[(df['region'].isin(top_econ))].sort_values(['region', 'year'])

df_top.reset_index(drop=True)
# Happiness Rank Across the World in 2019



map_plot = dict(type = 'choropleth', 

           locations = df_2019['region'],

           locationmode = 'country names',

           z = df_2019['rank'], 

           text = df_2019['region'],

          colorscale = 'rdylgn', reversescale = True)

layout = dict(title = 'Happiness Rank Across the World in 2019', 

             geo = dict(showframe = False, 

                       projection = {'type': 'equirectangular'}))

choromap = go.Figure(data = [map_plot], layout=layout)

iplot(choromap)
#Happiness Score Across the World in 2019



map_plot = dict(type = 'choropleth', 

           locations = df_2019['region'],

           locationmode = 'country names',

           z = df_2019['score'], 

           text = df_2019['region'],

          colorscale = 'rdylgn', reversescale = False)

layout = dict(title = 'Happiness Score Across the World in 2019', 

             geo = dict(showframe = False, 

                       projection = {'type': 'equirectangular'}))

choromap = go.Figure(data = [map_plot], layout=layout)

iplot(choromap)
#Happiness GDP Across the World in 2019



map_plot = dict(type = 'choropleth', 

           locations = df_2019['region'],

           locationmode = 'country names',

           z = df_2019['gdp'], 

           text = df_2019['region'],

          colorscale = 'rdylgn', reversescale = False)

layout = dict(title = 'Happiness GDP Across the World in 2019', 

             geo = dict(showframe = False, 

                       projection = {'type': 'equirectangular'}))

choromap = go.Figure(data = [map_plot], layout=layout)

iplot(choromap)
#Perception of Social Support Across the World in 2019



map_plot = dict(type = 'choropleth', 

           locations = df_2019['region'],

           locationmode = 'country names',

           z = df_2019['life_exp'], 

           text = df_2019['region'],

          colorscale = 'rdylgn', reversescale = False)

layout = dict(title = 'Perception of Social Support Across the World in 2019', 

             geo = dict(showframe = False, 

                       projection = {'type': 'equirectangular'}))

choromap = go.Figure(data = [map_plot], layout=layout)

iplot(choromap)
#Perception of Freedom Across the World in 2019



map_plot = dict(type = 'choropleth', 

           locations = df_2019['region'],

           locationmode = 'country names',

           z = df_2019['freedom'], 

           text = df_2019['region'],

          colorscale = 'rdylgn', reversescale = False)

layout = dict(title = 'Perception of Freedom Across the World in 2019', 

             geo = dict(showframe = False, 

                       projection = {'type': 'equirectangular'}))

choromap = go.Figure(data = [map_plot], layout=layout)

iplot(choromap)
#Trust in government Across the World in 2019



map_plot = dict(type = 'choropleth', 

           locations = df_2019['region'],

           locationmode = 'country names',

           z = df_2019['corruption'], 

           text = df_2019['region'],

          colorscale = 'rdylgn', reversescale = False)

layout = dict(title = 'Trust in government Across the World in 2019', 

             geo = dict(showframe = False, 

                       projection = {'type': 'equirectangular'}))

choromap = go.Figure(data = [map_plot], layout=layout)

iplot(choromap)
#Quartile  groups [4:'Top',3:'Top-Mid', 2:'Low-Mid', 1:'Low' ] Across the World in 2019



map_plot = dict(type = 'choropleth', 

           locations = df_2019['region'],

           locationmode = 'country names',

           z = df_2019['quarter_n'], 

           text = df_2019['region'],

          colorscale = 'rdylgn', reversescale = False)

layout = dict(title = 'Quartile % Group  in 2019', 

             geo = dict(showframe = False, 

                       projection = {'type': 'equirectangular'}))

choromap = go.Figure(data = [map_plot], layout=layout)

iplot(choromap)
drop_rank = df.drop(["rank", 'quarter', 'quarter_n'], axis = 1)
plt.rcParams['figure.figsize'] = (10,10)

sns.pairplot(drop_rank, hue = 'year', corner=True);
# LET'S TAKE A SECOND LOOK INTO CORRELATIONS

df_clean = df.drop(["rank", 'quarter', 'quarter_n', 'year'], axis=1)

                   

kendall_corr = df_clean.corr(method='kendall')

kendall_corr
spearman_corr = df_clean.corr(method='spearman')

spearman_corr

pearson_corr = df_clean.corr(method='pearson')

pearson_corr
# VISUALIZE CORRELATIONS WITH HEATMAPS

fig, ax = plt.subplots(ncols=3,figsize=(20,5) )

sns.heatmap(kendall_corr, vmin=-1, vmax=1, ax=ax[0], center=0, cmap="RdBu_r", annot=True);

sns.heatmap(spearman_corr, vmin=-1, vmax=1, ax=ax[1], center=0, cmap="RdBu_r", annot=True);

sns.heatmap(pearson_corr, vmin=-1, vmax=1, ax=ax[2], center=0, cmap="RdBu_r", annot=True);
df_model = df_clean.drop(['region'], axis = 1)

df_model
# TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split



X = df_model.drop('score', axis =1)

y = df_model.score.values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
# MULTIPLE LR

import statsmodels.api as sm



X_sm = X = sm.add_constant(X)

model = sm.OLS(y,X_sm)

model.fit().summary()
from sklearn.linear_model import LinearRegression, Lasso

from sklearn.model_selection import cross_val_score



lm = LinearRegression()

lm.fit(X_train, y_train)



print('Negative MAE: ', np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
print("Estimated Intercept (constant) is", lm.intercept_)

print("The number of coefficients in this model are", lm.coef_)
coef = zip(X.columns, lm.coef_)

coef_df = pd.DataFrame(list(zip(X.columns, lm.coef_)), columns=['features', 'coefficients'])

coef_df
# LASSO REGRESSION

lm_l = Lasso()

lm_l.fit(X_train,y_train)

np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))



alpha = []

error = []



for i in range(1,10):

    alpha.append(i/1000)

    lml = Lasso(alpha=(i/1000))

    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))

    

#plt.plot(alpha,error)



err = tuple(zip(alpha,error))

df_err = pd.DataFrame(err, columns = ['alpha','error'])

df_err[df_err.error == max(df_err.error)]
# RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()



print ('Negative MAE: ', np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3)))

# TUNE MODELS GRIDSEARCHCV

from sklearn.model_selection import GridSearchCV



parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}



gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)

gs.fit(X_train,y_train)



gs.best_score_

gs.best_estimator_


# TEST ENSEMBLES

tpred_lm = lm.predict(X_test)

tpred_lml = lm_l.predict(X_test)

tpred_rf = gs.best_estimator_.predict(X_test)



from sklearn.metrics import mean_absolute_error

print('MAE Linear Regression:          ',mean_absolute_error(y_test,tpred_lm))

print('MAE Multiple Linear Regression: ',mean_absolute_error(y_test,tpred_lml))

print('MAE Random Forest Regression:   ',mean_absolute_error(y_test,tpred_rf))

print('Average MAE LM+RF:              ',mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2))





# PICKLE MODEL

import pickle

pickl = {'model': gs.best_estimator_}

pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )



file_name = "model_file.p"

with open(file_name, 'rb') as pickled:

    data = pickle.load(pickled)

    model = data['model']



model.predict(np.array(list(X_test.iloc[1,:])).reshape(1,-1))[0]





print('Best Estimator: ', gs.best_estimator_)

print('Best score: ', gs.best_score_)



list(X_test.iloc[1,:])


