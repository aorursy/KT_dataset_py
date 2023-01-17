import numpy as np

import pandas as pd

import calendar

from scipy.stats import kendalltau

import seaborn as sns

from sklearn import preprocessing

from yellowbrick.features import PCA

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.model_selection import train_test_split

from yellowbrick.target import FeatureCorrelation

from yellowbrick.regressor import ResidualsPlot

from pdpbox import pdp, info_plots

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 100
df = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

df.head()
df['datetime'] = pd.to_datetime(df['datetime']) 

df['year'] = df['datetime'].dt.year #Extract Year feature

df['month'] = df['datetime'].dt.month #Extract Month feature

df['hour'] = df['datetime'].dt.hour #Extract Hour feature

df.head()
continious_variables = df[['temp', 'atemp', 'humidity', 'windspeed','count']]

continious_variables.columns = ['temp', 'atemp', 'humidity', 'windspeed','number_of_rentals']

sns.pairplot(continious_variables,x_vars=['temp', 'atemp', 'humidity', 'windspeed'],y_vars=['number_of_rentals'],

            height=5, aspect=.8, kind="reg")

plt.show()             
corr = continious_variables.corr() #Check Pearson Correlation

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns,

        annot=True, fmt="f",cmap="Blues")
visualizer = FeatureCorrelation(method='mutual_info-regression', labels=continious_variables.columns)



visualizer.fit(continious_variables.loc[:,continious_variables.columns!='number_of_rentals'], \

               continious_variables.loc[:,continious_variables.columns=='number_of_rentals'], \

               random_state=0)

visualizer.show()
fig, axes, summary_df = info_plots.target_plot(

    df=continious_variables, feature='temp', feature_name='temp', target='number_of_rentals', show_percentile=True

)
summary_df
categorical_data = df[['season', 'holiday', 'workingday', 'weather','month','year','hour','count']]

#Extract the category codes

categorical_data = categorical_data.astype('category')

categorical_data['season'] = categorical_data['season'].cat.codes

categorical_data['holiday'] = categorical_data['holiday'].cat.codes

categorical_data['workingday'] = categorical_data['workingday'].cat.codes

categorical_data['weather'] = categorical_data['weather'].cat.codes
visualizer = FeatureCorrelation(method='mutual_info-regression', labels=categorical_data.columns)



visualizer.fit(categorical_data.loc[:,categorical_data.columns!='count'], \

               categorical_data.loc[:,categorical_data.columns=='count'], \

               random_state=0)

visualizer.show()
sns.lmplot(x="temp", y="count", col="season", data=df,

           aspect=.5);
sns.lmplot(x="temp", y="count", col="holiday", data=df,

           aspect=.5);
#Quantify 'temp' -> 'holiday' relationship

model = smf.ols(formula='temp ~ C(holiday)', data=df)

res = model.fit()

print('ANOVA - temp~holiday')

summary = res.summary()

print(summary.tables[0])



#Quantify 'atemp' -> 'holiday' relationship

model = smf.ols(formula='atemp ~ C(holiday)', data=df)

res = model.fit()

print('ANOVA - temp~holiday')

summary = res.summary()

print(summary.tables[0])
#Quantify 'temp' -> 'month' relationship

model = smf.ols(formula='temp ~ C(month)', data=df)

res = model.fit()

print('ANOVA - temp~month')

summary = res.summary()

print(summary.tables[0])



#Quantify 'atemp' -> 'month' relationship

model = smf.ols(formula='atemp ~ C(month)', data=df)

res = model.fit()

print('ANOVA - atemp~month')

summary = res.summary()

print(summary.tables[0])
#Quantify 'temp' -> 'season' relationship

model = smf.ols(formula='temp ~ C(season)', data=df)

res = model.fit()

print('ANOVA - temp~season')

summary = res.summary()

print(summary.tables[0])



#Quantify 'atemp' -> 'season' relationship

model = smf.ols(formula='atemp ~ C(season)', data=df)

res = model.fit()

print('ANOVA - atemp~season')

summary = res.summary()

print(summary.tables[0])
X,y = df[['year','hour','temp','humidity','windspeed','holiday','workingday']],df['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestRegressor(n_estimators=200)

visualizer = ResidualsPlot(model, qqplot=True)



visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.show()
pdp_temp = pdp.pdp_isolate(

    model=model, dataset=X_train, model_features=X_train.columns, feature='temp'

)

fig, axes = pdp.pdp_plot(pdp_temp, 'temp',center=False, cluster=True,n_cluster_centers=2,\

                         plot_lines=True, x_quantile=True, show_percentile=True, plot_pts_dist=True)
print('Get the PDP values',pdp_temp.pdp)
X,y = df[['year','hour','temp','atemp','humidity','windspeed','holiday','workingday']],df['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestRegressor(n_estimators=200)

visualizer = ResidualsPlot(model, qqplot=True)



visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.show()
pdp_temp = pdp.pdp_isolate(

    model=model, dataset=X_train, model_features=X_train.columns, feature='temp'

)

fig, axes = pdp.pdp_plot(pdp_temp, 'temp',center=False, cluster=True,n_cluster_centers=2,\

                         plot_lines=True, x_quantile=True, show_percentile=True, plot_pts_dist=True)
print('Get the PDP values',pdp_temp.pdp)
#Get all the extrapolated 'temp' values used for the PDP

X_train['month'] = df[df.index.isin(X_train.index)]['month']

X_train['month'] = X_train['month'].apply(lambda x: calendar.month_abbr[x])

#Extract all the extrapolated data points

X_train['extrapolated_temp'] = [pdp_temp.feature_grids.tolist()]*len(X_train)

X_train = X_train.explode('extrapolated_temp')

X_train.head()
X,y = df[['year','hour','season','humidity','windspeed','holiday','workingday']],df['count']

X = pd.get_dummies(X,columns=['season']) # one hot enocode 'season'

#Encode each season

X = X.rename(columns = {'season_1':'spring','season_2':'summer',

                       'season_3':'fall','season_4':'winter'})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train.head()
model = RandomForestRegressor(n_estimators=200)

visualizer = ResidualsPlot(model, qqplot=True)



visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.show()
pdp_season = pdp.pdp_isolate(

    model=model, dataset=X_train, model_features=X_train.columns, 

    feature=['spring', 'summer', 'fall', 'winter']

)

fig, axes = pdp.pdp_plot(pdp_season,'season', center=False, cluster=True,n_cluster_centers=2,\

                         plot_lines=True, x_quantile=True, show_percentile=True)
print('Categorical PDP values',pdp_season.pdp)
X,y = df[['year','hour','temp','season','humidity','windspeed','holiday','workingday']],df['count']

X = pd.get_dummies(X,columns=['season']) # one hot enocode 'season'

X = X.rename(columns = {'season_1':'spring','season_2':'summer',

                       'season_3':'fall','season_4':'winter'})



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train.head()
model = RandomForestRegressor(n_estimators=200)

visualizer = ResidualsPlot(model, qqplot=True)



visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.show()
pdp_season = pdp.pdp_isolate(

    model=model, dataset=X_train, model_features=X_train.columns, 

    feature=['spring', 'summer', 'fall', 'winter']

)

fig, axes = pdp.pdp_plot(pdp_season,'season', center=False, cluster=True,n_cluster_centers=2,\

                         plot_lines=True, x_quantile=True, show_percentile=True)
print('Categorical PDP values',pdp_season.pdp)
#Get all the extrapolated 'season' categories used for the PDP

X_train['month'] = df[df.index.isin(X_train.index)]['month']

X_train['month'] = X_train['month'].apply(lambda x: calendar.month_abbr[x])

#Extract all the extrapolated data points

X_train['extrapolated_season'] = [pdp_season.feature_grids.tolist()]*len(X_train)

X_train = X_train.explode('extrapolated_season')

X_train.head()