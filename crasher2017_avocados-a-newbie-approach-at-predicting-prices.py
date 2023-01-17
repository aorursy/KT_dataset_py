import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv('../input/avocado.csv', index_col=0)

df.head(5)
df = df.reset_index(drop=True)
df['type'] = df['type'].astype('category')
df['type'] = df['type'].cat.codes
filter1=df.region!='TotalUS'
data1=df[filter1]

sorted_average = data1.groupby(["region"])['Total Volume'].aggregate(np.mean).reset_index().sort_values('Total Volume')

fig, ax = plt.subplots(figsize=(12, 8))
plt.xticks(rotation=90)
ax=sns.barplot(x='region',y='Total Volume', data=data1, palette='magma', order=sorted_average['region'])
filter2=df['type']==1
data2=df[filter2]

sorted_average = data2.groupby(["region"])['AveragePrice'].aggregate(np.mean).reset_index().sort_values('AveragePrice')

fig, ax = plt.subplots(figsize=(12, 8))
plt.xticks(rotation=90)
plt.title('Organic, Average Price')
ax=sns.barplot(x='region',y='AveragePrice', data=data2, palette='magma', order=sorted_average['region'])
filter2=df['type']==0
data2=df[filter2]


sorted_average = data2.groupby(["region"])['AveragePrice'].aggregate(np.mean).reset_index().sort_values('AveragePrice')

fig, ax = plt.subplots(figsize=(12, 8))
plt.xticks(rotation=90)
plt.title('Conventional, Average Price')
ax=sns.barplot(x='region',y='AveragePrice', data=data2, palette='magma', order=sorted_average['region'])
filter3=df['region']!='TotalUS'
data3=df[filter3]

fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Average Price per year')
g = sns.barplot(x = 'year', y = 'AveragePrice', hue='type', data=data3)
filter3=df['region']=='TotalUS'
data3=df[filter3]

fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Total Volume per year (TotalUS only)')
g = sns.barplot(x = 'year', y = 'Total Volume', hue='type', data=data3, estimator=sum)
filter5=df['type']==0
data5=df[filter5]

g = sns.lmplot(x='Total Volume',y='AveragePrice', data=data5, fit_reg=True, height=8, aspect=1.2)
fig = g.fig
fig.suptitle("Conventional: Volume vs. Average Price")
plt.show()
filter5=df['type']==1
data5=df[filter5]

g = sns.lmplot(x='Total Volume',y='AveragePrice', data=data5, fit_reg=True, height=8, aspect=1.2)
fig = g.fig
fig.suptitle("Organic: Volume vs. Average Price")
plt.show()
sns.clustermap(df.corr(), center=0, cmap="vlag", annot = True, linewidths=.75, figsize=(13, 13));
g = sns.PairGrid(df)
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter);
df['fregion'] = df['region'].values
df = pd.get_dummies(df, columns=['fregion'])
df.head()
drop_list = ['AveragePrice', 'Date', 'region']

X = df.drop(drop_list, axis=1)
y = df['AveragePrice'].values.ravel()
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=1)

clf = RandomForestRegressor(n_estimators= 100, random_state=42)
clf.fit(Xtrain,ytrain)

scores = cross_val_score(clf, Xtrain, ytrain, cv=3, n_jobs=-1)

print(f"{round(np.mean(scores),3)*100}% accuracy")
print(f"MSE {mean_squared_error(y_pred=clf.predict(Xtest), y_true=ytest)}")
predictions = clf.predict(Xtest)

#print(predictions)
# Calculate the absolute errors
errors = abs(predictions - ytest)

fig, ax = plt.subplots(figsize=(12, 8))
plt.hist(errors, bins = 10, edgecolor = 'black');
# Print out the mean absolute error (mae)
#print(f"Test: {mean_squared_error(ytest, predictions)} ")
print(f"R2 score: {r2_score(ytest, predictions)}")
print(f"Mean absolute error:  {mean_absolute_error(ytest, predictions)} USD")
fig, ax = plt.subplots(figsize=(12, 8))
g = sns.regplot(x = predictions,y = ytest)
# Get numerical feature importances
importances = list(clf.feature_importances_)

feature_list = list(X.columns)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
EconomicAnalysisRegion = []

for region in df['region']:
    if region in ['California', 'LasVegas', 'LosAngeles', 'Portland', 'Sacramento', 'SanDiego', 'SanFrancisco', 'Seattle', 'Spokane']:
        EconomicAnalysisRegion.append('Far West')
    elif region in ['Chicago', 'CincinnatiDayton', 'Columbus', 'Detroit', 'GrandRapids', 'Indianapolis']:
        EconomicAnalysisRegion.append('Great Lakes')
    elif region in ['GreatLakes']:
        EconomicAnalysisRegion.append('GreatLakes')
    elif region in ['Albany', 'BaltimoreWashington', 'BuffaloRochester', 'HarrisburgScranton', 'HartfordSpringfield', 'NewYork', 'Philadelphia', 'Pittsburgh', 'Syracuse']:
        EconomicAnalysisRegion.append('Mideast')
    elif region in ['Midsouth']:
        EconomicAnalysisRegion.append('Midsouth') 
    elif region in ['Boston', 'HartfordSpringfield']:
        EconomicAnalysisRegion.append('New England') 
    elif region in ['Northeast']:
        EconomicAnalysisRegion.append('Northeast')
    elif region in ['NorthernNewEngland']:
        EconomicAnalysisRegion.append('NorthernNewEngland')
    elif region in ['Plains', 'StLouis']:
        EconomicAnalysisRegion.append('Plains')
    elif region in ['Boise', 'Denver']:
        EconomicAnalysisRegion.append('Rocky Mountains')
    elif region in ['SouthCarolina']:
        EconomicAnalysisRegion.append('SouthCarolina')
    elif region in ['SouthCentral']:
        EconomicAnalysisRegion.append('SouthCentral')
    elif region in ['Atlanta', 'Charlotte', 'Jacksonville', 'Louisville', 'MiamiFtLauderdale', 'Nashville', 'NewOrleansMobile', 'Orlando', 'RaleighGreensboro', 'RichmondNorfolk', 'Roanoke', 'Southeast']:
        EconomicAnalysisRegion.append('Southeast')
    elif region in ['DallasFtWorth', 'Houston', 'PhoenixTucson', 'Tampa']:
        EconomicAnalysisRegion.append('SouthWest')
    elif region in ['TotalUS']:
        EconomicAnalysisRegion.append('TotalUS')
    elif region in ['West']:
        EconomicAnalysisRegion.append('West')
    elif region in ['WestTexNewMexico']:
        EconomicAnalysisRegion.append('WestTexNewMexico')
        

df['Economic Analysis Region'] = EconomicAnalysisRegion

df = pd.get_dummies(df, columns=['Economic Analysis Region'])
df['Date'] = pd.to_datetime(df['Date'])
df['month'] = df['Date'].dt.month

seasons = []

for month in df['month']:
    if month in [1, 2, 12]:
        seasons.append('winter')
    elif month in [3, 4, 5]:
        seasons.append('spring')
    elif month in [6, 7, 8]:
        seasons.append('summer')
    elif month in [9, 10, 11]:
        seasons.append('fall')
                
df['season'] = seasons
df = pd.get_dummies(df, columns=['season'])
df['Date'] = pd.to_datetime(df['Date'])
df['week'] = df['Date'].dt.week.shift(-2).ffill()
df.head(5)
drop_list = ['AveragePrice', 'Date', '4046', '4225', '4770', 'Total Bags', 'Total Volume', 'region', 'Small Bags', 'Large Bags', 'XLarge Bags']

X = df.drop(drop_list, axis=1)
y = df['AveragePrice'].values.ravel()
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestRegressor(n_estimators= 100, random_state=42)

clf.fit(Xtrain,ytrain)

scores = cross_val_score(clf, Xtrain, ytrain, cv=3, n_jobs=-1)

print(f"{round(np.mean(scores),3)*100}% accuracy")
print(f"MSE {mean_squared_error(y_pred=clf.predict(Xtest), y_true=ytest)}")
predictions = clf.predict(Xtest)

#print(predictions)
# Calculate the absolute errors
errors = abs(predictions - ytest)
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.hist(errors, bins = 10, edgecolor = 'black');
# Print out the mean absolute error (mae)
print('Mean Absolute Error: USD', round(np.mean(errors), 3))
# Get numerical feature importances
importances = list(clf.feature_importances_)

feature_list = list(X.columns)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
from fbprophet import Prophet
m = Prophet()
df = pd.read_csv('../input/avocado.csv', index_col=0)
df['Date'] = pd.to_datetime(df['Date'])

mask = (df['region'] == 'TotalUS') & (df['type'] == 'conventional')
df = df[mask]

# Change column names as Prophet requires.
df.rename(columns={'Date': 'ds', 'AveragePrice': 'y'}, inplace=True)
m.fit(df);
future = m.make_future_dataframe(periods=52,freq='w')
future.tail(3)
forecast = m.predict(future)
m.plot(forecast, xlabel = 'Date', ylabel = 'Price');
fig2 = m.plot_components(forecast)
cmp_df = df.join(forecast.set_index('ds'), on='ds')
cmp_df = cmp_df[cmp_df['y'].notnull()]
print(f"MSE {mean_squared_error(y_pred=cmp_df.yhat, y_true=cmp_df.y)}")
print(f"R2 score: {r2_score(cmp_df.y, cmp_df.yhat)}")
print(f"Mean absolute error:  {mean_absolute_error(cmp_df.y, cmp_df.yhat)} USD")
errors = abs(cmp_df.yhat - cmp_df.y)
plt.hist(errors, bins = 15, edgecolor = 'black');
sns.set(rc={'figure.figsize':(12,10)})
g = sns.regplot(x = cmp_df.yhat,y = cmp_df.y)
m = Prophet()
df = pd.read_csv('../input/avocado.csv', index_col=0)
df['Date'] = pd.to_datetime(df['Date'])

mask = (df['region'] == 'Houston') & (df['type'] == 'organic')
df = df[mask]

df.rename(columns={'Date': 'ds', 'AveragePrice': 'y'}, inplace=True)
m.fit(df);
future = m.make_future_dataframe(periods=52,freq='w')
forecast = m.predict(future)
m.plot(forecast, xlabel = 'Date', ylabel = 'Price');
cmp_df = df.join(forecast.set_index('ds'), on='ds')
cmp_df = cmp_df[cmp_df['y'].notnull()]

print(f"MSE {mean_squared_error(y_pred=cmp_df.yhat, y_true=cmp_df.y)}")
print(f"R2 score: {r2_score(cmp_df.y, cmp_df.yhat)}")
print(f"Mean absolute error:  {mean_absolute_error(cmp_df.y, cmp_df.yhat)} USD")
