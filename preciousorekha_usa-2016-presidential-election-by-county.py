import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df = pd.read_csv('../input/us-elections-dataset/usa-2016-presidential-election-by-county.csv', sep = ';')
df
df.info()
df.shape
df.isnull().sum()
df.describe()
df
df.Votes.sum() # total votes casted in all states
df.County.nunique()  # total no of Counties
df[['Democrats 2016', 'Republicans 2016']].sum() # total votes by party
df.groupby('Votes')['County'].value_counts().sort_values(ascending = False).head()
df.groupby('Votes')['State'].value_counts().sort_values(ascending = False).head()
df.groupby(['Votes','State'])['Republicans 08 (Votes)'].max().sort_values(ascending = False).head()
df.groupby(['Votes','State'])['Democrats 08 (Votes)'].min().sort_values(ascending = False).head()
sns.distplot(df['Democrats 2016'].dropna(), kde=False, bins=20)
import cufflinks as cf
cf.go_offline()
df['County'].dropna().iplot(kind='bar',)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap = 'viridis')
df.isnull().sum().sort_values(ascending= False)
df.isna().sum().count()
df.shape
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
missing_value_df.sort_values('percent_missing', inplace=True, ascending = False)
missing_value_df
df.dropna(axis=1,thresh=0.7*len(df), inplace=True) # using a thresh function to get certain range of values to drop
df.isnull().sum().max()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap = 'viridis')
df.isnull().sum().sort_values(ascending=False)
df.fillna(method='ffill', inplace=True )
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap = 'viridis')
df
correlated_features = set()
correlation_matrix = df.corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
correlated_features
df = df[['Asian','At Least High School Diploma','Black','Child.Poverty.living.in.families.below.the.poverty.line',
        'Democrats 08 (Votes)','Democrats 12 (Votes)','Democrats 2008','Democrats 2012','Graduate Degree',
        'Nearest County','Poverty.Rate.below.federal.poverty.threshold','Republicans 08 (Votes)',
         'Republicans 12 (Votes)','Republicans 2008','Republicans 2012','Total Population','Votes',
        'White','White  Asian','total08','total12','total16', 'Democrats 2016', 'Republicans 2016']]
df = df.dropna()
df.head()
features = df.drop(columns=['Democrats 2016', 'Republicans 2016'])
target = df[['Democrats 2016', 'Republicans 2016']]
print(target.shape)
print(features.shape)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( features, target, test_size=0.3, random_state=42)
print(X_test.shape)
print(X_train.shape)
print(y_test.shape)
print(y_train.shape)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalised_X_train = scaler.fit_transform(X_train)
normalised_X_train = pd.DataFrame(normalised_X_train, columns=X_train.columns)
normalised_X_train.head()

normalised_X_test = scaler.transform(X_test)
normalised_X_test = pd.DataFrame(normalised_X_test, columns=X_test.columns)
normalised_X_test.head()
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
max_depth = 30
regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
max_depth=max_depth,
random_state=0))
regr_multirf.fit(normalised_X_train, y_train)

regr_rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth,
random_state=2)
regr_rf.fit(normalised_X_train, y_train)
y_multirf = regr_multirf.predict(normalised_X_test)
y_rf = regr_rf.predict(normalised_X_test)
y_multirf # comparing the predicted results
y_test.head() # Original result
from sklearn import metrics
# MAE values
MAE = metrics.mean_absolute_error(y_test,y_multirf)
round(MAE,2) # this tell us our model actually predicts an average of 2.76 more or less value which is very impressive 
# R Squared values
r2 = metrics.r2_score(y_test,y_multirf)
round(r2,2)  # This gives us a better rating of our model that it is actually very impressive, it tells us that 
               #  the regression line has fitted our dataset very well

# RMSE
RMSE = np.sqrt(metrics.mean_squared_error(y_test,y_multirf))
RMSE


y_test.describe()
y_multirf = pd.DataFrame(y_multirf)
y_multirf.sum() # Republicans emerged winners according to this discription

sns.jointplot(data = df, x = 'Democrats 2016', y = 'Votes')
sns.jointplot(data = df, x = 'Republicans 2016', y = 'Votes')

