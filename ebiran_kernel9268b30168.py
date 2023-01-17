
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from subprocess import check_output

df = pd.read_csv('../input/clean_data.csv')
df.info()


dff = df[['Elevation','MeanTemp','HeatDegDays','Total_Precip_mm']]
dff.describe()
stations = df['Name'].unique()
num_of_stations = stations.size
print('Total # of stations: ' + str(num_of_stations))
      
    
df1 = df.groupby(['Name','Year'])['Total_Precip_mm'].sum()
df1 = pd.DataFrame(df1)

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
df1.groupby('Name').mean().sort_values(by='Total_Precip_mm', ascending=False)['Total_Precip_mm'].plot('bar', color='r',width=0.3,title='Average AnnualPrecipitation', fontsize=20)
plt.xticks(rotation = 90)
plt.ylabel('Average Annual Percipitation (mm)')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111)
dfg = df.groupby('Year')['Total_Precip_mm'].sum()
dfg.plot('line', title='Overall Annual Percipitation', fontsize=20)
plt.ylabel('Overall Percipitation (mm)')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
print('Max: ' + str(dfg.max()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.max()].index.values[0:]))
print('Min: ' + str(dfg.min()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.min()].index.values[0:]))
print('Mean: ' + str(dfg.mean()))

dfm = df.groupby('Month')['Total_Precip_mm'].sum()
dfm = pd.DataFrame(dfm)

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
dfm.groupby('Month').mean().sort_values(by='Total_Precip_mm')['Total_Precip_mm'].plot('bar', color='r',width=0.3,title='Average Monthly Precipitation', fontsize=20)
plt.xticks(rotation = 90)
plt.ylabel('Average Monthly Percipitation (mm)')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
dff = df[['Elevation','MeanTemp','HeatDegDays','Total_Precip_mm']]
dff.boxplot()
dff.hist()
#Variable Corrolation
corr_matrix = df.corr()
corr_matrix['Total_Precip_mm'].sort_values(ascending=False)

#Scatter by Total Precipitation
from pandas.tools.plotting import scatter_matrix
df.plot(kind="scatter", x='Longitude', y= 'Latitude', alpha=0.4, figsize=(10,7),
    c= df.Total_Precip_mm, cmap=plt.get_cmap("jet"), colorbar=True,sharex=False)

#Linear Regression Plot
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

Mean = df.groupby(['Month','Day'])['MeanTemp'].mean()
Rain = df.groupby(['Month','Day'])['Total_Precip_mm'].mean()

Mean = Mean.values[:,np.newaxis]
Rain = Rain.values

modell = LinearRegression()
modell.fit(Mean, Rain)
plt.scatter(Mean, Rain,color='r')
plt.ylabel('Precipitation')
plt.xlabel('Mean Temp')
plt.plot(Mean, modell.predict(Mean),color='k')
plt.show()

#Linear Regression
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression

df = df.fillna(df.mean())
Mean = df.groupby(['Month','Day'])['MeanTemp'].max()
Rain = df.groupby(['Month','Day'])['Total_Precip_mm'].max()

X = Mean.values[:,np.newaxis]
y = Rain.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

regressor = LinearRegression()
regressor = regressor.fit(X_train, y_train)

scores = cross_val_score(regressor, X, y, cv=5)
print('Scores: ',scores)
print('average score: {}'.format(scores.mean()))

y_pred = regressor.predict(X_test)
print('Linear Regression R squared": %.3f' % (regressor.score(X_test, y_test) * 100), '%')

#root-mean-square error (RMSE)
from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(y_pred, y_test)
lin_rmse = np.sqrt(lin_mse)
print('Linear Regression RMSE: %.3f' % lin_rmse)

#Mean absolute error (MAE):
from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(y_pred, y_test)
print('Linear Regression MAE: %.3f' % lin_mae)

#ensemble Method (Gradient boosting ) for Linear regre
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

model = ensemble.GradientBoostingRegressor()
model.fit(X_train, y_train)
print('Gradient Boosting R squared": %.3f' % (regressor.score(X_test, y_test) * 100), '%')

y_pred = model.predict(X_test)
model_mse = mean_squared_error(y_pred, y_test)
model_rmse = np.sqrt(model_mse)
print('Gradient Boosting RMSE: %.3f' % model_rmse)
#Decision tree precision 
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

df5 = pd.read_csv('../input/clean_data.csv')
df5 = df5[['Year','Month','Day','MeanTemp','HeatDegDays','Total_Precip_mm']]
Mean = df5.groupby(['Month','Day'])['MeanTemp'].max()
Rain = df5.groupby(['Month','Day'])['Total_Precip_mm'].max()

X = Mean.values[:,np.newaxis]
y = Rain.values
X, y = make_classification(n_samples=100, n_informative=10, n_classes=3)
sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.3, random_state=0)
for train_idx, test_idx in sss:
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    #fit the model
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print('Accuracy: %.3f' % (accuracy_score(y_test, y_pred) * 100) ,'%')
    
    print('Score: ',f1_score(y_test, y_pred, average="macro"))
    print('Precision: ',precision_score(y_test, y_pred, average="macro"))
    print('Recall: ',recall_score(y_test, y_pred, average="macro"))  
#ensemble Method (Gradient boosting ) For Decision tree
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

df5 = pd.read_csv('../input/clean_data.csv')
df5 = df5[['Year','Month','Day','MeanTemp','HeatDegDays','Total_Precip_mm']]
Mean = df5.groupby(['Month','Day'])['MeanTemp'].max()
Rain = df5.groupby(['Month','Day'])['Total_Precip_mm'].max()

X = Mean.values[:,np.newaxis]
y = Rain.values
X, y = make_classification(n_samples=100, n_informative=10, n_classes=3)
sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.3, random_state=0)
for train_idx, test_idx in sss:
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

model = ensemble.GradientBoostingRegressor()
model.fit(X_train, y_train)
print('Gradient Boosting R squared": %.3f' % (model.score(X_test, y_test) * 100), '%')

#Decision Tree plot: True vs, Predicted
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

df5 = pd.read_csv('../input/clean_data.csv')
df5 = df5[['Year','Month','Day','MeanTemp','HeatDegDays','Total_Precip_mm']]
Mean = df5.groupby(['Month','Day'])['MeanTemp'].max()
Rain = df5.groupby(['Month','Day'])['Total_Precip_mm'].max()

X = Mean.values[:,np.newaxis]
y = Rain.values
clf= DecisionTreeRegressor()
clf.fit(X, y)

predicted = clf.predict(X)
expected = y

from matplotlib import pyplot as plt
plt.figure(figsize=(4, 3))
plt.scatter(expected, predicted)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True Precipitation')
plt.ylabel('Predicted Precipitation')
plt.tight_layout()

#Feature importance
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tabulate import tabulate

df = pd.read_csv('../input/clean_data.csv')
df = df.fillna(df.mean())
df = df[['Total_Precip_mm','MeanTemp','Year','Month','Day']]
X = df.drop(['Total_Precip_mm'], axis=1)
y = np.array(df['Total_Precip_mm'],dtype='long')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
clf = RandomForestRegressor(n_estimators=100)
model1 = clf.fit(X_train, y_train)
headers = ["name", "score"]
values = sorted(zip(X_train.columns, model1.feature_importances_), key=lambda x: x[1] * -1)
print(tabulate(values, headers, tablefmt="plain"))