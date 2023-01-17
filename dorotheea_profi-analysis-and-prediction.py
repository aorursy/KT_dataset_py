# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
features = pd.read_csv("../input/Features data set.csv")

sales = pd.read_csv("../input/sales data-set.csv")

#stores = pd.read_csv("../input/stores data-set.csv")
features['Date'] = pd.to_datetime(features['Date'])

features = features.fillna(0)

features.head()
sales['Date'] = pd.to_datetime(sales['Date'])

sales.head()
sales
import matplotlib.pyplot as plt



data_in = pd.merge(features, sales, on=['Store', 'Date', 'IsHoliday'], how='inner')
filter_dept = data_in['Dept']== 5

filtered_data = data_in[filter_dept]



filter_sale = filtered_data['Weekly_Sales'] == 259955.820000

filtered_data[filter_sale]
data_in = pd.merge(features, sales, on=['Store', 'Date', 'IsHoliday'], how='inner')



max_sales_per_store = data_in.groupby(by=['Store'], as_index=False)['Weekly_Sales'].max()

max_sales_per_store



max_sale_features = pd.DataFrame(columns=list(data_in))

for entry in max_sales_per_store['Weekly_Sales']:

    filter_sale =data_in['Weekly_Sales']== entry

    max_sale_features = max_sale_features.append(pd.DataFrame(data_in[filter_sale], columns=list(data_in)))



max_sale_features
data_in_analysis = data_in.groupby(by=['Date'], as_index=False)['Weekly_Sales'].sum()

data_in_analysis.sort_values('Weekly_Sales', ascending=False)



plt.figure(figsize=(20,5))

plt.plot(data_in_analysis.Date, data_in_analysis.Weekly_Sales)

plt.show()
data_analysis = data_in.groupby(by=['Store'], as_index=False)['Weekly_Sales'].sum()

data_analysis
def scatter(dataset, feature):

    plt.figure()

    plt.scatter(data_in[feature] , data_in['Weekly_Sales'])

    plt.ylabel('weeklySales')

    plt.xlabel(feature)
scatter(data_in, 'Fuel_Price')

scatter(data_in, 'Store')

scatter(data_in, 'Dept')

scatter(data_in, 'Temperature')

scatter(data_in, 'CPI')

scatter(data_in, 'IsHoliday')

scatter(data_in, 'Unemployment')
scatter(data_in, 'MarkDown1')

scatter(data_in, 'MarkDown2')

scatter(data_in, 'MarkDown3')

scatter(data_in, 'MarkDown4')

scatter(data_in, 'MarkDown5')
import seaborn as sns



corr = data_in.corr()

corr = abs(corr)

plt.figure(figsize=(10,10))

sns.heatmap(corr, 

            annot=True, fmt=".3f",

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

plt.show()
corr['Weekly_Sales'].sort_values(ascending=False)
data_in = pd.merge(features, sales, on=['Store', 'Date', 'IsHoliday'], how='inner')

data_in = data_in.drop(columns = ['Date'])



print(data_in.values.shape)

data_in[:][100:105]
from sklearn.model_selection import train_test_split



x = data_in.drop(columns=['Weekly_Sales'])

y = data_in['Weekly_Sales']



xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = 0)

print(xTrain.shape, yTrain.shape)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, make_scorer



model = LinearRegression()

model.fit(xTrain, yTrain)

yPredict = pd.Series(model.predict(xTest))

err = mean_absolute_error(yTest, yPredict)

print('{:.3f}'.format(err))
model.score(xTest, yTest)
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import GridSearchCV





model = MLPRegressor(hidden_layer_sizes=(20,),  activation='tanh', solver = 'adam', max_iter=30, alpha=0.001, batch_size='auto',

    verbose=3)

model.fit(xTrain, yTrain)

yPredict = pd.Series(model.predict(xTest))

err = mean_absolute_error(yTest, yPredict)

print('{:.3f}'.format(err))



from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=5)



model.fit(xTrain, yTrain)

yPredict = pd.Series(model.predict(xTest))

err = mean_absolute_error(yTest, yPredict)

print('{:.3f}'.format(err))
model.score(xTest, yTest)
from sklearn.ensemble import ExtraTreesRegressor



model = ExtraTreesRegressor(n_estimators=100,max_features='auto', verbose=1)



model.fit(xTrain, yTrain)

yPredict = pd.Series(model.predict(xTest))

err = mean_absolute_error(yTest, yPredict)

print('{:.3f}'.format(err))
model.score(xTest, yTest)
from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(n_estimators=50, max_features=0.99, min_samples_leaf=2,

                          n_jobs=-1, oob_score=True)

model.fit(xTrain, yTrain)

yPredict = pd.Series(model.predict(xTest))

err = mean_absolute_error(yTest, yPredict)

print('{:.3f}'.format(err))
model.score(xTest, yTest)
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators= 400, max_depth=4, min_samples_split=2,

          learning_rate=0.1, loss='ls')



model.fit(xTrain, yTrain)

yPredict = pd.Series(model.predict(xTest))

err = mean_absolute_error(yTest, yPredict)

print('{:.3f}'.format(err))
model.score(xTest, yTest)
data_in = pd.merge(features, sales, on=['Store', 'Date', 'IsHoliday'], how='inner')

data_in = data_in.drop(columns = ['Date'])

data_in.describe()
x = data_in.drop(columns=['Weekly_Sales'])

y = data_in['Weekly_Sales']



xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = 0)

print(xTrain.shape, yTrain.shape)
from sklearn.ensemble import ExtraTreesRegressor



model = ExtraTreesRegressor(n_estimators=100,max_features='auto', verbose=1)



model.fit(xTrain, yTrain)

yPredict = pd.Series(model.predict(xTest))

err = mean_absolute_error(yTest, yPredict)

print('{:.3f}'.format(err))
score_tree_1 = model.score(xTest, yTest)

print(score_tree_1)
from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(n_estimators=50, max_features=0.99, min_samples_leaf=2,

                          n_jobs=-1, oob_score=True)

model.fit(xTrain, yTrain)

yPredict = pd.Series(model.predict(xTest))

err = mean_absolute_error(yTest, yPredict)

print('{:.3f}'.format(err))
score_forest_1 = model.score(xTest, yTest)

print(score_forest_1)
data_in = pd.merge(features, sales, on=['Store', 'Date', 'IsHoliday'], how='inner')

data_in = data_in.drop(columns = ['Date'])

data_in.describe()
data_in = data_in.drop(columns=['Fuel_Price', 'Temperature',  'IsHoliday', 'MarkDown2', 'CPI',  'Unemployment',])
x = data_in.drop(columns=['Weekly_Sales'])

y = data_in['Weekly_Sales']



xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = 0)

print(xTrain.shape, yTrain.shape)
from sklearn.ensemble import ExtraTreesRegressor



model = ExtraTreesRegressor(n_estimators=100,max_features='auto', verbose=1)



model.fit(xTrain, yTrain)

yPredict = pd.Series(model.predict(xTest))

err = mean_absolute_error(yTest, yPredict)

print('{:.3f}'.format(err))
score_tree_2 = model.score(xTest, yTest)

print(score_tree_2)
model = RandomForestRegressor(n_estimators=50, max_features=0.99, min_samples_leaf=2,

                          n_jobs=-1, oob_score=True)

model.fit(xTrain, yTrain)

yPredict = pd.Series(model.predict(xTest))

err = mean_absolute_error(yTest, yPredict)

print('{:.3f}'.format(err))
score_forest_2 = model.score(xTest, yTest)

print(score_forest_2)
data_in = pd.merge(features, sales, on=['Store', 'Date', 'IsHoliday'], how='inner')

data_in = data_in.drop(columns = ['Date'])

data_in.describe()
data_in = data_in.drop(columns=['Fuel_Price', 'Temperature',  'IsHoliday', 'MarkDown2', 'CPI',  'Unemployment', 'MarkDown1', 'MarkDown3', 'MarkDown4', 'MarkDown5'])
x = data_in.drop(columns=['Weekly_Sales'])

y = data_in['Weekly_Sales']



xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = 0)

print(xTrain.shape, yTrain.shape)
data_in.describe()
from sklearn.ensemble import ExtraTreesRegressor



model = ExtraTreesRegressor(n_estimators=100,max_features='auto', verbose=1)



model.fit(xTrain, yTrain)

yPredict = pd.Series(model.predict(xTest))

err = mean_absolute_error(yTest, yPredict)

print('{:.3f}'.format(err))
score_tree_3 = model.score(xTest, yTest)

print(score_tree_3)
model = RandomForestRegressor(n_estimators=50, max_features=0.99, min_samples_leaf=2,

                          n_jobs=-1, oob_score=True)

model.fit(xTrain, yTrain)

yPredict = pd.Series(model.predict(xTest))

err = mean_absolute_error(yTest, yPredict)

print('{:.3f}'.format(err))
score_forest_3 = model.score(xTest, yTest)

print(score_forest_3)
data_scores = [[score_tree_1, score_forest_1],

              [score_tree_2, score_forest_2],

              [score_tree_3, score_forest_3]]



data_scores
from sklearn.ensemble import RandomForestRegressor



data_in = pd.merge(features, sales, on=['Store', 'Date', 'IsHoliday'], how='inner')

data_in = data_in.drop(columns = ['Date'])



x = data_in.drop(columns=['Weekly_Sales'])

y = data_in['Weekly_Sales']



xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = 0)

print(xTrain.shape, yTrain.shape)



model = RandomForestRegressor(n_estimators=50, max_features=0.99, min_samples_leaf=2,

                          n_jobs=-1, oob_score=True)

model.fit(xTrain, yTrain)

yPredict = pd.Series(model.predict(xTest))

err = mean_absolute_error(yTest, yPredict)

print('{:.3f}'.format(err))



names = xTrain.columns

print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), names), 

             reverse=True))