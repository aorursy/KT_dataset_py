import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv('../input/2015.csv')
print(data.columns)
data[:10]
y = data['Happiness Score'] # target variable
happiness_score_predictors = ['Country','Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',
       'Generosity', 'Dystopia Residual']
X = data[happiness_score_predictors] # predictors
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

model_decision_tree = DecisionTreeRegressor()

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
model_decision_tree.fit(train_X.drop(['Country'], axis=1), train_y)
prediction_tree = model_decision_tree.predict(val_X.drop(['Country'], axis=1))
#val_y - prediction # to see the actual difference between expected and calculated values
error_tree = mean_absolute_error(val_y, prediction_tree)
print(error_tree)
from sklearn.ensemble import RandomForestRegressor

model_random_forest = RandomForestRegressor()
model_random_forest.fit(train_X.drop(['Country'], axis=1), train_y)
prediction_forest = model_random_forest.predict(val_X.drop(['Country'], axis=1))
error_forest = mean_absolute_error(val_y, prediction_forest)
print(error_forest)
import matplotlib.pyplot as plt

dt = data[:40]['Country']
sorted_val_y = val_y.sort_values(ascending=False)

plt.plot(np.sort(prediction_tree), marker='o', label='Decision Tree model')
plt.plot(np.sort(prediction_forest), marker='o', label='Random Forest model')
plt.plot(np.sort(val_y.values), marker='o', label='Actual')
plt.legend()
plt.xticks(range(len(val_y.values)), val_X['Country'], rotation = 60)

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 30
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size
plt.show()