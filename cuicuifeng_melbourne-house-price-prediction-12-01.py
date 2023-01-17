import pandas as pd

data = pd.read_csv('/kaggle/input/melbourne-housing-snapshot/melb_data.csv')

data.head()
data.columns
data.describe()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.distplot(data['Price'])
sns.heatmap(data.corr())
sns.pairplot(data)
y = data.Price

features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = data[features]
X.describe()
X.head()
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state = 10)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
plt.scatter(y_test, predictions)
sns.distplot((y_test-predictions), bins=50)
from sklearn.metrics import mean_absolute_error

predict_train = model.predict(X_train)

mean_absolute_error(y_train, predict_train)
from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor



def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)
for max_leaf_nodes in [5, 50, 500, 5000]:

    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))