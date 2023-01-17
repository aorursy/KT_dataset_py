import pandas as pd



data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')



pd.set_option('max_column', 200)



# data.head()



X = data[['LotArea','OverallQual']]

y = data['SalePrice']



#X.head()

#y.head()



# MODEL:

from sklearn.model_selection import cross_val_predict

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_log_error



model = DecisionTreeRegressor(max_depth=3)



y_pred_for_scoring = cross_val_predict(model, X, y, cv=5)



score = mean_squared_log_error(y, y_pred_for_scoring) # Lower is better

print(f'The extimated MSLE is {score:.3f} (lower is better)')



model.fit(X,y) 



data_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



X_test = data_test[['LotArea','OverallQual']]

y_pred = model.predict(X_test)



answer = pd.DataFrame({

    'Id':data_test['Id'],

    'SalePrice': y_pred

})



# answer.head()



answer.to_csv('answer.csv', index=False)
import pandas as pd

data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
pd.set_option('max_column', 200)

data.head()
import plotly.express as px

import plotly.offline as py



fig1 = px.scatter(data, x='LotArea',y='SalePrice', title='SalePrice vs LotArea')

fig2 = px.scatter(data, x='OverallQual', y='SalePrice', title='SalePrice vs OverallQual')



py.iplot(fig1)

py.iplot(fig2)
X = data[['LotArea','OverallQual']]

y = data['SalePrice']
X.head()
y.head()
from sklearn.tree import DecisionTreeRegressor



model = DecisionTreeRegressor()
from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import cross_val_predict



y_pred = cross_val_predict(model, X, y, cv=5)



mean_squared_log_error(y, y_pred)
import seaborn as sns

import matplotlib.pyplot as plt



fig = sns.jointplot(y, y_pred)



fig.ax_joint.set_title('Predicted Vs Actual')

fig.ax_joint.set_xlabel('Actaul SalePrice')

fig.ax_joint.set_ylabel('Predicted SalePrice')



plt.tight_layout()
data_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

data_test.head()
model.fit(X,y) # Training on the training data
from sklearn.tree import plot_tree



plt.figure(figsize=(20,10))

_ = plot_tree(model, feature_names=X.columns, max_depth=3, fontsize=10, precision=0)
X_test = data_test[['LotArea','OverallQual']]

y_pred = model.predict(X_test) # Predicting on the test data
y_pred
df = pd.DataFrame({

    'Id': data_test['Id'],

    'SalePrice': y_pred

})
df.head()
# Saving to a CSV

df.to_csv('answer.csv', index=False)