# Code you have previously used to load data
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'

home_data = pd.read_csv(iowa_file_path)
# print the list of columns in the dataset to find the name of the prediction target
home_data.columns
y = home_data['SalePrice']
# Create the list of features 
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']

# select data corresponding to features in feature_names
X =home_data[feature_names]
# print description or statistics from X
print(X.describe())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
from sklearn.tree import DecisionTreeRegressor
#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
model = DecisionTreeRegressor(random_state=1)

# Fit the model
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(predictions)
import seaborn as sns 
import matplotlib.pyplot as plt 
fig, ax = plt.subplots()
ax.scatter(predictions, y_test, edgecolors=(0, 0, 1))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.show()
# model evaluation for testing set

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
r2 = metrics.r2_score(y_test, predictions)

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))