import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
# data_dir = "----------INPUT: YOUR DIRECTORY----------" # Uncomment this line if you work on local computer
data_dir = '/kaggle/input/eee-datathon-challenge-2020/' # Comment this line if you work on local computer
diamonds = pd.read_csv(data_dir+"train.csv",index_col = False)
# diamonds = pd.read_csv("train.csv",index_col = False)
diamonds.head() # display the data on the top of the table
diamonds.info()
# Price is int64, best if all numeric attributes have the same datatype, especially as float64
diamonds["price"] = diamonds["price"].astype(float)

# Preview dataset again
diamonds.head()
diamonds.info()
#  Have a rough idea on the distribution of each attributes 
#  X stands for the value and Y stands for the amount
diamonds.hist(bins = 50, figsize = (20, 15))
plt.show()
# Create a correlation matrix between every pair of attributes and try to find the relationship between them
corr_matrix = diamonds.corr()
plt.subplots(figsize = (10, 8))
sns.heatmap(corr_matrix, annot = True )
plt.show()
# See the relationship  of every pair of attributes
sns.pairplot(data = diamonds)
# Show the relationship between price and carat
sns.jointplot(x='carat' , y='price' , data=diamonds , size=5)
# Count the number of diffent cut
sns.factorplot(x='cut', data=diamonds , kind='count',aspect=2.5 )
# Show the relationship between price and quality
sns.factorplot(x='cut', y='price', data=diamonds, kind='box' ,aspect=2.5 )
# Show the percentage of Clarity Categories
labels = diamonds.clarity.unique().tolist()
sizes = diamonds.clarity.value_counts().tolist()
explode = (0.1, 0.0, 0.1, 0, 0.1, 0, 0.1,0)
plt.pie(sizes, explode=explode, labels=labels,autopct='%1.1f%%', shadow=True)
plt.title("Percentage of Clarity Categories")
plt.plot()
plt.show()
# Show the relationship between price and clarity
sns.factorplot(x='color', y='price', data=diamonds, kind='box' ,aspect=2.5 )
scaler = StandardScaler()
# Drop categorical (non-numeric) columns.
diamonds_num = diamonds.drop(["cut", "color", "clarity"], axis = 1)
diamonds_num.head()
hot = OneHotEncoder(sparse=False)
diamonds_cat = diamonds[["cut", "color", "clarity"]]
diamonds_cat_hot = hot.fit_transform(diamonds_cat)

# Output type of a fit_transform is ndarray so we need to convert it back to DataFrame
diamonds_cat_hot = pd.DataFrame(diamonds_cat_hot,index=diamonds_cat.index)
diamonds_cat_hot.head()
# Concat both DataFrames
diamonds_ready = pd.concat([diamonds_num,diamonds_cat_hot],axis=1)
diamonds_ready.head()
X_train, X_test, y_train, y_test = train_test_split(diamonds_ready.drop('price',axis=1),diamonds_ready['price'],test_size=0.2)

model = LinearRegression()
model.fit(diamonds_ready.drop('price',axis=1),diamonds_ready['price'])

# Evaluate model using train set
train_r2 = model.score(diamonds_ready.drop('price',axis=1),diamonds_ready['price']) #R^2 score
pred = model.predict(X_train)
train_mse = mean_squared_error(y_train,pred)
print('train_r2:  ', train_r2)
print('train_mse: ', train_mse)

# Evaluate model using train set
test_r2 = model.score(X_test,y_test) # R^2 score
pred_test = model.predict(X_test)
test_mse = mean_squared_error(y_test,pred_test)
print('test_r2:   ', test_r2)
print('test_mse:  ', test_mse)
diamonds_submission = pd.read_csv(data_dir+"test.csv",index_col = False)
diamonds_submission.head()
scaler_submission = StandardScaler()
diamonds_submission_num = diamonds_submission.drop(["cut", "color", "clarity"], axis = 1)
diamonds_submission_num.head()
hot_submission = OneHotEncoder(sparse=False)
diamonds_submission_cat = diamonds_submission[["cut", "color", "clarity"]]
diamonds_submission_cat_hot = hot_submission.fit_transform(diamonds_submission_cat)

# Output type of a fit_transform is ndarray so we need to convert it back to DataFrame
diamonds_submission_cat_hot = pd.DataFrame(diamonds_submission_cat_hot,index=diamonds_submission_cat.index)
diamonds_submission_cat_hot.head()
# Concat both DataFrames
diamonds_submission_ready = pd.concat([diamonds_submission_num,diamonds_submission_cat_hot],axis=1)
diamonds_submission_ready.head()
diamonds_submission_pred = model.predict(diamonds_submission_ready)
diamonds_submission_pred
price_submission = pd.read_csv(data_dir+"submission_sample.csv",index_col = False)
price_submission['price'] = diamonds_submission_pred
price_submission.head()
price_submission.to_csv("Team_AE_1.csv", index=False)