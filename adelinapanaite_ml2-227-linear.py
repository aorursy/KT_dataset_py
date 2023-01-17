# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


data = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

print("The training set shape is : %s" % str(data.shape))
data.head(n=10)
sum_result = data.isna().sum(axis=0).sort_values(ascending=False)
missing_values_columns = sum_result[sum_result > 0]
print('They are %s columns with missing values : \n%s' % (missing_values_columns.count(), [(index, value) for (index, value) in missing_values_columns.iteritems()]))
###Data exploration- the whole dataset

data.hist(bins=50, figsize=(15,15))
plt.show()
data = data.copy()
data.plot(kind="scatter", x="price", y="sqft_living", alpha=0.4,
            s=data["bedrooms"]/100, label="bedrooms",
            c="sqft_living15", cmap=plt.get_cmap("jet"), colorbar=True,
            figsize=(15,7))
plt.legend()
#The target variable sqft_living 

sns.distplot(data['sqft_living'])
data["sqft_living"].describe()
#Exploring the independent variables

numeric_cols = list(data.select_dtypes(include=[np.number]))
numerical_values = data[list(data.select_dtypes(include=[np.number]))]
# Get the more correlated variables by sorting in descending order for the SalePrice column
ix = numerical_values.corr().sort_values('sqft_living', ascending=False).index
df_sorted_by_correlation = numerical_values.loc[:, ix]
# take only the first 15 more correlated variables
fifteen_more_correlated = df_sorted_by_correlation.iloc[:, :15]
corr = fifteen_more_correlated.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    # display a correlation heatmap
    ax = sns.heatmap(corr, mask=mask, annot=True)
corr_matrix = data.corr().abs()
corr_matrix
##Exploring the dependent variables

sns.distplot(data['price'])
sns.distplot(data['bedrooms'])
sns.distplot(data['bathrooms'])
sns.distplot(data['sqft_above'])
sns.distplot(data['grade'])
sns.distplot(data['sqft_living15'])
data.columns
#Train & test data
x = data[['bedrooms', 'bathrooms', 'sqft_above', 'grade', 'sqft_living15', 'price', 'sqft_lot']]
#separate the other attributes from the predicting attribute

y = data[['sqft_living']]
#separte the predicting attribute into Y for model training
from sklearn.model_selection import train_test_split

#import model selection train test split for splitting the data into test and train for 
#model validation.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
#loading the model constructor
lm.fit(x_train,y_train)
#training or fitting the train data into the model #training the algorithm
print(lm.intercept_)
lm.coef_
#examining the co-efficients of the fitted model.
x_train.columns

cdf = pd.DataFrame(data=lm.coef_.reshape(7,1),index=x_train.columns,columns=['Coeff'])
cdf
###prediction 

predictions = lm.predict(x_test)
plt.scatter(y_test,predictions)
#to visualise the predictions and the test Y !! almost it is forming a linear line with less deviation
plt.scatter(y_test,predictions)
#to visualise the predictions and the test Y !! almost it is forming a linear line with less deviation
sns.distplot((y_test-predictions))
###Model evaluation

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
y_test_predict = lm.predict(x_test)
r2_score(y_test, y_test_predict)

# plotting the y_test vs y_pred
# ideally should have been a straight line
plt.scatter(y_test, y_test_predict)
plt.show()