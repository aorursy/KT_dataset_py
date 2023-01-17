import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

from sklearn.metrics import precision_score, recall_score

from sklearn import preprocessing

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train.head()
test.head()
test.describe()
train.describe()
test.dtypes

train.dtypes
# We will see if there's a relationship between SalesPrice and OverallQual: overall material and finish of the house.

# 1-10 where 1=Very Poor and 10=Very Excellent

sns.barplot(train.OverallQual,train.SalePrice)

# Scatter plot to see the relationship between numerical values



#lotarea = px.scatter(house, x='LotArea', y='SalePrice')

#lotarea.update_layout(title='Sales Price Vs Area',xaxis_title="Area",yaxis_title="Price")

#lotarea.show()





plt.scatter(x =train.LotArea,y = train.SalePrice,c = 'black')

plt.title('Sales Price Vs Area')

plt.xlabel('LotArea')

plt.ylabel('SalePrice')

plt.show()
# Distribution of SalesPrice 

print(train['SalePrice'].describe())

sns.distplot(train['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.5});



train.dtypes
# we will now only inlcude the numerical variables to see numerical distribution as a whole

train_num = train.select_dtypes(include = ['float64', 'int64'])

train_num.head()
train_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);



# pulling data into  the target (y) which is the SalePrice and predictors (X)

train_y = train.SalePrice

pred_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
#prediction data

train_x = train[pred_cols]



model =  LogisticRegression()



model.fit(train_x, train_y)
# pulling same columns from the test data

test_x = test[pred_cols]

pred_prices = model.predict(test_x)

print(pred_prices)
#save file

ayesha_submission = pd.DataFrame({'Id': test.Id, 'SalePrice' : pred_prices})

ayesha_submission.to_csv('submission.csv', index=False)
#sns.distplot(train.loc[:,feature], norm_hist=True, ax = ax1)
