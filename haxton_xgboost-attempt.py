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

train = pd.read_csv("../input/train.csv")
sample = pd.read_csv("../input/sample_submission.csv")
test = pd.read_csv("../input/test.csv")

#Clean - Remove all NaN
train_cleaned = train.fillna(0)
test_cleaned = test.fillna(0)

#categorising - one hot encoding using get)dummies
train_category = train_cleaned
train_category[train_category.select_dtypes("object").columns.values] = train_cleaned.select_dtypes("object").astype("category")

####Check Data Types
#train_category.dtypes
#train_category.info()

pd.get_dummies(train_category).head(5)



# library & dataset
import seaborn as sns
import matplotlib.pyplot as plt

df = train_category.select_dtypes(include=[np.number])
df = df.drop(['Id'], axis=1)
train_norm = train_category.copy()

#Normalising numerical numbers
from sklearn import preprocessing
x = df.values #returns a numpy array
max_abs_scaler = preprocessing.MaxAbsScaler()
x_scaled = max_abs_scaler.fit_transform(x)
norm = pd.DataFrame(x_scaled)
norm.columns = train_category.select_dtypes(include=[np.number]).drop(['Id'], axis=1).columns.values
train_norm[train_norm.select_dtypes(include=[np.number]).drop(['Id'], axis=1).columns.values] = norm
train_norm.head(5)
#feature engineering

from sklearn.ensemble import RandomForestRegressor

df = train_norm
df = df.drop(['Id','SalePrice'], axis=1)
model = RandomForestRegressor(random_state=1, max_depth=10)

#One-hot encoding for categorical data
df=pd.get_dummies(df)
model.fit(df,train_norm.SalePrice)

features = df.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-15:]  # top 15 features
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

train_featured = train_category[features[indices]]

###Future testing
#find optimal number of features to be selected

#Are variables independent?
#Correlation map in the top 15 features
train_featured = train_category[features[indices]]
#train_featured['SalePrice'] = train_norm['SalePrice'] 
corr = train_featured.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


#GrlivArea is second important feature to consider
#we see that GarageArea, GarageCars, and GarageYrBlt are significantly correlated and we can easily see why
#So ar: YearBuilt & OverallQual, GrLivArea & TotRmsAbvGrd, TotalBsmtSF
#To remove unwanted features out of each group
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer # fills nan

#Persistent dictionary instead of one-hot encoding
from collections import defaultdict #dictionary for labelled
from sklearn.preprocessing import LabelEncoder 
d = defaultdict(LabelEncoder)

#Split 
y = train_category.reindex(range(0,1461)).SalePrice.fillna(0) # Y from encoding
fit = train.reindex(range(0,1461)).select_dtypes(exclude=['number']).drop(['Alley','MSZoning'],axis=1).astype(str).apply(lambda x: d[x.name].fit_transform(x)) #encoding
#X = pd.concat([fit.select_dtypes(exclude=['object']), train_category.reindex(range(0,1461)).select_dtypes(exclude=['category']).drop('Id',axis=1)], axis=1, sort=False).fillna(0) #join tables
X = train_featured.select_dtypes(exclude=['object']).reindex(range(0,1461)).fillna(0)
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

#Imputer helps to quickly fill NaN with various strategy, for our case
my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)
#XGBoost 
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000,learning_rate=0.05)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)

#To run different machine learning techniques in future
# make predictions
predictions = my_model.predict(test_X)

#get Mean Absolute Error
from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

#plt.plot(predictions, 'bo')
#plt.plot(test_y, 'g^')

#get Mean Absolute Error
plt.plot(pd.DataFrame((predictions - test_y)).abs())

###Previous test result
#16817.41171875 for 14 features 
#17683.63653681507 for get_dummies all features (304)
#18735.594625170765 15 feature

#for submission
##NOTE: replaced with dictionary
#final_test = test.copy().drop(['Alley','MSZoning'],axis=1)
#final_test.Utilities = final_test.Utilities.fillna(final_test.Utilities.value_counts().idxmax())
#final_test.Exterior1st = final_test.Exterior1st.fillna(final_test.Exterior1st.value_counts().idxmax())
#final_test.Exterior2nd = final_test.Exterior2nd.fillna(final_test.Exterior1st.value_counts().idxmax())
#final_test.MasVnrType = final_test.MasVnrType.fillna(0)
#for i in final_test.select_dtypes(exclude=['number']):
#    print(i)
#    if(final_test[i].isna().sum()):
#        final_test[i] = final_test[i].fillna(0)#final_test[i].value_counts().idmax())
#final_test = final_test.fillna(0)
#final_fit = final_test.select_dtypes(exclude=['number']).astype(str).apply(lambda x: d[x.name].transform(x)) #encoding
#final_X = pd.concat([fit.select_dtypes(exclude=['object']), final_test.select_dtypes(exclude=['object'])], axis=1, sort=False) #join tables

final_fit = test.select_dtypes(exclude=['number']).drop(['Alley','MSZoning'],axis=1).astype(str).apply(lambda x: d[x.name].transform(x)) #encoding
#final_X = pd.concat([fit.select_dtypes(exclude=['object']), test.select_dtypes(exclude=['object']).fillna(0)], axis=1, sort=False) #join tables
X = test[train_featured.select_dtypes(exclude=['object']).columns.values].fillna(0)


final = my_imputer.transform(X)
predictions = my_model.predict(final)

#Test Data has missing variables, "Alley" and "MSZoning". To implement to fill NaN in future
submission = pd.concat([test['Id'],pd.DataFrame(predictions)],axis=1, sort=False)
submission.columns = ['Id','SalePrice']
#submission = submission[:-2].astype(int)
submission.Id = submission.Id.astype(int)
submission.to_csv('submission2.csv', index=False)
submission.tail()

