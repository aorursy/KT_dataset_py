import pandas as pd

import numpy as np

import os 
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_excel('/kaggle/input/Reporting Analyst Test.xlsx', sheet_name='Daily Data')

df.head(6)
df.columns =['Date', 'Likes','Engaged users','reach','impressions'] 
df.head()
df.drop(df.index[[0]],inplace=True)
df.head()
df.reset_index().head()
df.reset_index(inplace=True, drop=True)
df.head(6)
df.Date.head()
df.info()
df['Date'] = df['Date'].astype('datetime64[ns]') 
data=df.drop(columns='Date',axis=1)
data.head()
data.describe()
data.isnull().any().sum()
numeric_var_names=[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]

cat_var_names=[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['object']]

print (numeric_var_names)

print (cat_var_names)
def cat_summary(x):

    return pd.Series([x.count(), x.isnull().sum(), x.value_counts()], 

                  index=['N', 'NMISS', 'ColumnsNames'])



cat_summary=data.apply(lambda x: cat_summary(x))
cat_summary
import pandas_profiling

pandas_profiling.ProfileReport(data)
# function to create dummy variable

def create_dummies( df, colname ):

    col_dummies = pd.get_dummies(df[colname], prefix=colname)

    col_dummies.drop(col_dummies.columns[0], axis=1, inplace=True)

    df = pd.concat([df, col_dummies], axis=1)

    df.drop( colname, axis = 1, inplace = True )

    return df
# c_feature in categorical_features

data_cat=df[[ 'Engaged users', 'reach', 'impressions']]

for c_feature in [ 'Engaged users', 'reach', 'impressions']:

    data_cat[c_feature] = data_cat[c_feature].astype('category')

    data_cat = create_dummies(data_cat , c_feature )
data_cat.head()
data_cat.columns
data_new = pd.concat([data_cat, data], axis=1)
data_new.columns
#Splitting the data

feature_columns = data_new.columns.difference( ['Likes'] )

from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split( data_new[feature_columns],

                                                  data_new['Likes'],

                                                  test_size = 0.5,

                                                  random_state = 123 )
from sklearn.ensemble import RandomForestRegressor
radm_clf = RandomForestRegressor( n_estimators=100)

radm_clf.fit( train_X, train_y )
radm_test_pred = pd.DataFrame( { 'actual-y':  test_y,

                            'predicted-x': radm_clf.predict( test_X ) } )
radm_test_pred.head()
importances=list(radm_clf.feature_importances_)
import pandas as pd

feature_importances = pd.DataFrame(radm_clf.feature_importances_,

                                   index = train_X.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances.columns
features=feature_importances.head(4)
features
print (len( train_X ))

print (len( test_X))
feature_cols = ['reach','Engaged_users','impressions']

X = data_new[feature_cols] # Features

y = data_new.Likes # Target variable
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.5,random_state=123)
X.shape
y.shape
import statsmodels.api as sm

lm=sm.OLS(train_y.astype(float),train_X.astype(float)).fit()

print(lm.summary())
y_pred = lm.predict(test_X)

# calculate these metrics by hand!

from sklearn import metrics

import numpy as np

print ('MAE:', metrics.mean_absolute_error(test_y, y_pred))

print ('MSE:', metrics.mean_squared_error(test_y, y_pred))

print ('RMSE:', np.sqrt(metrics.mean_squared_error(test_y, y_pred)))
deta=pd.DataFrame({'Actual-y': test_y, 'Predicted-y': y_pred})

deta.head()
import matplotlib.pyplot as plt 

df1 = deta.head(25)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()