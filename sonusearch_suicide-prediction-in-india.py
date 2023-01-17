import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/Suicides in India 2001-2012.csv")
df = df.drop(df[(df['State'] == 'Total (All India)') |  (df['State'] =='Total (States)') |  (df['State'] =='Total (Uts)')].index)
df_dowry_dispute = df[(df['Type_code'] == 'Causes') & (df['Type']=='Dowry Dispute')]
df_dowry_dispute_female = df_dowry_dispute[(df_dowry_dispute['Gender'] == 'Female') & (df_dowry_dispute['Total'] >0)]
df_dowry_dispute_female.isnull().sum()
df_dowry_dispute_female_total_suicides = df_dowry_dispute_female['Total'].sum()
df_dowry_dispute_female_total_suicides
df_female_suicide_by_state = df_dowry_dispute_female.groupby('State')['Total'].sum()
df_female_suicide_by_state
df_dowry_dispute_female.head()
X = df_dowry_dispute_female.iloc[:,:-5].values
#X = df_dowry_dispute_female[['State','Year']]
X.shape
len(X)
y = df_dowry_dispute_female.iloc[:,6].values
len(y)
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
from numpy import array
from numpy import argmax
X[:,0] = labelEncoder.fit_transform(array(X[:,0]))
X[:,1] = labelEncoder.fit_transform(array(X[:,1]))
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
##Remember OneHotEncoder is required only for the State and not for the year. State is categorical where one state can not be greater or less than other state.
##On the contrary, year is progressive and can be greater or less than other year. So, OneHotCoding is not applicable on year
onehotencoder_x = OneHotEncoder(categorical_features = [0], sparse=True)
X = onehotencoder_x.fit_transform(X).toarray()
# Encoding the Dependent Variable
#### ?? Do we need to use LabelEncoder on y ??
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

# Dummy variable trap
X = X[:, 1:]
X
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
## ?? What is the difference between StandardScaler and MinMaxScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
len(y_test)
# if Sparse = True then X_train[0].toarray() Sparse is False when using with Keras. Sparse not compatible with Keras
len(X_test)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
y_pred
y_test
import statsmodels.formula.api as sm 
X = np.append(arr=np.ones((662, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

