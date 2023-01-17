### Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#### read in data
df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv',index_col='Serial No.')
df.head()
df.isnull().sum()####no missing values
df.info()
### pairplot
sns.pairplot(df)
sns.heatmap(df.corr(),annot=True)
### Gre score is out of 340
### TOEFL is out of 120
df['Total_Perc_Score'] = ((df['GRE Score'] + df['TOEFL Score'])/(340+120))
df.head()
sns.heatmap(df.drop(['GRE Score','TOEFL Score'],axis=1).corr(),annot=True)
### Research
df['Research'].value_counts()
### Let us convert 0 to -1
# df['Research'] = df['Research'].replace(0,-1)
### Research
df['Research'].value_counts()
## SOP+LOR TOTAL
df['SOP_LOR_Total'] = ((df['SOP'] + df['LOR ']) /10)
df.head()
df['Total_Perc_Score']=df['Total_Perc_Score'].apply(lambda x : np.round(x,2))
sns.heatmap(df[['SOP_LOR_Total','Total_Perc_Score','Research','University Rating']].corr(),annot=True)
## As we can see, multicollinearity has been drasitcally reduced
df.head()
df.columns
df_input = df[['University Rating','CGPA',
       'Research', 'Chance of Admit ', 'Total_Perc_Score', 'SOP_LOR_Total']]
df_input.head()
#### Arrays
X = df_input.drop('Chance of Admit ',axis=1).values
y = df['Chance of Admit '].values
X.shape
y.shape
## Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
linear_regressor = LinearRegression()
rfc_regressor = RandomForestRegressor()
sgd_regressor = SGDRegressor()
np.mean(cross_val_score(linear_regressor,X_train,y_train,cv=5))
np.mean(cross_val_score(rfc_regressor,X_train,y_train,cv=5))
np.mean(cross_val_score(sgd_regressor,X_train,y_train,cv=5))
###
from sklearn.model_selection import RandomizedSearchCV
ridge = Ridge()
np.mean(cross_val_score(ridge,X_train,y_train,cv=5))
linear_regressor.fit(scaled_X_train,y_train)
predictions_linear = linear_regressor.predict(scaled_X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error
print(mean_absolute_error(y_test,predictions_linear))
print((mean_squared_error(y_test,predictions_linear)**(1/2)))
#Mean absolute error = 0.04
#rmse =0.06 

#linear models are the best!