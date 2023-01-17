# Import relevant libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings 
warnings.filterwarnings('ignore')
raw_data=pd.read_csv('../input/boston-housing-dataset/HousingData.csv')    # Load Data
raw_data.head()
raw_data.describe(include='all')
raw_data.info()
data=raw_data.copy()
data.dropna(axis=0,inplace = True)     # Drop null values
data.isnull().sum()
q1=data.quantile(0.25)
q3=data.quantile(0.75)
iqr=q3 - q1 
iqr
data_clean=data[~((data < (q1 - 1.5 * iqr)) | (data > (q3 + iqr))).any(axis=1)]
data_clean.skew()
sns.pairplot(data_clean)
log_medv=np.log(data_clean['MEDV'])
data_clean['log_MEDV'] = log_medv
data_clean.head()
data_clean=data_clean.drop(['MEDV'],axis = 1)
data_clean.corr().abs() # Analysing Correlation 
from statsmodels.stats.outliers_influence import variance_inflation_factor  # Checking for Multi-collinearity
variables = data_clean[['PTRATIO','INDUS','AGE','RM','LSTAT']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns
vif
data_clean.drop('PTRATIO',axis=1,inplace= True)
data_pre=data_clean.copy()
data_pre = data_clean[['INDUS','AGE','RM','LSTAT','log_MEDV']]
targets = data_pre['log_MEDV']
inputs = data_pre.drop(['log_MEDV'],axis = 1)
from sklearn.preprocessing import StandardScaler    # Scaling Data
scaler=StandardScaler()
scaler.fit(inputs)
inputs_sc = scaler.transform(inputs) 
from sklearn.feature_selection import f_regression     # Checking p_values
f_regression(inputs_sc,targets)
p_values = f_regression(inputs_sc,targets)[1]
p_values.round(3)
from sklearn.model_selection import train_test_split    # Splitting data in Train - Test
x_train, x_test, y_train, y_test = train_test_split(inputs_sc, targets, test_size=0.2, random_state=365)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
y_hat = reg.predict(x_train)
sns.scatterplot(x=y_train,y=y_hat)
sns.distplot(y_train - y_hat)
reg.score(x_train,y_train)    # R2 
def adj_r2(x,y):
    r2 = reg.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2
adj_r2(inputs_sc,targets)
reg.intercept_
reg.coef_
y_hat_test = reg.predict(x_test)
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
df_pf.head()
df_pf['Target'] = np.exp(y_test)
df_pf
y_test = y_test.reset_index(drop=True)
y_test.head()
df_pf['Target'] = np.exp(y_test)
df_pf
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf
pd.options.display.max_rows = 999
# Moreover, to make the dataset clear, we can display the result with only 2 digits after the dot 
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Finally, we sort by difference in % and manually check the model
df_pf.sort_values(by=['Difference%'])