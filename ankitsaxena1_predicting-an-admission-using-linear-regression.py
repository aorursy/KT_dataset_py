import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import mean_absolute_error
dataset = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
dataset.info()
dataset.describe()
dataset = dataset.drop('Serial No.', axis=1)
sns.pairplot(data=dataset, kind='reg')
dataset.corr(method='pearson')
y=dataset['Chance of Admit ']
dataset=dataset.drop('Chance of Admit ',axis=1)

def create_stats_fit_model(x):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
    regressor_OLS = sm.OLS(endog=y_train,exog=x_train).fit()
    print(regressor_OLS.summary())
    predictions =  regressor_OLS.predict(x_test)
    print("Mean absolute error is ",mean_absolute_error(y_test,predictions))
dataset_temp = np.append(arr = np.ones((500,1)).astype(int), values = dataset ,axis=1)
dataset_temp1 = dataset_temp[:, [0,1,2,3,4,5,6,7]]
create_stats_fit_model(dataset_temp1)
dataset_temp1 = dataset_temp[:, [0,1,2,3,5,6,7]]
create_stats_fit_model(dataset_temp1)
dataset_temp1 = dataset_temp[:, [0,1,2,5,6,7]]
create_stats_fit_model(dataset_temp1)
