import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
%matplotlib inline
#ls
df_insurance = pd.read_csv('../input/insurance/insurance.csv')
df_insurance.head()
df_insurance.shape
df_insurance.info()
col = ['sex','children','smoker','region']
for i in col :
    print(i)
    print(df_insurance[i].value_counts())
df_insurance['sex'] = df_insurance['sex'].map({"male": 1, "female" : 0})
df_insurance['smoker'] = df_insurance['smoker'].map({"yes":1,"no":0})
df_insurance['region'] = df_insurance['region'].map({"southeast" :0 ,"northwest":1,"southwest":2,"northeast":3})
df_insurance.head()
df_insurance.corr()
plt.figure(figsize=(10,10))
sns.heatmap(df_insurance.corr(),annot=True)
x = df_insurance[['age','bmi','smoker']]
y = df_insurance['charges']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=.25,random_state=42)
lr = LinearRegression()
lr.fit(xtrain,ytrain)
y_pred = lr.predict(xtest)
print(y_pred[:5])

from sklearn.metrics import r2_score,mean_squared_error
print(r2_score(y_pred,ytest))
print(np.sqrt(mean_squared_error(y_pred,ytest)))