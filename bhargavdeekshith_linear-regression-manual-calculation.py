import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.DataFrame({'bmi':[4.0,5.5,6.8,7.2,7.8,9.8,9.7,8.8,11.0,13.0],'glu':[60,135,90,175,240,220,300,370,360,365]})
df
plt.scatter(x = 'bmi',y = 'glu',data = df)
plt.xlabel('BMI')
plt.ylabel('glucose')
ß1 = np.cov(df['bmi'],df['glu'],ddof = 1)/np.var(df['bmi'],ddof = 1)
ß1
beta1 = np.sum((df['bmi']-np.mean(df['bmi']))*(df['glu']-np.mean(df['glu'])))/np.sum((df['bmi']-np.mean(df['bmi']))**2)
beta1
ß0 = (np.mean(df['glu'])) - (beta1 * np.mean(df['bmi']))
ß0
glu_pred = ß0 + beta1 * df['bmi']
glu_pred
plt.scatter(x = 'bmi',y = 'glu',data = df)
plt.plot(df['bmi'],glu_pred,color = 'red')
plt.xlabel('BMI')
plt.ylabel('glucose')
mse = np.sum((df['glu'] - glu_pred)**2)/10
mse
sse = np.sum((df['glu'] - glu_pred)**2)
sse
rmse = np.sqrt(mse)
rmse
average = np.mean(df['glu'])
mse_of_base_line_model = np.sum((df['glu'] - average)**2)/10 
rmse_of_base_line_model = np.sqrt(mse_of_base_line_model)
print(average)
print(mse_of_base_line_model)
print(rmse_of_base_line_model)
from sklearn.linear_model import LinearRegression
x = df[['bmi']] #bmi is a series. Model expects a dataframe. So it is passed inside two brackets.
y = df['glu'] #dependent variable
model = LinearRegression()
model.fit(x,y)
model.coef_
model.intercept_
Glu_pred = model.predict(x)
Glu_pred
R=np.corrcoef(df['glu'],glu_pred)
R
R2 =  0.86584815**2
print(R2)
from sklearn.metrics import r2_score
r2_score(df['glu'],Glu_pred)