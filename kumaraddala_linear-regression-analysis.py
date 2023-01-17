# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt # Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Reading csv file
df=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
#printing first 5 rows of data frame
df.head()
df.shape
df.dtypes
df.columns
X=df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR ', 'CGPA', 'Research']]
y=df['Chance of Admit ']
import matplotlib.pyplot as plt
plt.scatter(X['GRE Score'],y)
plt.scatter(X['TOEFL Score'],y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=14)
X_tested=X_test[['GRE Score']]
X_trained=X_train[['GRE Score']]
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_trained,y_train)
yhat=LR.predict(X_tested)
print(np.sqrt(mean_squared_error(y_test,yhat)))#out of sample accuracy
print(r2_score(y_test,yhat))
yhated=LR.predict(X_trained)
plt.scatter(X_trained,y_train,color='blue')
plt.plot(X_trained,yhated,color='red')
plt.title('Predicting College Admission')
plt.xlabel('GRE Score')
plt.ylabel('Chance of Admission')
plt.show()
X_test1=X_test[['GRE Score','CGPA','University Rating']]
X_train1=X_train[['GRE Score','CGPA','University Rating']]
mlr=LinearRegression()
mlr.fit(X_train1,y_train)
yhatmul=mlr.predict(X_test1)
print(np.sqrt(mean_squared_error(y_test,yhatmul)))#out of sample accuracy
print(r2_score(y_test,yhatmul)) #general
data1=df[['GRE Score','CGPA','Chance of Admit ']]
data1=data1.rename(columns={'GRE Score':'GRE','Chance of Admit ':'Chance_of_Admit'})
data1.head()
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
model = smf.ols(formula='Chance_of_Admit  ~ GRE + CGPA', data=data1)
results_formula = model.fit()
results_formula.params
x_surf, y_surf = np.meshgrid(np.linspace(data1.GRE.min(), data1.GRE.max(), 100),np.linspace(data1.CGPA.min(), data1.CGPA.max(), 100))
onlyX = pd.DataFrame({'GRE': x_surf.ravel(), 'CGPA': y_surf.ravel()})
fittedY=results_formula.predict(exog=onlyX)
fittedY=np.array(fittedY)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data1['GRE'],data1['CGPA'],data1['Chance_of_Admit'],c='red', marker='o', alpha=0.5)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
ax.set_xlabel('GRE')
ax.set_ylabel('CGPA')
ax.set_zlabel('Chance_of_Admit')
plt.show()
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
Xs=ss.fit_transform(X)
from sklearn.model_selection import train_test_split
X_trains,X_tests,y_trains,y_tests=train_test_split(Xs,y,test_size=0.2,random_state=14)
mlrs=LinearRegression()
mlrs.fit(X_trains,y_trains)
yhats=mlrs.predict(X_tests)
print(np.sqrt(mean_squared_error(y_tests,yhats)))#out of sample accuracy
print(r2_score(y_tests,yhats)) #statndardizatin
from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
Xm=mm.fit_transform(X)
from sklearn.model_selection import train_test_split
X_trainm,X_testm,y_trainm,y_testm=train_test_split(Xm,y,test_size=0.2,random_state=14)
mlrm=LinearRegression()
mlrm.fit(X_trainm,y_trainm)
yhatm=mlrm.predict(X_testm)
print(np.sqrt(mean_squared_error(y_testm,yhatm)))#out of sample accuracy
print(r2_score(y_testm,yhatm)) #Normalization
from sklearn.linear_model import Ridge
RR=Ridge(alpha=0.01)
RR.fit(X_train1,y_train)
yhatrr=RR.predict(X_test1)
print(np.sqrt(mean_squared_error(y_test,yhatrr)))#out of sample accuracy
print(r2_score(y_test,yhatrr))
from sklearn.linear_model import Lasso
LR=Lasso(alpha=0.01)
LR.fit(X_train1,y_train)
yhatlr=RR.predict(X_test1)
print(np.sqrt(mean_squared_error(y_test,yhatlr)))#out of sample accuracy
print(r2_score(y_test,yhatlr))
from sklearn.linear_model import ElasticNet
ER=ElasticNet(alpha=0.01)
ER.fit(X_train1,y_train)
yhater=ER.predict(X_test1)
print(np.sqrt(mean_squared_error(y_test,yhater)))#out of sample accuracy
print(r2_score(y_test,yhater))
