# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets import load_boston
boston_dataset=load_boston()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
boston=pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)
boston.head()
boston['MEDV']=boston_dataset.target
boston.head()
boston.describe()
boston.info()
boston.nunique()
boston.isnull().sum()
import matplotlib.pyplot as plt
plt.plot(boston['MEDV'])
plt.show()
from scipy.stats import skew
boston['MEDV'].skew()
(np.log1p(boston['MEDV'])).skew()
correlation_matrix=boston.corr().round(2)
plt.figure(figsize=(10,6))
sns.heatmap(data=correlation_matrix,annot=True)
plt.figure(figsize=(20,5))
features=['LSTAT','RM']
target=boston['MEDV']
for i,col in enumerate(features):
    plt.subplot(1,len(features),i+1)
    x=boston[col]
    y=target
    plt.scatter(x,y,marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
X=pd.DataFrame(np.c_[boston['LSTAT'],boston['RM']],columns=['LSTAT','RM'])
Y=boston['MEDV']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

X
Y
X_train
Y_test
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
lin_model=LinearRegression()
lin_model.fit(X_train,Y_train)
df1=boston.iloc[:,0:13]
df1.head()
y_test_predict=lin_model.predict(X_test)
rmse=(np.sqrt(mean_squared_error(Y_test,y_test_predict)))
R2=r2_score(Y_test,y_test_predict)
print('The model performane for test data')
print('RMSE IS {}'.format(rmse))
print('R2_score is {}'.format(R2))
X=boston.drop(columns=['MEDV','RAD'])
Y=boston['MEDV']
import seaborn as sns
sns.boxplot(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=7)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from  sklearn.preprocessing import StandardScaler
from  sklearn.preprocessing import MinMaxScaler
from  sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
el = StandardScaler()
ms = MinMaxScaler()
rb = RobustScaler()
pw = PowerTransformer()
X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)
# Y_train = pd.DataFrame(Y_train)  
Y_train = Y_train.values.reshape(-1,1)
Y_test = Y_test.values.reshape(-1,1)
Y_train = pw.fit_transform(Y_train)
Y_test = pw.transform(Y_test)
model_lin=LinearRegression()
model_lin.fit(X_train,Y_train)
y_test_predictv=model_lin.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test_predictv,Y_test))

r2=r2_score(y_test_predictv,Y_test)

print('the predicted values has')
print('RMSE={}'.format(rmse))
print('r2 score={}'.format(r2))
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(X_train,Y_train)
#el=StandardScaler()
#X_train=el.fit_transform(X_train)
#X_test=el.transform(X_test)
y_test_predictv=pipe.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test_predictv,Y_test))

r2=r2_score(y_test_predictv,Y_test)

print('the predicted values has')
print('RMSE={}'.format(rmse))
print('r2 score={}'.format(r2))
model_lin=LinearRegression()
model_lin.fit(X_train,Y_train)
y_test_predictv=model_lin.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test_predictv,Y_test))

r2=r2_score(y_test_predictv,Y_test)

print('the predicted values has')
print('RMSE={}'.format(rmse))
print('r2 score={}'.format(r2))
from sklearn.linear_model import Lasso
lasso_model=Lasso(alpha=0.1)
lasso_model.fit(X_train,Y_train)
y_test_predict=lasso_model.predict(X_test)
rmse=(np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)
print('rmse is {}'.format(rmse))
print('r2_score is {}'.format(r2))
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso_reg=Lasso()
parameters = {"alpha": [0.001,0.01,0.1,0.3,0.5,0.8,1,4,9,10,30],
              "fit_intercept": [True, False],
             }
grid=GridSearchCV(estimator=lasso_reg,param_grid=parameters,cv=2,n_jobs=-1)
grid.fit(X_train,Y_train)
grid.best_params_
grid.best_score_
L_model = Lasso(alpha=0.3)
L_model.fit(X_train, Y_train)

y_test_predict = L_model.predict(X_test)

rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
from sklearn.linear_model import Ridge
R_model=Ridge()
R_model.fit(X_train, Y_train)
y_test_predict=R_model.predict(X_test)
rmse=(np.sqrt(mean_squared_error(Y_test,y_test_predict)))
r2=r2_score(Y_test,y_test_predict)
print('rmse is {}'.format(rmse))
print('r2_score is {}'.format(r2))
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
lasso_reg = Ridge()
parameters = {"alpha": [0.001,0.01,0.1,0.3,0.5,0.8,1,4,9,10,30],
              "fit_intercept": [True, False],
             }
grid = GridSearchCV(estimator=lasso_reg, param_grid = parameters, cv = 2, n_jobs=-1)
grid.fit(X_train, Y_train)
grid.best_params_
grid.best_score_
R_model=Ridge()
R_model.fit(X_train,Y_train)

y_test_predict = R_model.predict(X_test)

rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
from sklearn.linear_model import ElasticNet
E_model = Ridge()
E_model.fit(X_train, Y_train)

y_test_predict = E_model.predict(X_test)

rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

r2 = r2_score(Y_test, y_test_predict)

print("The model performance")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
#final = L*0.1 + r*0.6 + e*0.3
#rmse = (np.sqrt(mean_squared_error(Y_test, final)))
#r2 = r2_score(Y_test, final)

#print("The model performance for testing set")
#print('RMSE is {}'.format(rmse))
#print('R2 score is {}'.format(r2))
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
lasso_reg = ElasticNet()
parameters = {"alpha": [0.001,0.01,0.1,0.3,0.5,0.8,1,4,9,10,30],
              "l1_ratio":[0.1,0.3,0.5,0.8],
              "fit_intercept": [True, False],
             }
grid = GridSearchCV(estimator=lasso_reg, param_grid = parameters, cv = 7, n_jobs=-1)
grid.fit(X_train, Y_train)
grid.best_params_
elastica_model = ElasticNet(alpha=0.001,
                           l1_ratio=0.8)
elastica_model.fit(X_train,Y_train)

y_test_predict=elastica_model.predict(X_test)

rmse=np.sqrt(mean_squared_error(Y_test,y_test_predict))

r2=r2_score(Y_test,y_test_predict)

print('RMSE={}'.format(rmse))

print('r2 score={}'.format(r2))
Y_test = pw.inverse_transform(Y_test)
y_test_predict = pw.inverse_transform(y_test_predictv)
from matplotlib.pyplot import plot
plot(y_test_predict, label='Pred')
plot(Y_test, label='Actual')
plt.legend(loc='best')
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=7)
y_pred= np.mean(y_test_predict, 0)
y_pred
from sklearn.model_selection import KFold 
score = []
kfold = KFold(n_splits=4, random_state=234, shuffle=True)
for train, test in kfold.split(X):
    x_train, x_test = X.iloc[train], X.iloc[test]
    y_train, y_test = Y[train], Y[test]
    
    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)
    y_test_predict = lin_model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    print(rmse)
    score.append(rmse)
        
        
average_score = np.mean(score)
print('The average RMSE is ', average_score)
from sklearn.datasets import load_wine
wine_dataset = load_wine()
wine = pd.DataFrame(wine_dataset.data, columns=wine_dataset.feature_names)

wine['quality'] = wine_dataset.target
wine.head()
wine = wine[wine.quality !=2]
X = wine.drop(columns='quality')
Y = wine['quality']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=32)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, Y_train)
pred = model.predict(X_test)
from sklearn.metrics import   f1_score, accuracy_score
accuracy_score(pred, Y_test)
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, random_state=27, shuffle=True)
score = []
for train, test in kfold.split(X, Y):
    X_train, X_test = X.iloc[train], X.iloc[test]
    Y_train, Y_test = Y[train], Y[test]
    
    model = LogisticRegression(max_iter = 10000)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    f1 = accuracy_score(pred, Y_test)
    print(f1)
    score.append(f1)
        
average_score = np.mean(score)
print('The average accuracy is ', average_score) 