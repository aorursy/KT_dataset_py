#DPF = DiabetesPedigreeFunction
import numpy as np
import pandas as pd

data = pd.read_csv('diabetes.csv')
data.head()
data.shape
x=data.iloc[:,:-1].values
y=data.iloc[:,8].values
data.isnull().sum()
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy="mean")
x[:,2:7]=imputer.fit_transform(x[:,2:7])
from pandas import DataFrame
z=DataFrame(x)
z.isnull().sum()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LinearRegression
re = LinearRegression()
re.fit(x_train,y_train)
pred = re.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,pred)
#x=np.append(arr=np.ones((768,1)).astype(int), values=data,axis=1)
y_test