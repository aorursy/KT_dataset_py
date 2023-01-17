
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

df = pd.read_csv('../input/superconductors/superconductor.csv')
df.describe()
df.isnull().sum()
df.mean()
df.info()
corr = df.corr()
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=2)

sc = StandardScaler()
sc.fit_transform(x_train)
sc.fit_transform(x_test)

lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)

df = pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
print(df)

print("Mean squared error is {}".format(mse(y_test,y_pred)))
print("rmse value is {}".format(np.sqrt(mse(y_test,y_pred))))

print(lr.score(x_train,y_train))

saved_model = pickle.dumps(lr) 
  
# Load the pickled model 
classifier_from_pickle = pickle.loads(saved_model) 
  
# Use the loaded pickled model to make predictions 
classifier_from_pickle.predict(x_test)
