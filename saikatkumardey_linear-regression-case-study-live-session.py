import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')





from sklearn import metrics



import numpy as np



# allow plots to appear directly in the notebook

%matplotlib inline
data = pd.read_csv('../input/advertising.csv/Advertising.csv',index_col=0)

data.head(10)
data['TV'].head(5)
data['radio'].head(5)
data['TV'].values[0]
data.info()
data['TV'].min(), data['TV'].max()
data.describe()
data['TV'].plot(kind='hist');
data['radio'].plot(kind='hist');
data['newspaper'].plot(kind='hist');
data['sales'].plot(kind='hist');
data['newspaper'].plot(kind='box');
data['TV'].plot(kind='box');
data['radio'].plot(kind='box');
data.head()
sns.scatterplot(x="TV", y="sales", data=data);
sns.scatterplot(x="radio", y="sales", data=data);
sns.scatterplot(x="newspaper", y="sales", data=data);
from sklearn.model_selection import train_test_split
data.head()
X_cols = ['TV','radio','newspaper']



X = data[X_cols]

X
y = data['sales']

y.head(10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train.shape
y_train.shape
X_test.shape
y_test.shape
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
X_train.head()
y_train.head()
model.predict(X_train)[:5]
y_pred = model.predict(X_test)
X_test.shape
y_pred
y_test
fig,ax = plt.subplots(figsize=(10,8))



sns.scatterplot(y_test,y_pred,ax=ax);



ax.set_xlabel("Actual Sales");



ax.set_ylabel("Predicted Sales");
from sklearn.metrics import mean_absolute_error
error = mean_absolute_error(y_true = y_test, y_pred = y_pred)
error
mean_sales = y_train.mean()

mean_sales
y_test.shape
len(y_test)
mean_prediction = [mean_sales]* len(y_test)

mean_prediction
error_benchmark = mean_absolute_error(y_true = y_test, y_pred = mean_prediction)



error_benchmark
import joblib
model_filename = "lr_model.joblib"
joblib.dump(model,model_filename)
ls
model_new = joblib.load(model_filename)
model_new
model_new.predict(X_test)
model.coef_