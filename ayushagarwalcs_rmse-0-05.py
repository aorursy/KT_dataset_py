import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd

filename = "/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv"

df = pd.read_csv(filename)

df.head()
df.rename(columns={"Serial No.":"SerialNo","Chance of Admit ":"ChanceOfAdmit", "LOR ":"LOR"}, inplace=True)
df.SerialNo.nunique()
for variable in df:

    plt.scatter(df["ChanceOfAdmit"],df[variable])
corr_matrix = df.corr()

corr_matrix
plt.figure(figsize=(20,30))

i=1

for variable in df:

    plt.subplot(5,4,i)

    plt.scatter(df["ChanceOfAdmit"], df[variable])

    plt.title(variable)

    i=i+1
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



X = df[['CGPA','SOP','LOR']]

Y = df['ChanceOfAdmit']

X_train=X[:400]

Y_train=Y[:400]

X_test=X[400:]

Y_test=Y[400:]
import math
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators = 250, random_state = 20) 

forest.fit(X_train, Y_train)

Y_pred=forest.predict(X_test)
mse=mean_squared_error(Y_pred, Y_test)

rmse = math.sqrt(mse)

rmse
model = LinearRegression(fit_intercept=True, normalize=True).fit(X_train, Y_train)

predictions= model.predict(X_test)

mse=mean_squared_error(predictions, Y_test)

rmse = math.sqrt(mse)

rmse