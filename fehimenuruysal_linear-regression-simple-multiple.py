# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error , classification_report

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/advertising-dataset/advertising.csv")

df.head()
df.describe().T
df.info()
sns.heatmap(df.corr() , cmap="RdPu" , annot = True , linewidth = .7);
figure , ax = plt.subplots(3 , figsize = (8,5))

plt_tv = sns.boxplot(df["TV"] , ax = ax[0])

plt_radio = sns.boxplot(df["Radio"] , ax = ax[1])

plt_newspaper = sns.boxplot(df["Newspaper"] , ax = ax[2])

plt.tight_layout()
X = df[["TV"]]

y = df[["Sales"]]
plt.scatter(x = X , y = y , color = "red")

plt.xlabel("TV" , fontsize = 8)

plt.ylabel("Sales" , fontsize = 8)

plt.show()
sns.jointplot(x = "TV" , y = "Sales" , data = df , kind="reg");
model = LinearRegression().fit(X,y)
model.intercept_
model.coef_
model.score(X,y)
y_pred = model.predict(X)
plt.scatter(X , y , color = "red")

plt.scatter(X , y_pred , color = "blue");
real_y_values = y[:5]

predict_y_values = pd.DataFrame(y_pred[:5])

MSE = mean_squared_error(real_y_values , predict_y_values)

print("MSE = {}".format(MSE))
X_test = X[:5]

results = pd.DataFrame(model.predict(X_test), columns =["Pred_y"])
predict_real_y_df = pd.concat([results , real_y_values[:5]] , axis=1)

predict_real_y_df
X = df.iloc[:,:3]

y = df.iloc[:,3]
multi_linear_model = LinearRegression().fit(X,y)
multi_linear_model.intercept_
multi_linear_model.coef_
multi_linear_model.score(X,y)
y_pred = multi_linear_model.predict(X)
RMSE = np.sqrt(mean_squared_error(y, y_pred))

RMSE
pred_y_df = pd.DataFrame(y_pred[:5] , columns =["Y Pred"])

result_df = pd.concat([pred_y_df , y[:5]] , axis=1)

result_df
df.head(1)
X_pre = [[230.1 , 37.8 , 69.2]]

multi_linear_model.predict(X_pre)