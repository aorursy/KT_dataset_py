# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

plt.style.use("seaborn-whitegrid")       

import pandas_profiling as pp 



import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv")
df.head()
df.shape
df.info()
df.describe().T


sns.scatterplot(x = "YearsExperience", y = "Salary",hue = "YearsExperience",data = df);

sns.pairplot(df,kind= "reg");
sns.jointplot(x="YearsExperience", y="Salary",color = 'blue',kind = "reg", data=df,size = 5);
X = df['YearsExperience']

y = df['Salary']
import statsmodels.api as sm

X = sm.add_constant(X)

lm = sm.OLS(y, X)

model = lm.fit()

model.summary()



#You can use stats model to learn statistical details (such as p_value, model parameters ..)

#You can create a model with using stats model in order to predict values. 

#the stats model is mostly used to look at statistical details, to understand the data and model.
print("f_pvalue: ", "%.4f" % model.f_pvalue)
model.params
STM = "Salary = " +  str("%.4f" % model.params[0]) + " + " + "YearsExperience*" + str("%.4f" % model.params[1])

print(STM)
g = sns.regplot(df["YearsExperience"], df["Salary"], ci=None, scatter_kws={'color':'r', 's':9});

g.set_title(" Salary = 25321.58 + YearsExperience*9423.82");

g.set_ylabel("Salary");

g.set_xlabel("YearsExperience");

#plt.xlim(-10,310)

#plt.ylim(bottom=0)
X = df['YearsExperience']

y = df['Salary']


from sklearn.model_selection import train_test_split,cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 42)
X_train.shape
X_train = X_train[:,np.newaxis]

X_test = X_test[:,np.newaxis]



#numpy.newaxis is used to increase the dimension of the existing array by one more dimension, when used once.
X_train.shape



#After used numpy.newaxis, shape of X_train was change. In order to use fit(), your feature shape must be column vector.
X_test.shape
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X_train,y_train)
model.intercept_
model.coef_ 
LR = "Salary = " +  str("%.2f" % model.intercept_ ) + " + YearsExperience" + "*" + str("%.2f" % model.coef_)

print(LR)
y_pred_train = model.predict(X_train)

y_pred_test = model.predict(X_test)
ax1 = sns.distplot(model.predict(X_train), hist=False, color="r", label="Actual Train Value")

sns.distplot(y_train, hist=False, color="b", label="Fitted Train Values" , ax=ax1);
ax1 = sns.distplot(model.predict(X_test), hist=False, color="r", label="Actual Test Value")

sns.distplot(y_test, hist=False, color="b", label="Fitted Test Values" , ax=ax1);
trainscr = model.score(X_train, y_train)

testscr = model.score(X_test, y_test)



print("train score:{}\ntest score:{}".format(trainscr,testscr))
from sklearn.metrics import mean_squared_error, r2_score



rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))





rmse1 = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))





print("train rmse:{}\ntest rmse:{}".format(rmse,rmse1))
print(STM)

print(LR)