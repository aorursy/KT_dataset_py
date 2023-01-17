# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as  pd
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
matplotlib.get_backend()
import numpy as np
import scipy as scp
import sklearn as sk
import seaborn as sns
import random
import scipy.stats as stats

%matplotlib inline
#NFA DATABASE----------------------------------------------------------------------------------------------------
EFland=pd.read_csv("../input/NFA 2018.csv")
#Remove unknown values
EFlandclean=EFland.dropna()
#Filter the data
t=EFlandclean[EFlandclean['record']=="EFProdTotGHA"]
#Group by continent and year
EF3=t.groupby(['UN_region','year']).sum()
EF3.describe()
fb=EF3.boxplot(figsize=(16,12))
f=fb.get_figure()
f.savefig('box.png')
#CORRELATION BETWEEN THE DIFFERENT VARIABLES
corr = EF3.corr()
plt.figure(figsize=(16,12))
sns_plot=sns.heatmap(corr,annot=True,fmt='.2f',cmap='coolwarm')
fig = sns_plot.get_figure()
fig.savefig("correlation.png")
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#Independent variables
#X = EF3.iloc[:, [0,1,2,7,8]]. 'crop_land', 'grazing_land', 'forest_land',values GDP,population
X = EF3.iloc[:, [0,1,2,7,8]].values
#X = EF3.iloc[:, [0,1,2]].values
print("The independent variables are:",EF3.iloc[:, [0,1,2,7,8]].columns )
#Target variable
Y = EF3.iloc[:, [5]].values
# Diviser notre dataset en training set et test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
# Feature Scaling <=> Normalisation 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
# Fittage du modèle de régression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prédiction des résultats
y_pred = regressor.predict(X_test)

# Ajouter une constante 1
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((len(EF3),1)).astype(int), values = X, axis = 1)
X_opt = X[:, :]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
col1 = EF3.apply ( lambda row : ((row["Percapita GDP (2010 USD)"])*row["population"]), axis = 1 ) 
EF3["Total_GDP"] = col1
corr =EF3.corr()
plt.figure(figsize=(16,12))
sns_plot=sns.heatmap(corr,annot=True,fmt='.2f',cmap='coolwarm')
fig = sns_plot.get_figure()
fig.savefig("correlationFinal.png")
