# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import seaborn as sns

from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/fish-market/Fish.csv')
print(plt.style.available)
plt.style.use('ggplot')
data.head()
data.columns = ['Species', 'Weight', 'SL', 'FL', 'TL', 'BD', 'BT']
data.head()
data.info()
print(str('Is there any NaN value in the dataset: '), data.isnull().values.any())
data.describe()
sp = data['Species'].value_counts()
sp = pd.DataFrame(sp)
sp.T
sns.barplot(x=sp.index, y=sp['Species']);
plt.xlabel('Species')
plt.ylabel('Counts of Species')
plt.show()
fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(data[["Weight", "SL", "FL", "TL", "BD", "BT",]].corr(), annot = True)
plt.show()
def corr(species):
    data1 = data[data['Species'] == species]
    fig, ax = plt.subplots(figsize=(5,5)) 
    sns.heatmap(data1[["Weight", "SL", "FL", "TL", "BD", "BT",]].corr(), annot = True)
    plt.title("Correlation heat map of {} ".format(species))
    plt.show()
species_list = list(data['Species'].unique())
for s in species_list:
    corr(s)
g = sns.pairplot(data, kind='scatter', hue='Species')
g.fig.set_size_inches(10,10)
def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers
data.loc[detect_outliers(data,["Weight", "SL", "FL", "TL", "BD", "BT"])]
df_pike = data[data['Species'] == 'Pike']
df_pike.loc[detect_outliers(df_pike,["Weight", "SL", "FL", "TL", "BD", "BT"])]
species_list = list(data['Species'].unique())
print(species_list)
df_s = data[data['Species'] == 'Bream']
df_s.loc[detect_outliers(df_s,["Weight", "SL", "FL", "TL", "BD", "BT"])]
df_s = data[data['Species'] == 'Roach']
df_s.loc[detect_outliers(df_s,["Weight", "SL", "FL", "TL", "BD", "BT"])]
df_s = data[data['Species'] == 'Whitefish']
df_s.loc[detect_outliers(df_s,["Weight", "SL", "FL", "TL", "BD", "BT"])]
df_s = data[data['Species'] == 'Parkki']
df_s.loc[detect_outliers(df_s,["Weight", "SL", "FL", "TL", "BD", "BT"])]
df_s = data[data['Species'] == 'Perch']
df_s.loc[detect_outliers(df_s,["Weight", "SL", "FL", "TL", "BD", "BT"])]
df_s = data[data['Species'] == 'Smelt']
df_s.loc[detect_outliers(df_s,["Weight", "SL", "FL", "TL", "BD", "BT"])]
data1 = data.drop([35, 54, 157,158])
data1.info()
data1.columns[data1.isnull().any()]
data1.head()
df_bream = data1[data1['Species'] == 'Bream']

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

x = df_bream.TL.values.reshape(-1,1)
y = df_bream.Weight.values.reshape(-1,1)

linear_reg.fit(x,y)
y_head = linear_reg.predict(x)

plt.scatter(df_bream.TL,df_bream.Weight)
plt.plot(x,y_head,color= "black")
plt.xlabel("Total Length")
plt.ylabel("Weight")
plt.show()
print('The weight of a 36 cm Bream is: ', linear_reg.predict([[36]]), 'grams')
y = df_bream['Weight'] # Dependant Var
X = df_bream.iloc[:,[4,5]]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print('Samples in the test and train datasets are:')
print('X_train: ', np.shape(X_train))
print('y_train: ', np.shape(y_train))
print('X_test: ', np.shape(X_test))
print('y_test: ', np.shape(y_test))
ML_reg = LinearRegression()
ML_reg.fit(X_train, y_train)
print('y = ' + str('%.2f' % ML_reg.intercept_) + ' + ' + str('%.2f' % ML_reg.coef_[0]) + '*X1 ' + ' + ' + str('%.2f' % ML_reg.coef_[1]) + '*X2 ')
print('The weight of a TL=31cm and BD=12cm Bream is: ', ML_reg.predict(np.array([[30,11.52]])), 'grams')
x = df_bream.TL.values.reshape(-1,1)
y = df_bream.Weight.values.reshape(-1,1)

linear_reg.fit(x,y)
y_head = linear_reg.predict(x)

from sklearn.preprocessing import PolynomialFeatures
PL_reg = PolynomialFeatures(degree = 2)

x_polynomial = PL_reg.fit_transform(x)

L_reg = LinearRegression()
L_reg.fit(x_polynomial,y)
y_head2 = L_reg.predict(x_polynomial)
plt.scatter(df_bream.TL,df_bream.Weight)
plt.plot(x,y_head,color= "orange")
plt.plot(x,y_head2,color= "green")
plt.xlabel("Total Length")
plt.ylabel("Weight")
plt.show()
data1.head()
x1 = data1.iloc[:,1].values.reshape(-1,1)
y1 = data1.iloc[:,4].values.reshape(-1,1)

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x1,y1)

x1_ = np.arange(min(x1), max(x1), 0.01).reshape(-1,1)
y1_head = tree_reg.predict(x1_)
plt.scatter(x1,y1, color = "red")
plt.plot(x1_,y1_head,color = "green")
plt.xlabel("Total Length")
plt.ylabel("Weight")
plt.show()
x2 = data1.iloc[:,1].values.reshape(-1,1)
y2 = data1.iloc[:,4].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(x2,y2)

x2_ = np.arange(min(x2),max(x2),0.01).reshape(-1,1)
y2_head = rf.predict(x2_)

plt.scatter(x2,y2,color = "red")
plt.plot(x2_,y2_head,color = "green")
plt.xlabel("Total Length")
plt.ylabel("Weight")
plt.show()
# Separate variables
y = df_bream['Weight']
X = df_bream.iloc[:,[4,5]]

# Divide dataset for train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Regression model
ML_reg = LinearRegression()
ML_reg.fit(X_train, y_train)

#Predict weight values from train dataset
y_head = ML_reg.predict(X_train)
from sklearn.metrics import r2_score
r2_score(y_train, y_head)
from sklearn.model_selection import cross_val_score
cross_val_score_train = cross_val_score(ML_reg, X_train, y_train, cv=4, scoring='r2')
print(cross_val_score_train)
cross_val_score_train.mean()
y_pred = ML_reg.predict(X_test)
print(r2_score(y_test, y_pred))
plt.scatter(X_test['TL'], y_test, color='red', alpha=0.4) #Real data
plt.scatter(X_test['TL'], y_pred, color='blue', alpha=0.4) #Predicted data
plt.xlabel('Total Length in cm')
plt.ylabel('Weight of the fish in grams')
plt.title('Linear Regression Model for Weight Estimation');
# Separate variables
yRF = data1.iloc[:,1].values.reshape(-1,1)
XRF = data1.iloc[:,4].values.reshape(-1,1)

# Divide dataset for train and test
from sklearn.model_selection import train_test_split
XRF_train, XRF_test, yRF_train, yRF_test = train_test_split(XRF, yRF, test_size=0.2, random_state=1)

# Regression model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(XRF_train,yRF_train)

#Predict weight values from train dataset
yRF_head = rf.predict(XRF_train)
r2_score(yRF_train, yRF_head)
from sklearn.model_selection import cross_val_score
cross_val_score_train = cross_val_score(rf, XRF_train, yRF_train, cv=10, scoring='r2')
print(cross_val_score_train)
cross_val_score_train.mean()
yRF_pred = rf.predict(XRF_test).reshape(-1,1)
print(r2_score(yRF_test, yRF_pred))
plt.scatter(XRF_test, yRF_test, color='red', alpha=0.4) #Real data
plt.scatter(XRF_test, yRF_pred, color='blue', alpha=0.4) #Predicted data
plt.xlabel('Total Length in cm')
plt.ylabel('Weight of the fish in grams')
plt.title('Random Forest Regression Model for Weight Estimation');