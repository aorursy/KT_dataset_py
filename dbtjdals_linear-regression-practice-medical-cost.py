#essentials

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.tools as tls

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly.graph_objs as go

import plotly.express as px

init_notebook_mode(connected=True)



#machine learning

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics



#show input file directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#read data

insurance_data = pd.read_csv('/kaggle/input/insurance/insurance.csv')
#basic info

insurance_data.info()
#basic stats

insurance_data.describe()
#check head of data

insurance_data.head()
#check distribution of age

insurance_data['age'].hist(bins=20)
#explore relationship across dataset

sns.pairplot(insurance_data)
#compare charges between male and female

sns.stripplot(x='sex',y='charges',data=insurance_data)
#compare charges between smokers and non smokers

sns.stripplot(x='smoker',y='charges',data=insurance_data)
g = sns.FacetGrid(data=insurance_data,col='smoker')

g.map(sns.distplot,'charges',bins=30,kde=False)
#plot bmi vs. charges in relationship with smoker y/n

px.scatter(insurance_data,x='bmi',y='charges',color='smoker',color_discrete_sequence=['red','blue'])
#plot bmi vs. charges in relationship with smoker y/n

sns.lmplot(x='bmi',y='charges',data= insurance_data, col = 'smoker')
#plot number of children vs. charges

sns.barplot(x='children',y='charges',data = insurance_data)
#plot region vs. charges

sns.barplot(x='region',y='charges',data = insurance_data)
#plot bmi vs. region

sns.stripplot(x='region',y='bmi',data=insurance_data)
insurance_data.info()
#assign dummy variable to categorical features



insurance_data=pd.get_dummies(insurance_data, columns=['sex'])

insurance_data=pd.get_dummies(insurance_data, columns=['smoker'])

insurance_data=pd.get_dummies(insurance_data, columns=['children'])



insurance_data.drop('sex_female', axis=1, inplace=True)

insurance_data.drop('smoker_no', axis=1, inplace=True)

insurance_data.drop('children_5', axis=1, inplace=True)
#create dummy variables for region

insurance_data=pd.get_dummies(insurance_data, columns=['region'])



insurance_data.drop('region_northeast', axis=1, inplace=True)
#scale normalize numerical variables



from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

num_vars = ['age', 'bmi', 'charges']

insurance_data[num_vars] = scaler.fit_transform(insurance_data[num_vars])
insurance_data.head()
insurance_data.info()
#Correlation using heatmap

plt.figure(figsize = (10, 10))

sns.heatmap(insurance_data.corr(), annot = True, cmap="YlGnBu")

plt.show()
#assign features and labels

X = insurance_data.drop(['charges'],axis=1)

y = insurance_data['charges']
#split train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
#assign linear model object

lm = LinearRegression(fit_intercept=False, normalize=True)
#fit, train, test, and show results

def fit_show_results(X_train, y_train, X_test, y_test):

    

    #fit linear model

    lm.fit(X_train,y_train)

    

    #call predictions

    predictions = lm.predict(X_test)

    

    #regression plot of the real test values versus the predicted values

    plt.figure(figsize=(16,8))

    sns.regplot(y_test,predictions)

    plt.xlabel('Predictions')

    plt.ylabel('Actual')

    plt.title("Linear Model Predictions")

    plt.grid(False)

    plt.show()

    

    #show dataframe of coefficients

    coeffecients = pd.DataFrame(lm.coef_,X_train.columns)

    coeffecients.columns = ['Coeffecient']

    print(coeffecients)

    

    #calculate metrics

    print('\n')

    print('MAE:', metrics.mean_absolute_error(y_test, predictions))

    print('MSE:', metrics.mean_squared_error(y_test, predictions))

    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    

    #calculate r squared

    SS_Residual = sum((y_test-predictions)**2)

    SS_Total = sum((y_test-np.mean(y_test))**2)

    r_squared = 1 - (float(SS_Residual))/SS_Total

    print('R Squared:', r_squared)
#fit and check p-values

import statsmodels.api as sm



Xt = sm.add_constant(X_train)

lin_model = sm.OLS(y_train, Xt).fit()

print(lin_model.summary())
#only select features with p-values lower than significance value of 0.05

X_train2 = X_train[['age','bmi','smoker_yes']]

X_test2 = X_test[['age','bmi','smoker_yes']]
#fit and run model

fit_show_results(X_train2, y_train, X_test2, y_test)