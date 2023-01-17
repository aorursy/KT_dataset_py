# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mplib

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

# /kaggle/input/life-expectancy-who/led.csv
data = pd.read_csv("../input/life-expectancy-who/led.csv")

data_df = pd.DataFrame(data)

print(data_df.info())

print(data_df.describe())

# data_df.head()

print("Null values:\n", data_df.isnull().sum())

print("Percent missingness:\n", data_df.isnull().sum() / data_df.count())

print("Shape:\n", data_df.shape)

print("Data Types:\n", data_df.dtypes)
corrMatrix = data_df.corr()

corrMatrix.style.background_gradient(cmap='plasma', low=.5, high=0).highlight_null('red') # from https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
df_last5 = data_df[data_df['Year'].isin([2011,2012,2013,2014,2015])]
df_last5_avg = df_last5.groupby(['Country'],as_index=False).mean()
print(df_last5[df_last5['Country']=='Italy'])

print(df_last5_avg[df_last5_avg['Country']=='Italy'])
print('Percent Missingness:\n', df_last5_avg.isnull().sum()/df_last5_avg.count())
df_last5_avg.drop(['Population'],1,inplace=True)

df_last5_avg.drop(['Year'],1,inplace=True)
# pd.set_option("display.max_rows", None, "display.max_columns", None) # this allows you to see the full dataset
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values = np.nan, strategy='median')

imp = imp.fit(df_last5_avg[['GDP']])

df_last5_avg['GDP'] = imp.transform(df_last5_avg[['GDP']])
imp = SimpleImputer(missing_values = np.nan, strategy='median')

imp = imp.fit(df_last5_avg[['Lifeexpectancy']])

df_last5_avg['Lifeexpectancy'] = imp.transform(df_last5_avg[['Lifeexpectancy']])
imp = SimpleImputer(missing_values = np.nan, strategy='median')

imp = imp.fit(df_last5_avg[['AdultMortality']])

df_last5_avg['AdultMortality'] = imp.transform(df_last5_avg[['AdultMortality']])
imp = SimpleImputer(missing_values = np.nan, strategy='median')

imp = imp.fit(df_last5_avg[['Alcohol']])

df_last5_avg['Alcohol'] = imp.transform(df_last5_avg[['Alcohol']])
imp = SimpleImputer(missing_values = np.nan, strategy='median')

imp = imp.fit(df_last5_avg[['HepatitisB']])

df_last5_avg['HepatitisB'] = imp.transform(df_last5_avg[['HepatitisB']])
imp = SimpleImputer(missing_values = np.nan, strategy='median')

imp = imp.fit(df_last5_avg[['BMI']])

df_last5_avg['BMI'] = imp.transform(df_last5_avg[['BMI']])
imp = SimpleImputer(missing_values = np.nan, strategy='median')

imp = imp.fit(df_last5_avg[['Totalexpenditure']])

df_last5_avg['Totalexpenditure'] = imp.transform(df_last5_avg[['Totalexpenditure']])
imp = SimpleImputer(missing_values = np.nan, strategy='median')

imp = imp.fit(df_last5_avg[['thinness1-19years']])

df_last5_avg['thinness1-19years'] = imp.transform(df_last5_avg[['thinness1-19years']])
imp = SimpleImputer(missing_values = np.nan, strategy='median')

imp = imp.fit(df_last5_avg[['thinness5-9years']])

df_last5_avg['thinness5-9years'] = imp.transform(df_last5_avg[['thinness5-9years']])
imp = SimpleImputer(missing_values = np.nan, strategy='median')

imp = imp.fit(df_last5_avg[['Incomecompositionofresources']])

df_last5_avg['Incomecompositionofresources'] = imp.transform(df_last5_avg[['Incomecompositionofresources']])
imp = SimpleImputer(missing_values = np.nan, strategy='median')

imp = imp.fit(df_last5_avg[['Schooling']])

df_last5_avg['Schooling'] = imp.transform(df_last5_avg[['Schooling']])
print("Percent missingness:\n", df_last5_avg.isnull().sum() / df_last5_avg.count())
X = df_last5_avg['AdultMortality']

y = df_last5_avg['Lifeexpectancy']

plt.scatter(X,y)

plt.ylabel('Life Expectancy')

plt.xlabel('Adult Mortality per 1000 pop.')

plt.show()
X = df_last5_avg['GDP']

y = df_last5_avg['Lifeexpectancy']

plt.scatter(X,y)

plt.ylabel('Life Expectancy')

plt.xlabel('GDP')

plt.show()
from sklearn import linear_model

reg = linear_model.LinearRegression()

reg.fit(df_last5_avg[['AdultMortality']],df_last5_avg['Lifeexpectancy'])

prediction_space = np.linspace(min(df_last5_avg['AdultMortality']),max(df_last5_avg['AdultMortality'])).reshape(-1,1)

plt.scatter(df_last5_avg['AdultMortality'],df_last5_avg['Lifeexpectancy'],color='yellow')

plt.plot(prediction_space,reg.predict(prediction_space),color='blue',linewidth=3)

plt.show()
correlated_features = set()

correlation_matrix = df_last5_avg.drop('Lifeexpectancy', axis=1).corr()



for i in range(len(correlation_matrix.columns)):

    for j in range(i):

        if abs(correlation_matrix.iloc[i, j]) > 0.8: #.iloc method is used to extract rows

            colname = correlation_matrix.columns[i]

            correlated_features.add(colname)

print(correlated_features)
df_last5_avg.drop(['Diphtheria'],1,inplace=True)

df_last5_avg.drop(['thinness5-9years'],1,inplace=True)

df_last5_avg.drop(['under-fivedeaths'],1,inplace=True)

df_last5_avg.drop(['Schooling'],1,inplace=True)
df_last5_avg.head()

df_last5_avg.set_index('Country',inplace=True)
from sklearn.feature_selection import RFE



X = df_last5_avg.drop('Lifeexpectancy',axis=1)

y = df_last5_avg['Lifeexpectancy']



lr = linear_model.LinearRegression()

rfe = RFE(estimator=lr, n_features_to_select=8, step=1)

rfe.fit(X, y)



#print(rfe.get_support)

print(rfe.ranking_)
from sklearn.model_selection import train_test_split

#no of features

nof_list=np.arange(1,13)            

high_score=0

#Variable to store the optimum features

nof=0           

score_list =[]

for n in range(len(nof_list)):

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

    model = linear_model.LinearRegression()

    rfe = RFE(model,nof_list[n])

    X_train_rfe = rfe.fit_transform(X_train,y_train)

    X_test_rfe = rfe.transform(X_test)

    model.fit(X_train_rfe,y_train)

    score = model.score(X_test_rfe,y_test)

    score_list.append(score)

    if(score>high_score):

        high_score = score

        nof = nof_list[n]

print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))
cols = list(X.columns)

model = linear_model.LinearRegression()

#Initializing RFE model

rfe = RFE(model, 7)             

#Transforming data using RFE

X_rfe = rfe.fit_transform(X,y)  

#Fitting the data to model

model.fit(X_rfe,y)             

temp = pd.Series(rfe.support_,index = cols)

selected_features_rfe = temp[temp==True].index

print(selected_features_rfe)

coefs = model.fit(X_rfe,y).coef_  

_ = plt.plot(coefs)

_ = plt.xticks(np.arange(7),('AdultMortality', 'Alcohol', 'HepatitisB', 'Totalexpenditure',

       'HIV/AIDS', 'thinness1-19years', 'Incomecompositionofresources'), rotation=60)

_ = plt.ylabel('Coefficients')

plt.show()
coef_dict = dict(enumerate(coefs))

coef_dict = {'AdultMortality':coef_dict[0],'Alcohol':coef_dict[1],'HepatitisB':coef_dict[2],

             'Totalexpenditure':coef_dict[3],'HIV/AIDS':coef_dict[4],

             'thinness1-19years':coef_dict[5],'Incomecompositionofresources':coef_dict[6]}

coef_dict
import seaborn as sns

corr = df_last5_avg.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right')
df_regress = df_last5_avg[['AdultMortality', 'Alcohol', 'HepatitisB', 'Totalexpenditure',

       'HIV/AIDS', 'thinness1-19years', 'Incomecompositionofresources','Lifeexpectancy']]

X = df_regress.drop('Lifeexpectancy',axis=1)

y = df_regress['Lifeexpectancy']



from sklearn.linear_model import Ridge

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)

ridge = Ridge(alpha = 0.1, normalize=True)

ridge.fit(X_train,y_train)

ridge_pred=ridge.predict(X_test)

ridge_pred1=ridge.predict(X_train)

print(ridge.score(X_train,y_train))

print(ridge.score(X_test,y_test))
from sklearn.linear_model import Lasso



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)

lasso = Lasso(alpha = 0.1, normalize=True)

lasso.fit(X_train,y_train)

lasso_pred=lasso.predict(X_test)

lasso.score(X_test,y_test)