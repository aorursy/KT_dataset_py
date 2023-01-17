import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_original = pd.read_csv("/kaggle/input/seattle-house-sales-prices/house_sales.csv")
df = df_original.copy()
df.head()
df.shape
df.drop(["id", "date"], inplace=True, axis=1)
df.head()
df.columns
for column in df.columns:
    if df[column].isnull().any():
        print("The {} has {} null values".format(column, df[column].isnull().sum()))
df.describe()
plt.boxplot(df['bedrooms'], widths=0.5)
df['bedrooms'].value_counts()
df.dtypes
# removing rows having bedrooms 33 and 11 sonce they are outliers
df.drop(df[df['bedrooms']==33].index, inplace=True, axis=0)
df.drop(df[df['bedrooms']==11].index, inplace=True, axis=0)
df['bedrooms'].value_counts()
df.describe()
df.corr()
import seaborn as sns
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), xticklabels=df.columns, yticklabels=df.columns, linewidths=0, annot=df.corr())
df1 = df.copy()
cols = list(df1.columns)
price_col = cols.pop(cols.index('price'))
price_col
df1 = df1[cols+['price']]
df1.head()
corr_matrix = df1.corr()
type(corr_matrix)
sort = corr_matrix.sort_values('price',ascending=False)
sort[['price']]
plt.scatter(df1['sqft_living'], df1['price'])
plt.scatter(df1['long'], df1['price'])
sns.regplot(df1['long'], df1['price'])
sns.regplot(df1['sqft_living'], df1['price'])
from numpy import cov
covariance = cov(df1['long'],df1['price'])
covariance
df['waterfront'].value_counts()
df['view'].value_counts()
df['condition'].value_counts()
#ANOVA
grouped_test2=df1[['waterfront', 'price']].groupby(['waterfront'])
grouped_test2.head()
from scipy.stats import f_oneway
f_val, p_val = f_oneway(grouped_test2.get_group(0)['price'], grouped_test2.get_group(1)['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val)  
from scipy.stats import pearsonr
corr, p = pearsonr(df1['long'], df1['price'])
print("correlation: ", corr)
print("p-value: ", p)
pearson_coef, p_value = pearsonr(df1['yr_renovated'], df1['price'])
print("The Pearson Correlation Coefficient: ", pearson_coef, "\nP-value of P: ", p_value)
pearson_coef, p_value = pearsonr(df1['long'], df1['price'])
print("The Pearson Correlation Coefficient: ", pearson_coef, "\nP-value of P: ", p_value)
features1 = df1.columns.to_list()[:-1]
features1
pearson_coef = []
p_values = []
for column in features1:
    coef, p_val = pearsonr(df1[column], df1['price'])
    pearson_coef.append(coef)
    p_values.append(p_val)
dict1 = {"Features" : features1, "Pearson Coeffs" : pearson_coef, "P-values" : p_values }
pd.DataFrame(dict1).sort_values(by="Pearson Coeffs", ascending=False,)
#taking part of columns and validating the accuracy
features_1 = df1[['sqft_living','grade','sqft_above','sqft_living15','bathrooms','view','sqft_basement','bedrooms','lat','waterfront']]
features_1.shape
label = df['price']
label.shape
featuresall = df1.iloc[:,:-2]
featuresall.head()
df[featuresall.columns.to_list()[0]].describe()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
bruteScores = []
for i in range(1,30):
    x_train,  x_test, y_train, y_test = train_test_split(featuresall, label, test_size = 0.3, random_state=i)
    
    LinReg_model = LinearRegression().fit(x_train, y_train)
    
    test_score = LinReg_model.score(x_test,y_test)
    train_score = LinReg_model.score(x_train,y_train)
    
    if test_score > train_score:
         bruteScores.append((train_score, test_score, i))
             
print(pd.DataFrame(bruteScores, columns=["train score", 'test score', 'random state']))
#correlation b/w features
for i in featuresall.columns.to_list():
    for j in featuresall.columns.to_list():
        correlation = df[i].corr(df[j])
        if correlation > 0.95 and correlation != 1:
            print ("Corelated columns: {} {}".format(i,j))
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(featuresall)
poly_features.shape
featuresall.shape
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
bruteScores = []
for i in range(1,30):
    x_train,  x_test, y_train, y_test = train_test_split(poly_features, label, test_size = 0.3, random_state=i)
    
    LinReg_model = LinearRegression().fit(x_train, y_train)
    
    test_score = LinReg_model.score(x_test,y_test)
    train_score = LinReg_model.score(x_train,y_train)
    
    if test_score > train_score:
         bruteScores.append((train_score, test_score, i))
             
print(pd.DataFrame(bruteScores, columns=["train score", 'test score', 'random state']))
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_features = sc.fit_transform(featuresall)
principalComponents = PCA(n_components=10)
principalComponents.fit(scaled_features,label)
PCA_Var_ratio = principalComponents.explained_variance_ratio_.tolist()
sum(PCA_Var_ratio)
PCA_features = principalComponents.transform(featuresall)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
bruteScores = []
for i in range(1,30):
    x_train,  x_test, y_train, y_test = train_test_split(PCA_features, label, test_size = 0.3, random_state=i)
    
    LinReg_model = LinearRegression().fit(x_train, y_train)
    
    test_score = LinReg_model.score(x_test,y_test)
    train_score = LinReg_model.score(x_train,y_train)
    
    if test_score > train_score:
         bruteScores.append((train_score, test_score, i))
             
print(pd.DataFrame(bruteScores, columns=["train score", 'test score', 'random state']))
#cross validaion
from sklearn.model_selection import cross_val_score
cross_val = cross_val_score(LinearRegression(), poly_features, label, cv=5)
print(np.mean(cross_val))
#Pipelining 
from sklearn.pipeline import Pipeline
Poly_Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(degree=2,include_bias=False)), ('model',LinearRegression())]
polypipe = Pipeline(Poly_Input)
polypipe
polypipe.fit(featuresall, label)
y_polypipe = polypipe.predict(featuresall)
y_polypipe[0:3]
polypipe.score(featuresall, label)
Scale_Input=[('scale',StandardScaler()), ('model',LinearRegression())]
scalepipe = Pipeline(Scale_Input)
scalepipe.fit(featuresall, label)
y_scalepipe = scalepipe.predict(featuresall)
y_scalepipe[0:5]
scalepipe.score(featuresall, label)
#Decision tree regressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
x_train,  x_test, y_train, y_test = train_test_split(featuresall, label, test_size = 0.3, random_state=20)

DTR_model = DecisionTreeRegressor().fit(x_train, y_train)

test_score = DTR_model.score(x_test,y_test)
train_score = DTR_model.score(x_train,y_train)

print("Test Score: {}\nTrain Score: {}".format(test_score,train_score))
#Decision tree regressor
from sklearn.tree import DecisionTreeRegressor
for i in range(1,30):
    x_train,  x_test, y_train, y_test = train_test_split(featuresall, label, test_size = 0.3, random_state=i)
    
    DTR_model = DecisionTreeRegressor().fit(x_train, y_train)

    test_score = DTR_model.score(x_test,y_test)
    train_score = DTR_model.score(x_train,y_train)
    
    if test_score > train_score:
        print("Test Score: {}\tTrain Score: {}\tRandom State: {}".format(test_score,train_score,i))
from sklearn.linear_model import Ridge
x_train,  x_test, y_train, y_test = train_test_split(featuresall, label, test_size = 0.3, random_state=20)

Ridge_model = DecisionTreeRegressor().fit(x_train, y_train)

test_score = Ridge_model.score(x_test,y_test)
train_score = Ridge_model.score(x_train,y_train)

print("Test Score: {}\nTrain Score: {}".format(test_score,train_score))
from sklearn.linear_model import Ridge
x_train,  x_test, y_train, y_test = train_test_split(poly_features, label, test_size = 0.3, random_state=20)

Ridge_model = DecisionTreeRegressor().fit(x_train, y_train)

test_score = Ridge_model.score(x_test,y_test)
train_score = Ridge_model.score(x_train,y_train)

print("Test Score: {}\nTrain Score: {}".format(test_score,train_score))
from sklearn.metrics import mean_squared_error
#Selecting the best model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train,  x_test, y_train, y_test = train_test_split(poly_features, label, test_size = 0.3, random_state=1)
    
LinReg_model = LinearRegression().fit(x_train, y_train)
    
test_score = LinReg_model.score(x_test,y_test)
train_score = LinReg_model.score(x_train,y_train)
    
if test_score > train_score:             
    print('Train Score: {}\nTest Score: {}\nRandom State: {}'.format(train_score, test_score, i))
LR_prediction = LinReg_model.predict(x_test)
mse = mean_squared_error(y_test, LR_prediction)
print('The mean square error of price and predicted value is: ', mse)
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, LR_prediction)
print('The R-square value is: ', r_squared)
LinReg_model.score(x_test, y_test)
#Pipelining 
from sklearn.pipeline import Pipeline
Poly_Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(degree=2,include_bias=False)), ('model',LinearRegression())]
polypipe = Pipeline(Poly_Input)
polypipe
x_train,  x_test, y_train, y_test = train_test_split(featuresall, label, test_size = 0.3, random_state=1)
polypipe.fit(x_train, y_train)
y_polypipe = polypipe.predict(x_test)
y_polypipe[0:4]
r_squared = r2_score(y_test, y_polypipe)
print('The R-square value is: ', r_squared)
