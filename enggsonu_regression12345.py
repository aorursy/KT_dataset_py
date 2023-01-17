# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sn

import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
boston = load_boston()
x,y=boston["data"],boston["target"]
boston.feature_names
df=pd.DataFrame(x,columns=boston.feature_names)
df.head()
df.shape
df['target']=boston['target']
df.head()
df.shape
df.describe()
df.isnull().sum()
sn.heatmap(df.corr().round(2),annot=True)
sn.set_style("ticks")

sn.set(rc={"figure.figsize":(10,7)})

sn.heatmap(df.corr().round(2),annot=True)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,Ridge,Lasso
df_x=df[df.columns[:12]]
X_train,X_test,Y_train,Y_test = train_test_split(df_x,df['target'],test_size=0.2,random_state=42)
model_Lr = LinearRegression()

model_Lr.fit(X_train,Y_train)

print("train accuracy",model_Lr.score(X_train,Y_train))

print("test accuracy",model_Lr.score(X_test,Y_test))
model_Lr.coef_
for idx, col_name in enumerate(df_x.columns):

    print("the coefficient for {} is {}".format(col_name,model_Lr.coef_[idx]))
model_Lr = LinearRegression(normalize=True)

model_Lr.fit(X_train,Y_train)

print("train accuracy",model_Lr.score(X_train,Y_train))

print("test accuracy",model_Lr.score(X_test,Y_test))
for idx,col_name in enumerate(df_x.columns):

    print("the coefficient for {} is {}".format(col_name,model_Lr.coef_[idx]))
model_Lr_1=Ridge()

model_Lr_1.fit(X_train,Y_train)

print("train accuracy",model_Lr_1.score(X_train,Y_train))

print("test accuracy",model_Lr_1.score(X_test,Y_test))
model_Lr_1.coef_
for idx,col_name in enumerate(df_x.columns):

    print("the coefficient for {} is {}".format(col_name, model_Lr_1.coef_[idx]))

    
model_Lr_10=Ridge(alpha=10)

model_Lr_10.fit(X_train,Y_train)

print("the train accuracy",model_Lr_10.score(X_train,Y_train))

print("the test accuracy",model_Lr_10.score(X_test,Y_test))
for idx,col_name in enumerate(df_x.columns):

    print("the coefficient for {} is {}".format(col_name,model_Lr_10.coef_[idx]))
model_Lr_10.coef_
model_Lr_100 = Ridge(alpha=100)

model_Lr_100.fit(X_train,Y_train)

print("train accuracy",model_Lr_100.score(X_train,Y_train))

print("test accuracy",model_Lr_100.score(X_test,Y_test))
model_Lr_100.coef_
model_Lr = Ridge(alpha=0.1)

model_Lr.fit(X_train,Y_train)

print("train accuracy",model_Lr.score(X_train,Y_train))

print("test accuracy",model_Lr.score(X_test,Y_test))
plt.plot(model_Lr_1.coef_, 's', label="Ridge alpha=1")

plt.plot(model_Lr_10.coef_, '^', label="Ridge alpha=10")

plt.plot(model_Lr_100.coef_, 'v', label="Ridge alpha=100")



plt.plot(model_Lr.coef_, 'o', label="LinearRegression")

plt.xlabel("Coefficient index")

plt.ylabel("Coefficient magnitude")



plt.legend()
model_Lass=Lasso()

model_Lass.fit(X_train,Y_train)

# score Returns the coefficient of determination R^2 of the prediction.

print("Train accuracy",model_Lr.score(X_train,Y_train))

print("Test accuracy",model_Lr.score(X_test,Y_test))
plt.figure(figsize=(8, 4))

plt.plot(model_Lass.coef_, 'o', label="alpha=.001")

plt.legend()
plt.figure(figsize=(8,4))

plt.plot(model_Lass.coef_, 'o', label="alpha=.01")

plt.legend()
plt.figure(figsize=(8,4))

plt.plot(model_Lass.coef_, 'o', label="alpha=10")



plt.legend()
plt.figure(figsize=(8, 8))

#plt.plot(model_Lass.coef_, 'o', label="alpha=01",alpha=0.5)

plt.plot(model_Lass.coef_, 'o', label="alpha=001",alpha=0.5)

plt.plot(model_Lass.coef_, 'o', label="alpha=10",alpha=0.5)

plt.legend()
model_Lass_10 = Lasso(alpha=10)

model_Lass_10.fit(X_train,Y_train)

print("train accuracy",model_Lass_10.score(X_train,Y_train))

print("test accuracy",model_Lass_10.score(X_test,Y_test))
for idx, col_name in enumerate(df_x.columns):

    print("the coefficient for {} is {} ".format(col_name,model_Lass_10.coef_[idx]))
model_Lass_10.coef_
model_Lass_01=Lasso(alpha=0.01)

model_Lass_01.fit(X_train,Y_train)

# score Returns the coefficient of determination R^2 of the prediction.

print("Train accuracy",model_Lass_01.score(X_train,Y_train))

print("Test accuracy",model_Lass_01.score(X_test,Y_test))
model_Lass_001=Lasso(alpha=0.001)

model_Lass_001.fit(X_train,Y_train)

# score Returns the coefficient of determination R^2 of the prediction.

print("Train accuracy",model_Lass_001.score(X_train,Y_train))

print("Test accuracy",model_Lass_001.score(X_test,Y_test))
len(df_x.columns)
g = sn.PairGrid(df, y_vars=["target"], x_vars=df.columns[:6])

g.map(sn.scatterplot)
g = sn.PairGrid(df, y_vars=["target"], x_vars=df.columns[6:13])

g.map(sn.scatterplot)
sn.pairplot(df)
df.columns
ax=sn.boxplot(data=df_x)
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler
scaler=StandardScaler()

scaler.fit(X_train)

X_scaled_train=scaler.transform(X_train)
ax=sn.boxplot(data=X_scaled_train)
ridge=Ridge().fit(X_scaled_train,Y_train)

X_test_scaled=scaler.transform(X_test)

ridge.score(X_test_scaled,Y_test)
ridge=Ridge().fit(X_train,Y_train)

ridge.score(X_test,Y_test)
scaler=MinMaxScaler()

scaler.fit(X_train,Y_train)

X_train_scaler_MM = scaler.transform(X_train)

X_Test_scaled=scaler.transform(X_test)
ridge=Ridge().fit(X_train_scaler_MM,Y_train)

ridge.score(X_Test_scaled,Y_test)
Lasso_reg=Lasso().fit(X_scaled_train,Y_train)

Lasso_reg.score(X_test_scaled,Y_test)
from sklearn.pipeline import make_pipeline



pipe=make_pipeline(StandardScaler(),Ridge())

pipe.fit(X_train,Y_train)

pipe.score(X_test,Y_test)
pipe_NoScaling=make_pipeline(Ridge())

pipe_NoScaling.fit(X_train,Y_train)

pipe_NoScaling.score(X_test,Y_test)
pipe_l=make_pipeline(StandardScaler(),Lasso())

pipe_l.fit(X_train,Y_train)

pipe_l.score(X_test,Y_test)
df_cat = pd.DataFrame(

    {'boro': ['Manhattan', 'Queens', 'Manhattan', 'Brooklyn', 'Brooklyn', 'Bronx'],

     'vegan': ['No', 'No','No','Yes', 'Yes', 'No']})
df_cat
df_cat.info()
pd.get_dummies(df_cat)
df_cat['boro_ordinal'] = df_cat.boro.astype("category").cat.codes

df_cat
df_cat_example = pd.DataFrame(

    {'boro': ['Manhattan', 'Queens', 'Manhattan', 'Brooklyn', 'Brooklyn', 'Bronx'],

     'vegan': ['No', 'No','No','Yes', 'Yes', 'No'],

     'gender':['1','2','1','2','1','2']})
df_cat_example
df_cat_example.head()
df_cat_example.info()
pd.get_dummies(df_cat_example)
df_cat_example_int = pd.DataFrame(

    {'boro': ['Manhattan', 'Queens', 'Manhattan', 'Brooklyn', 'Brooklyn', 'Bronx'],

     'vegan': ['No', 'No','No','Yes', 'Yes', 'No'],

     'gender':[1,2,1,2,1,2]})
df_cat_example_int.info()
df_cat_example_int
pd.get_dummies(df_cat_example_int)
pd.get_dummies(df_cat_example_int,columns=['gender','boro','vegan'])
dfff = pd.DataFrame({'salary': [103, 89, 142, 54, 63, 219],

                   'boro': ['Manhattan', 'Queens', 'Manhattan',

                            'Brooklyn', 'Brooklyn', 'Bronx']})
dfff
from sklearn.preprocessing import OneHotEncoder
transform_one = OneHotEncoder().fit(dfff)



transform_one.transform(dfff).toarray()
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures()

X_poly=poly.fit_transform(df_x)



print(X_poly.shape),print(df_x.shape)
df.columns
X_poly_train,X_poly_test,y_tr,y_tt=train_test_split(X_poly,df['target'],test_size=0.3)
poly_Ridge=Ridge().fit(X_poly_train,y_tr)

poly_Ridge.score(X_poly_test,y_tt)
poly_interact=PolynomialFeatures(degree=2,interaction_only=True)

X_poly_interactions=poly_interact.fit_transform(X_scaled_train)

print(X_poly_interactions.shape),print(X_poly.shape),print(X_scaled_train.shape)
poly_interact=PolynomialFeatures(degree=3,interaction_only=True)

X_poly_interactions=poly_interact.fit_transform(X_scaled_train)

print(X_poly_interactions.shape),print(X_poly.shape),print(X_scaled_train.shape)
X_train.columns
poly_interact=PolynomialFeatures(degree=2,interaction_only=True)

X_poly_interactions=poly_interact.fit_transform(X_train)
poly_interact.get_feature_names(X_train.columns)
scale_pipe=make_pipeline(StandardScaler(),Ridge())

scale_pipe.fit(X_train,Y_train)

scale_pipe.score(X_test,Y_test)
scale_pipe=make_pipeline(StandardScaler(),Ridge())

scale_pipe.fit(X_train,Y_train)

scale_pipe.score(X_test,Y_test)