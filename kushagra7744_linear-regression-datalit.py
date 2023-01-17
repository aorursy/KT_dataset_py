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
df=pd.read_csv('../input/Automobile_data.csv')

df.head()
print(df.columns)
df['horsepower']=pd.to_numeric(df['horsepower'],errors='coerce')

df['curb-weight']=pd.to_numeric(df['curb-weight'],errors='coerce')

df['price']=pd.to_numeric(df['price'],errors='coerce')

df.dropna(subset=['horsepower','price','curb-weight'],inplace=True)

df.head()
from scipy.stats import pearsonr
pearsonr(df.horsepower,df.price)
pearsonr(df['curb-weight'],df.price)

#higher correlation than horsepower
from bokeh.io import output_notebook

from bokeh.plotting import figure, show, ColumnDataSource

output_notebook()
datsrc= ColumnDataSource(data=(dict(x=df['curb-weight'],y=df['price'],make=df.make)))

ttips=[('price','$y{$0}'),('make','@make'),('curb-weight','$x')]



pl= figure(plot_width=600,plot_height=400,tooltips=ttips)



pl.xaxis.axis_label='curb-weight'

pl.yaxis.axis_label='price'



pl.circle('x','y',source=datsrc,size=10,color='red',alpha=0.5)



show(pl)
from sklearn.model_selection import train_test_split

from sklearn import linear_model
train_d,test_d=train_test_split(df,test_size=0.25)
mymodel=linear_model.LinearRegression()

#model is expecting a 2d matrix as an input at x. Thus x has to be reshaped.

trn_x=np.array(train_d['curb-weight']).reshape(-1,1)#returning a 2d matrix

trn_y=np.array(train_d['price'])



mymodel.fit(trn_x,trn_y)
slope=np.asscalar(np.squeeze(mymodel.coef_))

intercept=mymodel.intercept_



print("Slope= ",slope,"Intercept= ",intercept)
from bokeh.models import Slope



best_fit=Slope(gradient=slope,y_intercept=intercept,line_color='black',line_width=3)

pl.add_layout(best_fit)

show(pl)
from sklearn.metrics import accuracy_score, mean_squared_error,mean_absolute_error,r2_score
predictions_train=np.array(mymodel.predict(trn_x))





maet=mean_absolute_error(trn_y,predictions_train)

mset=mean_squared_error(trn_y,predictions_train)

r2scoret=r2_score(trn_y,predictions_train)

print("For TRAIN DATA\nMean Absolute Error : ",maet,"\nMean Squared Error: ",mset,"\nR2 Score: ",r2scoret)
test_x=np.array(test_d['curb-weight']).reshape(-1,1)

test_y=np.array(test_d['price'])



predictions=(mymodel.predict(test_x))
mae=mean_absolute_error(test_y,predictions)

mse=mean_squared_error(test_y,predictions)

r2score=r2_score(test_y,predictions)



print("For TEST DATA\nMean Absolute Error : ",mae,"\nMean Squared Error: ",mse,"\nR2 Score: ",r2score)
cols=['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',

        'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',

        'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',

        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke',

        'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg',

        'highway-mpg', 'price']



for col in cols:

    df[col]=pd.to_numeric(df[col],errors='coerce')

df.dropna(subset=['price','horsepower','curb-weight'])



for col in cols:

    print(col,pearsonr(df[col],df.price))
cols=['horsepower','curb-weight','length','width','engine-size']



multi_x=np.column_stack(tuple(df[col] for col in cols))



multi_train_x,multi_test_x,multi_train_y,multi_test_y=train_test_split(multi_x,df['price'],test_size=0.25)

multi_model=linear_model.LinearRegression()

multi_model.fit(multi_train_x,multi_train_y)

multi_coef=dict(zip(cols,multi_model.coef_))

multi_intercept=multi_model.intercept_

#for training data

pred= multi_model.predict(multi_train_x)



mae=mean_absolute_error(multi_train_y,pred)

mse=mean_squared_error(multi_train_y,pred)

r2score=r2_score(multi_train_y,pred)



print("For TRAIN DATA\nMean Absolute Error : ",mae,"\nMean Squared Error: ",mse,"\nR2 Score: ",r2score)
#for training data

pred= multi_model.predict(multi_test_x)



mae=mean_absolute_error(multi_test_y,pred)

mse=mean_squared_error(multi_test_y,pred)

r2score=r2_score(multi_test_y,pred)



print("For TRAIN DATA\nMean Absolute Error : ",mae,"\nMean Squared Error: ",mse,"\nR2 Score: ",r2score)
from sklearn.model_selection import GridSearchCV
ridge=linear_model.Ridge()

alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

parameters={'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

ridgeReg=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)


ridgeReg.fit(multi_train_x,multi_train_y)

pred=ridgeReg.predict(multi_test_x)



r2=r2_score(pred,multi_test_y)

r2
lasso= linear_model.Lasso()

alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

parameters={'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

lassoreg=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)



lassoreg.fit(multi_train_x,multi_train_y)



lpred=lassoreg.predict(multi_test_x)



mae=mean_absolute_error(lpred,multi_test_y)

mse=mean_squared_error(lpred,multi_test_y)

r2score=r2_score(lpred,multi_test_y)

print("For TRAIN DATA\nMean Absolute Error : ",mae,"\nMean Squared Error: ",mse,"\nR2 Score: ",r2score)
lpred= lassoreg.predict(multi_train_x)

mae=mean_absolute_error(lpred,multi_train_y)

mse=mean_squared_error(lpred,multi_train_y)

r2score=r2_score(lpred,multi_train_y)

print("For TRAIN DATA\nMean Absolute Error : ",mae,"\nMean Squared Error: ",mse,"\nR2 Score: ",r2score)
lpred=lassoreg.predict(multi_test_x)



mae=mean_absolute_error(lpred,multi_test_y)

mse=mean_squared_error(lpred,multi_test_y)

r2score=r2_score(lpred,multi_test_y)

print("For TRAIN DATA\nMean Absolute Error : ",mae,"\nMean Squared Error: ",mse,"\nR2 Score: ",r2score)