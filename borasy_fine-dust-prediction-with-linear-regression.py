# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

from sklearn import linear_model, model_selection

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import Imputer



df = pd.read_csv("../input/SeoulHourlyAvgAirPollution.csv")

df.head()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#df.rename(columns={'측정일시':'Date/Time','측정소명':'Location','이산화질소농도(ppm)':'NO2', '오존농도(ppm)','O3','일산화탄소농도(ppm)','CO','아황산가스(ppm)','SO2','미세먼지(㎍/㎥)','Fine Dust','초미세먼지(㎍/㎥)','Ultrafine Dust'}, inplace=True)

df.rename(columns={'측정일시':'Date/Time','측정소명':'Location','이산화질소농도(ppm)':'NO2', '오존농도(ppm)':'O3','일산화탄소농도(ppm)':'CO','아황산가스(ppm)':'SO2','미세먼지(㎍/㎥)':'Fine Dust','초미세먼지(㎍/㎥)':'Ultrafine Dust'}, inplace=True)

df.head()
# Find missing values

print(df.isnull().sum())

#df = df.fillna(0)

#df = df.fillna(df.mean())

df = df.interpolate()

print(df.isnull().sum())

df = pd.get_dummies(df, columns=['Location'])

df.head()
df['Target'] = df['Fine Dust']

df.drop(['Fine Dust'],axis=1,inplace=True)

df.head()
y = df.Target.values

features_col = [i for i in list(df.columns) if i != 'Target']

#print (list(df.columns))

#print (features_col)

x = df.loc[:,features_col].as_matrix()

print (y.shape)

print (x.shape)



x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=0)



print (x_train.shape)

print (x_test.shape)

print (y_train.shape)

print (y_test.shape)
# Create linear regression object

regr = linear_model.LinearRegression()



# Imputation

#my_imputer = Imputer()

#x_train_imputed = my_imputer.fit_transform(x_train)

#x_test_imputed = my_imputer.fit_transform(x_test)



# Train the model using the training sets

regr.fit(x_train, y_train)



# Make predictions using the testing set

y_pred = regr.predict(x_test)



# The mean squared error

print("Mean squared error: %.8f" % mean_squared_error(y_test, y_pred))

      

# Explained variance score: 1 is perfect prediction

print('Variance R2 score: %.8f' % r2_score(y_test, y_pred))
# Plot outputs

fig, ax = plt.subplots()

ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))

#ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')



#plt.xticks(())

#plt.yticks(())



plt.show()