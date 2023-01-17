

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
#Import the dataset

df = pd.read_csv("../input/Admission_Predict.csv")

#Lets checkd the dataset

df.info() #Clearly we see the there is no missing data
# Rename the columns name which have to required change.

df=df.rename(columns={'Serial No.':'SerialNo', 'GRE Score':'GRE', 'TOEFL Score':'TOEFL',

                      'University Rating':'UniversityRating','LOR ':'LOR','Chance of Admit ':'ChanceOfAdmit'})

df.columns
#Lets check our target variable

import seaborn as sns

import matplotlib.pyplot as plt

sns.distplot(df['ChanceOfAdmit'])

print('Skewness: %f', df['ChanceOfAdmit'].skew())

print("Kurtosis: %f" % df['ChanceOfAdmit'].kurt())
#Lets check the correlation by ploting heatmap

corr = df.corr()

colormap = sns.diverging_palette(220, 10, as_cmap = True)

plt.figure(figsize = (8,6))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            annot=True,fmt='.2f',linewidths=0.30,

            cmap = colormap, linecolor='white')

plt.title('Correlation of df Features', y = 1.05, size=15)
#Lets look the correlation score

print (corr['ChanceOfAdmit'].sort_values(ascending=False), '\n')
#We have our target variable ChanceOfAdmit and some correlated features.

#Lets see the top three coreleated features.

#At first check the relation with CGPA

sns.jointplot(x =df['CGPA'], y = df['ChanceOfAdmit'], color = 'deeppink')
#Now check the relation with GRE

sns.jointplot(x =df['GRE'], y = df['ChanceOfAdmit'], color = 'green')
#Now check the relation with TOEFL

sns.jointplot(x =df['TOEFL'], y = df['ChanceOfAdmit'], color = 'skyblue')
#Now we create our matrices of features for ML model

x = df.iloc[:,1:8].values

y = df.iloc[:, 8].values
#Spliting Dataset into the traning and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#Fitting the Random Forest Regression

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100, random_state = 10)

regressor.fit(x_train, y_train)
#Predicting the Test set Result

y_pred = regressor.predict(x_test)
#Now Lets Check the Training and Test set Accuracy

accuracy_train = regressor.score(x_train, y_train)

accuracy_test = regressor.score(x_test, y_test)

print(accuracy_train)

print(accuracy_test)
#Now Check the error for regression

from sklearn import metrics

print('MAE :'," ", metrics.mean_absolute_error(y_test,y_pred))

print('MSE :'," ", metrics.mean_squared_error(y_test,y_pred))

print('RMAE :'," ", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#Visualising the Acutal and predicted Result

plt.plot(y_test, color = 'deeppink', label = 'Actual')

plt.plot(y_pred, color = 'blue', label = 'Predicted')

plt.grid(alpha = 0.3)

plt.xlabel('Number of Candidate')

plt.ylabel('Score')

plt.title('Actual vs Predicted')

plt.legend()

plt.show()