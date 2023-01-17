# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from pandas_profiling import ProfileReport
fish = pd.read_csv('../input/fish-market/Fish.csv')
fish.head()
fish.shape
ProfileReport(fish)
fish.info()
# knowing some descriptive statistic about data

fish.describe()
fish['Species'].unique()
sns.countplot(fish['Species'])

plt.title('The number of each type of fish')

plt.show()
def box(var):

    # this function take the variable and return a boxplot for each type of fish

    sns.boxplot(x="Species", y=var, data=fish,palette='rainbow')

    
fig, ax = plt.subplots(2, 3,figsize=(20,15))

plt.subplot(2,3,1)

box('Weight')

plt.subplot(2,3,2)

box('Length1')

plt.subplot(2,3,3)

box('Length2')

plt.subplot(2,3,4)

box('Length3')

plt.subplot(2,3,5)

box('Height')

plt.subplot(2,3,6)

box('Width')
sns.pairplot(data=fish)
sns.pairplot(data=fish,hue='Species')

plt.title('pairwise relationships in a dataset')
sns.heatmap(fish.corr(),cmap='coolwarm',annot=True,linecolor='white',linewidths=4)
fish.drop(['Length1','Length2','Length3'],axis=1,inplace=True)
fish.head()
fish.head()
filt_df = fish.iloc[:,1:]
#computing percentiles

low = .05

high = .95

quant_df = filt_df.quantile([low, high])

print(quant_df)
filt_df = filt_df.apply(lambda x: x[(x>quant_df.loc[low,x.name]) & 

                                    (x < quant_df.loc[high,x.name])], axis=0)
filt_df = pd.concat([fish.iloc[:,0], filt_df], axis=1)
filt_df.info()
filt_df.describe()
fish =filt_df.dropna()
X = fish.drop('Weight',axis=1)

y=fish['Weight']
f_type =pd.get_dummies(X['Species'],drop_first=True)
f_type.head()
X.drop('Species',axis=1,inplace=True)
X = pd.concat([X,f_type],axis=1)
X.head()
#spliting the dataset into training set and test set

from sklearn.model_selection import train_test_split

X_train ,X_test , y_train , y_test =train_test_split(X,y, test_size = 0.2 , random_state=4)
from sklearn.linear_model import LinearRegression

#Create an instance of a LinearRegression() model named lm.

lm=LinearRegression()

# fit the model

lm.fit(X_train,y_train)
print('Coefficients: \n', lm.coef_)
print('intercept: \n',lm.intercept_)
s=pd.DataFrame(X_train)

s.head()
# Predicting the Test set results

predictions = lm.predict( X_test)

plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
# calculate these metrics by hand!

from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
sns.distplot((y_test-predictions),bins=20);
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
print('intercept: \n',lm.intercept_)