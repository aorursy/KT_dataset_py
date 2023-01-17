import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline
advert = pd.read_csv('/kaggle/input/advertising-dataset/advertising.csv')

advert.head()
advert.info()
advert.columns
import seaborn as sns

sns.distplot(advert.Sales)
sns.distplot(advert.Newspaper)
sns.distplot(advert.Radio)
sns.distplot(advert.TV)
sns.pairplot(advert, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=7, 

            aspect=0.7, kind='reg')
# Correlation value between TV and Sales

advert.TV.corr(advert.Sales)
# Correlation value between Radio and Sales

advert.Radio.corr(advert.Sales)
# Correlation value between Newspaper and Sales

advert.Newspaper.corr(advert.Sales)
advert.corr()
# Heatmap for showing the correlation values

sns.heatmap(advert.corr(), annot=True)
X = advert[['TV']]

X.head()
print(type(X))

print(X.shape)
y = advert.Sales

print(type(y))

print(y.shape)
# Train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# Displaying the shape of each of the train test dataframe

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# Creating the model

from sklearn.linear_model import LinearRegression



linreg = LinearRegression()

linreg.fit(X_train, y_train)
print(linreg.intercept_)

print(linreg.coef_)
y_pred = linreg.predict(X_test)

y_pred[:5]
from sklearn import metrics
# Calculating RMSE (Root mean Squared Error)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))