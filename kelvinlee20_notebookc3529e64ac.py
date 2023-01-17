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
import pandas as pd

df = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')
df
df.info()
s = df.cut.value_counts()
s
s.plot.bar()
def f(s):
    if (s=='Fair'): return 1.0
    if (s=='Good'): return 2.0
    if (s=='Very Good'): return 3.0
    if (s=='Premium'): return 4.0
    if (s=='Ideal'): return 5.0

df['_cut'] = df.cut.apply(f)
df.head(3)
df.carat.hist(bins=100)
df.price.hist(bins=100)
import matplotlib.pyplot as plt

plt.scatter(df.carat, df.price)
import seaborn as sns

plt.figure(figsize=(15,8))
sns.scatterplot(data=df, x="carat", y="price", hue="cut", linewidth=0, s=9, alpha=0.5)
import seaborn as sns
sns.boxplot(x="cut", y="price", data=df)
import seaborn as sns
sns.boxplot(x="color", y="price", data=df)
df.corr()
features = df[['carat', 'x', 'y', 'z']]
features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform(features) 
features.loc[:,:] = scaled_values
features
label = df[['price']]
label
from sklearn.model_selection import train_test_split 
X = features
y = label
Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.3)
from sklearn.linear_model import LinearRegression
from sklearn import metrics

regressor = LinearRegression()  
regressor.fit(Xtr, ytr) #training the algorithm

#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)
from sklearn.metrics import classification_report, confusion_matrix  
predicted = regressor.predict(Xte)
import numpy as np

print('Mean Absolute Error:', metrics.mean_absolute_error(yte, predicted))  
print('Mean Squared Error:', metrics.mean_squared_error(yte, predicted))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yte, predicted)))
predicted_df = pd.DataFrame({'Actual': yte['price'], 'Predicted': predicted.flatten()})
predicted_df