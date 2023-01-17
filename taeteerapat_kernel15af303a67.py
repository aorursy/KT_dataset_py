import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


filename='/kaggle/input/top50spotify2019/top50.csv'
T=pd.read_csv(filename,encoding='ISO-8859-1')
T.head()
filename='/kaggle/input/top50spotify2019/top50.csv'
spoti=pd.read_csv(filename,encoding='ISO-8859-1')
spoti.head(50)
select_d = ['Artist.Name','Genre','Beats.Per.Minute','Length.','Loudness..dB..','Popularity','Valence.']
spoti = spoti[select_d]
spoti.head()
spoti.describe()
skew=T.skew()
print(skew)
# Removing the skew by using the boxcox transformations
transform=np.asarray(T[['Popularity']].values)

# Plotting a histogram to show the difference 
plt.hist(T['Popularity'],bins=10) #original data
plt.show()


# Check correlations
sns.heatmap(spoti.corr(), annot=True)
categories = ['Artist.Name','Genre']
spoti1 = pd.get_dummies(spoti.copy(), columns=categories,drop_first=True)
X = spoti1.drop(columns=['Popularity'],axis=1)
y = spoti['Popularity']

# แบ่ง X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=20)
dataTrain = LinearRegression()  
dataTrain.fit(X_train, y_train) #training the algorithm
y_pred = dataTrain.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))