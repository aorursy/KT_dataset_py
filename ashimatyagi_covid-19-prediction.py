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
df = pd.read_csv('/kaggle/input/india-coronavirus/Monthly covid-19 India.csv')
df
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df
import matplotlib.pyplot as plt
plt.bar(df['Days'], df['Total Confirmed'], color = (0.5,0.1,0.5,0.6))

plt.xlabel('No. of Days')
plt.ylabel('Total Confirmed cases')
plt.show()


import seaborn as sns
sns.jointplot(x='Days',y='Daily Confirmed',data=df,kind='scatter', color='r')


from sklearn import linear_model
X = df['Days']
y=df['Daily Confirmed']
X=X.values.reshape(-1,1)
y=y.values.reshape(-1,1)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)
X_test= np.arange(115,133)
print(X_test)
print(len(X_test))
X_test=X_test.reshape(18,1)
X_test
#y_pred is the prediction daily cases
y_pred= regressor.predict(X_test)
y_pred
regressor.score(X_test, y_pred)
#reshaping the 2d array to 1d array
X_test = X_test.reshape(-1)
y_pred= y_pred.reshape(-1)

dailycases_predicted = pd.DataFrame(y_pred, X_test)
dailycases_predicted
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(X_test.astype(str), y_pred, color='blue', linewidth=3)

plt.xlabel('No. of Days')
plt.ylabel('Predicted Daily Confirmed Cases')

#adding predictions and X_test data to the file
df2.to_csv("Covid_predictions.csv")
x2_train = np.array([[44],[50],[55],[60],[65],[65],[70],[75],[80],[85],[90],[100],[105],[110]])
y2_train = np.array([[3656],[4311],[3808],[5720],[6414],[8364],[9847],[9981],[11405],[14740],[16868],[18339],[24018],[25790]])
regressor.score(x2_train,y2_train)