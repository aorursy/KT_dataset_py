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
weather = pd.read_csv('/kaggle/input/weatherww2/Summary of Weather.csv')

weather.head()
temp = weather[['MaxTemp','MinTemp']]

temp.describe()
temp.isnull().sum()
import matplotlib.pyplot as plt 

import seaborn as sns

sns.set()

plt.scatter(temp['MaxTemp'],temp['MinTemp'])

plt.xlabel('Maximum Temperature', fontsize = 13)

plt.ylabel('Minimum Temperature', fontsize = 13)

plt.title('Min Max Relationship', fontsize = 20)

plt.show()
sns.distplot(temp['MaxTemp'])
outliers = temp['MaxTemp'].quantile(0.01)

min_max = temp[temp['MaxTemp']>outliers]
sns.distplot(min_max['MaxTemp'])
sns.distplot(temp['MinTemp'])
outliers1 = temp['MinTemp'].quantile(0.01)

min_max = temp[temp['MinTemp']>outliers1]

sns.distplot(min_max['MinTemp'])
min_max_data = min_max.reset_index(drop=True)

min_max.describe()
plt.scatter(min_max_data['MaxTemp'],min_max_data['MinTemp'])

plt.xlabel('Maximum Temperature', fontsize = 13)

plt.ylabel('Minimum Temperature', fontsize = 13)

plt.title('Min Max Relationship', fontsize = 20)

plt.show()
x = min_max_data['MinTemp']

x_matrix = x.values.reshape(-1,1)

y = min_max_data['MaxTemp']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_matrix, y, test_size=0.2, random_state=200)
from sklearn.linear_model import LinearRegression
r = LinearRegression()

r.fit(x_train,y_train)
r.score(x_train,y_train)
r.intercept_
r.coef_
plt.scatter(x_matrix,y)

yhat = x_matrix*0.87961507+11.474340834273745

fig = plt.plot(x_matrix,yhat, lw=4, c='orange', label ='regression line')

plt.xlabel('Minimun Temperature', fontsize = 20)

plt.ylabel('Maximum Temperature', fontsize = 20)

plt.show()
y_hat = r.predict(x_train)
y_hat_test = r.predict(x_test)
test_predictions= pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])

test_predictions
y_test = y_test.reset_index(drop=True)
test_predictions['Target'] = np.exp(y_test)

test_predictions.head()
