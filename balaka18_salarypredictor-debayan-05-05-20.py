# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as ml

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

%matplotlib inline

ml.style.use('ggplot')



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')

print(data.shape)

data.head()
data.info()
data.describe()
sns.boxplot(x='YearsExperience',data=data)

plt.show()



sns.boxplot(x='Salary',data=data)

plt.show()



fig = go.Figure()

fig.add_trace(go.Scatter(x=data.YearsExperience, y=data.Salary,

                    mode='lines+markers'))

fig.update_layout(title='SALARY VS YEAR_OF_EXPERIENCE',

                   xaxis_title='YearsExperience',

                   yaxis_title='Salary')

fig.show()
data.corr()
X = np.array(data.YearsExperience).reshape(-1,1)

Y = np.array(data.Salary).reshape(-1,1)



trainx,testx,trainy,testy = train_test_split(X,Y,test_size=0.2,random_state=17)

print("Train X size = ",trainx.shape,", Test X size = ",testx.shape)

print("Train Y size = ",trainy.shape,", Test Y size = ",testy.shape)

lr = LinearRegression(fit_intercept=True)

lr.fit(trainx,trainy)
y_pred = lr.predict(testx)

print("Mean Squared error = ",mean_squared_error(testy,y_pred,squared=True))

print("R^2 score = ",r2_score(testy,y_pred))