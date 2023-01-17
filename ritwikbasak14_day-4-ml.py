# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

curve = pd.read_csv("../input/curvecsv/curve.csv")

curve.head()
def fit_poly(degree):

    p = np.polyfit(curve.x, curve.y, deg = degree)

    curve['fit'] = np.polyval(p, curve.x)

    sns.regplot(curve.x, curve.y, fit_reg = False)

    return plt.plot(curve.x, curve.fit, label='Fitting')
fit_poly(5)

plt.xlabel("x values")

plt.ylabel("y values")
curve.head()
from sklearn.model_selection import train_test_split

from sklearn import metrics



X_train, X_test, y_train, y_test = train_test_split(curve.x, curve.y,test_size=0.2,random_state=10)



rmse = {

    'degree' : [],

    'rmse_train' : [],

    'rmse_test' : []

}



for degree in range(1,15):

    p = np.polyfit(X_train, y_train, deg=degree)

    rmse['degree'].append(degree)

    rmse['rmse_train'].append(metrics.mean_squared_error(y_train, np.polyval(p, X_train)))

    rmse['rmse_test'].append(metrics.mean_squared_error(y_test, np.polyval(p, X_test)))

rmseDf = pd.DataFrame(rmse)
rmseDf
plt.plot(rmseDf.degree,rmseDf.rmse_train,label='RMSE_TRAIN',c='red')

plt.plot(rmseDf.degree,rmseDf.rmse_test,label='RMSE_TEST',c='green')

plt.xlabel("Degree")

plt.ylabel("RMSE")

plt.legend()



import pandas as pd

student = pd.read_csv("../input/student.csv")

student.shape
student = student.drop('Name',axis='columns')



student



test_data = [6.5 ,1]



import numpy as np



lstDist = []



for index in student.index:

    distance = np.sqrt((test_data[0] - student['Aptitude'][index])**2 + (test_data[1] - student['Communication'][index])**2)

    lstDist.append([distance, student['Class'][index]])

    

df = pd.DataFrame(lstDist,columns=['Distance','Class'])

df_sorted = df.sort_values('Distance')

n = 4

df_sorted.head(n)


