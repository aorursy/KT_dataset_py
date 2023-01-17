# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import warnings

import warnings

# ignore warnings

warnings.filterwarnings("ignore")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

data=pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')

# df=pd.DataFrame(data)

data.head()
data.info()
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],

                                       c=color_list,

                                       figsize= [12,12],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()
sns.countplot(x="class", data=data)

data.loc[:,'class'].value_counts()
data1 = data[data['class'] == 'Abnormal']

x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1)

y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)
plt.figure(figsize=[10,10])

plt.scatter(x,y)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
# LinearRegression

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

# Predict space

predict_space = np.linspace(min(x), max(x)).reshape(-1,1)

# Fit

reg.fit(x,y)

predicted = reg.predict(predict_space)

y_head=reg.predict(x)



#%% rsquare

residual=y-y_head

SSR=sum(sum(residual**2))

print("SSR", SSR)

y_avg=sum(y)/len(y)

SST=sum((y-y_avg)**2)

print("SST",SST)

Rsq=1-(SSR/SST)

print("Rsq",Rsq)



#%% rsquare functionally

from sklearn.metrics import r2_score

print("r_score=",r2_score(y,y_head))



# R^2 from sklearn linear regression score

print('R^2 score: ', reg.score(x, y))
# Plot regression line and scatter

plt.figure(figsize=(15,10))

plt.plot(predict_space, predicted, color='black', linewidth=4)

plt.scatter(x=x,y=y)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()