# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for data visualization
import seaborn as sns #for data visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#load the from csv file
data=pd.read_csv('../input/column_2C_weka.csv')
#peek at the data first 5 rows
data.head()
#getting an overview of our data
data.info()
data.describe()
#unique class values
data['class'].unique()

#visualization
data['class'].value_counts()
#visalizaton
sns.countplot(data['class'])
#visualization
color_list=['purple' if i=='Abnormal' else 'aqua' for i in data.loc[:,'class']]
pd.plotting.scatter_matrix(data.loc[:,data.columns!='class'],
                           c=color_list,
                           figsize=[15,15],
                           diagonal='hist',
                           alpha=0.5,
                           s=200,
                           marker='.',
                           edgecolor='black')
plt.show()
data1=data[data['class']=='Normal']
x=np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1)
y=np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)
#scatter
plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y,color="purple")
plt.xlabel("pelvic incidence")
plt.ylabel("sacral slope")
plt.show()
#LinearRegression
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()

predict_space=np.linspace(min(x),max(x)).reshape(-1,1)
#fit
linear_reg.fit(x,y)
#predict
y_predict=linear_reg.predict(predict_space)
#r^2
print('r^2 score:',linear_reg.score(x,y))
#plot regression line and scatter
plt.figure(figsize=[10,10])
plt.plot(predict_space,y_predict,color="black",linewidth=3)
plt.scatter(x=x,y=y,color='purple')
plt.xlabel('pelvic incidence')
plt.ylabel('sacral slope')
plt.show()