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
import matplotlib.pyplot as plt              # Using 'as' after import allows you to use an alias for the library

import seaborn as sns                        # matplotlib,seaborn are data visualisation libraries

sns.set(color_codes = True)

df = pd.read_csv('/kaggle/input/iris/Iris.csv',index_col = 'Id')

df.head() 
df.info() # getting more information on the data.
# checking if the data has any missing values, in this case there are none.

df.isnull().sum()
# making a pie chart showing the distribution of species

#f , ax = plt.subplots(1,2,figsize = (18,8))

df['Species'].value_counts().plot.pie(shadow = True,autopct = '%1.2f%%',figsize = (10,8))
# Swarmplot, using data visialisation helps us to judge the data and come up with the best possible models .

fig= plt.gcf()

fig.set_size_inches(10,6)

fig = sns.swarmplot(data = df,x = 'Species',y= 'PetalWidthCm')
from sklearn import metrics                            # Helps calculate accuracy of our models

from sklearn.model_selection import train_test_split   # Splitting data for data analysis



x = df.iloc[:,:-1]

y = df.iloc[:,-1]



# Splitting the dataset into train_test models

x_train , x_test ,y_train , y_test = train_test_split(x,y,test_size=0.3 )
# The model I plan on using is support vector classifier

from sklearn.svm import SVC

svc = SVC(kernel = 'rbf')

svc.fit(x_train,y_train)

prediction = svc.predict(x_test)

x = metrics.accuracy_score(y_test, prediction)

print(x)