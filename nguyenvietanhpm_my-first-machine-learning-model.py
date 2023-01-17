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
import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn import preprocessing

from sklearn import model_selection
df = pd.read_csv('/kaggle/input/air-quality-csv-file-sep/AirQualityUCI.csv',sep = ';')
df.head()
for col in df.columns:

    count = 0

    for value in df.isnull()[col]:

        if value ==True:

            count +=1

    print(col,count,sep = ' ')
df.drop(['Unnamed: 15','Unnamed: 16'],axis = 1,inplace = True)

df.dropna(inplace = True)

df.isnull().any()
df.dtypes
df.head()
cols = ['CO(GT)','C6H6(GT)','T','RH','AH']

for col in cols:

    df[col].replace(df[col].values,df[col].str.replace(',','.').values.astype(np.float64),inplace = True)
df.dtypes
def distribution_plot(data,cols):

    for col in cols:

        sns.set_style('darkgrid')

        sns.distplot(data[col])

        plt.title('Distribution in ' + col)

        plt.show()

distribution_plot(df,['CO(GT)','PT08.S1(CO)','NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)',

                      'NOx(GT)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH'])
X = preprocessing.scale(df.drop(['Date','Time','AH'],axis = 1))

y = preprocessing.scale(df['AH'])

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.5)
model = LinearRegression(n_jobs= 1,normalize = True)

model.fit(X_train,y_train)

predictions = model.predict(X_test)
model.score(X_test,y_test)