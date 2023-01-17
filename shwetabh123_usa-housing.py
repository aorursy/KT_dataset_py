# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/USA_Housing.csv")
data1=data.to_parquet('output.parquet')

#import matplotlib.pyplot as plt

#data.head()

data1

#%matplotlib inline
import seaborn as  sns
#data.columns

sns.distplot(data['Price'])
import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline
df =pd.read_csv('../input/USA_Housing.csv')
df.info()
df.describe()
df.head
sns.pairplot(df)





sns.distplot(df['Price'])
sns.heatmap(df.corr() ,annot =True)
df.columns
x=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',

       'Avg. Area Number of Bedrooms', 'Area Population']]

y =df['Price']
from sklearn.model_selection import train_test_split
x_train,x_test ,y_train ,y_test =train_test_split(x ,y ,test_size =0.4,random_state =101)
from sklearn.linear_model import  LinearRegression
lm = LinearRegression()
lm.fit(x_train ,y_train)
print(lm.intercept_)
pd.DataFrame(lm.coef_ ,x.columns ,columns =['coeff'])
prediction =lm.predict(x_test)
prediction
y_test
plt.scatter(y_test ,prediction)
sns.distplot((y_test-prediction))
from sklearn import metrics
metrics.mean_absolute_error(y_test ,prediction)
metrics.mean_squared_error(y_test ,prediction)
np.sqrt(y_test ,prediction)