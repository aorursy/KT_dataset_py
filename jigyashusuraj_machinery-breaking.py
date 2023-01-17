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
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel(r"../input/machinery-breaking-analysis/Combined_Cycle_powerplant.xlsx")
df.shape
df.columns
df.head()
df.info()
df.describe()
#Check for missing values
df.duplicated().sum()
#Drop duplicates
df.drop_duplicates(inplace=True)
#check for missing values
df.isnull().sum()
#it comapres all column with all the columns and shows the graph
sns.pairplot(df)
plt.show()
cor=df.corr()
plt.figure(figsize=(9,7))
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()
#Separate features and label
x = df[['AT','V','AP','RH']]
y = df[['PE']]
# split data into train and test
from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts = train_test_split(x,y,test_size=0.2)
# we have to split the data into 80% as train and 20% as test so we have specified test_size as 0.2
print(x.shape)
print(xtr.shape)
print(xts.shape)
print(y.shape)
print(ytr.shape)
print(yts.shape)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
#train the model with the training data
model.fit(xtr,ytr)
new_data=np.array([[13.97,39.16,1016.05,84.6]])
model.predict(new_data)
#get prediction of xts
ypred = model.predict(xts)

#calculating r2score
from sklearn.metrics import r2_score
r2_score(yts,ypred)

#To find the error
from sklearn.metrics import mean_squared_error
mean_squared_error(yts,ypred)
import joblib
#from sklearn.externals import joblib
joblib.dump(model,r"..\output\kaagle\working\ccpp_model.pkl")
