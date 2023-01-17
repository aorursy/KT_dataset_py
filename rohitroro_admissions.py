import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score,accuracy_score, mean_squared_error
import numpy as np
import seaborn as sns
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
lst=[]
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        lst.append(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv(lst[1])
data = data.drop(columns = ["Serial No."])
data.describe()   # Data description

data.isnull().sum() #Nulls?
#Distribution of all attributes
fig,axes = plt.subplots(2,3, figsize = (10,5))
axes[0][0].hist(data["GRE Score"])
axes[0][0].set_xlabel("GRE Score")
axes[0][1].hist(data["TOEFL Score"])
axes[0][1].set_xlabel("TOEFL Score")
axes[0][2].hist(data["University Rating"])
axes[0][2].set_xlabel("University Rating")
axes[1][0].hist(data["SOP"])
axes[1][0].set_xlabel("SOP")
axes[1][1].hist(data["LOR "])
axes[1][1].set_xlabel("LOR ")
axes[1][2].hist(data["CGPA"])
axes[1][2].set_xlabel("CGPA")
fig.tight_layout()
#CGPA and LOR have outliers, they do not lie very far away from IQR but still no extra data points nearby, can be cause
#of noise removing them

# data.drop(data[data["CGPA"] < 7.0].index, inplace = True)
# data.drop(data[data["LOR "] < 1.5].index, inplace = True)
data.corr().tail(1)
X=data[['GRE Score','TOEFL Score','CGPA','University Rating','SOP',"LOR ","Research"]]
y=data["Chance of Admit "]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=50)
Linear=LinearRegression()
Linear.fit(X_train,y_train)
y_pred=Linear.predict(X_test)
print("mean_absolute_error  of the model is ",mean_absolute_error(y_pred,y_test))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df["Error"] = df["Actual"] - df["Predicted"]
sns.distplot(df["Error"])
validation = data.tail(100)
pred = Linear.predict(validation[['GRE Score','TOEFL Score','CGPA','University Rating','SOP',"LOR ","Research"]])
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(validation["Chance of Admit "], pred)))