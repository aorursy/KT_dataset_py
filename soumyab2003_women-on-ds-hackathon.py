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
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
train_df=pd.read_csv("../input/womenintheloop-data-science-hackathon/train.csv")
test_df=pd.read_csv("../input/womenintheloop-data-science-hackathon/test_QkPvNLx.csv")
train_df.shape
train_df.head()
train_df["Sales"].plot(kind = 'density')
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train_df["Sales"], "log(1+price)":np.log1p(train_df["Sales"])})
prices.hist()
train_df.isnull().sum()
train_df.Competition_Metric.isnull().sum()
percentagemissing=(train_df.Competition_Metric.isnull().sum()/len(train_df))*100
percentagemissing
train_df=train_df.dropna(inplace=False)
train_df=train_df.reset_index(drop=True)
train_df.isnull().sum()
train_df.corr()
corrMatrix = train_df.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

train_df.Day_No.nunique()
grouped=train_df.groupby(['ID','Course_ID','Course_Domain']).Sales.sum()
grouped
grouped=grouped.to_frame()
grouped
print(grouped.Sales.max())
print(grouped.Sales.idxmax())
grouped
train_df.head()
train_df['Day_of_week_v'] = train_df['Day_No']%7
test_df['Day_of_week_v']=test_df['Day_No']%7
Trial_df=train_df[['Course_Domain','Course_Type','Short_Promotion','Public_Holiday','Long_Promotion','Sales','Competition_Metric','Day_of_week_v']]
Trial_df.head()
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
Trial_df = pd.get_dummies(Trial_df, prefix_sep='_', drop_first=True)

Trial_df.head()
X=Trial_df[['Long_Promotion','Public_Holiday','Short_Promotion','Course_Domain_Development','Course_Domain_Finance & Accounting','Course_Domain_Software Marketing','Course_Type_Degree','Course_Type_Program','Competition_Metric','Day_of_week_v']]
y=np.log1p(Trial_df.Sales)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.ensemble import RandomForestRegressor 

rf= RandomForestRegressor(n_estimators = 20, random_state = 0) 
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
y_train_predicted = rf.predict(x_train)


# The mean squared error
score = mean_squared_log_error(np.expm1(y_test), np.expm1(y_pred))
error=1000*score
error
y_pred[0:5]
y_test[0:5]
from sklearn.model_selection import cross_val_predict

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Ground Truth vs Predicted")
plt.show()
test_df.head()
test_df.isnull().sum()
test_df=test_df.dropna(inplace=False)

test_df = test_df.reset_index(drop=True)

test1_df=test_df[['Course_Domain','Course_Type','Short_Promotion','Public_Holiday','Long_Promotion','Competition_Metric','Day_of_week_v']]
test1_df = pd.get_dummies(test1_df, prefix_sep='_', drop_first=True)

test1_df.head()
x_test=test1_df[['Long_Promotion','Public_Holiday','Short_Promotion','Course_Domain_Development','Course_Domain_Finance & Accounting','Course_Domain_Software Marketing','Course_Type_Degree','Course_Type_Program','Competition_Metric','Day_of_week_v']]
y_pred_test=rf.predict(x_test)
y_pred_test
result = pd.DataFrame({"ID": test_df['ID'], 'Sales': np.expm1(y_pred_test)})
result
result.plot(x='ID',y='Sales')
