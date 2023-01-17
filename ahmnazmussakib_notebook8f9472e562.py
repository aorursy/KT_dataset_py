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
train_data = pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")
test_data = pd.read_csv("../input/health-insurance-cross-sell-prediction/test.csv")
print(train_data.head(10))
X_init = train_data.drop(columns = 'Response')
y = train_data["Response"]
X
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
X_init["_Gender"] = lb_make.fit_transform(X_init["Gender"])
X_init["_Region_Code"] = lb_make.fit_transform(X_init["Region_Code"])
X_init["_Vehicle_Age"] = lb_make.fit_transform(X_init["Vehicle_Age"])
X_init["_Vehicle_Damage"] = lb_make.fit_transform(X_init["Vehicle_Damage"])
X_init["_Policy_Sales_Channel"] = lb_make.fit_transform(X_init["Policy_Sales_Channel"])

vintage_bins = [0,30,60,90,120,150,180,210,240,270,300]
vintage_labels = [1,2,3,4,5,6,7,8,9,10]
X_init["_Vintage"] = pd.cut(X_init["Vintage"], bins = vintage_bins, labels = vintage_labels)

age_bins = [0,20,30,40,50,60,70,80,90]
age_labels = [1,2,3,4,5,6,7,8]
X_init["_Age"] = pd.cut(X_init["Age"], bins = age_bins, labels = age_labels)

from scipy import stats
X_init["Premium"] = np.round(stats.zscore(x["Annual_Premium"]))

x = X_init.drop(columns = ["Gender","Region_Code","Vehicle_Age","Vehicle_Damage","Policy_Sales_Channel","Vintage","Age","Annual_Premium"])

x
X = x.drop(columns=["id"])
#create model

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#initialize cross-validation

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=10, random_state=1, shuffle=True)


#train model

scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

print(scores)
from scipy import stats

np.round(stats.zscore(x["Annual_Premium"]))
