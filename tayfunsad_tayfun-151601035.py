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
import pandas as pd
import seaborn as sns
import numpy as np
covd_19 = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200131.csv")
covd_19
print(covd_19.columns)
covd_19.info()
covd_19.head(20)
print(covd_19.groupby("Confirmed").size())
covd_19.tail(20)
print(covd_19.describe())
covd_19.isnull().sum()
print(covd_19.corr())
covd_19.describe()
sns.pairplot(covd_19)
covd_19['Recovered'].mean()
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
x=covd_19["Confirmed"].replace(np.NaN, covd_19["Confirmed"].mean())
y=covd_19["Recovered"].replace(np.NaN, covd_19["Recovered"].mean())
x=np.array(x)
y=np.array(y)
reg = linear_model.LinearRegression(normalize='Ture')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)


reg.fit(X_train, y_train)

print(reg.score(X_train, y_train))
sns.regplot(x=X_train,y=y_train)
covd_19.isna().sum() 
country_details 
province_state_country = pd.pivot_table(covd_19,index=["Province/State"] ,aggfunc=np.sum).sort_values(by='Confirmed', ascending=False)
province_state_country[:10]
province_state_country[:10].plot(kind='bar' ,figsize=(10, 4), width=1,zorder=12)
province_state_country[:5].plot(kind='pie', subplots=True, figsize=(100, 100))
province_state_country[1:10].plot(kind='bar' ,figsize=(10, 4), width=2)
country_details[0:5].plot(kind='bar' ,figsize=(10, 4), width=1,zorder=12,rot=2)
country_details[1:6].plot(kind='bar' ,figsize=(10, 4), width=1,zorder=12,rot=1)