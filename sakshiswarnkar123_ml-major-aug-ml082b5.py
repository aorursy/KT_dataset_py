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
match = pd.read_csv('/kaggle/input/ipl2017/ipl2017.csv')
match.head(10)
match.info()
match1 = match.drop(["mid","date","venue"],axis=1)
match1.head(10)
match1.info()
match1["bat_team"].unique()
match1["bowl_team"].unique()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
match1.bat_team = le.fit_transform(match1.bat_team )
match1.bowl_team = le.fit_transform(match1.bowl_team )
match1.batsman = le.fit_transform(match1.batsman )
match1.bowler = le.fit_transform(match1.bowler )
match1.head(1000)
match1['bat_team'].unique()
match1['bowl_team'].unique()
match1['bowler'].unique()
match1['batsman'].unique()
Y = match1['total']
X = match1.drop('total',axis=1)
Y.head(10)
X.head(10)
X.info()
type(X)
type(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 42)
X

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(X_train,y_train)
model.score(X_test,y_test)
X_dataframe = X_train.tolist()                                  # Converting numpy array to list
X_dataframe = pd.DataFrame(X_train)                             # Coverting list to dataframe
feature_important = model.feature_importances_
feature_important
total = sum(feature_important)
new = [value * 100 / total for value in feature_important]
new = np.round(new,2)
keys = list(X_dataframe.columns)
feature_importances = pd.DataFrame()
feature_importances['Features'] = keys
feature_importances['Importance (%)'] = new
feature_importances = feature_importances.sort_values(['Importance (%)'],ascending=False).reset_index(drop=True)
feature_importances
dict2={0:[1,2,4,13,12,5,3,4,2,6],1:[4,3,11,14,1,10,1,5,13,2],2:[12,43,78,99,2,71,33,123,172,111],3:[21,88,65,69,33,34,100,200,126,220],4:[69,59,12,10,22,24,59,120,90,87],5:[1,3,5,7,9,0,2,4,6,8],6:[0.2,0.4,19.1,15.1,12.3,16.5,12.1,9.5,2.1,1.5],7:[55,10,21,43,22,33,44,55,11,10],8:[2,1,0,3,1,3,4,1,3,4],9:[20,0,11,33,14,99,12,23,4,55],10:[44,122,32,0,54,21,2,4,55,60]}
data2=pd.DataFrame(dict2)            
model.predict(data2)                                  # Predicting output on new dataset