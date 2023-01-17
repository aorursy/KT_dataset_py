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
import numpy as np

from sklearn import preprocessing, metrics

retweet = pd.read_csv("/kaggle/input/turkishnewstwitter/retweet_kaggle.csv")
giris = pd.read_csv("/kaggle/input/turkishnewstwitter/giris_kaggle.csv")

giris['retweet'] = retweet["retweet_toplam"]
dataset = giris.copy()
dataset=dataset[dataset["retweet"]<30]
dataset=dataset[dataset["wiki_deger"]<45000]


X = dataset.iloc[:,1:]
X = X.drop(['retweet'], axis=1)

y = dataset.iloc[:,-1]
y = pd.DataFrame(y)


X.head()
y.head()
###############################################################
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
def compute_score(y_pred, y_test): 
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    var = explained_variance_score(y_test, y_pred)
    r2sc = r2_score(y_test, y_pred)
    
    print("Mean Squarred Error: %.4f" % mse)
    print("Root Mean Squarred Error: %.4f" % rmse)
    print("Mean Absolute Error: %.4f" % mae)
    print("Variance Score (Best possible score is 1): %.4f" % var)
    print("R2Score (Best possible score is 1): %.4f" % r2sc)

###################################################################

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
from sklearn.ensemble import RandomForestRegressor
regrRM2 = RandomForestRegressor(n_estimators=200, max_depth = 50, min_samples_split = 5,min_samples_leaf =4, random_state=42)
regrRM2.fit(x_train, y_train)
regrRM2_score = regrRM2.score(x_train, y_train)
print(regrRM2_score)
compute_score(y_test, y_pred)