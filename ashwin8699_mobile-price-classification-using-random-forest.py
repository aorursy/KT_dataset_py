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
train = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')
test = pd.read_csv('/kaggle/input/mobile-price-classification/test.csv')

train.head()
train['power per wt'] = train['battery_power']/train['mobile_wt']
train['wt per vol'] = train.apply(lambda x: x['mobile_wt']/(x['sc_h']*x['sc_w']*x['m_dep']) if x['sc_w']!=0 else 0 , axis = 1)
train['power per core'] = train['battery_power']/train['n_cores']
train['pixels'] = train['px_height']*train['px_width']
train['area'] = train['sc_h']*train['sc_w']
train['px_density'] = train.apply(lambda x:x['pixels']/x['area'] if x['area']!=0 else 0 , axis = 1)
train['sc_size'] = np.sqrt(train['sc_h']**2 + train['sc_w']**2)
train['time'] = train['battery_power']/train['talk_time']
train['power per clock'] = train['battery_power']/train['clock_speed']
train.isna().sum()
cols = ['battery_power','clock_speed','mobile_wt','n_cores','px_height','px_width','ram','sc_w','talk_time','power per wt','wt per vol', 'power per core','pixels','area','px_density','sc_size','time','power per clock',"price_range"] 
train2 = train[cols]
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
train , test = train_test_split(train2, test_size = 0.25)

x_train = train.drop('price_range', axis=1)
y_train = train['price_range']

x_test = test.drop('price_range', axis = 1)
y_test = test['price_range']
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
param_grid = {'n_estimators':range(10,110,10) , 'criterion':['gini', 'entropy'],
             'max_depth' : [1,2,3,4,5,6,7,8,9,10],'max_features':['auto', 'sqrt', 'log2']}

clf = RandomizedSearchCV(RandomForestClassifier(),param_grid,cv = 10,n_iter = 30)
clf = clf.fit(x_train, y_train)
from sklearn.metrics import classification_report,accuracy_score
accuracy_score(y_test , clf.predict(x_test))
from sklearn.metrics import classification_report,accuracy_score
accuracy_score(y_train , clf.predict(x_train))
