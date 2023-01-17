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

data1= pd.read_csv('/kaggle/input/emg-4/0.csv',header=None)
data2= pd.read_csv('/kaggle/input/emg-4/1.csv',header=None)
data3= pd.read_csv('/kaggle/input/emg-4/2.csv',header=None)
data4= pd.read_csv('/kaggle/input/emg-4/3.csv',header=None)
dfs=[data1,data2,data3,data4]
data = pd.concat(dfs)

y = data[64]
x = data.drop(64,axis=1)

def Grad_boost(x,y):
  from sklearn.ensemble import GradientBoostingClassifier
  import numpy as np
  from sklearn.model_selection import GridSearchCV
  param_grid2 = [
    {'min_samples_split': np.array([7]),
    'max_depth': np.array([4]),
    'min_samples_leaf':np.array([5]),
    'learning_rate': np.array([0.41]),
    'criterion':['friedman_mse']
    }
  ]
  modelo = GradientBoostingClassifier(n_estimators=50)
  gridGradientBoostingClassifier= GridSearchCV(estimator=modelo,param_grid=param_grid2,cv=3,n_jobs=-1)
  gridGradientBoostingClassifier.fit(x,y)
  print('Mínimo split:', gridGradientBoostingClassifier.best_estimator_.min_samples_split)
  print('Máxima profundidade:', gridGradientBoostingClassifier.best_estimator_.max_depth)
  print('Mínimo leaf:', gridGradientBoostingClassifier.best_estimator_.min_samples_leaf)
  print('Mínimo learning rate:', gridGradientBoostingClassifier.best_estimator_.learning_rate)
  print('Mínimo criterion:', gridGradientBoostingClassifier.best_estimator_.criterion)    
  print('R2:',gridGradientBoostingClassifier.best_score_)
Grad_boost(x,y)