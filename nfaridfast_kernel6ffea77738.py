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
data = pd.read_csv('../input/internet-advertisements-data-set/add.csv', index_col=0,low_memory=False)
data.head(5)
data.head()
data.info()
data['0'] = pd.to_numeric(data['0'], errors='coerce')
data['1'] = pd.to_numeric(data['1'], errors='coerce')
data['2'] = pd.to_numeric(data['2'], errors='coerce')
data['3'] = pd.to_numeric(data['3'], errors='coerce')
data = data.dropna()
data[['0','1','2','3']] = data[['0','1','2','3']].astype('int64')
y = data['1558']
data['1558'].values
data['1558'].eq('ad.').mul(1)
x = data.drop(columns=['1558'])
x.info()
from sklearn import model_selection 
from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier

accuracies = []
for i in range(1,11):
    bagPercentSize = 10*i
    tenfold = model_selection.KFold(n_splits=10)
    model = BaggingClassifier(base_estimator=DecisionTreeClassifier())
    results = model_selection.cross_val_score(model,x,y,cv=tenfold)
    print ("baf prct size: %d, the accuracy is %f" %(bagPercentSize,results.mean()))
    accuracies.append([bagPercentSize,results.mean()])
final = pd.DataFrame(accuracies,columns=['bagPercentSize','accuracies'])
final.plot(x='bagPercentSize',y='accuracies')
