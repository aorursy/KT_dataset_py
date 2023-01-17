import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv("../input/FIFA 2018 Statistics.csv")

data.shape # colomn과 row의 수
data.head(5)
gc = np.array([], dtype = 'int')

for i in range(0, 128, 2):
    gc = np.append(gc, data.loc[i+1, "Goal Scored"])
    gc = np.append(gc, data.loc[i, "Goal Scored"])
    
data.insert(4, "Goal Conceded", pd.Series(gc))
data.head(5)
condition = [(data['Goal Scored'] > data['Goal Conceded']), (data['Goal Scored'] == data['Goal Conceded']), 
             (data['Goal Scored'] < data['Goal Conceded'])]
result = np.array([0, 1 , 2], dtype = 'int')
data.insert(5, 'Result', pd.Series(np.select(condition, result, default = -1)))
data.head(5)
