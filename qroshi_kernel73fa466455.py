# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/train.csv.zip', index_col="id")
data
data.info()
data.describe()
data["color"].hist()
data["type"].astype('category')
data["type"] = data["type"].astype('category') 
data["color"] = data["color"].astype('category') 
data["color"]
for color in data["color"].cat.categories: 
    print("Color: ", color.upper())
    for monster_type in data["type"].cat.categories: 
        print("\t", monster_type, data.query('color == @color & type == @monster_type')["type"].count())
data.loc[(data["color"] == "green") & (data["type"] == "Ghost"), "type"].count() / data.loc[data["color"] == "green", "type"].count()
#green ghosts.. strange, isn't it? 
#color splits monsters into 3 nearly equal parts so I think it is useless
data.drop("color", axis=1, inplace=True)  
data
from sklearn.linear_model import SGDClassifier 
from sklearn.model_selection import cross_val_score 
from sklearn.ensemble import RandomForestClassifier

print(cross_val_score(SGDClassifier(), data.drop('type', axis=1), data["type"], cv=3)) 
print(cross_val_score(RandomForestClassifier(), data.drop(['type'], axis=1), data["type"], cv=3)) 
linear = SGDClassifier(early_stopping=True, max_iter=10000, learning_rate='adaptive', eta0=2)
linear.fit(data.drop('type', axis=1), data['type'])
linear.score(data.drop('type', axis=1), data['type'])
test_data = pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/test.csv.zip', index_col="id")
test_data
predictions = linear.predict(test_data.drop("color", axis=1))


pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/sample_submission.csv.zip')
result = pd.DataFrame({'id': test_data.index,
                       'type': predictions})
result.to_csv("monsters_submission.csv", index=False)
result["type"]