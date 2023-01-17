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
data = pd.read_csv("/kaggle/input/titanic/train.csv")
sum(data.Survived) / len(data)
data.Pclass.value_counts()
data.groupby("Pclass").sum() / data.groupby("Pclass").count()
import seaborn as sb

sb.lineplot(x = "Pclass", y = "Survived", data = data, hue = "Sex")
data.describe()
gender = data.Sex == "male"

data["Gender"] = gender
X = data.fillna(data.mean())
y = X.pop("Survived")
df = X[X.describe().columns]
passengerid = df.pop("PassengerId")
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = 1000).fit(df, y)
sb.scatterplot(x = model.predict(df), y = y)