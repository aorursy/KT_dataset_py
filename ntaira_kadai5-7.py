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
#課題5
data = pd.read_csv("../input/bostonhoustingmlnd/housing.csv")
print(data)
#課題6
df = pd.read_csv("../input/pandasdata/pandasdata2.csv", skiprows=range(1,11))
df.to_csv("/kaggle/working/pandasdata3.csv", index=False, columns=["age", "hobby"], encoding="shift_jis")
df2 = pd.read_csv("/kaggle/working/pandasdata3.csv", encoding="shift_jis")
print(df2)
#課題7
df = pd.read_csv("../input/titanic/train.csv")
df = df.dropna(subset=["Embarked"])
dp_cols = ["Name", "Ticket", "Cabin"]
df = df.drop(dp_cols, axis=1)
df.to_csv("/kaggle/working/train2.csv", index=False)
df2 = pd.read_csv("/kaggle/working/train2.csv")
print(df2)
#課題8
df = pd.read_csv("/kaggle/working/train2.csv")
