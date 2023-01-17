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
array = np.array(range(1, 6))
print(array)
print(array ++ 1)
w = np.array(range(1,  5))
x = np.array(range(6, 10))
b = 4

# to [w1*x1 + w2*x2 + ... + b]
y = w.dot(x) + b

print(y)
array = np.array(
    [[1,  2,   3]
    ,[6, 50, 400]
    ,[5, 10, 100]]
)
print(array.sum(axis=0)[0] / 3)
print(np.average(array, axis=0)[0])
index = ["Taro", "Jiro", "Saburo", "Hanako", "Yoshiko"]
data  = [90, 100, 70, 80, 100]
series = pd.Series(data, index = index)
print(series)
csv = pd.read_csv("../input/bostonhoustingmlnd/housing.csv", delimiter=",")
print(csv)
# from subprocess import check_output
# print(check_output(["pwd", ""]).decode("utf8"))
csv = pd.read_csv("../input/pandasdata2csv/pandasdata2.csv", delimiter=",")
dataFrame = pd.DataFrame(csv)
sliced = dataFrame.loc[range(11, 13),["age", "hobby"]]
print(sliced)
sliced.to_csv("pandasdata3.csv")
df = pd.read_csv("../input/titanic/train.csv", delimiter=",")
# 欠損データ対応：Embarked
# → 'Embarked'列にNaNを持つデータ行を削除
df = df.dropna(subset=['Embarked'])

df = df.drop("Name", axis=1)
df = df.drop("Ticket", axis=1)
df = df.drop("Cabin", axis=1)

print(df)
df.to_csv("train2.csv")
df = pd.read_csv("../input/titanic/train.csv", delimiter=",")
# 欠損データ対応：Embarked
# → 'Embarked'列にNaNを持つデータ行を削除
df = df.dropna(subset=['Embarked'])

df = df.drop("Name", axis=1)
df = df.drop("Ticket", axis=1)
df = df.drop("Cabin", axis=1)

isMale   = df["Sex"] == "male"
isFemale = df["Sex"] == "female"
df["male"]   = isMale
df["female"] = isFemale
df = df.drop("Sex", axis=1)

isC = df["Embarked"] == "C"
isQ = df["Embarked"] == "Q"
isS = df["Embarked"] == "S"
df["embarked c"] = isC
df["embarked q"] = isQ
df["embarked s"] = isS


print(df)
df.to_csv("train3.csv")