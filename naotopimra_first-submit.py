# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
df= pd.read_csv("../input/train.csv").replace("male",0).replace("female",1)
df["Age"].fillna(df.Age.median(), inplace=True)
split_data = []
for survived in [0,1]:

    split_data.append(df[df.Survived==survived])
print(split_data)
temp = [i["Pclass"].dropna() for i in split_data]
print(temp)
plt.hist(temp, histtype="barstacked", bins=3)
temp = [i["Age"].dropna() for i in split_data]
plt.hist(temp, histtype="barstacked", bins=3)
plt.hist(temp, histtype="barstacked", bins=16)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df2 = df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)
df2.dtypes
train_data=df2.values
xs = train_data[:,2:]
print(xs)
y=train_data[:,1]
forest = RandomForestClassifier(n_estimators=100)
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(xs, y)
test_df = pd.read_csv("../input/test.csv").replace("male", 0).replace("female",1)
test_df["Age"].fillna(df.Age.median(), inplace=True)
test_df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

test_df2 = test_df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)
print(test_df2)
test_data=test_df2.values
print(test_data)
xs_test = test_data[:, 1:]
output = forest.predict(xs_test)
print(len(test_data[:,0]), len(output))
zip_data = zip(test_data[:,0].astype(int), output.astype(int))
predict_data = list(zip_data)
print(predict_data)
import csv
with open("predict_result_data.csv", "w") as f:

    writer = csv.writer(f, lineterminator='\n')

    writer.writerow(["PassengerId", "Survived"])

    for pid, survived in zip(test_data[:,0].astype(int), output.astype(int)):

        writer.writerow([pid, survived])
pwd
ls