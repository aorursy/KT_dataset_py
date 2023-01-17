# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd



df = pd.read_csv("../input/train.csv")
# 필요없는 변수 제거

# descript는 자세한 범죄를 나타낸것이기 때문에 category에 포함되어 제거한다.

# address는 약 2만개의 범주가 나타나므로 분석에 무의미하여 제거한다.

df = df.drop(['X', 'Y', 'Descript', 'Address'], axis=1)
# 범주형 변수를 수치형 변수로 바꿔준다.

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Category'] = le.fit_transform(df.Category)

df['DayOfWeek'] = le.fit_transform(df.DayOfWeek)

df['PdDistrict'] = le.fit_transform(df.PdDistrict)

df['Resolution'] = le.fit_transform(df.Resolution)
# 몇개의 범주로 나눠지는지 확인

df.Category.max()
df.DayOfWeek.max()
df.PdDistrict.max()
df.Resolution.max()
# 날짜 데이터에서 시간만 출력

# 시간대에 따른 범죄종류를 확인하기 위해서

Dates = []

for i in range(0, 878049):

    Dates.append(df.Dates[i].split(" ")[1].split(":")[0])

df.Dates = Dates
# 정제한 파일 저장

df.to_csv("df.csv", index=False)
# x, y 분류

y = df.Resolution

y = pd.DataFrame(y)

x = df.drop(["Resolution", "PdDistrict", "Category"], axis=1)

x = pd.DataFrame(x)
# 정확한 분석을 위해 training set과 test set으로 분류

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn import datasets

rf = RandomForestClassifier()

rf.fit(x_train, y_train)

accuracy_score(y_test, rf.predict(x_test))