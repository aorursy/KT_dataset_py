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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.head()
np.shape(train_data.values)
test_data.head()
np.shape(test_data.values)
# チュートリアルより抜粋

women = train_data.loc[train_data.Sex == "female"]["Survived"]

rate_women = sum(women)/len(women)

# 女性で生き残った人の割合

print("% of women who survived:",rate_women * 100)

men = train_data.loc[train_data.Sex == "male"]["Survived"]

rate_men = sum(men)/len(men)

# 男性で生き残った人の割合

print("% of men who survived:",rate_men * 100)
# 船上に家族がいるか

s = train_data.loc[(train_data.SibSp + train_data.Parch) == 0]["Survived"]

rate_s = sum(s)/len(s)

print("% of single who survived:",rate_s * 100)

print("volume:",len(s))

f = train_data.loc[(train_data.SibSp + train_data.Parch) > 0]["Survived"]

rate_f = sum(f)/len(f)

print("% of family who survived:",rate_f * 100)

print("volume:",len(f))



# 席の値段

s = train_data.loc[train_data.Pclass == 3]["Survived"]

rate_s = sum(s)/len(s)

print("% of 3rd class who survived:",rate_s * 100)

print("volume:",len(s))

f = train_data.loc[train_data.Pclass == 2]["Survived"]

rate_f = sum(f)/len(f)

print("% of 2nd class who survived:",rate_f * 100)

print("volume:",len(f))

c = train_data.loc[train_data.Pclass == 1]["Survived"]

rate_c = sum(c)/len(c)

print("% of 1st class who survived:",rate_c * 100)

print("volume:",len(c))



# 訓練データの欠損値

train_data.isnull().sum()
# テストデータの欠損値

test_data.isnull().sum()
feats = ["Age", "Embarked","Fare","Sex","Pclass","Family"]

union = pd.concat([train_data,test_data],axis = 0)

union["Age"] = union["Age"].fillna(union["Age"].mean())

union["Embarked"] = union["Embarked"].fillna("S")

union["Fare"] = union["Fare"].fillna(union["Fare"].mean())

union["Survived"] = union["Survived"].fillna(0)

union["Family"] = union["Parch"] + union["SibSp"]



import category_encoders as ce



list_cols = ['Embarked',  'Sex']

ce_ohe = ce.OneHotEncoder(cols=list_cols,handle_unknown='impute')

union = ce_ohe.fit_transform(union)

union["Embarked_1"] *= 3

union["Embarked_2"] *= 2

union["Embarked"] = union["Embarked_1"] + union["Embarked_2"] + union["Embarked_3"]

union["Sex_1"] *= 2

union["Sex"] = union["Sex_1"] + union["Sex_2"]



trainX = union.iloc[0:891,:]

trainY = trainX["Survived"]

trainX = trainX[feats]



testX = union.iloc[891:1309,:]

testX = testX[feats]



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 1000, max_depth = 6, random_state = 1)

model.fit(trainX, trainY)



predictions = model.predict(testX)



output = pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived':predictions})

output.to_csv('submission.csv', index = False)

output.describe()
output