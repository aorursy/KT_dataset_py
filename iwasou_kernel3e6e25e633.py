# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

from numpy.random import *

from statsmodels import api as sm

from sklearn import preprocessing, metrics, linear_model

from sklearn.svm import SVC

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
##########################################

# private functions

##########################################



# データ標準化

def ss(df_data):

    # 標準化

    ss = preprocessing.StandardScaler()

    ss.fit(arrange(df_train.drop('Survived', 1)))



    std_df_data = ss.transform(df_data)

    

    return pd.DataFrame(std_df_data)





# データ整形

def arrange(df_data):

    # 傾向がないフィールドを除去（=カテゴリフィールド且つ種類が多いフィールド）

    df_data = df_data.drop('PassengerId', 1)

    df_data = df_data.drop('Name', 1)

    df_data = df_data.drop('Ticket', 1)

    df_data = df_data.drop('Cabin', 1)

    

    # カテゴリフィールドをダミーデータ化

    df_data_Sex = pd.get_dummies(df_data["Sex"])

    df_data_Sex.columns = ['Sex_femail', 'Sex_mail']

    df_data = pd.concat([df_data.drop('Sex', 1), pd.get_dummies(df_data_Sex)], axis=1)

    

    # カテゴリフィールドをダミーデータ化

    df_data_Pclass = pd.get_dummies(df_data["Pclass"])

    df_data_Pclass.columns = ['Pclass_1', 'Pclass_2', 'Pclass_3']

    df_data = pd.concat([df_data.drop('Pclass', 1), pd.get_dummies(df_data_Pclass)], axis=1)

    

    # カテゴリフィールドをダミーデータ化

    df_data_Embarked = pd.get_dummies(df_data["Embarked"])

    df_data_Embarked.columns = ['Embarked_C', 'Embarked_Q', 'Embarked_S']

    df_data = pd.concat([df_data.drop('Embarked', 1), pd.get_dummies(df_data_Embarked)], axis=1)

    

    # マルチコ（多重共線性の）排除

    df_data = df_data.drop('Sex_mail', 1)

    df_data = df_data.drop('Pclass_3', 1)

    df_data = df_data.drop('Embarked_S', 1)

    

    return df_data





# 欠損データに正規分布によるランダム値を埋め込む

def embed_to_na(df_data):



    means = df_data.mean()

    stds = df_data.std()



    for column in df_data.columns.values:

        df_data = df_data.fillna({column : normal(means[column], stds[column])})



    return df_data
# データの読み込み

df_gender_submission = pd.read_table('/kaggle/input/titanic/gender_submission.csv', ',')

df_test = pd.read_table('/kaggle/input/titanic/test.csv', ',')

df_train = pd.read_table('/kaggle/input/titanic/train.csv', ',')
df_gender_submission
# データ整形

# → 傾向がないフィールドを除去（=カテゴリフィールド且つ種類が多いフィールド）

# 　 → PassengerId, Name, Ticket, Cabin

# → カテゴリフィールドをダミーデータ化

# 　 → Sex, Pclass, Embarked

# → マルチコ排除

df_test_arranged = arrange(df_test)

df_train_arranged = arrange(df_train)
# データ標準化

ss_df_test_arranged = ss(df_test_arranged)

ss_df_train_arranged = pd.concat([ss(df_train_arranged.drop('Survived', 1)), df_train_arranged['Survived']], axis=1)
# 欠損データに正規分布によるランダム値を埋め込む

ss_df_test_arranged_embeded = embed_to_na(ss_df_test_arranged)

ss_df_train_arranged_embeded = embed_to_na(ss_df_train_arranged)
# 学習

clf = SVC(kernel='linear', random_state=None,C=0.1)   # サポートベクターマシン

#clf = linear_model.LogisticRegression()   # ロジスティック回帰

clf.fit(ss_df_train_arranged_embeded.drop('Survived', 1), ss_df_train_arranged_embeded['Survived'])
# 予測

prediction = clf.predict(ss_df_test_arranged_embeded)
# スコア 標準化なし：0.9401913875598086

ac_score = metrics.accuracy_score(df_gender_submission.drop('PassengerId', 1), prediction)

print(ac_score)
# submit

df_prediction = pd.concat([df_test['PassengerId'], pd.DataFrame(prediction)], axis=1)

df_prediction.columns = ['PassengerId', 'Survived']

df_prediction.to_csv("/kaggle/working/titanic_prediction.csv",index=False)