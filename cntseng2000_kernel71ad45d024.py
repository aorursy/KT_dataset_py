import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np



from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor



import warnings

warnings.filterwarnings('ignore')

data_path = '/kaggle/input/3rd-ml100marathon-midterm/'

df_train = pd.read_csv(data_path + 'train_data.csv')



df_test = pd.read_csv(data_path + 'test_features.csv')



df_train.head()
df_train.describe()
df_train['poi'] = df_train['poi'].astype(float)

train_Y = df_train['poi']

ids = df_test['name']



df_train = df_train.drop(['name', 'email_address', 'poi'] , axis=1)

df_test = df_test.drop(['name', 'email_address'] , axis=1)



df = pd.concat([df_train,df_test])

df = df.fillna(0)

df.info()
#finalcial 和 email 數據的數值資訊都捨去, 有就是1, 沒有(na)就是0

for col in df:

    df[col] = df[col].apply(lambda x: 1 if abs(x)>0 else 0)



#先前測試, 發現有時候把 'bonus' 和 'deferred_income' 結合成新的 feature 結果比較好

df ['com2'] = df['bonus'] + df['deferred_income']

df.drop(['bonus', 'deferred_income'], axis = 1)

        

df.describe()
#測試幾種model後, 發現 gdbt 和 clf 分的比較好



gdbt = GradientBoostingClassifier(tol=100, subsample=0.75, n_estimators=250, max_features=9,

                                  max_depth=6, learning_rate=0.03)

         

train_num = train_Y.shape[0]

train_X = df[:train_num]

test_X = df[train_num:]

            

X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.25, random_state=4)

          

gdbt.fit(train_X, train_Y)

gdbt_pred = gdbt.predict_proba(test_X)[:,1]

gdbt_score = cross_val_score(gdbt, train_X, train_Y, cv=5).mean()



sub = pd.DataFrame({'name': ids, 'poi': gdbt_pred})

sub.to_csv('mid_abs_com2_gdbt1027.csv', index=False)

            



# 建立模型

clf = DecisionTreeClassifier()

clf.fit(train_X, train_Y)

clf_pred = clf.predict_proba(test_X)[:,1]

clf_score = cross_val_score(clf, train_X, train_Y, cv=5).mean()



sub = pd.DataFrame({'name': ids, 'poi': clf_pred})

sub.to_csv('mid_abs_com2_clf1027.csv', index=False)





print('gdbt:')

print(gdbt_score)



print('clf:')

print(clf_score)            


