import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# !unzip /kaggle/input/springleaf-marketing-response/sample_submission.csv.zip

# !unzip /kaggle/input/springleaf-marketing-response/test.csv.zip

# !unzip /kaggle/input/springleaf-marketing-response/train.csv.zip
df_train = pd.read_csv('/kaggle/input/springleaf-marketing-response/train.csv.zip')

df_test = pd.read_csv('/kaggle/input/springleaf-marketing-response/test.csv.zip')
df_train
df_train.info()
#пропущенные данные

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
missing_data[missing_data["Total"] > 1] 
#отрбрасываем столбцы с пропущенными значениями

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_test = df_test.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train
df_train["target"].unique()
#находим стоблцы с категориальными признаками

df_train.select_dtypes(include=['O'])
df_train["VAR_0001"].unique()
# преобразуем категориальные признаки в численные

df_train['0001H'] = df_train['VAR_0001'].map({'H': 1, 'R': 0, 'Q': 0})

df_train['0001R'] = df_train['VAR_0001'].map({'H': 0, 'R': 1, 'Q': 0})

df_train['0001Q'] = df_train['VAR_0001'].map({'H': 0, 'R': 0, 'Q': 1})

df_train.pop("VAR_0001")



df_test['0001H'] = df_test['VAR_0001'].map({'H': 1, 'R': 0, 'Q': 0})

df_test['0001R'] = df_test['VAR_0001'].map({'H': 0, 'R': 1, 'Q': 0})

df_test['0001Q'] = df_test['VAR_0001'].map({'H': 0, 'R': 0, 'Q': 1})

df_test.pop("VAR_0001")
df_train["VAR_0005"].unique()
# преобразуем категориальные признаки в численные

df_train['0005C'] = df_train['VAR_0005'].map({'C': 1, 'B': 0, 'N': 0, 'S': 0})

df_train['0005B'] = df_train['VAR_0005'].map({'C': 1, 'B': 0, 'N': 0, 'S': 0})

df_train['0005N'] = df_train['VAR_0005'].map({'C': 1, 'B': 0, 'N': 0, 'S': 0})

df_train['0005S'] = df_train['VAR_0005'].map({'C': 1, 'B': 0, 'N': 0, 'S': 0})

df_train.pop("VAR_0005")



df_test['0005C'] = df_test['VAR_0005'].map({'C': 1, 'B': 0, 'N': 0, 'S': 0})

df_test['0005B'] = df_test['VAR_0005'].map({'C': 1, 'B': 0, 'N': 0, 'S': 0})

df_test['0005N'] = df_test['VAR_0005'].map({'C': 1, 'B': 0, 'N': 0, 'S': 0})

df_test['0005S'] = df_test['VAR_0005'].map({'C': 1, 'B': 0, 'N': 0, 'S': 0})

df_test.pop("VAR_0005")
df_train["VAR_1934"].unique()
# преобразуем категориальные признаки в численные

df_train['1934_IAPS'] = df_train['VAR_1934'].map({'IAPS': 1, 'RCC': 0, 'BRANCH': 0, 'MOBILE': 0, 'CSC': 0})

df_train['1934_RCC'] = df_train['VAR_1934'].map({'IAPS': 0, 'RCC': 1, 'BRANCH': 0, 'MOBILE': 0, 'CSC': 0})

df_train['1934_BRANCH'] = df_train['VAR_1934'].map({'IAPS': 0, 'RCC': 0, 'BRANCH': 1, 'MOBILE': 0, 'CSC': 0})

df_train['1934_MOBILE'] = df_train['VAR_1934'].map({'IAPS': 0, 'RCC': 0, 'BRANCH': 0, 'MOBILE': 1, 'CSC': 0})

df_train['1934_CSC'] = df_train['VAR_1934'].map({'IAPS': 0, 'RCC': 0, 'BRANCH': 0, 'MOBILE': 0, 'CSC': 1})

df_train.pop("VAR_1934")



df_test['1934_IAPS'] = df_test['VAR_1934'].map({'IAPS': 1, 'RCC': 0, 'BRANCH': 0, 'MOBILE': 0, 'CSC': 0})

df_test['1934_RCC'] = df_test['VAR_1934'].map({'IAPS': 0, 'RCC': 1, 'BRANCH': 0, 'MOBILE': 0, 'CSC': 0})

df_test['1934_BRANCH'] = df_test['VAR_1934'].map({'IAPS': 0, 'RCC': 0, 'BRANCH': 1, 'MOBILE': 0, 'CSC': 0})

df_test['1934_MOBILE'] = df_test['VAR_1934'].map({'IAPS': 0, 'RCC': 0, 'BRANCH': 0, 'MOBILE': 1, 'CSC': 0})

df_test['1934_CSC'] = df_test['VAR_1934'].map({'IAPS': 0, 'RCC': 0, 'BRANCH': 0, 'MOBILE': 0, 'CSC': 1})

df_test.pop("VAR_1934")
# проверямем, остались ли категориальные признаки

sum(df_train.select_dtypes(include=['O']))
# заполняем пропущенные данные

df_train = df_train.fillna(-1)

df_test = df_test.fillna(-1)
df_train['target1'] = df_train["target"]

df_train.pop('target')

df_train
from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.metrics import accuracy_score
featureNames = df_train.columns[1:-1]
print("sampling train to get around 8GB memory limitations\n")

df_train = df_train.sample(n=20000)





print("training a XGBoost classifier\n")

dtrain = xgb.DMatrix(df_train[featureNames].values, label=df_train['target1'].values)



param = {'max_depth':2, 

         'eta':1, 

         'objective':'binary:logistic', 

         'eval_metric': 'auc'}

clf = xgb.train(param, dtrain, 20)





print("making predictions in batches due to 8GB memory limitation\n")

submission = df_test[['ID']]

submission['target1'] = np.nan

step = len(submission)/10000

for rows in range(0, len(submission), int(step)):

    submission.loc[rows:rows+step, "target"] = clf.predict(xgb.DMatrix(df_test.loc[rows:rows+step, featureNames].values))





print("saving the submission file\n")

submission.to_csv("xgboost_submission.csv", index=False)