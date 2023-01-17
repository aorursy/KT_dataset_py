import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

%matplotlib inline
train = pd.read_csv("/kaggle/input/GiveMeSomeCredit/cs-training.csv")

test = pd.read_csv("/kaggle/input/GiveMeSomeCredit/cs-test.csv")

# print(train.columns)

# print(test.columns)
# 合并处理

df = pd.concat([train, test], ignore_index=True)

df.head()
# 检查缺失值情况

df.isnull().sum()
raw_columns = df.columns

for i, col in enumerate(raw_columns):

    print("这是第 %2d 个变量: %s" % (i, col))

    

# 第0个变量是无效的用来作为唯一标识符的值
sns.countplot(df.SeriousDlqin2yrs)

df['y'] = df.SeriousDlqin2yrs
df.RevolvingUtilizationOfUnsecuredLines.describe()  # 按理不该存在大于1的值
df['RUOUL'] = df.RevolvingUtilizationOfUnsecuredLines.copy()

# df.loc[df.RUOUL > 1, 'RUOUL'] = 1

df.RUOUL.describe()



# 分布

plt.figure(figsize=(16, 9))

sns.distplot(df.RUOUL, kde=False)
# 分箱操作

N = 11

df['RUOUL'] = pd.qcut(df.RUOUL, N, labels=[i/10 for i in range(N)], duplicates='drop')  # 'drop' vs. 'raise'

print(df.RUOUL.value_counts())


# plot

sns.countplot(df['RUOUL'], hue=df['SeriousDlqin2yrs'])
df['AGE'] = df.age.copy()



# plot

sns.countplot(df.AGE)
# 分箱操作

bins = [0, 30, 40, 50, 60, 70, 120]

# print(bins)

labels = [i/(len(bins)-2) for i in range(len(bins) - 1)]

# print(labels)

df.AGE = pd.cut(df.AGE, bins, right=0, labels=labels)

# plot

sns.countplot(x='AGE', hue='SeriousDlqin2yrs', data=df)
df['NOT30D'] = df['NumberOfTime30-59DaysPastDueNotWorse'].copy()



# plot

sns.countplot(df.NOT30D)
# 直接进行二类映射表示有此逾期记录

df.loc[df.NOT30D > 0, 'NOT30D'] = 1

sns.countplot(df.NOT30D)
df['DR'] = df.DebtRatio.copy()

df['DR'].describe()  # 存在异常值
plt.figure(figsize=(16, 9))

sns.boxplot(y=df['DR'])
# 分箱操作

N = 11

df['DR'] = pd.qcut(df.DR, N, labels=[i/10 for i in range(N)], duplicates='drop')  # 'drop' vs. 'raise'
# plot

sns.countplot(df.DR, hue=df.SeriousDlqin2yrs)

# sns.boxplot(y=df['DR'])

df.DR.value_counts()
df['MI'] = df.MonthlyIncome.copy()

df['MI'] = df.MI.fillna(df.MI.median())

df.MI.describe()
# 分箱操作

N = 11

df['MI'] = pd.qcut(df.MI, N, labels=[i/10 for i in range(N-1)], duplicates='drop')  # 'drop' vs. 'raise'
# plot

sns.countplot(df.MI)  # , hue=df.SeriousDlqin2yrs

df.MI.value_counts()
df['NOOCLAL'] = df.NumberOfOpenCreditLinesAndLoans.copy()

df.NOOCLAL.describe()
# 分箱操作

N = 11

df['NOOCLAL'] = pd.qcut(df.NOOCLAL, N, labels=[i/10 for i in range(N)], duplicates='drop')  # 'drop' vs. 'raise'
# plot

sns.countplot(df.NOOCLAL)  # , hue=df.SeriousDlqin2yrs

df.NOOCLAL.value_counts()

df.NOOCLAL.isnull().sum()
df['NOT90D'] = df.NumberOfTimes90DaysLate.copy()

# 直接进行二类映射表示有此逾期记录

df.loc[df.NOT90D > 0, 'NOT90D'] = 1

sns.countplot(df.NOT90D)
df['NRELO'] = df.NumberRealEstateLoansOrLines.copy()

df['NRELO'].describe()
# 分组统计

bins = [0, 1, 3, df.NRELO.max()]

labels = [0, 0.5, 1.0]

df['NRELO'] = pd.cut(df['NRELO'], bins, right=0, labels=labels)



# plot

sns.countplot(df.NRELO, hue=df.y)
df['NOT60D'] = df['NumberOfTime60-89DaysPastDueNotWorse'].copy()

# 直接进行二类映射表示有此逾期记录

df.loc[df.NOT60D > 0, 'NOT60D'] = 1

sns.countplot(df.NOT60D)
df['NOD'] = df['NumberOfDependents'].copy()

df['NOD'] = df.NOD.fillna(0)

df.NOD.describe()
# 分组统计

bins = [0, 1, 2, 4, 8, df.NOD.max()]

labels = [0, 0.25, 0.5, 0.75, 1.0]

df['NOD'] = pd.cut(df['NOD'], bins, right=0, labels=labels)



# plot

sns.countplot(df.NOD, hue=df.y)
df.drop(raw_columns, axis=1, inplace=True)
df.head()
df.dtypes
df[:len(train)]['y'].value_counts()  # 不均衡数据集
# 可以通过复制进行数据集升采样

train_0_data = df[:len(train)][df.y == 0]

train_1_data = df[:len(train)][df.y == 1]

print(train_0_data.shape, train_1_data.shape)
# UpSample

# train_1_data_up = train_0_data.copy()

# print(train_1_data_up.shape)

# for i in np.arange(len(train_0_data)):

#     idx = np.random.randint(0, train_1_data.shape[0])

#     train_1_data_up.iloc[i] = train_1_data.iloc[idx]

train_1_data_up = pd.concat([train_1_data] * 14, ignore_index=True)

print(train_1_data_up.shape)

print(train_1_data_up.iloc[0, :], '\n' + '-'*50)

print(train_1_data_up.iloc[0+train_1_data.shape[0], :])
print(train_0_data.shape, train_1_data_up.shape)
train_data = pd.concat([train_0_data, train_1_data_up], ignore_index=True)

print(train_data.isnull().sum())  # 再次检查缺失值

train_data.dropna(axis=0, how='any', inplace=True) # 直接删除

print(train_data[train_data.NRELO.isnull()])



X, y = train_data.iloc[:, 1:].values, train_data.y.values.ravel()

X, y = X.astype('f4'), y.astype('f4')

print(X.dtype, y.dtype)

print(X.shape, y.shape)

print(X[-1], y[-1])
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=2019)

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score



model = LogisticRegression(C=3, solver='lbfgs').fit(X_train, y_train)

print(model.score(X_val, y_val))

print(roc_auc_score(y_val, model.predict(X_val)))
import xgboost as xgb



# model = xgb.XGBClassifier(

#     max_depth=5,

#     eta=0.025,

#     silent=1,

#     objective='binary:logistic',

#     eval_matric='auc',

#     minchildweight=10.0,

#     maxdeltastep=1.8,

#     colsample_bytree=0.4,

#     subsample=0.8,

#     gamma=0.65,

#     numboostround=391

# )



model = xgb.XGBClassifier(

    max_depth=6,

    eta=1,

    silent=1,

    objective='binary:logistic',

    eval_matric='f1'

)



model.fit(X_train, y_train)

print(model.score(X_val, y_val))

print(roc_auc_score(y_val, model.predict(X_val)))
X_test = df[len(train):]

print(X_test.isnull().sum(), '\n' + '-'*100)

print(X_test[X_test.NOD.isnull()])

X_test['NOD'] = X_test['NOD'].astype("float64")  # 为什么会有一个缺失值的出现

X_test['NOD'] = X_test.NOD.fillna(X_test.NOD.median())

X_test = X_test.iloc[:, 1:]

X_test.head()
y_test = model.predict(X_test.values)
submission = pd.DataFrame()

submission['ID'] = np.arange(1, len(y_test) + 1)

submission['Probability'] = y_test

submission.to_csv('submission.csv', header=True, index=False)