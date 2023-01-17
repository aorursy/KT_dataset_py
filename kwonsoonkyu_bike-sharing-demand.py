import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import sklearn as skl
import seaborn as sns
import scipy as sci

%matplotlib inline
train = pd.read_csv("../input/train.csv", parse_dates=["datetime"])
test = pd.read_csv("../input/test.csv", parse_dates=["datetime"])
print(train.shape)
print(test.shape)
# 결측치 있는지 확인
# train.isna().sum()
train.isnull().sum()
print(train.columns)
print(test.columns)
common_columns = set(train.columns) & set(test.columns) #test에도 있는 columns만 사용하기 위해 공통으로 있는 column 정리
common_columns.add("count")

common_columns
train = train.loc[:,common_columns] # train에서 common column만 사용
train.shape
# datetime을 연월일시분초로 분할
train["year"] = train.datetime.dt.year
train["month"] = train.datetime.dt.month
train["day"] = train.datetime.dt.day
train["hour"] = train.datetime.dt.hour
train["min"] = train.datetime.dt.minute
train["sec"] = train.datetime.dt.second

test["year"] = test.datetime.dt.year
test["month"] = test.datetime.dt.month
test["day"] = test.datetime.dt.day
test["hour"] = test.datetime.dt.hour
test["min"] = test.datetime.dt.minute
test["sec"] = test.datetime.dt.second

train = train.loc[:, train.columns != "datetime"]
category_columns = ["holiday","season","workingday","weather", "year", "month", "day", "hour", "min", "sec"]
numerical_columns = list(set(train.columns) - set(category_columns))
# categorical variable을 category로 변환
for col in category_columns:
    train.loc[:,col] = train.loc[:,col].astype('category')
print(len(category_columns), len(numerical_columns))
nrows = 5
ncols = 3
figure, axes = plt.subplots(nrows = nrows, ncols = ncols)
figure.set_size_inches(15,30)
# figure.set(title = "Frequency")

for i in range(len(category_columns)):
    row_index = i // ncols
    col_index = i % ncols
    sns.countplot(data = train, x = category_columns[i], ax = axes[row_index][col_index])

for i in range(len(numerical_columns)):
    row_index = (i + len(category_columns)) // ncols
    col_index = (i + len(category_columns)) % ncols
    sns.distplot(train[numerical_columns[i]], ax = axes[row_index][col_index])
# count의 distplot이 왼쪽으로 skewed되어 있으므로 log변환이 필요하다
# windspeed의 값이 0인게 많은데 결측치로 생각할 수 있다
print(train["count"][train["count"] == 0].count())

np.power(train["count"], 0.1).plot(color="red", kind="density", )
np.power(train["count"], 0.2).plot(color="orange", kind="density")
np.power(train["count"], 0.3).plot(color="yellow", kind="density")
np.log(train["count"]).plot(color="green", kind="density")
np.log10(train["count"]).plot(color="blue", kind="density")
np.log1p(train["count"]).plot(color="navy", kind="density")
plt.legend(["^0.1","^0.2","^0.3","log","log10","log1p"])
# log변환이 가장 정규분포에 가까움
train["log_count"] = np.log(train["count"])
train = train.drop("count", axis=1)
corr = train.corr()
corr[np.abs(corr) > 0.1]
# wind speed와 다른 변수들간의 상관관계가 낮기 때문에 최빈값으로 imputation
train["windspeed"].value_counts().iloc[:2,]
windspeed_mode = 8.9981
train.loc[train["windspeed"] == 0, "windspeed"] = windspeed_mode
corr = train.corr()
corr
# atemp와 temp의 상관계수가 높으므로 atemp는 삭제
train = train.drop(["atemp"], axis=1)
train_dummy = pd.get_dummies(train)
ncols = 2
nrows = 5
figure, axes = plt.subplots(nrows=nrows, ncols=ncols)
figure.set_size_inches(15,30)

lists = ["year","month","day","hour","min","sec","holiday","workingday","season","weather"]
for i in range(10):
    sns.barplot(data = train, x = lists[i], y = "log_count", ax = axes[i // ncols][i % ncols])
    axes[i // ncols][i % ncols].set(ylabel="log_count", xlabel=lists[i], title="Count by " + lists[i])
test["day"][test["day"] < 20].count()
# 분, 초는 의미가 없으므로 삭제, train set에 day는 19일까지 밖에 없고 test set에 20일 이후가 있으므로 의미있게 사용할 수 없으므로 삭제
train = train.drop(["min","sec", "day"], axis=1)
test = test.drop(["min","sec","day"], axis=1)
a = range(1,5)
for i in a:
    print(i)
s = "season"
w = "weather"
table = pd.crosstab(train[s],train[w])
table
check_independence_list = ["season","weather","workingday","holiday"] # check할 category variable들 목록
p_val_mat = np.zeros((len(check_independence_list),len(check_independence_list))) # 빈 matrix 생성

for i in range(len(check_independence_list)):
    for j in range(i+1, len(check_independence_list)):
        table = pd.crosstab(train[check_independence_list[i]], train[check_independence_list[j]])
        p_value = sci.stats.chi2_contingency(table)[1] # p-value
        p_val_mat[i][j] = p_value
        print("p-value of independence test between " + check_independence_list[i]
              + " and " + check_independence_list[j] + " : " + "%.10f" %p_value) # p-value가 낮으면 두 변수가 독립이 아니다.
        
pd.DataFrame(np.round(p_val_mat, 4), index=check_independence_list, columns=check_independence_list)
# train.to_csv("datasets/train_EDA.csv")
# test.to_csv("datasets/test_EDA.csv")
