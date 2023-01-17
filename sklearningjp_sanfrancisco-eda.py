import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df_0 = pd.read_csv('../input/sf-salaries/Salaries.csv')
df_0.head()
df_0.shape
df_0.isnull().sum()
df_0.fillna(0)
df_0['JobTitle'].value_counts()
plt.figure(figsize=(13,8))
job = df_0['JobTitle'].value_counts()[0:30]
sns.barplot(x=job.values,y=job.index, alpha=0.6)
plt.show()
df_0['Agency'].value_counts()
df_0['Year'].value_counts()
df_0.dtypes
df_0["BasePay"] = pd.to_numeric(df_0["BasePay"], errors='coerce')
df_0["OvertimePay"] = pd.to_numeric(df_0["OvertimePay"], errors='coerce')
df_0["OtherPay"] = pd.to_numeric(df_0["OtherPay"], errors='coerce')
df_0["Benefits"] = pd.to_numeric(df_0["Benefits"], errors='coerce')
df_0.dtypes
data = ['JobTitle', 'TotalPay', 'Year']
df_1 = df_0[data]
df_1.head()
df_2 = df_1.sort_values(by="TotalPay", ascending=False)
df_２.set_index("JobTitle",inplace=True)
df_2[0:20].plot.bar()
df_11 = df_1.query('Year == "2011"')
df_12 = df_1.query('Year == "2012"')
df_13 = df_1.query('Year == "2013"')
df_14 = df_1.query('Year == "2014"')
df_11 = df_11.sort_values(by="TotalPay", ascending=False)
df_11.head()
df_12 = df_12.sort_values(by="TotalPay", ascending=False)
df_12.head()
df_13 = df_13.sort_values(by="TotalPay", ascending=False)
df_13.head()
df_14 = df_14.sort_values(by="TotalPay", ascending=False)
df_14.head()
df_0.iloc[0]
# OtherPayが多い
df_0.iloc[36159]
# OvertimePayが多い
df_0.iloc[72927]
# OvertimePayが多い
df_0.iloc[110531]
# OtherPayが多い
# 各年度の人数

df_0.groupby(['Year'])['Id'].count()
# 各年度のBasePay合計

df_0.groupby(['Year'])['BasePay'].sum()
# 各年度の一人当たりBasePay

df_b = df_0.groupby(['Year'])['BasePay'].sum() / df_0.groupby(['Year'])['Id'].count()
df_b
df_b.plot.bar()
# 各年度のOvertimePay合計
print(df_0.groupby(['Year'])['OvertimePay'].sum())

# 各年度の一人当たりOvertimePay
df_ov = df_0.groupby(['Year'])['OvertimePay'].sum() / df_0.groupby(['Year'])['Id'].count()
print(df_ov)
df_ov.plot.bar()
# 各年度のOtherPay 合計
print(df_0.groupby(['Year'])['OtherPay'].sum())

# 各年度の一人当たりOtherPay 
df_ot = df_0.groupby(['Year'])['OtherPay'].sum() / df_0.groupby(['Year'])['Id'].count()
print(df_ot)
df_ot.plot.bar()
# 各年度のBenefits合計
print(df_0.groupby(['Year'])['Benefits'].sum())

# 各年度の一人当たりBenefits
df_be = df_0.groupby(['Year'])['Benefits'].sum() / df_0.groupby(['Year'])['Id'].count()
print(df_be)
df_be.plot.bar()
# 各年度のTotalPay合計
print(df_0.groupby(['Year'])['TotalPay'].sum())

# 各年度の一人当たりTotalPay
df_t = df_0.groupby(['Year'])['TotalPay'].sum() / df_0.groupby(['Year'])['Id'].count()
print(df_t)
df_t.plot.bar()
# 各年度のTotalPayBenefits合計
print(df_0.groupby(['Year'])['TotalPayBenefits'].sum())

# 各年度の一人当たりTotalPayBenefits
df_to = df_0.groupby(['Year'])['TotalPayBenefits'].sum() / df_0.groupby(['Year'])['Id'].count()
print(df_to)
df_to.plot.bar()
df_2.head(20)
# 給与のばらつき

x = df_0.Year
y = df_0.TotalPay
sns.boxplot(x, y)
X = df_1['JobTitle']
y = df_1['TotalPay']
X = pd.get_dummies(X)
X.head()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_train, y_train)
print(lr.coef_)
print(lr.intercept_)
y_pred = lr.predict(X_test)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_pred, y_test)