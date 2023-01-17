import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
train=pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
test=pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')
train.head()
test.head()
print(train.shape)
print(train.isnull().sum())
print(train.dtypes)
print(train['Vehicle_Age'].unique())
train.groupby(['Vehicle_Age','Response'])['id'].count()
print(train['Policy_Sales_Channel'].unique())
train.groupby(['Policy_Sales_Channel','Response'])['id'].count()
df=train.groupby(['Previously_Insured','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
df
sns.catplot(x="Previously_Insured", y="count",col="Response",data=df, kind="bar",height=6, aspect=.7);
# すでに加入している人は、ほとんど興味を示さない
train.describe()
train.Response.value_counts()
# 興味ある率は１２．3％くらい
sns.distplot(train.Age)
# 顧客は若い人が多い
sns.countplot(train.Gender)
sns.countplot(train.Driving_License)
# 免許はほぼ持っている
sns.countplot(train.Previously_Insured)
# 自動車保険未加入の人が多い
sns.countplot(train.Vehicle_Damage)
# 過去の事故や故障があった人は半分くらい
sns.countplot(train.Response)
sns.countplot(train.Vehicle_Age)
# ２年以内の人が大多数
df２=train.groupby(['Vehicle_Age','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
df２
sns.catplot(x="Vehicle_Age", y="count",col="Response",data=df２, kind="bar",height=6, aspect=.7);
# １年以上２年未満の人の方が興味がある！
df３ = train.groupby(['Previously_Insured','Vehicle_Damage'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
df3
sns.catplot(x="Previously_Insured", y="count",col="Vehicle_Damage",data=df3, kind="bar",height=6, aspect=.7);
# ??? 過去に車両破損経験者の方が自動車保険に入ってない。
# 自動車保険に入っている人の方が車両破損しない。→安全運転、慎重？
train.plot(kind='scatter', x='Age', y='Annual_Premium', figsize=(10, 7))
# 年齢とプレミアム保険料を払っている額
train.groupby('Response')['Age'].plot.hist(bins=20, alpha=0.5, legend=True)
# 興味ある人の年齢の分布：40代付近が多い。若い人は興味ない。
(ax0, ax1), = pd.crosstab(pd.cut(train['Age'], range(0, 101, 5), right=False), train['Response']).plot.barh(subplots=True, layout=(1, 2), sharex=False)
ax0.invert_xaxis() # 左側のグラフのX軸を反転する
ax1.set_yticklabels([]) # 右側のグラフのY軸のラベルを消す
(ax0, ax1), = pd.crosstab(pd.cut(train['Annual_Premium'], range(0, 60000, 5000), right=False), train['Response']).plot.barh(subplots=True, layout=(1, 2), sharex=False)
ax0.invert_xaxis() # 左側のグラフのX軸を反転する
ax1.set_yticklabels([]) # 右側のグラフのY軸のラベルを消す
sns.boxplot( x=train['Gender'], y=train['Age'] )
# 性別と年齢
sns.boxplot( x=train['Vehicle_Age'], y=train['Age'] )
# 車両年数と年齢
# 買ったばかりの１年未満は20歳代が多い
# 買って１年以上2年未満は40歳50歳代が多い＝乗り換え？
sns.boxplot( x=train['Response'], y=train['Age'] )
# 興味ある人と年齢
sns.boxplot( x=train['Response'], y=train['Annual_Premium'] )
# 興味ある人と年間プレミアム保険料支払い者：あまり関係ない？
train.loc[train['Gender'] == 'Male', 'Gender'] = 1
train.loc[train['Gender'] == 'Female', 'Gender'] = 0
test.loc[test['Gender'] == 'Male', 'Gender'] = 1
test.loc[test['Gender'] == 'Female', 'Gender'] = 0

train.loc[train['Vehicle_Age'] == '> 2 Years', 'Vehicle_Age'] = 2
train.loc[train['Vehicle_Age'] == '1-2 Year', 'Vehicle_Age'] = 1
train.loc[train['Vehicle_Age'] == '< 1 Year', 'Vehicle_Age'] = 0
test.loc[test['Vehicle_Age'] == '> 2 Years', 'Vehicle_Age'] = 2
test.loc[test['Vehicle_Age'] == '1-2 Year', 'Vehicle_Age'] = 1
test.loc[test['Vehicle_Age'] == '< 1 Year', 'Vehicle_Age'] = 0

train.loc[train['Vehicle_Damage'] == 'Yes', 'Vehicle_Damage'] = 1
train.loc[train['Vehicle_Damage'] == 'No', 'Vehicle_Damage'] = 0
test.loc[test['Vehicle_Damage'] == 'Yes', 'Vehicle_Damage'] = 1
test.loc[test['Vehicle_Damage'] == 'No', 'Vehicle_Damage'] = 0
for col in train.columns:
    train[col] = train[col].astype(np.int32)

train.head(10)
train_corr = train.corr()
matplotlib.pyplot.figure(figsize=(10,10))
sns.heatmap(train_corr, vmax=.9, square=True, annot=True, linewidths=.3, cmap="YlGnBu", fmt='.３f')
train.plot(kind='scatter', x='Age', y='Policy_Sales_Channel', figsize=(10, 7))
# 流入経路は年齢との相関が強い
# 40−50、80−120あたりが40歳50歳代を獲得できるチャネルコード
x = train.drop("Response",axis=1)
y = train["Response"]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
dt_clf = DecisionTreeClassifier(criterion='entropy',max_depth = 100,random_state=0)
dt_clf.fit(x_train,y_train)
from sklearn.metrics import accuracy_score,classification_report
dt_pred = dt_clf.predict(x_test)
dt_accuracy = accuracy_score(y_test,dt_pred)
dt_accuracy
rf_clf = RandomForestClassifier(n_estimators = 200,random_state=0)
rf_clf.fit(x_train,y_train)
rf_pred = rf_clf.predict(x_test)
rf_accuracy = accuracy_score(y_test,rf_pred)
rf_accuracy
lr_clf = LogisticRegression(random_state=0)
lr_clf.fit(x_train,y_train)
lr_pred = lr_clf.predict(x_test)
lr_accuracy = accuracy_score(y_test,lr_pred)
lr_accuracy
lgbm_clf = LGBMClassifier(n_estimators=1000,learning_rate=0.007,random_state=0)#1000
lgbm_clf.fit(x_train,y_train)
lgbm_pred = lgbm_clf.predict(x_test)
lgbm_accuracy = accuracy_score(y_test,lgbm_pred)
lgbm_accuracy
knn_clf = KNeighborsClassifier(n_neighbors=20)
knn_clf.fit(x_train,y_train)
knn_pred = knn_clf.predict(x_test)
knn_accuracy = accuracy_score(y_test,knn_pred)
knn_accuracy
acc_df = pd.DataFrame({"Decision Tree":dt_accuracy,"Random Forest":rf_accuracy,
                       "LightGBM":lgbm_accuracy,"Logistic Regression" : lr_accuracy,"KNN":knn_accuracy},index = ["Accuracy"])
acc_df.style.background_gradient(cmap = "Reds")
features=['Gender','Age','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Annual_Premium',
'Policy_Sales_Channel','Vintage']
target = 'Response'
from sklearn.ensemble import RandomForestClassifier

model_rfc = RandomForestClassifier(n_estimators=100,random_state=9,n_jobs=-1)

model_rfc.fit(train[features],train[target]) 
predictions = model_rfc.predict(test[features])

predictions
submission = pd.DataFrame({'id':test['id'],'Response':predictions})

submission.head(10)
filename = 'submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)