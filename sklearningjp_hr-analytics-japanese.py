import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# 説明のファイル
dic = pd.read_excel('../input/hr-analytics-case-study/data_dictionary.xlsx')
dic
# ファイル読み込んでいく
employee = pd.read_csv('../input/hr-analytics-case-study/employee_survey_data.csv')
general = pd.read_csv('../input/hr-analytics-case-study/general_data.csv')
intime = pd.read_csv('../input/hr-analytics-case-study/in_time.csv')
outtime = pd.read_csv('../input/hr-analytics-case-study/out_time.csv')
mgrsurvey = pd.read_csv('../input/hr-analytics-case-study/manager_survey_data.csv')
employee.head()
# 満足度とWLBの指標
general.head()
# 全体的なデータ
# IDでマージできそう
intime.head()
# 勤怠の出社時間
outtime.head()
# 勤怠の退社時間
mgrsurvey.head()
# 仕事への関与とパフォーマンスの評価
intime = intime.apply(pd.to_datetime)
outtime = outtime.apply(pd.to_datetime)

time = outtime - intime
time.head()
time = time/ np.timedelta64(1, 's')
time.head()
intime = intime.applymap(lambda x: 3600*x.hour + 60*x.minute + x.second)
outtime = outtime.applymap(lambda x: 3600*x.hour + 60*x.minute + x.second)
general['in_mean'] = intime.mean(axis=1)
general['out_mean'] = outtime.mean(axis=1)

general['in_std'] = intime.std(axis=1)
general['out_std'] = outtime.std(axis=1)
general.head()
# generalに労働時間を入れる
general['time_std'] = general['out_std'] - general['in_std']
general.head()
general.isnull().sum()
# 列を強制的に全表示する
pd.set_option('display.max_columns', 100)
general.head()
# general, employee, mgrsurveyをEmployeeIDで結合する

df1 = pd.merge(general, employee, on='EmployeeID')
df = pd.merge(df1, mgrsurvey, on='EmployeeID')
df.head()
# 婚姻状況での違い

df_mar = pd.crosstab(df['Attrition'], df['MaritalStatus'])
print(df_mar)

df_mar.plot.bar()
# Singleの離職が多い
# 役職での違い

df_job = pd.crosstab(df['Attrition'], df['JobRole'])
print(df_job)

df_job.plot.bar()
# あまり大きな関係なさそう
# 性別での違い

df_gen = pd.crosstab(df['Attrition'], df['Gender'])
print(df_gen)

df_gen.plot.bar()
# 性別は関係なさそう
# 専攻での違い

df_edu = pd.crosstab(df['Attrition'], df['EducationField'])
print(df_edu)

df_edu.plot.bar()
# 関係なさそう
# 部署別の退職数

df_dep = pd.crosstab(df['Attrition'], df['Department'])
print(df_dep)

df_dep.plot.bar()
# 比率変わらないのであまり関係ない？
# 出張の頻度別の退職数

df_biz = pd.crosstab(df['Attrition'], df['BusinessTravel'])
print(df_biz)

df_biz.plot.bar()
# 出張の頻度は関係なさそう
# 環境への満足度別の退職者数

df_es = pd.crosstab(df['Attrition'], df['EnvironmentSatisfaction'])
print(df_es)
df_es.plot.bar()
# 辞める人は高くても辞めている
# 仕事への満足度別の退職者数

df_js = pd.crosstab(df['Attrition'], df['JobSatisfaction'])
print(df_js)
df_js.plot.bar()
# 辞める人は高くても辞めている
# ワークライフバランス別の退職者数

df_wb = pd.crosstab(df['Attrition'], df['WorkLifeBalance'])
print(df_wb)
df_wb.plot.bar()
# 3.0=Betterでも辞めている（社員に占める比率が多い）
# 評価（仕事への関与）別の退職者数

df_jb = pd.crosstab(df['Attrition'], df['JobInvolvement'])
print(df_jb)
df_jb.plot.bar()
# 3.0=Highでも辞めている（社員に占める比率が多い）
# 評価（パフォーマンス）別の退職者数

df_pr = pd.crosstab(df['Attrition'], df['PerformanceRating'])
print(df_pr)
df_pr.plot.bar()
# PerformanceRatingは３と４しかない？のね・・・すごい会社

df.groupby(['PerformanceRating'])['EmployeeID'].count()
sns.boxplot(x="Attrition", y="Age", data=df)
# 労働時間と退職
sns.boxplot(x="Attrition", y="time_std", data=df)
# 月給と退職
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df)
# 勤続年数と退職
sns.boxplot(x="Attrition", y="TotalWorkingYears", data=df)
# 給与UP率と退職
sns.boxplot(x="Attrition", y="PercentSalaryHike", data=df)
df_att = df.loc[df.Attrition =="Yes"]
df_att.head()
df_natt = df.loc[df.Attrition =="No"]
df_natt.head()
print(df_att.mean())
print(df_natt.mean())
df_s = df_natt.mean() - df_att.mean()
df_s
df_d = df_natt.mean() / df_att.mean()
df_d
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Attrition']=le.fit_transform(df['Attrition'])
df['BusinessTravel'] = le.fit_transform(df['BusinessTravel'])
df['Department'] = le.fit_transform(df['Department'])
df['EducationField'] = le.fit_transform(df['EducationField'])
df['Gender'] = le.fit_transform(df['Gender'])
df['JobRole'] = le.fit_transform(df['JobRole'])
df['MaritalStatus'] = le.fit_transform(df['MaritalStatus'])
df['Over18'] = le.fit_transform(df['Over18'])
corr_cols = df[['Age','Attrition','BusinessTravel','DistanceFromHome',
                'Education', 'EducationField','Gender', 'JobLevel', 'JobRole',
                'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked',
                'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears',
                'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',
                'YearsWithCurrManager','time_std', 'EnvironmentSatisfaction',
                'JobSatisfaction', 'WorkLifeBalance','JobInvolvement', 
                'PerformanceRating']]
corr = corr_cols.corr()
plt.figure(figsize=(20,15))
sns.heatmap(corr, annot = True)
plt.show()
col = df[['time_std', 'TotalWorkingYears', 'MaritalStatus', 'YearsWithCurrManager',
      'Age','YearsAtCompany', 'EnvironmentSatisfaction', 'JobSatisfaction']]
col.isnull().sum()
col.fillna(0,inplace =True)
col.isnull().any()
X = col
y = df['Attrition']
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state=50)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
# 労働時間と給与UPに退職の相関があるのか？　（28：time_std, 16：PercentSalaryHike）

plt.scatter(df.iloc[:, 28], df.iloc[:, 16], c=df['Attrition'])
# 労働時間と月給に退職の相関があるのか？　（28：time_std, 1３：MonthlyIncome）

plt.scatter(df.iloc[:, 28], df.iloc[:, 13], c=df['Attrition'])
# 労働時間が長くて月給が低い層に退職（黄色）は多い
# 年齢と月収に退職の相関があるのか？　（0：Age, 1３：MonthlyIncome）

plt.scatter(df.iloc[:, 0], df.iloc[:, 13], c=df['Attrition'])
# 給与は年齢では決まってないが、若手で月収低い層に退職が多い
# 給与が高い層、高年齢層でも退職は出現
# 新たな変数を作成＝年齢と月給

df['ageincome'] = df['Age'] * df['MonthlyIncome'] /100
# 労働時間と新変数（年齢＊月給）に退職の相関があるのか？　（28：time_std, ３４：ageincome）

plt.figure(figsize=(15,10))
plt.scatter(df.iloc[:, 28], df.iloc[:, 34], c=df['Attrition'])
# 労働時間２０００超えかつ、新変数２００００以下で退職が多い
# 部門ごとの労働時間

sns.boxplot(x="Department", y="time_std",data=df)
# パフォーマンス評価と給与UP率

sns.boxplot(x="PerformanceRating", y="PercentSalaryHike",data=df)
# パフォーマンス評価と月給

sns.boxplot(x="PerformanceRating", y="MonthlyIncome",data=df)

# 評価は月収に反映されていない？？？
# パフォーマンス評価と年齢

sns.boxplot(x="PerformanceRating", y="Age",data=df)