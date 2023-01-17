# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb



from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn import model_selection

from sklearn import metrics
data = pd.read_csv('../input/HR_comma_sep.csv')

data.info()   # 没有缺失值
data.head()
data.describe()
data[data.isnull().values == True]
data_copy = data.copy(deep = True)  #对原始数据做一个拷贝

data_copy.describe()
data['satisfBin'] = pd.cut(data['satisfaction_level'],[0.08,0.44,0.6129,0.82,1.0])

data['evaluBin'] = pd.cut(data['last_evaluation'], [0.35,0.56,0.72,0.87,1.0])

data['hoursBin'] = pd.qcut(data['average_montly_hours'], 4) 



label = LabelEncoder()

data['satisf_code'] = label.fit_transform(data['satisfBin'])

data['evalu_code'] = label.fit_transform(data['evaluBin'])

data['sales_code'] = label.fit_transform(data['sales'])

data['salary_code'] = label.fit_transform(data['salary'])

data['hours_code'] = label.fit_transform(data['hoursBin'])



Target = ['left']

data_x = ['satisfBin','evaluBin','number_project','hoursBin','time_spend_company','Work_accident',

          'promotion_last_5years','sales','salary',]



data_xy_Bin = data_x + Target  # 将数据转换为类别

print('Bin X Y: ', data_xy_Bin, '\n')



data_dummy = pd.get_dummies(data.drop(columns='left'))  # 将类别变量转换为分类变量

data_x_dummy = data_dummy.columns.tolist()

data_xy_dummy = Target + data_x_dummy

print('Dummy X Y: ', data_xy_dummy, '\n')



data_dummy.head()
salary_left = pd.crosstab(data['salary'], data['left'])

salary_left = salary_left.reindex(index = ['low','medium','high'])

salary_left.plot(kind='bar')
sales_left = pd.crosstab(data['sales'], data['left'])

sales_left.plot(kind = 'bar')   
plt.figure(figsize=[10,6])



plt.subplot(121)

plt.boxplot(data['satisfaction_level'], showmeans = True, meanline = True)

plt.title('satisfaction_level Boxplot')

plt.ylabel('satisfaction_level (#)')



plt.subplot(122)

sb.countplot(x = data['left'],data = data, hue = 'satisfBin')



satis_left_ = pd.crosstab(data['satisfBin'],data['left'])

satis_left_.plot.pie(y=1, autopct='%.2f', figsize=(10,6))
plt.figure(figsize=[16,8])



plt.subplot(131)

plt.boxplot(data['last_evaluation'], showmeans = True, meanline = True)

plt.title('last_evaluation of staffs')

plt.ylabel('last_evaluation (#)')



plt.subplot(133)

plt.boxplot(data[data['left']==1]['last_evaluation'], showmeans = True, meanline = True)

plt.title('last_evaluation of left')

plt.ylabel('last_evaluation (#)')



plt.subplot(132)

sb.countplot(x = data['left'],data = data, hue = 'evaluBin')



satis_left_ = pd.crosstab(data['evaluBin'],data['left'])

satis_left_.plot.pie(y=1, autopct='%.2f', figsize=(10,6))
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))



sb.boxplot(x = 'promotion_last_5years', y = 'time_spend_company', hue = 'left', data = data, ax = axis1)

axis1.set_title('promotion vs work_year left Comparison')



sb.boxplot(x = 'promotion_last_5years', y = 'average_montly_hours', hue = 'left', data = data, ax = axis2)

axis2.set_title('promotion vs montly_hours left Comparison')



sb.boxplot(x = 'promotion_last_5years', y ='number_project', hue = 'left', data = data, ax = axis3)

axis3.set_title('promotion vs number_project left Comparison')
X = data_dummy

y = data_copy[Target]

Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size = 0.3, random_state = 42)

model = LogisticRegression()

model.fit(Xtrain, ytrain)
pred = model.predict(Xtest)

metrics.accuracy_score(ytest, pred)  # 输出预测精确度
metrics.confusion_matrix(ytest, pred)  # 输出混淆矩阵,行表示真实值，列表示预测值
print(metrics.classification_report(ytest, pred))
model_selection.cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)  # 输出10份交叉验证
result = ytest.copy(deep=True)

result.insert(1,'predict_left', pred)

result.to_csv("left_predictions.csv", index=False)