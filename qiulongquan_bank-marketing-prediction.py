import os

import numpy as np 

import pandas as pd

import missingno as msno

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

import phik



# Modeling libraries

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression



# Other settings

sns.set_style("whitegrid")

sns.set(rc={'figure.figsize':(10,8)})

# plt.rcParams['font.sans-serif'] = ['SimHei']

# plt.rcParams['axes.unicode_minus'] = False 

# sns.set(font='SimHei')

matplotlib.rc('font', family='TakaoPGothic')

sns.set_context("paper", font_scale=1.1)

# ignore warning for axis logger of matplotlib

matplotlib.axes._axes._log.setLevel('ERROR')

%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# set basically parameter

path='/kaggle/input/bank-marketing/bank-additional-full.csv'



# Set the number of rows and columns displayed

pd.options.display.max_rows = 100

pd.options.display.max_columns = 100

base_info=pd.read_csv(path,sep = ';')

# 予測結果の列はyに保存されます（预测结果列保存到y）

y = pd.get_dummies(base_info['y'], columns = ['y'], prefix = ['y'], drop_first = True)

base_info.head()
base_info.columns.values,len(base_info.columns.values)
# 欠損値がないかどうかを確認します（确认是否有missing丢失值）

msno.matrix(base_info,labels=True)
base_info.info()
client_info=base_info.iloc[:,0:7]

client_info.head()
print("Job \n {}\n {}".format(len(client_info['job'].unique()),client_info['job'].unique()))
print("Marital \n {}\n {}".format(len(client_info['marital'].unique()),client_info['marital'].unique()))
print("Education \n {}\n {}".format(len(client_info['education'].unique()),client_info['education'].unique()))
print("Default \n {}\n {}".format(len(client_info['default'].unique()),client_info['default'].unique()))
print("Housing \n {}\n {}".format(len(client_info['housing'].unique()),client_info['housing'].unique()))
print("Loan \n {}\n {}".format(len(client_info['loan'].unique()),client_info['loan'].unique()))
base_info['age'].describe()
sns.boxplot(x=base_info["age"],data=base_info)
fig,ax=plt.subplots()

fig.set_size_inches(15,8)

sns.countplot(x='age',data=base_info)

ax.set_xlabel("Age",fontsize=15)

ax.set_ylabel("Count",fontsize=15)

ax.set_title("Age count distribution chart",fontsize=15)
# 计算outlier范围higher_outlier和lower_outlier（Calculate the outlier range higher_outlier and lower_outlier）

print(client_info['age'].describe())

higher_outlier=client_info['age'].quantile(q = 0.75)+1.5*(client_info['age'].quantile(q = 0.75)-client_info['age'].quantile(q = 0.25))

lower_outlier=client_info['age'].quantile(q = 0.25)-1.5*(client_info['age'].quantile(q = 0.75)-client_info['age'].quantile(q = 0.25))

print("higher_outlier={}\nlower_outlier={}".format(higher_outlier,lower_outlier))
outlier_count=client_info[client_info['age']>higher_outlier]['age'].count()

print("outlier count={}".format(outlier_count))

print("outlier rate={} %".format(round(outlier_count*100/len(client_info),2)))
fig,ax=plt.subplots()

fig.set_size_inches(15,6)

sns.countplot(client_info['job'])

ax.set_xlabel("Job",fontsize=10)

ax.set_ylabel("Count",fontsize=10)

ax.set_title("Job distribution chart")

sns.despine()
fig,ax=plt.subplots()

fig.set_size_inches(10,6)

sns.countplot(client_info['marital'])

ax.set_xlabel("marital",fontsize=10)

ax.set_ylabel("Count",fontsize=10)

ax.set_title("marital distribution chart")

sns.despine()
fig,ax=plt.subplots()

fig.set_size_inches(15,6)

sns.countplot(client_info['education'])

ax.set_xlabel("education",fontsize=10)

ax.set_ylabel("Count",fontsize=10)

ax.set_title("education distribution chart")

sns.despine()
fig,(ax1,ax2,ax3)=plt.subplots(nrows=1,ncols=3,figsize=(20,6))

sns.countplot(x='default',ax=ax1,data=client_info,order = ['yes', 'no', 'unknown'])

ax1.set_title("default columns data distribution chart",fontsize=15)

sns.countplot(x='housing',ax=ax2,data=client_info,order = ['yes', 'no', 'unknown'])

ax2.set_title("housing columns data distribution chart",fontsize=15)

sns.countplot(x='loan',ax=ax3,data=client_info,order = ['yes', 'no', 'unknown'])

ax3.set_title("loan columns data distribution chart",fontsize=15)
print("number of 'yes' in default={}\n".format(client_info[client_info['default']=='yes']['age'].count()))

print("number of 'no' in default={}\n".format(client_info[client_info['default']=='no']['age'].count()))

print("number of 'unknown' in default={}\n".format(client_info[client_info['default']=='unknown']['age'].count()))
print("number of 'yes' in housing={}\n".format(client_info[client_info['housing']=='yes']['age'].count()))

print("number of 'no' in housing={}\n".format(client_info[client_info['housing']=='no']['age'].count()))

print("number of 'unknown' in housing={}\n".format(client_info[client_info['housing']=='unknown']['age'].count()))
print("number of 'yes' in loan={}\n".format(client_info[client_info['loan']=='yes']['age'].count()))

print("number of 'no' in loan={}\n".format(client_info[client_info['loan']=='no']['age'].count()))

print("number of 'unknown' in loan={}\n".format(client_info[client_info['loan']=='unknown']['age'].count()))
# 从下图结果发现，不订阅的人数远远大于订阅的人数

# From the result of the figure below, it is found that the number of people who do not subscribe is far greater than the number of people who subscribe

fig,ax=plt.subplots()

fig.set_size_inches(8,5)

sns.countplot(base_info['y'])

# ax.set_xlabel("education",fontsize=10)

# ax.set_ylabel("Count",fontsize=10)

ax.set_title("y column distribution chart")

sns.despine()
# 订阅的人的年龄分布状况（Age distribution of subscribers）

fig,ax=plt.subplots()

fig.set_size_inches(15,8)

sns.countplot(x='age',data=base_info[base_info['y']=='yes'])

ax.set_title("subscriptioner's age distribution chart",fontsize=15)
# 根据下图发现订阅主要集中在30到50岁的人群，不订阅集中在32到47岁的人群

# （According to the figure below, subscriptions are mainly concentrated in people aged 30 to 50, and non-subscriptions are concentrated in people aged 32 to 47.）

plt.rcParams['figure.figsize'] = 10,5

sns.boxplot(x='y', y='age', data=base_info)

plt.title("Correlation between y column and age",fontsize=15)

plt.show()
# 不同年龄段和y列的相关性表示（Correlation representation of different age groups and y column）

from matplotlib.colors import LogNorm

sns.heatmap(base_info.groupby(["age", "y"]).size().unstack(),norm=LogNorm())
# 根据heatmap展示y列和那些字段有相关性（According to the heatmap, the y column is related to those fields）

fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(base_info.phik_matrix(), vmin=-1, vmax=1, cmap='Blues', cbar=True, annot=False, ax=ax)
client_info_encoder=client_info.copy()

# 标签编码转换，将现在的字符转换成数字(Label encoding conversion, convert current characters into numbers)

labelencoder = LabelEncoder()

client_info_encoder['job']      = labelencoder.fit_transform(client_info_encoder['job']) 

client_info_encoder['marital']  = labelencoder.fit_transform(client_info_encoder['marital']) 

client_info_encoder['education']= labelencoder.fit_transform(client_info_encoder['education']) 

client_info_encoder['default']  = labelencoder.fit_transform(client_info_encoder['default']) 

client_info_encoder['housing']  = labelencoder.fit_transform(client_info_encoder['housing']) 

client_info_encoder['loan']     = labelencoder.fit_transform(client_info_encoder['loan']) 
client_info_encoder.head()
#function to creat group of ages, this helps because we have 78 differente values here

def age(dataframe):

    dataframe.loc[dataframe['age'] <= 32, 'age'] = 1

    dataframe.loc[(dataframe['age'] > 32) & (dataframe['age'] <= 47), 'age'] = 2

    dataframe.loc[(dataframe['age'] > 47) & (dataframe['age'] <= 70), 'age'] = 3

    dataframe.loc[(dataframe['age'] > 70) & (dataframe['age'] <= 98), 'age'] = 4

           

    return dataframe



age(client_info_encoder);
# print(client_info_encoder['age'].unique())

fig,ax=plt.subplots(figsize=(10,6))

sns.countplot(x='age',data=client_info_encoder,order = [1, 2, 3, 4])

ax.set_title("Age group distribution status")

ax.set_xlabel("Age group")
related_info=base_info.iloc[:,7:11]
related_info.head()
fig,ax=plt.subplots()

fig.set_size_inches(10,6)

sns.countplot(x='duration',data=related_info)
# 计算outlier范围higher_outlier和lower_outlier（Calculate the outlier range higher_outlier and lower_outlier）

print(related_info['duration'].describe())

higher_outlier=related_info['duration'].quantile(q = 0.75)+1.5*(related_info['duration'].quantile(q = 0.75)-related_info['duration'].quantile(q = 0.25))

lower_outlier=related_info['duration'].quantile(q = 0.25)-1.5*(related_info['duration'].quantile(q = 0.75)-related_info['duration'].quantile(q = 0.25))

print("higher_outlier={}\nlower_outlier={}".format(higher_outlier,lower_outlier))
#function to creat group of duration, because of duration is importance item and according to grouping seems accuracy improve and decrease training time，Treat outliers separately as a group

# 因为持续时间是重要的项目并且根据分组似乎准确性提高并且减少了训练时间,把异常值单独作为一个组处理

def duration(dataframe):

    dataframe.loc[dataframe['duration'] <= 102, 'duration'] = 1

    dataframe.loc[(dataframe['duration'] > 102) & (dataframe['duration'] <= 180)  , 'duration']    = 2

    dataframe.loc[(dataframe['duration'] > 180) & (dataframe['duration'] <= 319)  , 'duration']   = 3

    dataframe.loc[(dataframe['duration'] > 319) & (dataframe['duration'] <= 644.5), 'duration'] = 4

    dataframe.loc[dataframe['duration']  > 644.5, 'duration'] = 5



duration(related_info);
fig,ax=plt.subplots()

fig.set_size_inches(10,6)

sns.countplot(x='duration',data=related_info)

ax.set_xlabel("After convert duration distribution chart")

ax.set_title("Duration distribution chart")
fig,(ax1,ax2,ax3)=plt.subplots(nrows=1,ncols=3,figsize=(15,6))

sns.countplot(x='contact',data=related_info,ax=ax1)

sns.countplot(x='month',data=related_info,ax=ax2,order=['mar', 'apr','may', 'jun', 'jul', 'aug','sep', 'oct', 'nov', 'dec'])

sns.countplot(x='day_of_week',data=related_info,ax=ax3)
related_info_encoder=related_info.copy()

labelencoder=LabelEncoder()

related_info_encoder['contact']=labelencoder.fit_transform(related_info_encoder['contact'])

related_info_encoder['month']=labelencoder.fit_transform(related_info_encoder['month'])

related_info_encoder['day_of_week']=labelencoder.fit_transform(related_info_encoder['day_of_week'])

related_info_encoder['duration']=labelencoder.fit_transform(related_info_encoder['duration'])
related_info_encoder.head()
social_info=base_info.iloc[:,15:20]

social_info.head()
# 根据下图，都是数值型数据，不需要转换（According to the figure below, they are all numerical data and do not need to be converted）

fig,(ax1,ax2,ax3,ax4,ax5)=plt.subplots(nrows=1,ncols=5,figsize=(20,6))

sns.set(rc={'figure.figsize':(10,5)})

sns.lineplot(data=social_info['emp.var.rate'],ax=ax1)

sns.lineplot(data=social_info['cons.price.idx'],ax=ax2)

sns.lineplot(data=social_info['cons.conf.idx'],ax=ax3)

sns.lineplot(data=social_info['euribor3m'],ax=ax4)

sns.lineplot(data=social_info['nr.employed'],ax=ax5)
other_info=base_info.iloc[:,11:15]

other_info.head()
print(other_info['campaign'].describe())

sns.lineplot(data=other_info['campaign'])
# 计算outlier范围higher_outlier和lower_outlier（Calculate the outlier range higher_outlier and lower_outlier）

higher_outlier=other_info['campaign'].quantile(q = 0.75)+1.5*(other_info['campaign'].quantile(q = 0.75)-other_info['campaign'].quantile(q = 0.25))

lower_outlier=other_info['campaign'].quantile(q = 0.25)-1.5*(other_info['campaign'].quantile(q = 0.75)-other_info['campaign'].quantile(q = 0.25))

print("higher_outlier={}\nlower_outlier={}".format(higher_outlier,lower_outlier))
#function to creat group of campaign. according to grouping seems accuracy improve and decrease training time，Treat outliers separately as a group

# 通过分组准确性提高并且减少了训练时间,把异常值单独作为一个组处理

def campaign(dataframe):

    dataframe.loc[dataframe['campaign'] <= 1, 'campaign'] = 1

    dataframe.loc[(dataframe['campaign'] > 1) & (dataframe['campaign'] <= 2)  , 'campaign']    = 2

    dataframe.loc[(dataframe['campaign'] > 2) & (dataframe['campaign'] <= 3)  , 'campaign']   = 3

    dataframe.loc[(dataframe['campaign'] > 3) & (dataframe['campaign'] <= 6), 'campaign'] = 4

    dataframe.loc[dataframe['campaign']  > 6, 'campaign'] = 5



campaign(other_info);
# 转换后的campaign状态（Campaign status after conversion）

sns.countplot(x='campaign',data=other_info)
sns.countplot(x='pdays',data=other_info)
labelencoder = LabelEncoder()

other_info['poutcome'] = labelencoder.fit_transform(other_info['poutcome']) 
sns.countplot(x='poutcome',data=other_info)
other_info.head()
bank= pd.concat([client_info_encoder, related_info_encoder, other_info, social_info], axis = 1)

bank.columns=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',

                     'contact', 'month', 'day_of_week', 'duration','campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 

                     'cons.conf.idx', 'euribor3m', 'nr.employed']

bank.shape
columns=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan','contact', 'month', 'day_of_week', 'duration','campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
bank.head()
x_train, x_test, y_train, y_test = train_test_split(bank, y, test_size = 0.15, random_state = 1234)

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
x_train.shape,y_train.shape
x_train.head()
sc_X = StandardScaler()

x_train_scale = sc_X.fit_transform(x_train)

x_test_scale = sc_X.transform(x_test)
x_train_scale_pd=pd.DataFrame(x_train_scale)

x_train_scale_pd.columns=[x_train.columns]

x_train_scale_pd
sc_X = StandardScaler()

bank_all_info_scale = sc_X.fit_transform(bank)

bank_all_info_scale=pd.DataFrame(bank_all_info_scale)

bank_all_info_scale.columns=[bank.columns]

bank_all_info_scale
bank_all_info_scale=pd.concat([bank_all_info_scale,y],axis = 1)

bank_all_info_scale.columns=[base_info.columns]

bank_all_info_scale
fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(bank_all_info_scale.corr(),

            vmin=-1,

            vmax=1,

            cmap='bwr',

            cbar=True,

            annot=False,

            ax=ax)
%%time

from sklearn.linear_model import LassoCV

CV_count=3

lasso = LassoCV(cv=CV_count, random_state=1234).fit(x_train_scale_pd, y_train.values.ravel())

coef = np.abs(lasso.coef_)

coef=coef.reshape(1,-1)

df_importance = pd.DataFrame(coef,columns=bank.columns)

df_importance
# Showing top 10 important features

n_top_features = 10

df_importance_top = df_importance.median().sort_values(ascending=False)[:n_top_features]

name_features = df_importance_top.index

left = name_features

height = df_importance_top.values

plt.bar(left, height,align="center",linewidth=1)

plt.xticks(rotation=270)

plt.title("feature importance")

plt.xlabel("feature variables")

plt.ylabel("degree")

plt.grid(True)

print(name_features)
x_train_scale = pd.DataFrame(sc_X.fit_transform(x_train[name_features]),

                              columns=x_train[name_features].columns)

x_test_scale = pd.DataFrame(sc_X.transform(x_test[name_features]),

                             columns=x_test[name_features].columns)
x_train_scale.shape,x_test_scale.shape
logmodel = LogisticRegression() 

model=logmodel.fit(x_train_scale,y_train.values.ravel())

logpred = model.predict(x_test_scale)
from sklearn.metrics import plot_confusion_matrix

print("Prediction accuracy={} %".format(round(accuracy_score(y_test, logpred),2)*100))

LOGCV = (cross_val_score(logmodel, x_train_scale, y_train.values.ravel(), cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())

print("Cross Validation={}".format(LOGCV))
# confusion matrix chart show

disp = plot_confusion_matrix(model, x_test_scale, y_test,

                                 display_labels=['subscribe','No subscribe'],

                                 cmap=plt.cm.Blues,

                                 normalize=None)

disp.ax_.set_title("Prediction Subscribe Result Show")

print("Prediction Subscribe Result Show")

print(disp.confusion_matrix)

plt.show()
from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(x_train_scale, y_train.values.ravel())

xgbprd = xgb.predict(x_test_scale)
print(confusion_matrix(y_test, xgbprd ))

print(round(accuracy_score(y_test, xgbprd),2)*100)

XGB = (cross_val_score(estimator = xgb, X = x_train_scale, y = y_train.values.ravel(), cv = 10).mean())
%%time



import time

from sklearn import svm

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from sklearn.model_selection import cross_val_score, StratifiedKFold





def objective(params):

    params = {

        'C': abs(float(params['C'])),

        "kernel": str(params['kernel']),

    }

    clf = svm.SVC(gamma='scale', **params)



    #     采用前5000条数据,数据量进行超参数优化测试（Using the first 5000 data,data volume for hyperparameter optimization test）

    score = -np.mean(

        cross_val_score(clf,

                        x_train_scale[:10000],

                        y_train.values.ravel()[:10000],

                        cv=3,

                        n_jobs=-1,

                        scoring="neg_mean_squared_error"))



    print("loss score {:.3f} params {}".format(score, params))

    return {'loss': score, 'status': STATUS_OK}





start_time = time.time()

space = {

    'C': hp.normal('C', 0, 50),

    "kernel": hp.choice('kernel', ['rbf']),

}



best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=5)

kernel = ['rbf']

best['kernel'] = kernel[best['kernel']]

best['C'] = abs(best['C'])

hyperparametertuning_time = round(time.time() - start_time, 2)

print(

    "SVC: Hyperopt estimated optimum {}. Hyper parameter tuning spend time {}".

    format(best, hyperparametertuning_time))
%%time

start_time = time.time()

svc = svm.SVC(gamma='scale', **best)

svc.fit(x_train_scale,y_train.values.ravel())

training_time = round(time.time() - start_time, 2)

print("SVC: Training spend time {}".format(training_time))
svcprd = svc.predict(x_test_scale)

print(confusion_matrix(y_test, svcprd))

print("Prediction accuracy={} %".format(round(accuracy_score(y_test, svcprd),2)*100))

SVC = (cross_val_score(logmodel, x_train_scale, y_train.values.ravel(), cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())

print("Cross Validation={}".format(SVC))
# confusion matrix chart show

disp = plot_confusion_matrix(svc, x_test_scale, y_test,

                                 display_labels=['subscribe','No subscribe'],

                                 cmap=plt.cm.Blues,

                                 normalize=None)

disp.ax_.set_title("Prediction Subscribe Result Show")

print("Prediction Subscribe Result Show")

print(disp.confusion_matrix)

plt.show()
models = pd.DataFrame({

                'Models': ['Logistic Classifier', 'XGBoost','SVClassifier' ],

                'Score':  [LOGCV,XGB,SVC]})



models.sort_values(by='Score', ascending=False)