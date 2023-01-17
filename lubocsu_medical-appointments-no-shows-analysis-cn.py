# Import the neccessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# Load relevant .csv files into DataFrames
# 加载数据集文件
df = pd.read_csv('../input/KaggleV2-May-2016.csv')
df.head()
# check missing values/data types/the numbers of columns and rows.
# 检查缺失的值/数据类型/列和行的数量
df.info()
# replace the column name with Show_rate and check the result.
# 将No-show列名替代为Show_rate,并且检查替代效果。
df = df.rename(columns = {'No-show':'Show_rate'})
df.columns
# reolace the column value with 0 or 1 and check the result.
# The Show_rate data uses 1 or 0 to represent the show condition, so the mean of "Show_rate" column is actually the Show rate.
# 将Show_rate列值替代为0,1用来表示预约出席率并检查结果。
# 将Show_rate列值替代为0,1用来表示预约出席情况，所以Show_rate列的平均值就表示为出席率

df['Show_rate'] = df['Show_rate'].replace(['No','Yes'],[1,0])
df.head()
# check for any erroneous values about every column and show the result.
# 检查每一列值的范围并且展示结果
print('Gender:%s' % df['Gender'].unique())
print('Age:%s' % sorted(df['Age'].unique()))
print('Scholarship:%s' % df['Scholarship'].unique())
print('Hipertension:%s' % df['Hipertension'].unique())
print('Diabetes:%s' % df['Diabetes'].unique())
print('Alcoholism:%s' % df['Alcoholism'].unique())
print('Handcap:%s' % df['Handcap'].unique())
print('SMS_received:%s' % df['SMS_received'].unique())
print('Show_rate:%s' % df['Show_rate'].unique()) 
# show the sample age distribution
# 显示样本的年龄分布
df.Age.hist()
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
# remove the impossible and the absurd from the data and check the result.
# 移除年龄列值中不合理的数值并显示结果
df = df[(df['Age'] >= 0) & (df['Age'] <= 100)]
print('Age:%s' % sorted(df['Age'].unique()))
# replace the absurd value with the reasonable value and check the result.
# 替代残障列中的不合理列值并检查结果
df['Handcap'] = df['Handcap'].replace([2,3,4],1)
df['Handcap'].unique()
# replace the columns number values with literal string to show the graph conveniently and check the result.
# 将列值中数字替代为文字以便图表显示并检查结果
for col in ['Hipertension','Diabetes','Alcoholism','Handcap']:
    df[col] = df[col].replace(1,'Yes')
    df[col] = df[col].replace(0,'No')
for col in ['Scholarship','SMS_received']:
    df[col] = df[col].replace(1,'Have')
    df[col] = df[col].replace(0,'None')
df.head()
# remove the unreasonable rows which datetime of AppointmentDay is earlier than ScheduledDay.
# 移除预约就诊日期早于预约日期不合理的行。
# convert the datatype of object into datetime64  
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
# calculate difference of  AppointmentDay and ScheduledDay as column Interval value.
df['Interval'] = df['AppointmentDay'].dt.normalize() - df['ScheduledDay'].dt.normalize()
df['Interval'] = df['Interval'].dt.days
# filter the row which Interval value is equal or above 0.
df_val = df[df['Interval'] >= 0]
df_val.info()
#check the number of duplicate rows. 
#检查重复的样本行
sum(df_val.duplicated())
# calculate and show the ratio of show-up in general.
# 计算和展示预约出席率
total_show = df_val['Show_rate'].sum()
total_no_show = df_val['Show_rate'].count() - df_val['Show_rate'].sum()
explode = [0,0.10]
plt.pie([total_no_show, total_show], shadow=True, startangle=90, labels=['No Show','Show'],autopct='%1.2f%%',explode=explode)
plt.title('General show rate') 
plt.axis('equal') 
plt.show()
# calculate and show the likelihood of show with respect to age.
# 计算不同年龄组的预约出席率
bins=[0, 10, 18, 30, 60, 80, 100]
group_name = ['children', 'teenager', 'youth', 'middle', 'old','very_old']
df_val['Age_group'] = pd.cut(df_val['Age'], bins=bins, labels=group_name)

print(df_val.groupby(['Age_group']).mean()['Show_rate'].describe())
df_val.groupby(['Age_group']).mean()['Show_rate'].plot(kind='bar',grid=True)
plt.yticks(np.arange(0,1,0.05))
plt.xlabel('Age_group')
plt.ylabel('Show rate')
plt.title('Show rate of age')
# 展示不同年龄阶段样本数量
df_val.groupby(['Age_group']).count()['PatientId'].plot(kind='bar',grid=True)
plt.ylabel('Frequency')
plt.title('Age Group Distribution')
# calculate and show the likelihood of show with respect to gender.
# 计算不同性别的预约出席率
print(df_val.groupby(['Gender']).mean()['Show_rate'])
df_val.groupby(['Gender']).mean()['Show_rate'].rename({'F':'Female','M':'Male'}).plot(kind='bar',grid=True)
plt.xlabel('Gender')
plt.ylabel('Show rate')
plt.title('Show rate of gender')
plt.yticks(np.arange(0,1.1,0.1))
# show the sample interval distribution 
# 展示样本按照预约时间间隔分布情况
# sns.set(style="whitegrid")
# ax = sns.stripplot(data = df_val, y = 'Interval',jitter=True)
df.Interval.hist()
plt.xlabel('Interval')
plt.ylabel('Frequency')
plt.title('Interval Distribution')
# calculate and show the likelihood of show with respect to interval.
# 计算和展示不同预约时间间隔的预约出席率。
print(df_val.groupby(['Interval']).mean()['Show_rate'].describe())
# plt.figure(figsize= (18 ,5))
df_val.groupby(['Interval']).mean()['Show_rate'].plot(kind='line')
plt.xlabel('Interval(day)')
plt.ylabel('Show rate')
plt.title('Show rate of interval')
plt.yticks(np.arange(0,1.1,0.1))
# calculate and show the likelihood of show with respect to neighbourhood.
# 计算和展示不同地点的预约出席率。
print(df_val.groupby(['Neighbourhood']).mean()['Show_rate'].describe())
plt.figure(figsize= (15 ,5))
df_val.groupby(['Neighbourhood']).mean()['Show_rate'].sort_values(ascending=False).plot(kind='bar',grid=True)
plt.xlabel('Neighbourhood')
plt.ylabel('Show rate')
plt.title('Show rate of Neighbourhood')
plt.yticks(np.arange(0,1.1,0.1))
# calculate and show the likelihood of show with respect to scholarship.
# 计算和展示医保金有无与预约出席率关系。
# 定义可以重复表示图表的函数show_graph()
def show_graph(feature):
    print(df_val.groupby([feature]).mean()['Show_rate'])
    df_val.groupby([feature]).mean()['Show_rate'].rename({0:'None',1:'Have'}).plot(kind='bar',grid=True)
    plt.xlabel(feature)
    plt.ylabel('Show rate')
    plt.title('Show rate of %s' % feature)
    plt.yticks(np.arange(0,1.1,0.1))
show_graph('Scholarship')    
# calculate and show the likelihood of show with respect to SMS_received.
# 调用可重复表示相似图表的函数show_graph
show_graph('SMS_received')  
# calculate and show the likelihood of show and no show with respect to diseases.
# 计算和展示不同疾病的预约出席率。
ser_hi = df_val.groupby(['Hipertension']).mean()['Show_rate']
ser_d = df_val.groupby(['Diabetes']).mean()['Show_rate']
ser_a = df_val.groupby(['Alcoholism']).mean()['Show_rate']
ser_ha = df_val.groupby(['Handcap']).mean()['Show_rate']
df_disease = pd.DataFrame({ser_hi.index.name:ser_hi.values,
                           ser_d.index.name:ser_d.values,
                           ser_a.index.name:ser_a.values,
                           ser_ha.index.name:ser_ha.values,
                          },index=['no show','show'])
print(df_disease)
df_disease.T.plot(kind='bar',y=['no show','show'],grid=True)
plt.xlabel('Diseases')
plt.ylabel('Show rate')
plt.title('Show rate of Diseases')
plt.yticks(np.arange(0,1.1,0.1))
# calculate and show  the sample distribution of show with respect to diseases/age/scholarship.
# 计算和展示疾病的种类、年龄大小、救助金与否怎样影响预约出席的人员分布。
df_val_show = df_val[df_val['Show_rate'] == 1]
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(15,10))
fig.suptitle('Show statistics related to diseases/age/scholarship')
ax = sns.violinplot(x='Hipertension', y='Age', data=df_val_show,inner=None, color=".8", ax=axes[0,0])
ax = sns.stripplot(x="Hipertension", y="Age", data=df_val_show, jitter=True, hue="Scholarship", palette="Set2", dodge=True, ax=axes[0,0])

ax = sns.violinplot(x='Diabetes', y='Age', data=df_val_show,inner=None, color=".8", ax=axes[0,1])
ax = sns.stripplot(x="Diabetes", y="Age", data=df_val_show, jitter=True, hue="Scholarship", palette="Set2", dodge=True, ax=axes[0,1])

ax = sns.violinplot(x='Alcoholism', y='Age', data=df_val_show,inner=None, color=".8", ax=axes[1,0])
ax = sns.stripplot(x="Alcoholism", y="Age", data=df_val_show, jitter=True, hue="Scholarship", palette="Set2", dodge=True, ax=axes[1,0])

ax = sns.violinplot(x='Handcap', y='Age', data=df_val_show,inner=None, color=".8", ax=axes[1,1])
ax = sns.stripplot(x="Handcap", y="Age", data=df_val_show, jitter=True, hue="Scholarship", palette="Set2", dodge=True, ax=axes[1,1])
# calculate and show  the sample distribution of show with respect to diseases/interval/scholarship.
df_val_show = df_val[df_val['Show_rate'] == 0]
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(15,10))
fig.suptitle('Show statistics related to diseases/interval/scholarship')
ax = sns.violinplot(x='Hipertension', y='Interval', data=df_val_show,inner=None, color=".8", ax=axes[0,0])
ax = sns.stripplot(x="Hipertension", y="Interval", data=df_val_show, jitter=True, hue="Scholarship", palette="Set2", dodge=True, ax=axes[0,0])

ax = sns.violinplot(x='Diabetes', y='Interval', data=df_val_show,inner=None, color=".8", ax=axes[0,1])
ax = sns.stripplot(x="Diabetes", y="Interval", data=df_val_show, jitter=True, hue="Scholarship", palette="Set2", dodge=True, ax=axes[0,1])

ax = sns.violinplot(x='Alcoholism', y='Interval', data=df_val_show,inner=None, color=".8", ax=axes[1,0])
ax = sns.stripplot(x="Alcoholism", y="Interval", data=df_val_show, jitter=True, hue="Scholarship", palette="Set2", dodge=True, ax=axes[1,0])

ax = sns.violinplot(x='Handcap', y='Interval', data=df_val_show,inner=None, color=".8", ax=axes[1,1])
ax = sns.stripplot(x="Handcap", y="Interval", data=df_val_show, jitter=True, hue="Scholarship", palette="Set2", dodge=True, ax=axes[1,1])
from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])
