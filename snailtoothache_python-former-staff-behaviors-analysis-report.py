import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv("/kaggle/input/human-resource-analytics/HR_Data.csv")

df.head()
# change column name

df = df.rename(columns={'sales':'department'})

df.head()
df.describe(include='all')
df.shape
df.info()
df.describe()
# Check duplicated values 查看重复值

df.duplicated().sum()
# remove duplicate values 删除重复值

df_NoDup=df.drop_duplicates()

df_NoDup.duplicated().sum()
df.shape
# check null values 检查空值

df.isnull().sum()
# Generate new dataframe 生成新列表

a = df['left'].value_counts()

b = pd.DataFrame(a)

b.head()
# Rename columns 更改列名

b.index =('Current','Left')

b.head()
# 对表b进行转置

b.T
plt.pie(b['left'],labels=b.index,shadow = True, autopct='%.0f%%',explode=[0.1,0])

plt.title('Overall Turnover Rate')

fig1 = plt.gcf()

fig1.set_size_inches(6,6)
# 生成新列表: 最多的离职部门

c = df['department'].groupby([df['left']])

d = c.value_counts()

d
# 生成新列表: 最多的离职部门

gd=pd.DataFrame(d)

gd=gd.drop(0)

gd
# 生成新列表: 最多的离职部门

gd = gd.rename(columns={'department':'department','department':'left_total_1'})

gd=gd.reset_index(level=[0,1])

gd
# 生成新列表: 最多的离职部门

sns.barplot(x='department',y='left_total_1',data=gd,palette='Blues_d').set_title('Top 3 left departments')

fig1 = plt.gcf()

fig1.set_size_inches(12,6)
# 生成新列表: 最多的离职部门

x = gd['department']

y = gd['left_total_1']

plt.barh(x,height=0.6,width=y,align='center')

plt.title('Top 3 left departments',loc='center')

for m,n in zip(x,y):

    plt.text(n,m,n,ha='left',va='center',fontsize=12)

plt.grid(False)

fig1 = plt.gcf()

fig1.set_size_inches(12,6)
# 生成新列表: 离职人员的满意度分布

e = df['satisfaction_level'].groupby([df['left']])

f = e.value_counts()

fd=pd.DataFrame(f)

fd=fd.drop(0)

fd.head()
fd = fd.rename(columns={'satisfaction_level':'satisfaction_level','satisfaction_level':'left_total_2'})

fd=fd.reset_index(level=[0,1])

fd.head()
sns.barplot(x=fd['satisfaction_level'],y=fd['left_total_2'])

fig1 = plt.gcf()

fig1.set_size_inches(15,6)

plt.title("Left staff satisfaction level distribution")
# 边界核密度估计图

sns.jointplot(x=fd['satisfaction_level'], y=fd["left_total_2"], kind='kde')

plt.title("Left staff satisfaction level distribution",loc='left')
sns.jointplot(x=fd['satisfaction_level'],y=fd['left_total_2'],kind='hex')

plt.title("Left staff satisfaction level distribution",loc='left')
# 生成新列表: 离职人员的工伤分布

g = df['Work_accident'].groupby([df['left']])

h = g.value_counts()

hd=pd.DataFrame(h)

hd=hd.drop(0)

hd.head()
hd = hd.rename(columns={'Work_accident':'Work_accident','Work_accident':'left_total_3'})

hd=hd.rename(index={0:'No Accident',1:'Accident'})

hd=hd.reset_index(level=[0,1])

hd.head()
# 生成新列表: 离职人员工伤分布

sns.barplot(x='Work_accident',y='left_total_3',data=hd,palette='Blues_d').set_title('Left staff work accident distribution ')

fig1 = plt.gcf()

fig1.set_size_inches(6,6)
# 生成新列表: 离职人员工伤分布

x1=hd['Work_accident']

y1=hd['left_total_3']

plt.barh(x1,height=0.6,width=y1,align='center')

plt.title('Left staff work accident distribution',loc='center')

for A,B in zip(x1,y1):

    plt.text(B,A,B,ha='left',va='center',fontsize=12)

plt.grid(False)

fig1 = plt.gcf()

fig1.set_size_inches(12,6)
# 生成新列表: 离职人员的升职分布

i = df['promotion_last_5years'].groupby([df['left']])

j = i.value_counts()

jd=pd.DataFrame(j)

jd=jd.drop(0)

jd.head()
jd = jd.rename(columns={'promotion_last_5years':'promotion_last_5years','promotion_last_5years':'left_total_4'})

jd=jd.rename(index={0:'No Promotion',1:'Promotion'})

jd=jd.reset_index(level=[0,1])

jd.head()
# 生成新列表: 离职人员的升职分布

sns.barplot(x='promotion_last_5years',y='left_total_4',data=jd,palette='Blues_d').set_title('Left staff promotion distribution ')

fig1 = plt.gcf()

fig1.set_size_inches(5,6)
# 生成新列表: 离职人员工伤分布

x2=jd['promotion_last_5years']

y2=jd['left_total_4']

plt.barh(x2,height=0.6,width=y2,align='center')

plt.title('Left staff promotion distribution',loc='center')

for A1,B1 in zip(x2,y2):

    plt.text(B1,A1,B1,ha='left',va='center',fontsize=12)

plt.grid(False)

fig1 = plt.gcf()

fig1.set_size_inches(12,6)
# 生成新列表: 离职人员的项目分布

p = df['number_project'].groupby([df['left']])

q = p.value_counts()

qd=pd.DataFrame(q)

qd=qd.drop(0)

qd
# 生成新列表: 离职人员的项目分布

qd = qd.rename(columns={'number_project':'number_project','number_project':'left_total_5'})

qd=qd.rename(index={2:'2 projects',6:'6 projects',5:'5 projects',4:'4 projects',7:'7 projects',3:'3 projects'})

qd=qd.reset_index(level=[0,1])

qd
# 生成新列表: 离职人员的项目分布

sns.barplot(x='number_project',y='left_total_5',data=qd,palette='Blues_d').set_title('Left staff projects distribution ')

fig1 = plt.gcf()

fig1.set_size_inches(12,6)
# 生成新列表: 离职人员的项目分布

x3=qd['number_project']

y3=qd['left_total_5']

plt.barh(x3,height=0.6,width=y3,align='center')

plt.title('Left staff projects distribution',loc='center')

for A2,B2 in zip(x3,y3):

    plt.text(B2,A2,B2,ha='left',va='center',fontsize=12)

plt.grid(False)

fig1 = plt.gcf()

fig1.set_size_inches(12,6)
# 生成新列表: 离职人员的服务时间分布

s = df['time_spend_company'].groupby([df['left']])

t = s.value_counts()

sd=pd.DataFrame(t)

sd=sd.drop(0)

sd
# 生成新列表: 离职人员的服务时间分布

sd = sd.rename(columns={'time_spend_company':'time_spend_company','time_spend_company':'left_total_6'})

sd=sd.rename(index={3:'3 Service Years',4:'4 Service Years',5:'5 Service Years',6:'6 Service Years',2:'2 Service Years'})

sd=sd.reset_index(level=[0,1])

sd
# 生成新列表: 离职人员的项目分布

sns.barplot(x='time_spend_company',y='left_total_6',data=sd,palette='Blues_d').set_title('Left staff serviced years distribution ')

fig1 = plt.gcf()

fig1.set_size_inches(10,6)
# 生成新列表: 离职人员的项目分布

x4=sd['time_spend_company']

y4=sd['left_total_6']

plt.barh(x4,height=0.6,width=y4,align='center')

plt.title('Left staff serviced years distribution',loc='center')

for A3,B3 in zip(x4,y4):

    plt.text(B3,A3,B3,ha='left',va='center',fontsize=12)

plt.grid(False)

fig1 = plt.gcf()

fig1.set_size_inches(12,6)
# 生成新列表: 离职人员的薪资分布

u = df['salary'].groupby([df['left']])

v = u.value_counts()

ud=pd.DataFrame(v)

ud=ud.drop(0)

ud
# 生成新列表: 离职人员的薪资分布

ud = ud.rename(columns={'salary':'salary','salary':'left_total_7'})

ud=ud.reset_index(level=[0,1])

ud
# 生成新列表: 离职人员的薪资分布

sns.barplot(x='salary',y='left_total_7',data=ud,palette='Blues_d').set_title('Left staff salary distribution ')

fig1 = plt.gcf()

fig1.set_size_inches(10,6)
# 生成新列表: 离职人员的薪资分布

x5=ud['salary']

y5=ud['left_total_7']

plt.barh(x5,height=0.6,width=y5,align='center')

plt.title('Left staff salary distribution',loc='center')

for A4,B4 in zip(x5,y5):

    plt.text(B4,A4,B4,ha='left',va='center',fontsize=12)

plt.grid(False)

fig1 = plt.gcf()

fig1.set_size_inches(12,6)
# 生成新列表: 离职人员的绩效结果分布

aa = df['last_evaluation'].groupby([df['left']])

bb = aa.value_counts()

xd=pd.DataFrame(bb)

xd=xd.drop(0)

xd.head()
# 生成新列表: 离职人员的绩效结果分布

xd = xd.rename(columns={'last_evaluation':'last_evaluation','last_evaluation':'left_total_8'})

xd=xd.reset_index(level=[0,1])

xd.head()
sns.barplot(x=xd['last_evaluation'],y=xd['left_total_8'])

fig1 = plt.gcf()

fig1.set_size_inches(15,6)

plt.title("Left staff peformance distribution")
sns.jointplot(x=xd["last_evaluation"], y=xd["left_total_8"], kind='scatter')

fig1.set_size_inches(10,10)

plt.title("Left staff peformance distribution",loc='left')
# 生成新列表: 离职人员的月工作时长分布

cc = df['average_montly_hours'].groupby([df['left']])

dd = cc.value_counts()

yd=pd.DataFrame(dd)

yd=yd.drop(0)

yd.head()
# 生成新列表: 离职人员的绩效结果分布

yd = yd.rename(columns={'average_montly_hours':'average_montly_hours','average_montly_hours':'left_total_9'})

yd = yd.reset_index(level=[0,1])

yd.head()
sns.barplot(x=yd['average_montly_hours'],y=yd['left_total_9'])

fig1 = plt.gcf()

fig1.set_size_inches(20,6)

plt.title("Left staff avg monthly working hours distribution")
sns.jointplot(x=yd["average_montly_hours"], y=yd["left_total_9"], kind='scatter')

fig1.set_size_inches(20,20)

plt.title("Left staff avg monthly working hours distribution",loc='left')
sns.jointplot(x=yd["average_montly_hours"], y=yd["left_total_9"], kind='reg')

fig1.set_size_inches(10,10)

plt.title("Left staff avg monthly working hours distribution",loc='right')
## Overall
# 公司所有人员的薪资分布

ee = df['salary']

ff = ee.value_counts()

ad = pd.DataFrame(ff)

ad
# 公司所有人员的薪资分布

ad = ad.reset_index()

ad
ad = ad.rename(columns={'index':'salary_level','salary':'total_1'})

ad
# 公司所有人员的薪资分布

sns.barplot(x='salary_level',y='total_1',data=ad,palette='Blues_d').set_title('Overall salary distribution ')

fig1 = plt.gcf()

fig1.set_size_inches(5,5)
# 公司所有人员的薪资分布

x6=ad['salary_level']

y6=ad['total_1']

plt.barh(x6,height=0.6,width=y6,align='center')

plt.title('Overall staff salary distribution',loc='center')

for A5,B5 in zip(x6,y6):

    plt.text(B5,A5,B5,ha='left',va='center',fontsize=12)

plt.grid(False)

fig1 = plt.gcf()

fig1.set_size_inches(12,6)
# 公司所有人员的满意度分布

gg = df['satisfaction_level']

hh = gg.value_counts()

bd = pd.DataFrame(hh)

bd.head()
bd.shape
bd = bd.reset_index()

bd.head()
# 公司所有人员的满意度分布

bd = bd.rename(columns={'index':'satisfaction_level','satisfaction_level':'total_2'})

bd.head()
# 公司所有人员的满意度分布

sns.barplot(x=bd['satisfaction_level'],y=bd['total_2'])

fig1 = plt.gcf()

fig1.set_size_inches(20,6)

plt.title("Overall staff satisfaction level distribution")
# 公司所有人员的满意度分布

sns.jointplot(x=bd["satisfaction_level"], y=bd["total_2"], kind='scatter')

fig1.set_size_inches(20,20)

plt.title("Overall staff satisfaction level distribution",loc='left')
# 公司所有人员的满意度分布

sns.jointplot(x=bd["satisfaction_level"], y=bd["total_2"], kind='reg')

fig1.set_size_inches(20,20)

plt.title("Overall staff satisfaction level distribution",loc='left')
# 公司所有人员的月工作时长分布

ii = df['average_montly_hours']

jj = ii.value_counts()

cd = pd.DataFrame(jj)

cd.head()
# 公司所有人员的月工作时长分布

cd = cd.reset_index()

cd.head()
# 公司所有人员的月工作时长分布

cd = cd.rename(columns={'index':'average_montly_hours','average_montly_hours':'total_3'})

cd.head()
# 公司所有人员的月工作时长分布

sns.barplot(x=cd['average_montly_hours'],y=cd['total_3'])

fig1 = plt.gcf()

fig1.set_size_inches(20,6)

plt.title("Overall staff monthly working hours distribution")
# 公司所有人员的月工作时长分布

sns.jointplot(x=cd["average_montly_hours"], y=cd["total_3"], kind='scatter')

fig1.set_size_inches(20,20)

plt.title("Overall staff monthly working hours distribution",loc='left')
# 公司所有人员的月工作时长分布

sns.jointplot(x=cd["average_montly_hours"], y=cd["total_3"], kind='reg')

fig1.set_size_inches(20,20)

plt.title("Overall staff monthly working hours distribution",loc='left')
# 公司所有人员的升职分布

kk = df['promotion_last_5years']

mm = kk.value_counts()

dd = pd.DataFrame(mm)

dd.head()
# 公司所有人员的升职分布

dd = dd.reset_index()

dd.head()
dd = dd.rename(columns={'index':'promotion_last_5years','promotion_last_5years':'total_4'})

dd
dd['promotion_last_5years'].replace({0:'No Promotion',1:'Promotion'},inplace=True)

dd
# 公司所有人员的升职分布

x7=dd['promotion_last_5years']

y7=dd['total_4']

plt.barh(x7,height=0.6,width=y7,align='center')

plt.title('Overall staff promotion distribution',loc='center')

for A6,B6 in zip(x7,y7):

    plt.text(B6,A6,B6,ha='left',va='center',fontsize=12)

plt.grid(False)

fig1 = plt.gcf()

fig1.set_size_inches(15,6)
# 公司所有人员的升职分布

sns.barplot(x='promotion_last_5years',y='total_4',data=dd,palette='Blues_d').set_title('Overall promotion distribution ')

fig1 = plt.gcf()

fig1.set_size_inches(5,5)
# 公司所有人员的工作年份分布

nn = df['time_spend_company']

pp = nn.value_counts()

ee = pd.DataFrame(pp)

ee
# 公司所有人员的工作年份分布

ee = ee.reset_index()

ee
# 生成新列表: 公司所有人员的工作年份分布

ee = ee.rename(columns={'index':'time_spend_company','time_spend_company':'total_5'})

ee
ee['time_spend_company'].replace({3:'3 Service Years',4:'4 Service Years',5:'5 Service Years',6:'6 Service Years',2:'2 Service Years',10:'10 Service Years',8:'8 Service Years',7:'7 Service Years'},inplace=True)

ee
# 生成新列表: 公司所有人员的工作年份分布



sns.barplot(x=ee['time_spend_company'],y=ee['total_5'],palette='Blues_d')

fig1 = plt.gcf()

fig1.set_size_inches(15,6)

plt.title("Overall staff serviced years distribution")
# 公司所有人员的工作年份分布

x8=ee['time_spend_company']

y8=ee['total_5']

plt.barh(x8,height=0.6,width=y8,align='center')

plt.title('Overall staff promotion distribution',loc='center')

for A7,B7 in zip(x8,y8):

    plt.text(B7,A7,B7,ha='left',va='center',fontsize=12)

plt.grid(False)

fig1 = plt.gcf()

fig1.set_size_inches(15,6)