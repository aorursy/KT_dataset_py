import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from sklearn.ensemble import RandomForestClassifier #for the model

from sklearn.tree import export_graphviz #plot tree

from sklearn.metrics import roc_curve, auc #for model evaluation

from sklearn.metrics import classification_report #for model evaluation

from sklearn.metrics import confusion_matrix #for model evaluation

from sklearn.model_selection import train_test_split #for data splitting

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split





sns.set(style='ticks')

plt.rcParams['font.family']='Arial Unicode MS'

plt.rcParams['axes.unicode_minus']=False

warnings.filterwarnings('ignore')
data=pd.read_csv('../input/heart-disease-uci/heart.csv')  #读取数据

print('行:',data.shape[0],',  列:',data.shape[1]) # 看多少行多少列

data.head(5)
data.describe()
countNoDisease = len(data[data.target == 0])

countHaveDisease = len(data[data.target == 1])

countfemale = len(data[data.sex == 0])

countmale = len(data[data.sex == 1])

print(f'没患病人数:{countNoDisease }',end=' ,')

print("没有得心脏病比率: {:.2f}%".format((countNoDisease / (len(data.target))*100)))

print(f'有患病人数:{countHaveDisease }',end=' ,')

print("患有心脏病比率: {:.2f}%".format((countHaveDisease / (len(data.target))*100)))

print(f'女性人数:{countfemale }',end=' ,')

print("女性比例: {:.2f}%".format((countfemale / (len(data.sex))*100)))

print(f'男性人数:{countmale }',end=' ,')

print("男性比例: {:.2f}%".format((countmale   / (len(data.sex))*100)))
# 画个饼图

labels = '男子','女子'

sizes = [countmale,countfemale]



# Plot

plt.figure(figsize=(6,6))

plt.pie(sizes, explode=(0, 0.1), labels=labels, colors=sns.color_palette("Blues"),

autopct='%1.1f%%', shadow=True, startangle=90)

plt.title('Pie Chart Ratio for Sex Distribution\n', fontsize=16)

sns.set_context("paper", font_scale=1.2)
fig, ax =plt.subplots(1,3)  #2个子区域

fig.set_size_inches(w=15,h=5)   # 设置画布大小

sns.countplot(x="sex", data=data,ax=ax[0])

plt.xlabel("性别 (0 = female, 1= male)")

sns.countplot(x="target", data=data,ax=ax[1])

plt.xlabel("是否患病 (0 = 未患病, 1= 患病)")

sns.swarmplot(x='sex',y='age',hue='target',data=data,ax=ax[2])

plt.xlabel("性别 (0 = female, 1= male)")

plt.show()
group2 = data.groupby(['sex','target'])

group2.count()
pd.crosstab(data.sex,data.target).plot(kind="bar",figsize=(15,6),color=['#30A9DE','#EFDC05' ])

plt.title('各性别下患病图示')

plt.xlabel('性别 (0 = 女性, 1 = 男性)')

plt.xticks(rotation=0)

plt.legend(["未患病", "患有心脏病"])

plt.ylabel('人数')

plt.show()
pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(25,8))

plt.title('患病变化随年龄分布图')

plt.xlabel('岁数')

plt.ylabel('比率')

plt.savefig('heartDiseaseAndAges.png')

plt.show()
plt.scatter(x=data.age[data.target==1], y=data.thalach[(data.target==1)], c="red")

plt.scatter(x=data.age[data.target==0], y=data.thalach[(data.target==0)], c='#41D3BD')

plt.legend(["患病", "未患病"])

plt.xlabel("年龄")

plt.ylabel("最大心率")

plt.show()
sns.violinplot(x=data.target,y=data.trestbps,data=data)
plt.scatter(x=data.age[data.target==1], y=data.trestbps[data.target==1], c="#FFA773")

plt.scatter(x=data.age[data.target==0], y=data.trestbps[data.target==0], c="#8DE0FF")

plt.legend(["患病",'未患病'])

plt.xlabel("年龄")

plt.ylabel("血压")

plt.show()
plt.scatter(x=data.thalach[data.target==1], y=data.trestbps[data.target==1], c="#FFA773")

plt.scatter(x=data.thalach[data.target==0], y=data.trestbps[data.target==0], c="#8DE0FF")

plt.legend(["患病",'未患病'])

plt.xlabel("心率")

plt.ylabel("血压")

plt.show()
plt.scatter(x=data.age[data.target==1], y=data.chol[data.target==1], c="orange")

plt.scatter(x=data.age[data.target==0], y=data.chol[data.target==0], c="green")

plt.legend(["患病",'未患病'])

plt.xlabel("年龄")

plt.ylabel("胆固醇")

plt.show()
sns.boxplot(x=data.target,y=data.chol,data=data)
sns.swarmplot(x='target',y='trestbps',hue='cp',data=data, size=6)

plt.xlabel('是否患病')

plt.show()
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.countplot(x='cp',data=data,hue='target',palette='Set3',ax=ax[0])

ax[0].set_xlabel("胸痛类型")

data.cp.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',explode=[0.01,0.01,0.01,0.01],shadow=True, cmap='Blues')

ax[1].set_title("胸痛类型")
plt.figure(figsize=(15,5))

sns.swarmplot(y='trestbps',data=data,x='ca',hue='target',palette='RdBu_r',size=7)

plt.xlabel('大血管数量')

plt.ylabel('静息血压')

plt.show()
plt.figure(figsize=(15,5))

sns.catplot(x="ca", y="age", hue="target", kind="swarm", data=data, palette='RdBu_r')

plt.xlabel('大血管显色数量')

plt.ylabel('年龄')
sns.boxplot(x="sex", y="age", hue="cp", data=data, palette="Paired")

plt.title("0: 女性, 1:男性",color="gray")

plt.legend

plt.ylabel('年龄')

plt.xlabel('性别')

plt.show()
plt.figure(figsize=(15,10))

ax= sns.heatmap(data.corr(),cmap=plt.cm.RdYlGn , annot=True ,fmt='.2f')

a,b =ax.get_ylim()

ax.set_ylim(a+0.5,b-0.5)