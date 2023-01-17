import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# 读取训练数据
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
# import Libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
# 数据列
train_data.columns
# 显示前5行
train_data.head(5)
# 统计数字类型列的概要信息
# 平均生存率 38.4% 平均年龄 29.7岁
train_data.describe()
# 统计非数字型列的概要信息
# 男性占比高 577/891
train_data.describe(include=[np.object])
# 用饼图显示男女比例
df = train_data['Sex'].value_counts()
plt.pie(df, labels=df.index, autopct='%.1f%%')
plt.axis('equal')
plt.show()
# 缺失数据分析
# 年龄字段缺失约 20%
total = train_data.isnull().sum().sort_values(ascending = False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total,percent], axis = 1, keys = ['Total', 'Percent'])
missing_data.head(5)
# 相关性分析，热力图展示
# 从图中看不出明显的相关性来
corrmat = train_data.corr()
f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(corrmat,cmap = "Blues",vmax = 0.8, square = True)
plt.show()
# 相关性量化分析
# 有点失望，相关性最大的 Fare 也才 25.7%
# 年龄等还存在负相关性，是否意味着年轻反而死亡率高？
correlations = train_data.corr()
correlations = correlations["Survived"].sort_values(ascending=False)
features = correlations.index[1:6]
correlations.head(10)
# 死亡相关性分析
k = 5
cols = corrmat.nlargest(k , 'Survived')['Survived'].index
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale = 1.00)
hm = sns.clustermap(cm , cmap = "Blues",cbar = True,square = True, 
                    yticklabels = cols.values, xticklabels = cols.values)
plt.show()
# 年龄/仓位 分析，可以看出年轻人集中在3等舱
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train_data, palette='viridis')
plt.show()
full_data = pd.concat([train_data, test_data], ignore_index=True)
# 空值统计
full_data.isnull().sum()
# 头衔处理
full_data['Title'] = full_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0])
full_data['Title'].value_counts()
# 船票空值查询
full_data[full_data['Fare'].isnull()]
# 船票空值处理
fill_Fare = full_data[full_data['Pclass']==3]['Fare'].median()
full_data['Fare'].fillna(fill_Fare, inplace=True)
# 确认修改成功
# full_data.iloc[1043]['Fare']