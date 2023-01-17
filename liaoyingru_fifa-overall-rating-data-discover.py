#导入所需要用到的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()
#读取数据
df = pd.read_csv('../input/soccer_player.csv')
df.head()
#检查重复行
sum(df.duplicated())
#检查缺失值和数据类型
df.info()
#丢弃重复行
df.drop_duplicates(inplace=True)
#再次检查重复行
sum(df.duplicated())
#丢弃存在缺失值的数据行
df = df.dropna()
#调整 birthday, date 的数据类型
df['birthday'] = pd.to_datetime(df['birthday'])
df['date'] = pd.to_datetime(df['date'])
df.info()
#增加 age 列
df['age'] = df['date'].dt.year - df['birthday'].dt.year
#增加 bmi 列
df['bmi'] = (df['weight'] * 0.45) / ((df['height'] * 0.01) * df['height'] * 0.01)
df['bmi'] = df['bmi'].round(0)
#按年龄分组并计算平均综合得分
age_orating = df.groupby('age').mean()['overall_rating']
#作图
age_orating.plot(kind='bar', figsize=(10, 5))
plt.title('Age vs Average overall_rating', fontsize=15)
plt.xlabel('age', fontsize=15)
plt.ylabel('average overall_rating', fontsize=15);
#根据球员的BMI值和综合得分作图
df.boxplot(column='overall_rating', by='bmi', figsize=(10, 5))
plt.ylabel('overall_rating', fontsize=15)
plt.xlabel('BMI', fontsize=15)
plt.title('BMI vs overall_rating', fontsize=15)
plt.suptitle('');
#根据球员的年龄、潜力得分和综合得分作图

x = df['overall_rating']
y = df['potential']
c = df['age']

plt.figure(figsize=(8, 8))
plt.hexbin(x, y, C=c, cmap='BuGn', gridsize=45)
plt.xticks(range(30, 110, 10))                                                                                                                        
plt.yticks(range(30, 110, 10))
plt.xlabel('overall_rating', fontsize=15)
plt.ylabel('potential', fontsize=15)
plt.colorbar().set_label('age', fontsize=15)
plt.title('Overall_rating vs Potential and Age shown in different colors', fontsize=15);
