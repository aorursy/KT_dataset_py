import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

import seaborn as sns
pd.set_option('display.max_columns', 100)  # 设置显示数据的最大列数，防止出现省略号…，导致数据显示不全

pd.set_option('expand_frame_repr', False)  # 当列太多时不自动换行

%matplotlib inline

font = FontProperties(fname="../input/msyh.ttf", size=14) # 指定文字,用来正常显示中文标签

sns.set_style('darkgrid')
red_df = pd.read_csv('../input/winequality-red.csv',sep = ';')

white_df = pd.read_csv('../input/winequality-white.csv',sep = ';')
red_df.head()
white_df.head()
red_df.rename(columns= {'fixed acidity':'fixed_acidity'} , inplace = True)

red_df.rename(columns= {'volatile acidity':'volatile_acidity'} , inplace = True)

red_df.rename(columns= {'citric acid':'citric_acid'} , inplace = True)

red_df.rename(columns= {'residual sugar':'residual_sugar'} , inplace = True)

red_df.rename(columns= {'total sulfur dioxide':'total_sulfur_dioxide'} , inplace = True)

red_df.rename(columns= {'free sulfur dioxide':'free_sulfur_dioxide'} , inplace = True)

red_df.head()
white_df.rename(columns= {'fixed acidity':'fixed_acidity'} , inplace = True)

white_df.rename(columns= {'volatile acidity':'volatile_acidity'} , inplace = True)

white_df.rename(columns= {'citric acid':'citric_acid'} , inplace = True)

white_df.rename(columns= {'residual sugar':'residual_sugar'} , inplace = True)

white_df.rename(columns= {'total sulfur dioxide':'total_sulfur_dioxide'} , inplace = True)

white_df.rename(columns= {'free sulfur dioxide':'free_sulfur_dioxide'} , inplace = True)

# white_df.to_csv('winequality-white.csv', index=False)

white_df.head()
red_df.info()
white_df.info()
#红白葡萄酒重复行

white_df.duplicated().sum(), red_df.duplicated().sum()
#删除重复数据

red_df.drop_duplicates(inplace = True)

white_df.drop_duplicates(inplace = True)

#检查是否有空值

red_df.isnull().sum().sum(), white_df.isnull().sum().sum()
#红葡萄酒质量唯一值

red_df.quality.unique()
# 唯一值计数

red_df.quality.value_counts()
# 红葡萄酒的平均密度

red_df[["density"]].mean()
# 白葡萄酒的平均密度

white_df[["density"]].mean()
red_df.shape, white_df.shape
# 为红葡萄酒数据框创建颜色数组

color_red = np.repeat('red',red_df.shape[0])

red_df['color'] = color_red

red_df.head()
# 为白葡萄酒数据框创建颜色数组

color_white = np.repeat('white',white_df.shape[0])

white_df['color'] = color_white

white_df.head()
wine_df = red_df.append(white_df)

wine_df.to_csv('winequality.csv', index=False)

wine_df.shape
df = pd.read_csv('winequality.csv')

df.head()
df.hist();
pd.plotting.scatter_matrix(df);
# 用 groupby 计算每个酒类型（红葡萄酒和白葡萄酒）的平均质量

df.groupby('color').mean()
# 用 Pandas 描述功能查看最小、25%、50%、75% 和 最大 pH 值

df['pH'].describe()
# 对用于把数据“分割”成组的边缘进行分组

bin_edges = [2.72 ,3.11 ,3.21 ,3.32 ,4.01 ] # 用刚才计算的五个值填充这个列表
# 四个酸度水平组的标签

bin_names = ['high' ,'mid-high' ,'mid' ,'low' ] # 对每个酸度水平类别进行命名
# 创建 acidity_levels 列

df['acidity_levels'] = pd.cut(df['pH'], bin_edges, labels=bin_names)



# 检查该列是否成功创建

df.head()
# 用 groupby 计算每个酸度水平的平均质量

df.groupby('acidity_levels').mean()
# 获取酒精含量的中位数

df.describe()
# 选择酒精含量小于中位数的样本

low_alcohol =df.query('alcohol<10.3')



# 选择酒精含量大于等于中位数的样本

high_alcohol =df.query('alcohol>=10.3')



# 确保这些查询中的每个样本只出现一次

num_samples = df.shape[0]

num_samples == low_alcohol['quality'].count() + high_alcohol['quality'].count() # 应为True
# 获取低酒精含量组和高酒精含量组的平均质量评分

low_alcohol['quality'].mean(), high_alcohol['quality'].mean()
# 获取残留糖分的中位数

df['residual_sugar'].median()
# 选择残留糖分小于中位数的样本

low_sugar =df.query('residual_sugar<=3.0')



# 选择残留糖分大于等于中位数的样本

high_sugar =df.query('residual_sugar>3.0')



# 确保这些查询中的每个样本只出现一次

num_samples == low_sugar['quality'].count() + high_sugar['quality'].count() # 应为True
# 获取低糖分组和高糖分组的平均质量评分

low_sugar['quality'].mean(), high_sugar['quality'].mean()
# 用查询功能选择每个组，并获取其平均质量

median = df['alcohol'].median()

low = df.query('alcohol < {}'.format(median))

high = df.query('alcohol >= {}'.format(median))



mean_quality_low = low['quality'].mean()

mean_quality_high = high['quality'].mean()

print("mean_quality_high:", mean_quality_high)

print("mean_quality_low:", mean_quality_low)
# 用合适的标签创建柱状图

locations = [1, 2]

heights = [mean_quality_low, mean_quality_high]

labels = ['Low', 'High']

plt.bar(locations, heights, tick_label=labels)

plt.title('Average Quality Ratings by Alcohol Content\n酒精含量的平均质量等级', FontProperties=font)

plt.xlabel('Alcohol Content \n酒精含量', FontProperties=font)

plt.ylabel('Average Quality Rating \n平均质量等级', FontProperties=font);
# 用查询功能选择每个组，并获取其平均质量

median = df['residual_sugar'].median()

low = df.query('residual_sugar < {}'.format(median))

high = df.query('residual_sugar >= {}'.format(median))



mean_quality_low = low['quality'].mean()

mean_quality_high = high['quality'].mean()

print("mean_quality_high:", mean_quality_high)

print("mean_quality_low:", mean_quality_low)
# 用合适的标签创建柱状图

locations = [1, 2]

heights = [mean_quality_low, mean_quality_high]

labels = ['Low', 'High']

plt.bar(locations, heights, tick_label=labels)

plt.title('Average Quality Ratings by Residual Sugar Content \n残糖量的平均质量等级', FontProperties=font)

plt.xlabel('Residual Sugar Content \n残糖量', FontProperties=font)

plt.ylabel('Average Quality Rating \n平均质量等级', FontProperties=font);
# 使用分组功能获取每个酸度水平的平均质量

# acidity_levels

# low mid mid-high high

low = df.loc[df['acidity_levels'] == 'low']

mid = df.loc[df['acidity_levels'] == 'mid']

mid_high = df.loc[df['acidity_levels'] == 'mid-high']

high = df.loc[df['acidity_levels'] == 'high']

ph_low = low['pH'].mean()

ph_mid = mid['pH'].mean()

ph_mid_high = mid_high['pH'].mean()

ph_high = high['pH'].mean()

print(ph_low, ph_mid, ph_mid_high, ph_high)
# 用合适的标签创建柱状图



locations = [1, 2, 3, 4]

heights = [ph_low, ph_mid, ph_mid_high, ph_high]

labels = ['Low', 'Mid', 'Mid-High', 'High']

plt.bar(locations, heights, tick_label=labels)

plt.title('Average Quality Ratings by pH Content \n按酸碱度含量划分的平均质量等级', FontProperties=font)

plt.xlabel('pH Content \npH含量', FontProperties=font)

plt.ylabel('Average Quality Rating \n平均质量等级', FontProperties=font);
# 获取每个等级和颜色的数量

color_counts = df.groupby(['color', 'quality']).count()['pH']

color_counts
# 获取每个颜色的总数

color_totals = df.groupby('color').count()['pH']

color_totals
# 将红葡萄酒等级数量除以红葡萄酒样本总数，获取比例

red_proportions = color_counts['red'] / color_totals['red']

red_proportions
# 将白葡萄酒等级数量除以白葡萄酒样本总数，获取比例

white_proportions = color_counts['white'] / color_totals['white']

white_proportions
red_proportions['9'] = 0

red_proportions
ind = np.arange(len(red_proportions))  # 组的 x 坐标位置

width = 0.35       # 条柱的宽度

b = ind + width

(ind, b)
# 绘制条柱

red_bars = plt.bar(ind, red_proportions, width, color='r', alpha=.7, label='Red Wine')

white_bars = plt.bar(b, white_proportions, width, color='w', alpha=.7, label='White Wine')



# 标题和标签

plt.ylabel('Proportion\n比例', FontProperties=font)

plt.xlabel('Quality\n质量', FontProperties=font)

plt.title('Proportion by Wine Color and Quality \n葡萄酒颜色和质量的比例', FontProperties=font)

locations = ind + width / 2  # x 坐标刻度位置

labels = ['3', '4', '5', '6', '7', '8', '9']  # x 坐标刻度标签

plt.xticks(locations, labels)

# 图例

plt.legend();