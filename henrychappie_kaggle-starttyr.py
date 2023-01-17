#系统初始化代码

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))

#使用pandas导入数据

import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as pl

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('../input/insurance.csv')
data.head()
#统计数据集中属性为空的记录的数量

data.isnull().sum()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

#性别

le.fit(data.sex.drop_duplicates()) 

data.sex = le.transform(data.sex)

# 是否是烟民

le.fit(data.smoker.drop_duplicates()) 

data.smoker = le.transform(data.smoker)

#地区

le.fit(data.region.drop_duplicates()) 

data.region = le.transform(data.region)
data.corr()['charges'].sort_values()
#绘制热度图，主要作用是查看特征之间的相关性

f, ax = pl.subplots(figsize=(10, 8))

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240,10,as_cmap=True),

            square=True, ax=ax)
from bokeh.io import output_notebook, show

from bokeh.plotting import figure

output_notebook()

import scipy.special

from bokeh.layouts import gridplot

from bokeh.plotting import figure, show, output_file

p = figure(title="Distribution of charges",tools="save",

            background_fill_color="#E8DDCB")

hist, edges = np.histogram(data.charges)

p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],

        fill_color="#036564", line_color="#033649")

p.xaxis.axis_label = 'x'

p.yaxis.axis_label = 'Pr(x)'
f= pl.figure(figsize=(12,5))



#烟民的医疗费用分布图

ax=f.add_subplot(121)

sns.distplot(data[(data.smoker == 1)]["charges"],color='c',ax=ax)

ax.set_title('Distribution of charges for smokers')



#非烟民的医疗费用分布图

ax=f.add_subplot(122)

sns.distplot(data[(data.smoker == 0)]['charges'],color='b',ax=ax)

ax.set_title('Distribution of charges for non-smokers')
#统计数据中烟民与非烟民的性别比例条形图

sns.catplot(x="smoker", kind="count",hue = 'sex', palette="pink", data=data)
#绘制violin图

sns.catplot(x="sex", y="charges", hue="smoker",

            kind="violin", data=data, palette = 'magma')
#绘制女性中烟民与非烟民的费用箱型图

pl.figure(figsize=(12,5))

pl.title("Box plot for charges of women")

sns.boxplot(y="smoker", x="charges", data =  data[(data.sex == 1)] , orient="h", palette = 'magma')
#绘制男性中烟民与非烟民的费用箱型图

pl.figure(figsize=(12,5))

pl.title("Box plot for charges of men")

sns.boxplot(y="smoker", x="charges", data =  data[(data.sex == 0)] , orient="h", palette = 'rainbow')
#绘制年龄分布图

pl.figure(figsize=(12,5))

pl.title("Distribution of age")

ax = sns.distplot(data["age"], color = 'g')
#对18岁的数据对象进行是否是烟民的统计

sns.catplot(x="smoker", kind="count",hue = 'sex', palette="rainbow", data=data[(data.age == 18)])

pl.title("The number of smokers and non-smokers (18 years old)")
#绘制箱型图

pl.figure(figsize=(12,5))

pl.title("Box plot for charges 18 years old smokers")

sns.boxplot(y="smoker", x="charges", data = data[(data.age == 18)] , orient="h", palette = 'pink')
#对非烟民的费用与年龄进行统计

g = sns.jointplot(x="age", y="charges", data = data[(data.smoker == 0)],kind="kde", color="m")

g.plot_joint(pl.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$X$", "$Y$")

ax.set_title('Distribution of charges and age for non-smokers')
#对烟民的费用和年龄进行统计

g = sns.jointplot(x="age", y="charges", data = data[(data.smoker == 1)],kind="kde", color="c")

g.plot_joint(pl.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$X$", "$Y$")

ax.set_title('Distribution of charges and age for smokers')
#绘制对非烟民的费用与年龄的圆点图

p = figure(plot_width=500, plot_height=450)

p.circle(x=data[(data.smoker == 0)].age,y=data[(data.smoker == 0)].charges, size=7, line_color="navy", fill_color="pink", fill_alpha=0.9)



show(p)
#绘制对烟民的费用与年龄的圆点图

p = figure(plot_width=500, plot_height=450)

p.circle(x=data[(data.smoker == 1)].age,y=data[(data.smoker == 1)].charges, size=7, line_color="navy", fill_color="red", fill_alpha=0.9)

show(p)
#绘制年龄与费用对是否是烟民的圆点图

sns.lmplot(x="age", y="charges", hue="smoker", data=data, palette = 'inferno_r', size = 7)

ax.set_title('Smokers and non-smokers')
#绘制体重指数的分布图

pl.figure(figsize=(12,5))

pl.title("Distribution of bmi")

ax = sns.distplot(data["bmi"], color = 'm')
#绘制对体重指数大于等于30的人的费用分布图

pl.figure(figsize=(12,5))

pl.title("Distribution of charges for patients with BMI greater than 30")

ax = sns.distplot(data[(data.bmi >= 30)]['charges'], color = 'm')
#绘制对体重指数小于三十的人的费用分布图

pl.figure(figsize=(12,5))

pl.title("Distribution of charges for patients with BMI less than 30")

ax = sns.distplot(data[(data.bmi < 30)]['charges'], color = 'b')
#绘制体重指数与年龄的分布图

g = sns.jointplot(x="bmi", y="charges", data = data,kind="kde", color="r")

g.plot_joint(pl.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$X$", "$Y$")

ax.set_title('Distribution of bmi and charges')
#统计是否是烟民这一属性中所有人的费用与体重指数的散点图

pl.figure(figsize=(10,6))

ax = sns.scatterplot(x='bmi',y='charges',data=data,palette='magma',hue='smoker')

ax.set_title('Scatter plot of charges and bmi')



sns.lmplot(x="bmi", y="charges", hue="smoker", data=data, palette = 'magma', size = 8)
#统计孩子的数量

sns.catplot(x="children", kind="count", palette="ch:.25", data=data, size = 6)
#对有孩子的人统计烟民与非烟民的数量

sns.catplot(x="smoker", kind="count", palette="rainbow",hue = "sex",

            data=data[(data.children > 0)], size = 6)

ax.set_title('Smokers and non-smokers who have childrens')
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.ensemble import RandomForestRegressor
#删除属性为费用的这一列作为训练数据

x = data.drop(['charges'], axis = 1)

#取出数据为费用的这一列作为标签

y = data.charges



#切分数据

x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0)

#使用线性回归训练数据

lr = LinearRegression().fit(x_train,y_train)

#训练好模型后进行预测

y_train_pred = lr.predict(x_train)

y_test_pred = lr.predict(x_test)

#打印出正确率

print(lr.score(x_test,y_test))
#将费用和地区删除后的数据作为训练数据

X = data.drop(['charges','region'], axis = 1)

#测试数据同样是将费用作为标签

Y = data.charges





#构建特征，设置多项式的度为2

quad = PolynomialFeatures (degree = 2)

x_quad = quad.fit_transform(X)



#切分数据集

X_train,X_test,Y_train,Y_test = train_test_split(x_quad,Y, random_state = 0)



#使用线性回归训练数据

plr = LinearRegression().fit(X_train,Y_train)



#预测数据

Y_train_pred = plr.predict(X_train)

Y_test_pred = plr.predict(X_test)



#打印正确率

print(plr.score(X_test,Y_test))
#使用随机森林进行回归任务【100棵树，使用均方误差作为评估】

forest = RandomForestRegressor(n_estimators = 100,

                              criterion = 'mse',

                              random_state = 1,

                              n_jobs = -1)

#训练与预测

forest.fit(x_train,y_train)

forest_train_pred = forest.predict(x_train)

forest_test_pred = forest.predict(x_test)



#打印正确率

print('MSE train data: %.3f, MSE test data: %.3f' % (

mean_squared_error(y_train,forest_train_pred),

mean_squared_error(y_test,forest_test_pred)))

print('R2 train data: %.3f, R2 test data: %.3f' % (

r2_score(y_train,forest_train_pred),

r2_score(y_test,forest_test_pred)))
#将上面的结果可视化

pl.figure(figsize=(10,6))



pl.scatter(forest_train_pred,forest_train_pred - y_train,

          c = 'black', marker = 'o', s = 35, alpha = 0.5,

          label = 'Train data')

pl.scatter(forest_test_pred,forest_test_pred - y_test,

          c = 'c', marker = 'o', s = 35, alpha = 0.7,

          label = 'Test data')

pl.xlabel('Predicted values')

pl.ylabel('Tailings')

pl.legend(loc = 'upper left')

pl.hlines(y = 0, xmin = 0, xmax = 60000, lw = 2, color = 'red')

pl.show()