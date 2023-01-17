import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
"question 4"

data_path='/kaggle/input/fifadata/data/fifa_countries_audience.csv'

df=pd.read_csv(data_path)

df.head()
for confederation in list(df['confederation'].unique()):

    try:

        f= plt.figure(figsize=(12,5))

        ax=f.add_subplot(122)



        sns.distplot(df[(df.confederation == confederation)]['tv_audience_share'],color='b',ax=ax)

        ax.set_title(confederation+' Distribution of tv_audience_share')

        plt.show()

    except:

        pass
df=df.sort_values(by='gdp_weighted_share').reset_index(drop=True)

df[0:20]


f, ax = plt.subplots(figsize=(12, 12))

data=df[['population_share','tv_audience_share','gdp_weighted_share']]

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(120,10,as_cmap=True),

            square=True, ax=ax)

plt.show()
ax = sns.scatterplot(x='gdp_weighted_share',y='tv_audience_share',data=df,palette='magma',hue='confederation')

ax.set_title('Scatter plot of charges and bmi')



sns.lmplot(x="gdp_weighted_share", y="tv_audience_share", hue="confederation", data=df, palette = 'magma', size = 8)
df.rename(columns={'tv_audience_share':'y','gdp_weighted_share':'x'},inplace=True) 

df=df[['x','y']]

df.head()


corr=df.corr()

corr
x=df['x'].values

y=df['y'].values

#线性回归拟合

x_m=np.mean(x)

y_m=np.mean(y)

x1=(x-x_m)

y1=y-y_m

x2=sum((x-x_m)**2)

xy=sum(x1*y1)

beta1=xy/x2

beta0=y_m-beta1*x_m

print('y=',beta0,'+',beta1,'*x')

a=np.linspace(1000,5000,1000)#b表示在(1000,5000)上生成1000个a值

b=[beta0+beta1*i for i in a] 

plt.plot(a,b,'r')

plt.show()
from scipy import stats

#方差

sigma2=sum((y-beta0-beta1*x)**2)/(18)

print("r2:",sigma2)

#标准差

sigma=np.sqrt(sigma2)

#求t值

t=beta1*np.sqrt(x2)/sigma

print('t=',t)

#已知临界值求p值

p=stats.t.sf(t,18)

print('p=',p)

 

#输出检验结果

if p<0.05:

    print ('the linear regression between x and y is significant')

else:

	print('the linear regression between x and y is not significant')

plt.figure(figsize=(6, 6))

sns.set_style("darkgrid") 

a=np.linspace(0,15,3)

b=[beta0+beta1*i for i in a] 

sns.lineplot(a, b, color="r", lw=1)

sns.scatterplot(x=x, y=y,color='b')

plt.xlabel('x')

plt.ylabel('y')

plt.tight_layout()

plt.show()