import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error as mae
data=pd.read_csv('../input/data-acquisitioncleaning/cleaned_data.csv')

data=data.set_index(['Unnamed: 0']) #DataFrame在存储为csv file以后原来的index会变为一个列，因此要重新设置index
data.shape
pie_plt=data.groupby(['brand']).sum()['comments'].sort_values(ascending=False)#统计每个品牌评论总数，以此作为我们对销量的估计
pie_plt
#绘制各个手机品牌估计销量的占比扇形图

fig,axes=plt.subplots(figsize=(12,12))

comment_sum=pie_plt.values.sum()

percen=[np.round(each/comment_sum*100,2) for each in pie_plt.values]

axes.pie(pie_plt.values,labels=pie_plt.index,labeldistance=1.2,autopct = '%3.1f%%')

axes.legend([pie_plt.index[i]+': '+str(percen[i])+"%" for i in range(len(percen))],loc='upper right',bbox_to_anchor=(1, 0, 1, 1))

axes.set_title('Estimated Handphone Market Share in China')

plt.show()
data[(data['brand']=='NOKIA')|(data['brand']=='Philips')]['price'].median()#诺基亚和飞利浦手机价格中位数
data[(data['brand']=='NOKIA')|(data['brand']=='Philips')]['price'].mean()#诺基亚和飞利浦手机价格平均数
correlation=data[(data['brand']!='Apple')&(data['price']!=9999)].corr() 
#绘制相应correlation matrix的heatmap

fig,axes=plt.subplots(figsize=(8,8))

cax=sns.heatmap(correlation,vmin=-0.25, vmax=1,square=True,annot=True)

axes.set_xticklabels(['RAM', 'ROM', 'battery', 'comments', 'price', 'rear camera',

       'resolution', 'screen size', 'weight'])

axes.set_yticklabels(['RAM', 'ROM', 'battery', 'comments', 'price', 'rear camera',

       'resolution', 'screen size', 'weight'])

axes.set_title('Heatmap of Correlation Matrix of numerical data')

plt.show()
data.groupby(['brand']).median()['price'].sort_values(ascending=False).values.std() #计算不同品牌价格中位数集合的标准差
bar_plt=data.groupby(['brand']).median()['price']



fig,axes=plt.subplots(figsize=(20,8))

axes.bar(bar_plt.index,bar_plt.values)

axes.set_title('Median price of handphones of various brands')
data.groupby(['screen material']).median()['price'].sort_values(ascending=False).values.std() #计算不同屏幕材料价格中位数集合的标准差
data.groupby(['screen material']).median()['price'].sort_values(ascending=False)
bar_plt2=data.groupby(['screen material']).median()['price']



fig,axes=plt.subplots(figsize=(18,8))

axes.bar(bar_plt2.index,bar_plt2.values)

axes.set_title('Median price of handphones of various screen materials')
data[(data['brand']=='NOKIA')|(data['brand']=='Philips')]['screen material'].value_counts()
#绘制屏幕材料为IPS或TFT手机的价格分布图

hist_plot=data[(data['screen material']=='IPS')|(data['screen material']=='TFT')]['price']#查看所有屏幕材料为IPS或TFT手机的价格

sns.distplot(hist_plot)

plt.title('Price Distribution Plot of Handphones Whose Screen Material is TFT or IPS ')
data.dropna(subset=['ROM','RAM','brand','price']).shape[0]/data.shape[0]
data.isnull().sum().sort_values(ascending=False)
df=data.loc[:,['price','rear camera','brand','weight']].dropna()
to_model=pd.get_dummies(df)#对非数值型数据进行独热编码
x=to_model.iloc[:,1:].values

y=to_model.iloc[:,0].values
model=DecisionTreeRegressor()

model.fit(x,y)
error_list=[]

for each in df['brand'].value_counts().index:   

    to_fill='brand_{}'.format(each)

    x_data=to_model[to_model[to_fill]==1].iloc[:,1:].values

    y_data=to_model[to_model[to_fill]==1].iloc[:,0].values

    

    

    test_result=model.predict(x_data)

    merror=mae(y_data.reshape(len(y_data),1),test_result.flatten())

    error=(np.abs(test_result-y_data)/y_data).mean()

    print(each,end=' : ') 

    print(np.round(merror,2),end=', ')

    print(str(np.round(error*100,3))+'%')

    error_list.append([each,merror,error])
error_df=pd.DataFrame(error_list,columns=['brand','mean_absolute_error','mean_proportional_error'])
error_df