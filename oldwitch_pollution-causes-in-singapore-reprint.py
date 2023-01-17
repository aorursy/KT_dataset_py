#导入分析过程中会用到的库
#pyplot是可以对图像做出一些改变的函数集合，和MATLAB类似,并且可以通过函数调用的方式来保存图像状态
#encoding='gbk'声明中文编码，在分析过程中用到中文字符时使用
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import nan as NaN
from functools import reduce
encoding='gbk'
#先将各个污染物含量随年份变化的数据文件导入，由于源文件的名字命名过于复杂，我将其做了简化命名处理,导入后的文件类型为dataframe
#CSV文件导入用到的函数为：pd.read_csv("文件路径"),比如："D:\koggle_data\pollutant1121\CO.csv"
df1=pd.read_csv("../input/CO.csv")
df2=pd.read_csv("../input/NO2.csv")
df3=pd.read_csv("../input/O3.csv")
df4=pd.read_csv("../input/Pb.csv")
df5=pd.read_csv("../input/SO2.csv")
df6=pd.read_csv("../input/pm10.csv")
df7=pd.read_csv("../input/pm2.5.csv")
#这里我们将各个表格进行合并，得到各个
#原作者是不断使用merge函数，将各个df连续合并，需要6行，我们使用reduce，只需两行就可以
#merge函数：pd.merge(表1,表2,on=合并列；left_on/rigtht_on-列名不一样时分别指定；
#left_index/right_index=True-相当于将索引作为合并列；sort=true/false-合并后的数据是否排序，false为默认；
#suffixes=["_r","_l"]-当两个表有相同的列名的时候分别加上后缀来区分；’indicater=true/名字-表示显示数据来自both/left_only/right_only；
#copy-将数据复制到数据结构中，一般不设；how=inner默认/outer/left/right,表示合并后合并列不一样时保留相同/全部/左边/右边)
#reduce函数：reduce(函数,元素列表，初始值) ，不设置初始值的时候，将元素列表依次进行函数运算，一次运算的结果和下一个元素进行运算，设初始值则以初始值为先
data_frames = [df1,df2,df3,df4,df5,df6,df7]
df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year'],
                                            how='outer'), data_frames)
df_merged

#这里提前做数据准备，后面会多次用到
year=range(2008,2015)#我们要比较的年份统一在2008年到2014年之间
DIMS=(16,8)#我们后面绘制图表的时候会统一图表的大小，所以事先声明，大小为(16,8)
#这里我们想要得到的结果是讲各个空气污染值做归一化处理，以便于在同一数量级进行比较
#dropna(axis=0/1表示去除带有NAN的行默认/列,how="all"表示全部为NAN才去掉)函数
#isin函数生成布尔值，利用布尔值对df进行筛选；df[df['列名'].isin(['值1','值2','值3'...])生成某一列的布尔结果]直接生成筛选后的df表
#isin多条件的筛选：df.列名1.isin([值1，值2，值3...])&df.列名2.isin([值1，值2，值3...])
#drop(行索引/列名,axis=0默认/1,inplace=True表示替换原数据,不需要再另外赋值)用于删除df的行列，默认删除行，指定列及axis=1时删除列
#数据归一：1.min-max标准化：(x-min)/(max-min) 结果位于[0-1]之间 2.Z-score标准化方法：(x-平均值)/标准差 结果符合正态分布,平均值=0,方差=1
df_std=df_merged.dropna().sort_values("year")
year=range(2008,2015)
df_std=df_std[df_std["year"].isin(year)]#生成2008年到2014年的数据表
df_std.drop('year',axis=1,inplace=True)
df_std=(df_std-df_std.mean())/df_std.std()#数据的归一化处理
df_std["year"]=year
df_std
#df_std为各个污染物按年份的数值变化表
#根据企业门户网站的统计，制造业占了新加坡生产总值的20%-25%，并且最大的两个行业是电子光学产品以及化工产品
#我们将各个产业进行统计求和,发现上述两个产业确实位于第一、第三位
#源文件里面只看了电子光学及化工产品的影响，我这里加上排名第二的精炼油产品的分析
#groupby为分类函数，配合运算函数进行分类汇总
#groupby(["列一","列二"...])["列三"].sum/count/mean()...相当于对列一列二进行分类，并对列三运算，列三可以不填，即对其它所有列运算
#sort_values是排序函数：sort_values(列名-即以哪一列的排序为准，axis=0/1默认0纵向排序，ascending=True/False,升序降序)
df_industry=pd.read_csv("../input/Industry_values.csv")
df_industry=df_industry[df_industry["year"].isin(year)]#年份筛选
df_industry.groupby("product_type").sum().sort_values("values",ascending=False)#统计分析
#这里想要得到和上面空气污染值变化表一样类型的表，即产品类型为列名，年份为行名
#stack/unstack(0/1)-为堆叠和不要堆叠，stack()倾向于将表格结构变为双索引结构，unstack则相反，默认将第二层索引变为列名，参数为1时移动第一次索引
#当堆叠表格有多列值时，在前面加列名选定某列，即只能针对一列去堆叠:df[列名].unstack()
#我是先通过将年份和产品类型分类求和，由于只有values可以求和，因此正好过滤了其他数据
#原作者先将年份和类型设置为索引，再针对values列去堆叠展开：df_industry=df_industry.set_index(["year","product_type"])["values"].unstack()
product_type=["Computer, Electronic & Optical Products","Refined Petroleum Products","Chemicals & Chemical Products"]
df_industry=df_industry[df_industry['product_type'].isin(product_type)]
df_industry=df_industry.groupby(["year","product_type"]).sum().unstack()
df_industry
#将values这一层的索引去除
#reset_index为重置索引,df.reset_index表示重置列索引，ser.df.reset_index表示重置行索引，drop=True表示重置的时候去掉原索引
df_industry=df_industry['values'].reset_index(drop=True)
df_industry
#数据归一化并加上年份
df_industry=(df_industry-df_industry.mean())/df_industry.std()
df_industry["year"]=year
df_industry

#将df_std气体变化表和df_industry工业产品产值变化表合并
df_industry2=pd.merge(df_industry,df_std,on=["year"],how="outer")
df_industry2
#pd.corr(),为计算相关系数，表示两列数据的相关性。0.3-0.8，可以认为弱相关。0.3以下，认为没有相关性，0.8以上，认为强相关
#corr的参数为method=pearson计算线性相关系数，连续值；spearman用于计算有序性相关系数，有先后；Kendall用于分类性数据的相关系数
#从下表中我们可以看到化工产品和臭氧的正相关性达0.821，电子产品和一氧化碳的相关性达正0.857，而精炼石油产品和各个气体都没有明显的正相关性
industry_corr=df_industry2.corr(method="spearman")
industry_corr

#因此我们通过图表来看一下化工产品和臭氧、电子产品和一氧化碳的变化关系
#先绘制臭氧随年份变化的图plot(x=x轴列,y=y轴列,kind=线型line/bar/barh/hist...grid=True显示网格,figsize=(长,宽),ax=画在哪张图像里，不指定则新建图像)
#plt.legeng(loc='位置1 位置2',bbox_to_anchor=(作比较点的位置))，作比较点的位置(1,0.5)表示一个图表长，0.5个图表宽的点的位置；
#loc表示作比较点相对于标签的位置-('center left')表示作比较点在标签的中间最左边的位置，注意格式-只有一个引号，且中间为空格
O3_graph=df_std.plot(x='year',y='O3_mean',kind='line',grid=True,figsize=DIMS)
df_industry.plot(x='year',y="Chemicals & Chemical Products",kind='line',grid=True,ax=O3_graph)
plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
plt.show()
#从图表可以看到臭氧和化工产品变化的趋势非常相似，同时相关系数表示两者确实存在较大的相关性，从而从图形和数学角度证明了臭氧的上升和化工产品有关
#同理，我们得到一氧化碳和电子光学产品的变化趋势图，结果和上面一样
CO_graph=df_std.plot(x='year',y='CO_mean',kind='line',grid=True,figsize=DIMS)
df_industry.plot(x='year',y="Computer, Electronic & Optical Products",kind='line',grid=True,ax=CO_graph)
plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
plt.show()
#下面我们来尝试验证假设2.建筑房屋数量增长将导致新加坡的污染增加
#首先导入居民居住用宅的数据,可以看到前面15行的数据
#新加坡的居住用房主要以HDB为主，所以我们只分析HDB，并且选择在建的类型，这里默认在建的建筑才会产生污染
df_hdb=pd.read_csv('../input/Residential_building.csv')
df_hdb.head(15)
#我们对年份和建筑类型以及状态为在建的数据同时做了筛选，得到如下结果
df_hdb=df_hdb[df_hdb['year'].isin(year)&df_hdb['building_type'].isin(['HDB Flats'])&df_hdb['state'].isin(['Under Construction'])]
df_hdb
#再来导入商业用房的数据,观察前面15行的数据，各个年份不同类型的商业建筑及处于同状态的数量不同
df_commercial=pd.read_csv('../input/Commercial_buildings.csv')
df_commercial.head(15)
#将各个建筑类型进行汇总排序，前三分别为：饭店、商超及小型市场，原作者只选择了前两者分析，我把第三者也加上去看一下结果
df_commercial2=df_commercial.groupby(['building_type'])['number'].sum().sort_values(ascending=False)
df_commercial2
#筛选出上述三种目标建筑，并结合年份、状态进行进一步筛选
df_commercial=df_commercial[df_commercial['year'].isin(year)&df_commercial['building_type'].isin(['Shops, Lock-Up Shops and Eating Houses','Emporiums and Supermarkets','Mini-markets'])&df_commercial['state'].isin(['Under Construction'])]
df_commercial
#我们对商业住宅的数据进行了处理，使其成为展示各个类型建筑数量的表格
df_commercial=df_commercial.set_index(['year','building_type'])['number'].unstack()
df_commercial

#将hdp的数量和commercial数量的数据合并成一个表格，用到了dataframe的筛选列、合并表、修改列名
#筛选df中的某几列形成新表：df[列名]；df[[‘列名1’，‘列名2’...]] 分别表示选择一列和多列的情况
#df修改列名：df.rename(columns={'a':'b','c':'d'...},inplace=True)
#使用切片工具选择了df的某几列-iloc是根据行号、列排序来索引；而loc是根据索引名称来索引
#iloc[:,1:5]表示全部行，第2列至第5列；iloc[[1]]表示第行列，注意需两个中括号；iloc[1:5]表示第2行至第5行。列号行号均从0开始计
#loc[[2]]表示行索引为2的行；loc[:,'a':'d']表示列名称为a的列到列名称为c的列，假设列名称分别为abcde
df_building=df_hdb[['year','number']]
df_building=pd.merge(df_building,df_commercial,on='year',how='outer')
df_building.rename(columns={'number':'hdb'},inplace=True)
df_building=df_building.iloc[:,1:5]
df_building
#查看df_building的信息，用data.info(),可以看到hdb这一列的数据不为数值型，可能后续无法运算，所以我们在后面进行转换
df_building.info()
#这里我们利用自定义函数的应用apply对hdb的数据类型进行了转换：df.apply(np.sum/mean/int..,axis=0/1分别表示列默认/行)
#转换好的数据便可进行归一化处理
df_building['hdb']=df_building['hdb'].apply(np.int)
df_building=(df_building-df_building.mean())/df_building.std()
df_building
#建筑数据加上年份后和气体数据合并
df_building['year']=year
df_building2=pd.merge(df_building,df_std,on='year',how='outer')
df_building2
#看一下相关系数，发现四种建筑都和臭氧和大于0.8的相关性
df_building2.corr(method='spearman')
#在绘制这条曲线的时候，填写y轴的时候我没有直接写出列名，而是用了df.columns.values.tolist()函数来获取df的列名列表，并且用remove函数移除了其中的‘year元素’
#tolist为转化为列表函数
building_graph=df_building.plot(x='year',y=df_building.columns.values.tolist().remove('year'),kind='line',grid=True,figsize=DIMS)
df_std.plot(x='year',y='O3_mean',kind='line',ax=building_graph)
plt.legend(loc='center left',bbox_to_anchor=[1,0.5])
plt.show()
#我们开始处理机动车数据，导入，查看前15行
df_vehicle=pd.read_csv('../input/vehicle population.csv')
df_vehicle.head(15)
#机动车数据进行年份筛选之后，我们根据年份进行分类求和，并重新设置索引，不需要设置drop=True
#并单独对number列进行归一化处理
df_vehicle=df_vehicle[df_vehicle['year'].isin(year)]
df_vehicle=df_vehicle.groupby('year').sum().reset_index()
df_vehicle['number']=(df_vehicle['number']-df_vehicle['number'].mean())/df_vehicle['number'].std()
df_vehicle.rename(columns={"number":"vehicle"},inplace=True)
df_vehicle
#将归一化的气体数据和归一化的机动车数据结合
df_vehicle2=pd.merge(df_std,df_vehicle,on='year',how='outer')
df_vehicle2
#计算相关系数，看到机动车和臭氧有85%的相关性，和二氧化氮有80%的相关性，下面绘制图表来看一下
df_vehicle2.corr(method="spearman")
vehicle_graph=df_vehicle.plot(x='year',y='vehicle',grid=True,figsize=DIMS,kind='line')
df_std.plot(x='year',y=['NO2_mean','O3_mean'],ax=vehicle_graph)
plt.legend(loc="center left",bbox_to_anchor=[1,0.5])
plt.show()
