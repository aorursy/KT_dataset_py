#setting the local path
your_local_path="C:\Program Files (x86)\Python36-32/"
#Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#reading the file
#WineData=pd.read_csv(your_local_path+'winequality.csv')
#WineData.head(10)http://localhost:8888/notebooks/EDA%20Project.ipynb#
WineData.groupby('quality').size()
WineData
#To understand the distribution of each variable , we plotted histograms
#WineData.hist('quality')
WineData.hist(figsize=(8,15),xlabelsize=10,grid=False)
plt.show()

#ax = plt.subplots(figsize=(10,10))
#WineData.boxplot(['chlorides','alcohol','density','pH'],figsize=(20,10))
#plt.show()
#fixed acidity	volatil acidity	citric acid	residual sugar	chlorides	free sulfur dioxide	total sulfur dioxide	density	pH	sulphates	alcohol	quality
plt.figure(figsize=(10,8))
plt.subplots_adjust(hspace= 0.2, wspace= 0.5)

plt.subplot(2,6,1)
WineData.boxplot('fixed_acidity',widths=0.3)
plt.subplot(2,6,2)
WineData.boxplot('volatile_acidity',widths=0.3)
plt.subplot(2,6,3)
WineData.boxplot('citric_acid',widths=0.3)
plt.subplot(2,6,4)
WineData.boxplot('residual_sugar',widths=0.3)
plt.subplot(2,6,5)
WineData.boxplot('chlorides',widths=0.3)
plt.subplot(2,6,6)
WineData.boxplot('free_S_dioxide',widths=0.3)
plt.subplot(2,6,7)
WineData.boxplot('total_S_dioxide',widths=0.3)
plt.subplot(2,6,8)
WineData.boxplot('density',widths=0.3)
plt.subplot(2,6,9)
WineData.boxplot('pH',widths=0.3)
plt.subplot(2,6,10)
WineData.boxplot('sulphates',widths=0.3)
plt.subplot(2,6,11)
WineData.boxplot('alcohol',widths=0.3)
plt.subplot(2,6,12)
WineData.boxplot('quality',widths=0.3)
plt.show()
WineData_Summary=WineData.describe().T
WineData_Summary

#WineData=WineData[WineData.apply(lambda x: np.abs(x - x.mean()) < 2*x.std())]
WineData_refined=WineData[WineData.apply(lambda x: x < (x.quantile(0.75)+1.5*(x.quantile(0.75)-x.quantile(0.25))))]
WineData_refined
WineData_refined['quality'] = WineData_refined['quality'].fillna(value=8)

#ax = plt.subplots(figsize=(10,10))
#WineData.boxplot(['chlorides','alcohol','density','pH'],figsize=(20,10))
#plt.show()
#fixed acidity	volatil acidity	citric acid	residual sugar	chlorides	free sulfur dioxide	total sulfur dioxide	density	pH	sulphates	alcohol	quality
plt.figure(figsize=(10,8))
plt.subplots_adjust(hspace= 0.2, wspace= 0.5)

plt.subplot(2,6,1)
WineData_refined.boxplot('fixed_acidity',widths=0.3)
plt.subplot(2,6,2)
WineData_refined.boxplot('volatile_acidity',widths=0.3)
plt.subplot(2,6,3)
WineData_refined.boxplot('citric_acid',widths=0.3)
plt.subplot(2,6,4)
WineData_refined.boxplot('residual_sugar',widths=0.3)
plt.subplot(2,6,5)
WineData_refined.boxplot('chlorides',widths=0.3)
plt.subplot(2,6,6)
WineData_refined.boxplot('free_S_dioxide',widths=0.3)
plt.subplot(2,6,7)
WineData_refined.boxplot('total_S_dioxide',widths=0.3)
plt.subplot(2,6,8)
WineData_refined.boxplot('density',widths=0.3)
plt.subplot(2,6,9)
WineData_refined.boxplot('pH',widths=0.3)
plt.subplot(2,6,10)
WineData_refined.boxplot('sulphates',widths=0.3)
plt.subplot(2,6,11)
WineData_refined.boxplot('alcohol',widths=0.3)
plt.subplot(2,6,12)
WineData_refined.boxplot('quality',widths=0.3)
plt.show()
#WineData_refined.quality.fillna('8')
df.groupby('quality').size()


#df1[df1.free_S_dioxide==3]
df=WineData_refined.fillna(WineData.mean())
df
def f(col):
    if col['quality'] > 6 :
        val = 'High'
    elif col['quality'] > 4  :
        val = 'Med'
    else:
        val = 'Low'
    return val
df['Category'] = df.apply(f, axis=1)
df.head(45)
#df1=df.sort_values(by='quality')
#df1
df['Category'].value_counts().plot(kind='bar')
plt.show()


#ax = plt.subplots(figsize=(10,10))
#WineData.boxplot(['chlorides','alcohol','density','pH'],figsize=(20,10))
#plt.show()
#fixed acidity	volatil acidity	citric acid	residual sugar	chlorides	free sulfur dioxide	total sulfur dioxide	density	pH	sulphates	alcohol	quality
plt.figure(figsize=(10,8))
plt.subplots_adjust(hspace= 0.2, wspace= 0.5)

plt.subplot(2,6,1)
WineData.boxplot('fixed_acidity',widths=0.3)
plt.subplot(2,6,2)
WineData.boxplot('volatile_acidity',widths=0.3)
plt.subplot(2,6,3)
WineData.boxplot('citric_acid',widths=0.3)
plt.subplot(2,6,4)
WineData.boxplot('residual_sugar',widths=0.3)
plt.subplot(2,6,5)
WineData.boxplot('chlorides',widths=0.3)
plt.subplot(2,6,6)
WineData.boxplot('free_S_dioxide',widths=0.3)
plt.subplot(2,6,7)
WineData.boxplot('total_S_dioxide',widths=0.3)
plt.subplot(2,6,8)
WineData.boxplot('density',widths=0.3)
plt.subplot(2,6,9)
WineData.boxplot('pH',widths=0.3)
plt.subplot(2,6,10)
WineData.boxplot('sulphates',widths=0.3)
plt.subplot(2,6,11)
WineData.boxplot('alcohol',widths=0.3)
plt.subplot(2,6,12)
WineData.boxplot('quality',widths=0.3)
plt.show()
plt.figure(figsize=(15,10))
plt.subplots_adjust(hspace= 0.2, wspace= 0.5)

plt.subplot(2,6,1)
sns.boxplot(x='Category',y='fixed_acidity',data=df,linewidth=3)
plt.subplot(2,6,2)
sns.boxplot(x='Category',y='volatile_acidity',data=df,linewidth=3)
plt.subplot(2,6,3)
sns.boxplot(x='Category',y='citric_acid',data=df,linewidth=3)
plt.subplot(2,6,4)
sns.boxplot(x='Category',y='residual_sugar',data=df,linewidth=3)
plt.subplot(2,6,5)
sns.boxplot(x='Category',y='chlorides',data=df,linewidth=3)
plt.subplot(2,6,6)
sns.boxplot(x='Category',y='free_S_dioxide',data=df,linewidth=3)
plt.subplot(2,6,7)
sns.boxplot(x='Category',y='total_S_dioxide',data=df,linewidth=3)
plt.subplot(2,6,8)
sns.boxplot(x='Category',y='density',data=df,linewidth=3)
plt.subplot(2,6,9)
sns.boxplot(x='Category',y='pH',data=df,linewidth=3)
plt.subplot(2,6,10)
sns.boxplot(x='Category',y='sulphates',data=df,linewidth=3)
plt.subplot(2,6,11)
sns.boxplot(x='Category',y='alcohol',data=df,linewidth=3)

plt.show()
#WineData.set_index('quality',inplace=True)
WineData.reset_index(inplace=True)
#WineData.sort_values(by='quality')

plt.figure(figsize=(15,5))
plt.subplots_adjust(hspace= 0.2, wspace= 0.5)
plt.subplot(1,3,1)
plt.scatter(df.quality,df.residual_sugar)
plt.xlabel('quality')
plt.ylabel('residual_sugar')
plt.subplot(1,3,2)
plt.xlabel('quality')
plt.ylabel('residual_sugar')
plt.scatter(df.quality,df.density)
plt.subplot(1,3,3)
plt.scatter(df.quality,df.alcohol)
plt.xlabel('quality')
plt.ylabel('residual_sugar')

plt.show()
df
x = range(100)
y = range(100,200)
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(x[:4], y[:4], s=10, c='b', marker="s", label='first')
ax1.scatter(x[40:],y[40:], s=10, c='r', marker="o", label='second')
plt.legend(loc='upper left');
plt.show()
plt.scatter(df['alcohol'], df['sulphates'], color='black', label='High')
plt.scatter(df['Category'], df['alcohol'], color='red', label='Low')
plt.show()
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([2,1,3,6,7])

cluster = np.array([1,1,1,2,2]) 

fig, ax = plt.subplots()

ax.scatter(x[cluster==1],y[cluster==1], marker='^')
ax.scatter(x[cluster==2],y[cluster==2], marker='s')

plt.show()
df_m=df[['alcohol','sulphates','citric_acid','volatile_acidity','Category']]
df_m.groupby("Category").size()
#ata1=sns.load_dataset("df_m")
sns.pairplot(df_m, hue="Category",palette="husl")
plt.show()
df_mm=df_m[~(df_m.Category=='Med')]
df_mm.groupby("Category").size()
sns.set(style="ticks")
color_codes=True
sns.pairplot(df_mm, hue="Category",markers=["o", "s"])
plt.show()
