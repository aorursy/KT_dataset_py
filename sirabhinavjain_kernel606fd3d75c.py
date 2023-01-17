from IPython.display import Image

Image(url= 'https://bit.ly/2HV74du')
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

%matplotlib inline



train = pd.read_csv('../input/iris/Iris.csv',encoding = 'utf-8')

train.drop(['Id'],axis = 1,inplace = True)

train.head()





train.info()



train_categoury = train.select_dtypes(include = ['object'])

print('Number of Object Type Columns:',train_categoury.shape)

train_categoury.head(3)



train_Numeric = train.select_dtypes(include = ['float64'])

train_Numeric.head(3)





train.describe()



train_Avg = train.groupby('Species').SepalLengthCm.mean()

train_Avg1=pd.DataFrame(train_Avg)

train_Avg1



Y = train['Species'].unique()

y=pd.DataFrame(Y)

Y



plt.plot(Y,train_Avg,marker = 's')

plt.xlabel('Species')

plt.ylabel('Average_Sepal_Length_(Cm)')

plt.title('Species_Vs_Average_Sepal_Width_(Cm)',size = 15)
list(train_Numeric)
train_avg2 = train.groupby('Species').SepalWidthCm.mean()

train_avg2.head()

plt.plot(Y,train_avg2,marker = 's')

plt.xlabel('Species')

plt.ylabel('Average_Sepal_Width_(Cm)')

plt.title('Species_Vs_Average_Sepal_Width_(Cm)',size = 15)
train_avg3 = train.groupby('Species').PetalLengthCm.mean()
plt.plot(Y,train_avg3,marker = 's')

plt.xlabel('Species')

plt.ylabel('Average_Petal_Length_(Cm)')

plt.title('Species_Vs_Average_Petal_Length_(Cm)',size = 15)
train_avg4 = train.groupby('Species').PetalWidthCm.mean()
plt.plot(Y,train_avg4,marker = 's')

plt.xlabel('Species')

plt.ylabel('Average_Petal_Width_(Cm)')

plt.title('Species_Vs_Average_Petal_Width_(Cm)',size = 15)
plt.figure(figsize = (12,8))

sns.set_style('whitegrid')

plt.plot(Y,train_Avg,marker = 's')

plt.plot(Y,train_avg2,marker = 's')

plt.plot(Y,train_avg3,marker = 's')

plt.plot(Y,train_avg4,marker = 's')

plt.xlabel('Species',size = 15)

plt.ylabel('Ave_of_numeric_variables',size = 15)

plt.title(' Species Vs Numeric_variables',size = 15)

plt.legend()
train.head()
plt.close()

sns.set_style('whitegrid')

sns.pairplot(train,hue = 'Species',size = 3)

plt.show()
plt.rcParams['figure.figsize']=(12,9)






for i in list(train_Numeric):

    sns.distplot(train[i],bins = 23,hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2}, color = 'darkblue',kde=True,hist=True)

    plt.xlabel(i,size = 15)

    plt.ylabel('percentage of distribution',size = 15)

    plt.title('histogram of '+ i,size = 15)

    plt.xticks(np.arange(0,10,1))

    plt.figure(figsize=(20,18))

    plt.show()
train.SepalWidthCm.std()
#plt.figure(figsize= (8,6))

for i in train_Numeric:

    plt.figure(figsize= (8,6))

    a = sns.boxplot(x='Species',y = i ,data = train)

    plt.title('Finding number of outliers in ' + i,size = 15)

    a.set_xlabel('Species',fontsize = 15)

    a.set_ylabel(i,fontsize=15)

    plt.show()
from string import ascii_letters



corr = train.corr()



mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Fig:1',size=15)
sns.heatmap(corr,cmap = 'Blues',square=True)

plt.title('Fig:2',size=20)
for i in train_Numeric:

    

    x = np.sort(train[i])

    y = np.arange(1 , len(x)+1) / len(x)

    plt.plot(x,y,marker='.',linestyle='none')

    #plt.margins(0.02)

    plt.legend()

    plt.show()
list(train_Numeric)
plt.figure(figsize=(8,6))



x_SepalLengthCm = np.sort(train['SepalLengthCm'])

x_SepalWidthCm = np.sort(train['SepalWidthCm'])

x_PetalLengthCm = np.sort(train['PetalLengthCm'])

x_PetalWidthCm = np.sort(train['PetalWidthCm'])



y1 = np.arange(1 , len(x_SepalLengthCm)+1) / len(x_SepalLengthCm)

y2 = np.arange(1 , len(x_SepalWidthCm)+1) / len(x_SepalWidthCm)

y3 = np.arange(1 , len(x_PetalLengthCm)+1) / len(x_PetalLengthCm)

y4 = np.arange(1 , len(x_PetalWidthCm)+1) / len(x_PetalWidthCm)



plt.plot(x_SepalLengthCm,y1,marker='.',linestyle='none',label='SepalLengthCm')

plt.plot(x_SepalWidthCm,y2,marker='.',linestyle='none',label='SepalWidthCm')

plt.plot(x_PetalLengthCm,y3,marker='.',linestyle='none',label='PetalLengthCm')

plt.plot(x_PetalWidthCm,y4,marker='.',linestyle='none',label='PetalWidthCm')



plt.margins(0.02)

plt.legend(loc=4,fontsize='large')

plt.show()