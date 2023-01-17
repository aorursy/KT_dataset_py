import matplotlib.pyplot as plt

import numpy as np
x = np.array([15,20,30,35])

plt.plot(x,x**2,label="square")

##lable in x and  y axis



plt.xlabel('x')

plt.ylabel('y')

plt.title('line graph')

plt.plot(x,x*10,label="mult.")

plt.legend()

# plt.show() --- to break chart data
## 0,0 for 1st

## 0,1 for 2nd 

## 1,0 for 3rd

## 1,1 for 4ht





fig,axes = plt.subplots(nrows=2,ncols=2)

axes[0,0].plot(x,x**2)
import math
def cal(x):

    p=x**2

    q=x**1/2

    r=math.exp(x)

    s=math.log(x)

    return (p,q,r,s)



y=np.array(list(map(cal,x)))

y
k=0

fig,axes = plt.subplots(nrows=2,ncols=2)

for i in range(len(axes)):

    for j in range(axes.shape[1]):

        axes[i,j].plot(x,y[:,k])

        k+=1
import seaborn as sb
df=sb.load_dataset('tips')

df.head()
sb.distplot(df['total_bill'],kde=False)
sb.scatterplot(df['total_bill'],df['tip'],hue=df['sex'])

sb.lmplot(x='total_bill',y='tip',data=df)
sb.barplot(x='day',y='total_bill',data=df) #mean value to be plotted
sb.barplot(y='day',x='total_bill',data=df,estimator=np.sum)
sb.boxplot(y='total_bill',data=df)  # also called box viscous



## fixed y axis as numeric variable
sb.boxplot(y='total_bill',x='day',data=df) #seprate data for each class



#box 3 lines

## below q1 ,middle as q2, top as q3

## q2 line if lies between any side then it will be right or left skew if middle then normal distribution



#topmost value is q3+1.5IQR and below is Q3-1.5IQR
sb.violinplot(y='total_bill',x='day',data=df)
sb.violinplot(y='total_bill',x='day',data=df,hue='sex',split=True)
tip_cor=df.corr()

tip_cor
sb.heatmap(tip_cor,annot=True,cmap='Pastel1') # annot to print values also, # cmap gives color *matplotlib cmap
df1=sb.load_dataset('flights')

df1.head()
flight_pvt=df1.pivot(index='year',columns='month',values='passengers')

flight_pvt
sb.clustermap(flight_pvt,cmap='Pastel1')
sb.clustermap(flight_pvt,cmap='Pastel2')
df.head()
y1=['total_bill','tip','size']

x1=['sex','smoker','day','time']

for i in range(len(y1)):

    for j in range(len(x1)):

        sb.boxplot(x=x1[j],y=y1[i],data=df)  ##can also use "ax=axes[i,j]" in boxplot

        plt.show()

        

        
y1=['total_bill','tip','size']

x1=['sex','smoker','day','time']

k=1

for i in range(len(y1)):    

    for j in range(len(x1)):

        plt.subplot(4,3,k)

        sb.boxplot(x=x1[j],y=y1[i],data=df)  ##can also use "ax=axes[i,j]" in boxplot       

        k+=1

        

        
plt.figure(figsize=(20,10))

plt.plot(df['size'][0:10],color='r',linewidth=1.0,marker='o',markeredgecolor='y',markeredgewidth=10)

plt.plot(df['tip'][10:20],color='g',linewidth=0.5,markeredgecolor='b',marker='o',markerfacecolor='w')

plt.xlabel('x')

plt.ylabel('y')

plt.title('line graph')

plt.legend()
import matplotlib.pyplot as plt

import numpy as np



evens=list(range(2,100,2))

evens=np.array(evens)

plt.plot(evens,evens**2)

plt.show()



plt.plot(evens,evens**2,label='x^2')

plt.legend() # to display label

plt.savefig('output.png',dpi=100) #savin graph as image

plt.show()
pwd
import random

years=[2009,2010,2011,2012,2013,2014,2015]

android=np.random.randint(20,1200,7)

ios=np.random.randint(50,700,7)

microsoft=np.random.randint(12,1000,7)



plt.plot(years,android,label='android')

plt.plot(years,ios,label='ios')

plt.plot(years,microsoft,label='microsoft')

plt.legend()
import matplotlib as mpl

#prepare 4 lines with different slopes

x=np.linspace(0,200,100) # prepare 100 even spaced number between 0 to 200

y1=x*2

y2=x*3

y3=x*4

y4=x*5

# set line width to 2 for clarity

mpl.rcParams['lines.linewidth']=2

# drawing the 4 lines

plt.plot(x,y1,label='2x',c='0')

plt.plot(x,y2,label='3x',c='0.2',ls='--')

plt.plot(x,y3,label='4x',c='0.4',ls='-.')

plt.plot(x,y4,label='5x',c='0.6',ls=':')

plt.legend()

plt.show()
fontparams={'size':16,'fontweight':'light','family':'monospace'}

x=[1,2,3]

y=[2,3,4]

plt.plot(x,y)

plt.xlabel('xlabel',size=20,fontweight='semibold',family='serif')

plt.ylabel('ylabel',fontparams)

plt.show()
#prepare a curve of square nos.

import matplotlib.pyplot as plt

x=np.linspace(0,200,100) # prepare 100 even spaced number between 0 to 200

y1=x

y2=x+20

# plot a curve of a square nos.

plt.plot(x,y1,label='$x$')

plt.plot(x,y2,label=r'$x^3+\beta$')

plt.legend()

plt.show()
tech=['google','apple','microsoft','samsung']

y_pos=np.arange(len(tech))



web=[1,2,3,4]

plt.bar(y_pos,web,align='center',alpha=0.9)

plt.xticks(y_pos,tech,rotation=75)

plt.ylabel('live starts count')

plt.title('online advert tech usage')

plt.show()
x= np.linspace(0,10,100)

y2=np.sin(x**2)

y1=x**2



#initiate a figure with a subplot axis

fig,ax1=plt.subplots()



# set the insert plot dim

left,bottom,width,height=[0.22,0.45,0.3,0.35]

ax2=fig.add_axes([left,bottom,width,height])



#draw the plots

ax1.plot(x,y1)

ax2.plot(x,y2)

plt.show()