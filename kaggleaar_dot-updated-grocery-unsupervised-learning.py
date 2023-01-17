import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

import warnings



import itertools

warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)
#loading the data set



ws_df = pd.read_csv('../input/Wholesale%20customers%20data.csv')

ws_df.head(500)
ws_df.drop(labels=(['Channel','Region']),axis=1,inplace=True)
ws_df.info()

ws_df.shape
ws_df.rename(columns={'Fresh':'Rice','Milk':'Sugar','Grocery':'Dal','Frozen':'Flour','Detergents_Paper':'Salt','Delicassen':'Oil'},inplace=True)
print(ws_df.columns)
ws_df.info()
ws_df.head(5)
ws_df.head(5).plot.bar()

plt.title('Amount Weight description')

plt.xlabel('---Items---')

plt.ylabel('---Weight in m.u---')
ws_df.describe().T
ws_df.describe().T.plot.bar()
import itertools



attr_col = [i for i in ws_df.columns if i not in 'strength']

length = len(attr_col)

cs = ["r","c","b","g","k"]

fig = plt.figure(figsize=(15,25))



for i,j,k in itertools.zip_longest(attr_col,range(length),cs):

    plt.subplot(4,2,j+1)

    ax = sns.distplot(ws_df[i],color=k,rug=True)

    ax.set_facecolor("w")

    plt.axvline(ws_df[i].mean(),linestyle="dashed",label="mean",color="brown")

    plt.legend(loc="best")

    plt.title(i,color="red")

    plt.xlabel("")
#Summary View of all attribute , The we will look into all the boxplot individually to trace out outliers



ax = sns.boxplot(data=ws_df, orient="h")
from sklearn.preprocessing import normalize



X_std = normalize(ws_df)

X_std = pd.DataFrame(X_std, columns=ws_df.columns)

X_std.head(6)

#RANDOM ANALYSIS

#The main aim of the below graphs is to understand relation between products or items and their consumption using random analysis.
import matplotlib.pyplot as plt 

  

# defining labels 

activities = ['Rice', 'Sugar', 'Dal','Flour', 'Salt' , 'Oil'] 

  

# portion covered by each label 

slices = [8, 5, 7, 2, 4, 6] 

  

# color for each label 

colors = ['g', 'b', 'r', 'pink','y', 'violet'] 

  

# plotting the pie chart 

plt.pie(slices, labels = activities, colors=colors,  

        startangle=90, shadow = True, explode = (0, 0, 0.1, 0,0,0), 

        radius = 2.3, autopct = '%1.1f%%') 

  

# plotting legend 

plt.legend() 

  

# showing the plot 

plt.show() 
import matplotlib.pyplot as plt 



# x-coordinates of left sides of bars 

left = [1, 3, 5, 7, 9, 11] 



# heights of bars 

height = [33.3, 20.8, 29.2, 17.7, 16.7, 8.8] 



# labels for bars 

tick_label = ['Rice', 'Sugar', 'Dal', 'Flour', 'Salt' , 'Oil'] 



# plotting a bar chart 

plt.bar(left, height, tick_label = tick_label, 

		width = 0.8, color = ['red', 'y']) 



# naming the x-axis 

plt.xlabel('---products/items---') 

# naming the y-axis 

plt.ylabel('---percentage/proportion---') 

# plot title 

plt.title('Consumption plot') 



# function to show the plot 

plt.show() 

import matplotlib.pyplot as plt



days = [1,2,3,4,5]



Rice = [7,8,6,11,7]

Sugar =   [2,3,4,3,2]

Dal =  [7,8,7,2,2]

Oil =  [7,8,7,1,2]

Flour =  [8,5,7,8,13]

Salt  = [7,4,5,9,11]







plt.plot([],[],color='m', label='Rice', linewidth=5)

plt.plot([],[],color='c', label='Sugar', linewidth=5)

plt.plot([],[],color='r', label='Dal', linewidth=5)

plt.plot([],[],color='y', label='Oil', linewidth=5)

plt.plot([],[],color='k', label='Flour', linewidth=5)

plt.plot([],[],color='b', label='Salt', linewidth=5)





plt.stackplot(days,Rice,Sugar,Dal,Flour,Salt,Oil, colors=['m','c','r','y','k','b'])



plt.xlabel('--Along x-axis--')

plt.ylabel('--Along y-axis--')

plt.title('--Avg annual Proprortion expenditure--')

plt.legend()

plt.show()
import matplotlib.pyplot as plt



slices = [10,12,15,7,5, 11]

activities = ['Rice','Sugar','Dal','Oil','Flour','Salt']

cols = ['c','m','r','b','y','pink']



plt.pie(slices,

        labels=activities,

        radius=2.1,

        colors=cols,

        startangle=90,

        shadow= True,

        explode=(0,0.1,0,0,0,0),

        autopct='%1.1f%%')



plt.title('---Monthly Expenditure in m.u---')

plt.show()
import matplotlib.pyplot as plt

import numpy as np

import urllib

import matplotlib.dates as mdates



def bytespdate2num(fmt, encoding='utf-8'):

    strconverter = mdates.strpdate2num(fmt)

    def bytesconverter(b):

        s = b.decode(encoding)

        return strconverter(s)

    return bytesconverter

    



def graph_data(stock):



    fig = plt.figure()

    ax1 = plt.subplot2grid((1,1), (0,0))

    

    # Unfortunately, Yahoo's API is no longer available

    # feel free to adapt the code to another source, or use this drop-in replacement.

    stock_price_url = 'https://pythonprogramming.net/yahoo_finance_replacement'

    source_code = urllib.request.urlopen(stock_price_url).read().decode()

    stock_data = []

    split_source = source_code.split('\n')

    for line in split_source[1:]:

        split_line = line.split(',')

        if len(split_line) == 7:

            if 'values' not in line and 'labels' not in line:

                stock_data.append(line)



    date, closep, highp, lowp, openp, adj_closep, volume = np.loadtxt(stock_data,

                                                          delimiter=',',

                                                          unpack=True,

                                                          converters={0: bytespdate2num('%Y-%m-%d')})



    ax1.plot_date(date, closep,'-', label='Average Price(Random)')

    for label in ax1.xaxis.get_ticklabels():

        label.set_rotation(45)

    ax1.grid(True)#, color='g', linestyle='-', linewidth=5)



    plt.xlabel('Date')

    plt.ylabel('Price')

    plt.title('--Time series depiction of expenditure--')

    plt.legend()

    plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)

    plt.show()

graph_data('TSLA')
heat_map=sns.heatmap(ws_df.head(5))

plt.title('Item wt description of 1st few rows')

plt.show()

heat_map=sns.heatmap(ws_df.tail(5),cmap='YlGnBu')

plt.title('Item wt description of last few rows')

plt.show()