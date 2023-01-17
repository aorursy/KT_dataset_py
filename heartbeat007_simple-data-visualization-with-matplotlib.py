import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns
### line graph of the matplotlib
starting   = 1.0
ending     = 10.0
num        = 10
x          = range(10)
y          = np.linspace(1.0,10.0,10)


plt.plot(x,y)
## legend title and lebels within matplotlib
import math
x      = np.linspace(1,10,1000)
y1     = np.sin(x)
y2     = np.cos(x) 



plt.plot(x,y1 ,label = "sin curve")
plt.plot(x,y2 ,label = "cos curve")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("sin wave")
plt.legend()
## doing bar chart and histogram
## bar chart show the actual value
## the histogram will show the distibution
x     = [x for x in range(1,11) if x%2==0]
y     = [x for x in range(11) if x%2!=0]



plt.bar(x,y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("bar chart")
## multiple bar chart with legend
x1     = [1,2,3,4,5,6,7,8,9,10]
x2     = [5,6,7,8,9,10,11,12,13,14,15]
y1     = [5,6,4,3,2,3,4,56,6,7]
y2     = [23,12,34,56,3,23,34,34,12,10,20]


## for bar chart you need do give color 
## with the label
plt.bar(x1,y1,color = "r",label="first")
plt.bar(x2,y2,color="b",label="second")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("multiple bar char")
plt.show()
data      = [x+np.random.randint(100) for x in range(100) if x%4==0]
x         = [x for x in range(len(data))]


plt.bar(x,data)
## histogram use to condense the data with bins
data      = [x-np.random.randint(100) for x in range(100) if x%4==0]


plt.hist(data,label="histogram")
plt.legend()
## scatter plot simple graph
y1      =  [x+np.random.randint(100) for x in range(50)]
y2      =  [x+np.random.randint(100) for x in range(50)]
x       =  [x for x in range(len(y1))]




plt.scatter(x,y1,label = "first")
plt.scatter(x,y2,label = "second",marker = "*",s=150) ## s for sizes
plt.legend()
# with different marker size
plt.scatter(x,y1,label = "first")
plt.scatter(x,y2,label = "second",marker = "*",s=400) ## s for sizes
plt.legend()
## stack plot for matplotlib
days         = [x for x in range(1,8)]
sleeping     = [10,6,8,7,6,9,3]
working      = [8,5,6,4,8,10,2]
eating       = [2,4,3,5,4,3,2]
playing      = [1,3,2,3,2,3,2]

## you can have label in the stack plot
plt.plot([],[],color='m',label="sleeping",linewidth=10)
plt.plot([],[],color='c',label="working",linewidth=10)
plt.plot([],[],color='r',label="eating",linewidth=10)
plt.plot([],[],color='k',label="playing",linewidth=10)
## these are the fake plots for the label


plt.stackplot(days,sleeping,working,eating,playing,colors=['m','c','r','k'])
plt.legend()
## pie chars using matplotlib
hours    = [8,8,4,2,2]
work     = ['sleep','work','play','eat','other']
explode = (0, 0.1, .1, 0,.1) 



plt.pie(hours,labels = work,explode=explode)
plt.legend()
## loading data from csv data and plotting it
## loading the iris data from github
import pandas as pd
df = pd.read_csv('https://gist.githubusercontent.com/tanviredu/43e5ebea680d1630d9544f722aecce53/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
df.head()
X = []
Y = []
for item in df.values:
  X.append(item[0])
  Y.append(item[1])
plt.scatter(X,Y)  
## we can add multiple data for scatering
X = []
Y = []
Z = []
for item in df.values:
  X.append(item[0])
  Y.append(item[1])
  Z.append(item[2])
plt.scatter(X,Y,label  = "first  properties")
plt.scatter(X,Z, label = "Second properties")
plt.legend()



df = pd.read_csv('https://raw.githubusercontent.com/tanviredu/Google-Stock-Price-Prediction/master/Google_Stock_Price_Train.csv')
df.head()
df = df.set_index("Date")
df.head()
df['Close'] = pd.to_numeric(df['Close'], errors='coerce').astype('float64')

df[['Open','Close']].plot()
plt.xlabel("")
plt.figure()
plt.show()

# creating subplot
import numpy              as np
import matplotlib.pyplot  as pd
def curve(sin=True):
  if sin==True:

    X = np.linspace(1,10,100)
    Y = np.sin(X)
    return X,Y
  else:
    X = np.linspace(1,10,100)
    Y = np.cos(X)
    return X,Y


x1,y1 = curve(sin=True)
x2,y2 = curve(sin=False)
plt.plot(x1,y1,label = "sin curve")
plt.plot(x2,y2,label  = "cos curve") 
plt.legend()

## figure with a just one subplot
fig, ax = plt.subplots()
ax.plot(x1, y1)
ax.set_title('A single plot')
## stacking subplot
row = 2
fig, ax = plt.subplots(row)
fig.suptitle('Vertically stacked subplots')
ax[0].plot(x1, y1,color='r')
ax[1].plot(x2, y2)

## horijontal plot
row = 1
column = 2
fig, (ax1, ax2) = plt.subplots(row, column)
fig.suptitle('Horizontally stacked subplots')
ax1.plot(x1, y1,color='r')
ax2.plot(x2, y2)
x1,y1 = curve(sin=True)
x2,y2 = curve(sin=True)
x3,y3 = curve(sin=False)
x4,y4 = curve(sin=False)
col = 2
row = 2
fig, axs = plt.subplots(row, col)
axs[0, 0].plot(x1, y1)
axs[0, 0].set_title('Axis [0,0]')
axs[0, 1].plot(x2, y2, 'tab:orange')
axs[0, 1].set_title('Axis [0,1]')
axs[1, 0].plot(x3, y3, 'tab:green')
axs[1, 0].set_title('Axis [1,0]')
axs[1, 1].plot(x4, y4, 'tab:red')
axs[1, 1].set_title('Axis [1,1]')

row = 3
fig, axs = plt.subplots(row, sharex=True, sharey=True)
fig.suptitle('Sharing both axes')
axs[0].plot(x1, y1)
axs[1].plot(x2, y2, 'o')
axs[2].plot(x3, y3, '+')
row = 3
fig, axs = plt.subplots(row, sharex=True, sharey=True)
fig.suptitle('Sharing both axes')
axs[0].plot(df['Open'])
axs[1].plot(df['Close'], 'o')
axs[2].plot(df['High'], '+')
import pandas as pd
df = pd.read_csv('https://gist.githubusercontent.com/tanviredu/43e5ebea680d1630d9544f722aecce53/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
df.head()
print(df['variety'].unique())
explode = (0, 0.1, .1) 
plt.pie(df['variety'].value_counts(),labels=df['variety'].unique(),explode=explode)
plt.legend()

