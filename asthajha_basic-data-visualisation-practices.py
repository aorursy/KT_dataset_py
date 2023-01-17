# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt 

import numpy as np
#line plot



x=np.arange(10)

y1=x**2 

y2=2*x+3

print(x)

print(y1)

print(y2)
themes=plt.style.available

print(themes)
plt.style.use("seaborn-dark-palette")
plt.plot(x,y1,color='red',label="Apple",marker="o")    # to plot 

#plt.show()    #  if both has to be printed on different graphs 

plt.plot(x,y2,color='green',label="Kiwi",linestyle="dashed")



plt.xlabel("Time")

plt.ylabel("Price")

plt.title("Prices of fruit over time ")

plt.legend()

plt.show()          # prints on the same graph



# agar x axis k liye data na bhi ho 

prices=np.array([1,2,3,4])**2

print(prices)

plt.plot(prices)
# SCATTER PLOTS 

plt.scatter(x,y1)

plt.show()
# adjust the size of any plot 

#plt.figure(figsize=(5,5))





plt.scatter(x,y1,color='red',label="Apple",marker="^")     

#plt.show()    

plt.scatter(x,y2,color='green',label="Kiwi",linestyle="dashed")



plt.xlabel("Time")

plt.ylabel("Price")

plt.title("Prices of fruit over time ")

plt.legend()

plt.show()         



plt.bar([0,1,2],[10,20,15]) #current year

plt.bar([0,1,2],[20,10,12])  #next year 

plt.show()



# this is overlapping , but we want side by side  
x_coor= np.array([0,1,2])*2

plt.bar(x_coor-0.25,[10,20,15],width=0.5,label="current year" , tick_label=["gold","silver","platinum"]) #current year

plt.bar(x_coor+0.25 ,[20,10,12],width=0.5,label="next year")  #next year 

plt.legend()

plt.ylim(0,40)   # for scaling

plt.xlim(-2,5)

plt.xlabel("Metal")

plt.ylabel("Price")

plt.title("Metal Price comparison")

plt.show()

#plt.style.use("dark_background")

x_coor= np.array([0,1,2])*2

plt.bar(x_coor-0.25,[10,20,15],width=0.5,label="current year" , tick_label=["gold","silver","platinum"]) #current year

plt.bar(x_coor+0.25 ,[20,10,12],width=0.5,label="next year",color="orange")  #next year 

plt.legend()

plt.xlabel("Metal")

plt.ylabel("Price")

plt.title("Metal Price comparison")

plt.show()

subjects="ME","ITC","DCCN","DSP"

weightage= [10,20,15,5]

plt.pie(weightage,labels=subjects)
subjects="ME","ITC","DCCN","DSP"

weightage= [10,20,15,5]

plt.pie(weightage,labels=subjects,explode=(1,0,0.1,0),autopct='%1.1f%%')

plt.show()
import pandas as pd



pwd             # default directory
df=pd.read_csv("https://raw.githubusercontent.com/coding-blocks-archives/machine-learning-online-2018/master/2.%20Working%20with%20Libraries/movie_metadata.csv")

print(df.head(n=10))
df.columns
titles = list(df.get(('movie_title')))
print(titles[:5])

print(titles[0][:-1])       #special chars ko hatane k liye
freq_title={}

for x in titles:

    l=len(x)

    

    if freq_title.get(l) is None:

        freq_title[l]=1

                                       # if a particular length is coming for first time then we make it as 1

    else:

        freq_title[l]+=1

        
freq_title          # 138 movies in which the title length is 7 and so on
print(freq_title)
X=np.array(list(freq_title.keys()))

Y=np.array(list(freq_title.values()))

#print(X,Y)
plt.scatter(X,Y)

plt.xlabel("length of movie title")

plt.ylabel("no. of movies having this much long title ")

plt.title("movie data visualization problem")

plt.show()               # ye kuch gaussian sa aa rha hai