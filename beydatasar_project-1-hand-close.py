# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/EL_KAPAMA_DATA_SET_300.csv')
data.info()
data.describe()
data.head(5)
data.tail(5)
data.corr() # fetureler arasındaki korelasyonu veriyor
#correlation map
f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.columns
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
f,ax = plt.subplots(figsize=(20, 10))
plt.subplot(3,1,1)
data.MAX_1.plot(kind = 'line', color = 'green',label = 'MAX_1',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.MAX_2.plot(kind = 'line', color = 'blue',label = 'MAX_2',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.MAX_3.plot(kind = 'line', color = 'black',label = 'MAX_3',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.MAX_4.plot(kind = 'line', color = 'yellow',label = 'MAX_4',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.MAX_5.plot(kind = 'line', color = 'pink',label = 'MAX_5',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.MAX_6.plot(kind = 'line', color = 'purple',label = 'MAX_6',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.MAX_7.plot(kind = 'line', color = 'red',label = 'MAX_7',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.MAX_8.plot(kind = 'line', color = 'gray',label = 'MAX_8',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Sample')              # label = name of label
plt.ylabel('MAX')
plt.title('MAX-SAMPLE Plot')            # title = title of plot


plt.subplot(3,1,2)
data.ETR_1.plot(kind = 'line', color = 'green',label = 'ETR_1',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.ETR_2.plot(kind = 'line', color = 'blue',label = 'ETR_2',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.ETR_3.plot(kind = 'line', color = 'black',label = 'ETR_3',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.ETR_4.plot(kind = 'line', color = 'yellow',label = 'ETR_4',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.ETR_5.plot(kind = 'line', color = 'pink',label = 'ETR_5',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.ETR_6.plot(kind = 'line', color = 'purple',label = 'ETR_6',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.ETR_7.plot(kind = 'line', color = 'red',label = 'ETR_7',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.ETR_8.plot(kind = 'line', color = 'gray',label = 'ETR_8',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Sample')              # label = name of label
plt.ylabel('ENTROPY')
plt.title('ENTROPY-SAMPLE Plot')            # title = title of plot

plt.subplot(3,1,3)
data.VAR_1.plot(kind = 'line', color = 'green',label = 'VAR_1',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.VAR_2.plot(kind = 'line', color = 'blue',label = 'VAR_2',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.VAR_3.plot(kind = 'line', color = 'black',label = 'VAR_3',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.VAR_4.plot(kind = 'line', color = 'yellow',label = 'VAR_4',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.VAR_5.plot(kind = 'line', color = 'pink',label = 'VAR_5',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.VAR_6.plot(kind = 'line', color = 'purple',label = 'VAR_6',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.VAR_7.plot(kind = 'line', color = 'red',label = 'VAR_7',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data.VAR_8.plot(kind = 'line', color = 'gray',label = 'VAR_8',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Sample')              # label = name of label
plt.ylabel('VARIANCE')
plt.title('VARIANCE-SAMPLE Plot')            # title = title of plot

#plt.show()
beyda=data[data.RESULT==1]
alper=data[data.RESULT==2]
ahmet=data[data.RESULT==3]
irem=data[data.RESULT==4]

f,ax = plt.subplots(figsize=(20, 10))
plt.subplot(2,4,1)
plt.scatter(beyda.MAX_1,beyda.ETR_1,color="red",label="beyda")
plt.scatter(alper.MAX_1,alper.ETR_1,color="blue",label="alper")
plt.scatter(ahmet.MAX_1,ahmet.ETR_1,color="green",label="ahmet")
plt.scatter(irem.MAX_1,irem.ETR_1,color="yellow",label="irem")

plt.legend()
plt.xlabel("MAX_1")
plt.ylabel("ETR_1")
plt.title("MAX_1-ETR_1 Scatter Plot")


plt.subplot(2,4,2)
plt.scatter(beyda.MAX_2,beyda.ETR_2,color="red",label="beyda")
plt.scatter(alper.MAX_2,alper.ETR_2,color="blue",label="alper")
plt.scatter(ahmet.MAX_2,ahmet.ETR_2,color="green",label="ahmet")
plt.scatter(irem.MAX_2,irem.ETR_2,color="yellow",label="irem")

plt.legend()
plt.xlabel("MAX_2")
plt.ylabel("ETR_2")
plt.title("MAX_2-ETR_2 Scatter Plot")


plt.subplot(2,4,3)
plt.scatter(beyda.MAX_3,beyda.ETR_3,color="red",label="beyda")
plt.scatter(alper.MAX_3,alper.ETR_3,color="blue",label="alper")
plt.scatter(ahmet.MAX_3,ahmet.ETR_3,color="green",label="ahmet")
plt.scatter(irem.MAX_3,irem.ETR_3,color="yellow",label="irem")

plt.legend()
plt.xlabel("MAX_3")
plt.ylabel("ETR_3")
plt.title("MAX_3-ETR_3 Scatter Plot")


plt.subplot(2,4,4)
plt.scatter(beyda.MAX_4,beyda.ETR_4,color="red",label="beyda")
plt.scatter(alper.MAX_4,alper.ETR_4,color="blue",label="alper")
plt.scatter(ahmet.MAX_4,ahmet.ETR_4,color="green",label="ahmet")
plt.scatter(irem.MAX_4,irem.ETR_4,color="yellow",label="irem")

plt.legend()
plt.xlabel("MAX_4")
plt.ylabel("ETR_4")
plt.title("MAX_4-ETR_4 Scatter Plot")

plt.subplot(2,4,5)
plt.scatter(beyda.MAX_5,beyda.ETR_5,color="red",label="beyda")
plt.scatter(alper.MAX_5,alper.ETR_5,color="blue",label="alper")
plt.scatter(ahmet.MAX_5,ahmet.ETR_5,color="green",label="ahmet")
plt.scatter(irem.MAX_5,irem.ETR_5,color="yellow",label="irem")

plt.legend()
plt.xlabel("MAX_5")
plt.ylabel("ETR_5")
plt.title("MAX_5-ETR_5 Scatter Plot")

plt.subplot(2,4,6)
plt.scatter(beyda.MAX_6,beyda.ETR_6,color="red",label="beyda")
plt.scatter(alper.MAX_6,alper.ETR_6,color="blue",label="alper")
plt.scatter(ahmet.MAX_6,ahmet.ETR_6,color="green",label="ahmet")
plt.scatter(irem.MAX_6,irem.ETR_6,color="yellow",label="irem")

plt.legend()
plt.xlabel("MAX_6")
plt.ylabel("ETR_6")
plt.title("MAX_6-ETR_6 Scatter Plot")


plt.subplot(2,4,7)
plt.scatter(beyda.MAX_7,beyda.ETR_7,color="red",label="beyda")
plt.scatter(alper.MAX_7,alper.ETR_7,color="blue",label="alper")
plt.scatter(ahmet.MAX_7,ahmet.ETR_7,color="green",label="ahmet")
plt.scatter(irem.MAX_7,irem.ETR_7,color="yellow",label="irem")

plt.legend()
plt.xlabel("MAX_7")
plt.ylabel("ETR_7")
plt.title("MAX_7-ETR_7 Scatter Plot")

plt.subplot(2,4,8)
plt.scatter(beyda.MAX_8,beyda.ETR_8,color="red",label="beyda")
plt.scatter(alper.MAX_8,alper.ETR_8,color="blue",label="alper")
plt.scatter(ahmet.MAX_8,ahmet.ETR_8,color="green",label="ahmet")
plt.scatter(irem.MAX_8,irem.ETR_8,color="yellow",label="irem")

plt.legend()
plt.xlabel("MAX_8")
plt.ylabel("ETR_8")
plt.title("MAX_8-ETR_8 Scatter Plot")
plt.show()

f,ax = plt.subplots(figsize=(20, 10))
plt.subplot(2,4,1)
plt.scatter(beyda.MAX_1,beyda.VAR_1,color="red",label="beyda")
plt.scatter(alper.MAX_1,alper.VAR_1,color="blue",label="alper")
plt.scatter(ahmet.MAX_1,ahmet.VAR_1,color="green",label="ahmet")
plt.scatter(irem.MAX_1,irem.VAR_1,color="yellow",label="irem")

plt.legend()
plt.xlabel("MAX_1")
plt.ylabel("VAR_1")
plt.title("VAR_1-VAR_1 Scatter Plot")


plt.subplot(2,4,2)
plt.scatter(beyda.MAX_2,beyda.VAR_2,color="red",label="beyda")
plt.scatter(alper.MAX_2,alper.VAR_2,color="blue",label="alper")
plt.scatter(ahmet.MAX_2,ahmet.VAR_2,color="green",label="ahmet")
plt.scatter(irem.MAX_2,irem.VAR_2,color="yellow",label="irem")

plt.legend()
plt.xlabel("MAX_2")
plt.ylabel("VAR_2")
plt.title("MAX_2-VAR_2 Scatter Plot")


plt.subplot(2,4,3)
plt.scatter(beyda.MAX_3,beyda.VAR_3,color="red",label="beyda")
plt.scatter(alper.MAX_3,alper.VAR_3,color="blue",label="alper")
plt.scatter(ahmet.MAX_3,ahmet.VAR_3,color="green",label="ahmet")
plt.scatter(irem.MAX_3,irem.VAR_3,color="yellow",label="irem")

plt.legend()
plt.xlabel("MAX_3")
plt.ylabel("VAR_3")
plt.title("MAX_3-VAR_3 Scatter Plot")


plt.subplot(2,4,4)
plt.scatter(beyda.MAX_4,beyda.VAR_4,color="red",label="beyda")
plt.scatter(alper.MAX_4,alper.VAR_4,color="blue",label="alper")
plt.scatter(ahmet.MAX_4,ahmet.VAR_4,color="green",label="ahmet")
plt.scatter(irem.MAX_4,irem.VAR_4,color="yellow",label="irem")

plt.legend()
plt.xlabel("MAX_4")
plt.ylabel("VAR_4")
plt.title("MAX_4-VAR_4 Scatter Plot")

plt.subplot(2,4,5)
plt.scatter(beyda.MAX_5,beyda.VAR_5,color="red",label="beyda")
plt.scatter(alper.MAX_5,alper.VAR_5,color="blue",label="alper")
plt.scatter(ahmet.MAX_5,ahmet.VAR_5,color="green",label="ahmet")
plt.scatter(irem.MAX_5,irem.VAR_5,color="yellow",label="irem")

plt.legend()
plt.xlabel("MAX_5")
plt.ylabel("VAR_5")
plt.title("MAX_5-VAR_5 Scatter Plot")

plt.subplot(2,4,6)
plt.scatter(beyda.MAX_6,beyda.VAR_6,color="red",label="beyda")
plt.scatter(alper.MAX_6,alper.VAR_6,color="blue",label="alper")
plt.scatter(ahmet.MAX_6,ahmet.VAR_6,color="green",label="ahmet")
plt.scatter(irem.MAX_6,irem.VAR_6,color="yellow",label="irem")

plt.legend()
plt.xlabel("MAX_6")
plt.ylabel("VAR_6")
plt.title("MAX_6-VAR_6 Scatter Plot")


plt.subplot(2,4,7)
plt.scatter(beyda.MAX_7,beyda.VAR_7,color="red",label="beyda")
plt.scatter(alper.MAX_7,alper.VAR_7,color="blue",label="alper")
plt.scatter(ahmet.MAX_7,ahmet.VAR_7,color="green",label="ahmet")
plt.scatter(irem.MAX_7,irem.VAR_7,color="yellow",label="irem")

plt.legend()
plt.xlabel("MAX_7")
plt.ylabel("VAR_7")
plt.title("MAX_7-VAR_7 Scatter Plot")

plt.subplot(2,4,8)
plt.scatter(beyda.MAX_8,beyda.VAR_8,color="red",label="beyda")
plt.scatter(alper.MAX_8,alper.VAR_8,color="blue",label="alper")
plt.scatter(ahmet.MAX_8,ahmet.VAR_8,color="green",label="ahmet")
plt.scatter(irem.MAX_8,irem.VAR_8,color="yellow",label="irem")

plt.legend()
plt.xlabel("MAX_8")
plt.ylabel("VAR_8")
plt.title("MAX_8-VAR_8 Scatter Plot")
plt.show()

#%% histogram
f,ax = plt.subplots(figsize=(20, 5))
plt.subplot(1,4,1)
plt.hist(beyda.MAX_1,bins=20,color="red")
plt.legend()
plt.xlabel("Frekans")
plt.ylabel("MAX_1")
plt.title(" Beyda_MAX_1_Histogram Plot")

plt.subplot(1,4,2)
plt.hist(alper.MAX_1,bins=20,color="blue")
plt.xlabel("Frekans")
plt.ylabel("MAX_1")
plt.title(" Alper_MAX_1_Histogram Plot")

plt.subplot(1,4,3)
plt.hist(ahmet.MAX_1,bins=20,color="black")
plt.xlabel("Frekans")
plt.ylabel("MAX_1")
plt.title(" Ahmet_MAX_1_Histogram Plot")

plt.subplot(1,4,4)
plt.hist(irem.MAX_1,bins=20,color="yellow")
plt.xlabel("Frekans")
plt.ylabel("MAX_1")
plt.title(" Irem_MAX_1_Histogram Plot")
plt.show()
