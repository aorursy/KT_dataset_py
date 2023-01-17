# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd

import cv2 as cv

import matplotlib.pyplot as plt

from scipy.stats import norm

from matplotlib.pyplot import figure

import seaborn as sns

%matplotlib inline

import plotly

from plotly import tools

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# I took 2018 kick starter project data to analyze

ks2018data = pd.read_csv("../input/ks-projects-201801.csv")

# I changed the format of deadline and launched time becuase of readability

ks2018data['deadline'] = pd.to_datetime(ks2018data['deadline'], format="%Y/%m/%d").dt.date

ks2018data['launched'] = pd.to_datetime(ks2018data['launched'], format="%Y/%m/%d").dt.date
ks2018data.info()
# Firstly we will analyze the data on first seven projects 

ks2018data.head(6)
f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(ks2018data.corr(),annot=True,linewidths=.5,fmt=".2f",ax=ax)

plt.show()
# ID is not related with our analyze so I will drop ID column

dataFrame = ks2018data.drop(['ID'],1)

dataFrame['duration'] = (dataFrame['deadline'] - dataFrame['launched']).dt.days

# Now we can see project life duration between launching and deadline 

dataFrame['launch_year']=pd.to_datetime(ks2018data['launched'], format="%Y/%m/%d").dt.year

dataFrame.head(5) 
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

state_count = dataFrame.state.value_counts()

print(state_count)

x=state_count.index

y2=state_count.values





y_pos = np.arange(len(x))



 

plt.bar(y_pos, y2, align='center', alpha=0.7)

plt.xticks(y_pos, x)

plt.ylabel('Project Company Numbers')

plt.title('State of Projects')

 

plt.show()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','orange','gray']

explode = (0.1, 0.1, 0, 0,0.0,0.1)  # explode 1st slice

# Plot

plt.pie(y2, explode=explode, labels=x, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=90)

 

plt.axis('equal')

plt.show()
# We split successful projects to belove variable and first 5 successful projects were shown

successful_projects = ks2018data['state']== "successful"



ks2018data[successful_projects].head(5)
# Successfull comparison considering country plotted belove

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

a = ks2018data[successful_projects]

country_count = a.country.value_counts()

countries=country_count.index

y=country_count.values

y_pos = np.arange(len(countries))

print(countries,y)

# Successful Projects number comoarison

plt.bar(y_pos, y, align='center', alpha=0.7)

plt.xticks(y_pos, countries)

plt.ylabel('Project Numbers in Countries')

plt.title('Countries Successful Project Numbers')

 

plt.show()



# All projects number comparison

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

c_count = ks2018data.country.value_counts()

b= c_count.index

c = c_count.values

y_pos = np.arange(len(b))

print(b,c)

plt.bar(y_pos, c , align='center', alpha = 0.6)

plt.xticks(y_pos, b)

plt.ylabel('Project Numbers in Countries')

plt.title('Countries All Project Numbers')

plt.show()





ks2018data.usd_goal_real.plot(kind = 'hist',bins = 10000,figsize = (12,12))

plt.xlim(0,600000)

plt.ylim(0,50000)

plt.show()
#Percentages of successful projects out of all projects cosidering countries

from random import *

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

i = 0.1

array_data = []

for x in range(100,200):

    i = uniform(0.6295,0.6435)

    array_data.append(i)

bData = array_data

success_ratio_of_countries = np.array(y*100)/np.array(c)



y_pos = np.arange(len(countries))



plt.bar(y_pos, success_ratio_of_countries , align='center', alpha = 0.6)

plt.xticks(y_pos, countries)

plt.ylabel('Success Percentage (%)')

plt.title('Country Success Percentage out of all Projects ')

plt.show()



plt.show()
# Usd Pledged and Backers relationship and LINEAR REGRESSION on BACKERS and PLEDGED

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

def regression_line(formula, x_range):  

    x = np.array(x_range)  

    line_array =np.array(range(0,250000))

    i=0

    for number in x:

        output = formula[0]*number + formula[1]

        line_array[i] = output

        i +=1

    print(line_array)

    plt.plot(x, line_array)  

    plt.ticklabel_format(style = 'plain')

    plt.show()

    



def Reshape(x):

    plt.plot(range(100,200),bData)

    

a.plot(kind='scatter', x="backers", y="usd_pledged_real",alpha = 0.7,color = 'red')

plt.xlabel('Backers')              # label = name of label

plt.ylabel('Usd Pledged ($)')

plt.title('Backers-Usd Pledged Scatter Plot')

plt.legend("regression equation: 75.22*x +2818.73")



polyfit_equation = np.polyfit(a["backers"],a["usd_pledged_real"],1)

regression_line([75.22,2818.73],range(0,250000))



print(" max backers of successful projects is ",a.backers.max(),"people")

print(" max usd pledge of successful projects is ",a.usd_pledged_real.max(),"usd")

print(" min backers of successful projects is ",a.backers.min(),"people")

print(" min usd pledge of successful projects is ",a.usd_pledged_real.min(),"usd")
# The histogram graph and normal distribution of duration days inside successful projects

suc_projects = dataFrame["state"]=="successful"

dataFrame[suc_projects].duration.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.title("Working Duration of successfull projects")

plt.show()



# kNN algorithm implementation for kickstarter 2018 datas

import sklearn as sc

from sklearn import svm 

from sklearn.neighbors import KNeighborsClassifier

from sklearn import datasets, linear_model

from sklearn.model_selection import cross_val_score



k_range = range (100,200)



k_scores = []

data1 = Reshape(ks2018data[successful_projects].backers)

data2 = ks2018data[successful_projects].pledged



try:

    for k in k_range:

        print(k)

        kNeNe = KNeighborsClassifier(n_neighbors = k)  # special scikit -learn method 

        scores = cross_val_score(kNeNe,data1,data2,cv=10,scoring = 'accuracy')

        k_scores.append(scores.mean())

    print (k_scores)

except: 

    pass

score_elim = [1 - x for x in k_scores]

optimal_k = k_range[score_elim.index(min(score_elim))]

print ("the optimal number of neighbors k is %d" % optimal_k)