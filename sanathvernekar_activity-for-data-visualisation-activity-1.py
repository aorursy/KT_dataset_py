

import numpy as np

import pandas as pd

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#Author :Sanath vernekar

'''

Problem Statement:-

A TV cable service provider has 170 customers distributed over 8 km radius. The service provider

wishes to restrict his services over 2km radius and retain maximum number of customers possible.

The remaining customers will be transferred to other service providers. Estimate the new position

of the base station (Assuming the current position of the service provider as (0,0)), and the

customers who can avail the services by the service provider. The data of customer ID and their

corresponding position is provided.



Tasks:

Design an algorithm to generate a list of customers to be retained and the list of customers to

be transferred.

Plot the graph of initial data and processed data, so that we can present the visualization of

the developed algorithm to the service provider.



'''

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os



#reading data



dfs=pd.read_excel("../input/Activity_1_Data.xlsx",sheetname="Sheet1")



#Here dfs is the dataframeset

#To see the info of the data frame 

dfs.info()

dfs.columns

#To know the labels of the data set 
dfs.head(5)

#to know top 5 samples of the data
#Extract the individual columns into X and Y arrays

x=np.array(dfs.iloc[0:,1])

#print(x)

#print(len(x))

y=np.array(dfs.iloc[0:,2])

#print(y)

#print(len(y))

print("Data Extracted")
#Initially plot the Scatter plot for the given data

print("Initial Plot of Customer Distribution")

plt.figure(1)

plt.scatter(x,y)

plt.show()
#Calculate Mean of X and Y

meanx=np.mean(x)

print("Mean of x data",meanx)

meany=np.mean(y)

print("Mean of Y data ",meany)
#calculate Median for the X and Y

medianx=np.median(x)

print("Median of X data",medianx)

mediany=np.median(y)

print("Median of Y data",mediany)
# Calculate Standard Deviation and Variance of X and Y

stdx=np.std(x)

print("Standard deviation of X is",stdx)

print("Variance of X is ",stdx*stdx)

stdy=np.std(y)

print("Standard deviation of Y is ",stdy)

print("Variance of Y is ",stdy*stdy)
#declaration of Distance formula

def dist(x,y):

    res=np.sqrt((x*x)+(y*y))

    return(res)
#since much of the data lies in the centre i.e 150/170 points lie at the centre therefore

#It would be better if we consider Median rather than mean ,since we have 20 outliers here

#Here in our case these outliers are far from the centre ,so we will consider Median

# Here if we consider even mean also it doesnot effect much ,since we have more points concentrated near centre

#Here iam considering median as the baseLocation

baseradius=dist(medianx,mediany)

print("Our New BaseLocation should be  X= ",medianx," Y=",mediany,",to gain maximum customers")

temp=dist((medianx),(mediany))

print("Distance from initial location to New base location is",temp)

#Here in our case Initial Location is (0,0)
#Retaining Maximum Customers within redius of 2 km

custx=[]

custy=[]

clist=[]

#myarr=[]

print("\n")

l=len(x)

ncust=np.arange(1,l+1)



for i in range(l):

    xa=x[i]

    ya=y[i]

    temp=dist((medianx-xa),(mediany-ya))

    if(temp<2):

        clist.append(i+1)

        custx.append(xa)

        custy.append(ya)



tcust=list(set(ncust)-set(clist))

tcust=sorted(tcust)

#Here tcust is the array containing list of customers id ,who doesnot come undersTV operator's service

tcust=np.array(tcust)

tcustlen=len(tcust)





clist=np.array(clist)

#Here Custx and custy are the locations of Customers within 2km radius from the baseLocation in x and y axis

custx=np.array(custx)

custy=np.array(custy)

#myarr=np.array(myarr)





print("Retained Customers Graph")

plt.figure(2)

plt.scatter(custx,custy)

plt.show()



#Plot of Customers who doesnot come under 2km radius

#Transferred customers plot

print("transferred Customers Plot")

tcustx=[]

tcusty=[]

for i in range(tcustlen):

    

    tcustx.append(x[tcust[i]-1])

    tcusty.append(y[tcust[i]-1])



plt.figure(3)

plt.scatter(tcustx,tcusty)

plt.show()



#To save the arrays of the customers who are within 2km radius and the customers out of the range

#we use pandas library's inbuilt function(.to_excel) to save them to .xlsx files



df=pd.DataFrame(clist)

try:

    df.to_excel("Activity_1_Retain.xlsx", sheet_name='Retained Customers',index=False)

    print("Activity_1_Retain.xlsx file is successfully saved in ",os.getcwd(),"directory")

except:

    print("Could not save file")

df=pd.DataFrame(tcust)

try:

    df.to_excel("Activity_1_Transfer.xlsx", sheet_name='Transferred Customers',index=False)

    print("Activity_1_Transfer.xlsx file is successfully saved in ",os.getcwd(),"directory")

except:

    print("Could not save file")
#Finally Plot all the graphs for Comparision

print("Initial Plot of Customer Distribution")

plt.figure(1)

plt.scatter(x,y)

plt.show()

print("Retained Customers plot")

plt.figure(2)

plt.scatter(custx,custy)

plt.show()

print("transferred Customers Plot")

plt.figure(3)

plt.scatter(tcustx,tcusty)

plt.show()







#Thank You


