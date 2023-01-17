# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import sys
import os
import math as math
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/WOE_data.csv')
df.head(3)
ServiceType	 = df['ServiceType']
ServiceStatus = df['ServiceStatus']
print (df['ServiceType'].head(3))
print (df['ServiceStatus'].head(3))
def calculate_sum(name):
    name = (ServiceStatus == name)
    return name.sum()

Complete_of_all = calculate_sum(name = 'Complete')
Quit_of_all = calculate_sum(name = 'Quit')
print("Number of complete is "+str(Complete_of_all))
print("Number of quiz is "+str(Quit_of_all))
B = df[df.ServiceType == 'B']
B[ServiceStatus == 'Complete']
df[(df.ServiceType == 'B')&(ServiceStatus == 'Complete')].head(3)

#Calculate sum of one type
def calculate_bool_sum(Type,Status):
    job = (df.ServiceType == Type)&(ServiceStatus == Status)
    return job.sum()


#Calculate sum of each type
def calculate_each_sum(Types):
    for x in Types:
        x_complete = calculate_bool_sum(Type =x,Status = 'Complete')
        x_quit = calculate_bool_sum(Type =x,Status = 'Quit')
        outcome = f'Complete of {x} is {x_complete}   Quit of {x} is {x_quit}\n'
        print(outcome)
i = 0   
Types = ['B','C','E','H','I','M','N','S']
calculate_each_sum(Types)
WOE_IV = pd.DataFrame(index=Types, columns = ['WOE', 'IV'])

def calculate_WOE_IV(Types):
    i = 0
    for x in Types:
        x_complete = calculate_bool_sum(Type =x,Status = 'Complete')
        x_quit = calculate_bool_sum(Type =x,Status = 'Quit')
        x_com_pro = x_complete / Complete_of_all
        x_quit_pro = x_quit / Quit_of_all
        WOE_x = math.log(x_com_pro/x_quit_pro)
        IV_x = (x_com_pro - x_quit_pro)*WOE_x
        #Add WOE&IV to dataframe
        WOE_IV['WOE'].iloc[[i]] = WOE_x
        WOE_IV['IV'].iloc[[i]] = IV_x
        i=i+1

calculate_WOE_IV(Types)
WOE_IV
WOE_IV.to_csv('WOE_IV.csv',index=False)
