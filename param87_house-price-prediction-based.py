# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



#read the data from the kc_house_data and drop the columns which are not needed



dataFrame = pd.read_csv('../input/kc_house_data.csv',nrows=1000)#read the CSV file only 1000 dataset



Cols = ['price','sqft_living'] #these are the the columns which are needed

dataFrame = dataFrame[Cols] #consider only those columns which are required and drop the other columns

dataFrame[['price']] = dataFrame[['price']]/1000



print(dataFrame.head())#print the data



print('no of dataset:',len(dataFrame))#no of dataset



#data_points = dataFrame.as_matrix() #conver the data to the matrix form



#simply plotting of data in 2d form

plt.scatter(dataFrame['sqft_living'],dataFrame['price'])

plt.title(' sqft_living vs price ')

plt.xlabel('sqft_living area')

plt.ylabel('price k $')

plt.show()



#b,m are the constant of equation  linear rgression y = m*x +b

init_consts = np.array([0,0])#inital parameter of best fit which is assign to b=0,m=0

criteria = 8000

epsi = 1e-5 #epsilon 



N = len(dataFrame.index)#length of dataset

total_living =  sum(dataFrame['sqft_living'])#sum of all sqft_living

sq_total_living = sum(np.power(dataFrame['sqft_living'],2))# sum of sqft_living^2



#Initialize hessian matrix

H = [[-N,-total_living],[-total_living,-sq_total_living]]



#update newton method to give new points

def newton_method_update(old_consts, H, J):

    new_consts = np.array(np.subtract(old_consts, np.dot(np.linalg.pinv(H),J)))

    

    return new_consts

    

price = np.array(dataFrame['price'])#conver to array

living_sqft = np.array(dataFrame['sqft_living'])#conver to array



new_consts = init_consts#initialie new parameter



#this condition for looping

while criteria > epsi:

    old_consts = new_consts

    

    J_position1 = np.nansum(price) - N * old_consts[0] - total_living * old_consts[1]

    J_position2 = np.nansum(price * living_sqft) - total_living * old_consts[0] - sq_total_living * old_consts[1]

    J = np.array([J_position1,J_position2])

    

    new_consts = newton_method_update(old_consts, H, J)

    

    criteria = np.linalg.norm(new_consts - old_consts)#criteria check every time for looping

    

#this is point obtains which of best fit

#were m = new_points[1] and b=new_points[0]

#

print(new_consts)





#plot the line of best fit

plt.plot(price, new_consts[1] * price + new_consts[0],'red')

#data with respect to sqft_living vs price

plt.scatter(dataFrame['sqft_living'],dataFrame['price'],)

plt.title(' sqft_living vs price ')

plt.xlabel('sqft_living area')

plt.ylabel('price k $')

plt.show()