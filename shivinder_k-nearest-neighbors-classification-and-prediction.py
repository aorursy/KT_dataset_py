import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

import random

from collections import Counter

from sklearn import preprocessing

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

from matplotlib import style

style.use('fivethirtyeight')





# read the data set

df=pd.read_csv('../input/Iris.csv')

#What's in the data set??

print(df.head(10))
#to plot data in the data set

#Figure contains all the plots

fig=plt.figure()

#Set the background of figure to be white in color

fig.patch.set_facecolor('White')

#Creating axes on which all the three group of species will be plotted

ax1=plt.subplot2grid((1,1),(0,0))

#setting the background of axes to be white in color

ax1.set_axis_bgcolor('White')



grpd=df.groupby('Species')



#Getting the group 'Iris-setosa' and plotting its values for 'SepalLengthCm' vs 'SepalWidthCm'

grpd.get_group('Iris-setosa').plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',ax=ax1,label='Iris-setosa')



#Plotting 'SepalLengthCm' vs 'SepalWidthCm' and 'PetalLengthCm' vs 'PetalWidthCm' for 'Iris-versicolor'

grpd.get_group('Iris-versicolor').plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',ax=ax1,color='g',label='Iris-versicolor')



#Plotting 'SepalLengthCm' vs 'SepalWidthCm' and 'PetalLengthCm' vs 'PetalWidthCm' for 'Iris-versicolor'

grpd.get_group('Iris-virginica').plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',ax=ax1,color='black',label='Iris-virginica')



ax1.legend(loc=4,fontsize ='xx-small',scatterpoints=2,shadow='True',title='Species')

plt.show()
fig=plt.figure()

#Set the background of figure to be white in color

fig.patch.set_facecolor('White')

#Creating axes on which all the three group of species will be plotted

ax1=plt.subplot2grid((1,1),(0,0))

#setting the background of axes to be white in color

ax1.set_axis_bgcolor('White')



grpd=df.groupby('Species')



#Plotting 'PetalLengthCm' vs 'PetalWidthCm' for 'Iris-setosa'

grpd.get_group('Iris-setosa').plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',ax=ax1,label='Iris-setosa')



#Plotting 'PetalLengthCm' vs 'PetalWidthCm' for 'Iris-versicolor'

grpd.get_group('Iris-versicolor').plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',ax=ax1,color='g',label='Iris-versicolor')



#Plotting 'PetalLengthCm' vs 'PetalWidthCm' for 'Iris-versicolor'

grpd.get_group('Iris-virginica').plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',ax=ax1,color='black',label='Iris-virginica')



ax1.legend(loc=4,fontsize ='xx-small',scatterpoints=2,shadow='True',title='Species')

plt.show()
# After visualising the data, classification is done using k nearest neighbor algorithm.



# First we will design our own K Nearest Neighbor Algorithm.

# Our data set Iris has three classes:

# 1:Iris-setosa

# 2:Iris-versicolor

# 3:Iris-virginica



# In this algorithm input parameter 'Eval_data' is in the form of python dictionary.

# K defines the number of nearest neighbors.K must be greater than the number of classes.

# And Pred_data is used to predict the class to which it belongs.



def algo_KNN(Eval_data,Pred_data,k):

    #First we will check that if K should not be less than the classes in the data set.

    #If it is, generate a warning.

    if len(Eval_data)>k:

        warning.warn('K should be greater than total voting groups')



    # K Nearest Neighbour uses eucledian distance to decide whether to which class the data set belongs



    Distance_from_group=[]

    #Calculates eucledian distance of the Pred_data from every data set in Eval_data

    for voting_group in Eval_data:

        for data in Eval_data[voting_group]:

            dis=np.linalg.norm(np.array(data)-np.array(Pred_data))

            Distance_from_group.append([dis,voting_group])

            

    #After obtaining the distances, fetch class for k shortest distances from the list DIstance_from_group

    vote=[i[1] for i in sorted(Distance_from_group)[:k]]

    

    #From these K shortest distances get the most frequently occuring class.

    #That class will be the class to which the data set belongs

    vote_res=Counter(vote).most_common(1)[0][0]

    

    return vote_res
#Now we will use the algorithm above on data set Iris and check its accuracy.

df=pd.read_csv('../input/Iris.csv')

#First we will drop the id column otherwise it will affect the classification.

df.drop(['Id'],1,inplace='True')

#Now we will convert our classes to numerial values to ease our algorithm.

le=preprocessing.LabelEncoder()

le.fit(['A','Iris-setosa','Iris-versicolor','Iris-virginica'])

df['Species']=le.transform(df['Species'])



#Shuffle our data set to create randomness in the data set

df=shuffle(df)



#Defining our test size to be 20% of the whole data base

test_size=0.2



#Creating 80% training data

train=df[:-int(test_size*len(df))]



#Creating 20% testing data

test=df[-int(test_size*len(df)):]



#Initialise train_set and test_set in the form of dictionaries

train_set={1:[],2:[],3:[]}

test_set={1:[],2:[],3:[]}



#Filling train_set dictionary with data from train

for i in range(len(train)):

    x=df.iloc[i].values.tolist()

    train_set[(x[-1])].append(x[:-1])

    

#Filling test_set dictionary with data from test        

for i in range(len(test)):

    x=df.iloc[i].values.tolist()

    test_set[(x[-1])].append(x[:-1])

        

        

count=0

total=0

ls=[]



#Passing the data from the test_set one by one for prediction

for i in test_set:

    for j in test_set[i]:

        vote=algo_KNN(train_set,j,k=5)

        ls.append([i,vote])

        if i==vote:

            count+=1

        

        total+=1

        

accuracy=(count/total)*100

accuracy = float("{0:.2f}".format(accuracy))

print('Accuracy is:',accuracy,'%')
#Let's predict using algorith above

#Take a sample data and try to determine its class

Sample_data=[5,2.1,6.3,2.5]

#Calling the algorithm function

res=algo_KNN(train_set,Sample_data,k=5)



print('Species is:',le.inverse_transform(res))



#Similarly u can create more data set to predict their species class.



#K Nearest Neighbor Algorithm works fine when data set is small or sufficiently moderate.

#But as the data set increases, algorithms ability falls and processing time increses significantly.
#Display results

random.shuffle(ls)

df1=pd.DataFrame(ls,columns=['Actual','Predicted'])

df1.plot(kind='Bar')

plt.tick_params( axis='x',which='both',bottom='off',top='off',labelbottom='off')

plt.yticks(df1.index,[' ','Iris-setosa','Iris-versicolor','Iris-virginica'])

plt.axis([-1,30,0,3])

plt.legend(loc=0,fontsize ='xx-small',shadow='True')

plt.show()


