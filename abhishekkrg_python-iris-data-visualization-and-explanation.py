#Importing Package 

import sys

print('Python: {}'.format(sys.version))

import numpy

import pandas

from pandas.tools.plotting import scatter_matrix

import matplotlib.pyplot as plt

import numpy as np

import warnings
#Reading IRIS Dataset in Pandas Dataframe

url="../input/Iris.csv"

dataset = pandas.read_csv(url)

#Printing DataFrame -only first 20 rows to understand what data look like

print(dataset.head(20))

#Data have 5 Columns ,First four are features and fifth is Classfication of the Iris type
#findout unique classification/type of iris flower.

dataset['Species'].unique()
#findout no of rows for each Species.

print(dataset.groupby('Species').size())
#Create 3 DataFrame for each Species

setosa=dataset[dataset['Species']=='Iris-setosa']

versicolor =dataset[dataset['Species']=='Iris-versicolor']

virginica =dataset[dataset['Species']=='Iris-virginica']



print(setosa.describe())

print(versicolor.describe())

print(virginica.describe())
print(dataset.describe())

#count tells that all the 4 features have 150 rows

#in general ,From Mean we can say that sepal is larger than petal.
#Plotting Petal Length vs Petal Width & Sepal Length vs Sepal width

#warnings.simplefilter("ignore")#Supress any warning

plt.figure()

fig,ax=plt.subplots(1,2,figsize=(17, 9))

dataset.plot(x="SepalLengthCm",y="SepalWidthCm",kind="scatter",ax=ax[0],sharex=False,sharey=False,label="sepal",color='r')

dataset.plot(x="PetalLengthCm",y="PetalWidthCm",kind="scatter",ax=ax[1],sharex=False,sharey=False,label="petal",color='b')

ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')

ax[1].set(title='Petal Comparasion',  ylabel='petal-width')

ax[0].legend()

ax[1].legend()

# plt.show()

# plt.close()



# we can see that  there are some petals which are smaller than rest of petal.

#Let's examine them
#for each Species ,let's check what is petal and sepal distibutuon

plt.figure()

fig,ax=plt.subplots(1,2,figsize=(21, 10))



setosa.plot(x="SepalLengthCm", y="SepalWidthCm", kind="scatter",ax=ax[0],label='setosa',color='r')

versicolor.plot(x="SepalLengthCm",y="SepalWidthCm",kind="scatter",ax=ax[0],label='versicolor',color='b')

virginica.plot(x="SepalLengthCm", y="SepalWidthCm", kind="scatter", ax=ax[0], label='virginica', color='g')



setosa.plot(x="PetalLengthCm", y="PetalWidthCm", kind="scatter",ax=ax[1],label='setosa',color='r')

versicolor.plot(x="PetalLengthCm",y="PetalWidthCm",kind="scatter",ax=ax[1],label='versicolor',color='b')

virginica.plot(x="PetalLengthCm", y="PetalWidthCm", kind="scatter", ax=ax[1], label='virginica', color='g')



ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')

ax[1].set(title='Petal Comparasion',  ylabel='petal-width')

ax[0].legend()

ax[1].legend()

# plt.show()

# plt.close()



#satosa   - satosa Petal are relatively smaller than rest of species .can be easily separable from rest of Species 

#versicolor & virginica are also separable in Petal comprasion

#satoa sepal are smallest in length and largest in Width than other species


###############################Histogram# Plot all feature for all species###########################################################################

plt.figure()



fig,ax=plt.subplots(4,3,figsize=(17, 8))

setosa["SepalLengthCm"].plot(kind="hist", ax=ax[0][0],label="setosa",color ='r',fontsize=10)

versicolor["SepalLengthCm"].plot(kind="hist", ax=ax[0][1],label="versicolor",color='b',fontsize=10)

virginica["SepalLengthCm"].plot( kind="hist",ax=ax[0][2],label="virginica",color='g',fontsize=10)



setosa["SepalWidthCm"].plot(kind="hist", ax=ax[1][0],label="setosa",color ='r',fontsize=10)

versicolor["SepalWidthCm"].plot(kind="hist", ax=ax[1][1],label="versicolor",color='b',fontsize=10)

virginica["SepalWidthCm"].plot( kind="hist",ax=ax[1][2],label="virginica",color='g',fontsize=10)



setosa["PetalLengthCm"].plot(kind="hist", ax=ax[2][0],label="setosa",color ='r',fontsize=10)

versicolor["PetalLengthCm"].plot(kind="hist", ax=ax[2][1],label="versicolor",color='b',fontsize=10)

virginica["PetalLengthCm"].plot( kind="hist",ax=ax[2][2],label="virginica",color='g',fontsize=10)





setosa["PetalWidthCm"].plot(kind="hist", ax=ax[3][0],label="setosa",color ='r',fontsize=10)

versicolor["PetalWidthCm"].plot(kind="hist", ax=ax[3][1],label="versicolor",color='b',fontsize=10)

virginica["PetalWidthCm"].plot( kind="hist",ax=ax[3][2],label="virginica",color='g',fontsize=10)



plt.rcParams.update({'font.size': 10})

plt.tight_layout()



ax[0][0].set(title='SepalLengthCm')

ax[0][1].set(title='SepalLengthCm')

ax[0][2].set(title='SepalLengthCm')

ax[1][0].set(title='SepalWidthCm ')

ax[1][1].set(title='SepalWidthCm ')

ax[1][2].set(title='SepalWidthCm ')

ax[2][0].set(title='PetalLengthCm')

ax[2][1].set(title='PetalLengthCm ')

ax[2][2].set(title='PetalLengthCm')

ax[3][0].set(title='PetalWidthCm')

ax[3][1].set(title='PetalWidthCm')

ax[3][2].set(title='PetalWidthCm')



ax[0][0].legend()

ax[0][1].legend()

ax[0][2].legend()

ax[1][0].legend()

ax[1][1].legend()

ax[1][2].legend()

ax[2][0].legend()

ax[2][1].legend()

ax[2][2].legend()

ax[3][0].legend()

ax[3][1].legend()

ax[3][2].legend()





plt.show()

plt.close()







###########################################################################################################