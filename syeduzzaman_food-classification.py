import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn import datasets

from sklearn.cluster import KMeans

import sklearn.metrics as sm

from scipy.cluster.hierarchy import linkage, dendrogram

import os

# read csv file 

df = pd.read_csv("../input/food_coded.csv")



# data cleaning : drop the NA rows 

df=df.dropna()





# Set some pandas options

pd.set_option('display.notebook_repr_html', False)

pd.set_option('display.max_rows', 60)

pd.set_option('display.max_columns', 60)

pd.set_option('display.width', 1000)

 

%matplotlib inline

type(df)

print("Row: ",df.shape[0])

print("Column: ",df.shape[1])
### Counting Gender of students: 

#1 Gender count 

gend = df['Gender'].value_counts()

print(gend)
#2 student weight 

weight=df.weight # create pandas series from dataframe 

weight.describe() # print different aspects of data
#3 student recommendation on veggie  

veggies_day=df.veggies_day # create pandas series from dataframe 

veggies_day.describe() # print different aspects of data 
#4 student recommendation on Indian Food

indian_food=df.indian_food # create pandas series from dataframe 

indian_food.describe() # print different aspects of data 
from pandas import DataFrame, Series

df['weight']=df['weight'].apply(lambda x: 0 if str(type(x))=="<class 'str'>" else x) # non munerical sets as 0



Dendrogram_data = df[['indian_food','veggies_day','weight']]

from sklearn.metrics.pairwise import euclidean_distances

Dendrogram_data_D = DataFrame(euclidean_distances(Dendrogram_data))



from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list

Z = linkage(Dendrogram_data_D, 'average')

plt.figure(figsize=(10, 10))

D = dendrogram(Z=Z, orientation="right", leaf_font_size=10)

fig = plt.figure(figsize=(10,10))

df2=Dendrogram_data

plt.scatter(Dendrogram_data.indian_food, Dendrogram_data.weight, edgecolor = 'black', s = 80)

plt.title('Foods')

plt.xlabel('veggies day')

plt.ylabel('indain food')





centroid1 = np.array([2,4])

centroid2 = np.array([3, 3])

centroid3 = np.array([4, 2])



fig = plt.figure(figsize=(10,10))

plt.scatter(Dendrogram_data.veggies_day, Dendrogram_data.indian_food, edgecolor = 'black', s = 80)

plt.title('Foods')

plt.xlabel('veggies day')

plt.ylabel('indain food')

circle1 = plt.Circle(centroid1, radius=0.15, edgecolor = 'black', fc='yellow')

circle2 = plt.Circle(centroid2, radius=0.15, edgecolor = 'black', fc='red')

circle3 = plt.Circle(centroid3, radius=0.15, edgecolor = 'black', fc='green')

plt.gca().add_patch(circle1)

plt.gca().add_patch(circle2)

plt.gca().add_patch(circle3)





Dendrogram_data["Dist_C1"] = np.sqrt((Dendrogram_data.veggies_day - centroid1[0])**2 

                             + (Dendrogram_data.indian_food - centroid1[1])**2)

Dendrogram_data["Dist_C2"] = np.sqrt((Dendrogram_data.veggies_day - centroid2[0])**2 + 

                             (Dendrogram_data.indian_food - centroid2[1])**2)

Dendrogram_data["Dist_C3"] = np.sqrt((Dendrogram_data.veggies_day - centroid3[0])**2 + 

                             (Dendrogram_data.indian_food - centroid3[1])**2)



Dendrogram_data["Association"] = np.where(

    (Dendrogram_data.Dist_C1 < Dendrogram_data.Dist_C2) & 

    (Dendrogram_data.Dist_C1 < Dendrogram_data.Dist_C3), 1, 

     np.where((Dendrogram_data.Dist_C2 < Dendrogram_data.Dist_C1) & 

    (Dendrogram_data.Dist_C2 < Dendrogram_data.Dist_C3) , 2, 3))



fig = plt.figure(figsize=(10,10))

 

# Create a colormap

colormap = np.array(['black', 'yellow', 'red', 'green'])

 

plt.scatter(Dendrogram_data.veggies_day, Dendrogram_data.indian_food, 

            c=colormap[Dendrogram_data.Association], edgecolor = 'black', s = 80)

plt.title('Foods')

plt.xlabel('veggies day')

plt.ylabel('indain food')
centroid1 = Dendrogram_data[Dendrogram_data.Association == 1][["veggies_day", "indian_food"]].mean()

centroid2 = Dendrogram_data[Dendrogram_data.Association == 2][["veggies_day", "indian_food"]].mean()

centroid3 = Dendrogram_data[Dendrogram_data.Association == 3][["veggies_day", "indian_food"]].mean()



fig = plt.figure(figsize=(10,10))



# Create a colormap

colormap = np.array(['black', 'yellow', 'red', 'green'])

 

plt.scatter(Dendrogram_data.veggies_day, Dendrogram_data.indian_food, c=colormap[Dendrogram_data.Association], edgecolor = 'black', s = 80)



#plt.scatter(Dendrogram_data.fav_food,Dendrogram_data.pay_meal_out, edgecolor = 'black', s = 80)



plt.title('Foods')

plt.xlabel('veggies day')

plt.ylabel('indain food')

circle1 = plt.Circle(centroid1, radius=0.1, edgecolor = 'black', fc='yellow')

circle2 = plt.Circle(centroid2, radius=0.1, edgecolor = 'black', fc='red')

circle3 = plt.Circle(centroid3, radius=0.1, edgecolor = 'black', fc='green')

plt.gca().add_patch(circle1)

plt.gca().add_patch(circle2)

plt.gca().add_patch(circle3)





Dendrogram_data["Dist_C1"] = np.sqrt((Dendrogram_data.veggies_day - centroid1[0])**2 + (Dendrogram_data.indian_food - centroid1[1])**2)

Dendrogram_data["Dist_C2"] = np.sqrt((Dendrogram_data.veggies_day - centroid2[0])**2 + (Dendrogram_data.indian_food- centroid2[1])**2)

Dendrogram_data["Dist_C3"] = np.sqrt((Dendrogram_data.veggies_day - centroid3[0])**2 + (Dendrogram_data.indian_food - centroid3[1])**2)

Dendrogram_data["Association"] = np.where((Dendrogram_data.Dist_C1 < Dendrogram_data.Dist_C2) & 

                                  (Dendrogram_data.Dist_C1 < Dendrogram_data.Dist_C3), 1, 

                                   np.where((Dendrogram_data.Dist_C2 < Dendrogram_data.Dist_C1) & 

                                  (Dendrogram_data.Dist_C2 < Dendrogram_data.Dist_C3) , 2, 3))

fig = plt.figure(figsize=(10,10))

 

# Create a colormap

colormap = np.array(['black', 'yellow', 'red', 'green'])

# Plot Sepal

plt.scatter(Dendrogram_data.veggies_day,Dendrogram_data.indian_food, c=colormap[Dendrogram_data.Association], edgecolor = 'black', s = 80)

plt.title('Foods')

plt.xlabel('veggies day')

plt.ylabel('indain food')

circle1 = plt.Circle(centroid1, radius=0.1, edgecolor = 'black', fc='yellow')

circle2 = plt.Circle(centroid2, radius=0.1, edgecolor = 'black', fc='red')

circle3 = plt.Circle(centroid3, radius=0.1, edgecolor = 'black', fc='green')

plt.gca().add_patch(circle1)

plt.gca().add_patch(circle2)

plt.gca().add_patch(circle3)

centroid1 = Dendrogram_data[Dendrogram_data.Association == 1][["veggies_day", "indian_food"]].mean()

centroid2 = Dendrogram_data[Dendrogram_data.Association == 2][["veggies_day", "indian_food"]].mean()

centroid3 = Dendrogram_data[Dendrogram_data.Association == 3][["veggies_day", "indian_food"]].mean()

Dendrogram_data["Dist_C1"] = np.sqrt((Dendrogram_data.veggies_day - centroid1[0])**2 + (Dendrogram_data.indian_food - centroid1[1])**2)

Dendrogram_data["Dist_C2"] = np.sqrt((Dendrogram_data.veggies_day - centroid2[0])**2 + (Dendrogram_data.indian_food - centroid2[1])**2)

Dendrogram_data["Dist_C3"] = np.sqrt((Dendrogram_data.veggies_day - centroid3[0])**2 + (Dendrogram_data.indian_food - centroid3[1])**2)

Dendrogram_data["Association"] = np.where((Dendrogram_data.Dist_C1 < Dendrogram_data.Dist_C2) & 

                                  (Dendrogram_data.Dist_C1 < Dendrogram_data.Dist_C3), 1, 

                                   np.where((Dendrogram_data.Dist_C2 < Dendrogram_data.Dist_C1) & 

                                  (Dendrogram_data.Dist_C2 < Dendrogram_data.Dist_C3) , 2, 3))

fig = plt.figure(figsize=(10,10))

 

# Create a colormap

colormap = np.array(['black', 'yellow', 'red', 'green'])

 

# Plot Sepal

plt.scatter(Dendrogram_data.veggies_day, Dendrogram_data.indian_food, c=colormap[Dendrogram_data.Association], edgecolor = 'black', s = 80)

plt.title('Foods')

plt.xlabel('veggies day')

plt.ylabel('indain food')

circle1 = plt.Circle(centroid1, radius=0.1, edgecolor = 'black', fc='yellow')

circle2 = plt.Circle(centroid2, radius=0.1, edgecolor = 'black', fc='red')

circle3 = plt.Circle(centroid3, radius=0.1, edgecolor = 'black', fc='green')

plt.gca().add_patch(circle1)

plt.gca().add_patch(circle2)

plt.gca().add_patch(circle3)
centroid1 = Dendrogram_data[Dendrogram_data.Association == 1][["veggies_day", "indian_food"]].mean()

centroid2 = Dendrogram_data[Dendrogram_data.Association == 2][["veggies_day", "indian_food"]].mean()

centroid3 = Dendrogram_data[Dendrogram_data.Association == 3][["veggies_day", "indian_food"]].mean()

Dendrogram_data["Dist_C1"] = np.sqrt((Dendrogram_data.veggies_day - centroid1[0])**2 + (Dendrogram_data.indian_food - centroid1[1])**2)

Dendrogram_data["Dist_C2"] = np.sqrt((Dendrogram_data.veggies_day - centroid2[0])**2 + (Dendrogram_data.indian_food - centroid2[1])**2)

Dendrogram_data["Dist_C3"] = np.sqrt((Dendrogram_data.veggies_day - centroid3[0])**2 + (Dendrogram_data.indian_food - centroid3[1])**2)

Dendrogram_data["Association"] = np.where((Dendrogram_data.Dist_C1 < Dendrogram_data.Dist_C2) & 

                                  (Dendrogram_data.Dist_C1 < Dendrogram_data.Dist_C3), 1, 

                                   np.where((Dendrogram_data.Dist_C2 < Dendrogram_data.Dist_C1) & 

                                  (Dendrogram_data.Dist_C2 < Dendrogram_data.Dist_C3) , 2, 3))

fig = plt.figure(figsize=(10,10))

 

# Create a colormap

colormap = np.array(['black', 'yellow', 'red', 'green'])

 

# Plot Sepal

plt.scatter(Dendrogram_data.veggies_day, Dendrogram_data.indian_food, c=colormap[Dendrogram_data.Association], edgecolor = 'black', s = 80)

plt.title('Foods')

plt.xlabel('veggies day')

plt.ylabel('indain food')

circle1 = plt.Circle(centroid1, radius=0.1, edgecolor = 'black', fc='yellow')

circle2 = plt.Circle(centroid2, radius=0.1, edgecolor = 'black', fc='red')

circle3 = plt.Circle(centroid3, radius=0.1, edgecolor = 'black', fc='green')

plt.gca().add_patch(circle1)

plt.gca().add_patch(circle2)

plt.gca().add_patch(circle3)
centroid1 = Dendrogram_data[Dendrogram_data.Association == 1][["veggies_day", "indian_food"]].mean()

centroid2 = Dendrogram_data[Dendrogram_data.Association == 2][["veggies_day", "indian_food"]].mean()

centroid3 = Dendrogram_data[Dendrogram_data.Association == 3][["veggies_day", "indian_food"]].mean()

Dendrogram_data["Dist_C1"] = np.sqrt((Dendrogram_data.veggies_day - centroid1[0])**2 + (Dendrogram_data.indian_food - centroid1[1])**2)

Dendrogram_data["Dist_C2"] = np.sqrt((Dendrogram_data.veggies_day - centroid2[0])**2 + (Dendrogram_data.indian_food - centroid2[1])**2)

Dendrogram_data["Dist_C3"] = np.sqrt((Dendrogram_data.veggies_day - centroid3[0])**2 + (Dendrogram_data.indian_food - centroid3[1])**2)

Dendrogram_data["Association"] = np.where((Dendrogram_data.Dist_C1 < Dendrogram_data.Dist_C2) & 

                                  (Dendrogram_data.Dist_C1 < Dendrogram_data.Dist_C3), 1, 

                                   np.where((Dendrogram_data.Dist_C2 < Dendrogram_data.Dist_C1) & 

                                  (Dendrogram_data.Dist_C2 < Dendrogram_data.Dist_C3) , 2, 3))

fig = plt.figure(figsize=(10,10))

 

# Create a colormap

colormap = np.array(['black', 'yellow', 'red', 'green'])

 

# Plot Sepal

plt.scatter(Dendrogram_data.veggies_day, Dendrogram_data.indian_food, c=colormap[Dendrogram_data.Association], edgecolor = 'black', s = 80)

plt.title('Foods')

plt.xlabel('veggies day')

plt.ylabel('indain food')
df2 = df[['veggies_day','indian_food']]

df3 = df[['veggies_day','indian_food','weight']]



# K Means Cluster

model = KMeans(n_clusters = 3)

model.fit(df2)



# K Means Cluster

model = KMeans(n_clusters = 3)

model.fit(df2)



fig = plt.figure(figsize=(10, 10))

 

# Create a colormap

colormap = np.array(['green', 'red','yellow'])

 

plt.scatter(df2.veggies_day, df2.indian_food, c=colormap[model.labels_], edgecolor = 'black', s = 80)

plt.title('K Mean Classification')

plt.xlabel('veggies day')

plt.ylabel('indain food')
def plotMesh():

    h = 0.1

    # Create color maps

    cmap_light = ListedColormap(['#ffffcc', '#ff8080','#ccd9ff'])

    colormap = np.array(['yellow', 'red', 'blue'])



    x_min, x_max = df2.veggies_day.min() - 1, df2.veggies_day.max() + 1

    y_min, y_max = df2.indian_food.min() - 1, df2.indian_food.max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])



    # Put the result into a color plot

    Z = Z.reshape(xx.shape)

    fig = plt.figure()

    plt.pcolormesh(xx, yy, Z, cmap = cmap_light)



    # Plot also the training points

    plt.scatter(df2.veggies_day, df2.indian_food, c = colormap[model.labels_], edgecolor = 'black', s = 120)

    plt.xlim(xx.min(), xx.max())

    plt.ylim(yy.min(), yy.max())

    plt.title("4-Class classification \n(k = %i)\n\ Loan 1 - Yellow, Loan 2 - Red, Loan 3 - Blue, Loan 4 - green" % (5))

    plt.xlabel('veggies_day')

    plt.ylabel('indian_food')





    

    

plotMesh()
from sklearn.cluster import KMeans

model = KMeans(n_clusters = 3)

model.fit(df3)



def pred():

        colormap = np.array(['yellow', 'red', 'blue','magenta','black'])



        veggies_day = float(input('veggies day: '))

        indian_food = float(input('indian_food: '))

        weight = float(input('weight: '))



        data_class = model.predict(np.array([veggies_day, indian_food,weight]).reshape(1,-1))[0]

        if colormap[data_class]=='yellow':

        #Student likes the veggie food too much. Based on the cluster, he probably likes the Indian food to

            print("  You will get $2 coupon ")

            

        if colormap[data_class]=='red':

        #Student does not like the veggie food too much. Based on the cluster, he probably does not likes the Indian food to

            print("  You will get $10 coupon ")   

            

        if colormap[data_class]=='blue':

        #Student likes the veggie food too much. Based on the cluster, he probably does not likes the Indian food to

            print("  You will get $5 coupon ") 

pred()