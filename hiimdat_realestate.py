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
data = pd.read_csv("/kaggle/input/real-estate-price-prediction/Real estate.csv")
data.info()
data.head()
data = data.drop("No", axis = 1)
import matplotlib.pyplot as plt



#fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplot(3)



plt.figure(figsize = (20,35))

plt.subplot(6,2,1)

plt.title("X1 transaction date vs Y")

plt.xlabel("X1 transaction date")

plt.ylabel("Y house price of unit area")

plt.scatter(data["X1 transaction date"], data["Y house price of unit area"])

plt.plot()



plt.subplot(6,2,2)

plt.title("X2 house age vs Y")

plt.xlabel("X2 house age")

plt.ylabel("Y house price of unit area")

plt.scatter(data["X2 house age"], data["Y house price of unit area"])

plt.plot()



plt.subplot(6,2,3)

plt.title("X3 distance to the nearest MRT station vs Y")

plt.xlabel("X3 distance to the nearest MRT station")

plt.ylabel("Y house price of unit area")

plt.scatter(data["X3 distance to the nearest MRT station"], data["Y house price of unit area"])

plt.plot()



plt.subplot(6,2,4)

plt.title("X4 number of convenience stores vs Y")

plt.xlabel("X4 number of convenience stores")

plt.ylabel("Y house price of unit area")

plt.scatter(data["X4 number of convenience stores"], data["Y house price of unit area"])

plt.plot()



plt.subplot(6,2,5)

plt.title("X5 latitude vs Y")

plt.xlabel("X5 latitude")

plt.ylabel("Y house price of unit area")

plt.scatter(data["X5 latitude"], data["Y house price of unit area"])

plt.plot()



plt.subplot(6,2,6)

plt.title("X6 longitude vs Y")

plt.xlabel("X6 longitude")

plt.ylabel("Y house price of unit area")

plt.scatter(data["X6 longitude"], data["Y house price of unit area"])

plt.plot()
for indx,col in zip(range(0, len(data.columns)), data.columns):

    plt.title(data.columns[indx])

    plt.boxplot(data[col], vert = False)

    plt.show()

    
import seaborn as sb



sb.pairplot(data, height = 4)
import seaborn as sb

corrMatrix = data.corr()

sb.heatmap(corrMatrix, annot=True)
df_new = data.copy()
clus = df_new[['X5 latitude', 'X6 longitude']]

clus.dtypes
from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer



model = KMeans()

visualizer = KElbowVisualizer(model, k = (1, 18)) #k = 2 to 17

visualizer.fit(clus)

visualizer.show()
kmeans = KMeans(n_clusters = 4, random_state = 0) #k = 5

kmeans.fit(clus)
centroids = kmeans.cluster_centers_

centroids
clocation = pd.DataFrame(centroids, columns = ['X5 latitude', 'X6 longitude'])
clocation
plt.scatter(clocation['X5 latitude'], clocation['X6 longitude'], marker = "x", color = 'R', s = 200)
import folium

centroid = clocation.values.tolist()



m = folium.Map(location = [24.968, 121.53], zoom_start = 13)

for point in range(0, len(centroid)):

    folium.Marker(centroid[point], popup = centroid[point]).add_to(m)



m
label = kmeans.labels_

label
df_new['Clusters'] = label

df_new
sb.factorplot(data = df_new, x = "Clusters", kind = "count", size = 7, aspect = 2)
sorted(df_new['Y house price of unit area'].unique())
location=df_new[['X5 latitude','X6 longitude','Y house price of unit area']]

location['color']=location['Y house price of unit area'].apply(lambda price:"Black" if price>=100 else

                                         "green" if price>=90 and price<100 else

                                         "Orange" if price>=80 and price<90 else

                                         "darkblue" if price>=70 and price<80 else

                                         "red" if price>=60 and price<70 else

                                         "lightblue" if price>=50 and price<60 else

                                         "brown" if price>=40 and price<50 else

                                         "grey" if price>=30 and price<40 else

                                         "yellow" if price>=20 and price<30 else

                                         "blue" if price>=10 and price<20 else

                                         "white")

location['size']=location['Y house price of unit area'].apply(lambda price:19 if price>=100 else

                                         17 if price>=90 and price<100 else

                                         15 if price>=80 and price<90 else

                                         13 if price>=70 and price<80 else

                                         11 if price>=60 and price<70 else

                                         9 if price>=50 and price<60 else

                                         7 if price>=40 and price<50 else

                                         5 if price>=30 and price<40 else

                                         3 if price>=20 and price<30 else

                                         1 if price>=10 and price<20 else

                                         0.1)

location
m_2 = folium.Map(location = [24.968, 121.53], zoom_start = 13)

#location=location[0:2000]

for lat,lon,price,color,size in zip(location['X5 latitude'],location['X6 longitude'],location['Y house price of unit area'],location['color'],location['size']):

     folium.CircleMarker([lat, lon],

                            popup=price,

                            radius=size,

                            color='b',

                            fill=True,

                            fill_opacity=0.7,

                            fill_color=color,

                           ).add_to(m_2)

m_2
from folium import plugins

location_data = location[['X5 latitude', 'X6 longitude']]



# plot heatmap

m_2.add_children(plugins.HeatMap(location_data, radius=15))
from scipy.stats import skew



plt.hist(data[data.columns[2]])

skew(data[data.columns[2]])
import math



df_new[df_new.columns[2]] = df_new[df_new.columns[2]].astype(float) 



df_new[df_new.columns[2]] = np.log2(df_new[df_new.columns[2]])
plt.hist(df_new[df_new.columns[2]])

skew(df_new[df_new.columns[2]])
#x3 outliers

x3_q1 = data[data.columns[2]].quantile(0.25)

x3_q3 = data[data.columns[2]].quantile(0.75)

IQR_x3 = x3_q3 - x3_q1

up_out_x3 = x3_q3 + 1.5*IQR_x3

low_out_x3 = x3_q1 - 1.5*IQR_x3



#x5 outliers

x5_q1 = data[data.columns[4]].quantile(0.25)

x5_q3 = data[data.columns[4]].quantile(0.75)

IQR_x5 = x5_q3 - x5_q1

up_out_x5 = x5_q3 + 1.5*IQR_x5

low_out_x5 = x5_q1 - 1.5*IQR_x5



#x6 outliers

x6_q1 = data[data.columns[5]].quantile(0.25)

x6_q3 = data[data.columns[5]].quantile(0.75)

IQR_x6 = x6_q3 - x6_q1

up_out_x6 = x6_q3 + 1.5*IQR_x6

low_out_x6 = x6_q1 - 1.5*IQR_x6



#y outliers

y_q1 = data[data.columns[-1]].quantile(0.25)

y_q3 = data[data.columns[-1]].quantile(0.75)

IQR_y = y_q3 - y_q1

up_out_y = y_q3 + 1.5*IQR_y

low_out_y = y_q1 - 1.5*IQR_y
df_new = df_new[df_new[data.columns[2]] <= up_out_x3]    #x3





df_new = df_new[df_new[data.columns[4]] <= up_out_x5]    #x5

df_new = df_new[df_new[data.columns[4]] >= low_out_x5]



df_new = df_new[df_new[data.columns[5]] <= up_out_x6]    #x6

df_new = df_new[df_new[data.columns[5]] >= low_out_x6]



df_new = df_new[df_new[data.columns[-1]] <= up_out_y]    #y



df_new
df_prep = df_new.copy()
df_prep = df_prep.drop("Clusters", axis = 1)

for indx,col in zip(range(0, len(df_prep.columns)), df_prep.columns):

    plt.title(df_prep.columns[indx])

    plt.boxplot(df_prep[col], vert = False)

    plt.show()

    
corrMatrix = df_prep.corr()

corrMatrix
sb.heatmap(corrMatrix, annot=True)
df_new.to_csv('RealEstate.csv',index=False)