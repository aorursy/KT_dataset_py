# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pylab as plt
%matplotlib inline

import seaborn as snss
from sklearn.model_selection  import train_test_split

from scipy.stats import zscore


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# random ou manuellement
# On considére que le cluster correspond à l'indice dans le tableau centers
K = 4
centers = [[20,15], [50,35], [10,9], [34,60]]
df = pd.DataFrame(centers, columns=["x", "y"])
df.plot.scatter(x="x", y="y")
# Compute distance between 2 points
def euclidean_distance(p1, p2):
    # p --> (x,y)
    somme_square = 0
    for i in range(len(p1)):
        somme_square += (p1[i]-p2[i])**2
    return np.sqrt(somme_square)
# calcul of the mean with points as inputs
def get_mean(points):
    nb_points = len(points)
    size_point = len(points[0])
    mean_point = np.zeros(size_point)
    for point in points:
        for i in range(size_point):
            mean_point[i]+=point[i]
    return mean_point/nb_points
# Calcul of the variance
def get_variance(points):
    nb_points = len(points)
    mean_p = mean(points)
    print("la moyenne est de : ", mean_p)
    somme = 0
    for p in points:
        dist = euclidean_distance(p,mean_p)**2
        print("la distance de ", p, "à ", mean_p, " est de ", dist)
        somme+= dist
    return somme/nb_points
# Normalized the data
def get_normalized_data(data):
    data_norm = []
    mean = get_mean(data)
    variance = get_variance(data)
    for x in data:
        data_norm.append((x-mean)/variance)
    return data_norm
# Find the nearest central point
def find_nearest_center_cluster(p1, centers):
    min_dist = euclidean_distance(p1, centers[0])
    nearest_center_ind = 0
    for i_c in range(len(centers)):
        d = euclidean_distance(p1, centers[i_c])
        #print(p1, " --> ", centers[i_c])
        #print("\tdistance : ", d)
        if d < min_dist:
            nearest_center_ind = i_c
            min_dist = d
    return nearest_center_ind
def apply(vect, func, res=None):
    if len(vect)==0:
        return res
    else:
        return apply(vect[:-1], func, func(res,vect[-1]))
arr = [(1,2),(1,2),(1,2)]

def plus(a,b):
    return a+b

def mult(a,b):
    return a*b

# Trouve le centre d'un nuage de points de N dimension
def find_center_point(points):
    nb_points = len(points)
    size_point = len(points[0])
    center_coord = []
    for i in range(size_point):
        res = apply([p[i] for p in points],plus,0)
        center_coord.append(res/nb_points)
    return center_coord
# Iteration :
def iterate_clustering(data_points, centers):
    K = len(centers)
    clusters = []
    for k in range(K):
        clusters.append([])
        
    for point in data_points:
        cluster = find_nearest_center_cluster(point, centers)
        clusters[cluster].append(point)
        print("the point with point",point, "is among the cluster n°",cluster)
    return clusters
# Test iterate clustering
# data_points = [[54,23],[26,43],[18,32],[34,25],[42,31],[26,29],[54,62]]
# centers = [[20,15], [50,35], [10,9], [34,60]]

nRowsRead = 12203
df = pd.read_csv('/kaggle/input/sites-information-data-from-alexacom-dataset/alexa.com_site_info.csv', delimiter=',', nrows = nRowsRead)
df.dataframeName = 'alexa.com_site_info.csv'


c_1 = [max(df['keyword_opportunities_breakdown_keyword_gaps']), max(df['keyword_opportunities_breakdown_buyer_keywords'])]
c_2 = [max(df['keyword_opportunities_breakdown_keyword_gaps']) - (max(df['keyword_opportunities_breakdown_keyword_gaps']) - (min(df['keyword_opportunities_breakdown_keyword_gaps'])))/4, max(df['keyword_opportunities_breakdown_buyer_keywords']) - (max(df['keyword_opportunities_breakdown_buyer_keywords']) - (min(df['keyword_opportunities_breakdown_buyer_keywords'])))/4]
c_3 = [min(df['keyword_opportunities_breakdown_keyword_gaps']) + (max(df['keyword_opportunities_breakdown_keyword_gaps']) - (min(df['keyword_opportunities_breakdown_keyword_gaps'])))/4, min(df['keyword_opportunities_breakdown_buyer_keywords']) + (max(df['keyword_opportunities_breakdown_buyer_keywords']) - (min(df['keyword_opportunities_breakdown_buyer_keywords'])))/4]
c_4 = [min(df['keyword_opportunities_breakdown_keyword_gaps']), min(df['keyword_opportunities_breakdown_buyer_keywords'])]

centers = [c_1, c_2, c_3, c_4]

data_list = []
for i in range(len(df['keyword_opportunities_breakdown_keyword_gaps'])):
    data_list.append([df['keyword_opportunities_breakdown_keyword_gaps'][i], df['keyword_opportunities_breakdown_buyer_keywords'][i]])
print(data_list)

data_points = np.array(data_list)


clusters = iterate_clustering(data_points, centers)
# Recalculate the centers
# has error !
def recalculate_centers(new_clusters, old_centers):
    new_centers = []
    for i in range(len(new_clusters)):
        if new_clusters[i]==[]:
            new_centers.append(old_centers[i])
        else:
            new_centers.append(find_center_point(new_clusters[i]))
    return new_centers

# new_centers = recalculate_centers(clusters, centers)
# Verify if center are stable or not by verify if center changed or not
def centers_changed(old_centers, new_centers):
    for ocenter in old_centers:
        if ocenter not in new_centers:
            return 1
        else:
            return 0
import matplotlib as plt
# Testing the iteration function
data_points = np.array([[54,23],[26,43],[18,32],[34,25],[42,31],[26,29],[54,62]])
clusters = iterate_clustering(data_points, centers)
new_centers = recalculate_centers(clusters, centers)
while(centers_changed(centers,new_centers)):
    clusters = iterate_clustering(data_points, new_centers)
    n_centers = recalculate_centers(clusters, centers)
    centers = new_centers
    new_centers = n_centers
print(clusters)
print(new_centers)
# For c in clusters:
#    x,y = [data_points[p] for p in clusters]

# Verify if center are stable or not by verify if center changed or not
def centers_changed(old_centers, new_centers):
    for ocenter in old_centers:
        if ocenter not in new_centers:
            return 1
        else:
            return 0
import matplotlib as plt

# Testing the iteration function
clusters = iterate_clustering(data_points, centers)
new_centers = recalculate_centers(clusters, centers)
while(centers_changed(centers,new_centers)):
    clusters = iterate_clustering(data_points, new_centers)
    n_centers = recalculate_centers(clusters, centers)
    centers = new_centers
    new_centers = n_centers
print(clusters)
print(new_centers)
# For c in clusters:
#    x,y = [data_points[p] for p in clusters]


kmeans_model = KMeans(n_clusters=4, init=np.array(centers))
print(kmeans_model)
####################### New eddition ##################################
#Read the datset

# tech_supp_df = pd.read_csv("/kaggle/input/technical-customer-support-data/technical_support_data.csv")
# tech_supp_df.dtypes

nRowsRead = 12203
df = pd.read_csv('/kaggle/input/sites-information-data-from-alexacom-dataset/alexa.com_site_info.csv', delimiter=',', nrows = nRowsRead)
# df.dtypes

#Shape of the dataset
df.shape
#Displaying the first five rows of dataset 
df.head()
# print(df)
# Plotiing the pairplot
for item in df:
#     print(type(df[item][1]))
    if type(df[item][1]) == np.float64:
        for i in range(len(df[item])):
            df[item][i] = df[item][i].astype(np.int64)
#         print(item)
    if type(df[item][1]) == str:
        del(df[item])
    
print("ok")


# techSuppAttr=df.iloc[:,1:]
# techSuppScaled=techSuppAttr.apply(zscore)


# techSuppAttr
# techSuppScaled
# sns.pairplot(techSuppScaled,diag_kind='kde')
from scipy.spatial.distance import cdist
clusters=range(1,10)
meanDistortions=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    try:
        model.fit(techSuppScaled)
        prediction=model.predict(techSuppScaled)
        meanDistortions.append(sum(np.min(cdist(techSuppScaled, model.cluster_centers_, 'euclidean'), axis=1)) / techSuppScaled.shape[0])
    except:
        meanDistortions.append(6)


plt.plot(clusters, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
############## new edition for k-means clustring and display on diagram
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from dateutil import parser
import io
import base64
from IPython.display import HTML
from imblearn.under_sampling import RandomUnderSampler
from subprocess import check_output

nRowsRead = 12203
df = pd.read_csv('/kaggle/input/sites-information-data-from-alexacom-dataset/alexa.com_site_info.csv', delimiter=',', nrows = nRowsRead)
df.dataframeName = 'alexa.com_site_info.csv'

x_0 = min([min(df['keyword_opportunities_breakdown_optimization_opportunities']), min(df['keyword_opportunities_breakdown_buyer_keywords'])])
x_1 = max([max(df['keyword_opportunities_breakdown_optimization_opportunities']), max(df['keyword_opportunities_breakdown_buyer_keywords'])])

y_0 = min([min(df['all_topics_keyword_gaps_Avg_traffic_parameter_3']), min(df['all_topics_buyer_keywords_Avg_traffic_parameter_4'])])
y_1 = max([max(df['all_topics_keyword_gaps_Avg_traffic_parameter_3']), max(df['all_topics_buyer_keywords_Avg_traffic_parameter_4'])])

# c_1 = [max(df['keyword_opportunities_breakdown_keyword_gaps']), max(df['keyword_opportunities_breakdown_buyer_keywords'])]

xlim = [x_0, x_1]
ylim = [y_0, y_1]
# xlim = [-74.03, -73.77]
# ylim = [40.63, 40.85]
# df
df = df[(df.keyword_opportunities_breakdown_optimization_opportunities> xlim[0]) & (df.keyword_opportunities_breakdown_optimization_opportunities < xlim[1])]
df = df[(df.keyword_opportunities_breakdown_buyer_keywords> xlim[0]) & (df.keyword_opportunities_breakdown_buyer_keywords < xlim[1])]
df = df[(df.all_topics_keyword_gaps_Avg_traffic_parameter_3> ylim[0]) & (df.all_topics_keyword_gaps_Avg_traffic_parameter_3 < ylim[1])]
df = df[(df.all_topics_buyer_keywords_Avg_traffic_parameter_4> ylim[0]) & (df.all_topics_buyer_keywords_Avg_traffic_parameter_4 < ylim[1])]
# 'all_topics_keyword_gaps_Avg_traffic_parameter_3', 'all_topics_buyer_keywords_Avg_traffic_parameter_4'
longitude = list(df.all_topics_keyword_gaps_Avg_traffic_parameter_3) + list(df.all_topics_keyword_gaps_Avg_traffic_parameter_3)
# latitude = 
latitude = list(df.keyword_opportunities_breakdown_optimization_opportunities) + list(df.keyword_opportunities_breakdown_buyer_keywords)
plt.figure(figsize = (10,10))
plt.plot(longitude,latitude,'.', alpha = 0.4, markersize = 0.5)
plt.show()
loc_df = pd.DataFrame()
loc_df['longitude'] = longitude
loc_df['latitude'] = latitude

kmeans = KMeans(n_clusters=15, random_state=2, n_init = 10).fit(loc_df)
loc_df['label'] = kmeans.labels_

loc_df = loc_df.sample(12000)
plt.figure(figsize = (10,10))
for label in loc_df.label.unique():
    plt.plot(loc_df.longitude[loc_df.label == label],loc_df.latitude[loc_df.label == label],'.', alpha = 0.3, markersize = 0.3)
# %%HTML
plt.title('Clusters of data')
plt.show()
nRowsRead = 12203
df = pd.read_csv('/kaggle/input/sites-information-data-from-alexacom-dataset/alexa.com_site_info.csv', delimiter=',', nrows = nRowsRead)
df.dataframeName = 'alexa.com_site_info.csv'
import math
from operator import itemgetter

df = df.fillna(df.mean())

def display_plot(x, y):

    area = np.pi * 3
    plt.scatter(x, y, s=area, alpha=0.5)

    plt.title('Comaparison dataset columns')
    plt.xlabel('item_1')
    plt.ylabel('item_2')

    plt.plot()
    plt.show()


def cluster_by_categories(item_1, item_2):
    categories = []
    for item in df['category']:
        if item in categories:
            print('item in catgeries : ' + str(item))
        if item.split('/')[0] not in categories:
            categories.append(item.split('/')[0])


    distance = []
    
    associate = []
    min_dist_center = []
    
    average_coordinates = []
    for i in range(len(categories)):
        keyword_opportunities_breakdown_optimization_opportunities = []
        Daily_time_on_site = []
        sum_x = 0
        sum_y = 0
        min_dist_to_center = 0
        for j in range(len(df['category'])):
            try:
                if categories[i] in df['category'][j]:
                    keyword_opportunities_breakdown_optimization_opportunities.append(df[item_1][j])
                    Daily_time_on_site.append(df[item_2][j])
            except:
                pass
        try:
            for k in range(len(keyword_opportunities_breakdown_optimization_opportunities)):
                sum_x += df[item_1][k]
                sum_y += df[item_2][k]
        except:
            pass
        # print('sum_x : ')
        # print(sum_x)
        # print('sum_y : ')
        # print(sum_y)
        try: 
            average_x = sum_x / len(keyword_opportunities_breakdown_optimization_opportunities)                
            average_y = sum_y / len(keyword_opportunities_breakdown_optimization_opportunities)
            
            for k in range(len(keyword_opportunities_breakdown_optimization_opportunities)):
                min_dist_to_center += pow((pow((df[item_1][k] - average_x), 2) + pow((df[item_2][k] - average_y), 2)), 1/2)
            
            min_dist_center.append([categories[i], min_dist_to_center, average_x, average_y])
            average_coordinates.append([categories[i], average_x, average_y])
        except:
            pass

        # display_plot(keyword_opportunities_breakdown_optimization_opportunities, Daily_time_on_site)




    for i in range(len(average_coordinates)):
        for j in range(i + 1, len(average_coordinates)):
            distance.append([average_coordinates[i][0], average_coordinates[j][0], pow(pow((average_coordinates[i][1] - average_coordinates[j][1]) , 2) + pow((average_coordinates[i][2] - average_coordinates[j][2]) , 2), 1/2)])

    
    category_1 = []
    category_2 = []

    category_1_x = []
    category_1_y= []
    category_2_x = []
    category_2_y = []
    
    min_dist_center = sorted(min_dist_center, key=itemgetter(1), reverse=False)

    
    for i in range(len(min_dist_center)):
        for j in range(i + 1, len(min_dist_center)):
            if i < 10 and pow((pow((min_dist_center[i][2] - min_dist_center[j][2]), 2) + pow((min_dist_center[i][3] - min_dist_center[i][3]), 2)), 1/2) > 1000:
                flag = 0
                for k in range(len(associate)):
                    if min_dist_center[i][0] in associate[k] and min_dist_center[j][0] in associate[k]:
                        flag = 1
                if flag == 0:
                    associate.append([min_dist_center[i][0] , min_dist_center[j][0]])
    print(' Associates !!!! ')
    print(associate)
    for i in range(len(associate)):
        category_1_x = []
        category_1_y= []
        category_2_x = []
        category_2_y = []
        if associate != []:
            for k in range(len(df['category'])):
                if associate[i][0] in df['category'][k]:
                    category_1_x.append(df[item_1][k])
                    category_1_y.append(df[item_2][k])
                if associate[i][1] in df['category'][k]:
                    category_2_x.append(df[item_2][k])
                    category_2_y.append(df[item_2][k])
            print('Two categories which have maximum distance between their centers ...')
            print('_________________________________________________')
            print(distance[0][0] , distance[0][1])
            print('_________________________________________________')
            print(item_1, item_2)
            print('_________________________________________________')
            display_plot(category_1_x, category_1_y)
            display_plot(category_2_x, category_2_y)

            
##########  Maximum distance between two catgeories average axis 
#     category_1_x = []
#     category_1_y = []
#     category_2_x = []
#     category_2_y = []
#     distance = sorted(distance, key=itemgetter(2), reverse=True)
#     for i in range(len(df['category'])):
# #         print(distance[0][0], distance[0][1])
#         if distance[0][0] in df['category'][i]:
#             category_1_x.append(df[item_1][i])
#             category_1_y.append(df[item_2][i])
#         if distance[0][1] in df['category'][i]:
#             category_2_x.append(df[item_2][i])
#             category_2_y.append(df[item_2][i])
#     print('Two categories which have maximum distance between their centers ...')
#     print('_________________________________________________')
#     print(distance[0][0] , distance[0][1])
#     print('_________________________________________________')
#     print(item_1, item_2)
#     print('_________________________________________________')
#     display_plot(category_1_x, category_1_y)
#     display_plot(category_2_x, category_2_y)


    
numeric_columns = []
str_counter = 0
for item in df:
    try:
        y = 0
        x = df[item][1] * 2
        z = x + y
        numeric_columns.append(item)
    except:
        str_counter += 1
print(str_counter)
    
all_corr = []
for i in range(len(numeric_columns)):
    for j in range(i + 1, len(numeric_columns) - 1):
        if numeric_columns[i].split('parameter_')[0] not in numeric_columns[j] and 'keyword_opp' not in numeric_columns[i]:
            all_corr.append([numeric_columns[i], numeric_columns[j], df[numeric_columns[i]].corr(df[numeric_columns[j]])])

# all_corr = sorted(all_corr, key=itemgetter(2), reverse=False)
all_corr = sorted(all_corr, key=itemgetter(2), reverse=True)

for i in range(20):
    print(' iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii :::::::::::::::::::: ')
    print(all_corr[i][0], all_corr[i][1])
    print(' iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii :::::::::::::::::::: ')
    cluster_by_categories(all_corr[i][0], all_corr[i][1])
# distance.sort(key = 2, reverse = True)
              
print('ok--1')
# print(distance[0], distance[1], distance[2])
print('ok--2')

# print('distance : ')
# print(distance)

print('ok--3')

import math
from operator import itemgetter

df = df.fillna(df.mean())

def display_plot(x, y):

    area = np.pi * 3
    plt.scatter(x, y, s=area, alpha=0.5)

    plt.title('Comaparison dataset columns')
    plt.xlabel('item_1')
    plt.ylabel('item_2')

    plt.plot()
    plt.show()


def cluster_by_categories(item_1, item_2):
    categories = []
    for item in df['category']:
        if item in categories:
            print('item in catgeries : ' + str(item))
        if item.split('/')[0] not in categories:
            categories.append(item.split('/')[0])


    distance = []
    
    associate = []
    min_dist_center = []
    
    average_coordinates = []
    for i in range(len(categories)):
        keyword_opportunities_breakdown_optimization_opportunities = []
        Daily_time_on_site = []
        sum_x = 0
        sum_y = 0
        min_dist_to_center = 0
        for j in range(len(df['category'])):
            try:
                if categories[i] in df['category'][j]:
                    keyword_opportunities_breakdown_optimization_opportunities.append(df[item_1][j])
                    Daily_time_on_site.append(df[item_2][j])
            except:
                pass
        try:
            for k in range(len(keyword_opportunities_breakdown_optimization_opportunities)):
                sum_x += df[item_1][k]
                sum_y += df[item_2][k]
        except:
            pass
        # print('sum_x : ')
        # print(sum_x)
        # print('sum_y : ')
        # print(sum_y)
        try: 
            average_x = sum_x / len(keyword_opportunities_breakdown_optimization_opportunities)                
            average_y = sum_y / len(keyword_opportunities_breakdown_optimization_opportunities)
            
            for k in range(len(keyword_opportunities_breakdown_optimization_opportunities)):
                min_dist_to_center += pow((pow((df[item_1][k] - average_x), 2) + pow((df[item_2][k] - average_y), 2)), 1/2)
            
            min_dist_center.append([categories[i], min_dist_to_center, average_x, average_y])
            average_coordinates.append([categories[i], average_x, average_y])
        except:
            pass

        # display_plot(keyword_opportunities_breakdown_optimization_opportunities, Daily_time_on_site)




    for i in range(len(average_coordinates)):
        for j in range(i + 1, len(average_coordinates)):
            distance.append([average_coordinates[i][0], average_coordinates[j][0], pow(pow((average_coordinates[i][1] - average_coordinates[j][1]) , 2) + pow((average_coordinates[i][2] - average_coordinates[j][2]) , 2), 1/2)])

    
    category_1 = []
    category_2 = []
    
    min_dist_center = sorted(min_dist_center, key=itemgetter(1), reverse=False)

##########  best axis for nearest associate between two catgeories
#     for i in range(len(min_dist_center)):
#         for j in range(i + 1, len(min_dist_center)):
#             if i < 10 and pow((pow((min_dist_center[i][2] - min_dist_center[j][2]), 2) + pow((min_dist_center[i][3] - min_dist_center[i][3]), 2)), 1/2) > 1000:
#                 flag = 0
#                 for k in range(len(associate)):
#                     if min_dist_center[i][0] in associate[k] and min_dist_center[j][0] in associate[k]:
#                         flag = 1
#                 if flag == 0:
#                     associate.append([min_dist_center[i][0] , min_dist_center[j][0]])
#     print(' Associates !!!! ')
#     print(associate)
#     for i in range(len(associate)):
#         category_1_x = []
#         category_1_y= []
#         category_2_x = []
#         category_2_y = []
#         if associate != []:
#             for k in range(len(df['category'])):
#                 if associate[i][0] in df['category'][k]:
#                     category_1_x.append(df[item_1][k])
#                     category_1_y.append(df[item_2][k])
#                 if associate[i][1] in df['category'][k]:
#                     category_2_x.append(df[item_2][k])
#                     category_2_y.append(df[item_2][k])
#             print('Two categories which have maximum distance between their centers ...')
#             print('_________________________________________________')
#             print(distance[0][0] , distance[0][1])
#             print('_________________________________________________')
#             print(item_1, item_2)
#             print('_________________________________________________')
#             display_plot(category_1_x, category_1_y)
#             display_plot(category_2_x, category_2_y)

            
##########  Maximum distance between two catgeories average axis 
    category_1_x = []
    category_1_y = []
    category_2_x = []
    category_2_y = []
    distance = sorted(distance, key=itemgetter(2), reverse=True)
    for i in range(len(df['category'])):
#         print(distance[0][0], distance[0][1])
        if distance[0][0] in df['category'][i]:
            category_1_x.append(df[item_1][i])
            category_1_y.append(df[item_2][i])
        if distance[0][1] in df['category'][i]:
            category_2_x.append(df[item_2][i])
            category_2_y.append(df[item_2][i])
    print('Two categories which have maximum distance between their centers ...')
    print('_________________________________________________')
    print(distance[0][0] , distance[0][1])
    print('_________________________________________________')
    print(item_1, item_2)
    print('_________________________________________________')
    display_plot(category_1_x, category_1_y)
    display_plot(category_2_x, category_2_y)


    
numeric_columns = []
str_counter = 0
for item in df:
    try:
        y = 0
        x = df[item][1] * 2
        z = x + y
        numeric_columns.append(item)
    except:
        str_counter += 1
print(str_counter)
    
all_corr = []
for i in range(len(numeric_columns)):
    for j in range(i + 1, len(numeric_columns) - 1):
        if numeric_columns[i].split('parameter_')[0] not in numeric_columns[j] and 'keyword_opp' not in numeric_columns[i]:
            all_corr.append([numeric_columns[i], numeric_columns[j], df[numeric_columns[i]].corr(df[numeric_columns[j]])])

# all_corr = sorted(all_corr, key=itemgetter(2), reverse=False)
all_corr = sorted(all_corr, key=itemgetter(2), reverse=True)

for i in range(20):
    print(' iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii :::::::::::::::::::: ')
    print(all_corr[i][0], all_corr[i][1])
    print(' iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii :::::::::::::::::::: ')
    cluster_by_categories(all_corr[i][0], all_corr[i][1])
# distance.sort(key = 2, reverse = True)
              
print('ok--1')
# print(distance[0], distance[1], distance[2])
print('ok--2')

# print('distance : ')
# print(distance)

print('ok--3')




print(pow((pow(2, 2) + pow(2 ,2) + pow(2, 2) + pow(2 ,2)) , 1/ 2))