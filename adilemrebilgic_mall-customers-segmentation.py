import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans

from sklearn.metrics import homogeneity_completeness_v_measure, silhouette_score

from scipy.spatial.distance import cdist



#reading data

data=pd.read_csv('../input/Mall_Customers.csv')
data.head()
#You can use CustomerID as index

data.set_index('CustomerID',inplace=True)

data.head()
data.shape
data.dtypes
#print descriptive statistics

data.describe(include='all')
#plot the data

features=data.columns.values

fig=plt.figure(figsize=(18,12))

temp_plot_numb=1

for feature in features:

    fig.add_subplot(2,2,

                    temp_plot_numb)

    temp_plot_numb+=1

    title='{} Distribution'.format(feature)

    sns.countplot(data[feature])

    plt.title(title)

plt.tight_layout()
#Create a pair plot the visualize the relationship between features

sns.pairplot(data,

             hue='Gender',

             height=4);
#is there any missing value ?

data.isnull().sum().any()   
def check_extreme_values_and_visualize(data_frame,

                                       lower_limit_perc,

                                       upper_limit_perc):

    '''this user made function takes 3 inputs and returns a plot with extreme values marked differently'''

    lower_limit=np.percentile(data_frame, lower_limit_perc, axis=0)

    upper_limit=np.percentile(data_frame, upper_limit_perc, axis=0)

    select_extreme_values= np.logical_or(data_frame>=upper_limit,data_frame<=lower_limit)

    plt.plot(data_frame[select_extreme_values],'ro')

    plt.plot(data_frame[~select_extreme_values],'bo')

    plt.ylabel(data_frame.name)

    plt.title('{} Distribution and Extreme Values'.format(data_frame.name))



#visualize the extreme values    

fig=plt.figure(figsize=(25,12))

ax1=fig.add_subplot(1,3,1)

check_extreme_values_and_visualize(data_frame=data['Age'],

                                       lower_limit_perc=5,upper_limit_perc=95)

ax1=fig.add_subplot(1,3,2)

check_extreme_values_and_visualize(data_frame=data['Annual Income (k$)'],

                                       lower_limit_perc=5,upper_limit_perc=95)

ax1=fig.add_subplot(1,3,3)

check_extreme_values_and_visualize(data_frame=data['Spending Score (1-100)'],

                                       lower_limit_perc=5,upper_limit_perc=95)
MM_Scaler=MinMaxScaler() #define a scaler

data_standardized=data.copy() # keep the original data for further usage

#standardize Age

data_standardized['Age']=MM_Scaler.fit_transform(data['Age'].values.reshape(-1,1))

#standardize annual income

data_standardized['Annual Income (k$)']=MM_Scaler.fit_transform(data['Annual Income (k$)'].values.reshape(-1,1))

#standardize spending score

data_standardized['Spending Score (1-100)']=MM_Scaler.fit_transform(data['Spending Score (1-100)'].values.reshape(-1,1))
data_standardized=pd.get_dummies(data=data_standardized,columns=['Gender'])
#Find the best K parameter for best seperation



distortions = [] # create an empty list for distortion values

silhouette_scrs=[] #create an empty list for silhouette values



X=data_standardized[['Age','Annual Income (k$)','Spending Score (1-100)','Gender_Female','Gender_Male']]



#a look is created to try and compare different results of k parameters.

for cluster in np.arange(3,15):

    km=KMeans(n_clusters=cluster,random_state=2)

    col_name='Cluster_{}'.format(cluster)

    data_standardized[col_name]=km.fit_predict(X)

    distortions.append(sum(np.min(cdist(X,

                                        km.cluster_centers_,

                                        'euclidean'), 

                                  axis=1)) / X.shape[0])

    silhouette_scrs.append(silhouette_score(X,data_standardized[col_name]))



#Graphs are created to visualize the distortion and silhouette score distrubution after each clustering step with different k parameters

fig=plt.figure(figsize=(28,18))

ax1=fig.add_subplot(2,1,1)

plt.xticks(np.arange(0,12),np.arange(3,15))

ax1=plt.plot(distortions)

ax1=plt.plot(distortions,'ro')

plt.title('Distortion Distribution')

plt.xlabel('Number of Clusters')

plt.ylabel('Distortion')



ax2=fig.add_subplot(2,1,2)

plt.xticks(np.arange(0,12),np.arange(3,15))

ax2=plt.plot(silhouette_scrs)

ax2=plt.plot(silhouette_scrs,'ro')

plt.title('Silhouette Score Distribution')
#After examining the plots,best possible K Parameters is 10.

#Before profiling, I add the cluster and female gender columns to the original data.

data['Gender_Female']=data_standardized['Gender_Female']

data['Cluster_10']=data_standardized['Cluster_10']
#In this step, The 10 different customer segments/groups are profiled in order to identify them by their differentiates with the help of visualization



feature_names=['Age','Annual Income (k$)','Spending Score (1-100)','Gender']

#make plots of the cluster 10 and entire population

fig=plt.figure(figsize=(15,40))

plot_numb=1  # this parameter is used to place the plots on the screen.

plot_numb2=3 # this parameter is used to place the plots on the screen.

for feature_name in feature_names:

    ax1=fig.add_subplot(5,2,plot_numb)

    plot_numb +=2 

    ax2=fig.add_subplot(5,3,plot_numb2)

    plot_numb2 +=3

    

    if feature_name=='Gender':        # Historgram is used to visualize the gender feature.

        sns.countplot(hue='Gender',

                      data=data,

                      x='Cluster_10',

                      ax=ax1)

        sns.countplot(x='Gender',

                      data=data,ax=ax2)       

    else:

        sns.boxplot(x='Cluster_10',   # Boxplot is used to visualize the features except gender.

                    y=feature_name,

                    data=data,

                    ax=ax1)

        sns.boxplot(y=feature_name,

                    data=data,

                    ax=ax2)        

        

    plt1_title='{}--{}'.format(feature_name,'Cluster-10')

    ax1.set_title(plt1_title)

    ax1.set_xlabel('Cluster Number')

    plt2_title='{}--{}'.format(feature_name,'Population')

    ax2.set_title(plt2_title)

    ax2.set_xlabel('Cluster Number')
Cluster_Profile_df=pd.DataFrame({'Age':['Young','Old','Average','Average','Young','Old','Average','Average','Old','Young'],

                                 'Annual Income (k$)':['Average','Average','High','Low','Low','Average','High','High','Low','Low'],

                                  'Spending Score (1-100)':['Average','Average','High','Low','High','Average','Low','High','Low','Average'],

                                   'Gender':['Female Dominated','Male Dominated','Male Dominated','Female Dominated','Female Dominated','Female Dominated','Male Dominated','Female Dominated','Female Dominated','Male Dominated'], 

                                'Group Size':data['Cluster_10'].value_counts()} )

Cluster_Profile_df.sort_values(by=['Age','Annual Income (k$)','Gender','Spending Score (1-100)'])