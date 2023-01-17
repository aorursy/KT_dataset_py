# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv("../input/athletes.csv")

# Any results you write to the current directory are saved as output.
for i,row in enumerate(data.iterrows()):

    try:

        data.loc[i,'Age'] = 116 - float(row[1].dob[len(row[1].dob)-2:len(row[1].dob)])

    except TypeError:

        data.loc[i,'Age'] = 0

        

data['weight'] = data.weight.fillna(0)

data['height'] = data.height.fillna(0)

data = data[data.weight>0]

data = data[data.height>0]

data = data[data.Age>0]
males = data[data.sex == 'male']

females = data[data.sex == 'female']
import seaborn as sns

sns.distplot(males.height[males.Age>0], hist = True)

sns.distplot(females.height[females.Age>0], hist = True)

plt.legend(['Males.  std: ' + str(np.std(males.height))[:4],'Females.  std: ' + str(np.std(females.height))[:4]], loc = 2)
legend = []

sports = ['volleyball','basketball','football']

for sport in sports:

    plt.plot(males.height[males.sport == sport],males.weight[males.sport == sport], 'o')

    legend.append(sport)

    plt.hold(True)

    

plt.legend(legend, loc = 2,numpoints=1)

plt.title('Can we seperate different sports?')

plt.xlabel('Height [m]')

plt.ylabel('Weight [kg]')

plt.show()
legend = []

sports = ['volleyball','basketball','football']

for sport in sports:

    plt.plot(males.height[males.sport == sport],males.Age[males.sport == sport], 'o')

    legend.append(sport)

    plt.hold(True)

    

plt.legend(legend, loc = 2,numpoints=1)

plt.title('Can we seperate different sports?')

plt.xlabel('Height [m]')

plt.ylabel('Age')

plt.show()
ball_df = males[(males.sport == 'volleyball') |(males.sport == 'football') |(males.sport == 'basketball')]

ball_df = ball_df.drop([u'id', u'name', u'nationality', u'sex', u'dob', 

     u'gold', u'silver', u'bronze'], axis = 1)



from sklearn.decomposition import PCA

from sklearn.cluster import KMeans



ball_df['weight'] = ball_df.weight.fillna(0)

ball_df = ball_df[ball_df.weight>0]



pca = PCA(n_components=2)

pca.fit(ball_df.drop(['sport'],axis = 1))



kmeans_df =  pd.DataFrame(data=ball_df.drop(['sport'],axis = 1))

kmeans = KMeans(n_clusters = 3, random_state = 0).fit(kmeans_df)

kmeans.cluster_centers_

ball_df['label'] = kmeans.labels_



pca_mat = pca.transform(ball_df.drop(['sport','label'],axis = 1))

ball_df['pca_1'] = pca_mat[:,0]

ball_df['pca_2'] = pca_mat[:,1]
plt.figure()

plt.plot(ball_df.pca_1[ball_df.sport == 'volleyball'],ball_df.pca_2[ball_df.sport == 'volleyball'],'o')

plt.plot(ball_df.pca_1[ball_df.sport == 'basketball'],ball_df.pca_2[ball_df.sport == 'basketball'],'o')

plt.plot(ball_df.pca_1[ball_df.sport == 'football'],ball_df.pca_2[ball_df.sport == 'football'],'o')

plt.legend(['volleyball','basketball','football'],numpoints = 1)

plt.title('Age-Height-Weight reduced to 2D with PCA')

plt.show()



plt.figure()

plt.title('Using K-means to find 3 clusters - do they correspond with the division by sport?')

plt.plot(ball_df.pca_1[ball_df.label == 0],ball_df.pca_2[ball_df.label == 0],'o')

plt.plot(ball_df.pca_1[ball_df.label ==1],ball_df.pca_2[ball_df.label == 1],'o')

plt.plot(ball_df.pca_1[ball_df.label == 2],ball_df.pca_2[ball_df.label == 2],'o')

label_1 = 'height ' + str(kmeans.cluster_centers_[0,0])[0:4] + ' weight ' +  str(kmeans.cluster_centers_[0,1])[0:4]+ ' Age ' + str(kmeans.cluster_centers_[0,2])[0:4]

label_2 = 'height ' + str(kmeans.cluster_centers_[1,0])[0:4] + ' weight ' +  str(kmeans.cluster_centers_[1,1])[0:4]+ ' Age ' + str(kmeans.cluster_centers_[1,2])[0:4]

label_3 = 'height ' + str(kmeans.cluster_centers_[2,0])[0:4] + ' weight ' +  str(kmeans.cluster_centers_[2,1])[0:4]+ ' Age ' + str(kmeans.cluster_centers_[2,2])[0:4]

plt.legend([label_1,label_2,label_3],numpoints = 1,fontsize = 10)

plt.show()