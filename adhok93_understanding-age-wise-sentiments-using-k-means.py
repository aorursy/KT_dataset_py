import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture #For GMM clustering
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

review_data = pd.read_csv("../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv")
review_data.dropna(inplace=True)
#review_data=review_data.drop(['Unnamed: 0', 'Clothing ID'],axis=1)
print(review_data.dtypes)
review_data.head(5)
# remove all the columns that are categorical variables
review_data_k_means=review_data.drop(['Unnamed: 0', 'Clothing ID','Class Name','Department Name','Title','Division Name','Recommended IND'],axis=1)
## adding the minimum value of sentiment score so as to remove negative sentiment scores


import pickle

with open("../input/sentiment-score-new/sent.txt", "rb") as fp:   # Unpickling
    b = pickle.load(fp)

b

review_data_k_means['sent_score'] = b


min_sent = abs(np.min(review_data_k_means['sent_score']))
review_data_k_means['sent_score'] =  review_data_k_means['sent_score'] + abs(np.min(review_data_k_means['sent_score']))
review_data_k_means.drop('Review Text',axis=1,inplace=True)


sns.set_style('darkgrid')
plt.title('Distribution of Each Column in the Data')

for i,col in enumerate(review_data_k_means.columns):
    plt.figure(i)
    sns.distplot(review_data_k_means[col])
# box cox transform can help with the skewed transformation
from scipy.stats import boxcox
tmp = review_data_k_means 
# adding one to each data variable to make it positive
tmp = tmp+1
for i in tmp.columns:
    tmp[i]=np.log(tmp[i])
# log modified data    
review_data_mod = tmp
# checking the distributions after transforming
sns.set_style('darkgrid')
plt.title('Distribution of Each Column in the Data')

for i,col in enumerate(review_data_mod.columns):
    plt.figure(i)
    sns.distplot(review_data_mod[col])


# just take age and sent score - the variables that display a nearly normal distribution

#review_data_mod = review_data_mod[['Age','sent_score']]



review_data_mod = review_data_mod[['Age','sent_score']]


from scipy import stats
review_data_std = stats.zscore(review_data_mod)
review_data_std = np.array(review_data_std)





# This snippet is sourced from https://www.kaggle.com/mariadobreva/k-means-clustering-in-python
# also refer to https://stackoverflow.com/questions/32370543/understanding-score-returned-by-scikit-learn-kmeans/32371258
import pylab as pl
number_of_clusters = range(1,20)
kmeans = [KMeans(n_clusters=i,max_iter=1000,random_state=42) for i in number_of_clusters]
score = [-1*kmeans[i].fit(review_data_std).score(review_data_std) for i in range(len(kmeans))]
pl.plot((number_of_clusters),score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()


    






k_means_test = KMeans(n_clusters=6,max_iter=1000,random_state=42)
-1*k_means_test.fit(review_data_std).score(review_data_std)
review_data_k_means['labels'] = k_means_test.labels_
size_of_each_cluster = review_data_k_means.groupby('labels').size().reset_index()
size_of_each_cluster.columns = ['labels','number_of_points']
size_of_each_cluster['percentage'] = size_of_each_cluster['number_of_points']/np.sum(size_of_each_cluster['number_of_points'])

print(size_of_each_cluster)
# a look at Age and Sentiment Scores
# we subtract the added absolute value of the minimum sentiment score
review_data_k_means['sent_score'] = review_data_k_means['sent_score'] - min_sent
sns.lmplot('Age','sent_score',data=review_data_k_means,hue='labels',fit_reg=False)
plt.show()
sent_score_labels = review_data_k_means[['sent_score','labels']]

sent_score_labels.boxplot(by='labels',figsize=(20,10))
plt.xticks(rotation=90)
plt.show()


age_labels = review_data_k_means[['labels','Age']]

age_labels.boxplot(by='labels',figsize=(20,10))
plt.xticks(rotation=90)
plt.show()

