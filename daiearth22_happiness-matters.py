import numpy as np # linear algebra 線形代数

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) データ整形

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
churn = pd.read_csv('../input/churn.csv')

churn.head()
ci = pd.read_csv('../input/commentInteractions.csv')

ci.head()
ci.shape
cc = pd.read_csv('../input/comments_clean_anonimized.csv')

cc.head()
votes = pd.read_csv('../input/votes.csv')

votes.head()
len(votes['companyAlias'].unique())
companies = pd.Series(votes['companyAlias'].unique())

vc = [companies.values.tolist().index(company) for company in votes['companyAlias'].values]

churn_company = [companies.values.tolist().index(company) if company in companies.values else -1 for company in churn['companyAlias'].values ]

comment_company = [companies.values.tolist().index(company) if company in companies.values else -1 for company in ci['companyAlias'].values ]

comment_company2 = [companies.values.tolist().index(company) if company in companies.values else -1 for company in cc['companyAlias'].values ]



votes['companyAlias'] = vc

churn['companyAlias'] = churn_company

ci['companyAlias'] = comment_company

cc['companyAlias'] = comment_company2
dates = votes['voteDate'].str.replace('CET','')

dates = dates.str.replace('CEST','')

votes['voteDate']= dates
votes['voteDate'] = pd.to_datetime(votes['voteDate'],format="%a %b %d %H:%M:%S %Y")
votes['wday'] = votes['voteDate'].dt.dayofweek

votes['yday'] = votes['voteDate'].dt.dayofyear

votes['year'] = votes['voteDate'].dt.year
votes['year'].unique()
votes['year'] = votes['year']-2014
votes['employee'] = votes['companyAlias'].astype(str)+"_"+votes['employee'].astype(str)

churn['employee'] = churn['companyAlias'].astype(str)+"_"+churn['employee'].astype(str)

ci['employee'] = ci['companyAlias'].astype(str)+"_"+ci['employee'].astype(str)

cc['employee'] = cc['companyAlias'].astype(str)+"_"+cc['employee'].astype(str)



len(votes['employee'].unique())
employee = votes.groupby('employee',as_index=False).mean()

employee = employee.merge(churn,on=['employee','employee'],how='left').drop_duplicates(subset="employee")
employee['companyAlias'] = employee.companyAlias_x.astype(int)

employee = employee.drop(['companyAlias_x','companyAlias_y'],axis=1)

employee.head()
import seaborn as sns

import matplotlib.pyplot as plt



f,axarr = fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))

data =votes.groupby('companyAlias').mean()

sns.barplot(x=data.index,y= data['vote'])
week_happ = votes.groupby('wday').mean()['vote']

sns.barplot(x = week_happ.index, y = week_happ.values)
churn_employee = employee[employee['stillExists']==True]

churn_employee = churn_employee.groupby('companyAlias').count()

tmp = employee.groupby('companyAlias',as_index=False).count()

churn_perc = 1- churn_employee['stillExists'].astype(float)/tmp['stillExists']

churn_perc = [0 if np.isnan(perc) else perc for perc in churn_perc]
data['churn_perc'] = churn_perc
data.corr()
sns.regplot(data['vote'].values,data['churn_perc'].values)
likes = ci[ci['liked']==True].groupby('employee',as_index=False).count()

likes = likes[['employee','liked']]

hates = ci[ci['disliked']==True].groupby('employee',as_index=False).count()

hates = hates[['employee','disliked']]

hated = cc[cc['dislikes']==True].groupby('employee',as_index=False).count()

hated = hated[['employee','dislikes']]

loved = cc[cc['likes']==True].groupby('employee',as_index=False).count()

loved = loved[['employee','likes']]

employee = employee.merge(likes,on='employee',how='left').drop_duplicates(subset="employee")

employee = employee.merge(hates,on='employee',how='left').drop_duplicates(subset="employee")

employee = employee.merge(hated,on='employee',how='left').drop_duplicates(subset="employee")

employee = employee.merge(loved,on='employee',how='left').drop_duplicates(subset="employee")

employee.shape
f,ax = plt.subplots(1,1,figsize=(15,10))

red_emp = employee.drop(['companyAlias','lastParticipationDate','wday'],axis=1)

sns.heatmap(red_emp.corr())

plt.title('Features Correlation Heatmap',fontsize=24)

plt.show()
author_comments = cc[['employee','commentId']].drop_duplicates(subset='commentId')

author_dict = {commentId:author_comments['employee'].values[i] for i,commentId in enumerate(author_comments['commentId'].values)}

comments = [commentId for i,commentId in enumerate(author_comments['commentId'].values)]
#this is too computational intensive I will work on subsets

#authors = [author_dict[commentId] if commentId in comments else -1 for commentId in ci['commentId'].values]
from nxviz import CircosPlot
votes_feature = votes[['employee','vote','wday','yday']]

votes_feature['yday'] = votes['yday']+ votes['year']*365

dummies = pd.get_dummies(votes_feature['yday'])

for i,row in enumerate(votes_feature.values):

    dummies.loc[i,row[3]]= row[1]

votes_feature=pd.concat([votes_feature,dummies],axis=1)
dummies = votes_feature.groupby('employee',as_index=False).sum().drop(['vote','wday','yday','employee'],axis=1)

dummies.head(2)
from sklearn.decomposition import PCA



#votes_feature = votes[['vote','wday','yday']]

pca = PCA(n_components=6)

pca.fit(dummies)

components = pca.transform(dummies)

components = pd.DataFrame(components,columns=['c1','c2','c3','c4','c5','c6'])
components.head(2)
tocluster = pd.DataFrame(components[['c6','c4']])
from sklearn.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering



from sklearn.metrics import silhouette_score



clusterer = KMeans(n_clusters=4,random_state=42).fit(tocluster)

centers = clusterer.cluster_centers_

c_preds = clusterer.predict(tocluster)
import matplotlib

fig = plt.figure(figsize=(17,15))

colors = ['orange','blue','purple','green','brown','red','pink','white']

colored = [colors[k] for k in c_preds]

plt.scatter(tocluster['c6'],tocluster['c4'],  color = colored,s=10,alpha=0.5)

for ci,c in enumerate(centers):

    plt.plot(c[0], c[1], 'o', markersize=15, color='black', alpha=0.5, label=''+str(ci))

    plt.annotate(str(ci), (c[0],c[1]),fontsize=24,fontweight='bold')



plt.xlabel('x_values')

plt.ylabel('y_values')

plt.legend()

plt.show()
employee['cluster']=c_preds
tot_cluster = employee.groupby('cluster').count()['employee']

tot_churn = len(employee[employee['stillExists']==False])

tot_cluster
tmp = employee[employee['stillExists']==False].groupby('cluster',as_index='False').count()['employee']

churn_perc_totchurn = tmp/tot_churn

tmp=employee[employee['stillExists']==False].groupby('cluster',as_index='False').count()['employee']

churn_perc_totcluster = tmp/tot_cluster

fig1, axarr = plt.subplots(1,2,figsize=(17,10))

explode = (0, 0, 0.2, 0)

labels = 'Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'

axarr[0].pie(churn_perc_totchurn, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

axarr[0].axis('equal')  

axarr[0].set_title('Percentage of Churn in clusters with respect to the total of churn')

sns.barplot(x=[0,1,2,3],y=churn_perc_totcluster,ax=axarr[1])

axarr[1].set_title('Ratio of Churn in clusters with respect to the cluster population')

plt.show()
vote_cluster = employee.groupby('cluster',as_index='False').mean()['vote']

nvote_cluster = employee.groupby('cluster',as_index='False').mean()['numVotes']

likes_cluster = employee.groupby('cluster',as_index='False').mean()['likes']

dislikes_cluster = employee.groupby('cluster',as_index='False').mean()['dislikes']

liked_cluster = employee.groupby('cluster',as_index='False').mean()['liked']

disliked_cluster = employee.groupby('cluster',as_index='False').mean()['disliked']

fig,axarray = plt.subplots(3,2,figsize=(17,30))

sns.barplot(x=[0,1,2,3],y=vote_cluster,ax=axarray[0,1])

axarray[0,1].set_title('Happyness in Clusters')

sns.barplot(x=[0,1,2,3],y=vote_cluster,ax=axarray[0,0])

axarray[0,0].set_title('Happyness in Clusters')

sns.barplot(x=[0,1,2,3],y=nvote_cluster,ax=axarray[0,1])

axarray[0,1].set_title('#Votes in Clusters')

sns.barplot(x=[0,1,2,3],y=likes_cluster,ax=axarray[1,0])

axarray[1,0].set_title('#Likes Received in Clusters')

sns.barplot(x=[0,1,2,3],y=dislikes_cluster,ax=axarray[1,1])

axarray[1,1].set_title('#Dislikes Received in Clusters')

sns.barplot(x=[0,1,2,3],y=liked_cluster,ax=axarray[2,0])

axarray[2,0].set_title('#Liked Comments in Clusters')

sns.barplot(x=[0,1,2,3],y=disliked_cluster,ax=axarray[2,1])

axarray[2,1].set_title('#Disliked Comments in Clusters')

plt.show()
c1_sample = employee[employee['cluster']==2].head(5)
f,axarr = plt.subplots(3,2,figsize=(15,30))

sample = employee[employee['cluster']==2][0:6]

vtp = votes[votes['employee']==sample['employee'].iloc[0]]

axarr[0,0].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)

vtp = votes[votes['employee']==sample['employee'].iloc[1]]

axarr[0,1].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)

vtp = votes[votes['employee']==sample['employee'].iloc[2]]

axarr[1,0].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)

vtp = votes[votes['employee']==sample['employee'].iloc[3]]

axarr[1,1].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)

vtp = votes[votes['employee']==sample['employee'].iloc[4]]

axarr[2,0].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)

vtp = votes[votes['employee']==sample['employee'].iloc[5]]

axarr[2,1].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)



plt.xlim(0,365)

plt.show()
f,axarr = plt.subplots(3,2,figsize=(15,30))

sample = employee[employee['cluster']==1][20:26]

vtp = votes[votes['employee']==sample['employee'].iloc[0]]

axarr[0,0].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)

vtp = votes[votes['employee']==sample['employee'].iloc[1]]

axarr[0,1].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)

vtp = votes[votes['employee']==sample['employee'].iloc[2]]

axarr[1,0].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)

vtp = votes[votes['employee']==sample['employee'].iloc[3]]

axarr[1,1].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)

vtp = votes[votes['employee']==sample['employee'].iloc[4]]

axarr[2,0].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)

vtp = votes[votes['employee']==sample['employee'].iloc[5]]

axarr[2,1].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)



plt.xlim(0,365)

plt.show()