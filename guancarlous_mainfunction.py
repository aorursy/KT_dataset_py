# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as nx 
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.
TeamData = pd.read_csv('../input/Teams.csv',index_col='Id')
TeamMembershipData = pd.read_csv('../input/TeamMemberships.csv',index_col = 'Id')
CompetitionData = pd.read_csv('../input/Competitions.csv',index_col = 'Id')
UserData = pd.read_csv('../input/Users.csv',index_col='Id')
TeamMembershipData.head(5)
TeamMembershipData.info()
TeamData.info()
TeamMembershipData.isnull().any()
TeamMembershipData.RequestDate.isna().value_counts()
TeamMembershipData.RequestDate.notna().sum()/len(TeamMembershipData)
TeamMembershipData.fillna(method='bfill', axis = 0,inplace=True)
TeamData.isnull().any()
TeamData.TeamLeaderId.isnull().value_counts()
TeamData[TeamData.TeamLeaderId.isnull()].head(10)
missval_index_1 = TeamData[TeamData.TeamLeaderId.isnull()].index
TeamData.drop(missval_index_1,inplace=True)
test = pd.read_csv('../input/Teams.csv')
sit1 = pd.concat([test.CompetitionId,test.MedalAwardDate,test.Medal],axis = 1)
data2 = sit1[sit1.CompetitionId==2439]
data2.MedalAwardDate.value_counts()
data2.MedalAwardDate.isnull().value_counts()
data2.Medal.value_counts()
index_1_val = TeamData.ScoreFirstSubmittedDate.isnull()

TeamData[index_1_val].Medal = TeamData[index_1_val].Medal.fillna(5)
TeamData.MedalAwardDate.fillna('1/1/2199',inplace = True)
TeamData.Medal.fillna(4,inplace = True)

TeamData.PublicLeaderboardRank.isnull().value_counts()

TeamData.PrivateLeaderboardRank.isnull().value_counts()
TeamData.PublicLeaderboardRank.fillna('99999',inplace = True)
TeamData.PrivateLeaderboardRank.fillna('99999',inplace = True)
TeamData.ScoreFirstSubmittedDate.fillna('1/1/2199',inplace = True)
TeamData.LastSubmissionDate.fillna('1/1/2199',inplace = True)
cols = ['PublicLeaderboardSubmissionId','PrivateLeaderboardSubmissionId']
TeamData.drop(cols,axis = 1,inplace = True)

TeamData.head(10)
TeamData.index.name = 'TeamId'

MergeValue = pd.merge(TeamData,TeamMembershipData,on = 'TeamId',how='right')
MergeValue.drop('TeamName',inplace=True,axis =1)

MergeValue.info()
MergeValue.to_csv('TeamRelation.csv')
MergeValue.isnull().any()
groupresult =MergeValue.groupby(['UserId','TeamLeaderId'])
MergeValue = pd.DataFrame(groupresult.Medal.min()).reset_index()
TierMerge = pd.DataFrame(UserData['PerformanceTier'])
TierMerge.index.name= 'TeamLeaderId'

result = pd.merge(TierMerge,MergeValue,on='TeamLeaderId',how='right').rename(columns={'PerformanceTier':'LeaderTier'})

result.drop('TeamLeaderId',axis = 1,inplace = True)

result.LeaderTier.isnull().value_counts()

result.dropna(inplace=True)
result.to_csv('TeamLeaderTier.csv')
competitionData = pd.read_csv('../input/Competitions.csv')
competitionData.head(5)
competitionData.info()
competitionData.columns
competitionData.HostSegmentTitle.value_counts()
competitionData.columns
competitionData.isnull().any()
competitionData.ForumId.isnull().value_counts()
competitionData.ForumId.fillna(0,inplace=True)
competitionData.Subtitle.isnull().value_counts()
competitionData.Subtitle.value_counts()
competitionData.drop('Subtitle',axis =1 ,inplace = True)
competitionData.OrganizationId.isnull().value_counts()
competitionData.OrganizationId.fillna(0,inplace=True)
competitionData.HostName.value_counts()
competitionData.HostName.isnull().value_counts()
competitionData.HostName.fillna('None',inplace=True)
competitionData.LeaderboardDisplayFormat.value_counts()
competitionData.EvaluationAlgorithmName.value_counts()
competitionData.EvaluationAlgorithmName.isnull().value_counts()
competitionData.EvaluationAlgorithmName.fillna('None',inplace = True)
competitionData.ValidationSetName.value_counts()
competitionData.ValidationSetName.isnull().value_counts()
competitionData.ValidationSetName.fillna('None',inplace=True)
competitionData.MaxTeamSize.value_counts()
competitionData.MaxTeamSize.isnull().value_counts()
competitionData.MaxTeamSize.fillna(-1,inplace=True)
competitionData.RewardType.value_counts()
competitionData.RewardType.isnull().value_counts()
competitionData.RewardType.fillna('None',inplace=True)
competitionData.RewardQuantity.isnull().value_counts()
competitionData.RewardQuantity.value_counts()
competitionData.RewardQuantity.fillna(-1,inplace = True)
enableTimeData = competitionData.EnabledDate.str.split('/|\s',expand=True)[[0,2]]
enableTimeData[0] = enableTimeData[0].astype('int')//4+1
competitionData.EnabledDate = enableTimeData[2].astype('str')+enableTimeData[0].astype('str')

deadLineTimeData = competitionData.DeadlineDate.str.split('/|\s',expand=True)[[0,2]]
deadLineTimeData[0] = enableTimeData[0].astype('int')//4+1
competitionData.DeadlineDate = deadLineTimeData[2].astype('str')+deadLineTimeData[0].astype('str')

competitionData.head(5)
dropCols = competitionData.columns[competitionData.isnull().any()]
competitionData.drop(dropCols,inplace=True,axis =1)
competitionData.isnull().any()
competitionData.head(5)
competitionData.dtypes
competitionData.HostSegmentTitle.value_counts()
competitionDataResult = competitionData.drop(['ForumId','OrganizationId','ForumId','Slug','Title','EvaluationAlgorithmAbbreviation','EvaluationAlgorithmName','HostName'],axis = 1)
competitionDataResult[['EnabledDate','DeadlineDate']] = competitionDataResult[['EnabledDate','DeadlineDate']].astype('int64')
ObjectColumns = competitionDataResult.columns[(competitionDataResult.dtypes=='object').values]
competitionDataResult[ObjectColumns]
competitionDataResult.info()
competitionDataResult[competitionDataResult.columns[competitionDataResult.dtypes=='int64']].head(5)
competitionDataResult[competitionDataResult.columns[competitionDataResult.dtypes=='float64']].head(5)
competitionDataResult[competitionDataResult.columns[competitionDataResult.dtypes=='float64']].UserRankMultiplier.value_counts()
competitionDataResult[competitionDataResult.columns[competitionDataResult.dtypes=='bool']].head(5)
competitionDataResult.drop('CompetitionTypeId',axis = 1,inplace=True)
competitionDataResult[competitionDataResult.columns[competitionDataResult.dtypes=='bool']] = competitionDataResult[competitionDataResult.columns[competitionDataResult.dtypes=='bool']].astype('int')
competitionDataResult[competitionDataResult.columns[competitionDataResult.dtypes=='object']]
competitionDataResult.set_index('Id',inplace=True)
competitionDataResult
HotCodeData = pd.get_dummies(competitionDataResult)
HotCodeData
HotCodeData.to_csv('CompetitionData_hotCode.csv')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import datetime as dt
CompetitionData_HotCode = pd.read_csv('CompetitionData_hotCode.csv',index_col=0)
CompetitionData_HotCode.head(5)
CompetitionData_HotCode.info()
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=(0.90*(1-0.90)))
CompetitionData_selFeature = pd.DataFrame(sel.fit_transform(CompetitionData_HotCode),index=CompetitionData_HotCode.index)

CompetitionData_selFeature.head(5)
from sklearn.preprocessing import scale

CompetitionData_Scale = pd.DataFrame(scale(CompetitionData_selFeature.values),index=CompetitionData_selFeature.index)

CompetitionData_Scale
from sklearn.decomposition import PCA
pca = PCA(n_components=0.90)
pca.fit(CompetitionData_Scale)

print(pca.n_components)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

pca.explained_variance_.shape

CompetitionData_Scale_PCA = pd.DataFrame(pca.transform(CompetitionData_Scale),index=CompetitionData_Scale.index)

CompetitionData_Scale_PCA.head(5)
CompetitionData_Scale_PCA.shape
pca = PCA(n_components=3)
pca.fit(CompetitionData_Scale)
print(pca.n_components)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
CompetitionData_PCA_3D = pd.DataFrame(pca.transform(CompetitionData_Scale),index=CompetitionData_Scale.index)
CompetitionData_PCA_3D.head(5)
CompetitionData_PCA_3D.shape
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(figsize=(7,5))
ax = Axes3D(fig)
x = CompetitionData_PCA_3D[0]
y = CompetitionData_PCA_3D[1]
z = CompetitionData_PCA_3D[2]
ax.scatter(z,x,y)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.show()

from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def Visual_K_and_Meandisortions(X_value,start,end):
    start_time = dt.datetime.now()

    K = range(start,end)
    meandisortions  = []
    for k in K:
        costFunction = 0
        for i in range(40):
            Minkmeams = MiniBatchKMeans(n_clusters=k)
            Minkmeams.fit(X_value)
            costFunction += sum(np.min(
                cdist(X_value,Minkmeams.cluster_centers_,'euclidean'),axis = 1))/X_value.shape[0]
        costFunction/=40
        meandisortions.append(costFunction)
        print('%d=='%k,end='=')

    end_time = dt.datetime.now()
    print((end_time - start_time).seconds)
    plt.plot(K,meandisortions,'gx-')
    plt.xlabel('k')
    plt.ylabel('Mean Cost Function')
    plt.show()
    return Minkmeams,plt,meandisortions
Visual_K_and_Meandisortions(CompetitionData_Scale_PCA,1,40)
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

def Visual_K_and_silhouette(X_value,start,end):
    start_time = dt.datetime.now()

    K = range(start,end)
    KMeans_silhouette_re  = []
    for k in K:
        KMeans_silhouette = 0
        for i in range(0,70):
            Minkmeams = MiniBatchKMeans(n_clusters=k)
            Minkmeams.fit(X_value)
            KMeans_silhouette += silhouette_score(X_value,Minkmeams.labels_)
        KMeans_silhouette /= 70
        KMeans_silhouette_re.append(KMeans_silhouette)
        print('%d=='%k,end='=')

    end_time = dt.datetime.now()
    print((end_time - start_time).seconds)
    plt.plot(K,KMeans_silhouette_re,'gx-')
    plt.xlabel('k')
    plt.ylabel('silhouette_score')
    plt.show()
    return Minkmeams,plt,KMeans_silhouette_re
Mimi,_,KMeans_silhouette_re = Visual_K_and_silhouette(CompetitionData_Scale_PCA,2,40)
Mimi,_,KMeans_silhouette_re = Visual_K_and_silhouette(CompetitionData_Scale_PCA,2,40)
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def Visual_K_and_AggCluster_silhouette(X_value,start,end):
    start_time = dt.datetime.now()

    K = range(start,end)
    AgglomerativeCluster_silhouette_re  = []
    for k in K:

        AgglomerativeCluster = AgglomerativeClustering(linkage='average',n_clusters=k)
        AgglomerativeCluster.fit(X_value)
        AgglomerativeCluster_silhouette = silhouette_score(X_value,AgglomerativeCluster.labels_,metric='euclidean',random_state=0)
        AgglomerativeCluster_silhouette_re.append(AgglomerativeCluster_silhouette)
        print('%d=='%k,end='=')

    end_time = dt.datetime.now()
    print((end_time - start_time).seconds)
    plt.plot(K,AgglomerativeCluster_silhouette_re,'gx-')
    plt.xlabel('k')
    plt.ylabel('silhouette_score')
    plt.show()
    return AgglomerativeCluster_silhouette,plt,AgglomerativeCluster_silhouette_re
_,_,AgglomerativeCluster_silhouette_re = Visual_K_and_AggCluster_silhouette(CompetitionData_Scale_PCA,2,40)
x= range(2,40)
y1 = KMeans_silhouette_re
y2 = AgglomerativeCluster_silhouette_re

plt.plot(x,y1,'ro-')
plt.plot(x,y2,'bx-',label = 'AgglomerativeCluster')

plt.legend((u'KMeans', u'Agglomerative Cluster'),loc='best')# sets our legend for our graph.


plt.xlabel('Cluster number')
plt.ylabel('Silhouette')
plt.show()
from sklearn.cluster import AffinityPropagation

AP = AffinityPropagation()
AP.fit(CompetitionData_Scale_PCA)
AP.cluster_centers_indices_ 
AP.labels_
import seaborn as sns 
sns.distplot(AP.labels_)
from sklearn.cluster import Birch
bir = Birch(branching_factor=10,n_clusters=10)
bir.fit(CompetitionData_Scale_PCA)
bir.predict(CompetitionData_Scale_PCA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

plt.figure(figsize=(15,7))

#计算距离关联矩阵，两两样本间的欧式距离
row_clusters = linkage(pdist(CompetitionData_Scale_PCA,metric='euclidean'),method='complete')#使用抽秘籍距离矩阵
#层次聚类树
row_dendr = dendrogram(row_clusters,labels=CompetitionData_Scale_PCA.index)

plt.ylabel('Euclidean distance')
AgglomerativeCluster = AgglomerativeClustering(linkage='average',n_clusters=8)
AgglomerativeCluster.fit(CompetitionData_Scale_PCA)
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.3,min_samples=6)
dbscan.fit(CompetitionData_Scale_PCA)
print(dbscan)
print(dbscan.labels_)
VisualData_PCA_Model = PCA(n_components=3)
VisualData_PCA_Model.fit(CompetitionData_Scale)
VisualData = VisualData_PCA_Model.transform(CompetitionData_Scale)
VisualData_3D = pd.DataFrame(VisualData,index=CompetitionData_Scale.index)
VisualData_3D.head(5)
def DBSCAN_Result_Visual(dbscan,VisualData,CompetitionData_DBSCAN_To_Visual):
    dbscan_labels = pd.DataFrame(dbscan.labels_,CompetitionData_DBSCAN_To_Visual.index).rename(columns={0:'Labels'})
    CompetitionData_DBSCAN_To_Visual_AfMerge = pd.merge(CompetitionData_DBSCAN_To_Visual,dbscan_labels,on='Id')
    colors = np.array(['#DC143C','#FF69B4','#DA70D6','#EE82EE','#8B008B','#9400D3','#8A2BE2','#6A5ACD','#F8F8FF','#191970','#4169E1',
                       '#778899','#F0F8FF','#87CEEB','#B0E0E6','#E1FFFF','#00FFFF','#008B8B','#20B2AA','#00FA9A','#3CB371','#90EE90',
                       '#32CD32','#008000','#7CFC00','#F5F5DC','#FFFFE0','#BDB76B','#F0E68C','#DAA520','#F5DEB3','#FFEFD5','#FAEBD7',
                       '#FFE4C4','#CD853F','#D2691E','#A0522D','#FF4500','#FFE4E1','#F08080','#FF0000','#8B0000','#F5F5F5','#C0C0C0',
                       '#696969',
    ])
    CompetitionData_DBSCAN_To_Visual_AfMerge.Labels = colors[CompetitionData_DBSCAN_To_Visual_AfMerge.Labels]

    # 可视化模块
    fig = plt.figure(figsize=(10,7))
    ax = Axes3D(fig)
    x = VisualData[0]
    y = VisualData[1]
    z = VisualData[2]
    c = CompetitionData_DBSCAN_To_Visual_AfMerge.Labels
    ax.scatter(z,x,y,c=c)
    #ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    #ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    #ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()

DBSCAN_Result_Visual(dbscan,VisualData_3D,CompetitionData_Scale_PCA)
result = pd.DataFrame()
for i in range(1,15):
    if i == 1:
        result = pd.DataFrame([np.arange(0.1,1.6,0.1),np.ones(15)*i])
    else:
        result = pd.concat((result,pd.DataFrame([np.arange(0.1,1.6,0.1),np.ones(15)*i])),axis = 1)
result = result.T
result.reset_index(inplace=True)
result.drop('index',axis = 1,inplace=True)
result.index.name = 'Id'
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 

def Silhoutte_DBSCAN(list,values):
    result = []
    for i in range(0,len(list)):
        dbscan = DBSCAN(eps = list[0][i],min_samples = list[1][i])
        dbscan.fit(values)
        try:
            result.append(silhouette_score(values,dbscan.labels_,metric='euclidean'))
        except:
            result.append(0)
            pass
    MergeValue = pd.DataFrame(result,index=list.index)
    MergeValue.index.name='Id'
    MergeValue.rename(columns={0:'Slihoutte'},inplace=True)
    VisualValue = pd.merge(list,MergeValue,on='Id')
    
    fig = plt.figure(figsize=(7,5))
    ax = Axes3D(fig)
    x = VisualValue[0]
    y = VisualValue[1]
    z = VisualValue.Slihoutte
    ax.scatter3D(x,y,z)
    #ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    #ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    #ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    return VisualValue

TDVisualData = Silhoutte_DBSCAN(result,CompetitionData_Scale_PCA)
AgglomerativeCluster = AgglomerativeClustering(linkage='average',n_clusters=4)
AgglomerativeCluster.fit(CompetitionData_Scale_PCA)
AgglomerativeCluster_silhouette = silhouette_score(CompetitionData_Scale_PCA,AgglomerativeCluster.labels_,random_state=0)

pd.DataFrame(AgglomerativeCluster.labels_)[0].value_counts()
CompetitionData_Scale_PCA.iloc[10:20]
list(range(6))
birchCluster2 = Birch(n_clusters=4)
silhouette = 0
for i in range(len(CompetitionData_Scale_PCA)//1000):
    temp = CompetitionData_Scale_PCA.iloc[i*1000:(i+1)*1000]
    birchCluster2.partial_fit(temp)
    silhouette += silhouette_score(temp,birchCluster2.labels_,metric='euclidean')
    print(i,end='-')
silhouette/(len(CompetitionData_Scale_PCA)//1000+1)
birchCluster1 = Birch(n_clusters=4)
birchCluster1.fit(CompetitionData_Scale_PCA)
silhouette_score(CompetitionData_Scale_PCA,birchCluster1.labels_,metric='euclidean')
birchCluster = Birch(n_clusters=4)
birchCluster.partial_fit(CompetitionData_Scale_PCA.iloc[21:30])
birchCluster.labels_
from sklearn.ensemble import IsolationForest
iTree = IsolationForest(contamination=0.005)
iTree.fit(CompetitionData_Scale_PCA)
DropVlaue = pd.DataFrame(iTree.predict(CompetitionData_Scale_PCA),index = CompetitionData_Scale_PCA.index)
CompetitionData_Scale_PCA_OutLiners = CompetitionData_Scale_PCA[DropVlaue[0]==1]
CompetitionData_HotCode[DropVlaue[0]==-1]
_,_,AgglomerativeCluster_silhouette_re = Visual_K_and_AggCluster_silhouette(CompetitionData_Scale_PCA_OutLiners,2,40)
_,_,KMeans_silhouette_re = Visual_K_and_silhouette(CompetitionData_Scale_PCA,2,40)
x= range(2,40)
y1 = KMeans_silhouette_re
y2 = AgglomerativeCluster_silhouette_re

plt.plot(x,y1,'ro-')
plt.plot(x,y2,'bx-',label = 'AgglomerativeCluster')

plt.legend((u'KMeans', u'Agglomerative Cluster'),loc='best')# sets our legend for our graph.


plt.xlabel('Cluster number')
plt.ylabel('Silhouette')
plt.show()
AgglomerativeCluster = AgglomerativeClustering(linkage='average',n_clusters=4)
AgglomerativeCluster.fit(CompetitionData_Scale_PCA_OutLiners)
AgglomerativeCluster_silhouette = silhouette_score(CompetitionData_Scale_PCA_OutLiners,AgglomerativeCluster.labels_,random_state=0)

pd.DataFrame(AgglomerativeCluster.labels_)[0].value_counts()
AgglomerativeCluster = AgglomerativeClustering(linkage='average',n_clusters=4)
AgglomerativeCluster.fit(CompetitionData_Scale_PCA)
MergeValue = pd.DataFrame(AgglomerativeCluster.labels_,index = CompetitionData_Scale_PCA.index).rename(columns = {0:'Labels'})
ClusterResult_Save = pd.merge(CompetitionData_Scale_PCA,MergeValue,on = 'Id')
ClusterResult_Save.Labels.value_counts()
AgglomerativeCluster.children_
CompetitionData_HotCode[ClusterResult_Save.Labels==2]
ClusterResult_Save
colors = np.array(['#DC143C','#FF69B4','#DA70D6','#EE82EE','#8B008B','#9400D3','#8A2BE2','#6A5ACD','#F8F8FF','#191970','#4169E1',
                   '#778899','#F0F8FF','#87CEEB','#B0E0E6','#E1FFFF','#00FFFF','#008B8B','#20B2AA','#00FA9A','#3CB371','#90EE90',
                   '#32CD32','#008000','#7CFC00','#F5F5DC','#FFFFE0','#BDB76B','#F0E68C','#DAA520','#F5DEB3','#FFEFD5','#FAEBD7',
                   '#FFE4C4','#CD853F','#D2691E','#A0522D','#FF4500','#FFE4E1','#F08080','#FF0000','#8B0000','#F5F5F5','#C0C0C0',
                   '#696969',
])
ClusterResult_Save.Labels = colors[ClusterResult_Save.Labels]
fig = plt.figure(figsize=(7,5))
ax = Axes3D(fig)
x = ClusterResult_Save[0]
y = ClusterResult_Save[1]
z = ClusterResult_Save[2]
c = ClusterResult_Save.Labels
ax.scatter(z,x,y,c=c)
#ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
#ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
#ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.show()

ClusterResult_Save.to_csv('CompetitionClusterResult.csv')
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
ls
UserData = pd.read_csv('../input/Users.csv',index_col=0)
UserFollow = pd.read_csv('../input/UserFollowers.csv',index_col=0)
UserOrganization = pd.read_csv('../input/UserOrganizations.csv',index_col=0)
UserAchievements = pd.read_csv('../input/UserAchievements.csv',index_col=0)
UserData.info()
UserData.head(5)
UserFollow.info()
UserData.reset_index(inplace=True)
UserOrganization.head(5)
UserOrganization.info()
UserFollow.head(5)
UserFollow.info()
UserAchievements.head(5)
UserAchievements.info()
UserData.isnull().any()
UserData[UserData.UserName.isnull()]
# 查看了前5行的数据
UserData[UserData.DisplayName.isnull()].head(5) 
UserData[UserData.DisplayName.isnull()].shape
UserData.DisplayName.fillna('None',inplace=True)
UserData.dropna(inplace=True)
UserData.isnull().any()
UserData.head(5)
TimeAndNumber = pd.DataFrame(UserData['RegisterDate'])
regTimeVal = UserData.RegisterDate.str.split('/',expand=True)
regTimeVal.head(5)
UserData.head(5)
regTimeVal[0] = (regTimeVal[0].astype('int')//4+1).astype('object')
regTimeVal.shape
UserData.RegisterDate = regTimeVal[2].astype('str')+regTimeVal[0].astype('str')
UserFollow.head(5)
userFollowGroupResult = UserFollow.groupby('UserId')
userFollowNumber = pd.DataFrame(userFollowGroupResult.size())
# 进行一些微调，比如Index和列名
userFollowNumber.reset_index(inplace=True)
userFollowNumber.rename(columns={'UserId':'Id',0:'FollowersNumber'},inplace=True)

userFollowNumber.head(5)
UserData_add_followerNum = pd.merge(UserData,userFollowNumber,on='Id',how='left')
UserData_add_followerNum.head(5)
UserData_add_followerNum.isnull().any()
UserData_add_followerNum.FollowersNumber.fillna(0,inplace=True)
UserData_add_followerNum.shape
UserAchievements.head(5)
UserAchievements[UserAchievements.UserId==368].head(5)
UserAchievements[UserAchievements.AchievementType=='Scripts'].head(5)
UserAchievements.UserId.value_counts().sort_values(ascending=True).head(5)
cols = ['UserId','CurrentRanking','HighestRanking','TotalGold','TotalSilver','TotalBronze']
UserAchievements_to_process = UserAchievements[cols]
UserAchievements_to_process.head(5)
UserAchievements_complete = UserAchievements_to_process.groupby('UserId').sum()
UserAchievements_complete.reset_index(inplace=True)
UserAchievements_complete.rename(columns={'UserId':'Id'},inplace=True)
UserAchievements_complete.head(5)
UserData_Follow_Achievements = pd.merge(UserAchievements_complete,UserData_add_followerNum,on = 'Id',how='left')
UserData_Follow_Achievements.head(5)
UserData_Follow_Achievements.isnull().any()
UserData_Follow_Achievements[UserData_Follow_Achievements.UserName.isnull()]
UserData_Follow_Achievements.dropna(inplace=True)
UserData_Follow_Achievements.isnull().any()
UserOrganization.head(5)
UserOrganization.rename(columns={'UserId':'Id'},inplace=True)
UserOrganization.index.name = 'Index'
UserData_Follow_Organiz_Ach = pd.merge(UserData_Follow_Achievements,UserOrganization,on='Id',how='left')
UserData_Follow_Organiz_Ach_dropDup = UserData_Follow_Organiz_Ach.drop_duplicates(subset=['Id'],keep='first')
UserData_Follow_Organiz_Ach_dropDup.shape
UserData_Follow_Organiz_Ach_dropDup[UserData_Follow_Organiz_Ach_dropDup.OrganizationId.isnull()].head(5)
UserData_Follow_Organiz_Ach_dropDup[UserData_Follow_Organiz_Ach_dropDup.JoinDate.isnull()].shape
UserData_Follow_Organiz_Ach_dropDup.OrganizationId.fillna(0,inplace=True)
UserData_Follow_Organiz_Ach_dropDup.JoinDate.fillna('1/1/2000',inplace=True)
UserData_Follow_Organiz_Ach_dropDup.shape
# UserData_Follow_Organiz_Ach_dropDup.to_csv('userDataAfterEngineering.csv')
CompetitionInf = pd.read_csv('CompetitionClusterResult.csv',index_col=0)
RelationInf = pd.read_csv('TeamRelation.csv',index_col=0)
CompetitionInf.head(5)
CompetitionInf.index.name='CompetitionId'
CompetitionInf.rename(columns={'Labels':'CompetitionType'},inplace=True)
ComMergeValue = pd.DataFrame(CompetitionInf.CompetitionType).reset_index()
ComMergeValue
RelationInf.head(5)
RelationInf.UserId.value_counts()
UserFEResult = pd.merge(ComMergeValue,RelationInf,on='CompetitionId',how='right')
UserFEResult
UserFEResult.isnull().any()
UserFEGroup = UserFEResult.groupby(['UserId','CompetitionType'])

UserFEGroup_DF = pd.DataFrame(UserFEGroup.size()).reset_index()

UserFEGroup_DF.head(5)
Merge_To_User_Value = UserFEGroup_DF.pivot(columns='CompetitionType',index='UserId').fillna(0)
Merge_To_User_Value.head(5)
Merge_To_User_Value.shape
Merge_To_User_Value.index.name='Id'
MergeResult = pd.merge(UserData_Follow_Organiz_Ach_dropDup,Merge_To_User_Value,how= 'left',on = 'Id')
MergeResult.dropna(inplace=True)
MergeResult.shape
TeamTier = pd.read_csv('TeamLeaderTier.csv',index_col=0)
TeamTier = TeamTier[['LeaderTier','UserId']]
TeamTier.rename(columns={'UserId':'Id'},inplace=True)
TeamTier
LastResult = pd.merge(MergeResult,TeamTier,how = 'left',on = 'Id')
LastResult.isnull().any()
LastResult.fillna(-1,inplace=True)
LastResult.to_csv('User_Relation_Competition.csv')