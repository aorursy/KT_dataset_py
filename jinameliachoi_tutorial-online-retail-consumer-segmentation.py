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
import datetime

import math

import matplotlib.pyplot as plt

%matplotlib inline



df = pd.read_excel(io='/kaggle/input/online-retail-data-set-from-uci-ml-repo/Online Retail.xlsx')

df.head(3)
df.info()
df = df[df['Quantity'] > 0]

df = df[df['UnitPrice'] > 0]

df = df[df['CustomerID'].notnull()]

print(df.shape)

df.isnull().sum()
# `Country` 칼럼은 주문 고객 국가. 주요 주문 고객은 영국인데 그 이외에 다른 값들도 많음.

# 이번 데이터 분석에서는 영국만 다룸

df['Country'].value_counts()
df = df[df['Country'] == 'United Kingdom']

print(df.shape)
# `UnitPrice` 와 `Quantity`를 곱하여 주문 금액 데이터 만들기

# `CustomerID`도 편하게 식별하기 위해 int 형으로 변경

df['sale_amount'] = df['Quantity'] * df['UnitPrice']

df['CustomerID'] = df['CustomerID'].astype(int)
# 해당 데이터 세트는 주문 횟수와 금액이 압도적으로 특정 고객에게 많은 특성을 가지고 있음

# 개인 고객의 주문과 소매점의 주문이 함께 포함되어 있기 때문에 

# top5 주문건수와 주문금액 가진 고객 추출

print(df['CustomerID'].value_counts().head(5))

print(df.groupby('CustomerID')['sale_amount'].sum().sort_values(ascending=False)[:5])
df.groupby(['InvoiceNo', 'StockCode'])['InvoiceNo'].count().mean()
# dataframe의 groupby()의 다중 연산을 위해 agg() 사용

# Recency는 InvoiceDate 칼럼의 max()에서 데이터 가공

# Frequency는 InvoiceNo 칼럼의 count(), Monentary value는 sale_amount칼럼의 sum()



aggregations = {

    'InvoiceDate' : 'max',

    'InvoiceNo' : 'count',

    'sale_amount' : 'sum'

}



cust_df = df.groupby('CustomerID').agg(aggregations)



# groupby된 결과 값을 변경

cust_df = cust_df.rename(columns = {'InvoiceDate' : 'Recency',

                                   'InvoiceNo' : 'Frequency',

                                   'sale_amount' : 'Monetary'

                                   }

                        )

cust_df = cust_df.reset_index()

cust_df.head(3)
# Recency 칼럼은 개별 고객 당 가장 최근의 주문. 오늘 날짜를 기준으로 가장 최근 주문 일자를 뺀 날짜.

# 오늘 날짜는 '현재 날짜'가 아님. 오늘 날짜는 2011년 12월 10일로 간주.

import datetime as dt



cust_df['Recency'] = dt.datetime(2011, 12, 10) - cust_df['Recency']

cust_df['Recency'] = cust_df['Recency'].apply(lambda x: x.days + 1)

print('cust_df row와 columns 건수는 ', cust_df.shape)

cust_df.head(3)
# 칼럼 값 별 히스토그램

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 4), nrows=1, ncols=3)

ax1.set_title('Recency Histogram')

ax1.hist(cust_df['Recency'])



ax2.set_title('Frequency Histogram')

ax2.hist(cust_df['Frequency'])



ax3.set_title('Monetary Histogram')

ax3.hist(cust_df['Monetary'])
# 세 변수 모두 왜곡된 데이터 값 분포도를 가지고 있음

cust_df[['Recency', 'Frequency', 'Monetary']].describe()
# 왜곡 정도가 매우 높은 데이터 세트에서 k-평균 군집을 적용하면 변별력이 떨어지는 군집화 수행

# scaling 과정 필요

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score, silhouette_samples



x_features = cust_df[['Recency', 'Frequency', 'Monetary']]

x_features_scaled = StandardScaler().fit_transform(x_features)



kmeans = KMeans(n_clusters=3, random_state=0)

labels = kmeans.fit_predict(x_features_scaled)

cust_df['cluster_label'] = labels



print('실루엣 스코어는 : {0:.3f}'.format(silhouette_score(x_features_scaled, labels)))
# 안정적인 실루엣 스코어 보여줌

# 각 군집별 실루엣 계수 알아보기

# 여러 개의 클러스터링 갯수를 list로 받아 각각 실루엣 계수는 면적으로 나타내는 그래프



def visualize_silhouette(cluster_lists, x_features):

    from sklearn.datasets import make_blobs

    from sklearn.cluster import KMeans

    from sklearn.metrics import silhouette_samples, silhouette_score

    

    import matplotlib.pyplot as plt

    import matplotlib.cm as cm

    import math

    

    # 입력값으로 클러스터링 갯수를 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 계수 구함

    n_cols = len(cluster_lists)

    

    # 리스트에 기재된 클러스터링 수만큼 sub figures를 가지는 axs 생성

    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)

    

    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 계수 시각화

    for ind, n_cluster in enumerate(cluster_lists):

        cluster = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)

        cluster_labels = cluster.fit_predict(x_features)

        

        sil_avg = silhouette_score(x_features, cluster_labels)

        sil_values = silhouette_samples(x_features, cluster_labels)

        

        y_lower = 10

        axs[ind].set_title('Number of Cluster : ' + str(n_cluster) + '\n' \

                          'Silhouette Score : ' + str(round(sil_avg, 3)))

        axs[ind].set_xlabel('The silhouette coefficient values')

        axs[ind].set_ylabel('Cluster label')

        axs[ind].set_xlim([-0.1, 1])

        axs[ind].set_ylim([0, len(x_features) + (n_cluster + 1) * 10])

        axs[ind].set_yticks([])

        axs[ind].set_xticks([0, .2, .4, .6, .8, 1])

        

        # 클러스터링 갯수별로 막대 그래프 표현

        for i in range(n_cluster):

            ith_cluster_sil_values = sil_values[cluster_labels == i]

            ith_cluster_sil_values.sort()

            

            size_cluster_i = ith_cluster_sil_values.shape[0]

            y_upper = y_lower + size_cluster_i

            

            color = cm.nipy_spectral(float(i) / n_cluster)

            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \

                                  facecolor = color, edgecolor = color, alpha = 0.7)

            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + 10

            

        axs[ind].axvline(x=sil_avg, color = 'red', linestyle = '--')
def visualize_kmeans_plot_multi(cluster_lists, X_features):

    

    from sklearn.cluster import KMeans

    from sklearn.decomposition import PCA

    import pandas as pd

    import numpy as np

    

    # plt.subplots()으로 리스트에 기재된 클러스터링 만큼의 sub figures를 가지는 axs 생성 

    n_cols = len(cluster_lists)

    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)

    

    # 입력 데이터의 FEATURE가 여러개일 경우 2차원 데이터 시각화가 어려우므로 PCA 변환하여 2차원 시각화

    pca = PCA(n_components=2)

    pca_transformed = pca.fit_transform(X_features)

    dataframe = pd.DataFrame(pca_transformed, columns=['PCA1','PCA2'])

    

     # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 KMeans 클러스터링 수행하고 시각화

    for ind, n_cluster in enumerate(cluster_lists):

        

        # KMeans 클러스터링으로 클러스터링 결과를 dataframe에 저장. 

        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)

        cluster_labels = clusterer.fit_predict(pca_transformed)

        dataframe['cluster']=cluster_labels

        

        unique_labels = np.unique(clusterer.labels_)

        markers=['o', 's', '^', 'x', '*']

       

        # 클러스터링 결과값 별로 scatter plot 으로 시각화

        for label in unique_labels:

            label_df = dataframe[dataframe['cluster']==label]

            if label == -1:

                cluster_legend = 'Noise'

            else :

                cluster_legend = 'Cluster '+str(label)           

            axs[ind].scatter(x=label_df['PCA1'], y=label_df['PCA2'], s=70,\

                        edgecolor='k', marker=markers[label], label=cluster_legend)



        axs[ind].set_title('Number of Cluster : '+ str(n_cluster))    

        axs[ind].legend(loc='upper right')

    

    plt.show()
visualize_silhouette([2, 3, 4, 5], x_features_scaled)

visualize_kmeans_plot_multi([2, 3, 4, 5], x_features_scaled)
# Recency, Frequency, Monetary에 log 변환

cust_df['Recency_log'] = np.log1p(cust_df['Recency'])

cust_df['Frequency_log'] = np.log1p(cust_df['Frequency'])

cust_df['Monetary_log'] = np.log1p(cust_df['Monetary'])



# log 변환 데이터에 scaling 진행

x_features = cust_df[['Recency_log', 'Frequency_log', 'Monetary_log']].values

x_features_scaled = StandardScaler().fit_transform(x_features)



kmeans = KMeans(n_clusters=3, random_state=0)

labels = kmeans.fit_predict(x_features_scaled)

cust_df['cluster_label'] = labels



print('실루엣 스코어는 : {0:.3f}'.format(silhouette_score(x_features_scaled, labels)))
visualize_silhouette([2, 3, 4, 5], x_features_scaled)

visualize_kmeans_plot_multi([2, 3, 4, 5], x_features_scaled)