# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import ExcelFile
data = pd.read_excel('../input/splunk-matrix-v2/Dashboard_Query_Matrix.xlsx', sheet_name='Splunk Query Analysis')
data.head()
data.isnull().sum()
data['Query Index'] = data['Query Index'].fillna(method='ffill')
data.head(10)
# incorrect = data[(data['Field Index'] == 7) | (data['Field Index'] == 8)].index
# incorrect
# data.loc[incorrect, 'Field Index'] = 6
# data.loc[incorrect, :]
# data['Field Index'] = data['Field Index'].map(lambda x : x - 2 if (type(x) == int) and (x > 8) else x)
# data.head(50)
# data.to_csv('refine_index.csv', index=False)
# data.head(50)
origin_data_length = len(data)
print('Total fields number used in the dashboards: %d' % origin_data_length)
data = data[~(data['Field Index'].isnull()) & (data['Field Index'] != 'l')]
removed_internal_fields = len(data)
print('Interal filed used only by logstash: %d' % (origin_data_length - removed_internal_fields))
query_fields = data.groupby('Query Index')['Field Index'].unique()
query_fields.index = query_fields.index.astype(int)
print('Charts number: %d' % len(query_fields.index))
unique_fields_number = data['Field Index'].unique()
print('Unique fields number: %d' % len(unique_fields_number))
charts_fields_count = query_fields.map(lambda x : len(x))
plt.figure(figsize=(14, 8))
sns.barplot(query_fields.index, charts_fields_count)
plt.title("Charts & Fields distribution")
plt.ylabel('Fileds number', fontsize=14)
plt.xlabel('Chart Index', fontsize=14)
def plot_classify(charts_df, ax, title):
    color_names = ['r', 'g', 'b', 'black', 'yellow', 'pink']
    marker = ['.', '*', '+', '2', 'd','^']
    class_list = charts_df['class'].unique()
#     class_index = 0
    for class_value in class_list:
        class_query = charts_df[charts_df['class'] == class_value]
        query_idx = []
        field_idx = []
        for index, row in class_query.iterrows():
            for value in row['fields']:
                query_idx.append(index)
                field_idx.append(value)
        ax.scatter(field_idx, query_idx, c=color_names[class_value], 
                   label='index%d' % class_value, marker=marker[class_value])
#         class_index += 1
    
    ax.set_ylim(0,35)
    ax.set_yticks(range(0,36))
    ax.set_xticks(range(0,47))
    ax.grid(which='major', axis='y')
#     ax.set_minorticks_on()
#     ax.set_grid(b=False, which='minor')
    ax.set_xlabel('Field Index')
    ax.set_ylabel('Chart Index')
    ax.set_title(title, fontsize=16)
    ax.legend()
class_array = np.ones(len(query_fields), dtype=int)
query_df = pd.DataFrame({'fields': query_fields.values, 'class':class_array},index=query_fields.index)
fig, axs = plt.subplots(1,1, figsize=(16, 8))
plot_classify(query_df, axs, 'Fields distribution with 1 Index')
def convert_list_to_onehot(arr):
    row = []
    for i in range(46):
        value_i = i + 1
        if value_i in arr:
            row.append(1)
        else:
            row.append(0)
    return row
features = query_fields.map(convert_list_to_onehot)
feature_array = []
for value in features:
    feature_array.append(value)
feature_array = np.asarray(feature_array)
array_df = pd.DataFrame(feature_array, index=query_df.index, columns=range(1,47))
array_df
#relation_df = array_df.transpose().corr()
relation_df = array_df.dot(array_df.transpose()).applymap(lambda x: 1 if x >=1 else 0)
fig, axs = plt.subplots(1,2,figsize=(28,10))
matrix_prod  = sns.heatmap(relation_df, ax=axs[0])
matrix_prod.set_title('Matrix Product to represent Charts Relation', fontsize=16)

from sklearn.metrics.pairwise import pairwise_distances
jaccard_distance = pairwise_distances(feature_array, metric='jaccard', n_jobs=4)
jaccard_df = pd.DataFrame(jaccard_distance, columns=query_df.index, index=query_df.index)

im = sns.heatmap(jaccard_df, ax=axs[1])
im.set_title('Jaccard Distance for all the charts', fontsize=16)
from sklearn.cluster import AgglomerativeClustering
hier_jacc_cluster = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='average')
predicts = hier_jacc_cluster.fit_predict(jaccard_distance)
query_df['class'] = predicts
fig, axs = plt.subplots(1,1, figsize=(16, 8))
plot_classify(query_df, axs, 'Fields distribution with 2 Indexes')
fig, axs = plt.subplots(2,2, figsize=(28, 16))
axs = axs.ravel()

hier_jacc_cluster = AgglomerativeClustering(n_clusters=3, affinity='precomputed', linkage='average')
query_df['class'] = hier_jacc_cluster.fit_predict(jaccard_distance)
plot_classify(query_df, axs[0], 'Fields distribution with 3 Indexes')

hier_jacc_cluster = AgglomerativeClustering(n_clusters=4, affinity='precomputed', linkage='average')
query_df['class'] = hier_jacc_cluster.fit_predict(jaccard_distance)
plot_classify(query_df, axs[1], 'Fields distribution with 4 Indexes')

hier_jacc_cluster = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage='average')
query_df['class'] = hier_jacc_cluster.fit_predict(jaccard_distance)
plot_classify(query_df, axs[2], 'Fields distribution with 5 Indexes')

hier_jacc_cluster = AgglomerativeClustering(n_clusters=6, affinity='precomputed', linkage='average')
query_df['class'] = hier_jacc_cluster.fit_predict(jaccard_distance)
plot_classify(query_df, axs[3], 'Fields distribution with 6 Indexes')
orphan_feature = relation_df.sum().map(lambda x : 1 if x == 1 else 0)
array_df_orphan = array_df.copy()
array_df_orphan.insert(loc=0, column='0', value=orphan_feature)
array_df_orphan
jaccard_distance = pairwise_distances(array_df_orphan.values, metric='jaccard', n_jobs=4)
fig, axs = plt.subplots(2,2, figsize=(28, 16))
axs = axs.ravel()

hier_jacc_cluster = AgglomerativeClustering(n_clusters=3, affinity='precomputed', linkage='average')
query_df['class'] = hier_jacc_cluster.fit_predict(jaccard_distance)
plot_classify(query_df, axs[0], 'Fields distribution with 3 Indexes')

hier_jacc_cluster = AgglomerativeClustering(n_clusters=4, affinity='precomputed', linkage='average')
query_df['class'] = hier_jacc_cluster.fit_predict(jaccard_distance)
plot_classify(query_df, axs[1], 'Fields distribution with 4 Indexes')

hier_jacc_cluster = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage='average')
query_df['class'] = hier_jacc_cluster.fit_predict(jaccard_distance)
plot_classify(query_df, axs[2], 'Fields distribution with 5 Indexes')

hier_jacc_cluster = AgglomerativeClustering(n_clusters=6, affinity='precomputed', linkage='average')
query_df['class'] = hier_jacc_cluster.fit_predict(jaccard_distance)
plot_classify(query_df, axs[3], 'Fields distribution with 6 Indexes')
# filter_arr = query_df[(query_df['class'] == 3) | (query_df['class'] == 4) | (query_df['class'] == 5)]
# fig, axs = plt.subplots(1,1, figsize=(16, 8))
# plot_classify(filter_arr,axs, 'Indexes with few fields')

# query_df['class'] = query_df['class'].map(lambda x: 3 if x > 2 else x)
# fig, axs = plt.subplots(1,1, figsize=(16, 8))
# plot_classify(query_df,axs, 'Final Indexes')
#We decide 5 clusers are enough
fig, axs = plt.subplots(1,1, figsize=(16, 8))
hier_jacc_cluster = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage='average')
query_df['class'] = hier_jacc_cluster.fit_predict(jaccard_distance)
plot_classify(query_df, axs, 'Fields distribution with 5 Indexes')
query_df['chart index'] = query_df.index
cluster_result = query_df.groupby('class').apply(lambda x : (list(x['chart index']), np.unique(np.concatenate(list(x['fields'])))))
plt.figure(figsize=(14,8))
for index, row in cluster_result.iteritems():
#     print(row)
    plt.scatter(index + 1, len(row[0]), s=len(row[1]) * 1000, alpha=0.5)
    label = "chart:{}\nfields:{}".format(row[0], row[1])
    plt.annotate(label, (index + 1, len(row[0])), xytext=(index + 0.7, len(row[0])))

plt.ylim((0, 20))
plt.xlim((0,6))
plt.ylabel("Chart Number")
plt.xlabel('Index')
plt.title('Clustering Result')
plt.yticks(range(0,21))
# ax = plt.gca()
# rects = ax.patches
# for rect in rects:
#     ax.text(rect.get_x(), rect.get_height(), 'aa')
fig, axs = plt.subplots(1,5,figsize=(30, 12))
axs = axs.ravel()
class_group = query_df.groupby('class')
for class_group in class_group:
#     print(class_group[1])
    class_group[1]['fields']
    query_array = []
    fields_array = []
    group_fields = class_group[1]
    for row_fields in group_fields.iterrows():
#         print(row_fields[1]['fields'])
        fields_array = np.concatenate((fields_array, row_fields[1]['fields']))
        query_array = np.concatenate((query_array, [row_fields[0] for i in range(len(row_fields[1]['fields']))]))
    
#     print('class:%d'%(class_group[0] + 1))
#     print(fields_array)
    axs[class_group[0]].scatter(query_array.astype(int), fields_array)
    axs[class_group[0]].set_xlabel('Chart Index')
    axs[class_group[0]].set_ylabel('Fields Index')
    axs[class_group[0]].set_title('Index%d' % (class_group[0] + 1))
    axs[class_group[0]].set_xlim((0, 35))
    ymin = np.min(fields_array)
    ymax = np.max(fields_array)
    ymin -= 1
    ymax += 1
#     if ymin == ymax:
#         ymin -= 1
#         ymax += 1
    axs[class_group[0]].set_ylim((ymin, ymax))
    axs[class_group[0]].tick_params(axis='y', which='minor')
    axs[class_group[0]].grid(which='major', axis='both')
     


query_df['class'] = query_df['class'] + 1
query_df
data_classified = data
data_classified['Query Index'] = data['Query Index'].astype(int)
data_classified = pd.merge(data_classified, query_df, how='inner', left_on=data_classified['Query Index'], right_on=query_df.index)
data_classified.drop('key_0', axis=1, inplace=True)
data_classified
def output_schema(class_index, output):
    field_groups = data_classified[data_classified['class'] == class_index].groupby('Field Index')
    field_id_array = []
    field_regex_array = []
    for field_gp in field_groups:
        print(field_gp[0])
        field_id_array.append(field_gp[0])
        field_gp_value = field_gp[1]
        un_value = field_gp_value['Data Field Before Merge'].unique()
        print(un_value)
        field_regex_array.append(un_value[0])

    shema_df = pd.DataFrame({'field_id': field_id_array, 'regex':field_regex_array})
    shema_df.to_csv('%s.csv' % output, index=False)
output_schema(1, 'index1')
output_schema(2, 'index2')
output_schema(3, 'index3')
output_schema(4, 'index4')
output_schema(5, 'index5')
output_schema(6, 'index6')
