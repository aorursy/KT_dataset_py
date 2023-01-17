from bq_helper import BigQueryHelper
import bq_helper,requests,json
import matplotlib.pyplot as plt

bq_assistant = BigQueryHelper("bigquery-public-data", "stackoverflow")
stackOverflow = bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="stackoverflow")
query = "SELECT tag_name,count FROM `bigquery-public-data.stackoverflow.tags` ORDER BY count DESC;"
response = stackOverflow.query_to_pandas_safe(query)
tag_list = response["tag_name"].values.tolist()
freq_list = response["count"].values.tolist()
plt.figure(figsize=(30,10))
plt.bar(tag_list[0:49],freq_list[0:49],)
plt.show()



from bq_helper import BigQueryHelper
import bq_helper,requests,json
import matplotlib.pyplot as plt
import copy

bq_assistant = BigQueryHelper("bigquery-public-data", "stackoverflow")
stackOverflow = bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="stackoverflow")
set_data = {}
year = 2008

def clean_up(data):
    length = len(data)
    i = 0
    while i<length:
        data[i] = data[i].split("|")
        i += 1
    return data

while year<2019:
    query = "SELECT tags,view_count FROM `bigquery-public-data.stackoverflow.posts_questions` WHERE EXTRACT(YEAR FROM creation_date)="+str(year)+" ORDER BY RAND() LIMIT 10000;"
    response = stackOverflow.query_to_pandas_safe(query)
    tag_list = clean_up(response["tags"].values.tolist())
    freq_list = response["view_count"].values.tolist()
    data_set = [tag_list,freq_list]
    set_data[year] = copy.deepcopy(data_set)
    year += 1
def count_f(data):
    count = {}
    for i in data:
        for x in i:
            if x in count:
                count[x]+=1
            else:
                count[x]=1
    return count

def count_v(data):
    count = {}
    i = 0
    length = len(data[0])
    while i<length:
        for x in data[0][i]:
            if x in count:
                count[x]+=data[1][i]
            else:
                count[x]=data[1][i]
        i += 1
    return count
    

count_tag = {}
vcount_tag = {}
year = 2008
while year<2019:
    count_tag[year] = count_f(set_data[year][0])
    vcount_tag[year] = count_v(set_data[year])
    year += 1
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt

year = 2008

def label_spawn(row):
    if row["frequency"] in low:
        return 0
    #elif row["frequency"] in mid:
    #    return 1
    elif row["frequency"] in high:
        return 1
    #    return 2

df_list = []
    
while year<2019:
    temp_list = [[],[],[]]
    for i in count_tag[year]:
        temp_list[0].append(i)
        temp_list[1].append(count_tag[year][i])
        temp_list[2].append(vcount_tag[year][i])
    d_set = {"tag" : temp_list[0] ,"frequency" : temp_list[1] ,"view_frequency" : temp_list[2]}
    df = pd.DataFrame(data=d_set)
    df = df.sort_values(by=["frequency"])

    #low  = range(1,int(df["frequency"].iloc[-1] * 0.33)+1)
    #mid = range(int(df["frequency"].iloc[-1] * 0.33)+1,int(df["frequency"].iloc[-1] * 0.66)+1)
    #high = range(int(df["frequency"].iloc[-1] * 0.66)+1,int(df["frequency"].iloc[-1] * 1)+1)
    low  = range(1,int(df["frequency"].iloc[-1] * 0.5)+1)
    high = range(int(df["frequency"].iloc[-1] * 0.5)+1,int(df["frequency"].iloc[-1] * 1)+1)
    
    df["spawn_freq"] = df.apply(label_spawn,axis=1) 
    df_list.append(df.copy(deep=True))
    year += 1

#colormap = np.array(["blue","red","green"])
colormap = np.array(["blue","red"])
year = 2008

for i in df_list:
    df = i
    plt.scatter(df.frequency,df.view_frequency,c=colormap[df.spawn_freq],s=40)
    plt.title(str(year))
    plt.show()
    year += 1
from sklearn.cluster import KMeans
import copy

year = 2008
df_list_c = copy.deepcopy(df_list)
#colors = ["r","b","g"]
colors = ["b","r"] 
index = 0

for i in df_list_c:
    df = i.drop(columns=["tag"])
    #model = KMeans(n_clusters=3)
    model = KMeans(n_clusters=2)
    model.fit(df)
    #predY = np.choose(model.labels_, [1,0,2]).astype(np.int64)
    predY = np.choose(model.labels_, [0,1]).astype(np.int64)
    plt.scatter(df.frequency,df.view_frequency,c=colormap[predY],s=40)
    plt.title("K-Mean "+str(year))
    plt.show()
    df_list_c[index] = df
    
    labels = model.labels_
    centroids = model.cluster_centers_
    
    frequency_l = df["frequency"].values.tolist()
    vfrequency_l = df["view_frequency"].values.tolist()
    df_array = []
    for d in range(len(frequency_l)):
        df_array.append([frequency_l[d],vfrequency_l[d]])
    
    #for x in range(3):
    for x in range(2):
        points = np.array([df_array[j] for j in range(len(df_array)) if labels[j]==x])
        plt.scatter(points[:, 0], points[:, 1], s=40, c=colors[x])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="*", s=400, c="black")
    plt.title("K-Mean Centroid "+str(year))
    plt.show()
    
    year += 1
    index += 1
from sklearn import metrics
year = 2008

for i in df_list_c:
    k_range = range(2, 5)
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=1)
        km.fit(i)
        scores.append(metrics.silhouette_score(i, km.labels_))

    plt.plot(k_range, scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.grid(True)
    plt.title("Silhouette Index "+str(year))
    plt.show()
    
    year += 1
df_list_c2 = copy.deepcopy(df_list)
for i in range(len(df_list_c2)):
    df_list_c2[i] = df_list_c2[i].sort_values(by="frequency",ascending=False)

trend_tag = []
for i in df_list_c2:
    trend_tag.extend(i["tag"][:10].tolist())
trend_tag = set(trend_tag)

trend_change={}
freq_set = {}
for i in trend_tag:
    trend_change[i] = {"freq":[0], "v_freq":[0]}
    freq_set[i] = {"freq":[], "v_freq":[]}
    for x in df_list_c2:
        try:
            freq_set[i]["freq"].append(x.loc[x["tag"]==i]["frequency"].tolist()[0])
            freq_set[i]["v_freq"].append(x.loc[x["tag"]==i]["view_frequency"].tolist()[0])
        except(IndexError):
            freq_set[i]["freq"].append(0)
            freq_set[i]["v_freq"].append(0)

for i in trend_tag:
    for x in range(10):
        try:
            freq_p = (freq_set[i]["freq"][x+1]-freq_set[i]["freq"][x])/freq_set[i]["freq"][x]
            freq_p *= 100
            freq_p = float(format(freq_p,".1f"))
            v_freq_p = (freq_set[i]["v_freq"][x+1]-freq_set[i]["v_freq"][x])/freq_set[i]["v_freq"][x]
            v_freq_p *= 100
            v_freq_p = float(format(v_freq_p,".1f"))
        except(ZeroDivisionError):
            freq_p = 0
            v_freq_p = 0
        trend_change[i]["freq"].append(freq_p)
        trend_change[i]["v_freq"].append(v_freq_p)
        
title_list = ["Frequency","View Frequency"]
temp = ["freq","v_freq"]

for x in range(2):
    plt.figure(figsize=(20,10))
    plt.title(title_list[x]+" Graph")
    plt.grid(True)
    legend = []
    for i in trend_tag:
        plt.plot([2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018],freq_set[i][temp[x]])
        legend.append(i)
    plt.xlabel("Year")
    plt.ylabel(title_list[x])
    plt.legend(legend, loc='upper left')
    plt.show()
    
    plt.figure(figsize=(20,10))
    plt.title(title_list[x]+" Change in Percentage Graph")
    plt.grid(True)
    legend = []
    for i in trend_tag:
        plt.plot([2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018],trend_change[i][temp[x]])
        legend.append(i)
    plt.xlabel("Year")
    plt.ylabel(title_list[x]+"(%)")
    plt.legend(legend, loc='upper left')
    plt.show()