# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from collections import Counter

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

from matplotlib.ticker import PercentFormatter, FuncFormatter

import matplotlib.patches as mpatches

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



def pre_processing(df,value_for_na = -1):

    df["class"] = df["class"].map({"neg":0,"pos":1})

    df = df.replace("na",value_for_na)

    for col in df.columns:

        if col != "origin":

            df[col] = pd.to_numeric(df[col])

    return df



def get_tag(name):

    return name.split("_")[0]
train = pd.read_csv('../input/aps_failure_training_set.csv',)

test = pd.read_csv('../input/aps_failure_test_set.csv',)

train["origin"] = "train"

test["origin"] = "test"

data= pd.concat([train,test],sort=True)

#to avoid double index

data = data.reset_index()
columns_list = train.columns



all_columns_with_tags = [a for a in columns_list if "_" in a]

all_tags = [get_tag(a) for a in all_columns_with_tags]

hists = [k for k,v in Counter(all_tags).items() if v == 10]

hists_columns = [k for k in all_columns_with_tags if get_tag(k) in hists]

hists_dict = {k:[col for col in hists_columns if k in col] for k in hists if get_tag(k) in hists}

counter_columns = [k for k in all_columns_with_tags if get_tag(k) not in hists]
data = pre_processing(data,-1)

data = data.rename(columns={"class":"Class"}) #to avoid name collision with the class



fig , axs = plt.subplots(10,7,sharex=True,  sharey='row',figsize=(15,20))

cmap=plt.cm.get_cmap('Set1', 7)

df_top = data[(data.Class==0)&(data.ag_000>0)]

for axis,r in zip(axs,df_top.sample(5).iterrows()):

    for i,(ax,hist) in enumerate(zip(axis,hists)):

        ax.step(range(10),r[1][f"{hist}_000":f"{hist}_009"],where="mid",color=cmap(i))

        ax.tick_params(labelbottom=False)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: "%.1E"%x))

        ax.tick_params(axis='y', which='major', labelsize=10)

df_bottom = data[(data.Class==1)&(data["ag_000"]>0)]

for axis,r in zip(axs[5:],df_top.sample(5).iterrows()):

    for i,(ax,hist) in enumerate(zip(axis,hists)):

        ax.step(range(10),r[1][f"{hist}_000":f"{hist}_009"],where="mid",color=cmap(i))

        ax.tick_params(labelbottom=False)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: "%.1E"%x))

        ax.tick_params(axis='y', which='major', labelsize=10)



for ax,hist in zip(axs[0],hists):

    ax.set_xlabel(hist.upper())    

    ax.xaxis.set_label_position('top') 

plt.tight_layout()
for hist in hists:

    data[f"{hist}_total"] = sum(data[col] for col in hists_dict[hist])

data["system_age"] = data[[f"{hist}_total" for hist in hists]].max(axis=1)

data = data.replace(-10,-1) #Some totals of -1 to make it easy to detect NA

#data = data.drop([f"{hist}_total" for hist in hists],axis=1)





plt.figure(figsize=(15,5));

for_plotting = data[data.system_age>=0]

_,bins,_ = plt.hist(np.log(for_plotting[for_plotting.Class==0].system_age+1),bins=100,density=True,alpha=0.5,label="Class 0");

plt.hist(np.log(for_plotting[for_plotting.Class==1].system_age+1),bins=bins,density=True,alpha=0.5,label="Class 1");

plt.legend();

plt.ylabel("Percentage per categorie");

plt.xlabel("Number of measurements (a.k.a. System Age) for physical (logarithmic scale)");

plt.xlim(0,21);







font = {'family' : 'normal',

        'weight' : 'normal',

        'size'   : 12}

plt.rc('font', **font)

df_dist = {}

for dist in hists:

    pds_to_concat = []

    for i,col in enumerate(hists_dict[dist]):

        temp = data[[col,'Class']]

        temp["ref"] = col

        temp.loc[temp[col]>=0,col] = np.log(temp[col]+1)

        temp.columns = ["data","Class","ref"]

        pds_to_concat.append(temp)

    df_dist[dist] = pd.concat(pds_to_concat)

    df_dist[dist] = df_dist[dist][df_dist[dist].data>=0] # No NA values

#ag_dist.data = np.log(ag_dist.data+1)

fig , axs = plt.subplots(7)

fig.set_figheight(15)

fig.set_figwidth(15)

#sns.boxenplot(data=ag_dist,x="ref",y="data",hue="class",outlier_prop=0.00000000001)

for i,hist in enumerate(hists):

    sns.violinplot(data=df_dist[hist],x="ref",y="data",hue="Class",scale="width",scale_hue=True

                   , split=True,ax=axs[i],legend=(i==0));

fig.text(0.04, 0.5, 'Distribution of each feature per outcome', va='center', rotation='vertical');



variable =  "ag_002"

bucket_nb = 20 



_ , bins = pd.qcut(np.log(data[data[variable]>0][variable]),bucket_nb,retbins=True)

data[f"{variable}_buckets"] = pd.cut(np.log(data[variable]+1),[-0.1] + list(bins),labels=range(bucket_nb+1))

#pd.crosstab(data[f"{variable}_buckets"],data.Class,margins=True)

plt.figure(figsize=(15,10));

ax = sns.barplot(data=data,x=f"{variable}_buckets", y="Class")

ax.yaxis.set_major_formatter(PercentFormatter(1))

plt.xlabel("Distibutions of the feature AG_002");

plt.ylabel("Percentage of failure per range");

variable =  "cn_001"

bucket_nb = 20 

_ , bins = pd.qcut(np.log(data[data[variable]>0][variable]),bucket_nb,retbins=True)

data[f"{variable}_buckets"] = pd.cut(np.log(data[variable]+1),[-0.1] + list(bins),labels=range(bucket_nb+1))

#pd.crosstab(data[f"{variable}_buckets"],data.Class,margins=True)

plt.figure(figsize=(15,10));

ax = sns.barplot(data=data,x=f"{variable}_buckets", y="Class")

ax.yaxis.set_major_formatter(PercentFormatter(1))

plt.xlabel("Distibutions of the feature CN_001");

plt.ylabel("Percentage of failure per range");







plt.figure(figsize=(20,10))



df = data[(data.Class==0)&(data.ag_002>0)&(data.cn_001>0)]

cmap = sns.light_palette("blue", as_cmap=True)

_,xedges,yedges, _ = plt.hist2d(np.log(df["ag_002"]+2), np.log(df["cn_001"]+1),norm=LogNorm()

                                , bins=40,cmap=cmap, cmin=1,alpha=0.5,label="Sound APS");

df = data[(data.Class==1)&(data.ag_002>0)&(data.cn_001>0)]



class_0 = mpatches.Patch(color='blue',alpha=0.3, label='Class 0')

class_1 = mpatches.Patch(color='red',alpha=0.3, label='Class 1')

plt.legend(handles=[class_0,class_1],loc=2)



cmap = sns.light_palette("red", as_cmap=True)

plt.hist2d(np.log(df["ag_002"]+1)

                , np.log(df["cn_001"]+1), cmap=cmap,bins=[xedges,yedges], cmin=1

                ,alpha=0.5

                ,label="Failing APS");

plt.ylabel("AG_002 distribution using a logarithmic scale");

plt.xlabel("CN_001 distribution using a logarithmic scale");



for hist in hists:

    data[f"{hist}_total"] = sum(data[col] for col in hists_dict[hist])

data = data.replace(-10,-1) #Some totals of -1 to make it easy to detect NA





for hist in hists:

    data[f"{hist}_avg"] = 0

    for col in hists_dict[hist]:

        data[f"{col}_density"] = data[col]/data[f"{hist}_total"]

        data.loc[data[f"{hist}_total"] == -10, f"{col}_density"] = -1

        data.loc[data[f"{hist}_total"] == 0, f"{col}_density"] = 0

        data[f"{hist}_avg"] += int(col[3:])*data[col]

    data[f"{hist}_avg"] = data[f"{hist}_avg"]/data.system_age

    data.loc[data[f"{hist}_total"] == 0, f"{hist}_avg"] = 0

    data.loc[data[f"{hist}_total"] == -1, f"{hist}_avg"] = 0



data = data.drop([f"{hist}_total" for hist in hists],axis=1)

_, bins_for_total_feature = pd.qcut(data[(data["origin"]=="train")&(data["Class"]==1)&(data.system_age>0)].system_age,3,retbins=True)

bins_for_total_feature[3] = np.max(data.system_age)

data["total_cat"] = pd.cut(data.system_age.replace(np.nan,-1),[-10.1] + list(bins_for_total_feature)

                        ,labels=["null","low", "medium", "high"])



pds_to_concat = []

for i,hist in enumerate(hists):

    temp = data[data.total_cat!="null"][[f"{hist}_avg",'Class',"total_cat"]]

    temp["ref"] = f"{hist}_avg"

    #temp.loc[temp[f"{hist}_avg"]>=0,f"{hist}_avg"] = np.log(temp[f"{hist}_avg"]+1)

    temp.columns = ["data","Class","total_cat","ref"]

    pds_to_concat.append(temp)

all_avg_vals = pd.concat(pds_to_concat)

categories_to_display = data.total_cat.cat.categories[1:]

plt.rcParams.update({'font.size': 12})





fig , axs = plt.subplots(len(categories_to_display))

fig.set_figheight(15)

fig.set_figwidth(15)

#sns.boxenplot(data=ag_dist,x="ref",y="data",hue="class",outlier_prop=0.00000000001)

for ax,cat in zip(axs,categories_to_display):

    g = sns.violinplot(data=all_avg_vals[all_avg_vals.total_cat==cat]

                  , x="ref",y="data",hue="Class",split=True

                  , ax = ax)

    g.set_xlabel(f"For system_age {cat} values")

#fig.text(0.04, 0.5, 'Distribution of each feature per outcome', va='center', rotation='vertical');



categories_to_diplay = data.total_cat.cat.categories[1:]

fig , axs = plt.subplots(len(categories_to_diplay))

fig.set_figheight(15)

fig.set_figwidth(15)

#sns.boxenplot(data=ag_dist,x="ref",y="data",hue="class",outlier_prop=0.00000000001)

for ax,cat in zip(axs,categories_to_diplay):

    g = sns.pointplot(data=all_avg_vals[all_avg_vals.total_cat==cat]

                  , x="ref",y="data",hue="Class"

                  , ax = ax)

    g.set_xlabel(f"For total_max {cat} values")

#fig.text(0.04, 0.5, 'Distribution of each feature per outcome', va='center', rotation='vertical');


