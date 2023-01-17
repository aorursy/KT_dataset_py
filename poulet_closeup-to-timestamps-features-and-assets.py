import pandas as pd

with pd.HDFStore("../input/train.h5", "r") as train:

    df = train.get("train")
import matplotlib.pyplot as plt

df["y"].hist(bins=100)

plt.show()
print("Total timestamps: {}".format(len(df["timestamp"].unique())))

print("Total assets: {}".format(len(df["id"].unique())))
import re

derived=[]

fundamental=[]

technical=[]

for var_name in df.columns:

    if re.search("^derived",var_name) is not None:

        derived.append(var_name)

    if re.search("^fundamental",var_name) is not None:

        fundamental.append(var_name)

    if re.search("^technical",var_name) is not None:

        technical.append(var_name)



print("Total derived {}".format(len(derived)))

print("Total fundamental {}".format(len(fundamental)))

print("Total technical {}".format(len(fundamental)))



id_list=df["id"].unique().tolist()
df_timestamps=pd.DataFrame(None, columns=["id","count","min","max","diff"])

for asset in id_list:

    pt=df["timestamp"][df["id"]==asset]

    count_stamp=pt.count()

    max_stamp=pt.max()

    min_stamp=pt.min()

    diff_stamp=pt.diff().dropna().mean()

    nrow = pd.DataFrame([[asset, count_stamp, min_stamp, max_stamp, diff_stamp]], columns=["id","count","min","max","diff"])

    df_timestamps = df_timestamps.append(nrow, ignore_index=True)

df_timestamps["count"].hist(bins=100,figsize=(6,3))

plt.title('Count of timestamps per asset')

plt.show()

df_timestamps["min"].hist(bins=100,figsize=(6,3))

plt.title("Minimum timestamp")

plt.show()

df_timestamps["max"].hist(bins=100,figsize=(6,3))

plt.title('Maximum timestamp')

plt.show()

df_timestamps["diff"].hist(bins=100,figsize=(6,3))

plt.title('Difference between timestamps')

plt.show()
import numpy as np

fig =  plt.figure(figsize=(9,46))



features_list = derived + fundamental + technical



pt = pd.pivot_table(df, values="timestamp", index="id", aggfunc=np.count_nonzero)

pt = pt.sort_values(ascending=False)

fullt_ids=pt[pt==1812].index.tolist()

asset_id=fullt_ids[200] #pick any asset with complete timestamps



col_str="y"

#return

pt_id = df[["timestamp","y"]][df["id"]==asset_id].dropna()

ax = fig.add_subplot(23,1,1)

ax.plot(pt_id["timestamp"],pt_id["y"])

ax.plot(pt_id["timestamp"],(pt_id["y"] + 1).cumprod())

ax.set_xlim(0, 1812)

ax.tick_params(axis='both', which='major', labelsize=6)

for key,spine in ax.spines.items():

    spine.set_visible(False)

ax.set_title("return", size=10)

#vars

fig_num=6

for col_str in features_list:    

    ax = fig.add_subplot(23,5,fig_num)

    pt_id = df[["timestamp",col_str]][df["id"]==asset_id].dropna()

    if col_str in derived:

        color="red"

    elif col_str in fundamental:

        color="blue"

    else:

        color="green"       

    ax.scatter(pt_id["timestamp"],pt_id[col_str],s=1,c=color,marker=(1,2,0))

    for key,spine in ax.spines.items():

        spine.set_visible(False)

    ax.set_xlim(0, 2000)

    ax.get_xaxis().set_visible(False)

    ax.tick_params(bottom="off", top="off", left="off", right="off")

    ax.set_title(col_str, size=6)

    ax.tick_params(axis='both', which='major', labelsize=6)

    fig_num += 1



plt.show()
import math

column_str="derived_0"

asset=fullt_ids[0]



df_range = pd.DataFrame(None,columns=["id"]+features_list)

for asset in fullt_ids: #ignore the assets w/o complete timestamps

    s_range = df[df["id"]==asset].max()-df[df["id"]==asset].min()

    df_range = df_range.append(s_range[["id"]+features_list], ignore_index=True)

    df_range.loc[df_range.shape[0]-1,"id"]=asset
percentile=5



fig = plt.figure(figsize=(9,44))

fig.subplots_adjust(hspace=.5)

fig_num=1

for col_str in features_list:

    nona_column=df_range[col_str].dropna()

    nona_column=nona_column.sort_values()

    desc_column=nona_column.describe()

    lower_p=math.floor(len(nona_column)*percentile/100)

    higher_p=math.floor(len(nona_column)*(100-percentile)/100)

    

    ax = fig.add_subplot(22,5,fig_num)

    if col_str in derived:

        color="red"

    elif col_str in fundamental:

        color="blue"

    else:

        color="green"

    for key,spine in ax.spines.items():

        spine.set_visible(False)

    ax.tick_params(bottom="off", top="off", left="off", right="off")

    ax.set_title(col_str, size=10)

    nona_column[lower_p:higher_p].hist(bins=10, ax=ax, grid=False, color=color)

    fig_num+=1

plt.show()
qtl=.76



fig = plt.figure(figsize=(9,44))

fig.subplots_adjust(hspace=.5)

fig_num=1



for col_str in features_list:

    ax = fig.add_subplot(22,5,fig_num)

    for key,spine in ax.spines.items():

        spine.set_visible(False)

    ax.tick_params(bottom="off", top="off", left="off", right="off")

    ax.set_title(col_str, size=10)

    #nona_column[lower_p:higher_p].hist(bins=10, ax=ax, grid=False, color=color)

    nonan=df_range[col_str].dropna()

    try:

        ax.boxplot(nonan)

    except:

        print("Error at "+col_str)

    ax.set_ylim(nonan.quantile(0.95-qtl),nonan.quantile(qtl))

    fig_num+=1

plt.show()