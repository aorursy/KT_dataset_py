import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data=pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

data.head()
data.isnull().sum()
data.drop(["last_review","reviews_per_month"],axis=1,inplace=True)
df=data.dropna()

df[["name","host_name"]].describe()
data.head()
data.info()
df=data.dropna()

print(len(np.unique(df["host_name"])),len(np.unique(df["name"])))
data["host_name"]=data["host_name"].fillna("Other")
data["name"]=data["name"].fillna("NaN")
data.info()
categorical=["name","host_name","neighbourhood_group","neighbourhood","room_type"]

numerical=data.columns ^ categorical

print(numerical,categorical,sep="\n")
data[numerical].describe()
data[categorical].describe().head(4)
cols=["availability_365","calculated_host_listings_count","minimum_nights","number_of_reviews","price"]





sns.pairplot(data[cols])
# room_type

fig,ax=plt.subplots(1,1,figsize=(9,6))

df=data["room_type"]

sns.countplot(df,ax=ax)

ax.set_title("Room_Type");

total=df.shape[0]



for p in ax.patches:

    percentage = '{:.1f}%'.format(100 * p.get_height()/total)

    x = p.get_x() + p.get_width() / 2 - 0.05

    y = p.get_y() + p.get_height()

    ax.annotate(percentage, (x, y),size=12)

    

ax.set_title("Room-Type"+" ( Total "+str(total)+" )");

fig,ax=plt.subplots(1,1,figsize=(9,4))

sns.stripplot(y="price",data=data,x="room_type");

ax.set_title("Room-Type");
fig,ax=plt.subplots(1,1,figsize=(12,5))



df=data

sns.countplot(df["neighbourhood_group"],ax=ax);
fig,ax=plt.subplots(5,2,figsize=(15,18));

cols=np.unique(data["neighbourhood_group"])

labels=np.unique(data["room_type"])

index=0

explode=[0.1]*3



for i in range(5):

  ax[i][0].set_title(cols[index],fontsize=19);

  ax[i][1].set_title(cols[index],fontsize=19);

  df=data.loc[data["neighbourhood_group"]==cols[index]]

  sns.countplot(df["room_type"],ax=ax[i][0],order=labels)

  total_room_types=df["room_type"].shape[0]

  sizes=[df.loc[df["room_type"]==i].shape[0]/total_room_types for i in labels]

  ax[i][1].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)  



  index+=1





fig.tight_layout();



sns.boxplot(df["price"])
fig,ax=plt.subplots(1,1,figsize=(8,4))



df=data.loc[data["price"]<500]

sns.violinplot("neighbourhood_group","price",data=df,ax=ax)
fig,ax=plt.subplots(3,2,figsize=(25,20))

df=data.loc[data["price"]<500]



index=0

for r in range(3):

    for c in range(2):

        if(index==5):

            break

        ddf=df.loc[df["neighbourhood_group"]==cols[index]]

        ax[r][c].set_title(cols[index],fontsize=24)

        ax[r][c].xaxis.set_tick_params(labelsize=22)

        ax[r][c].yaxis.set_tick_params(labelsize=22)

        



        for i in range(3):

            sns.distplot(ddf.loc[ddf["room_type"]==labels[i]]["price"],label=labels[i],ax=ax[r][c])

        index+=1    





ax[-1, -1].axis('off')



fig.legend(labels,fontsize=21);





ax[0][0].plot([70,70 ],[0,0.025],"r--",alpha=0.6);



ax[0][1].plot([70,70 ],[0,0.025],"r--",alpha=0.6);



ax[1][0].plot([105,105 ],[0,0.025],"r--",alpha=0.6);



ax[1][1].plot([70,70 ],[0,0.025],"r--",alpha=0.6);



ax[2][0].plot([60,60 ],[0,0.025],"r--",alpha=0.6);
#min nights

# df=data.loc[data["minimum_nights"]<730]

sns.stripplot(y=data["minimum_nights"])
fig,ax=plt.subplots(1,1,figsize=(9,4))

sns.stripplot(y="minimum_nights",data=data,x="room_type");

ax.set_title("Room-Type");

plt.ylim([-1,570])
fig,ax=plt.subplots(1,1,figsize=(9,4))

sns.stripplot(y="minimum_nights",data=data,x="neighbourhood_group");

ax.set_title("Room-Type");

plt.ylim([0,570])
fig,ax=plt.subplots(1,3,figsize=(25,5))

map_cols=["neighbourhood_group","room_type","availability_365"]

for i in range(3):

  ax[i].set_title(map_cols[i],fontsize=21)

  sns.scatterplot(hue=map_cols[i],x="latitude",y="longitude",data=data,ax=ax[i])