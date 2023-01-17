import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pydicom

from PIL import Image

import os

import cv2

from tqdm import tqdm

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()

BASE_PATH = "../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/"

TRAIN_DIR = "stage_2_train/"



train_df = pd.read_csv(BASE_PATH + 'stage_2_train.csv')

train_df['filename'] = train_df['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")

train_df['type'] = train_df['ID'].apply(lambda st: st.split('_')[2])

train_df['id'] = train_df['ID'].apply(lambda st: st.split('_')[1])



fig, ax = plt.subplots(figsize=(15, 10))

sns.countplot(train_df.Label,ax=ax)

ax.set_xlabel("Label")

ax.set_ylabel("Count")

ax.set_title("Label distribution")
fig, ax = plt.subplots(figsize=(15, 10))

plt.rcParams['axes.labelsize'] = 20

plt.rcParams['axes.titlesize'] = 20

type_counts = train_df.groupby("type").Label.value_counts().unstack()

true_cases = type_counts.loc[:,1] / train_df.groupby("type").size() * 100

sns.barplot(x=true_cases.index,y=true_cases.values,ax=ax)

plt.yticks(rotation=0,size=15)

plt.xticks(rotation=0,size=15)

ax.set_xlabel("ICH Type")

ax.set_ylabel("%")

ax.set_title("Type distribution",pad=20)
fig, ax = plt.subplots(figsize=(15, 10))

multi_count = train_df.groupby("id").Label.sum()

sns.countplot(multi_count,ax=ax)

ax.set_title("Co-occurences")

ax.set_xlabel("Targets per image")

ax.set_ylabel("Frequency")
df = train_df[['Label', 'filename', 'type']].drop_duplicates().pivot(

    index='filename', columns='type', values='Label').reset_index()
df[df['any']==1]['any'].count()
df[df['any']==0]['any'].count()
df.count()


cols = ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural']

labels = df[cols]

output = pd.DataFrame(

{x:[(df[x] & df[y]).sum() for y in cols] for x in cols},

index=cols)

output

#output

output["epidural"][output["epidural"].index != "epidural"].sum()
normalized_df = output.copy().astype(np.float32)

for col in cols:

    total = output[col][col]

    total_col = total - output[col][output[col].index != col].sum()

    normalized_df[col][col] = total_col / total

    for other in cols:

        if other == col:

            continue

        normalized_df[col][other] = output[col][other] / total

normalized_df
sum(normalized_df['epidural'])
normalized_df = normalized_df.rename({

    "epidural":"EDH",

    "intraparenchymal":"IPH",

    "intraventricular":"IVH",

    "subarachnoid":"SAH",

    "subdural":"SDH"

},axis="index")

normalized_df = normalized_df.rename(index=str,columns={

    "epidural":"EDH",

    "intraparenchymal":"IPH",

    "intraventricular":"IVH",

    "subarachnoid":"SAH",

    "subdural":"SDH"

})
plt.rcParams['axes.labelsize'] = 40

plt.rcParams['axes.titlesize'] = 20

fig, ax = plt.subplots(figsize=(15, 10))



sns.heatmap(normalized_df,ax=ax,annot=True,annot_kws={"size": 20})

for i in np.arange(0,5,1.0):

    ax.axvline(i, color='white', lw=10)

plt.yticks(rotation=0,size=20)

plt.xticks(rotation=0,size=20)

ax.set_title("Co-occurring haemorrhages matrix",pad=20)

#ax.set_title("Co-occurence Matrix")
ax.get_xticks()
#for i in range(type_list):

#    title = cols[i]

#    dataset = type_list[i]

#    fig,ax = plt.subplots()

#    sns.countplot(dataset,ax=ax)

#    plt.show()