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

%matplotlib inline

import seaborn as sns

from PIL import Image

import gc
train_df = pd.read_csv("../input/train/train.csv")

test_df = pd.read_csv("../input/test/test.csv")



train_csv = pd.read_csv("../input/train/train.csv")
# Repeated in section 2 for ML - Zero to Deep Dive Homework



print("train.csv shape is {}".format(train_df.shape))

print("test.csv shape is {}".format(test_df.shape))
train_df.head()
test_df.head()
# Added for ML - Zero to Deep Dive Homework

train_df.dtypes
# Added for ML - Zero to Deep Dive Homework

test_df.dtypes
train_df.isnull().sum()
test_df.isnull().sum()
# Added for ML - Zero to Deep Dive Homework



train_df[0:-1].describe()
# Added for ML - Zero to Deep Dive Homework



test_df[0:-1].describe()
# Repeated from existing code above



print("train.csv shape is {}".format(train_df.shape))

print("test.csv shape is {}".format(test_df.shape))
fig, ax = plt.subplots(figsize=(6, 5))

ax.set_title("AdoptionSpeed count in train")

sns.countplot(x="AdoptionSpeed", data=train_df, ax=ax)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.set_title("Type count in train data")

#ax1.patch.set_facecolor("blue")

#ax1.patch.set_alpha(0.3)

ax2.set_title("Type count in test data")

#ax2.patch.set_facecolor("yellow")

#ax2.patch.set_alpha(0.3)

g = sns.catplot(data=train_df, x="Type", kind="count", ax=ax1)

plt.close(g.fig)

g = sns.catplot(data=test_df, x="Type", kind="count", ax=ax2)

plt.close(g.fig)
fig, ax1 = plt.subplots(figsize=(8, 5))

#ax1.set_title("train data")

#ax1.patch.set_facecolor("blue")

#ax1.patch.set_alpha(0.3)

g = sns.catplot(data=train_df, x="Type", hue="AdoptionSpeed", kind="count", ax=ax1)

plt.close(g.fig)
print("The amount of unique Name is {} in train".format(train_df.Name.nunique()))

print("train shape is {}".format(train_df.shape))

print("The amount of unique Name is {} in test".format(test_df.Name.nunique()))

print("test shape is {}".format(test_df.shape))
train_df.Name
print("Age in train")

print(train_df.Age.describe())

print("\n\n")

print("Age in test")

print(test_df.Age.describe())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharex=True) 

sns.distplot(train_df.Age, ax=ax1, bins=20, kde=False)

sns.distplot(test_df.Age, ax=ax2, bins=20, kde=False)

ax1.set_title("Age histgram(bin=10) in train data", fontsize=15)

ax2.set_title("Age histgram(bin=10) in test data", fontsize=15)

ax1.tick_params(labelsize=14)

ax2.tick_params(labelsize=14)
fig, ax = plt.subplots(figsize=(8, 5))

ax = sns.stripplot(x="AdoptionSpeed", y="Age", data=train_df, ax=ax)
train_df.loc[:, ["Breed1", "Breed2"]].head(10)
train_df.loc[:, ["Breed1", "Breed2"]].describe()
test_df.loc[:, ["Breed1", "Breed2"]].describe()
print("The number of mixed in train: {}({:.2%})".format(train_df.query("Breed2 != 0").Type.count(), train_df.query("Breed2 != 0").Type.count()/len(train_df)))

print("The number of mixed in test: {}({:.2%})".format(test_df.query("Breed2 != 0").Type.count(), test_df.query("Breed2 != 0").Type.count()/len(test_df)))
breed_df = pd.read_csv("../input/breed_labels.csv")

breed_df.head()
breed_df.tail()
fig, ax = plt.subplots()

sns.countplot(x="Type", data=breed_df, ax=ax)

g = ax.set_title("Breeder type count")
print("The number of Breed1=0 in train: {}".format(train_df.query("Breed1 == 0").Type.count()))

print("The number of Breed1=0 in test: {}".format(test_df.query("Breed1 == 0").Type.count()))

print("The number of Breed1=307 in train: {}".format(train_df.query("Breed1 == 307").Type.count()))

print("The number of Breed1=307 in test: {}".format(test_df.query("Breed1 == 307").Type.count()))
train_df.query("Breed1 == 0")
train_df.query("Breed1 == 307").head()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(x="Gender", data=train_df, ax=ax1)

sns.countplot(x="Gender", data=test_df, ax=ax2)

ax1.set_title("The amount of each Gender in train data", fontsize=15)

g = ax2.set_title("The amount of each Gender in test data", fontsize=15)
fig, ax = plt.subplots(figsize=(8, 5))

ax = sns.countplot(x="Gender", data=train_df, ax=ax, hue="AdoptionSpeed")
train_df.loc[:, ["Color1", "Color2", "Color3"]].head(10)
train_df.loc[:, ["Color1", "Color2", "Color3"]].describe()
test_df.loc[:, ["Color1", "Color2", "Color3"]].describe()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1 = sns.countplot(x="Color1", data=train_df, hue="AdoptionSpeed", ax=ax1)

ax2 = sns.countplot(x="Color1", data=test_df, ax=ax2)
color_df = pd.read_csv("../input/color_labels.csv")
color_df
print("The number of nothing color(color1,2,3 == 0) pet: {}".format(train_df.query("Color1==0 & Color2==0 & Color3==0").shape[0]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1 = sns.countplot(x="MaturitySize", data=train_df, hue="AdoptionSpeed", ax=ax1)

ax2 = sns.countplot(x="MaturitySize", data=test_df, ax=ax2)

ax1.set_title("The amount of each MaturitySize in train data", fontsize=15)

g = ax2.set_title("The amount of each MaturitySize in test data", fontsize=15)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1 = sns.countplot(x="FurLength", data=train_df, hue="AdoptionSpeed", ax=ax1)

ax2 = sns.countplot(x="FurLength", data=test_df, ax=ax2)

ax1.set_title("The amount of each FurLength in train data", fontsize=15)

g = ax2.set_title("The amount of each FurLength in test data", fontsize=15)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1 = sns.countplot(x="Vaccinated", data=train_df, hue="AdoptionSpeed", ax=ax1)

ax2 = sns.countplot(x="Vaccinated", data=test_df, ax=ax2)

ax1.set_title("The amount of each Vaccinated in train data", fontsize=15)

g = ax2.set_title("The amount of each Vaccinated in test data", fontsize=15)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1 = sns.countplot(x="Dewormed", data=train_df, hue="AdoptionSpeed", ax=ax1)

ax2 = sns.countplot(x="Dewormed", data=test_df, ax=ax2)

ax1.set_title("The amount of each Dewormed in train data", fontsize=15)

g = ax2.set_title("The amount of each Dewormed in test data", fontsize=15)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1 = sns.countplot(x="Sterilized", data=train_df, hue="AdoptionSpeed", ax=ax1)

ax2 = sns.countplot(x="Sterilized", data=test_df, ax=ax2)

ax1.set_title("The amount of each Sterilized in train data", fontsize=15)

g = ax2.set_title("The amount of each Sterilized in test data", fontsize=15)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1 = sns.countplot(x="Health", data=train_df, hue="AdoptionSpeed", ax=ax1)

ax2 = sns.countplot(x="Health", data=test_df, ax=ax2)

ax1.set_title("The amount of each Health in train data", fontsize=15)

g = ax2.set_title("The amount of each Health in test data", fontsize=15)
train_df.Quantity.describe()
test_df.Quantity.describe()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1 = sns.distplot(train_df.Quantity, kde=False, bins=20, ax=ax1)

ax2 = sns.distplot(test_df.Quantity, kde=False, bins=20, ax=ax2)

ax1.set_title("The amount of Quantity in train data", fontsize=15)

g = ax2.set_title("The amount of Quantity in test data", fontsize=15)
fig, ax = plt.subplots(figsize=(8, 5))

ax = sns.stripplot(x="Quantity", y="AdoptionSpeed", data=train_df, ax=ax)
train_df.Fee.describe()
test_df.Fee.describe()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1 = sns.distplot(train_df.Fee, kde=False, bins=20, ax=ax1)

ax2 = sns.distplot(test_df.Fee, kde=False, bins=20, ax=ax2)

ax1.set_title("The amount of Fee in train data", fontsize=15)

g = ax2.set_title("The amount of Fee in test data", fontsize=15)
fig, ax = plt.subplots(figsize=(8, 5))

ax = sns.stripplot(x="Fee", y="AdoptionSpeed", data=train_df, ax=ax)

#ax.set_xticks(np.arange(0, 10))
train_df.State.head(10)
test_df.State.head(10)
state_df = pd.read_csv("../input/state_labels.csv")

state_df.head()
print("StateID min in train: {}".format(train_df.State.min()))

print("StateID max in train: {}".format(train_df.State.max()))

print("StateID min in test: {}".format(test_df.State.min()))

print("StateID max in test: {}".format(test_df.State.max()))

print("StateID min in state_labels: {}".format(state_df.StateID.min()))

print("StateID max in state_labels: {}".format(state_df.StateID.max()))
print("train data num is {}, the number of unique RescureID is {}".format(train_df.shape[0], train_df.RescuerID.nunique()))

print("test data num is {}, the number of unique RescureID is {}".format(test_df.shape[0], test_df.RescuerID.nunique()))
sorted(train_df.groupby(["RescuerID"]).Type.count().values, reverse=True)[:15]
train_df.VideoAmt.describe()
test_df.VideoAmt.describe()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

ax1 = sns.distplot(train_df.VideoAmt, kde=False, bins=9, ax=ax1)

ax2 = sns.distplot(test_df.VideoAmt, kde=False, bins=9, ax=ax2)

ax1.set_title("The amount of VideoAmt in train data", fontsize=15)

g = ax2.set_title("The amount of VideoAmt in test data", fontsize=15)
fig, ax = plt.subplots(figsize=(8, 5))

ax = sns.stripplot(x="VideoAmt", y="AdoptionSpeed", data=train_df, ax=ax)

#ax.set_xticks(np.arange(0, 10))
train_df.PhotoAmt.describe()
test_df.PhotoAmt.describe()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

ax1 = sns.distplot(train_df.PhotoAmt, kde=False, bins=30, ax=ax1)

ax2 = sns.distplot(test_df.PhotoAmt, kde=False, bins=30, ax=ax2)

ax1.set_title("The amount of PhotoAmt in train data", fontsize=15)

g = ax2.set_title("The amount of PhotoAmt in test data", fontsize=15)
fig, ax = plt.subplots(figsize=(8, 5))

ax = sns.stripplot(x="PhotoAmt", y="AdoptionSpeed", data=train_df, ax=ax)

#ax.set_xticks(np.arange(0, 10))
train_df.Description.head()
test_df.Description.head()
# Added for ML - Zero to Deep Dive Homework



train_df.corr()
# Added for ML - Zero to Deep Dive Homework



test_df.corr()


g = pd.plotting.scatter_matrix(train_df, alpha=0.8, figsize=(12,12), range_padding=0.5)
fig, ax = plt.subplots(figsize=(12, 9)) 

ax = sns.heatmap(train_df.corr(), square=True, vmax=1, vmin=-1, center=0)
fig, axes = plt.subplots(4,3, figsize=(15, 20))

images_train = os.listdir("../input/train_images/")

fig.suptitle("train images")

for i, img in enumerate(np.random.choice(images_train, 12)):

    image = Image.open("../input/train_images/" + img)

    axes[i//3, i%3].imshow(image)
fig, axes = plt.subplots(4,3, figsize=(15, 20))

images_train = os.listdir("../input/test_images/")

fig.suptitle("test images")

for i, img in enumerate(np.random.choice(images_train, 12)):

    image = Image.open("../input/test_images/" + img)

    axes[i//3, i%3].imshow(image)