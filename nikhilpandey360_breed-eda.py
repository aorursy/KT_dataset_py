
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

df = pd.read_csv("../input/labels.csv")
df.info()
# let's see how many breeds(classes) are there
print("Breeds Present in CSV: ",len(df.breed.unique()))
each_label = df.groupby("breed").count()
each_label = each_label.rename(columns = {"id" : "count"})
each_label = each_label.sort_values("count", ascending=False)
each_label.head(5)
each_label.tail()
ax=pd.value_counts(df['breed'],ascending=True).plot(kind='barh',
                                                       fontsize="40",
                                                       title="Class Distribution",
                                                       figsize=(50,100))
ax.set(xlabel="Images per class", ylabel="Classes")
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)
ax.title.set_size(30)
import cv2
training_images = os.listdir("../input/train/")
ls = []
for i in training_images:
    img = cv2.imread("../input/train/"+str(i))
    h,w,_ = img.shape
    ls.append([h,w])
ls = np.array((ls),np.int)

def reject_outliers(data, m = 2.5):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/(mdev if mdev else 1.)
    return data[s<m]
h = reject_outliers(ls[:,0])
w = reject_outliers(ls[:,1])

max_ht = h.max()
mean_ht = h.mean()
min_ht = h.min()

max_wd = w.max()
mean_wd = w.mean()
min_wd = w.min()

print("width: Max {}, Mean {}, Min {}".format(max_wd,mean_wd,min_wd))
print("hight: Max {}, Mean {}, Min {}".format(max_ht,mean_ht,min_ht))
import seaborn as sns
import matplotlib.pyplot as plt
h = h.astype(np.int).tolist()
h.sort()
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.xlabel("height")
sns.distplot(h)

h = w.tolist()
h.sort()
plt.subplot(122)
plt.xlabel("width")
sns.distplot(h)
