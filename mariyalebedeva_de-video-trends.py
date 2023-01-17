import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
dfDE=pd.read_csv('../input/youtube-new/DEvideos.csv', sep=",")
#At the beginnig it's better to clean your data from NaNs (In original we have more than 40.000, after - 39288)

dfDE=dfDE.dropna()
#to check that our dataset has been correctly uploaded

dfDE.head(1)
dfDE.describe()
def hist_plot(array, kde, rug, color, length=16, width=6, bins=50):

    """Histogram plot with possible kde distribution curve and rug signs. 

    kde and rug are boolen, color-string, others are integers

    PS color=color - when we enter 'yellow' as arg function replace the argument 

    with the string, but we should wright color='yellow'"""

    plt.figure(figsize=(length, width))

    sns.distplot(array, bins=bins, kde=kde, rug=rug, color=color)
def kde_plot(array, shade, color, length=16, width=6):

    """KDE (Kernal distribution estimation) plot with shade. 

    shade is boolen, color-string, others are integers"""

    plt.figure(figsize=(length,width))

    sns.kdeplot(array, shade=shade, color=color)
Likes_array=np.array(dfDE["likes"])
hist_plot(Likes_array, False, True, "orange")
kde_plot(Likes_array, True, "green")
Likes_array.sort()
res=[]

num=len(Likes_array)//1000

for i in range (num):

    res.append(Likes_array[i*1000:(i*1000+1000)])

num_end=len(Likes_array)%1000

res.append(Likes_array[-1:-num_end-1:-1])

total_Likes = [sum(i)/len(i) for i in res]
#check the number of entries. It should be 40 (since there are more than 40.000 entries).

len(total_Likes)
hist_plot(total_Likes, False, False, "blue")
kde_plot(total_Likes, True, "purple")
Dislikes_array=np.array(dfDE["dislikes"])
hist_plot(Dislikes_array, False, True, "brown")
kde_plot(Dislikes_array, True, "lightgreen")
Dislikes_array.sort()
res=[]

num=len(Dislikes_array)//1000

for i in range (num):

    res.append(Dislikes_array[i*1000:(i*1000+1000)])

num_end=len(Dislikes_array)%1000

res.append(Dislikes_array[-1:-num_end-1:-1])

total_Dislikes = [sum(i)/len(i) for i in res]
#check the number of entries. It should be 41 (since there are more than 40.000 entries)

len(total_Dislikes)
#you may combine kde distribution with histogram

hist_plot(total_Dislikes, True, False, "green")
Views_array=np.array(dfDE["views"])
hist_plot(Views_array, True, False, "red")
res=[]

num=len(Views_array)//1000

for i in range (num):

    res.append(Views_array[i*1000:(i*1000+1000)])

num_end=len(Views_array)%1000

res.append(Views_array[-1:-num_end-1:-1])

total_Views = [sum(i)/len(i) for i in res]
plt.figure(figsize=(16,10))

sns.distplot(total_Views, rug="True", rug_kws={"color": "orange"},

            kde_kws={"color": "y", "lw": 5, "label": "KDE Views", "shade": "lightyellow"},

            hist_kws={"color": "blue", "histtype": "step", "linewidth": 3, "label": "Views distribution"})
with sns.axes_style("white"):

    (sns.jointplot(x=total_Likes, y=total_Views, height=8, kind="hex", color="r")

    .set_axis_labels("Likes", "Views"))
with sns.axes_style("white"):

    (sns.jointplot(x=total_Dislikes, y=total_Views, height=8, kind="hex", color="g")

    .set_axis_labels("Dislikes", "Views"))