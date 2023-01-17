from IPython.display import YouTubeVideo

YouTubeVideo('AfK9LPNj-Zo', width=800, height=300)
import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objects as go

from ipywidgets import widgets

from ipywidgets import *



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

from scipy.signal import find_peaks

train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv",delimiter=",",encoding="latin", engine='python')

test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv",delimiter=",",encoding="latin", engine='python')

train.head(10)
print(train.info())
train.dtypes.value_counts()
test.head(10)
count = train['Patient'].value_counts() 

print(count) 
print("Number of Patient in the train set {}".format(len( train['Patient'].unique()))) 

print("Number of Patient in the test set {}".format(len( test['Patient'].unique()))) 
YouTubeVideo('BmYCAp4dRuA', width=800, height=300)
fig = plt.figure(figsize = (20, 10))

ax = fig.add_subplot()

i = 0 

for id_patient in train["Patient"].unique()[0:6] : 

    y = train[train["Patient"] == id_patient]["FVC"].reset_index(drop=True)

    df = train[train["Patient"] == id_patient].reset_index(drop=True)

    max_peaks_index, _ = find_peaks(y, height=0) 

    doublediff2 = np.diff(np.sign(np.diff(-1*y))) 

    min_peaks_index = np.where(doublediff2 == -2)[0] + 1

    ax.plot(y, color = "blue", alpha = .6)



    if i == 0:

        ax.scatter(x = y[max_peaks_index].index, y = y[max_peaks_index].values, marker = "^", s = 150, color = "green", alpha = .6, label = "Peaks")

        ax.scatter(x = y[min_peaks_index].index, y = y[min_peaks_index].values, marker = "v", s = 150, color = "red", alpha = .6, label = "Troughs")

    else :

        ax.scatter(x = y[max_peaks_index].index, y = y[max_peaks_index].values, marker = "^", s = 150, color = "green", alpha = .6)

        ax.scatter(x = y[min_peaks_index].index, y = y[min_peaks_index].values, marker = "v", s = 150, color = "red", alpha = .6)

    for max_annot in max_peaks_index[:] :

        for min_annot in min_peaks_index[:] :



            max_text = df.iloc[max_annot]["FVC"]

            min_text = df.iloc[min_annot]["FVC"]



            max_text_w = df.iloc[max_annot]["Weeks"]

            min_text_w = df.iloc[min_annot]["Weeks"]



            ax.text(df.index[max_annot], y[max_annot] + 50, s = max_text, fontsize = 12, horizontalalignment = 'center', verticalalignment = 'center')

            ax.text(df.index[min_annot], y[min_annot] + 50, s = min_text, fontsize = 12, horizontalalignment = 'center', verticalalignment = 'center')



            ax.text(df.index[max_annot], y[max_annot] - 50, s = "Week : " + str(max_text_w), fontsize = 12, horizontalalignment = 'center', verticalalignment = 'center')

            ax.text(df.index[min_annot], y[min_annot] - 50, s = "Week : " + str(min_text_w), fontsize = 12, horizontalalignment = 'center', verticalalignment = 'center')

    ax.text(df.index[0], y[0] + 30, s = id_patient, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')

    i = i + 1

    ax.legend(loc = "upper left", fontsize = 10)

train['FVC_mean'] = train['FVC'].groupby(train['Patient']).transform('mean')

train['FVC_max'] = train['FVC'].groupby(train['Patient']).transform('max')

train['FVC_min'] = train['FVC'].groupby(train['Patient']).transform('min')

train['FVC_std'] = train['FVC'].groupby(train['Patient']).transform('std')
fig = plt.figure(figsize = (12, 6))

ax = fig.add_subplot(111) 



for Smoking in sorted(list(train["SmokingStatus"].unique())):

    Age = train[train["SmokingStatus"] == Smoking]["Age"]

    FVC_mean = train[train["SmokingStatus"] == Smoking]["FVC_mean"]

    ax.scatter(Age, FVC_mean, label = Smoking, s = 10)



ax.spines["top"].set_color("None") 

ax.spines["right"].set_color("None")

ax.set_xlabel("Age") 

ax.set_ylabel("FVC_mean")

ax.set_title("Scatter plot of Age vs FVC_mean.")

ax.legend(loc = "upper left", fontsize = 10)
fig = plt.figure(figsize = (12, 6))

ax = fig.add_subplot(111) 



for Smoking in sorted(list(train["SmokingStatus"].unique())):

    Percent = train[train["SmokingStatus"] == Smoking]["Percent"]

    FVC_mean = train[train["SmokingStatus"] == Smoking]["FVC_mean"]

    ax.scatter(Percent, FVC_mean, label = Smoking, s = 10)



ax.spines["top"].set_color("None") 

ax.spines["right"].set_color("None")



ax.set_xlabel("Percent") 

ax.set_ylabel("FVC_mean")



ax.set_title("Scatter plot of Percent vs FVC_mean.")

ax.legend(loc = "upper left", fontsize = 10)
import squarify

label_value = train["SmokingStatus"].value_counts().to_dict()

labels = ["{} has {} obs".format(class_, obs) for class_, obs in label_value.items()]

colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

plt.figure(figsize = (10, 5))

squarify.plot(sizes = label_value.values(), label = labels,  color = colors, alpha = 0.8)

plt.title("Smoking Status")
fig, axes = plt.subplots(1, 3, figsize=(20, 9))

p = sns.boxplot(x='Sex', y='Age', hue='SmokingStatus', data=train, ax=axes[0])

p.set_title('train')



p = sns.boxplot(x='Sex', y='FVC_mean', hue='SmokingStatus', data=train, ax=axes[1])

p.set_title('train')



p = sns.boxplot(x='Sex', y='FVC_std', hue='SmokingStatus', data=train, ax=axes[2])

p.set_title('train')
fig, axes = plt.subplots(1, 2, figsize=(20, 9))





for s in train["SmokingStatus"].unique():

    x = train[train["SmokingStatus"] == s]["Percent"]

    g1 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[0])

    g1.set_title('Percent vs SmokingStatus')

g1.legend()





for s in train["SmokingStatus"].unique():

    x = train[train["SmokingStatus"] == s]["Age"]

    g2 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[1])

    g2.set_title('Age vs SmokingStatus')

g2.legend()

   
fig, axes = plt.subplots(1, 2, figsize=(20, 9))





for s in train["Sex"].unique():

    x = train[train["Sex"] == s]["Percent"]

    g1 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[0])

    g1.set_title('Percent vs Sex')

g1.legend()





for s in train["Sex"].unique():

    x = train[train["Sex"] == s]["Age"]

    g2 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[1])

    g2.set_title('Age vs Sex')

g2.legend()

   
fig, axes = plt.subplots(2, 2, figsize=(20, 11))





for s in train["SmokingStatus"].unique():

    x = train[train["SmokingStatus"] == s]["FVC_mean"]

    g1 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[0,0])

    g1.set_title('FVC_mean vs SmokingStatus')

g1.legend()





for s in train["SmokingStatus"].unique():

    x = train[train["SmokingStatus"] == s]["FVC_std"]

    g2 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[0,1])

    g2.set_title('FVC_std vs SmokingStatus')

g2.legend()



for s in train["SmokingStatus"].unique():

    x = train[train["SmokingStatus"] == s]["FVC_max"]

    g3 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[1,0])

    g3.set_title('FVC_max vs SmokingStatus')

g3.legend()



for s in train["SmokingStatus"].unique():

    x = train[train["SmokingStatus"] == s]["FVC_min"]

    g4 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[1,1])

    g4.set_title('FVC_min vs SmokingStatus')

g4.legend()
fig, axes = plt.subplots(2, 2, figsize=(20, 11))





for s in train["Sex"].unique():

    x = train[train["Sex"] == s]["FVC_mean"]

    g1 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[0,0])

    g1.set_title('FVC_mean vs Sex')

g1.legend()





for s in train["Sex"].unique():

    x = train[train["Sex"] == s]["FVC_std"]

    g2 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[0,1])

    g2.set_title('FVC_std vs Sex')

g2.legend()



for s in train["Sex"].unique():

    x = train[train["Sex"] == s]["FVC_max"]

    g3 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[1,0])

    g3.set_title('FVC_max vs Sex')

g3.legend()



for s in train["Sex"].unique():

    x = train[train["Sex"] == s]["FVC_min"]

    g4 = sns.distplot(x, kde = True, label = "{}".format(s), ax=axes[1,1])

    g4.set_title('FVC_min vs Sex')

g4.legend()