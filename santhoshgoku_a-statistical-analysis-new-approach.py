import os

import pandas as pd



#ploting

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="darkgrid")



#plotly

import plotly.express as px



#color

from colorama import Fore, Style,Back



#pydicom

import pydicom



plt.style.use("seaborn-notebook")

plt.show()





# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')
root_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/'
df_train = pd.read_csv(os.path.join(root_path,"train.csv"))

df_test = pd.read_csv(os.path.join(root_path,"test.csv"))

submission = pd.read_csv(os.path.join(root_path,"sample_submission.csv"))

train_folder = root_path+'train/'

test_folder  = root_path+'train/'
df_train.head().style.bar(subset=["FVC"],color=['#F7DC6F'])
df_train.info()
plt.figure(figsize=(10,5))

a = sns.countplot(data=df_train,x="Sex",hue="Sex",color="blue",palette=["#F5B041","#58D68D"])

plt.title("Gender Distribution")

plt.legend(fontsize=10)
a = df_train["Age"].plot.hist(colormap="jet",legend=True,color="#BB8FCE")

plt.xlabel("Age")

plt.legend(fontsize=10)
a = df_train["Patient"].value_counts().plot.hist(legend=True,color="#85C1E9")

plt.xlabel("no of visits")

plt.legend(fontsize=10)
a = df_train["FVC"].plot.kde(legend=True,color="#F8C471",linewidth=2.8)

plt.legend(fontsize=10)

plt.xlabel("FVC")
print('The mean value of FCV is',Back.CYAN+Style.BRIGHT+Fore.BLACK, f'{df_train["FVC"].mean()}')
a = df_train.plot.scatter(x="FVC",y="Age",color="#E74C3C")

plt.legend(fontsize=10)
plt.figure(figsize=(45,8))

sns.set(font_scale=1.1)

ax = sns.catplot(x="SmokingStatus", hue="Sex", col="Sex",data=df_train, kind="count",palette=["#F8C471","#58D68D"])

plt.legend(fontsize=10)
sns.pairplot(df_train, hue="Sex", palette="Set2", diag_kind="kde", height=2.5)
# sns.violinplot(x="SmokingStatus", y="Age", data=df_train, size=7,palette=["#F5B7B1","#2ECC71","#E74C3C"])

fig = px.violin(df_train, y="Age", x="SmokingStatus", color="Sex", box=True, points="all")

fig.update_layout(

    autosize=False,

    width=1200,

    height=700,)

fig.show()
fig = px.violin(df_train, y="Age", x="Sex", color="Sex", box=True, points="all")

fig.update_layout(

    autosize=False,

    width=1000,

    height=700,)

fig.show()
df_train_corr = df_train.corr()

sns.clustermap(df_train_corr, cmap="GnBu",annot=True)

plt.legend(fontsize=10)