import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
%matplotlib inline
data=pd.read_csv('../input/cars.csv')
data.head()
data.info()
data.describe()
sns.set_style('whitegrid')
sns.lmplot('mpg',' cylinders',data=data,hue=' brand',fit_reg=False)
data.columns
from sklearn.cluster import KMeans
kmc=KMeans(n_clusters=3)
data.info()
data[data[' cubicinches']==" "]
df=data[data[' cubicinches']!=" "]
df[' cubicinches']=df[' cubicinches'].astype(float)
df[df[' weightlbs']==' ']
df=df[df[' weightlbs']!=' ']
df[' weightlbs']=df[' weightlbs'].astype(float)
df.info()
df.describe()
sns.pairplot(df,hue=' brand')
kmc.fit(df.drop(' brand',axis=1))
kmc.cluster_centers_
kmc.labels_
def converter(x):
    if x==' US.':
        return 2
    elif x==' Europe.':
        return 0
    elif x==' Japan.':
        return 1
    else:
        return 2
df['cluster']=df[' brand'].apply(converter)
df.head()
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['cluster'],kmc.labels_))
print(classification_report(df['cluster'],kmc.labels_))
plt.figure(figsize=(12,6))
sns.lmplot(x=' weightlbs',y=' cubicinches',data=df,hue='cluster',fit_reg=False)
df['pred_clusters']=kmc.labels_
df.head()
plt.figure(figsize=(12,6))
sns.lmplot(x=' weightlbs',y=' cubicinches',data=df,hue='pred_clusters',fit_reg=False)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(14,6))
ax1.set_title('K Means')
ax1.scatter(df[' weightlbs'],df[' cubicinches'],c=df['pred_clusters'],cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(df[' weightlbs'],df[' cubicinches'],c=df['cluster'],cmap='rainbow')