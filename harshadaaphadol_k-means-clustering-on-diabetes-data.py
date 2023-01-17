import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report

df=pd.read_csv("../input/diabetes.csv")
df.head()
sns.set_style('whitegrid')
sns.lmplot('BMI','Age',data=df, hue='Outcome',
           palette='coolwarm',height=6,aspect=1,fit_reg=False)
kmeans=KMeans(n_clusters=2)
kmeans.fit(df)
kmeans.cluster_centers_
def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0
df['Cluster'] = df['Outcome'].apply(converter)
df.head()
print("Confusion Matrix: \n" ,confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))

