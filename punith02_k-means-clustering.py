import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline
college_data=pd.read_csv('../input/us-news-and-world-reports-college-data/College.csv')
college_data.head()
college_data.rename(columns={'Unnamed: 0':'College Name'},inplace=True)
college_data.info()
college_data.describe()
sns.set_style('whitegrid')

plot = sns.FacetGrid(college_data,hue="Private",palette='coolwarm',size=6,aspect=2)

plot = plot.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
college_data.columns
sns.set_style('darkgrid')

g = sns.FacetGrid(college_data,hue="Private",palette='coolwarm',size=6,aspect=2)

g = g.map(plt.hist,'Accept',bins=20,alpha=0.7)
sns.boxplot(x='Private',y='Expend',data=college_data)
sns.scatterplot(y='F.Undergrad',x='Outstate',data=college_data,hue='Private')
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=2)
x=college_data[[ 'Apps', 'Accept', 'Enroll', 'Top10perc',

       'Top25perc', 'F.Undergrad', 'P.Undergrad', 'Outstate', 'Room.Board',

       'Books', 'Personal', 'PhD', 'Terminal', 'S.F.Ratio', 'perc.alumni',

       'Expend', 'Grad.Rate']]
kmeans.fit(x)
kmeans.labels_