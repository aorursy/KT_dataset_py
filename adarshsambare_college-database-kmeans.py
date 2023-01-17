## Importing the required Libraries
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
## Reading the Data into the enviroment
df = pd.read_csv('../input/us-news-and-world-reports-college-data/College.csv', 
                 index_col=0)
# Making column 1 as index
## Previewing the data
df.head()
## Information regarding the data base
df.info()
## For further ananlysis
df.describe()
## Room board and Grad Rate as per Private colleges
sns.lmplot(x= 'Room.Board',y='Grad.Rate',data=df, hue='Private',
           fit_reg=False,palette='coolwarm',size=6,aspect=1)
## Outstate and Undergrad with Private
sns.lmplot(x= 'Outstate',y='F.Undergrad',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)
## Outstate based on the Private Column
g = sns.FacetGrid(df, hue = 'Private', palette = 'coolwarm',
                 size = 6, aspect = 2)
g = g.map(plt.hist,'Outstate',bins = 25, alpha = 0.6)
## Outstate based on the Private Column
g = sns.FacetGrid(df, hue = 'Private', palette = 'coolwarm',
                 size = 6, aspect = 2)
g = g.map(plt.hist,'Grad.Rate',bins = 25, alpha = 0.6)
# Do notice we have an outlier of 120 in Grad.Rate Column
# As grad rate cannot be more than 100
## Checking which data point is having Grad.Rate more than 100
df[df['Grad.Rate']>100]
# We need to change that from 118% to 100%
## Changing the Grad.Rate to 100
df['Grad.Rate']['Cazenovia College'] = 100
# It may give you a warning but no error
## Checking if data point is having Grad.Rate more than 100 is replaced or not
df[df['Grad.Rate']>100]
## calling the kmeans from sklearn
from sklearn.cluster import KMeans
## Giving no of cluster as  we already know 
## The outcome have only two catogeries
kmeans = KMeans(n_clusters=2)
## Fitting the model on whole data
## As it is supervised learning with no target splitting is unnecessary
kmeans.fit(df.drop('Private',axis=1))
## Cluster centers
kmeans.cluster_centers_
## Convert Count Function:
def converter(private):
    if private=='Yes':
        return 1
    else:
        return 0
## Applying the convert count function to our Private column &
## Creating Cluster column
df['Cluster'] = df['Private'].apply(converter)
## Checking the dataframe
df.head()
## Confusion Matrix & Classification report
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))