# 1.0 Call required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
#2.0 Load dataset into dataframe

MallCustomerds = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
#2.1 rename columns 

MallCustomerds.columns
MallCustomerds.rename({'Annual Income (k$)':'Annual_Income','Spending Score (1-100)':
                       'Spending_Score'},inplace = True , axis = 1)

#3.0 Drawing plots to show relation between variables of the dataset

# 3.1 Draw boxplots
fig = plt.figure(figsize=(15,5))
ax=plt.subplot(1,3,1)
sns.boxplot('Gender','Annual_Income',data=MallCustomerds)
ax=plt.subplot(1,3,2)
sns.boxplot('Gender','Spending_Score',data=MallCustomerds)

# 3.2 Draw sctterplot
fig = plt.figure(figsize=(15,5))
ax=plt.subplot(1,3,1)
sns.scatterplot(x="Age",y="Annual_Income",data=MallCustomerds,hue="Gender")
ax=plt.subplot(1,3,2)
sns.scatterplot(x="Age",y="Spending_Score",data=MallCustomerds,hue="Gender")

# 3.3 Draw distribution plot
fig = plt.figure(figsize=(15,5))
ax=plt.subplot(1,3,1)
sns.distplot(MallCustomerds.loc[MallCustomerds['Gender']=="Male",'Annual_Income'])
ax=plt.subplot(1,3,2)
sns.distplot(MallCustomerds.loc[MallCustomerds['Gender']=="Female",'Annual_Income'])
#4.0 To Perform Gaussian Mixture in the dataset

# 4.1 Import GaussianMixture Class
from sklearn.mixture import GaussianMixture

# 4.2 Drop Gender and customerID

ds = MallCustomerds.drop(['CustomerID','Gender'],axis=1)

# 4.3  Use StandardScaler class for scaling of dataframe
ss = StandardScaler()
ss.fit(ds)
XX = ss.transform(ds)
XX.shape

#4.4 Find optimum no. of clusters by using bic and aic
bic = []
aic = []
for i in range(8):
    gm = GaussianMixture(n_components = i+1,
                         n_init = 10,
                         max_iter = 100);
    gm.fit(XX)
    bic.append(gm.bic(XX))
    aic.append(gm.aic(XX))
    
fig = plt.figure()
plt.plot(range(1,9), aic)
plt.plot(range(1,9), bic)
# 4.5 Perform TNSE to check relationship between variables of dataset

tsne = TSNE(n_components = 2)
tsne_out = tsne.fit_transform(XX)
plt.scatter(tsne_out[:, 0], tsne_out[:, 1],
            marker='x',
            s=50,              # marker size
            linewidths=5,      # linewidth of marker edges
            c=gm.predict(XX)   # Colour as per gmm
            )
# 5.0 Perform clustering

# 5.1 From above values of aic and bic it is appearant that no. of clusters 4 is optimum
gm = GaussianMixture(n_components = 4,
                         n_init = 10,
                         max_iter = 100);
    
# 5.2 Train the algorithm
gm.fit(XX)    

# 5.3 Where are the clsuter centers
gm.means_

# 5.4 Did algorithm converge?
gm.converged_


# 5.5 How many iterations did it perform?
gm.n_iter_

# 5.6 Clusters labels
gm.predict(XX)    

# 5.6 Draw scatter plot to show clusters
fig = plt.figure()
plt.scatter(XX[:, 0], XX[:, 1],
            c=gm.predict(XX),
            s=2)

# 6.0 Find anomalous datapoints


densities = gm.score_samples(XX)
densities

density_threshold = np.percentile(densities,4)
density_threshold

anomalies = XX[densities < density_threshold]
anomalies
anomalies.shape

# 6.1 Show anomalous points
fig = plt.figure()
plt.scatter(XX[:, 0], XX[:, 1], c = gm.predict(XX))
plt.scatter(anomalies[:, 0], anomalies[:, 1],
            marker='x',
            s=50,               # marker size
            linewidths=5,      # linewidth of marker edges
            color='red'
            )
plt.show()



# 6.2 seperate out unanomalous points
unanomalies = XX[densities >= density_threshold]
unanomalies.shape

# 6.3 Transform both anomalous and unanomalous data
#    to pandas DataFrame
df_anomalies = pd.DataFrame(anomalies, columns = ['Age', 'Annual_Income','Spending_Score'])
df_anomalies['z'] = 'anomalous'   # Create a IIIrd constant column
df_normal = pd.DataFrame(unanomalies, columns = ['Age', 'Annual_Income','Spending_Score'])
df_normal['z'] = 'unanomalous'    # Create a IIIrd constant column


# 6.4 Let us see density plots
fig = plt.figure(figsize=(15,5))
ax=plt.subplot(1,3,1)
sns.distplot(df_anomalies['Age'])
sns.distplot(df_normal['Age'])

ax=plt.subplot(1,3,2)
sns.distplot(df_anomalies['Annual_Income'])
sns.distplot(df_normal['Annual_Income'])

ax=plt.subplot(1,3,3)
sns.distplot(df_anomalies['Spending_Score'])
sns.distplot(df_normal['Spending_Score'])

# 6.4 Draw side-by-side boxplots
# 6.4.1 Ist stack two dataframes
df = pd.concat([df_anomalies,df_normal])
# 6.4.2 Draw featurewise boxplots
fig = plt.figure(figsize=(15,5))

ax=plt.subplot(1,3,1)
sns.boxplot(x = df['z'], y = df['Age'])
sns.boxplot(x = df['z'], y = df['Age'])

ax=plt.subplot(1,3,2)
sns.boxplot(x = df['z'], y = df['Annual_Income'])
sns.boxplot(x = df['z'], y = df['Annual_Income'])

ax=plt.subplot(1,3,3)
sns.boxplot(x = df['z'], y = df['Spending_Score'])
sns.boxplot(x = df['z'], y = df['Spending_Score'])

