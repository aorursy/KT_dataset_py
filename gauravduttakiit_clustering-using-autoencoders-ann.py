import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, normalize

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

# You have to include the full link to the csv file containing your dataset

creditcard_df = pd.read_csv(r"/kaggle/input/creditcard-marketing/4.Marketing_data.csv")

# CUSTID: Identification of Credit Card holder 

# BALANCE: Balance amount left in customer's account to make purchases

# BALANCE_FREQUENCY: How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)

# PURCHASES: Amount of purchases made from account

# ONEOFFPURCHASES: Maximum purchase amount done in one-go

# INSTALLMENTS_PURCHASES: Amount of purchase done in installment

# CASH_ADVANCE: Cash in advance given by the user

# PURCHASES_FREQUENCY: How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)

# ONEOFF_PURCHASES_FREQUENCY: How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)

# PURCHASES_INSTALLMENTS_FREQUENCY: How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)

# CASH_ADVANCE_FREQUENCY: How frequently the cash in advance being paid

# CASH_ADVANCE_TRX: Number of Transactions made with "Cash in Advance"

# PURCHASES_TRX: Number of purchase transactions made

# CREDIT_LIMIT: Limit of Credit Card for user

# PAYMENTS: Amount of Payment done by user

# MINIMUM_PAYMENTS: Minimum amount of payments made by user  

# PRC_FULL_PAYMENT: Percent of full payment paid by user

# TENURE: Tenure of credit card service for user
creditcard_df
creditcard_df.info()

# 18 features with 8950 points  
creditcard_df.describe()

# Mean balance is $1564 

# Balance frequency is frequently updated on average ~0.9

# Purchases average is $1000

# one off purchase average is ~$600

# Average purchases frequency is around 0.5

# average ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY, and CASH_ADVANCE_FREQUENCY are generally low

# Average credit limit ~ 4500

# Percent of full payment is 15%

# Average tenure is 11.5 years
# Let's see who made one off purchase of $40761!

creditcard_df[creditcard_df['ONEOFF_PURCHASES'] == creditcard_df['ONEOFF_PURCHASES'].max()]

creditcard_df['CASH_ADVANCE'].max()
# Let's see who made cash advance of $47137!

# This customer made 123 cash advance transactions!!

# Never paid credit card in full



creditcard_df[creditcard_df['CASH_ADVANCE'] == creditcard_df['CASH_ADVANCE'].max()]

# Let's see if we have any missing data, luckily we don't!

sns.heatmap(creditcard_df.isnull(), yticklabels = False, cbar = False, cmap="rainbow")

plt.show()

creditcard_df.isnull().sum()
# Fill up the missing elements with mean of the 'MINIMUM_PAYMENT' 

creditcard_df.loc[(creditcard_df['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = creditcard_df['MINIMUM_PAYMENTS'].mean()

# Fill up the missing elements with mean of the 'CREDIT_LIMIT' 

creditcard_df.loc[(creditcard_df['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = creditcard_df['CREDIT_LIMIT'].mean()
sns.heatmap(creditcard_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")

plt.show()
# Let's see if we have duplicated entries in the data

creditcard_df.duplicated().sum()
# Let's drop Customer ID since it has no meaning here 

creditcard_df.drop("CUST_ID", axis = 1, inplace= True)
creditcard_df.head()
n = len(creditcard_df.columns)

n
creditcard_df.columns
# distplot combines the matplotlib.hist function with seaborn kdeplot()

# KDE Plot represents the Kernel Density Estimate

# KDE is used for visualizing the Probability Density of a continuous variable. 

# KDE demonstrates the probability density at different values in a continuous variable. 



# Mean of balance is $1500

# 'Balance_Frequency' for most customers is updated frequently ~1

# For 'PURCHASES_FREQUENCY', there are two distinct group of customers

# For 'ONEOFF_PURCHASES_FREQUENCY' and 'PURCHASES_INSTALLMENT_FREQUENCY' most users don't do one off puchases or installment purchases frequently 

# Very small number of customers pay their balance in full 'PRC_FULL_PAYMENT'~0

# Credit limit average is around $4500

# Most customers are ~11 years tenure



plt.figure(figsize=(20,60))

for i in range(len(creditcard_df.columns)-1):

  plt.subplot(16, 2, i+1)

  sns.distplot(creditcard_df[creditcard_df.columns[i]], kde_kws={"color": "r", "lw": 1, "label": "KDE"}, hist_kws={"color": "g"})

  plt.title(creditcard_df.columns[i])



plt.tight_layout()
plt.figure(figsize = (10,5))

ax=sns.countplot(creditcard_df['TENURE'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.show()
correlations = creditcard_df.corr()

f, ax = plt.subplots(figsize = (20, 20))

sns.heatmap(correlations, annot = True)

plt.show()

# 'PURCHASES' have high correlation between one-off purchases, 'installment purchases, purchase transactions, credit limit and payments. 

# Strong Positive Correlation between 'PURCHASES_FREQUENCY' and 'PURCHASES_INSTALLMENT_FREQUENCY'

# Let's scale the data first

scaler = StandardScaler()

creditcard_df_scaled = scaler.fit_transform(creditcard_df)
creditcard_df_scaled.shape
creditcard_df_scaled
scores_1 = []



range_values = range(1, 20)



for i in range_values:

  kmeans = KMeans(n_clusters = i)

  kmeans.fit(creditcard_df_scaled)

  scores_1.append(kmeans.inertia_) 

plt.figure(figsize = (10,5))

plt.plot(scores_1, 'bx-')

plt.title('Finding the right number of clusters')

plt.xlabel('Clusters')

plt.ylabel('Scores') 

plt.show()



# From this we can observe that, 4th cluster seems to be forming the elbow of the curve. 

# However, the values does not reduce linearly until 8th cluster. 

# Let's choose the number of clusters to be 7.
kmeans = KMeans(8)

kmeans.fit(creditcard_df_scaled)

labels = kmeans.labels_
kmeans.cluster_centers_.shape


cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [creditcard_df.columns])

cluster_centers           
# In order to understand what these numbers mean, let's perform inverse transformation

cluster_centers = scaler.inverse_transform(cluster_centers)

cluster_centers = pd.DataFrame(data = cluster_centers, columns = [creditcard_df.columns])

cluster_centers



# First Customers cluster (Transactors): Those are customers who pay least amount of intrerest charges and careful with their money, Cluster with lowest balance ($104) and cash advance ($303), Percentage of full payment = 23%

# Second customers cluster (revolvers) who use credit card as a loan (most lucrative sector): highest balance ($5000) and cash advance (~$5000), low purchase frequency, high cash advance frequency (0.5), high cash advance transactions (16) and low percentage of full payment (3%)

# Third customer cluster (VIP/Prime): high credit limit $16K and highest percentage of full payment, target for increase credit limit and increase spending habits

# Fourth customer cluster (low tenure): these are customers with low tenure (7 years), low balance 

labels.shape # Labels associated to each data point
labels.max()
labels.min()
y_kmeans = kmeans.fit_predict(creditcard_df_scaled)

y_kmeans

# concatenate the clusters labels to our original dataframe

creditcard_df_cluster = pd.concat([creditcard_df, pd.DataFrame({'cluster':labels})], axis = 1)

creditcard_df_cluster.head()
# Plot the histogram of various clusters

for i in creditcard_df.columns:

  plt.figure(figsize = (35, 5))

  for j in range(8):

    plt.subplot(1,8,j+1)

    cluster = creditcard_df_cluster[creditcard_df_cluster['cluster'] == j]

    cluster[i].hist(bins = 20)

    plt.title('{}    \nCluster {} '.format(i,j))

  

  plt.show()



# Obtain the principal components 

pca = PCA(n_components=2)

principal_comp = pca.fit_transform(creditcard_df_scaled)

principal_comp
# Create a dataframe with the two components

pca_df = pd.DataFrame(data = principal_comp, columns =['pca1','pca2'])

pca_df.head()
# Concatenate the clusters labels to the dataframe

pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)

pca_df.head()
plt.figure(figsize=(10,10))

ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df, palette =['red','green','blue','pink','yellow','gray','purple', 'black'])

plt.show()


from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.initializers import glorot_uniform

from keras.optimizers import SGD



encoding_dim = 7



input_df = Input(shape=(17,))





# Glorot normal initializer (Xavier normal initializer) draws samples from a truncated normal distribution 



x = Dense(encoding_dim, activation='relu')(input_df)

x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)

x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)

x = Dense(2000, activation='relu', kernel_initializer = 'glorot_uniform')(x)



encoded = Dense(10, activation='relu', kernel_initializer = 'glorot_uniform')(x)



x = Dense(2000, activation='relu', kernel_initializer = 'glorot_uniform')(encoded)

x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)



decoded = Dense(17, kernel_initializer = 'glorot_uniform')(x)



# autoencoder

autoencoder = Model(input_df, decoded)



#encoder - used for our dimention reduction

encoder = Model(input_df, encoded)



autoencoder.compile(optimizer= 'adam', loss='mean_squared_error')

creditcard_df_scaled.shape
autoencoder.fit(creditcard_df_scaled, creditcard_df_scaled, batch_size = 128, epochs = 25,  verbose = 1)
autoencoder.summary()
autoencoder.save_weights('autoencoder.h5')
pred = encoder.predict(creditcard_df_scaled)
pred.shape
scores_2 = []



range_values = range(1, 20)



for i in range_values:

  kmeans = KMeans(n_clusters= i)

  kmeans.fit(pred)

  scores_2.append(kmeans.inertia_)

plt.figure(figsize=(10,10))

plt.plot(scores_2, 'bx-')

plt.title('Finding right number of clusters')

plt.xlabel('Clusters')

plt.ylabel('scores') 

plt.show()
plt.figure(figsize=(10,10))

plt.plot(scores_1, 'bx-', color = 'r')

plt.plot(scores_2, 'bx-', color = 'g')

plt.show()
kmeans = KMeans(4)

kmeans.fit(pred)

labels = kmeans.labels_

y_kmeans = kmeans.fit_predict(creditcard_df_scaled)
df_cluster_dr = pd.concat([creditcard_df, pd.DataFrame({'cluster':labels})], axis = 1)

df_cluster_dr.head()
pca = PCA(n_components=2)

prin_comp = pca.fit_transform(pred)

pca_df = pd.DataFrame(data = prin_comp, columns =['pca1','pca2'])

pca_df.head()
pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)

pca_df.head()
plt.figure(figsize=(10,10))

ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df, palette =['red','green','blue','yellow'])

plt.show()