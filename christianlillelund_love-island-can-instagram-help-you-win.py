# Load the dataset

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



file_name = '/kaggle/input/love-island-2019/Love Island 2019.xlsx'

df = pd.read_excel(file_name, sheet_name=0, skiprows=[0,1])



# Make some simplifications

df = df.rename(columns={"Instagram followres (June 2020) in millions": "Followers"})

df = df.drop(['Unnamed: 0'], axis=1)
# Show 10 participants

df.head(10)
# Print a contestant as a sample

df.iloc[5]
# Number of participants

print(f'Total number of contestants: {len(df)}')



NumM = len(df[df['Gender'] == 'M'])

print (f'Number of male contestants: {NumM}')



NumF = len(df[df['Gender'] == 'F'])

print (f'Number of female contestants: {NumF}')



AvgA = round(df['Age'].mean())

print (f'Average age of contestants: {AvgA} years old')



YoungP = df['Age'].min()

print (f'Youngest contestant is {YoungP} years old')



OldP = (df['Age'].max())

print (f'Oldest contestant is {OldP} years old')



TopT = (df['Hometown'].value_counts().index.tolist()[:5])

print (f'Top 5 towns in the show: {TopT}')



NumT = len(df['Hometown'].value_counts().index.tolist())

print (f'There are {NumT} different towns')



BotT = df['Hometown'].value_counts().index.tolist()[NumT-5:]

print (f'Bottom 5 towns in the show: {BotT}')



AvgF = round(df['Followers'].mean(), 2)

print (f'Average followers in millions: {AvgF}M')



df = df.assign(FirstName=(df['Name'].str.extract('(^\S*)')))

PMF = df.loc[df['Followers'].idxmax(), 'FirstName']

NMF = df['Followers'].max()

print (f'Contestant with the most followers is {PMF} with {NMF} millions')
# Plot age versus status



plt.figure(figsize=(8,6))

ax = sns.barplot("Status", y="Age", data=df, palette="Blues_d")
# Followers by progress

plt.figure(figsize=(8,6))

ax = sns.barplot(x="Status", y="Followers", hue="Gender", data=df)

ax.set(title='Followers versus status', xlabel='Status', ylabel='Followers')

plt.show()



f,ax1 = plt.subplots(figsize=(26,10))

sns.pointplot(x='FirstName', y='Followers',data=df, color='lime',alpha=0.8)

ax1.set(title='Followers versus progress')

plt.xlabel('Contestant',fontsize = 20,color='blue')

plt.ylabel('Followers in M',fontsize = 20,color='blue')

plt.grid()
# Plot numerical variables and compare them to a contestant's number of followers.



columns = list(df.select_dtypes(exclude='object').columns)

plt.figure(figsize=[20,30])

for i in range(len(df.select_dtypes(exclude='object').columns)):

    try:

        ax = plt.subplot(10,4,i+1)

        plt.scatter(df[columns[i]],df.Followers,alpha=0.5)

        plt.title(columns[i])

        plt.xlabel(columns[i])

        plt.ylabel('Followers')

        box = ax.get_position()

        box.y1 = box.y1 - 0.01 

        ax.set_position(box)

    except:

        pass

plt.show()
# Plot significant correlations and their values



df_num = df.select_dtypes(include = ['float64', 'int64'])

corr = df_num.corr()

plt.figure(figsize=(8, 8))



sns.heatmap(corr[(corr >= 0) | (corr <= -0.4)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
# Assign a score based on the status



pd.options.mode.chained_assignment = None

df = df.assign(Score=0)



df.Score[df.Status=='Winner'] = 5

df.Score[df.Status=='Runner-up'] = 4

df.Score[df.Status=='Third place'] = 3

df.Score[df.Status=='Fourth place'] = 2

df.Score[df.Status=='Dumped'] = 1
df.head(10)
# Run kmeans. Start by selecting the Followers and Score columns, then run Elbow.

# Courtesy of https://www.kaggle.com/vjchoudhary7/kmeans-clustering-in-customer-segmentation



from sklearn.cluster import KMeans



kmeans_df = df.iloc[:, [6, 9]].values

wcss=[]

max_clusters = 10



for i in range(1, max_clusters + 1):

    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)

    kmeans.fit(kmeans_df)

    wcss.append(kmeans.inertia_)

    

#Visualizing the ELBOW method to get the optimal value of K



plt.plot(range(1,11), wcss)

plt.title('The Elbow Method')

plt.xlabel('no of clusters')

plt.ylabel('wcss')

plt.show()
# 3 clusters seems appropriate, run Kmeans and visualize



kmeansmodel = KMeans(n_clusters=3, init='k-means++', random_state=0)

y_kmeans = kmeansmodel.fit_predict(kmeans_df)



plt.figure(figsize=[8,8])

plt.scatter(kmeans_df[y_kmeans == 0, 0], kmeans_df[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(kmeans_df[y_kmeans == 1, 0], kmeans_df[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(kmeans_df[y_kmeans == 2, 0], kmeans_df[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of participants')

plt.xlabel('Followers')

plt.ylabel('Score')

plt.legend()

plt.show()
import statsmodels.api as sm

from sklearn import preprocessing



# Extract X/y, then convert to arrays and scale to unit variance

X = df.Followers

y = df.Score



X_arr = np.array(X).reshape(-1, 1)

y_arr = np.array(y).reshape(-1, 1)



X_arr = np.divide(X_arr - X_arr.min(), X_arr.max() - X_arr.min())

y_arr = np.divide(y_arr - y_arr.min(), y_arr.max() - y_arr.min())



x_with_ones = sm.add_constant(X_arr, prepend=False)



# Make a linear regression model using OLS



logit_model = sm.Logit(y_arr, x_with_ones)

lm = logit_model.fit()

print(lm.summary())
# Print the confidence intervals for the model coefficients

lm.conf_int()
# Print the p-values for the model coefficients

lm.pvalues