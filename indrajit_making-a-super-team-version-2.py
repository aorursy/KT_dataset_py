import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import matplotlib



import warnings; warnings.simplefilter('ignore')



df = pd.read_csv('../input/FullData.csv')

Names = pd.read_csv('../input/PlayerNames.csv')

df.assign(Index=np.nan)

df['Index'] = [v.split('/')[2] for v in Names['url']]

del df['Nationality']

del df['National_Position']

del df['National_Kit']

df.head()
# Work Rate Quantification

#this can be done using for loop and if/else statement as well

work_rate_points = pd.DataFrame({'Work_Rate': df.Work_Rate.unique()})

work_rate_points['Work_Rate_Points'] = ['5','5','7.5','2.5','10','7.5','5','2.5','0']

df = df.merge(work_rate_points, on='Work_Rate', how='left')
# Age Quantification

age_points = pd.DataFrame({'Age': df.Age.unique(), 'Age_Points': [0,0,1,0,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]})

df = df.merge(age_points, on='Age', how='left')
# BMI Calculation

df['Weight'] = pd.DataFrame(df.Weight.str.split(' ',1).tolist())

df['Weight'] = df.Weight.astype(np.float)

df['Height'] = pd.DataFrame(df.Height.str.split(' ',1).tolist())

df['Height'] = df.Height.astype(np.float)

df['BMI'] = round(df.Weight*10000 / (df.Height * df.Height), 2)



df['BMI'][(df['BMI'] <= 18.5)] = 5

df['BMI'][(df['BMI'] > 18.5) & (df['BMI'] <= 25)] = 10

df['BMI'][(df['BMI'] > 25) & (df['BMI'] <= 30)] = 5

df['BMI'][(df['BMI'] > 30)] = 5
from sklearn.preprocessing import MinMaxScaler

from sklearn import cluster



#Making different datasets for defense, midfield and attack

#Defense

X_def_main = df[(df['Club_Position'] == 'CB') | (df['Club_Position'] == 'LCB') | (df['Club_Position'] == 'RCB') | (df['Club_Position'] == 'RB') | (df['Club_Position'] == 'LB') | (df['Club_Position'] == 'RWB') | (df['Club_Position'] == 'LWB')]

X_def = X_def_main[['Weak_foot','Skill_Moves','Ball_Control','Dribbling','Marking','Sliding_Tackle','Standing_Tackle','Aggression','Reactions','Attacking_Position','Interceptions','Vision','Composure','Crossing','Short_Pass','Long_Pass','Acceleration','Speed','Stamina','Strength','Balance','Agility','Jumping','Heading','Shot_Power','Finishing','Long_Shots','Curve','Freekick_Accuracy','Penalties','Volleys','GK_Positioning','GK_Diving','GK_Kicking','GK_Handling','GK_Reflexes','Work_Rate_Points', 'Age_Points', 'BMI']]

X_scaled_def = MinMaxScaler(feature_range=(0, 10)).fit_transform(X_def)

#Midfield

X_mid_main = df[(df['Club_Position'] == 'RCM') | (df['Club_Position'] == 'CAM') | (df['Club_Position'] == 'LCM') | (df['Club_Position'] == 'LM') | (df['Club_Position'] == 'LDM') | (df['Club_Position'] == 'RM') | (df['Club_Position'] == 'CDM') | (df['Club_Position'] == 'RDM') | (df['Club_Position'] == 'LAM') | (df['Club_Position'] == 'RAM')]

X_mid = X_mid_main[['Weak_foot','Skill_Moves','Ball_Control','Dribbling','Marking','Sliding_Tackle','Standing_Tackle','Aggression','Reactions','Attacking_Position','Interceptions','Vision','Composure','Crossing','Short_Pass','Long_Pass','Acceleration','Speed','Stamina','Strength','Balance','Agility','Jumping','Heading','Shot_Power','Finishing','Long_Shots','Curve','Freekick_Accuracy','Penalties','Volleys','GK_Positioning','GK_Diving','GK_Kicking','GK_Handling','GK_Reflexes','Work_Rate_Points', 'Age_Points', 'BMI']]

X_scaled_mid = MinMaxScaler(feature_range=(0, 10)).fit_transform(X_mid)

#Attack

X_att_main = df[(df['Club_Position'] == 'LW') | (df['Club_Position'] == 'RW') | (df['Club_Position'] == 'ST') | (df['Club_Position'] == 'RS') | (df['Club_Position'] == 'LF') | (df['Club_Position'] == 'LS') | (df['Club_Position'] == 'RF') | (df['Club_Position'] == 'CF')]

X_att = X_att_main[['Weak_foot','Skill_Moves','Ball_Control','Dribbling','Marking','Sliding_Tackle','Standing_Tackle','Aggression','Reactions','Attacking_Position','Interceptions','Vision','Composure','Crossing','Short_Pass','Long_Pass','Acceleration','Speed','Stamina','Strength','Balance','Agility','Jumping','Heading','Shot_Power','Finishing','Long_Shots','Curve','Freekick_Accuracy','Penalties','Volleys','GK_Positioning','GK_Diving','GK_Kicking','GK_Handling','GK_Reflexes','Work_Rate_Points', 'Age_Points', 'BMI']]

X_scaled_att = MinMaxScaler(feature_range=(0, 10)).fit_transform(X_att)
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans



#defense

pca = PCA(n_components=3)

X_r_def = pca.fit(X_scaled_def).transform(X_scaled_def)

kmeans = KMeans(init='k-means++', n_clusters=8, n_init=10)

kmeans.fit(X_r_def)

label_color_def = [matplotlib.cm.spectral(float(i) /10) for i in kmeans.labels_]

X_def_main['Cluster'] = kmeans.labels_



#midfield

X_r_mid = pca.fit(X_scaled_mid).transform(X_scaled_mid)

kmeans = KMeans(init='k-means++', n_clusters=8, n_init=10)

kmeans.fit(X_r_mid)

label_color_mid = [matplotlib.cm.spectral(float(i) /10) for i in kmeans.labels_]

X_mid_main['Cluster'] = kmeans.labels_



#attack

X_r_att = pca.fit(X_scaled_att).transform(X_scaled_att)

kmeans = KMeans(init='k-means++', n_clusters=8, n_init=10)

kmeans.fit(X_r_att)

label_color_att = [matplotlib.cm.spectral(float(i) /10) for i in kmeans.labels_]

X_att_main['Cluster'] = kmeans.labels_

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



#defense plot

xs = X_r_def[:,0]

ys = X_r_def[:,1]

zs = X_r_def[:,2]

ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs, ys, zs, c=label_color_def)



ax.set_xlabel('PC I')

ax.set_ylabel('PC II')

ax.set_zlabel('PC III')



plt.show()
check1 = X_def_main.groupby('Cluster', as_index=False).agg({'Rating': 'mean'})

check2 = X_def_main.groupby('Cluster', as_index=False).agg({'Rating': 'std'})

check1['Mean+Std'] = round(check1['Rating'] + check2['Rating'])

check1
#ss = X_def_main[(X_def_main['Cluster'] == 0) | (X_def_main['Cluster'] == 3) | (X_def_main['Cluster'] == 6) | (X_def_main['Cluster'] == 7)].sort_values('Rating', ascending = False)[:10]

ss = X_def_main[(X_def_main['Cluster'] == 6)].sort_values('Rating', ascending = False)[:10]

sns.set(style="white", context="talk")



# Set up the matplotlib figure

f, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=False)



x = np.array(list(ss['Name']))

y = np.array(list(ss['Rating']))

sns.barplot(x, y, palette=sns.cubehelix_palette(7), ax=ax)

ax.set_ylabel("Rating")

plt.tight_layout(h_pad=5)

plt.xticks(rotation=90)
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



#defense plot

xs = X_r_mid[:,0]

ys = X_r_mid[:,1]

zs = X_r_mid[:,2]

ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs, ys, zs, c=label_color_mid)



ax.set_xlabel('PC I')

ax.set_ylabel('PC II')

ax.set_zlabel('PC III')



plt.show()
check1 = X_mid_main.groupby('Cluster', as_index=False).agg({'Rating': 'mean'})

check2 = X_mid_main.groupby('Cluster', as_index=False).agg({'Rating': 'std'})

check1['Mean+Std'] = round(check1['Rating'] + check2['Rating'])

check1
ss = X_mid_main[(X_mid_main['Cluster'] == 5)].sort_values('Rating', ascending = False)[:10]

sns.set(style="white", context="talk")



# Set up the matplotlib figure

f, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=False)



x = np.array(list(ss['Name']))

y = np.array(list(ss['Rating']))

sns.barplot(x, y, palette=sns.cubehelix_palette(7), ax=ax)

ax.set_ylabel("Rating")

plt.tight_layout(h_pad=5)

plt.xticks(rotation=90)
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



#attack plot

xs = X_r_att[:,0]

ys = X_r_att[:,1]

zs = X_r_att[:,2]

ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs, ys, zs, c=label_color_att)



ax.set_xlabel('PC I')

ax.set_ylabel('PC II')

ax.set_zlabel('PC III')



plt.show()
check1 = X_att_main.groupby('Cluster', as_index=False).agg({'Rating': 'mean'})

check2 = X_att_main.groupby('Cluster', as_index=False).agg({'Rating': 'std'})

check1['Mean+Std'] = round(check1['Rating'] + check2['Rating'])

check1
ss = X_att_main[(X_att_main['Cluster'] == 2)].sort_values('Rating', ascending = False)[:10]

sns.set(style="white", context="talk")



# Set up the matplotlib figure

f, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=False)



x = np.array(list(ss['Name']))

y = np.array(list(ss['Rating']))

sns.barplot(x, y, palette=sns.cubehelix_palette(7), ax=ax)

ax.set_ylabel("Rating")

plt.tight_layout(h_pad=5)

plt.xticks(rotation=90)