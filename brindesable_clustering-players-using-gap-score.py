import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA



seed = 0

np.random.seed(seed=seed)



import warnings

warnings.filterwarnings('ignore')
players_df = pd.read_csv('../input/data.csv', index_col=0)



# The skills are rated with a grade up to 100

skills = players_df.columns[53:81]

print('Skills: ' + ', '.join(skills))



# Drop the lines containing NaN values

players_df.dropna(axis=0, inplace=True, subset=list(skills)+['Position'])
# Filter the goalkeepers, then count the remaining players

players_df = players_df[players_df.Position!='GK']

print('Number of players: {}'.format(players_df.shape[0]))
X = players_df[skills].as_matrix()

# Normalization

X = X/np.sum(X, axis=1).reshape(-1,1)
forward_features = X[players_df.Name=='A. Griezmann'].reshape(-1)

defender_features = X[players_df.Name=='R. Varane'].reshape(-1)



sort_idx = np.argsort(forward_features)[::-1]

forward_features = forward_features[sort_idx]

defender_features = defender_features[sort_idx]

skills_sorted = skills[sort_idx]



visualization_df = pd.DataFrame({

    'Norm. Ratings': np.concatenate((forward_features, defender_features)),

    'Skills': np.concatenate((skills_sorted, skills_sorted)),

    'Player': ['Griezmann (Forward)']*len(forward_features)+['Varane (Defender)']*len(defender_features)

})

f, ax = plt.subplots(figsize=(16, 8))

sns.barplot(x='Norm. Ratings', y='Skills', hue='Player', data=visualization_df)

sns.despine(left=True, bottom=True)
# Visualisation in 2 Dimensions

X_pca = PCA(n_components=2).fit_transform(X)

sns.kdeplot(X_pca[:,0],X_pca[:,1],cmap="Reds", shade=True, shade_lowest=False)

plt.show()
# Here we build a dataset that clearly contains 5 clusters

# We'll test our algorithm on this toy example

real_k = 5



x_coord = np.array([])

y_coord = np.array([])

for i in range(real_k):

    cluster_x_coord = np.random.normal(i*10, 1, 20)

    x_coord = np.concatenate([x_coord, cluster_x_coord])

    cluster_y_coord = np.random.normal(0, 1, 20)

    y_coord = np.concatenate([y_coord, cluster_y_coord])

X_toy = np.column_stack((x_coord, y_coord))



# plot

plt.scatter(x = X_toy[:,0], y = X_toy[:,1])

plt.axis('scaled')

plt.show()
def gen_rand_dataset(dataset):

    data_size, nb_dim = dataset.shape

    rand_dataset = []

    for dim in range(nb_dim):

        # For each dimension, we generate a uniform distribution of values between the original min and max.

        rand_dataset.append(np.random.uniform(min(dataset[:,dim]), max(dataset[:,dim]), data_size))

        

    return np.matrix(rand_dataset).T



# We give our toy dataset as input

X_rand = gen_rand_dataset(X_toy)

# plot

plt.scatter(x = X_rand[:,0].A1, y = X_rand[:,1].A1)

plt.axis('scaled')

plt.show()
def get_gap(X_, k):

    X_rand = gen_rand_dataset(X_)

    inertia = KMeans(n_clusters=k).fit(X_).inertia_

    inertia_rand = KMeans(n_clusters=k).fit(X_rand).inertia_

    return inertia_rand - inertia



def get_gap_scores(X_, kmax, B = 5):

    gap_scores = []

    gap_k = []

    

    # Special case where k=1

    def get_inertia_unique_cluster(X__):

        centroid = np.mean(X__, axis=0)

        # Calculus of inertia

        return np.sum(np.apply_along_axis(lambda v: np.sum(np.square(v-centroid)), 1, X__))

    

    gap = get_inertia_unique_cluster(gen_rand_dataset(X_)) - get_inertia_unique_cluster(X_)

    gap_scores.append(gap)

    gap_k.append(1)

    

    # For k in [2,kmax]

    for k in range(2, kmax+1):

        gap = np.mean([get_gap(X_, k) for b in range(B)])

        gap_scores.append(gap)

        gap_k.append(k)

    

    return gap_k, gap_scores
# Compute the gap scores

gap_k, gap_scores = get_gap_scores(X_toy, 10)

# Plot the gap score for each k

def plot_gap_score(gap_k, gap_scores):

    plt.plot(gap_k, gap_scores)

    plt.xlabel('k')

    plt.ylabel('Gap Score')

    plt.xticks(gap_k, gap_k)

    k_best = gap_k[np.argmax(gap_scores)]

    plt.axvline(x=k_best, color='r')

    plt.show()

    

plot_gap_score(gap_k, gap_scores)
gap_k, gap_scores = get_gap_scores(X, 10)

plot_gap_score(gap_k, gap_scores)
labels = KMeans(n_clusters=2, random_state=seed).fit(X).labels_

players_df['cluster'] = labels
print('Positions: ' + ', '.join(players_df.Position.unique()))
players_df['Pos_coord_X'] = np.zeros(players_df.shape[0])

players_df.loc[players_df.Position.isin(['CB','LCB','RCB','RM','CM','CAM','CF','ST']),'Pos_coord_X'] = 0

players_df.loc[players_df.Position.isin(['LB','LWB','LDM','RCM','LM','LAM','LW','LF','LS']),'Pos_coord_X'] = -1

players_df.loc[players_df.Position.isin(['RB','RWB','RDM','LCM','RM','RAM','RW','RF','RS']),'Pos_coord_X'] = 1



players_df['Pos_coord_Y'] = np.zeros(players_df.shape[0])

players_df.loc[players_df.Position.isin(['RCB','LCB','CB','LB','RB']),'Pos_coord_Y'] = 1

players_df.loc[players_df.Position.isin(['CDM','LWB','RWB']),'Pos_coord_Y'] = 2

players_df.loc[players_df.Position.isin(['LDM','RDM','RCM','LCM','RM','CM','LM','RM','RAM','LAM']),'Pos_coord_Y'] = 3

players_df.loc[players_df.Position.isin(['RW','LW','CAM']),'Pos_coord_Y'] = 4

players_df.loc[players_df.Position.isin(['RF','LF','CF', 'ST','LS','RS']),'Pos_coord_Y'] = 5
players_df.groupby(['Pos_coord_Y', 'cluster'], as_index=False).agg({'ID': 'count'}).pivot(index='Pos_coord_Y', columns='cluster', values='ID').plot.bar(stacked=True)

plt.show()
players_df.groupby(['Pos_coord_X', 'cluster'], as_index=False).agg({'ID': 'count'}).pivot(index='Pos_coord_X', columns='cluster', values='ID').plot.bar(stacked=True)

plt.show()
total = players_df.groupby(['Pos_coord_X', 'Pos_coord_Y'], as_index=False).agg({'ID': 'count'}).pivot(index='Pos_coord_X', columns='Pos_coord_Y', values='ID').as_matrix()



def plot_distribution(distribution):

    fig, ax = plt.subplots()

    im = ax.imshow(distribution)

    ax.set_xticks(np.arange(5))

    ax.set_yticks(np.arange(3))

    ax.set_xticklabels(['Defense','','Midfield','','Attack'])

    ax.set_yticklabels(['Left', 'Center', 'Right'])

    for i in range(3):

        for j in range(5):

            col = 'b' if distribution[i, j]>50 else 'w'

            text = ax.text(j, i, '{} %'.format(distribution[i, j]),

                           ha='center', va='center', color=col)
# Defenders cluster

distribution = players_df[players_df.cluster==0].groupby(['Pos_coord_X', 'Pos_coord_Y'], as_index=False).agg({'ID': 'count'}).pivot(index='Pos_coord_X', columns='Pos_coord_Y', values='ID').fillna(0).as_matrix()

distribution = np.round(distribution / total * 100)

plot_distribution(distribution)
# Forward players cluster

distribution = players_df[players_df.cluster==1].groupby(['Pos_coord_X', 'Pos_coord_Y'], as_index=False).agg({'ID': 'count'}).pivot(index='Pos_coord_X', columns='Pos_coord_Y', values='ID').fillna(0).as_matrix()

distribution = np.round(distribution / total * 100)

plot_distribution(distribution)