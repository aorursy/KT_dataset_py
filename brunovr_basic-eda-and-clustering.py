import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
pokemon_df = pd.read_csv('../input/complete-pokemon-dataset-updated-090420/pokedex_(Update_05.20).csv')

pokemon_df.head()
pokemon_df.tail()
pokemon_df.info()
pokemon_df.shape
pokemon_df.columns
pokemon_df = pokemon_df.drop('Unnamed: 0', axis=1)



columns_to_drop = ['japanese_name', 'german_name', 'against_normal', 'against_fire',

                  'against_water', 'against_electric', 'against_grass', 'against_ice',

                  'against_fight', 'against_poison', 'against_ground', 'against_flying',

                  'against_psychic', 'against_bug', 'against_rock', 'against_ghost',

                  'against_dragon', 'against_dark', 'against_steel', 'against_fairy'

                  ]



pokemon_df = pokemon_df.drop(columns_to_drop, axis = 1)
# Get index and print row of pokemon having highest total_points

highest_tot_points_idx = pokemon_df['total_points'].idxmax()

pokemon_df.loc[highest_tot_points_idx,:]
# Select mega pokemons, dinamax and alolan pokemons

mega_pokemons = pokemon_df.index[pokemon_df['name'].apply(lambda x: 'Mega ' in x)].tolist()

dinamax_pokemons = pokemon_df.index[pokemon_df['name'].apply(lambda x: 'max' in x)].tolist()

alolan_pokemons = pokemon_df[pokemon_df.name.apply(lambda x: 'Alolan' in x) == True].index.tolist()



# Concatenate

to_delete = np.concatenate((mega_pokemons, dinamax_pokemons, alolan_pokemons))



# Remove

pokemon_df = pokemon_df.drop(to_delete, axis=0)
# Cheacking again after dropping mega, dinamax and alolan:

# Get index and print row of pokemon having highest total_points

highest_tot_points_idx = pokemon_df['total_points'].idxmax()

pokemon_df.loc[highest_tot_points_idx,:]
features_stats = ['total_points', 'hp', 'attack', 'defense',

       'sp_attack', 'sp_defense', 'speed']
def find_min_and_max(column_name):

    '''

    Get pokemon name according to its max and min attribute: column_name

    column_name: array of int or float

    '''

    

    # Find max

    max_index = pokemon_df[column_name].idxmax()

    max_pokemon = pokemon_df.loc[max_index, 'name']

    

    # Find min

    min_index = pokemon_df[column_name].idxmin()

    min_pokemon = pokemon_df.loc[min_index, 'name']

    

    print(f'Pokemon with min {column_name}: {min_pokemon}\nPokemon with max {column_name}: {max_pokemon}\n')

    return min_index, max_index
features_stats


min_dict = {}

max_dict = {}

max_labels=[]

min_labels=[]



for stat in features_stats:

    min_index, max_index = find_min_and_max(stat)

    max_dict[stat] = pokemon_df.loc[max_index, stat]

    min_dict[stat] = pokemon_df.loc[min_index, stat]

    max_labels.append(pokemon_df.loc[max_index, 'name'])

    min_labels.append(pokemon_df.loc[min_index, 'name'])
X = np.arange(len(max_dict))

fig, ax = plt.subplots(1, figsize=(10,10))



p1 = ax.bar(X, max_dict.values(), width=0.4, color='b', align='center')

p2 = ax.bar(X-0.4, min_dict.values(), width=0.4, color='g', align='center')

ax.legend(('Max values','Min values'))

plt.xticks(X, max_dict.keys())

plt.title("Min Max values", fontsize=17)

plt.grid()



def autolabel(bar_plot, bar_label):

    for idx,rect in enumerate(bar_plot):

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,

                bar_label[idx],

                ha='center', va='bottom', rotation=45)

autolabel(p1, max_labels)

autolabel(p2, min_labels)

plt.ylim((0,900))

plt.show()
t1_by_gen = pd.crosstab(pokemon_df['generation'],pokemon_df['type_1'])

t1_by_gen
fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(t1_by_gen, annot=True, linewidths=0.5, cmap="BuPu");
t2_by_gen = pd.crosstab(pokemon_df['generation'],pokemon_df['type_2'])
fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(t2_by_gen, annot=True, linewidths=0.5, cmap="BuPu");
stats_df = pokemon_df[features_stats]
stats_df.head()
stats_df.describe()
stats_df.corr().round(2)
sns.jointplot(x=stats_df['attack'], y=stats_df['sp_defense'], kind="kde");
sns.jointplot(x=stats_df['speed'], y=stats_df['total_points'], kind="kde");
scatter_matrix(stats_df, figsize=[12,12])



plt.show()
# Checking sp attack vs. speed

scatter_matrix(stats_df.iloc[: , [4,6]])



plt.show()
# Using seaborn to analyze distributions



# Let's check total_points first



sns.distplot(stats_df['total_points']);
# Defining a function to plot distribtion of stat and fiting to a normal curve

# Function plots the fitting parameters



def fit_stats(df, stat, show = True, label=None):

    # attack

    ax = sns.distplot(df[stat], label=label, kde=False, fit=stats.norm);



    # Get the fitted parameters used by sns

    (mu, sigma) = stats.norm.fit(df[stat])



    print (f"mu={mu}, sigma={sigma}")

    # Legend and labels 

    plt.legend(["normal dist. fit ($\mu=${0:.2g}, $\sigma=${1:.2f})".format(mu, sigma)])

    plt.ylabel('Frequency')



    # Cross-check this is indeed the case - should be overlaid over black curve

    x_dummy = np.linspace(stats.norm.ppf(0.01), stats.norm.ppf(0.99), 100)

    ax.plot(x_dummy, stats.norm.pdf(x_dummy, mu, sigma))

    plt.legend(["normal dist. fit ($\mu=${0:.2g}, $\sigma=${1:.2f})".format(mu, sigma),

               "cross-check"])

        

    if not show:

        plt.close()



    return [mu, sigma]
fit_stats(stats_df,'attack')
fit_stats(stats_df,'speed')
features_stas_gen = ['total_points', 'hp', 'attack', 'defense',

       'sp_attack', 'sp_defense', 'speed', 'status', 'generation']
status_df = pokemon_df[features_stas_gen]

status_df.head()
fig, ax = plt.subplots(1,1,figsize=(8,8))

y = pd.value_counts(status_df['generation'].values) 

y.plot(kind='barh', ax=ax)

for i, v in enumerate(y):

    ax.text(v + 3, i , str(v), color='blue', fontweight='bold')

plt.xlim((0,185))

plt.xlabel('Count')

plt.ylabel('Generation')

plt.grid()
# Encoding 'status' column

le = LabelEncoder()

status_df['status_encoded'] = le.fit_transform(status_df['status'])
status_df.head()
status_df.tail()
status_df['status_encoded'].unique()
status_df['status'].unique()
status_name_dict = {'Legendary' : 0,

                    'Mythical' : 1,

                    'Normal' : 2,

                    'Sub Legendary' : 3}



scatter = plt.scatter(status_df['attack'], status_df['sp_attack'],

                      c = status_df['status_encoded'])



plt.xlabel('attack')

plt.ylabel('sp_attack')



plt.legend(handles=scatter.legend_elements()[0], labels = status_name_dict.keys())



plt.grid()

plt.show()
scatter = plt.scatter(status_df['attack'], status_df['sp_attack'],

                      c = status_df['generation'])



plt.xlabel('attack')

plt.ylabel('sp_attack')



plt.legend(handles=scatter.legend_elements()[0], labels = list(range(1,9)))



plt.grid()

plt.show()
# Filtering by generation:



gen_1 = status_df.loc[status_df['generation'] == 1]

gen_1.head()
gen_1.describe()
gen_dict = {}

for i in range(1,9):

    gen_dict[f'gen_{i}'] = status_df.loc[status_df['generation'] == i]
def plot_radar(gen):

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="polar")



    # theta has 6 different angles, and the first one repeated

    use = gen.drop(['total_points','generation','status_encoded'], axis=1).describe()

    theta = np.arange(len(use.columns) + 1) / float(len(use.columns)) * 2 * np.pi

    # values has the 6 values from stats, with the first element repeated

    values = use.loc['mean'].values

    values = np.append(values, values[0])



    # draw the polygon and the mark the points for each angle/value combination

    l1, = ax.plot(theta, values, color="C2", marker="o", label="Name of Col B")

    plt.xticks(theta[:-1], use.columns, color='grey', size=12)

    ax.tick_params(pad=10) # to increase the distance of the labels to the plot

    # fill the area of the polygon with green and some transparency

    ax.fill(theta, values, 'green', alpha=0.1)



    # plt.legend() # shows the legend, using the label of the line plot (useful when there is more than 1 polygon)

    plt.title("Mean generation 1 Pokemon")

    plt.show()
plot_radar(gen_dict['gen_1'])
def plot_radar_all(gen_dict):

    fig = plt.figure(figsize=(15,15))

    i=1

    for k in gen_dict:

        gen=gen_dict[k]

        ax = fig.add_subplot(2,4,i, projection="polar")



        # theta has 6 different angles, and the first one repeated

        use = gen.drop(['total_points','generation','status_encoded'], axis=1).describe()

        theta = np.arange(len(use.columns) + 1) / float(len(use.columns)) * 2 * np.pi

        # values has the 6 values from stats, with the first element repeated

        values = use.loc['mean'].values

        values = np.append(values, values[0])



        # draw the polygon and the mark the points for each angle/value combination

        l1, = ax.plot(theta, values, color="C2", marker="o", label="Name of Col B")

        plt.xticks(theta[:-1], use.columns, color='grey', size=12)

        ax.tick_params(pad=10) # to increase the distance of the labels to the plot

        # fill the area of the polygon with green and some transparency

        ax.fill(theta, values, 'green', alpha=0.1)



        # plt.legend() # shows the legend, using the label of the line plot (useful when there is more than 1 polygon)

        plt.title("Mean gen"+ str(i) +" Pokemon")

        i+=1

        fig.tight_layout(pad=3.0)

    plt.show()
plot_radar_all(gen_dict)
# Function to plot stat by generation

def plotStatGen(df, stat):

    gen_dict = {}

    for i in range(1,9):

        gen_dict[f'gen_{i}'] = df.loc[df['generation'] == i]

        

    gen_stat = []

    for k in gen_dict:

        gen_stat.append(gen_dict[k][stat].mean())

    

    plt.plot(range(1,9), gen_stat, 'o-')

    plt.title('Mean '+ stat + ' by genertion')

    plt.xlabel('Generation')

    plt.ylabel('Mean ' + stat)



    plt.grid()

    plt.show()
plotStatGen(status_df, 'attack')
plotStatGen(status_df, 'sp_attack')
for i in features_stats:

    plotStatGen(status_df, i)
# We can look at the mean and std of a stat distribution by generation using seaborn and the gen's dataframes.



# Example with 'attack' in gen 1

fit_stats(gen_dict['gen_1'], 'attack')

for k in gen_dict:

    fit_stats(gen_dict[k], 'attack', show=True, label=k)

plt.legend();
# Let's just plot the mean and std from fit by generation

gen_attack_params=[]

for k in gen_dict:

    gen_attack_params.append(fit_stats(gen_dict[k], 'attack', show=False, label=k))
gen_attack_params = np.array(gen_attack_params)

gen_attack_params
# As function

def mean_std_gen(gen_dict, stat):

    gen_params=[]

    

    for k in gen_dict:

        gen_params.append(fit_stats(gen_dict[k], stat, show=False, label=k))

    plt.close()

    gen_params = np.array(gen_params)

    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    ax1.plot(range(1,9),gen_params[:,0], 'o-')

    ax1.grid()

    ax1.set_xlabel('Generation')

    ax1.set_ylabel('Mean '+stat)

    ax2.plot(range(1,9),gen_params[:,1], 'o-')

    ax2.grid()

    ax2.set_xlabel('Generation')

    ax2.set_ylabel('std of '+stat)

    

    fig.suptitle('Parameters for '+stat)

mean_std_gen(gen_dict, 'total_points')
mean_std_gen(gen_dict, 'speed')
# Instantiate the scaler

scaler = StandardScaler()



# Compute mean and std to be used for scaling



scaler.fit(stats_df)
# Mean

print(scaler.mean_)



# Std

print(scaler.scale_)
X = scaler.transform(stats_df)



X
# Sanity check

X.mean(axis = 0)
X.std(axis=0)
# K-means modeling



# Instantiate

kmeans = KMeans(n_clusters = 3)



# Fit

kmeans.fit(X)



# Make predictions

y_preds = kmeans.predict(X)



print(y_preds)
unique_poke, counts_poke = np.unique(y_preds, return_counts=True)

print(unique_poke)

print(counts_poke)
# Turn into dict

clusters = dict(zip(unique_poke, counts_poke))



clusters
# Coordinates of the three centroids

kmeans.cluster_centers_
stats_df.head()
plt.scatter(X[:,0], X[:,1], c = y_preds)



# Identifying centroids

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],

            marker='*', s = 250, c = [0,1,2], edgecolors='k')



plt.xlabel('total_points')

plt.ylabel('hp')

plt.title('k-means - k=3')

plt.show();
plt.scatter(X[:,2], X[:,4], c = y_preds)



# Identifying centroids

plt.scatter(kmeans.cluster_centers_[:,2], kmeans.cluster_centers_[:,4],

            marker='*', s = 250, c = [0,1,2], edgecolors='k')



plt.xlabel('attack')

plt.ylabel('sp_attack')

plt.title('k-means - k=3')

plt.show();
# Calculate inertia for a range of clusters number

inertia = []



for i in np.arange(1,11):

    km = KMeans(n_clusters = i)

    km.fit(X)

    inertia.append(km.inertia_)

    

# Plotting

plt.plot(np.arange(1,11), inertia, marker = 'o')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')

plt.grid()

plt.show();
sns.jointplot(x=stats_df['speed'], y=stats_df['total_points'], kind="kde");
reduced_df = stats_df[['speed', 'total_points']]

reduced_df
scatter_matrix(reduced_df)



plt.show()
# Instantiate the scaler

scaler = StandardScaler()



# Compute mean and std to be used for scaling



scaler.fit(reduced_df)

X = scaler.transform(reduced_df)



# Instantiate

kmeans = KMeans(n_clusters = 2)



# Fit

kmeans.fit(X)



# Make predictions

y_preds = kmeans.predict(X)



print(y_preds)
plt.scatter(X[:,0], X[:,1], c = y_preds)



# Identifying centroids

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],

            marker='*', s = 250, c = [0,1], edgecolors='k')



plt.xlabel('speed')

plt.ylabel('total_points')

plt.title('k-means - k=2')

plt.show();
# Calculate inertia for a range of clusters number

inertia = []



for i in np.arange(1,11):

    km = KMeans(n_clusters = i)

    km.fit(X)

    inertia.append(km.inertia_)

    

# Plotting

plt.plot(np.arange(1,11), inertia, marker = 'o')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')

plt.grid()

plt.show();
cluster_df = pokemon_df

cluster_df['cluster'] = y_preds



cluster_df.head(10)
cluster_1 = cluster_df.loc[cluster_df['cluster'] == 0]

cluster_2 = cluster_df.loc[cluster_df['cluster'] == 1]
def cluster_dist(col):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    pd.value_counts(cluster_1[col].values).plot(kind='barh', ax = ax1)

    ax1.grid()

    ax1.set_xlabel('Cluster 1 count')

    ax1.set_ylabel(col)



    pd.value_counts(cluster_2[col].values).plot(kind='barh', ax = ax2)

    ax2.grid()

    ax2.set_xlabel('Cluster 2 count')

    ax2.set_ylabel(col)



    fig.suptitle('Cluster distribution')

    fig.tight_layout(pad=3.0)

    

cluster_dist('generation')

cluster_dist('type_1')
cluster_dist('status')