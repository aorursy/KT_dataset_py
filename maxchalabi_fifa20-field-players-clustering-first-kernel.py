# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.cluster import KMeans

from sklearn import preprocessing

import datetime as dt

from statistics import median

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

import math

from math import pi

from scipy.spatial.distance import cdist

from IPython.display import display, HTML



style.use('ggplot')
# reading in data

player_data = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_20.csv')



# Cleaning & Dropping Data



# dropping player ratings for each individual position as they are already contained in other cells or could be determined from other columns

player_data.drop(['ls','st','rs','lw','lf','cf','rf','rw','lam','cam','ram','lm','lcm','cm','rcm','rm','lwb','ldm','cdm','rdm','rwb','lb','lcb','cb','rcb','rb'], 1, inplace=True)



# dropping uninteresting columns that won't be used later on

player_data.drop(['sofifa_id','player_url','long_name', 'loaned_from'], 1, inplace=True)



# Only keep the first position listed for each player in the column player_position (their best position)

player_data['player_positions'] = player_data['player_positions'].str.split(',').str[0]



# Dropping all Goalkeepers from data

player_data = player_data[(player_data['player_positions'] != 'GK' )]



# Only keeping players with a rating higher than 75

player_data = player_data[(player_data['overall']  > 75)]

# player_data = player_data[(player_data['overall'] < 67 )]



# Dropping all GK skill-related columns, since we don't want to group field players based on their goalkeeping abilities

player_data.drop(['gk_diving',	'gk_handling',	'gk_kicking',	'gk_reflexes',	'gk_speed',	'gk_positioning',

                  'goalkeeping_diving',	'goalkeeping_handling',	'goalkeeping_kicking',	'goalkeeping_positioning',	'goalkeeping_reflexes'], 1, inplace=True)



# filling all missing values with a 0, so they'll be handled as an outlier

player_data.fillna(0, inplace=True)



# Even if many columns still contained in player_data won't be used for our clustering, I saved a copy into the variable original_data to plot some of those 

# columns (e.g.: height, age, work_rate, etc..) based on their cluster group.

original_data = pd.DataFrame.copy(player_data)



# Dropping all other columns except for skill attributes

player_data.drop(['player_positions'], 1, inplace=True)

player_data.drop(['overall'], 1, inplace=True)

player_data.drop(['short_name','age','dob','height_cm','weight_kg','nationality','club','potential','value_eur','wage_eur',

                  'preferred_foot','international_reputation','weak_foot','skill_moves','work_rate','body_type','real_face','release_clause_eur',

                  'player_tags','team_position','team_jersey_number','joined','contract_valid_until','nation_position','nation_jersey_number'], 1, inplace=True)

player_data.drop(['player_traits'], 1, inplace=True)





player_data.drop(['pace', 'shooting', 'dribbling', 'passing', 'physic', 'defending'], 1, inplace=True)



print('Length of dataset:', len(player_data))



fig, ax = plt.subplots(figsize=(10,10)) 

plt.title('Correlation heatmap of skill attributes in dataset')

sns.heatmap(player_data.corr())

# Defining and scaling X data (all skill attributes for each player)



X = np.array(player_data)

X = preprocessing.scale(X)



WCSS = []

K = range(1,30)

for k in K:

  kmeansmodel = KMeans(n_clusters=k)

  kmeansmodel.fit(X)

  # distortions.append(sum(np.min(cdist(X, SpectralClusteringModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

  WCSS.append(kmeansmodel.inertia_)



# Plot the elbow

fig = plt.figure(figsize=(25,10))

ax1 = fig.add_subplot(1,2,1)

ax1.plot(K, WCSS, 'bx-')

plt.xlabel('k')

plt.xticks(list(K), list(K))

plt.ylabel('Within Cluster Sum of Squares')

plt.title('The Elbow Method showing the optimal k')



price_series = pd.Series(WCSS)

ax2 = fig.add_subplot(1,2,2)

WCSS_as_pdframe = pd.Series(WCSS)

ax2.plot(WCSS_as_pdframe.pct_change())

plt.xlabel('k')

plt.xticks(list(K), list(K))

plt.ylabel('Percent Change')

plt.title('Percent change using Elbow Method')



plt.show()

from sklearn.metrics import silhouette_samples, silhouette_score



K = range(2,15)

for i, k in enumerate(K):

  fig, ax1 = plt.subplots(1, 1)

  fig.set_size_inches(18, 7)

  

  # Run the Kmeans algorithm

  clf = KMeans(n_clusters=k)

  labels = clf.fit_predict(X)

  centroids = clf.cluster_centers_



  # Get silhouette samples

  silhouette_vals = silhouette_samples(X, labels)



  # Silhouette plot

  y_ticks = []

  y_lower, y_upper = 0, 0

  for i, cluster in enumerate(np.unique(labels)):

      cluster_silhouette_vals = silhouette_vals[labels == cluster]

      cluster_silhouette_vals.sort()

      y_upper += len(cluster_silhouette_vals)

      ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)

      ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))

      y_lower += len(cluster_silhouette_vals)



  # Get the average silhouette score and plot it

  avg_score = np.mean(silhouette_vals)

  ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')

  ax1.set_yticks([])

  ax1.set_xlim([-0.1, 1])

  ax1.set_xlabel('Silhouette coefficient values')

  ax1.set_ylabel('Cluster labels')

  ax1.set_title('Silhouette plot for the various clusters', y=1.02);

  

  plt.tight_layout()

  plt.suptitle(f'Silhouette analysis using k = {k}',

                fontsize=16, fontweight='semibold', y=1.05);
n_clusters_ = 5

clf = KMeans(n_clusters=n_clusters_, n_init=100)

clf.fit(X)



centroids = clf.cluster_centers_

labels = clf.labels_



# creating a new column in original_data, assigning each player their cluster labels

original_data['cluster_group'] = np.nan

for i in range(len(X)):

  original_data['cluster_group'].iloc[i] = labels[i]



print('Cluster Sizes:')

for i in range(n_clusters_):

  print(len(original_data[(original_data['cluster_group']==i)]))
if n_clusters_ <= 4:

  sizer = n_clusters_

else: 

  sizer = 4



nrows_ = math.ceil(1*(n_clusters_/4))

ncols_ = 4



fig = plt.figure(figsize=(sizer*7,3*nrows_))

plt.subplots_adjust(top = 2)



def most_common_player_position(cluster_label):

  cluster = original_data[(original_data['cluster_group']==cluster_label)]

  v_counts = pd.value_counts(cluster['player_positions'])

  position = v_counts.index[0]

  return position



def bar_chart_value_counts(column, nr_columns, rotate_xlabels, text):

  fig = plt.figure(figsize=(sizer*7,6*nrows_))

  plt.subplots_adjust(hspace = 0.4)

  for group in range(n_clusters_):

    cluster = original_data[(original_data['cluster_group']==group)]

    v_counts = pd.value_counts(cluster[column])

    v_counts = v_counts[:nr_columns]

    ax1 = fig.add_subplot(nrows_, ncols_, group+1)

    ax1.bar(v_counts.index, v_counts.values)

    ax1.set(title=most_common_player_position(group), ylabel='Number of Players')

    if rotate_xlabels == True:

      for label in ax1.xaxis.get_ticklabels():

        label.set_rotation(45)

    if text == True:

      barchartindex = int(len(v_counts)/2)

      ax1.text(v_counts.index[barchartindex], 0.9*v_counts.values[0],

               'Top rated players:', style='italic',

               bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 7})

      nr_names = 5

      for name in range(nr_names):

        top_name = cluster['short_name'].values[name]

        position = cluster['player_positions'].values[name]

        top_name = top_name + '    ' + position

        y_index = (0.82 - (name*0.05)) * v_counts.values[0]

        ax1.text(v_counts.index[barchartindex], y_index, str(top_name))



bar_chart_value_counts('player_positions', 3, True, True)

bar_chart_value_counts('club', 6, True, False)
def boxplot_per_clustergroup(variable_of_interest):

  fig, ax = plt.subplots(figsize=(10,10))  

  sns.boxplot(x=original_data['cluster_group'], y=pd.to_numeric(original_data[variable_of_interest], downcast="float"))

  xlabellist = []

  for pos in range(n_clusters_):

    xlabellist.append(most_common_player_position(pos)) 

  plt.xticks(list(range(0, n_clusters_)), xlabellist)



boxplot_per_clustergroup('overall')

boxplot_per_clustergroup('potential')
# Taken and adapted from https://python-graph-gallery.com/392-use-faceting-for-radar-chart/

 

# Set data

shooting = []

passing = []

dribbling = []

defending = []

physic = []

pace = []



attribute_names = 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'pace'

attribute_means = [shooting, passing, dribbling, defending, physic, pace]





def get_attribute_per_cluster_means(attribute_names_list, attribute_means_list):

  cluster_names = []

  for group in range(n_clusters_):

    cluster = original_data[(original_data['cluster_group']==group)]

    cluster_names.append(most_common_player_position(group))

    for i, attribute in enumerate(attribute_names_list):

      mean = np.average(cluster[attribute])

      attribute_means_list[i].append(mean)

  dictionary = {'cluster': cluster_names}

  for i in range(len(attribute_names_list)):

    dictionary.update({attribute_names_list[i]: attribute_means_list[i]})

  df_means = pd.DataFrame(dictionary)

  return df_means



df_means_main = get_attribute_per_cluster_means(attribute_names, attribute_means)





# Radar Chart Function 

# ------- PART 1: Define a function that do a plot for one line of the dataset!

 

def make_spider(df_means,row, title, color):

  # number of variable

  categories=list(df_means)[1:]

  N = len(categories)

  

  # What will be the angle of each axis in the plot? (we divide the plot / number of variable)

  angles = [n / float(N) * 2 * pi for n in range(N)]

  angles += angles[:1]

  

  # Initialise the spider plot

  ax = plt.subplot(nrows_, ncols_,row+1, polar=True, )

  

  # If you want the first axis to be on top:

  ax.set_theta_offset(pi / 2)

  ax.set_theta_direction(-1)

  

  # Draw one axe per variable + add labels labels yet

  plt.xticks(angles[:-1], categories, color='grey', size=8)

  

  # Draw ylabels

  ax.set_rlabel_position(0)

  plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)

  plt.ylim(0,100)

  

  # Ind1

  values=df_means.loc[row].drop('cluster').values.flatten().tolist()

  values += values[:1]

  ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')

  ax.fill(angles, values, color=color, alpha=0.4)

  

  # Add a title

  plt.title(title, size=11, color=color, y=1.1)

 

# ------- PART 2: Apply to all individuals

# initialize the figure

my_dpi=96

plt.figure(figsize=(1750/my_dpi, 1750/my_dpi), dpi=my_dpi)

if n_clusters_ <= 8:

  bottom_ = 0.45

  top_ = 0.9

  hspace_ = 0.05

if n_clusters_ > 8 & n_clusters_ <= 12:

  bottom_ = 0.45

  top_ = 0.9

  hspace_ = 0.4

plt.subplots_adjust(bottom= bottom_, top=top_, hspace = hspace_)

 

# Create a color palette:

my_palette = plt.cm.get_cmap("Set2", len(df_means_main.index))

 

# Loop to plot

for row in range(len(df_means_main.index)):

  make_spider(df_means=df_means_main, row=row, title='Cluster '+most_common_player_position(row), color=my_palette(row))
attacking_crossing = []

attacking_finishing = []

attacking_heading_accuracy = []

attacking_short_passing = []

attacking_volleys = []

skill_dribbling = []

skill_curve = []

skill_fk_accuracy = []

skill_long_passing = []

skill_ball_control = []

movement_acceleration = []

movement_sprint_speed = []

movement_agility = []

movement_reactions = []

movement_balance = []

power_shot_power = []

power_jumping = []

power_stamina = []

power_strength = []

power_long_shots = []

mentality_aggression = []

mentality_interceptions = []

mentality_positioning = []

mentality_vision = []

mentality_penalties = []

mentality_composure = []

defending_marking = []

defending_standing_tackle = []

defending_sliding_tackle = []



attribute_names = ['attacking_crossing','attacking_finishing','attacking_heading_accuracy','attacking_short_passing',

                   'attacking_volleys','skill_dribbling','skill_curve','skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',

                   'movement_acceleration','movement_sprint_speed', 'movement_agility','movement_reactions','movement_balance',

                   'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength','power_long_shots',

                   'mentality_aggression','mentality_interceptions','mentality_positioning','mentality_vision','mentality_penalties','mentality_composure',

                   'defending_marking','defending_standing_tackle','defending_sliding_tackle']

attribute_means = [attacking_crossing,attacking_finishing,attacking_heading_accuracy,attacking_short_passing,

                   attacking_volleys,skill_dribbling,skill_curve,skill_fk_accuracy, skill_long_passing, skill_ball_control,

                   movement_acceleration,movement_sprint_speed, movement_agility,movement_reactions,movement_balance,

                   power_shot_power, power_jumping, power_stamina, power_strength,power_long_shots,

                   mentality_aggression,mentality_interceptions,mentality_positioning,mentality_vision,mentality_penalties,mentality_composure,

                   defending_marking,defending_standing_tackle,defending_sliding_tackle]



df_means_main = get_attribute_per_cluster_means(attribute_names, attribute_means)



my_dpi=120

plt.figure(figsize=(25,25), dpi=my_dpi)

if n_clusters_ <= 8:

  bottom_ = 0.6

  top_ = 0.9

  hspace_ = 0.1

if n_clusters_ > 8 & n_clusters_ <= 12:

  bottom_ = 0.45

  top_ = 0.9

  hspace_ = 0.1

plt.subplots_adjust(bottom= bottom_, top=top_, hspace = hspace_, wspace=0.5)

 

# Loop to plot

for row in range(0, len(df_means_main.index)):

  make_spider(df_means=df_means_main, row=row, title='Cluster '+most_common_player_position(row), color=my_palette(row))
columns = 'height_cm', 'weight_kg', 'value_eur', 'age'

xlabels = 'Height in cm', 'Weight in kg', 'Value in €', 'Age'



medians = []

plottingarrays = []

for i in range(len(xlabels)):

  medians.append([])

  plottingarrays.append([])

  for _ in range(n_clusters_):

    plottingarrays[i].append([])



for col, savingarray, mediandata in zip(columns, plottingarrays, medians):

  med = median(np.array(original_data[col]))

  mediandata.append(med)

  for group, array in zip(range(n_clusters_), savingarray):

    cluster = original_data[(original_data['cluster_group']==group)]

    att_data = np.array(cluster[col])

    array.append(att_data)



colours = 10*['g', 'b', 'c', 'y', 'k']



for pdata, xlabel, c, m in zip(plottingarrays, xlabels, colours, medians):

  fig = plt.figure(figsize=(sizer*7,3*nrows_))

  for data, group in zip(pdata, range(n_clusters_)):

    ax1 = fig.add_subplot(nrows_,ncols_, group+1)

    ax1.hist(data, bins='auto', histtype='bar', rwidth=0.8, color=c, alpha=0.45, linewidth=4)

    ax1.axvline(m)

    plt.xlabel(xlabel)

    plt.title(most_common_player_position(group))

  plt.subplots_adjust(hspace = 0.5)

  plt.show()
attribute_names = ['attacking_crossing','attacking_finishing','attacking_heading_accuracy','attacking_short_passing',

                   'attacking_volleys','skill_dribbling','skill_curve','skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',

                   'movement_acceleration','movement_sprint_speed', 'movement_agility','movement_reactions','movement_balance',

                   'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength','power_long_shots',

                   'mentality_aggression','mentality_interceptions','mentality_positioning','mentality_vision','mentality_penalties','mentality_composure',

                   'defending_marking','defending_standing_tackle','defending_sliding_tackle']



from scipy.stats import f_oneway, ttest_ind



Fvals_list = []

index_list = []

important_feature_names_per_cluster = []

sorted_Fval_array = []

for i in range(n_clusters_):

  Fvals_list.append([])

  index_list.append([])





for i, F_list, i_list in zip(range(n_clusters_), Fvals_list, index_list):

  for ii, index in zip(attribute_names, range(len(attribute_names))):

    cluster = original_data[(original_data['cluster_group']==i)]

    other_clusters = original_data[(original_data['cluster_group']!=i)]

    cluster_attribute = np.array(cluster[ii])

    mean_cluster = np.average(cluster_attribute)

    other_clusters_attribute = np.array(other_clusters[ii])

    mean_other_clusters = np.average(other_clusters_attribute)

    F_onewayResult, p_onewayResult = ttest_ind(cluster_attribute, other_clusters_attribute)

    if mean_cluster > mean_other_clusters:

      if p_onewayResult < (0.05/(len(attribute_names)*n_clusters_)):

        F_list.append(F_onewayResult)

        i_list.append(ii)





for list_index in range(len(index_list)):

  ff = np.array(Fvals_list[list_index])

  sorted_F_vals = sorted(ff, reverse=True)

  ind = index_list[list_index]

  sorted_feature_index = [x for _,x in sorted(zip(ff,ind))]

  important_feature_names_per_cluster.append(sorted_feature_index)

  sorted_Fval_array.append(sorted_F_vals)



for i in range(n_clusters_):

  fig = plt.figure()

  ax1 = fig.add_subplot(1,1,1)

  ax1.bar(list(range(len(sorted_Fval_array[i]))), sorted_Fval_array[i])

  plt.xticks(list(range(len(important_feature_names_per_cluster[i]))), important_feature_names_per_cluster[i])

  for label in ax1.xaxis.get_ticklabels():

    label.set_rotation(90)



  plt.title(f'{most_common_player_position(i)}- most important feature: {important_feature_names_per_cluster[i][0]}')





  plt.show()
scaled_Fvals = []

for i in range(n_clusters_):

  scaled_Fvals.append([])

for i, each_list in enumerate(Fvals_list):

  scaled_list = each_list/np.sum(each_list)

  scaled_Fvals[i].append(scaled_list)





weighted_scores_by_cluster_dfs = []

for i in range(n_clusters_):

  weighted_scores_by_cluster_dfs.append([])



for group in range(n_clusters_):

  cluster = original_data[(original_data['cluster_group']==group)]

  for jj, i_list in enumerate(index_list):

    df_for_weighting = cluster[i_list]

    weight_col_list = []

    for ii, col in enumerate(i_list):

      df_for_weighting[f'cluster_{most_common_player_position(jj)}_{col}_weighted'] = df_for_weighting[col]*scaled_Fvals[jj][0][ii]

      weight_col_list.append(f'cluster_{most_common_player_position(jj)}_{col}_weighted')  

    df_for_weighting['weighted_cluster_specific_score'] = df_for_weighting[weight_col_list].sum(axis = 1)

    df_for_weighting = df_for_weighting.sort_values(by='weighted_cluster_specific_score', ascending=False)

    weighted_scores_by_cluster_dfs[group].append(df_for_weighting['weighted_cluster_specific_score'])





for main_cluster in weighted_scores_by_cluster_dfs:

  for i, cluster in enumerate(main_cluster):

    for wscore, Uindex in zip(cluster.values, cluster.index):

      original_data.loc[Uindex, f'{most_common_player_position(i)}_score'] = wscore
top_names_savarrs = []

for i in range(n_clusters_):

  top_names_savarrs.append([])



for i in range(n_clusters_):

  cluster = original_data[(original_data['cluster_group']==i)]

  for j in range(n_clusters_):

    columns = ['short_name','player_positions',f'{most_common_player_position(j)}_score','overall']

    cluster_ranked_names = cluster[columns].sort_values(by=f'{most_common_player_position(j)}_score', ascending=False)

    cluster_ranked_names = cluster_ranked_names.round(2)

    top_names = cluster_ranked_names.reset_index()

    top_names.drop(['index'], axis=1, inplace=True)

    top_names_savarrs[i].append(top_names)





for i, cluster_savarr in enumerate(top_names_savarrs):

  all_topnames_per_cluster_df = pd.concat([savarr for savarr in cluster_savarr], axis=1)

  print('Cluster Group:',most_common_player_position(i))

  head_df = all_topnames_per_cluster_df.head(10)

  display(HTML(head_df.to_html()))