import scipy, random, pickle as pkl, numpy as np, pandas as pd, seaborn as sns, scipy as sc, igraph as ig, matplotlib as mpl, matplotlib.pyplot as plt

from collections import Counter

from datetime import datetime, timedelta

from scipy.spatial import distance

from scipy.stats import multivariate_normal

from sklearn.mixture import GaussianMixture

from sklearn.metrics import precision_recall_fscore_support





# Set paramters for plotting

width = 7

height = width / 1.618

mpl.rcParams['axes.titlesize'] = 28

mpl.rcParams['axes.labelsize'] = 20

mpl.rcParams['lines.linewidth'] = 2

mpl.rcParams['lines.markersize'] = 3

mpl.rcParams['xtick.labelsize'] = 18

mpl.rcParams['ytick.labelsize'] = 18

mpl.rcParams['figure.figsize'] = (width, height)



%matplotlib inline



# Reading in the pickle file

with open('../input/Dataset.pkl', 'rb') as f:

    hourly_locs = pkl.load(f)

    feature_data = pkl.load(f)

    friends_set = pkl.load(f)

    not_friends_set = pkl.load(f)

    friends_survey = pkl.load(f)

    sbj_ids = pkl.load(f)


entropy_list = []



# Loop over all UUIDs

for uuid in range(107):

    entropy_item = {}

    try:

        

        # Calculate frequency for each location

        places_f = Counter([p for p_list in (hourly_locs[uuid][h] for h in range(24)) for p in p_list if not np.isnan(p)])

        

        # TODO: Calculate probability for each location

        sample_space = sum(places_f.values())

        

        for key in places_f:

            places_f[key] /= sample_space

            

        

        entropy = 0.0

        

        for location, probab in places_f.items():

            entropy += probab * np.log2(probab)

        

        entropy = -1 * entropy

        entropy_item[uuid] = entropy

    # If UUID missing

    except:

        entropy_item[uuid] = -1

        # TODO: Assign entropy -1

        

    entropy_list.append(entropy_item)

    

print('Entropy for each UUID:', entropy_list)

# TODO: Low entropy user

uuid = 64

data = pd.DataFrame(hourly_locs[uuid])

data.fillna(-1.0)



# Create heatmap

cmap = mpl.cm.YlOrRd

fig, ax = plt.subplots(figsize=(1.7*width, height))

ax.set_title('Low Entropy User')

sns.heatmap(data, cmap=cmap, ax=ax)



# TODO: High entropy user

uuid = 20

data = pd.DataFrame(hourly_locs[uuid])

data.fillna(-1.0)

# Create heatmap

cmap = mpl.cm.YlOrRd

fig, ax = plt.subplots(figsize=(1.7*width, height))

ax.set_title('High Entropy User')

sns.heatmap(data, cmap=cmap, ax=ax)
ids_len = len(sbj_ids)

friends_matrix = np.empty((ids_len, ids_len))



# Create adjacency matrix for surveyed friendships

for i, sid in enumerate(sbj_ids):

    friends_matrix[i] = [friends_survey[sid][sid2] for sid2 in sbj_ids]



# Delete self loops

np.fill_diagonal(friends_matrix, 0)

    

# Create graph from adjacency matrix

g = ig.Graph.Adjacency(list(friends_matrix), mode=ig.ADJ_UNDIRECTED)



visual_style = {}

visual_style['vertex_size'] = 8

visual_style['bbox'] = (500,500)

visual_style['margin'] = 20



ig.plot(g, **visual_style)
gmm_accuracy = dict()

predictions_dict = dict()

ids_len = len(sbj_ids)-1

gmm_friends_matrix = np.empty((ids_len, ids_len))



# Loop over all subject IDs

for sid in list(feature_data.keys()):

    # Extract the features for current subject

    feature_table = feature_data[sid]

    x = feature_table.ix[:,:'callevent'].values

        

    # Fitting of the GMM to the features

    # TODO: Vary the number of components

    model = GaussianMixture(n_components=2, max_iter=500)

    model.fit(x)

        

    # Prediction of friendships between current subject and all others

    gmm_pred = model.predict(x)

        

    # Labels (0 or 1) are randomly assigned to 'friends' and 'not friends' but most common intuitively is the latter

    not_friend = Counter(gmm_pred).most_common()[0][0]

        

    # Create dict for predicted friendships

    predicted = pd.DataFrame({'subject': list(feature_table.index), 'isfriendP': [int(label != not_friend) for label in gmm_pred]}).set_index('subject')

    predictions_dict[sid] = predicted



    # Evaluate which predictions match the actual friendship

    acc = feature_table.assign(isfriendP = predicted['isfriendP']).pipe(lambda df: df.isfriend == df.isfriendP)

    

    # TODO: Compute accuracy

    

    counter = 0

    for val in acc:

        if val == True:

            counter += 1

        

    gmm_accuracy[sid] = counter / len(acc)





# Show accuracy for all users

print('Overall accuracy: {:.4f} +/- {:.4f}'.format(pd.Series(gmm_accuracy).mean(), pd.Series(gmm_accuracy).std()))



# Create inferred network (adjacency matrix)

for i, sid in enumerate(sbj_ids):

    try:

        gmm_friends_matrix[i-1] = predictions_dict[sid].isfriendP.values

    except:

        gmm_friends_matrix[i-1] = [0] * ids_len



# Delete self loops

np.fill_diagonal(gmm_friends_matrix, 0)



# Create graph from adjacency matrix

g = ig.Graph.Adjacency(list(gmm_friends_matrix), mode=ig.ADJ_UNDIRECTED)



ig.plot(g, **visual_style)
# Calculate the class-conditonal density

def conditional_density(x, GMM):

    prob = 0

    

    # Sum over all mixture components

    for k in range(GMM.n_components):

        # Define the 6-dimensional normal distribution for one component using the GMM parameters

        func = multivariate_normal(mean=GMM.means_[k], cov=GMM.covariances_[k], allow_singular=True)

        

        # Evaluate the function at point x, multiply with component's weight

        prob += GMM.weights_[k] * func.pdf(x)

        

    return prob

    



# Classify the 6-dimensional point x by comparing both GMM evaluations

def GMM_classify(x, GMM_fr, GMM_not_fr, prior_fr, prior_not_fr):

    

    prob_fr *= conditional_density(x, GMM_fr)

    prob_not_fr *= conditional_density(x, GMM_not_fr)

    

    if prob_fr > prob_not_fr:

        return 1

    

    return 0





# Split into training and test data by taking random samples the sets

# TODO: Vary the number of feature vectors

training_fr = list(random.sample(friends_set, 25))

training_not_fr = list(random.sample(not_friends_set, 30))

test = list(friends_set - set(training_fr)) + list(random.sample(set(not_friends_set - set(training_not_fr)), 15))



# Compute the relative class frequencies (priors)

num_pairs = len(training_fr) + len(training_not_fr)

prior_fr = len(training_fr) / num_pairs

prior_not_fr = len(training_not_fr) / num_pairs





# Generate GMM for 'friends'

x_fr = np.asarray([tup[0] for tup in training_fr])

GMM_fr = GaussianMixture(n_components=5, max_iter=500).fit(x_fr)



# Generate GMM for 'not friends'

x_not_fr = np.asarray([tup[0] for tup in training_fr])

GMM_not_fr = GaussianMixture(n_components=5, max_iter=500).fit(x_not_fr)



estimates = []

ground_truth = []



for x in test:

    # Classify the pair x as friends (1) or no friends (0)

    label = GMM_classify(x[0], GMM_fr, GMM_not_fr, prior_fr, prior_not_fr)



    estimates.append(label)

    ground_truth.append(x[1])





# Evaluating the classifier

print(estimates)

print(ground_truth)



performance = precision_recall_fscore_support(y_true=ground_truth, y_pred=estimates)

print('\nPrecision:', performance[0])

print('Recall:', performance[1])

print('F1 score:', performance[2])