# Code from the my kernel "Titanic Top 3% : one line of the prediction code": 

# https://www.kaggle.com/vbmokin/titanic-top-3-one-line-of-the-prediction-code

#

import pandas as pd

import numpy as np 

traindf = pd.read_csv('../input/titanic/train.csv').set_index('PassengerId')

testdf = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')

df = pd.concat([traindf, testdf], axis=0, sort=False)

df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

df['IsWomanOrBoy'] = ((df.Title == 'Master') | (df.Sex == 'female'))

df['LastName'] = df.Name.str.split(',').str[0]

family = df.groupby(df.LastName).Survived

df['WomanOrBoyCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).count())

df['WomanOrBoyCount'] = df.mask(df.IsWomanOrBoy, df.WomanOrBoyCount - 1, axis=0)

df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).sum())

df['FamilySurvivedCount'] = df.mask(df.IsWomanOrBoy, df.FamilySurvivedCount - df.Survived.fillna(0), axis=0)

df['WomanOrBoySurvived'] = df.FamilySurvivedCount / df.WomanOrBoyCount.replace(0, np.nan)

df.WomanOrBoyCount = df.WomanOrBoyCount.replace(np.nan, 0)

df['Alone'] = (df.WomanOrBoyCount == 0)

train_y = df.Survived.loc[traindf.index]

df2 = pd.concat([df.WomanOrBoySurvived.fillna(0), df.Alone, df.Sex.replace({'female': 0, 'male': 1})], axis=1)

test_x = df2.loc[testdf.index]



# The one line of the code for prediction : LB = 0.83253 (Titanic Top 3%) 

y_pred_top3 = (((test_x.WomanOrBoySurvived <= 0.238) & (test_x.Sex < 0.5) & (test_x.Alone > 0.5)) | \

          ((test_x.WomanOrBoySurvived > 0.238) & \

           ~((test_x.WomanOrBoySurvived > 0.55) & (test_x.WomanOrBoySurvived <= 0.633)))).astype(int)



# Saving the result

pd.DataFrame({'Survived': y_pred_top3}, \

             index=testdf.index).reset_index().to_csv('survived_top3.csv', index=False)

print('Mean =', y_pred_top3.mean(), ' Std =', y_pred_top3.std())
from sklearn import cluster, datasets, mixture

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.neighbors import kneighbors_graph

from itertools import cycle, islice

from scipy.spatial.distance import cosine



import time

import graphviz

import matplotlib.pyplot as plt

print(__doc__)



import warnings

warnings.filterwarnings("ignore")





np.random.seed(0)
# FE
#Thanks to: https://www.kaggle.com/kpacocha/top-6-titanic-machine-learning-from-disaster

#Title

df['Title'] = df['Title'].replace('Ms','Miss')

df['Title'] = df['Title'].replace('Mlle','Miss')

df['Title'] = df['Title'].replace('Mme','Mrs')
#Thanks to: https://www.kaggle.com/kpacocha/top-6-titanic-machine-learning-from-disaster

# Embarked

df['Embarked'] = df['Embarked'].fillna('S')
# Thanks to: https://www.kaggle.com/erinsweet/simpledetect

# Fare

med_fare = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]

df['Fare'] = df['Fare'].fillna(med_fare)
#Thanks to: https://www.kaggle.com/kpacocha/top-6-titanic-machine-learning-from-disaster

# Cabin, Deck

df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

df.loc[(df['Deck'] == 'T'), 'Deck'] = 'A'
#Thanks to: https://www.kaggle.com/erinsweet/simpledetect

#Age

df['Age'] = df.groupby(['Sex', 'Pclass', 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))
#Thanks to: https://www.kaggle.com/erinsweet/simpledetect

# Family_Size

df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
pd.set_option('max_columns',100)

traindf.head(3)
df.WomanOrBoySurvived = df.WomanOrBoySurvived.fillna(0)

df.WomanOrBoyCount = df.WomanOrBoyCount.fillna(0)

df.FamilySurvivedCount = df.FamilySurvivedCount.fillna(0)

df.Alone = df.Alone.fillna(0)
df.head(3)
train_y = df.Survived.loc[traindf.index]
cols_to_drop = ['Name','Ticket','Cabin','Survived']

df = df.drop(cols_to_drop, axis=1)
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

categorical_columns = []

features = df.columns.values.tolist()

for col in features:

    if df[col].dtype in numerics: continue

    categorical_columns.append(col)

categorical_columns
for col in categorical_columns:

    if col in df.columns:

        le = LabelEncoder()

        le.fit(list(df[col].astype(str).values))

        df[col] = le.transform(list(df[col].astype(str).values))
train_x_all, test_x_all = df.loc[traindf.index], df.loc[testdf.index]

train_x_all.head(3)
# The minimal percentage of similarity of the clustered feature with "Survived" for inclusion in the final dataset

limit_opt = 0.7
n_clusters_opt = 4 # number of clusters

# Thanks to: https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py

default_base = {'quantile': .2,

                'eps': .3,

                'damping': .9,

                'preference': -200,

                'n_neighbors': 10,

                'n_clusters': n_clusters_opt,

                'min_samples': 3,

                'xi': 0.05,

                'min_cluster_size': 0.05}
# Features list for clustering

feature_first = 'WomanOrBoySurvived'

clustered_features = ['Pclass', 'Age', 'Deck', 'Family_Size', 'Title', 'Embarked', 'Fare']
def generate_data(x1,x2,df,t):

    # x1, x2 as string - name of features from dataframe df

    # t=1 - with train_y, t=0 - without its

    X = pd.concat([df[x1], df[x2]], axis=1).values

    if t==1:

        y = train_y.values.astype(int)

        return (X, y)

    else:

        return X
title_plot = {}

for i in range(len(clustered_features)):

    title_plot[i] = 'W-'+str(clustered_features[i])

title_plot
# train dataset

datasets = []

for i in range(len(clustered_features)):

    datasets.append((generate_data(feature_first,clustered_features[i],train_x_all,1),{}))
# test dataset

datasets_test = []

rez = pd.DataFrame(index = test_x.index)

for i in range(len(clustered_features)):

    datasets_test.append(generate_data(feature_first,clustered_features[i],test_x_all,0))
# Thanks to: https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py

def generate_clustering_algorithms(Z,n_clusters):

    # generate clustering algorithms

    

    # estimate bandwidth for mean shift

    bandwidth = cluster.estimate_bandwidth(df, quantile=params['quantile'])



    # connectivity matrix for structured Ward

    connectivity = kneighbors_graph(

        Z, n_neighbors=params['n_neighbors'], include_self=False)

    

    # make connectivity symmetric

    connectivity = 0.5 * (connectivity + connectivity.T)



    # ============

    # Create cluster objects

    # ============

    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state = 1000)

    two_means = cluster.MiniBatchKMeans(n_clusters=n_clusters)

    ward = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',

        connectivity=connectivity)

    spectral = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack',

        affinity="nearest_neighbors")

    dbscan = cluster.DBSCAN(eps=params['eps'])

    optics = cluster.OPTICS(min_samples=params['min_samples'],

                            xi=params['xi'],

                            min_cluster_size=params['min_cluster_size'])

    affinity_propagation = cluster.AffinityPropagation(damping=params['damping'])

    average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock",

        n_clusters=params['n_clusters'], connectivity=connectivity)

    birch = cluster.Birch(n_clusters=params['n_clusters'])

    gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full')



    clustering_algorithms = (

        ('MiniBatchKMeans', two_means),

        ('KMeans', kmeans),

        ('AffinityPropagation', affinity_propagation),

        ('MeanShift', ms),

        ('SpectralClustering', spectral),

        ('Ward', ward),

        ('AgglomerativeClustering', average_linkage),

        ('DBSCAN', dbscan),

        ('OPTICS', optics),

        ('Birch', birch),

        ('GaussianMixture', gmm)

    )

    return clustering_algorithms
# Thanks to: https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py

# ============

# Set up cluster parameters

# ============

plt.figure(figsize=(10 * 2 + 2, 12.5))

plt.subplots_adjust(left=.02, right=.98, bottom=.01, top=.98, wspace=.05,

                    hspace=.01)



plot_num = 1

coord_xy_lim = 2.5

for i_dataset, (dataset, algo_params) in enumerate(datasets):

    # update parameters with dataset-specific values

    params = default_base.copy()

    params.update(algo_params)



    X, y = dataset

    

    # normalize dataset for easier parameter selection

    X = StandardScaler().fit_transform(X)

    datasets_test[i_dataset] = StandardScaler().fit_transform(datasets_test[i_dataset])        



    clustering_algorithms = generate_clustering_algorithms(X,params['n_clusters'])

    clustering_algorithms_test = generate_clustering_algorithms(datasets_test[i_dataset],params['n_clusters'])



    simil = {}

    i = 0



    for name, algorithm in clustering_algorithms:

        t0 = time.time()



        # catch warnings related to kneighbors_graph

        with warnings.catch_warnings():

            warnings.filterwarnings(

                "ignore",

                message="the number of connected components of the " +

                "connectivity matrix is [0-9]{1,2}" +

                " > 1. Completing it to avoid stopping the tree early.",

                category=UserWarning)

            warnings.filterwarnings(

                "ignore",

                message="Graph is not fully connected, spectral embedding" +

                " may not work as expected.",

                category=UserWarning)

            algorithm.fit(X)



        t1 = time.time()

        

        if hasattr(algorithm, 'labels_'):

            y_pred = algorithm.labels_.astype(np.int)

        else:

            y_pred = algorithm.predict(X)       

        

        simil[name] = 1 - cosine(y, y_pred)

        print(i_dataset, i, round(simil[name], 3), title_plot[i_dataset], name)

        

        if i == len(clustering_algorithms)-1:

            # determine the optimal clustering method 

            max_simil = max(simil, key=simil.get)

            print('Optimal ==> ', max_simil)

           

            # clustering data by the optimal method - synthesis of a new feature

            if simil[max_simil] > limit_opt:

                train_x_all[title_plot[i_dataset]] = y_pred

                algorithm_opt = dict(clustering_algorithms_test)[max_simil]

                algorithm_opt.fit(datasets_test[i_dataset])

                if hasattr(algorithm_opt, 'labels_'):

                    rez[title_plot[i_dataset]] = algorithm_opt.labels_.astype(np.int)

                else:

                    rez[title_plot[i_dataset]] = algorithm_opt.predict(datasets_test[i_dataset])          



        plt.subplot(len(datasets), len(clustering_algorithms) + 1, plot_num)

        if i_dataset == 0:

            plt.title(name, size=14)

        if name == 'MiniBatchKMeans':

            yt = plt.ylabel(title_plot[i_dataset], size=14,rotation=90)

            

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',

                                             '#f781bf', '#a65628', '#984ea3',

                                             '#999999', '#e41a1c', '#dede00']),

                                      int(max(y_pred) + 1))))

        # add black color for outliers (if any)

        colors = np.append(colors, ["#000000"])

        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])



        plt.xlim(-coord_xy_lim, coord_xy_lim)

        plt.ylim(-coord_xy_lim, coord_xy_lim)

        plt.xticks(())

        plt.yticks(())

        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),

                 transform=plt.gca().transAxes, size=15,

                 horizontalalignment='right')

        plot_num += 1

        i += 1

    # Survived

    plt.subplot(len(datasets), len(clustering_algorithms)+1, plot_num)

    if i_dataset == 0:

        plt.title("Survived", size=14)

    plt.scatter(X[:, 0], X[:, 1], c=y)

    plt.xlim(-coord_xy_lim, coord_xy_lim)

    plt.ylim(-coord_xy_lim, coord_xy_lim)

    plt.xticks(())

    plt.yticks(())

    plot_num += 1



plt.show()
rez.head(3)
train_x = pd.concat([train_x_all.WomanOrBoySurvived.fillna(0), 

                     train_x_all.Alone, 

                     train_x_all.Sex,

                     ], axis=1)

test_x = pd.concat([test_x_all.WomanOrBoySurvived.fillna(0), 

                     test_x_all.Alone, 

                     test_x_all.Sex,

                     ], axis=1)
rez_col = rez.columns.values.tolist()

rez_col
train_x = train_x.join(train_x_all[rez_col], how='left', lsuffix="_rez")

train_x.head(2)
test_x = test_x.join(rez[rez_col], how='left', lsuffix="_rez")

test_x.head(2)
# Tuning the DecisionTreeClassifier by the GridSearchCV

parameters = {'max_depth' : np.arange(2, 9, dtype=int),

              'min_samples_leaf' :  np.arange(1, 4, dtype=int)}

classifier = DecisionTreeClassifier(random_state=1000)

model = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)

model.fit(train_x, train_y)

best_parameters = model.best_params_

print(best_parameters)
model=DecisionTreeClassifier(max_depth = best_parameters['max_depth'], 

                             random_state = 1118)

model.fit(train_x, train_y)
# plot tree

dot_data = export_graphviz(model, out_file=None, feature_names=train_x.columns, class_names=['0', '1'], 

                           filled=True, rounded=False,special_characters=True, precision=7) 

graph = graphviz.Source(dot_data)

graph 
# Prediction by the DecisionTreeClassifier

y_pred = model.predict(test_x).astype(int)

print('Mean =', y_pred.mean(), ' Std =', y_pred.std())

# Mean = 0.3349282296650718  Std = 0.4719653701687156 ==> LB = 0.83253
train_y_pred = model.predict(train_x).astype(int)

diff = sum(abs(train_y-train_y_pred))*100/len(train_y)

diff

# LB = 0.83253 ==> 7.744107744107744 
# Saving the result

pd.DataFrame({'Survived': y_pred}, index=testdf.index).reset_index().to_csv('survived_new.csv', index=False)
train_x_all['Pred'] = train_y_pred

train_x_all['Survived'] = train_y
pd.set_option('max_columns',100)

pd.set_option('max_rows',100)
train_x_all[train_x_all['Survived'] != train_x_all['Pred']].sort_values(by=['Survived'])
diff_nrow = len(train_x_all[train_x_all['Survived'] != train_x_all['Pred']])

diff_nrow

# LB = 0.83253 ==> 49