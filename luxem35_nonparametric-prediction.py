# Importing main libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scipy.stats as st
# Import data



df = pd.read_csv('../input/911.csv', low_memory=False, nrows=1000)

df = df[['lat', 'lng', 'title']]



df.head(10)
from sklearn.datasets.samples_generator import make_blobs





for i in range (0, df.shape[0]):

    df.iloc[i, 2] = df.iloc[i, 2][: df.iloc[i, 2].find(":")]

    

temp = np.unique(df['title'])

types = dict()

for type_id in range(0, len(temp)):

    types[temp[type_id]] = type_id



del temp



y = df.iloc[:, 2]

y = y.replace(types)



n_components = len(types)

X, truth = make_blobs(n_samples=df.shape[0], centers=n_components)

plt.scatter(df.iloc[:, 0], df.iloc[:, 1], s=50, c = y)

plt.title(f"Example of a mixture of {n_components} distributions")

plt.xlabel("Latitude")

plt.ylabel("Longtitude");
def naive_estimator_evidence(neighbourhood, point, dataset):

    '''

        @arg neighbourhood: float

        @arg point: Array

        @arg dataset: pandas.DataFrame

        

        returns float

        

        neighbourhood:

            Radius around the given point, given as a float. Too small of a value may cause evidence value to be 0.

        point:

            Point of interest.

        dataset:

            Available dataset to calculate the probability of @point occuring.



    '''

    application_matrix = (point[0] - neighbourhood / 2) < dataset['lat']

    application_matrix *= (point[0] + neighbourhood / 2) >= dataset['lat']

    application_matrix *= (point[1] - neighbourhood / 2) < dataset['lng']

    application_matrix *= (point[1] + neighbourhood / 2) >= dataset['lng']

    

    return ((sum(application_matrix) / dataset.shape[0]))
def naive_estimator_likelihood(neighbourhood, point, dataset, emergency):

    '''

        @arg neighbourhood: float

        @arg point: array

        @arg dataset: pandas.DataFrame

        @arg emergency: String

        

        @returns float

                neighbourhood:

            Radius around the given point, given as a float. Too small of a value may cause evidence value to be 0.

        point:

            Point of interest.

        dataset:

            Available dataset to calculate the probability of @point occuring.

        emergency:

            Type of emergency occurence.

    '''

    emergency_set = dataset[dataset['title'] == emergency]

    likelihood = naive_estimator_evidence(neighbourhood, point, emergency_set)

    

    return (likelihood)
def prior(dataset, emergency):

    '''

        @arg dataset: pandas.DataFrame

        @arg emergency: String

        

        @returns float

        

        dataset:

            Available dataset to calculate the probability of @point occuring.

        emergency:

            Type of emergency occurence.

    '''

    emergency_set = dataset[dataset['title'] == emergency]

    prior = emergency_set.shape[0] / dataset.shape[0]

    

    return (prior)
def normalized_posterior(prior, likelihood, evidence):

    '''

        @arg prior: float

        @arg likelihood: float

        @arg evidence: float, evidence != 0

        

        @returns float

        

        prior:

            Prior probability value of the posterior to be calculated.

        likelihood:

            Likelihood value of the posterior to be calculated.

        evidence:

            Evidence value of the posterior to be calculated.

    '''

    

    posterior = prior * likelihood / evidence

    return(posterior)
def naive_estimator_posterior(neighbourhood, point, dataset, emergency):

    '''

        @arg neighbourhood: float

        @arg point: Array

        @arg dataset: pandas.DataFrame

        @arg emergency: String

        

        @return float

        

        neighbourhood:

            Radius around the given point, given as a float. Too small of a value may cause evidence value to be 0.

        point:

            Point of interest.

        dataset:

            Available dataset to calculate the probability of @point occuring.

        emergency:

            Type of emergency occurence.

    '''

    

    __evidence = naive_estimator_evidence(neighbourhood, point, df)

    __likelihood = naive_estimator_likelihood(neighbourhood, point, df, emergency)

    __prior = prior(df, emergency)

    __posterior = normalized_posterior(__prior, __likelihood, __evidence)

    

    return __posterior
def naive_estimator_posterior_expandable(neighbourhood, point, dataset, emergency, expantion_rate=0.001):

    '''

        @arg neighbourhood: float

        @arg point: Array

        @arg dataset: pandas.DataFrame

        @arg emergency: String

        @arg expantion_rate: float, default=0.001

        

        returns Array[float, float]

        

        neighbourhood:

            Radius around the given point, given as a float. Too small of a value may cause evidence value to be 0.

        point:

            Point of interest.

        dataset:

            Available dataset to calculate the probability of @point occuring.

        emergency:

            Type of emergency occurence.

        expantion_rate:

            In case of not finding any available data points inside the given area, function expands its area

            by the expantion_rate.

    '''

    

    lat_boundaries = [dataset['lat'].min(), dataset['lat'].max()]

    lng_boundaries = [dataset['lng'].min(), dataset['lng'].max()]

    

    __evidence = naive_estimator_evidence(neighbourhood, point, df)

    

    while (__evidence == 0):

        if (lat_boundaries[0] > (point[0] - neighbourhood) and lat_boundaries[1] <= (point[0] + neighbourhood) and lng_boundaries[0] > (point[1] - neighbourhood) and lng_boundaries[1] <= (point[1] + neighbourhood)):

            return [neighbourhood, 0]

        

        neighbourhood += expantion_rate

        __evidence = naive_estimator_evidence(neighbourhood, point, df)

    

    __likelihood = naive_estimator_likelihood(neighbourhood, point, df, emergency)

    __prior = prior(df, emergency)

    __posterior = normalized_posterior(__prior, __likelihood, __evidence)

    

    while (__posterior == 0):

        if (lat_boundaries[0] > (point[0] - neighbourhood) and lat_boundaries[1] <= (point[0] + neighbourhood) and lng_boundaries[0] > (point[1] - neighbourhood) and lng_boundaries[1] <= (point[1] + neighbourhood)):

            return [neighbourhood, 0]

        

        neighbourhood += expantion_rate

        __evidence = naive_estimator_evidence(neighbourhood, point, df)

        __likelihood = naive_estimator_likelihood(neighbourhood, point, df, emergency)

        __posterior = normalized_posterior(__prior, __likelihood, __evidence)



    

    return [__posterior, neighbourhood]
import math



def gaussian_kernel(x):

    result = (1/math.sqrt(2 * math.pi)) * math.exp(-(x**2)/2)

    return (result)
def euclidian_dist(x, y):

    '''

        @arg x: pandas.Series

        @arg y: pandas.Series

        

        @returns float

    '''

    result = math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

    return (result)
def kernel_estimator_evidence(neighbourhood, point, dataset):

    '''

        @arg neighbourhood: float

        @arg point: pandas.Series

        @arg dataset: pandas.DataFrame

        

        @return float

        

        neighbourhood:

            Radius around the given point, given as a float. Too small of a value may cause evidence value to be 0.

        point:

            Point of interest.

        dataset:

            Available dataset to calculate the probability of @point occuring.

    '''

    application_matrix = (point[0] - neighbourhood / 2) < dataset['lat']

    application_matrix *= (point[0] + neighbourhood / 2) >= dataset['lat']

    application_matrix *= (point[1] - neighbourhood / 2) < dataset['lng']

    application_matrix *= (point[1] + neighbourhood / 2) >= dataset['lng']

    

    neighbors = dataset[application_matrix]

    evidence = 0

    

    for i in range (0, neighbors.shape[0]):

        item = neighbors.iloc[i, :2]

        distance = euclidian_dist(point, item)

        gaussian_weight = gaussian_kernel(distance)

        evidence += gaussian_weight

    return evidence / dataset.shape[0]
def kernel_estimator_likelihood(neighbourhood, point, dataset, emergency):

    '''

        @arg neighbourhood: float

        @arg point: pandas.Series

        @arg dataset: pandas.DataFrame

        @arg emergency: String

        

        @return float

        

        neighbourhood:

            Radius around the given point, given as a float. Too small of a value may cause evidence value to be 0.

        point:

            Point of interest.

        dataset:

            Available dataset to calculate the probability of @point occuring.

        emergency:

            Type of emergency occurence.

    '''

    emergency_set = dataset[dataset['title'] == emergency]

    likelihood = kernel_estimator_evidence(neighbourhood, point, emergency_set)

    

    return (likelihood)
def kernel_estimator_posterior_expandable(neighbourhood, point, dataset, emergency, expantion_rate=0.001):

    '''

        @arg neighbourhood: float

        @arg point: pandas.Series

        @arg dataset: pandas.DataFrame

        @arg emergency: String

        @arg expantion_rate: float

        

        @return float

        

        neighbourhood:

            Radius around the given point, given as a float. Too small of a value may cause evidence value to be 0.

        point:

            Point of interest.

        dataset:

            Available dataset to calculate the probability of @point occuring.

        emergency:

            Type of emergency occurence.

        expantion_rate:

            In case of not finding any available data points inside the given area, function expands its area

            by the expantion_rate.

    '''

        

    lat_boundaries = [dataset['lat'].min(), dataset['lat'].max()]

    lng_boundaries = [dataset['lng'].min(), dataset['lng'].max()]

    

    __evidence = kernel_estimator_evidence(neighbourhood, point, df)

    

    while (__evidence == 0):

        if (lat_boundaries[0] > (point[0] - neighbourhood) and lat_boundaries[1] <= (point[0] + neighbourhood) and lng_boundaries[0] > (point[1] - neighbourhood) and lng_boundaries[1] <= (point[1] + neighbourhood)):

            return [neighbourhood, 0]

        

        neighbourhood += expantion_rate

        __evidence = kernel_estimator_evidence(neighbourhood, point, df)

    

    __likelihood = kernel_estimator_likelihood(neighbourhood, point, df, emergency)

    __prior = prior(df, emergency)

    __posterior = normalized_posterior(__prior, __likelihood, __evidence)

    

    while (__posterior == 0):

        if (lat_boundaries[0] > (point[0] - neighbourhood) and lat_boundaries[1] <= (point[0] + neighbourhood) and lng_boundaries[0] > (point[1] - neighbourhood) and lng_boundaries[1] <= (point[1] + neighbourhood)):

            return [neighbourhood, 0]

        

        neighbourhood += expantion_rate

        __evidence = kernel_estimator_evidence(neighbourhood, point, df)

        __likelihood = kernel_estimator_likelihood(neighbourhood, point, df, emergency)

        __posterior = normalized_posterior(__prior, __likelihood, __evidence)



    

    return [__posterior, neighbourhood]
from sklearn.model_selection import train_test_split

X = df.iloc[:, :2]

y = df.iloc[:, 2]

train, test = train_test_split(df, test_size=0.2, random_state=21)



X_test = test.iloc[:, :2]

y_test = test.iloc[:, 2]
# Calculate accuracy for Naive Estimator



result = []

temp = np.unique(df['title'])



for i in range(0, X_test.shape[0]):

    best_posterior = 0

    best_class = ''

    for title in temp:

        posterior = naive_estimator_posterior(0.001, X_test.iloc[i, :], train, title)

        

        if(posterior > best_posterior):

            best_posterior = posterior

            best_class = title

    result.append(best_class)

    

print("Accuracy for naive_estimator_posterior with 0.001 neighbour: ", sum(result == y_test) / len(result) * 100)

                                              
# Calculate accuracy for Extended Naive Estimator



result = []

temp = np.unique(df['title'])



for i in range(0, X_test.shape[0]):

    best_posterior = 0

    best_class = ''

    for title in temp:

        posterior = naive_estimator_posterior_expandable(0.001, X_test.iloc[i, :], train, title, expantion_rate=0.001)

        

        if(posterior[0] > best_posterior):

            best_posterior = posterior[0]

            best_class = title

    result.append(best_class)

    

print("Accuracy for naive_estimator_posterior with 0.001 neighbour: ", sum(result == y_test) / len(result) * 100)

                                              
# Calculate accuracy for Gaussian Kernel Estimator



result = []

temp = np.unique(df['title'])



for i in range(0, X_test.shape[0]):

    best_posterior = 0

    best_class = ''

    for title in temp:

        posterior = kernel_estimator_posterior_expandable(0.001, X_test.iloc[i, :], train, title, expantion_rate=0.001)

        

        if(posterior[0] > best_posterior):

            best_posterior = posterior[0]

            best_class = title



    result.append(best_class)

    

print("Accuracy for naive_estimator_posterior with 0.001 neighbour: ", sum(result == y_test) / len(result) * 100)

                                              