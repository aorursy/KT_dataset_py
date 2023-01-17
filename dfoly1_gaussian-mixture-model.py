# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import multivariate_normal as mvn

import math



from numpy.linalg import slogdet, det, solve

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import time



from sklearn.datasets import load_digits

from sklearn.mixture.base import BaseMixture

from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA

from sklearn.preprocessing import normalize



import os, sys, email,re

from nltk.corpus import stopwords

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
samples = np.load('../input/samples/samples.npz')

X = samples['data']

pi0 = samples['pi0']

mu0 = samples['mu0']

sigma0 = samples['sigma0']

plt.scatter(X[:, 0], X[:, 1], c='grey', s=30)

plt.axis('equal')

plt.show()

print(pi0, mu0)
kmeans_test = KMeans(n_clusters= 3, init="k-means++", max_iter=500, algorithm = 'auto')

fitted = kmeans_test.fit(X)

prediction = kmeans_test.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=prediction, s=30)

plt.axis('equal')

plt.show()



kmeans_test.cluster_centers_
plt.figure(figsize = (10,8))

from scipy.spatial.distance import cdist

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):

    labels = kmeans.fit_predict(X)



    # plot the input data

    ax = ax or plt.gca()

    ax.axis('equal')

    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)



    # plot the representation of the KMeans model

    centers = kmeans.cluster_centers_

    radii = [cdist(X[labels == i], [center]).max()

             for i, center in enumerate(centers)]

    for c, r in zip(centers, radii):

        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))

        

plot_kmeans(kmeans_test, X)
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3).fit(X)

labels = gmm.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');

probs = gmm.predict_proba(X)

# print(probs[:5].round(3))
# print(gmm._get_parameters()[0])

# print(gmm._get_parameters()[1])

#gmm.lower_bound_

# kmeans mean values

# array([[1.20514208, 5.82885457],

#        [0.9733235 , 0.94371856],

#        [6.40546402, 4.50975916]])
def calculate_mean_covariance(X, prediction):

    C = 3

    d = X.shape[1]

    labels = np.unique(prediction)

    initial_means = np.zeros((C, d))

    initial_cov = np.zeros((C, d, d))

    initial_pi = np.zeros(C)

        

    counter=0

    for label in sorted(labels):

        ids = np.where(prediction == label) # returns indices

        initial_pi[counter] = len(ids[0]) / X.shape[0] 

        initial_means[counter,:] = np.mean(X[ids], axis = 0)

        de_meaned = X[ids] - initial_means[counter,:]

        Nk = X[ids].shape[0]

        initial_cov[counter,:, :] = np.dot(initial_pi[counter] * de_meaned.T, de_meaned) / Nk

        counter+=1

    assert np.sum(initial_pi) == 1

    return (initial_means, initial_cov, initial_pi)

    

n_clusters = 3

kmeans = KMeans(n_clusters= n_clusters, max_iter=500, algorithm = 'auto')

fitted = kmeans.fit(X)

prediction = kmeans.predict(X)



m, c, pi = calculate_mean_covariance(X, prediction)
class GMM:

    """ Gaussian Mixture Model

    

    Parameters

    -----------

        k: int , number of gaussian distributions

        

        seed: int, will be randomly set if None

        

        max_iter: int, number of iterations to run algorithm, default: 200

        

    Attributes

    -----------

       centroids: array, k, number_features

       

       cluster_labels: label for each data point

       

    """

    def __init__(self, C, n_runs):

        self.C = C # number of Guassians/clusters

        self.n_runs = n_runs

        

    

    def get_params(self):

        return (self.mu, self.pi, self.sigma)

    

    

        

    def calculate_mean_covariance(self, X, prediction):

        """Calculate means and covariance of different

            clusters from k-means prediction

        

        Parameters:

        ------------

        prediction: cluster labels from k-means

        

        X: N*d numpy array data points 

        

        Returns:

        -------------

        intial_means: for E-step of EM algorithm

        

        intial_cov: for E-step of EM algorithm

        

        """

        d = X.shape[1]

        labels = np.unique(prediction)

        self.initial_means = np.zeros((self.C, d))

        self.initial_cov = np.zeros((self.C, d, d))

        self.initial_pi = np.zeros(self.C)

        

        counter=0

        for label in labels:

            ids = np.where(prediction == label) # returns indices

            self.initial_pi[counter] = len(ids[0]) / X.shape[0]

            self.initial_means[counter,:] = np.mean(X[ids], axis = 0)

            de_meaned = X[ids] - self.initial_means[counter,:]

            Nk = X[ids].shape[0] # number of data points in current gaussian

            self.initial_cov[counter,:, :] = np.dot(self.initial_pi[counter] * de_meaned.T, de_meaned) / Nk

            counter+=1

        assert np.sum(self.initial_pi) == 1    

            

        return (self.initial_means, self.initial_cov, self.initial_pi)

    

    

    

    def _initialise_parameters(self, X):

        """Implement k-means to find starting

            parameter values.

            https://datascience.stackexchange.com/questions/11487/how-do-i-obtain-the-weight-and-variance-of-a-k-means-cluster



        Parameters:

        ------------

        X: numpy array of data points

        

        Returns:

        ----------

        tuple containing initial means and covariance

        

        _initial_means: numpy array: (C*d)

        

        _initial_cov: numpy array: (C,d*d)

        

        

        """

        n_clusters = self.C

        kmeans = KMeans(n_clusters= n_clusters, init="k-means++", max_iter=500, algorithm = 'auto')

        fitted = kmeans.fit(X)

        prediction = kmeans.predict(X)

        self._initial_means, self._initial_cov, self._initial_pi = self.calculate_mean_covariance(X, prediction)

        

        

        return (self._initial_means, self._initial_cov, self._initial_pi)

            

        

        

    def _e_step(self, X, pi, mu, sigma):

        """Performs E-step on GMM model



        Parameters:

        ------------

        X: (N x d), data points, m: no of features

        pi: (C), weights of mixture components

        mu: (C x d), mixture component means

        sigma: (C x d x d), mixture component covariance matrices



        Returns:

        ----------

        gamma: (N x C), probabilities of clusters for objects

        """

        N = X.shape[0] 

        self.gamma = np.zeros((N, self.C))



        const_c = np.zeros(self.C)

        

        

        self.mu = self.mu if self._initial_means is None else self._initial_means

        self.pi = self.pi if self._initial_pi is None else self._initial_pi

        self.sigma = self.sigma if self._initial_cov is None else self._initial_cov



        for c in range(self.C):

            # Posterior Distribution using Bayes Rule

            self.gamma[:,c] = self.pi[c] * mvn.pdf(X, self.mu[c,:], self.sigma[c])



        # normalize across columns to make a valid probability

        gamma_norm = np.sum(self.gamma, axis=1)[:,np.newaxis]

        self.gamma /= gamma_norm



        return self.gamma

    

    

    def _m_step(self, X, gamma):

        """Performs M-step of the GMM

        We need to update our priors, our means

        and our covariance matrix.



        Parameters:

        -----------

        X: (N x d), data 

        gamma: (N x C), posterior distribution of lower bound 



        Returns:

        ---------

        pi: (C)

        mu: (C x d)

        sigma: (C x d x d)

        """

        N = X.shape[0] # number of objects

        C = self.gamma.shape[1] # number of clusters

        d = X.shape[1] # dimension of each object



        # responsibilities for each gaussian

        self.pi = np.mean(self.gamma, axis = 0)



        self.mu = np.dot(self.gamma.T, X) / np.sum(self.gamma, axis = 0)[:,np.newaxis]



        for c in range(C):

            x = X - self.mu[c, :] # (N x d)

            

            gamma_diag = np.diag(self.gamma[:,c])

            x_mu = np.matrix(x)

            gamma_diag = np.matrix(gamma_diag)



            sigma_c = x.T * gamma_diag * x

            self.sigma[c,:,:]=(sigma_c) / np.sum(self.gamma, axis = 0)[:,np.newaxis][c]



        return self.pi, self.mu, self.sigma

    

    

    def _compute_loss_function(self, X, pi, mu, sigma):

        """Computes lower bound loss function

        

        Parameters:

        -----------

        X: (N x d), data 

        

        Returns:

        ---------

        pi: (C)

        mu: (C x d)

        sigma: (C x d x d)

        """

        N = X.shape[0]

        C = self.gamma.shape[1]

        self.loss = np.zeros((N, C))



        for c in range(C):

            dist = mvn(self.mu[c], self.sigma[c],allow_singular=True)

            self.loss[:,c] = self.gamma[:,c] * (np.log(self.pi[c]+0.00001)+dist.logpdf(X)-np.log(self.gamma[:,c]+0.000001))

        self.loss = np.sum(self.loss)

        return self.loss

    

    

    

    def fit(self, X):

        """Compute the E-step and M-step and

            Calculates the lowerbound

        

        Parameters:

        -----------

        X: (N x d), data 

        

        Returns:

        ----------

        instance of GMM

        

        """

        

        d = X.shape[1]

        self.mu, self.sigma, self.pi =  self._initialise_parameters(X)

        

        try:

            for run in range(self.n_runs):  

                self.gamma  = self._e_step(X, self.mu, self.pi, self.sigma)

                self.pi, self.mu, self.sigma = self._m_step(X, self.gamma)

                loss = self._compute_loss_function(X, self.pi, self.mu, self.sigma)

                

                if run % 10 == 0:

                    print("Iteration: %d Loss: %0.6f" %(run, loss))



        

        except Exception as e:

            print(e)

            

        

        return self

    

    

    

    

    def predict(self, X):

        """Returns predicted labels using Bayes Rule to

        Calculate the posterior distribution

        

        Parameters:

        -------------

        X: ?*d numpy array

        

        Returns:

        ----------

        labels: predicted cluster based on 

        highest responsibility gamma.

        

        """

        labels = np.zeros((X.shape[0], self.C))

        

        for c in range(self.C):

            labels [:,c] = self.pi[c] * mvn.pdf(X, self.mu[c,:], self.sigma[c])

        labels  = labels .argmax(1)

        return labels 

    

    def predict_proba(self, X):

        """Returns predicted labels

        

        Parameters:

        -------------

        X: N*d numpy array

        

        Returns:

        ----------

        labels: predicted cluster based on 

        highest responsibility gamma.

        

        """

        post_proba = np.zeros((X.shape[0], self.C))

        

        for c in range(self.C):

            # Posterior Distribution using Bayes Rule, try and vectorise

            post_proba[:,c] = self.pi[c] * mvn.pdf(X, self.mu[c,:], self.sigma[c])

    

        return post_proba
model = GMM(3, n_runs = 100)



fitted_values = model.fit(X)



predicted_values = model.predict(X)

# compute centers as point of highest density of distribution

centers = np.zeros((3,2))

for i in range(model.C):

    density = mvn(cov=model.sigma[i], mean=model.mu[i]).logpdf(X)

    centers[i, :] = X[np.argmax(density)]

    

plt.figure(figsize = (10,8))

plt.scatter(X[:, 0], X[:, 1],c=predicted_values ,s=50, cmap='viridis')



plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);

# print("initial means: ", model._initial_means)

# print('--------------------------')

# print("initial pi: ", model._initial_pi)

# print('--------------------------')

# print('GMM means: ', model.get_params()[1])

# print('--------------------------')

# print('GMM pi: ', model.get_params()[0])

# print('--------------------------')

# print('sklearn GMM means: ', gmm._get_parameters()[1])

# print('--------------------------')

# print('sklearn GMM pi: ', gmm._get_parameters()[0])
# Credit to python data science handbook for the code to plot these distributions

from matplotlib.patches import Ellipse



def draw_ellipse(position, covariance, ax=None, **kwargs):

    """Draw an ellipse with a given position and covariance"""

    ax = ax or plt.gca()

    

    # Convert covariance to principal axes

    if covariance.shape == (2, 2):

        U, s, Vt = np.linalg.svd(covariance)

        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))

        width, height = 2 * np.sqrt(s)

    else:

        angle = 0

        width, height = 2 * np.sqrt(covariance)

    

    # Draw the Ellipse

    for nsig in range(1, 4):

        ax.add_patch(Ellipse(position, nsig * width, nsig * height,

                             angle, **kwargs))

        
plt.figure(figsize = (10,8))

plt.scatter(X[:, 0], X[:, 1],c=predicted_values ,s=50, cmap='viridis')

plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);



w_factor = 0.2 / model.pi.max()

for pos, covar, w in zip(model.mu, model.sigma, model.pi):

    draw_ellipse(pos, covar, alpha=w * w_factor)
def E_step(X, pi, mu, sigma):

    """Performs E-step on GMM model



        Parameters:

        ------------

        X: (N x m), data points, m: no of features

        pi: (C), weights of mixture components

        mu: (C x m), mixture component means

        sigma: (C x m x m), mixture component covariance matrices



        Returns:

        ----------

        gamma: (N x C), probabilities of clusters for objects

        """

    

    N = X.shape[0] 

    C = pi0.shape[0] # number of clusters

    gamma = np.zeros((N, C))



    const_c = np.zeros(C)



    for c in range(C):

#         pi_c = pi0[c]

#         mu_c = mu0[c, :] 

#         sigma_c = sigma0[c] 



        # Posterior Distribution using Bayes Rule

        gamma[:,c] = pi0[c] * mvn.pdf(X, mu0[c,:], sigma0[c])



    # normalize across columns to make a valid probability

    gamma_norm = np.sum(gamma, axis=1)[:,np.newaxis]

    gamma /= gamma_norm



    return gamma



gamma = E_step(X, pi0, mu0, sigma0)

print(np.sum(gamma, axis=0))
def M_step(X, gamma):

    """Performs M-step of the GMM

    We need to update our priors, our means

    and our covariance matrix.

    

    Parameters:

    -----------

    X: (N x d), data 

    gamma: (N x C), posterior distribution of lower bound 

    

    Returns:

    ---------

    pi: (C)

    mu: (C x d)

    sigma: (C x d x d)

    """

    N = X.shape[0] # number of objects

    C = gamma.shape[1] # number of clusters

    d = X.shape[1] # dimension of each object



    ### YOUR CODE HERE

    mu = np.zeros((C, d))

    sigma = np.zeros((C, d, d))

    

    # responsibilities for each gaussian

    pi = np.mean(gamma, axis = 0)

    mu = np.dot(gamma.T, X) / np.sum(gamma, axis = 0)[:,np.newaxis]

     

    for c in range(C): # for each Gaussian

        x_mu = X - mu[c, :] # (N x d)

        

        gamma_diag = np.diag(gamma[:,c])

        x_mu = np.matrix(x_mu)

        gamma_diag = np.matrix(gamma_diag)

        

        sigma_c = x_mu.T * gamma_diag * x_mu

        sigma[c,:,:]=(sigma_c) / np.sum(gamma, axis = 0)[:,np.newaxis][c]

    

    return pi, mu, sigma

pi, mu, sigma = M_step(X, gamma)

print(pi, mu, sigma)
def compute_loss_function(X, gamma, pi, mu, sigma):

    N = X.shape[0]

    C = gamma.shape[1]

    loss = np.zeros((N,C))

    

    for c in range(C):

        dist = mvn(mu[c], sigma[c],allow_singular=True)

        loss[:,c] = gamma[:,c] * (np.log(pi[c]+0.00001)+dist.logpdf(X)-np.log(gamma[:,c]+0.000001))



    return np.sum(loss)
compute_loss_function(X, gamma, pi, mu, sigma)
pi, mu, sigma = pi0, mu0, sigma0

gamma = E_step(X, pi, mu, sigma)

pi, mu, sigma = M_step(X, gamma)

loss = compute_loss_function(X, gamma, pi, mu, sigma)
df = pd.read_csv('../input/enron-email-dataset/emails.csv',nrows = 35000)
# create list of email objects

emails = list(map(email.parser.Parser().parsestr,df['message']))



# extract headings such as subject, from, to etc..

headings  = emails[0].keys()



# Goes through each email and grabs info for each key

# doc['From'] grabs who sent email in all emails

for key in headings:

    df[key] = [doc[key] for doc in emails]



    

##Useful functions

def get_raw_text(emails):

    email_text = []

    for email in emails.walk():

        if email.get_content_type() == 'text/plain':

            email_text.append(email.get_payload())

    return ''.join(email_text)



df['body'] = list(map(get_raw_text, emails))

df.head()

df['user'] = df['file'].map(lambda x: x.split('/')[0])





#Unique to and From

print('Total number of emails: %d' %len(df))

print('------------')

print('Number of unique received: %d '%df['To'].nunique())

print('------------')

print('Number of unique Sent: %d '%df['From'].nunique())





def clean_column(data):

    if data is not None:

        stopwords_list = stopwords.words('english')

        #exclusions = ['RE:', 'Re:', 're:']

        #exclusions = '|'.join(exclusions)

        data =  data.lower()

        data = re.sub('re:', '', data)

        data = re.sub('-', '', data)

        data = re.sub('_', '', data)

        # Remove data between square brackets

        data =re.sub('\[[^]]*\]', '', data)

        # removes punctuation

        data = re.sub(r'[^\w\s]','',data)

        data = re.sub(r'\n',' ',data)

        data = re.sub(r'[0-9]+','',data)

        # strip html 

        p = re.compile(r'<.*?>')

        data = re.sub(r"\'ve", " have ", data)

        data = re.sub(r"can't", "cannot ", data)

        data = re.sub(r"n't", " not ", data)

        data = re.sub(r"I'm", "I am", data)

        data = re.sub(r" m ", " am ", data)

        data = re.sub(r"\'re", " are ", data)

        data = re.sub(r"\'d", " would ", data)

        data = re.sub(r"\'ll", " will ", data)

        data = re.sub('forwarded by phillip k allenhouect on    pm', '',data)

        data = re.sub(r"httpitcappscorpenroncomsrrsauthemaillinkaspidpage", "", data)

        

        data = p.sub('', data)

        if 'forwarded by:' in data:

            data = data.split('subject')[1]

        data = data.strip()

        return data

    return 'No Subject'





df['Subject_new'] = df['Subject'].apply(clean_column)

df['body_new'] = df['body'].apply(clean_column)





from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)

to_add = ['FW', 'ga', 'httpitcappscorpenroncomsrrsauthemaillinkaspidpage', 'cc', 'aa', 'aaa', 'aaaa',

         'hou', 'cc', 'etc', 'subject', 'pm']



for i in to_add:

    stopwords.add(i)
df['body_new'].head()
from sklearn.feature_extraction.text import TfidfVectorizer

data = df['body_new']

# data.head()



tf_idf_vectorizor = TfidfVectorizer(stop_words = stopwords,#tokenizer = tokenize_and_stem,

                             max_features = 5000)

%time tf_idf = tf_idf_vectorizor.fit_transform(data)

tf_idf_norm = normalize(tf_idf)

tf_idf_array = tf_idf_norm.toarray()

pd.DataFrame(tf_idf_array, columns=tf_idf_vectorizor.get_feature_names()).head()
from sklearn.cluster import KMeans

n_clusters = 3

sklearn_pca = PCA(n_components = 2)

Y_sklearn = sklearn_pca.fit_transform(tf_idf_array)

kmeans = KMeans(n_clusters= n_clusters, max_iter=600, algorithm = 'auto')

%time fitted = kmeans.fit(Y_sklearn)

prediction = kmeans.predict(Y_sklearn)



plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],c=prediction ,s=50, cmap='viridis')



centers2 = fitted.cluster_centers_

plt.scatter(centers2[:, 0], centers2[:, 1],c='black', s=300, alpha=0.6);
from matplotlib.patches import Ellipse



def draw_ellipse(position, covariance, ax=None, **kwargs):

    """Draw an ellipse with a given position and covariance"""

    ax = ax or plt.gca()

    

    # Convert covariance to principal axes

    if covariance.shape == (2, 2):

        U, s, Vt = np.linalg.svd(covariance)

        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))

        width, height = 2 * np.sqrt(s)

    else:

        angle = 0

        width, height = 2 * np.sqrt(covariance)

    

    # Draw the Ellipse

    for nsig in range(1, 4):

        ax.add_patch(Ellipse(position, nsig * width, nsig * height,

                             angle, **kwargs))
from sklearn.mixture import GaussianMixture

sklearn_pca = PCA(n_components = 2)

Y_sklearn = sklearn_pca.fit_transform(tf_idf_array)

gmm = GaussianMixture(n_components=3, covariance_type='full').fit(Y_sklearn)

prediction_gmm = gmm.predict(Y_sklearn)

probs = gmm.predict_proba(Y_sklearn)



centers = np.zeros((3,2))

for i in range(3):

    density = mvn(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(Y_sklearn)

    centers[i, :] = Y_sklearn[np.argmax(density)]



plt.figure(figsize = (10,8))

plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],c=prediction_gmm ,s=50, cmap='viridis')

plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);
# diag, Gaussians are aligned with th axis

plt.figure(figsize = (10,8))

plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],c=prediction_gmm ,s=50, cmap='viridis', zorder=1)

plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);



w_factor = 0.2 / gmm.weights_.max()

for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):

    draw_ellipse(pos, covar, alpha=w*.75)
print("GMM weights: ", gmm.weights_)

print("GMM means: ", gmm.means_)

print("GMM Covariance: ", gmm.covariances_)

print('-----------------------------')

print("My model weights: ", model.pi)

print("My model means: ", model.mu)

print("My model Covariance: ", model.sigma)
plt.figure(figsize = (10,8))

from scipy.spatial.distance import cdist

def plot_kmeans(kmeans, X, n_clusters=3, rseed=0, ax=None):

    labels = kmeans.fit_predict(X)



    # plot the input data

    ax = ax or plt.gca()

    ax.axis('equal')

    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)



    # plot the representation of the KMeans model

    centers = kmeans.cluster_centers_

    radii = [cdist(X[labels == i], [center]).max()

             for i, center in enumerate(centers)]

    for c, r in zip(centers, radii):

        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))

        

plot_kmeans(kmeans, Y_sklearn)
plt.figure(figsize = (10,8))

plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],c=prediction_gmm ,s=50, cmap='viridis', zorder=1)

plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);



w_factor = 0.2 / gmm.weights_.max()

for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):

    draw_ellipse(pos, covar, alpha=w*.75)
model = GMM(3, n_runs = 50)



fitted_values = model.fit(Y_sklearn)

predicted_values = model.predict(Y_sklearn)



# # compute centers as point of highest density of distribution

centers = np.zeros((3,2))

for i in range(model.C):

    density = mvn(cov=model.sigma[i], mean=model.mu[i]).logpdf(Y_sklearn)

    centers[i, :] = Y_sklearn[np.argmax(density)]

    

plt.figure(figsize = (10,8))

plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],c=predicted_values ,s=50, cmap='viridis', zorder=1)



plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.5, zorder=2);



w_factor = 0.2 / model.pi.max()

for pos, covar, w in zip(model.mu, model.sigma, model.pi):

    draw_ellipse(pos, covar, alpha = w)