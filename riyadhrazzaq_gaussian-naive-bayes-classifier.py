# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import math



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
class GaussNB:

    

    def __init__(self):

        """

        No params are needed for basic functionality.

        """

        pass

    

    def _mean(self,X): # CHECKED

        """

        Returns class probability for each 

        """

        mu = dict()

        for i in self.classes_:

            idx = np.argwhere(self.y == i).flatten()

            mean = []

            for j in range(self.n_feats):

                mean.append(np.mean( X[idx,j] ))

            mu[i] = mean

        return mu

    

    def _stddev(self,X): # CHECKED

        sigma = dict()

        for i in self.classes_:

            idx = np.argwhere(self.y==i).flatten()

            stddev = []

            for j in range(self.n_feats):

                stddev.append( np.std(X[idx,j]) )

            sigma[i] = stddev

        return sigma

    

    def _prior(self): # CHECKED

        """Prior probability, P(y) for each y

        """

        P = {}

        for i in self.classes_:

            count = np.argwhere(self.y==i).flatten().shape[0]

            probability = count / self.y.shape[0]

            P[i] = probability

        return P

    

    def _normal(self,x,mean,stddev): # CHECKED

        """

        Gaussian Normal Distribution

        $P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma^2_y}\right)$

        """

        

        multiplier = (1/ float(np.sqrt(2 * np.pi * stddev**2))) 

        exp = np.exp(-((x - mean)**2 / float(2 * stddev**2)))

        return multiplier * exp



    

    def P_E_H(self,x,h):

        """

        Uses Normal Distribution to get, P(E|H) = P(E1|H) * P(E2|H) .. * P(En|H)

        

        params

        ------

        X: 1dim array. 

            E in P(E|H)

        H: class in y

        """

        pdfs = []

        

        for i in range(self.n_feats):

            mu = self.means_[h][i]

            sigma = self.stddevs_[h][i]

            pdfs.append( self._normal(x[i],mu,sigma) )

            

        p_e_h = np.prod(pdfs)

        return p_e_h

        

        

    def fit(self, X, y):

        self.n_samples, self.n_feats = X.shape

        self.n_classes = np.unique(y).shape[0]

        self.classes_ = np.unique(y)

        self.y = y

        

        self.means_ = self._mean(X) # dict of list {class:feats}

        self.stddevs_ = self._stddev(X) # dict of list {class:feat}

        self.priors_ = self._prior() # dict of priors 

        

    def predict(self,X):

        samples, feats = X.shape

        if samples!=self.n_samples or feats!=self.n_feats:

            raise DimensionError("No dimension match with training data!")

            

        result = []

        for i in range(samples):

            distinct_likelyhoods = []

            for h in self.classes_:

                tmp = self.P_E_H(X[i],h)

                distinct_likelyhoods.append( tmp * self.priors_[h])

            marginal = np.sum(distinct_likelyhoods)

            tmp = 0

            probas = []

            for h in self.classes_:

                numerator = self.priors_[h] * distinct_likelyhoods[tmp]

                denominator = marginal

                probas.append( numerator / denominator )

                tmp+=1

            # predicting maximum

            idx = np.argmax(probas)

            result.append(self.classes_[idx])

        return result
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()

sk_pred = gnb.fit(X_train, y_train).predict(X_test)

print("Sci-kit Learn: ",accuracy_score(y_test,sk_pred))



nb = GaussNB()

nb.fit(X_train,y_train)

me_pred = nb.predict(X_test)

print("Custom GaussNB: ",accuracy_score(y_test,me_pred))