import pandas as pd

import numpy as np



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD



from sklearn.neighbors import BallTree

from sklearn.base import BaseEstimator



from sklearn.pipeline import make_pipeline



import warnings

warnings.filterwarnings("ignore")
# Read file

lines = [line.rstrip('\n').replace('\\n',' ').replace('>','') for line in open('../input/input (Cleaned).txt')]
# Make dataframe with context and reply

# I assume that each line is a reply for the previous replic



subtitles = pd.DataFrame(columns=['context', 'reply'])

subtitles['context'] = lines

subtitles['context'] = subtitles['context'].apply(lambda x: x.lower())

subtitles['reply'] = lines[1:] + ['...']

subtitles['reply'] = subtitles['reply'].apply(lambda x: x.lower())
# Add whitespaces before every sign

for sign in ['!', '?', ',', '.', ':']:

    subtitles['context'] = subtitles['context'].apply(lambda x: x.replace(sign, f' {sign}'))

    subtitles['reply'] = subtitles['reply'].apply(lambda x: x.replace(sign, f' {sign}'))
subtitles.info()
subtitles.iloc[100:120]
# Lets vectorize our context corpus

vectorizer = TfidfVectorizer()

vectorizer.fit(subtitles.context)



matrix_big = vectorizer.transform(subtitles.context)
matrix_big.shape
# SVD dimensionality reduction

# You may try to increase number of components, but performance will become lower and may rise memory error

svd = TruncatedSVD(n_components=300, algorithm='arpack')



svd.fit(matrix_big)

matrix_small = svd.transform(matrix_big)



# Print new dimensionality and explained variance ratio

print(matrix_small.shape)

print(svd.explained_variance_ratio_.sum())
# Probability  function for choosing one of the relevant answers

def softmax(x):

    proba = np.exp(-x)

    return proba/sum(proba)



# Choosing one of the k nearest neighbors with BallTree algorithm

class NeighborSampler(BaseEstimator):

    def __init__(self, k=5, temperature = 1.0):

        self.k = k

        self.temperature = temperature

    

    def fit(self, X, y):

        self.tree_ = BallTree(X)

        self.y_ = np.array(y)

        

    def predict(self, X, random_state = None):

        distances, indeces = self.tree_.query(X, return_distance = True, k = self.k)

        result = []

        for distance, index in zip(distances, indeces):

            result.append(np.random.choice(index, p = softmax(distance * self.temperature)))

            

        return self.y_[result]
ns = NeighborSampler()

ns.fit(matrix_small, subtitles.reply)



# Vectorize, SVD and then chose an answer

pipe = make_pipeline(vectorizer, svd, ns)
print(pipe.predict(['Hello !']))
print(pipe.predict(['Do you like me?']))
print(pipe.predict(['Do you like me?']))
print(pipe.predict(['you are weird']))
print(pipe.predict(['how are you ?']))
print(pipe.predict(['lets kill him']))
print(pipe.predict(['i am sorry']))
print(pipe.predict(['check out this thing']))
print(pipe.predict(['konoha is in danger']))
print(pipe.predict(['i will destroy this city']))
print(pipe.predict(['prepare for battle']))
print(pipe.predict(['never forget me']))