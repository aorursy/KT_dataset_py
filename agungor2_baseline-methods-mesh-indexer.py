#Read Files

import os

import time

#Check input files

print(os.listdir("../input"))
from scipy.io import loadmat

data_train_file = "../input/mesh-indexer/training/training.mat"

data_vocabulary_file = "../input/mesh-indexer/training/vocabulary.mat"

data_testing_file = "../input/mesh-indexer/testing/testing.mat"



start = time.time()

training_file = loadmat(data_train_file)

uniquewords_file = loadmat(data_vocabulary_file)

testing_file = loadmat(data_testing_file)

print("Reading time:", time.time()-start)

Xn = training_file["Xn"]

Yn = training_file["Yn"]

Xt = testing_file["Xt"]

vocab = uniquewords_file["vocab"]
#Explore what the data looks like

print("Training Set Size: ", Xn.shape)

print("Training Set Label Size: ", Yn.shape)

print("Testing Set Size: ", Xt.shape)

print("Vocabulary Size: ", vocab.shape)
! pip install pygal
import pygal

from IPython.display import display, HTML



base_html = """

<!DOCTYPE html>

<html>

  <head>

  <script type="text/javascript" src="http://kozea.github.com/pygal.js/javascripts/svg.jquery.js"></script>

  <script type="text/javascript" src="https://kozea.github.io/pygal.js/2.0.x/pygal-tooltips.min.js""></script>

  </head>

  <body>

    <figure>

      {rendered_chart}

    </figure>

  </body>

</html>

"""



def galplot(chart):

    rendered_chart = chart.render(is_unicode=True)

    plot_html = base_html.format(rendered_chart=rendered_chart)

    display(HTML(plot_html))



def plot_class_dist(colum_names):

    line_chart = pygal.Bar()

    line_chart.title = 'Class Distribution in (%)'

    

    for element in range(1,colum_names):

        line_chart.add(str(element),Yn.sum(axis=0)[0,element-1]/Yn.shape[0]*100)

        if Yn.sum(axis=0)[0,element-1]==0:

            print("No Instance in this class: ", element)

    galplot(line_chart)

    

#Class Distribution

plot_class_dist(127)
from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

from wordcloud import WordCloud



horse_mask = np.array(Image.open('../input/images/horse.jpg'))

lion_mask = np.array(Image.open('../input/images/lion.jpg'))



def Compare_WordCloud(class_name, class_image):

    d = {}

    word_freq_matrix = Xn[Yn[:,class_name].indices,:].sum(axis=0)

    for i in range(vocab.shape[1]):

        d[vocab[0][i][0]] = word_freq_matrix[0,i]





    wordcloud = WordCloud(mask = class_image)

    wordcloud.generate_from_frequencies(frequencies=d, max_font_size= 40)

    plt.figure(figsize=(16,13))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    tmp = "WordCloud Data Visualization for Class " + str(class_name + 1)

    plt.title(tmp, fontsize=20)

    plt.imshow(wordcloud.recolor( colormap= 'Pastel2' , random_state=17), alpha=0.98)

    plt.axis('off')

    

#Let's compare the word frequency distribution betweeen two highest mesh term labeled papers 

Compare_WordCloud(11, horse_mask)

Compare_WordCloud(51, lion_mask)
#Read Word Vectors

from tqdm import tqdm

import numpy as np

import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin', binary=True)

#Create a sentence vector from average of all words in the data instance

from scipy import sparse

from tqdm import tqdm

import time

def sentence2vec(s):

    I,J,V = sparse.find(s)

    M = []

    for w in range(len(V)):

        try:

            M.append(V[w]*model[vocab[0][J[w]][0]])

        except:

            continue

    M = np.array(M)

    v = M.sum(axis=0)

    if type(v) != np.ndarray:

        return np.zeros(300)

    return v / np.sqrt((v ** 2).sum())

#start = time.time()

#Create sentence vectors

#xtrain_word2vec = [sentence2vec(Xn[i,:]) for i in tqdm(range(Xn.shape[0]))]

#xtest_word2vec = [sentence2vec(Xt[i,:]) for i in tqdm(range(Xt.shape[0]))]

#print("Sentence2vec Creation Time: ", time.time()-start)
#Uncomment the parts if you wish to make a prediction with Word2Vec Features

#scores = []

#submission = pd.DataFrame()

#start = time.time()

#for i in tqdm(range(Yn.shape[1])):

#    train_target = Yn[:,i].toarray()

#    classifier = LogisticRegression(C=0.1, solver='sag')

#    cv_score = np.mean(cross_val_score(classifier, xtrain_word2vec, train_target, cv=3, scoring='f1'))

#    scores.append(cv_score)

#    print('CV score for class {} is {}'.format(i+1, cv_score))

#    classifier.fit(xtrain_word2vec, train_target)

#    submission[str(i+1)] = classifier.predict(xtest_word2vec)

#print('Total CV score is {}'.format(np.mean(scores)))

#print("Total Computation Time: ", time.time()-start)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

import numpy as np

import pandas as pd

import warnings

import xgboost as xgb

from sklearn.naive_bayes import MultinomialNB

from tqdm import tqdm



warnings.filterwarnings("ignore")

def training_example(clf, x=0):

    scores = []

    submission = pd.DataFrame()

    start = time.time()

    for i in tqdm(range(Yn.shape[1])):

        train_target = Yn[:,i].toarray()

        classifier = clf

        cv_score = np.mean(cross_val_score(classifier, Xn, train_target, cv=3, scoring='f1'))

        scores.append(cv_score)

        print('CV score for class {} is {}'.format(i+1, cv_score))

        classifier.fit(Xn, train_target)

        submission[str(i+1)] = classifier.predict(Xt)

        print(i+1)

    print('Total CV score is {}'.format(np.mean(scores)))

    print("Total Computation Time: ", time.time()-start)

    if x==1:

        return submission



LR = LogisticRegression(C=0.1, solver='sag')

MNB = MultinomialNB()

XGB = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)



#Will skip training other classifiers for now

#classifier_list = [LR, MNB, XGB]

#for clf in classifier_list:

#    print(clf)

#    submission_file = training_example(clf)

#training_example(LR)
#from pandas import Series

#submission_file = training_example(LR, x=1)

#submission_file = submission_file.apply(lambda x: x.index[x.astype(bool)].tolist(), 1)

#submission_file.columns = ["Labels"]

#submission_file['ID'] = Series(np.arange(1,len(submission_file)+1), index=submission_file.index)

#submission_file.to_csv('submission.csv', index=False)