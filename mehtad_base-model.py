# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
df_train.head()
df_train['diagnosis'].value_counts()/len(df_train)
def load_dataset(path):

    eye_files = os.listdir(path)

    return eye_files
train_files = load_dataset('../input/train_images')

test_files = load_dataset('../input/test_images')
dis_classes = df_train['diagnosis'].unique()
print('There are %d total disease categories' %len(dis_classes))

print('There are %s total eye images. \n' % len(np.hstack([train_files, test_files])))

print('There are %d training eye images. \n' % len(train_files))

print('There are %d test eye images. \n' % len(test_files))
import cv2

import matplotlib.pyplot as plt

%matplotlib inline

from glob import glob



train_files = np.array(glob("../input/train_images/*"))

test_files = np.array(glob("../input/test_images/*"))

img = cv2.imread(train_files[1])

plt.imshow(img)

plt.show()

train_files[1]
df_train[df_train.id_code == 'cd01672507c9']
import random

for i in range(10):

    plt.figure(figsize=(10,10))

    i = random.choice(os.listdir('../input/train_images'))

    i_c = i.split('.')[0]

    #print(os.path.join('../input/train_images', i))

    img = cv2.imread(os.path.join('../input/train_images', i))

    print(i, df_train[df_train.id_code == i_c])

    plt.imshow(img)

    plt.show()
def ed_hu_moments(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    feature = cv2.HuMoments(cv2.moments(image)).flatten()

    return feature
gray = cv2.cvtColor(cv2.imread('../input/train_images/3e61703b5ab2.png'), cv2.COLOR_BGR2GRAY)
image=cv2.imread('../input/train_images/3e61703b5ab2.png')

plt.imshow(image)
bins= 8

def ed_histogram(image, mask=None):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([image], [0,1,2], None, [bins,bins,bins],[0,256,0,256,0,256,] )

    cv2.normalize(hist,hist)

    return hist.flatten()
ed_histogram(image)
dis_classes
labels=[]

global_features=[]

for x in train_files:

    image = cv2.imread(x)

    

    x_c = x.split('.')[2].split('/')[3]

    current_label =  np.array(df_train.loc[df_train.id_code == x_c,'diagnosis'])

    labels.append(current_label)

    

    fv_hu_moments = ed_hu_moments(image)

    fv_histogram = ed_histogram(image)

    

    global_feature = np.hstack([fv_hu_moments,fv_histogram])

    global_features.append(global_feature)
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

scaler = MinMaxScaler(feature_range=(0,1))

scaled_features=scaler.fit_transform(np.array(global_features))



le = LabelEncoder()

target = le.fit_transform(labels)



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



model1 = LogisticRegression(multi_class='ovr')

model2 = RandomForestClassifier()

model1.fit(global_features,labels)

model2.fit(global_features,labels)
print(model1.score(global_features, labels))

print(model2.score(global_features, labels))
import numpy as np

import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.datasets import load_digits

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit





def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    """

    Generate a simple plot of the test and training learning curve.



    Parameters

    ----------

    estimator : object type that implements the "fit" and "predict" methods

        An object of that type which is cloned for each validation.



    title : string

        Title for the chart.



    X : array-like, shape (n_samples, n_features)

        Training vector, where n_samples is the number of samples and

        n_features is the number of features.



    y : array-like, shape (n_samples) or (n_samples, n_features), optional

        Target relative to X for classification or regression;

        None for unsupervised learning.



    ylim : tuple, shape (ymin, ymax), optional

        Defines minimum and maximum yvalues plotted.



    cv : int, cross-validation generator or an iterable, optional

        Determines the cross-validation splitting strategy.

        Possible inputs for cv are:

          - None, to use the default 3-fold cross-validation,

          - integer, to specify the number of folds.

          - :term:`CV splitter`,

          - An iterable yielding (train, test) splits as arrays of indices.



        For integer/None inputs, if ``y`` is binary or multiclass,

        :class:`StratifiedKFold` used. If the estimator is not a classifier

        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.



        Refer :ref:`User Guide <cross_validation>` for the various

        cross-validators that can be used here.



    n_jobs : int or None, optional (default=None)

        Number of jobs to run in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.

        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`

        for more details.



    train_sizes : array-like, shape (n_ticks,), dtype float or int

        Relative or absolute numbers of training examples that will be used to

        generate the learning curve. If the dtype is float, it is regarded as a

        fraction of the maximum size of the training set (that is determined

        by the selected validation method), i.e. it has to be within (0, 1].

        Otherwise it is interpreted as absolute sizes of the training sets.

        Note that for classification the number of samples usually have to

        be big enough to contain at least one sample from each class.

        (default: np.linspace(0.1, 1.0, 5))

    """

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt





title = r"Learning Curves Logistic Regression"

# Cross validation with 100 iterations to get smoother mean test and train

# score curves, each time with 20% data randomly selected as a validation set.

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

plot_learning_curve(model1, title, global_features, labels, cv=cv, n_jobs=4)



plt.show()
title = r"Learning Curves RandomForest Claassifier"

# Cross validation with 100 iterations to get smoother mean test and train

# score curves, each time with 20% data randomly selected as a validation set.

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

plot_learning_curve(model2, title, global_features, labels, cv=cv, n_jobs=4)



plt.show()
id_cds=[]

type(id_cds)
test_features = []

id_cds = []

for x in test_files:

    image = cv2.imread(x)

    

    x_c = x.split('.')[2].split('/')[3]

    id_cds.append(x_c)

    

    fv_hu_moments = ed_hu_moments(image)

    fv_histogram = ed_histogram(image)

    

    test_feature = np.hstack([fv_hu_moments,fv_histogram])

    test_features.append(test_feature)

    
test_preds = model2.predict(test_features)
combined_results = pd.DataFrame({'id_code': id_cds, 'diagnosis': test_preds})
combined_results.head()
combined_results.shape
combined_results.to_csv("submission.csv",index=False)