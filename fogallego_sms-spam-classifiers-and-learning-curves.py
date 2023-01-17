import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

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

          - An object to be used as a cross-validation generator.

          - An iterable yielding train/test splits.



        For integer/None inputs, if ``y`` is binary or multiclass,

        :param train_sizes:

        :class:`StratifiedKFold` used. If the estimator is not a classifier

        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.



        Refer :ref:`User Guide <cross_validation>` for the various

        cross-validators that can be used here.



    n_jobs : integer, optional

        Number of jobs to run in parallel (default 1).

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
df = pd.read_csv('../input/spam.csv', encoding='latin-1')

df.drop(df.columns[[2, 3, 4]], axis=1, inplace=True)

df['v1'] = df['v1'].astype('category').cat.codes

df.rename(index=str, columns={"v1": "isspam", "v2": "text"}, inplace=True)

df = df[['text', 'isspam']]

df.head()
# Preprocess:

# lowercase

df.text = df.text.str.lower()

# numbers

df.text = df.text.str.replace('\d+', ' number ')

# urls

df.text = df.text.str.replace('(http|https)://[^\s]*', ' httpaddr ')

# email adresses

df.text = df.text.str.replace('[^\s]+@[^\s]+', ' emailaddr ')



df.head()
df[df['text'].str.contains("[£]+")].head(10)
df[df['text'].str.contains("&lt;#&gt;")].head()
df[df['text'].str.contains("<|>")].head()
df[df['text'].str.contains("&lt;#&gt;")].groupby('isspam').describe()
df[df['text'].str.contains("<|>")].groupby('isspam').describe()
df.text = df.text.str.replace('&lt;#&gt;', ' impltgt ')

df.text = df.text.str.replace('<|>', ' expltgt ')

df.text = df.text.str.replace('[^a-zA-Z0-9\s]+', ' othersym ')

df.text = df.text.str.replace('\s+', ' ')

df.head()
count_vect = CountVectorizer(max_df=0.99, min_df=0.01)

X = count_vect.fit_transform(df.text)

y = df.isspam

X.shape
# Shuffle for learning curves

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
from sklearn.svm import SVC

model_svc = SVC(kernel='rbf', gamma=0.001, C=1.0)

plot_learning_curve(model_svc, 'Learning Curve (SVC)', X, y, (0.7, 1.01), cv=cv, n_jobs=4)

plt.show()
from sklearn.naive_bayes import GaussianNB

model_gnb = GaussianNB()

plot_learning_curve(model_gnb, 'Learning Curve (Gaussian NB)', X.toarray(), y, (0.5, 1.01), cv=cv, n_jobs=4)

plt.show()
from sklearn.naive_bayes import MultinomialNB

model_mnb = MultinomialNB()

plot_learning_curve(model_mnb, 'Learning Curve (Multinomial NB)', X.toarray(), y, (0.7, 1.01), cv=cv, n_jobs=4)

plt.show()
from sklearn.ensemble import RandomForestClassifier

model_rb = RandomForestClassifier(random_state=0)

plot_learning_curve(model_rb, 'Learning Curve (Random Forest)', X, y, (0.7, 1.01), cv=cv, n_jobs=4)

plt.show()
from sklearn.neural_network import MLPClassifier

model_mlp = MLPClassifier()

plot_learning_curve(model_rb, 'Learning Curve (Multilayer Perceptron)', X, y, (0.7, 1.01), cv=cv, n_jobs=4)

plt.show()