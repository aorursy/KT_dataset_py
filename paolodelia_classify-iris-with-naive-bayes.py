import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

from matplotlib import pyplot as plt

import pylab

import seaborn as sns

from IPython.core.display import display, HTML







filename = '../input/naivebayesleariningsamples/Iris_Data.csv'

print('Setup Complete!')
df = pd.read_csv(filename)



df
df.hist(

    column=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', "species"],

    figsize=(10, 10)

)

pylab.suptitle("Analyzing distribution for the series", fontsize="xx-large")
import scipy.stats as stats



for param in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:

    z, pval = stats.normaltest(df[param])

    

    if(pval < 0.055):

        print("%s has a p-value of %f - distribution is not normal" % (param, pval))

    else:

        print("%s has a p-value of %f" % (param, pval))
display(HTML('<h1>Analyzing the ' +

             '<a href="https://en.wikipedia.org/wiki/Pearson_correlation_coefficient">' +

             'Pearson correlation coefficient</a></h1>'))



# data without the indexes

dt = df[df.columns[0:]]



corr = dt.corr(method="pearson") #returns a dataframe, so it can be reused



# eliminate upper triangle for readability

bool_upper_matrix = np.tril(np.ones(corr.shape)).astype(np.bool)

corr = corr.where(bool_upper_matrix)

display(corr)



# seaborn matrix here

#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

#            square=True, ax=ax)

sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
from sklearn.model_selection import train_test_split

from sklearn import preprocessing



X_train, X_test, y_train, y_test = train_test_split(df.loc[:, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], df.loc[:, 'species'], test_size=0.33, random_state=42)



le = preprocessing.LabelEncoder()



y_train = le.fit_transform(y_train)

y_test = le.transform(y_test)



y_train
from sklearn.naive_bayes import GaussianNB



model = GaussianNB()



model.fit(X_train, y_train)
from sklearn import metrics



y_pred = []



for index, row in X_test.iterrows():

    pred = model.predict([[row[0], row[1],row[2],row[3]]])

    y_pred.append(pred[0])



print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')