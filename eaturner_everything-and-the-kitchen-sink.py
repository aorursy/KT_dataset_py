# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/training_variants')
test = pd.read_csv('../input/test_variants')
trainx = pd.read_csv('../input/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv('../input/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

train = pd.merge(train, trainx, how='left', on='ID')

test = pd.merge(test, testx, how='left', on='ID')
def catCount( train, test, col ):
    
    train.loc[:, col + '_count']  = train[col].apply(lambda x: len(x.split()))
    test.loc[:, col + '_count'] = test[col].apply(lambda x: len(x.split()))

    return train, test

train, test = catCount(train, test, 'Text')
def newCatFeatures( train_df, test_df, col, n_comp = 25 ):
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    
    print ( 'Features Before: ' + str(train_df.shape[1]) )
    
    wordSpace = train_df[col].append(test_df[col])
    
    wordCounts = [len(x.split()) for x in wordSpace]
    
    if np.max(wordCounts) > 20:
        tfidf = TfidfVectorizer(strip_accents='unicode',lowercase =True, analyzer='char_wb', 
	               ngram_range = (2,3), norm = 'l2', sublinear_tf = True, min_df = 1e-2,
                                   stop_words = 'english').fit(wordSpace)
    else:
        tfidf = TfidfVectorizer(strip_accents='unicode',lowercase =True, analyzer='char', 
	            ngram_range = (1,8), norm = 'l2', sublinear_tf = True, 
                                stop_words = 'english').fit(wordSpace)
        
    print ('Found term frequencies')
    
    svd = TruncatedSVD(n_components = n_comp, n_iter=25, random_state=12)
    
    Xtr = svd.fit_transform( tfidf.transform( train_df[col] ) )
    Xtst = svd.transform( tfidf.transform( test_df[col] ) )

    print ('Performed SVD')
    
    features_ = [ col + '_tfidf_svd_' + str(i+1) for i in range(Xtr.shape[1]) ]

    train_df = pd.concat( [train_df, pd.DataFrame(Xtr, columns = features_) ], axis = 1)
    test_df = pd.concat( [test_df, pd.DataFrame(Xtst, columns = features_) ], axis = 1)

    print ( 'Features After: ' + str(train_df.shape[1]) + '\n')
    
    train_df.drop(col, axis = 1, inplace = True)
    test_df.drop(col, axis = 1, inplace = True)
    
    return train_df, test_df
train, test = newCatFeatures(train, test, 'Gene', n_comp = 20)

train, test = newCatFeatures(train, test, 'Variation', n_comp = 20)

train, test = newCatFeatures(train, test, 'Text', n_comp = 50)
train.drop('ID', axis = 1, inplace = True)

classes = ['Class' + str(i + 1) for i in range(9)]

df_sub = pd.DataFrame( columns = ['ID'] )

df_sub['ID'] = test.pop('ID')
import matplotlib.pyplot as plt
import seaborn as sns

cor_ = train.corr()

sns.heatmap(cor_, vmax=.8, square=True)

plt.show()

train_labels = train.pop('Class') - 1
#base learners
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

#model selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#aid in model selection
from scipy.stats import expon

param_grid = {"n_neighbors": range(2,7)}
param_dist = {'C': expon(scale=100) }

clf_list = [
    GridSearchCV(KNeighborsClassifier(weights = 'uniform'), param_grid, cv = 5, scoring = 'neg_log_loss'),
    GridSearchCV(KNeighborsClassifier(weights = 'distance'), param_grid, cv = 5, scoring = 'neg_log_loss'),
    LogisticRegressionCV(cv = 5, solver = 'sag', multi_class = 'multinomial', n_jobs = -1),
    MLPClassifier(activation = 'logistic', learning_rate = 'adaptive', warm_start = True),
    MLPClassifier(activation = 'identity', learning_rate = 'adaptive', warm_start = True),
    MLPClassifier(activation = 'tanh', learning_rate = 'adaptive', warm_start = True),
    MLPClassifier(activation = 'relu', learning_rate = 'adaptive', warm_start = True)
]

n = len(clf_list)

predictions = np.zeros( (test.shape[0], 9) )

for i in range(n):
    print('At classifer: ' + str(i + 1) )
    clf = clf_list[i]
    
    clf.fit(train, train_labels)
    
    predictions = predictions + clf.predict_proba( test )

predictions = (1.0/n) * predictions

df_pred = pd.DataFrame(predictions, columns = classes)
df_sub = pd.concat([df_sub, df_pred], axis = 1)

df_sub.to_csv('submission.csv', index=False)
df_sub.head(10)