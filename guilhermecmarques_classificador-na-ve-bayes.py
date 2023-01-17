import os # accessing directory structure
print(os.listdir('../input/email-data'))
%matplotlib inline

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# import dataset as a dataframe
spam_df = pd.read_csv("../input/email-data/train_data.csv",
        sep=r'\s*,\s*',
        engine='python')
spam_df = spam_df.replace(np.nan,' ', regex=True)

# import test data
testdf = pd.read_csv("../input/email-data/test_features.csv",
        sep=r'\s*,\s*',
        engine='python')
testdf = testdf.replace(np.nan,' ', regex=True)
spam_df.info()
spam_df.head()
spam_df.describe()
Y = spam_df.ham
X = spam_df.drop(['ham','Id'], axis=1)
# Create correlation matrix
corr_matrix = X.corr().abs()

plt.matshow(corr_matrix)
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.8
correlate = [column for column in upper.columns if any(upper[column] > 0.8)]

correlate
X = X.drop(correlate, axis=1)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# feature extraction
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 4)
rfe_fit = rfe.fit(X, Y)

# obtain the names of the selected features
feat_name = X.columns[rfe_fit.get_support(True)]
feat_name
ham = spam_df.query('ham == 1')

spam = spam_df.query('ham == 0')
ham_remove = np.mean(ham['word_freq_remove'])
spam_remove = np.mean(spam['word_freq_remove'])

labels = ["Ham", "Spam"]

plt.bar([0,1], [ham_remove, spam_remove], tick_label=labels)
plt.title('Palavra "remove" X Categoria')
plt.ylabel('Média "remove"')
plt.xlabel('Categoria')
ham_000 = np.mean(ham['word_freq_000'])
spam_000 = np.mean(spam['word_freq_000'])

plt.bar([0,1], [ham_000, spam_000], tick_label=labels)
plt.title('Palavra "000" X Categoria')
plt.ylabel('Média "000"')
plt.xlabel('Categoria')
ham_george = np.mean(ham['word_freq_george'])
spam_george = np.mean(spam['word_freq_george'])

plt.bar([0,1], [ham_george, spam_george], tick_label=labels)
plt.title('Palavra "george" X Categoria')
plt.ylabel('Média "george"')
plt.xlabel('Categoria')
ham_cif = np.mean(ham['char_freq_$'])
spam_cif = np.mean(spam['char_freq_$'])

plt.bar([0,1], [ham_cif, spam_cif], tick_label=labels)
plt.title('Palavra "$" X Categoria')
plt.ylabel('Média "$"')
plt.xlabel('Categoria')
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

naive = MultinomialNB()
result = 0
Xfinal = X
feats = feat_name

for n in range(1,60):
    # repeat the RFE method to every n possible features
    rfe = RFE(model, n)
    
    # apply the RFE
    rfe_fit = rfe.fit(X, Y)
    
    # obtain the feature names
    feat_name = X.columns[rfe_fit.get_support(True)]
    
    # filter the names out
    X_iter = X.filter(feat_name, axis=1)
    
    # train the estimator based on the defined features
    nbfit = naive.fit(X_iter, Y)
    nb_res = naive.score(X_iter,Y)
    
    
    if nb_res > result:
        result = nb_res
        best_n = n
        Xfinal = X_iter
        feats = feat_name
        
        print([best_n, result])
nbfit = naive.fit(Xfinal, Y)

y_pred = nbfit.predict(Xfinal)

from sklearn.metrics import fbeta_score
fbeta_score(Y, y_pred, average='macro', beta=3)
Xtest = testdf.filter(feats, axis=1)
Ypred = nbfit.predict(Xtest)

preds = pd.DataFrame(testdf.Id)
preds["ham"] = Ypred
preds
preds.to_csv("prediction.csv", index=False)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors=5)
knn_scores = cross_val_score(knn, Xfinal, Y, cv=10)
np.mean(knn_scores)
nb_scores = cross_val_score(nbfit, Xfinal, Y, cv=10)
np.mean(nb_scores)
