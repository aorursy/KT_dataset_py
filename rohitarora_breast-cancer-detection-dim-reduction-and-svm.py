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
import pandas as pd

from numpy import mean, std

from matplotlib import pyplot as plt



# input_file = os.listdir("../input")[0]

# df = pd.read_csv(input_file)

df = pd.read_csv("../input/wdbc.csv")

df
def normalize_by_column(df): 

    for i in list(df):

        df[i] = list( (df[i].values - mean(df[i].values) ) / std(df[i].values) )

    return df



df_copy = df.copy()

df_for_analyses = df.drop(['id', 'diagnosis_numeric', 'diagnosis'], axis=1)

df_for_analyses_normed = normalize_by_column(df_for_analyses)

array_for_analyses = df_for_analyses_normed.values

diagnosis_numeric = list(df['diagnosis_numeric'].values)
# Run PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(array_for_analyses)

P = pca.transform(array_for_analyses)



diagnosis_pos = [val for ind_, val in enumerate(P) if diagnosis_numeric[ind_]==1]

diagnosis_neg = [val for ind_, val in enumerate(P) if diagnosis_numeric[ind_]==0]



plt.figure(figsize=(4,3))

plt.scatter(list(zip(*diagnosis_pos))[0], list(zip(*diagnosis_pos))[1], label="malignant", facecolors='none', color="red", s=3)

plt.scatter(list(zip(*diagnosis_neg))[0], list(zip(*diagnosis_neg))[1], label="benign", facecolors='none', color="green", s=3)

plt.xticks(fontsize=8); plt.yticks(fontsize=8)

plt.legend(); plt.title("PC1 vs PC2"); plt.xlabel("PC1"); plt.ylabel("PC2")

plt.tight_layout()
import umap

reducer = umap.UMAP(random_state=42, n_neighbors=5)

embedding = reducer.fit_transform(df_for_analyses)



diagnosis_pos_umap = [val for ind_, val in enumerate(embedding) if diagnosis_numeric[ind_]==1]

diagnosis_neg_umap = [val for ind_, val in enumerate(embedding) if diagnosis_numeric[ind_]==0]



plt.figure(figsize=(4,3))

plt.scatter(list(zip(*diagnosis_pos_umap))[0], list(zip(*diagnosis_pos_umap))[1], label="malignant", facecolors='none', color="red", s=3)

plt.scatter(list(zip(*diagnosis_neg_umap))[0], list(zip(*diagnosis_neg_umap))[1], label="benign", facecolors='none', color="green", s=3)

plt.xticks(fontsize=8); plt.yticks(fontsize=8)

plt.legend(); plt.title("UMAP1 vs UMAP2"); plt.xlabel("UMAP1"); plt.ylabel("UMAP2")

plt.tight_layout()
# Run T-SNE



from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=5, learning_rate=100, n_iter=10000, n_iter_without_progress=1000)

tsne_embedding = tsne.fit_transform(array_for_analyses)



diagnosis_pos_tsne = [val for ind_, val in enumerate(tsne_embedding) if diagnosis_numeric[ind_]==1]

diagnosis_neg_tsne = [val for ind_, val in enumerate(tsne_embedding) if diagnosis_numeric[ind_]==0]



plt.figure(figsize=(4,3))

plt.scatter(list(zip(*diagnosis_pos_tsne))[0], list(zip(*diagnosis_pos_tsne))[1], label="malignant", facecolors='none', color="red", s=3)

plt.scatter(list(zip(*diagnosis_neg_tsne))[0], list(zip(*diagnosis_neg_tsne))[1], label="benign", facecolors='none', color="green", s=3)

plt.xticks(fontsize=8); plt.yticks(fontsize=8)

plt.legend(); plt.title("TSNE1 vs TSNE2"); plt.xlabel("TSNE1"); plt.ylabel("TSNE2")

plt.tight_layout()
from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.metrics import roc_curve, auc

from numpy import mean, median, array

from random import shuffle



n_split=3



def svm_roc_auc(x_train, x_test, y_train, y_test):

	kernel='linear'

	clf = svm.SVC(kernel=kernel, C=1, probability=True)

	#auc_list=[]

	#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1./n_split, stratify=Y)

	fitted_clf = clf.fit(x_train, y_train)

	y_score = fitted_clf.predict_proba(x_test)[:, 1]

	fpr, tpr, _ = roc_curve(y_test, y_score)

	auc_roc = auc(fpr, tpr)

	return fpr, tpr, auc_roc



def plot_roc(fpr, tpr, auc_roc, title=""):

	plt.figure(figsize=(4,3))

	plt.plot(fpr, tpr, color='darkorange', lw=2, label='auc = %0.2f)' % auc_roc)

	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.legend()

	plt.title(title, fontsize=8)

	plt.xlabel('FPR', fontsize=8); plt.ylabel('TPR', fontsize=8)

	plt.tight_layout()



x_train_P, x_test_P, y_train_P, y_test_P = train_test_split(P, diagnosis_numeric, test_size=1./n_split, stratify=diagnosis_numeric)

fpr_P, tpr_P, auc_roc_P = svm_roc_auc(x_train_P, x_test_P, y_train_P, y_test_P)

# Randomize labels to get auc. This shows if we can get similar auc with random labeling

diagnosis_numeric_random = array(diagnosis_numeric).copy()

shuffle(diagnosis_numeric_random)

x_train_P_r, x_test_P_r, y_train_P_r, y_test_P_r = train_test_split(P, diagnosis_numeric_random, test_size=1./n_split, stratify=diagnosis_numeric)

fpr_P_r, tpr_P_r, auc_roc_P_r = svm_roc_auc(x_train_P_r, x_test_P_r, y_train_P_r, y_test_P_r)

#

plot_roc(fpr_P, tpr_P, auc_roc_P, title="ROC on PCA reduced data")

plot_roc(fpr_P_r, tpr_P_r, auc_roc_P_r, title="ROC (rand labels) on PCA reduced data")
# Repeat with 2000 random train_tests_splits and find the mean and median auc



n_repeats=2000

auc_list_P=[]

for i in range(n_repeats):

	x_train_P, x_test_P, y_train_P, y_test_P = train_test_split(P, diagnosis_numeric, test_size=1./n_split, stratify=diagnosis_numeric)

	fpr_P, tpr_P, auc_roc_P = svm_roc_auc(x_train_P, x_test_P, y_train_P, y_test_P)

	auc_list_P.append(auc_roc_P)



"%.2f" % median(auc_list_P)

'0.99'



# D the same for the randomized label set

auc_list_P_random=[]

for i in range(n_repeats):

	x_train_P_r, x_test_P_r, y_train_P_r, y_test_P_r = train_test_split(P, diagnosis_numeric_random, test_size=1./n_split, stratify=diagnosis_numeric_random)

	fpr_P_r, tpr_P_r, auc_roc_P_r = svm_roc_auc(x_train_P_r, x_test_P_r, y_train_P_r, y_test_P_r)

	auc_list_P_random.append(auc_roc_P_r)



"%.2f" % median(auc_list_P_random)

'0.50'



plt.figure(figsize=(4,3)); plt.title("auc distribution n=2000", fontsize=8)

plt.hist(auc_list_P_random, bins=50, label="random labeling", density=True)

plt.hist(auc_list_P, bins=50, label="labeling from data", density=True); plt.legend()

plt.tight_layout()
df_copy_for_analyses = df_copy.drop(['id', 'diagnosis', 'diagnosis_numeric'], axis=1)

number_of_features_selected = len(list(df_copy_for_analyses))

number_of_rows=10; number_of_columns=3



plt.figure(figsize=(30,50))

for i in range(1, number_of_features_selected+1):

	plt.subplot(10,3,i)

	color_by = list(df_copy_for_analyses)[i-1] # Recall we start i with 1

	_=plt.scatter(list(zip(*P))[0], list(zip(*P))[1], label="malignant", facecolors='none', c=df_copy[color_by].values, cmap='PuBuGn')

	_=plt.xticks(fontsize=4); plt.yticks(fontsize=4)

	plt.title(color_by, fontsize=15)

	cbar=plt.colorbar()

	cbar.ax.tick_params(labelsize=6)

	plt.subplots_adjust(hspace=0.9)