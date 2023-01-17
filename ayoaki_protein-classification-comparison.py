import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from textwrap import wrap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from IPython.display import Image, clear_output
import scipy.sparse
import math
np.random.seed(7)

# Load sequence data for each protein
all_seqs_df = pd.read_csv('../input/protein-data-set/pdb_data_seq.csv')
# Load characteristic data for each protein
all_charcs_df = pd.read_csv('../input/protein-data-set/pdb_data_no_dups.csv')
protein_charcs = all_charcs_df[all_charcs_df.macromoleculeType == 'Protein'].reset_index(drop=True)
protein_seqs = all_seqs_df[all_seqs_df.macromoleculeType == 'Protein'].reset_index(drop=True)

print(protein_charcs.head())
# print(protein_seqs.head())
# protein_df.isna().sum()
# protein_df.columns

protein_charcs = protein_charcs[['structureId','classification', 'residueCount', 'structureMolecularWeight',\
                         'crystallizationTempK', 'densityMatthews', 'densityPercentSol','phValue']]
protein_seqs = protein_seqs[['structureId','sequence']]

# combine protein characteristics df with their sequences using structureId
protein_all = protein_charcs.set_index('structureId').join(protein_seqs.set_index('structureId'))
protein_all = protein_all.dropna()

# capitalize all classification values to avoid missing any values in the next step
protein_all.classification = protein_all.classification.str.upper()

# drop all proteins with an unknown function; note -- the tilde returns the inverse of a filter
protein_all = protein_all[~protein_all.classification.str.contains("UNKNOWN FUNCTION")]

print(protein_all.head())
class_count = protein_all.classification.value_counts()
functions = np.asarray(class_count[(class_count > 800)].index)
data = protein_all[protein_all.classification.isin(functions)]
data = data.drop_duplicates(subset=["classification","sequence"])  # leaving more rows results in duplciates / index related?
data.head()
data.loc[~data['classification'].str.contains('ASE'), 'classification'] = 'OTHER'
data = data.loc[~data['classification'].str.contains("OTHER")]
data.loc[data['classification'].str.contains('TRANSFERASE/TRANSFERASE INHIBITOR'), 'classification'] = 'TRANSFERASE'
data.loc[data['classification'].str.contains('OXIDOREDUCTASE/OXIDOREDUCTASE INHIBITOR'), 'classification'] = 'OXIDOREDUCTASE'
data.loc[data['classification'].str.contains('HYDROLASE/HYDROLASE INHIBITOR'), 'classification'] = 'HYDROLASE'

print(data.classification.value_counts())
groups = np.asarray(data.classification.value_counts().index)
data
aac_data = data.drop(columns=['residueCount','structureMolecularWeight','classification','crystallizationTempK','densityMatthews','densityPercentSol','phValue'])
aac_data
aa_list = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
def AAC(seq):
    aac = []
    for i in aa_list:
        aac.append(seq.count(i)/len(seq))
    return aac
aac_data['AAC'] = aac_data.sequence.apply(AAC)
aac_data.drop(columns='sequence',inplace=True)
aac_data.head(10)
def dictionary_inator(list):
    return dict(zip(aa_list,list))
aac_data['dictAAC'] = aac_data.AAC.apply(dictionary_inator)
aac_data = aac_data.dictAAC.apply(pd.Series)
aac_data.head(10)
seq_data = data
seq_data['dipeptides'] = seq_data.sequence.apply(lambda string: wrap(string, 2))
seq_data['dipeptides'] = seq_data['dipeptides'].str.join(' ')
seq_data.head()
X = aac_data
y = data['classification']
aacX_train, aacX_test, aacy_train, aacy_test = train_test_split(X,y, train_size=0.85)
svm_aac = svm.SVC()

svm_aac.fit(aacX_train, aacy_train)

s_a_predictions = svm_aac.predict(aacX_test)
s_a_score = accuracy_score(aacy_test, s_a_predictions)
print(s_a_score)
knn_aac = KNeighborsClassifier(n_neighbors = 1)

knn_aac.fit(aacX_train, aacy_train)

k_a_predictions = knn_aac.predict(aacX_test)
k_a_score = accuracy_score(aacy_test, k_a_predictions)
print(k_a_score)
bdt_aac = tree.DecisionTreeClassifier()

bdt_aac.fit(aacX_train, aacy_train)

b_a_predictions = bdt_aac.predict(aacX_test)
b_a_score = accuracy_score(aacy_test, b_a_predictions)
print(b_a_score)
X = seq_data['dipeptides']
y = data['classification']
seqX_train, seqX_test, seqy_train, seqy_test = train_test_split(X,y, train_size=0.85)
vectorizer = TfidfVectorizer(ngram_range = (1,1))
seqX_train_tfidf = vectorizer.fit_transform(seqX_train)
svm_seq = svm.LinearSVC()

svm_seq.fit(seqX_train_tfidf, seqy_train)

s_s_predictions = svm_seq.predict(vectorizer.transform(seqX_test))
s_s_score = accuracy_score(seqy_test, s_s_predictions)
print(s_s_score)
knn_seq = KNeighborsClassifier(n_neighbors = 1)

knn_seq.fit(seqX_train_tfidf, seqy_train)

k_s_predictions = knn_seq.predict(vectorizer.transform(seqX_test))
k_s_score = accuracy_score(seqy_test, k_s_predictions)
print(k_s_score)
bdt_seq = tree.DecisionTreeClassifier()

bdt_seq.fit(seqX_train_tfidf, seqy_train)

b_s_predictions = bdt_seq.predict(vectorizer.transform(seqX_test))
b_s_score = accuracy_score(seqy_test, b_s_predictions)
print(b_s_score)
accuracy_dictionary = {'kNN, AAC':k_a_score,'kNN, Seq':k_s_score,'SVM, AAC':s_a_score,'SVM, Seq':s_s_score, 'BDT, AAC': b_a_score, 'BDT, Seq': b_s_score}
keys = accuracy_dictionary.keys()
values = accuracy_dictionary.values()
plt.bar(keys, values, color=['#848d96','#96c0f2','#848d96','#96c0f2','#848d96','#96c0f2'])
plt.show()
for k,v in accuracy_dictionary.items():
    print(f'Accuracy of {k} is ~{round(v,5)}')