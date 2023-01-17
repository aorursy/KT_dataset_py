# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import scipy.stats
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from nltk.metrics import *
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import wordnet as wn
from itertools import product
from sklearn.cluster import KMeans

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
# print(os.listdir("../working/"))
# print(os.listdir("../input/datamatchingdataset-1/query-impala-402338.csv"))
# df2 = pd.read_csv("../input/datamatchingdataset-1/query-impala-402336.csv")
# df2.head()
!wget "http://nlp.stanford.edu/data/wordvecs/glove.6B.zip"
!unzip -o glove.6B.zip
def read_data(file_name):
    with open(file_name,'r') as f:
        word_vocab = set() # not using list to avoid duplicate entry
        word2vector = {}
        for line in f:
            line_ = line.strip() #Remove white space
            words_Vec = line_.split()
            word_vocab.add(words_Vec[0])
            word2vector[words_Vec[0]] = np.array(words_Vec[1:],dtype=float)
    print("Total Words in DataSet:",len(word_vocab))
    return word_vocab,word2vector
word2vector = read_data("glove.6B.300d.txt")[1]
# glove_vectors = pd.read_csv("glove.6B.300d.txt", sep="\t", header=None, encoding='utf-8')
# glove_vectors.head()
word2vector["hilarious"]
def cos_sim(u,v):
    """
    u: vector of 1st word
    v: vector of 2nd Word
    """
    numerator_ = u.dot(v)
    denominator_= np.sqrt(np.sum(np.square(u))) * np.sqrt(np.sum(np.square(v)))
    return numerator_/denominator_
hilarious_v = word2vector["hilarious"]
funny_v = word2vector["funny"]
cos_sim(hilarious_v,funny_v)
hilarious_v = word2vector["hilarious"]
serious_v = word2vector["serious"]
cos_sim(hilarious_v,serious_v)
df1 = pd.read_excel("../input/parattlist/att_list.xlsx","att_list")
df1 = df1.sort_values(by=['PAR_ATT_LIST_PK'])
df1.head()

# word2vector["customer"].reshape((1, 300))
# word_list = ["customer", "id", "customer id", "client id", "client", "id"]
word_list_set = set(word_list)
return_matrix_ = np.random.randn(len(word_list_set),300)
i = 0
for word in word_list_set:
    word = word.lower()
    if "_" in word:
        word = word.replace("_"," ")
    if " " in word:
        tokenized_words = word.split(" ")
        sum_1 = np.zeros((1,300))
        for t1 in tokenized_words:
            t1 = t1.lower().strip()
            if t1 in word2vector:
                sum_1 = np.add(sum_1,word2vector[t1].reshape((1, 300)))
        return_matrix_[i] = sum_1
    else:
        if word.strip() in word2vector[word]:
            return_matrix_[i] = word2vector[word.lower()].reshape((1, 300))
        else:
            print("released")
            return_matrix_[i] = np.zeros((1,300))
    i+=1
return_matrix_
# word1 = word2vector["customer"]
# word2 = word2vector["id"]
# w1w2 = np.add(word1,word2)
# word3 = word2vector["client"]
# w3w4 = np.add(word3,word2)


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
# return_matrix_ = [word1, word3]
# random_words = ["important", "imperative"]
# return_matrix_ = [word1, word2, w1w2, word3, w3w4]
# random_words = ["customer", "id", "customer id", "client", "client id"]
# return_matrix_ = return_matrix(random_words)
pca_ = PCA(n_components=2)
viz_data = pca_.fit_transform(return_matrix_)
viz_data
plt.figure(figsize=(35,35))
plt.scatter(viz_data[:,0],viz_data[:,1],cmap=plt.get_cmap('Spectral'))
for label,x,y in zip(word_list,viz_data[:,0],viz_data[:,1]):
    plt.annotate(
        label,
        xy=(x,y),
        xytext=(-14, 14),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
    )
plt.xlabel('PCA Component 1 ')
plt.ylabel('PCA Component 2')
plt.title('PCA representation for Word Embedding')
plt.xlim(-50,50)
plt.ylim(-30,30)
fig = plt.figure()
fig = plt.figure(figsize=(50,50))
fig.canvas.draw()
word_list = set(final_dataframe["column_1"]).union(set(final_dataframe["column_2"]))
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1,perplexity=3,method='exact')
tsne_results = tsne.fit_transform(return_matrix_)
# plt.figure(figsize=(35,35))
plt.scatter(tsne_results[:,0],tsne_results[:,1],cmap=plt.get_cmap('Spectral'))
for label,x,y in zip(word_list,tsne_results[:,0],tsne_results[:,1]):
    plt.annotate(
        label,
        xy=(x,y),
        xytext=(-14, 14),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
    )
plt.xlabel('TSNE Component 1 ')
plt.ylabel('TSNE Component 2')
plt.title('TSNE representation for Word Embedding')
# plt.xlim(-50,50)
# plt.ylim(-30,30)
# df1 = pd.read_csv("../input/datamatchingdataset-2/query-impala-402328.csv")
# df1.head()
df2 = pd.read_excel("../input/parattvalues/att_values.xlsx", "Export Worksheet")
df2 = df2.sort_values(by=['PAR_FK'])
df2.head()
# df3 = pd.read_csv("../input/dataset-1-details/SRC_LMES_MTR_DTLS.csv")
# df3.head()
import re
regex_pat1 = re.compile(r'\([0-9]+\)', flags=re.IGNORECASE)
regex_pat2 = re.compile(r'[0-9]+', flags=re.IGNORECASE)
regex_pat3 = re.compile(r'PAR_ATT', flags=re.IGNORECASE)
regex_pat4 = re.compile(r'_PK', flags=re.IGNORECASE)
regex_pat6 = re.compile(r'PAR_', flags=re.IGNORECASE)
regex_pat7 = re.compile(r'VAL_', flags=re.IGNORECASE)
df3 = pd.read_csv("../input/parattlist-datatypes/PAR_ATT_LIST.csv")
df3["col_name"] = df3["col_name"].str.strip()
df3["data_type"] = df3["data_type"].str.strip()
df3["data_type"] = df3["data_type"].str.replace(regex_pat1,"")
df3["data_type"] = df3["data_type"].str.replace(regex_pat2,"")
df3["data_type"] = df3["data_type"].str.replace("NOT NULL","")
# df3["data_type"] = df3["data_type"].str.replace('[^A-Za-z\s]+', '')
# df3["col_name"] = df3["col_name"].str.replace(regex_pat3,'PARAMETER_ATTRIBUTE')
# df3["col_name"] = df3["col_name"].str.replace(regex_pat4,'_PRIMARY_KEY')
df3.head(13)
# df4 = pd.read_csv("../input/dataset-2-details/SRC_LMES_MTR_COUNTS.csv")
# df4.head()
df4 = pd.read_csv("../input/parattvalues-datatypes/PAR_ATT_VALUES.csv")
# df4.head()
regex_pat1 = re.compile(r'\([0-9]+,{0,}[0-9]+\)', flags=re.IGNORECASE)
regex_pat5 = re.compile(r'_FK', flags=re.IGNORECASE)
df4["col_name"] = df4["col_name"].str.strip()
df4["data_type"] = df4["data_type"].str.strip()
df4["data_type"] = df4["data_type"].str.replace(regex_pat1,"")
df4["data_type"] = df4["data_type"].str.replace(regex_pat2,"")
df4["data_type"] = df4["data_type"].str.replace("NOT NULL","")
# df4["data_type"] = df4["data_type"].str.replace('[^A-Za-z\s]+', '')
# df4["col_name"] = df4["col_name"].str.replace(regex_pat3,'PARAMETER_ATTRIBUTE')
# df4["col_name"] = df4["col_name"].str.replace(regex_pat4,'_PRIMARY_KEY')
# df4["col_name"] = df4["col_name"].str.replace(regex_pat5,'_FOREIGN_KEY')
# len(df4)
df4.head(12)
df2 = df2.head(2000)
len(df2)
TABLE_NAME_1 = "PARAMETER_ATTRIBUTE_LIST"
TABLE_NAME_2 = "PARAMETER_ATTRIBUTE_VALUES"

data = {}
data["column_1"] = []
data["column_2"] = []
data["table_name_1"] = []
data["table_name_2"] = []
data["data_type_1"] = []
data["data_type_2"] = []
data["dataset_1"] = []
data["dataset_2"] = []

# df4.columns
# df4.get_value(df4.index[df4['col_name'] == "location"][0], "data_type")
# def return_data_type(col1):
    

final_df = pd.DataFrame(data)
for c1 in df1.columns:
    for c2 in df2.columns:
        data["column_1"].append(c1)
        data["column_2"].append(c2)
        data["table_name_1"].append(TABLE_NAME_1)
        data["table_name_2"].append(TABLE_NAME_2)
        data["data_type_1"].append(df3.get_value(df3.index[df3['col_name'] == c1][0], "data_type"))
        data["data_type_2"].append(df4.get_value(df4.index[df4['col_name'] == c2][0], "data_type"))
        data["dataset_1"].append(np.array(df1[c1]))
        data["dataset_2"].append(np.array(df2[c2]))

final_dataframe = pd.DataFrame(data=data)
final_dataframe.head()
final_dataframe["column_1"] = final_dataframe["column_1"].str.replace(regex_pat3,'PARAMETER_ATTRIBUTE')
final_dataframe["column_1"] = final_dataframe["column_1"].str.replace(regex_pat4,'_PRIMARY_KEY')
final_dataframe["column_1"] = final_dataframe["column_1"].str.replace(regex_pat5,'_FOREIGN_KEY')
final_dataframe["column_1"] = final_dataframe["column_1"].str.replace(regex_pat6,'PARAMETER_')
final_dataframe["column_1"] = final_dataframe["column_1"].str.replace(regex_pat7,'VALUES_')

final_dataframe["column_2"] = final_dataframe["column_2"].str.replace(regex_pat3,'PARAMETER_ATTRIBUTE')
final_dataframe["column_2"] = final_dataframe["column_2"].str.replace(regex_pat4,'_PRIMARY_KEY')
final_dataframe["column_2"] = final_dataframe["column_2"].str.replace(regex_pat5,'_FOREIGN_KEY')
final_dataframe["column_2"] = final_dataframe["column_2"].str.replace(regex_pat6,'PARAMETER_')
final_dataframe["column_2"] = final_dataframe["column_2"].str.replace(regex_pat7,'VALUES_')

final_dataframe.head()
calculated_columns = []

# Feature 1 calculation:
def jaccard_distance_func(row1):
    jcd = jaccard_distance(set(row1[4]),set(row1[5]))
    return float(jcd)

def match_data_type_func(row1):
    d1 = row1["data_type_1"].strip('\n')
    d2 = row1["data_type_2"].strip('\n')
    d1 = row1["data_type_1"].strip('\t')
    d2 = row1["data_type_2"].strip('\t')
    d1 = row1["data_type_1"].strip()
    d2 = row1["data_type_2"].strip()
    if  d1 == d2:
        return 1
    else:
        if (d1 in d2) or (d2 in d1):
            return 0.5
        else:
            return 0.0
#     return edit_distance(row1[4], row1[5])

def tokenize_words(word):
    return word.split("_")

def edit_distance_between_columns_with_tokenize(row1):
    col1 = tokenize_words(row1[0])
    col2 = tokenize_words(row1[1])
    distance_metric = 0
    for col_1 in col1:
        for col_2 in col2:
            distance_metric += edit_distance(col_1, col_2)
    return distance_metric/(len(col1)*len(col2))

def edit_distance_between_columns_without_tokenize(row1):
    return edit_distance(row1[0], row1[1])


def determine_similarity_between_columns(row1):
    t_sim = 0
    words1 = tokenize_words(row1[0])
    words2 = tokenize_words(row1[1])
    for word1 in words1:
        for word2 in words2:    
            syns1 = set(wn.synsets(word1))
            syns2 = set(wn.synsets(word2))
            if len(syns1) == 0 or len(syns2) == 0:
                t_sim = max(t_sim,0)
            else:
                t_sim = max(t_sim, max([0 if(wn.wup_similarity(s1,s2)) is None else wn.wup_similarity(s1,s2) for s1,s2 in product(syns1, syns2)]))
    return t_sim

def determine_composite_word_matrix(words):
    sum_1 = np.zeros((1,300))
    tokenized_words = words.split('_')
    for word in tokenized_words:
        if word.lower() in word2vector:
            sum_1 = np.add(sum_1,word2vector[word.lower()].reshape((1, 300)))
    return sum_1
    
def determine_semantic_similarity_from_glove_vectors(row1):
    return_matrix_ = np.random.randn(2,300)
    col1 = row1[0]
    col2 = row1[1]
    return_matrix_[0] = determine_composite_word_matrix(col1)
    return_matrix_[1] = determine_composite_word_matrix(col2)
    return cos_sim(return_matrix_[0], return_matrix_[1])

def LCSubStr(row1):
    # Create a table to store lengths of
    # longest common suffixes of substrings. 
    # Note that LCSuff[i][j] contains the 
    # length of longest common suffix of 
    # X[0...i-1] and Y[0...j-1]. The first
    # row and first column entries have no
    # logical meaning, they are used only
    # for simplicity of the program.
     
    # LCSuff is the table with zero 
    # value initially in each cell
    X = row1[0]
    Y = row1[1]
    m = len(X)
    n = len(Y)
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)]
     
    # To store the length of 
    # longest common substring
    result = 0
    # Following steps to build
    # LCSuff[m+1][n+1] in bottom up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            elif (X[i-1] == Y[j-1]):
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                result = max(result, LCSuff[i][j])
            else:
                LCSuff[i][j] = 0
    return result

final_dataframe["jcd"] = final_dataframe.apply(jaccard_distance_func, axis = 1)
final_dataframe["jcs"] = 1-final_dataframe["jcd"]
final_dataframe["data_type_match"] = final_dataframe.apply(match_data_type_func, axis = 1)
final_dataframe["edit_distance_between_columns"] = final_dataframe.apply(edit_distance_between_columns_without_tokenize, axis = 1)
final_dataframe["sim_score_between_columns"] = final_dataframe.apply(determine_semantic_similarity_from_glove_vectors, axis = 1)
final_dataframe["longest_common_substring"] = final_dataframe.apply(LCSubStr, axis = 1)
final_dataframe.head()
# final_dataframe = preprocessing.normalize(final_dataframe["edit_distance"],norm='l1',axis=0)

extracted_data_frame = final_dataframe[["jcs","data_type_match","edit_distance_between_columns","sim_score_between_columns","longest_common_substring"]]
extracted_data_frame.head()
scaler = MinMaxScaler()
scaler.fit(extracted_data_frame)
X_scaled = scaler.transform(extracted_data_frame)
X_scaled
X_scaled[:,2] = 1-X_scaled[:,2]
X_scaled
avg_matrix = np.mean(X_scaled, axis = 1)
x_matches = final_dataframe[ avg_matrix[0:,]>0.6 ]
len(x_matches)
final_output = pd.DataFrame({'column1': x_matches['column_1'], 
                             'column2': x_matches['column_2'], 
                             'avg_value': avg_matrix[avg_matrix[0:,]>0.6], 
                             'jcs': x_matches['jcs'],
                             'data_type_1': x_matches['data_type_1'],
                             'data_type_2': x_matches['data_type_2'],
                             'data_type_match': x_matches['data_type_match'],
                             'sim_score_between_columns': x_matches['sim_score_between_columns'],
                             'edit_distance_between_columns': x_matches['edit_distance_between_columns']
                            })
final_output.head(15)
# final_output.to_csv("output.csv", index=False)
# !wget "http://nlp.stanford.edu/data/wordvecs/glove.6B.zip"
# from nltk.wsd import lesk

# sent = ["invalid","assay","count"]
# print(lesk(sent, 'assay').definition())
# import gensim
# from nltk.corpus import brown
# model = gensim.models.Word2Vec(brown.sents())
# model.save('brown.embedding')
# new_model = gensim.models.Word2Vec.load('brown.embedding')
# import numpy as np
# from scipy import spatial

# a1 = new_model["assay"]
# a2 = new_model["reason"]
# a3 = new_model["count"]
# w1 = new_model["assay"]
# w2 = new_model["reason"]
# w3 = new_model["code"]
# af = np.add(a1,a2, a3)
# wf = np.add(w1,w2, w3)
# result = 1 - spatial.distance.cosine(af, wf)
# result
# new_model.similarity('procedure','analysis')

# X_scaled = preprocessing.MinMaxScaler(extracted_data_frame)
# print(X_scaled)
# replace_dict_1 = {}
# replace_dict_2 = {}
# output_dict = {}
# output_dict["column_1"] = []
# output_dict["column_2"] = []
# output_dict["jaccard_dist"] = []
# #final_output = pd.DataFrame({'opportunity_number':o_no, 'result':preds})
# for column_name in df1.columns:
#     if df1[column_name].dtype == "object":
#         replace_dict_1[column_name] = "NA"
#     elif df1[column_name].dtype == "float64":
#         replace_dict_1[column_name] = 0.0
#     elif df1[column_name].dtype == "int64":
#         replace_dict_1[column_name] = 0

# for column_name in df2.columns:
#     if df2[column_name].dtype == "object":
#         replace_dict_2[column_name] = "NA"
#     elif df2[column_name].dtype == "float64":
#         replace_dict_2[column_name] = 0.0
#     elif df2[column_name].dtype == "int64":
#         replace_dict_2[column_name] = 0

# df1 = df1.fillna(value=replace_dict_1)
# df2 = df2.fillna(value=replace_dict_2)
# column_pairs = []
# for col_name_1 in df1.columns.values:
#     for col_name_2 in df2.columns.values:
#         column_pairs.append([col_name_1, col_name_2])
# for column_pair in column_pairs:
#         if df1[column_pair[0]].dtype == df2[column_pair[1]].dtype:
#                 df_3 = np.unique(df1[[column_pair[0]]])
#                 df_4 = np.unique(df2[[column_pair[1]]])
#                 #stats1, p_value = scipy.stats.kruskal(df_3, df_4)
#                 output_dict["column_1"].append(column_pair[0])
#                 output_dict["column_2"].append(column_pair[1])
#                 output_dict["jaccard_dist"].append(jaccard_distance(set(df_3),set(df_4)))
# #                 if stats1 < 0.1:
# #                     print(column_pair, "similar")
# #                     print(stats1)
# #                     print(p_value)
# #                 else:
# #                     print(column_pair, "dis-similar")
#         else:
#             pass
# #             print(column_pair,"datatype-dis-similarity")

# final_output = pd.DataFrame(data=output_dict)
# final_output.head()
# final_output.to_csv("output.csv")
# stats, p_value = stats.kruskal(np.array(np.unique(df1[["component_value"]])), np.array(np.unique(df1[["component_value_str"]])))
# print(stats)
# if stats < 1.0:
#     print("similar")
# else:
#     print("dis-similar")
#print(column_pairs)
# for column_pair in column_pairs:
#     print(column_pair)
#     stats, p_value = stats.kruskal(df1[[column_pair[0]]], df2[[column_pair[1]]])
#     if stats < 1.0:
#         print(column_pair, "similar")
#     else:
#         print(column_pair, "dis-similar")

# Any results you write to the current directory are saved as output.