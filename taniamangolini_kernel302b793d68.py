# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import networkx as nx
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# read the data file

df = pd.ExcelFile('/kaggle/input/covid19/dataset.xlsx').parse('All'); 
dfcpy = df.copy()
dfcpy = dfcpy[['SARS-Cov-2 exam result',
               'Hematocrit',
               'Hemoglobin',
               'Leukocytes', 
               'Lymphocytes', 
               'Neutrophils', 
               'Basophils', 
               'Monocytes',
               'Urea', 
               'Potassium', 
               'Sodium'          
              ]]
dfcpy.shape
dfcpy = dfcpy.dropna()
dfcpy.shape
dfcpy.describe()
df_positive = dfcpy[dfcpy['SARS-Cov-2 exam result'] == 'positive']
df_positive = df_positive[[ 'Hematocrit','Hemoglobin','Leukocytes', 
               'Lymphocytes', 'Neutrophils', 'Basophils', 'Monocytes',
               'Urea', 'Potassium', 'Sodium'
                ]]
df_positive = df_positive.dropna()
df_positive.head()
df_negative = dfcpy[dfcpy['SARS-Cov-2 exam result'] == 'negative']
df_negative = df_negative[[ 'Hematocrit','Hemoglobin','Leukocytes', 
               'Lymphocytes', 'Neutrophils', 'Basophils', 'Monocytes',
               'Urea', 'Potassium', 'Sodium'
                ]]
df_negative = df_negative.dropna()
df_negative.head()
#standardization
scaler = StandardScaler()
std_data_pos= scaler.fit_transform(df_positive.values)
std_df_positive = pd.DataFrame(std_data_pos, index=df_positive.index, columns=df_positive.columns)
std_df_positive.head()
#standardization
scaler = StandardScaler()
std_data = scaler.fit_transform(df_negative.values)
std_df_negative = pd.DataFrame(std_data, index=df_negative.index, columns=df_negative.columns)
std_df_negative.head()
#Hematocrit
positives = std_df_positive['Hematocrit']
negatives = std_df_negative['Hematocrit']

plt.hist([positives, negatives], label=['positives', 'negatives'],stacked=True)
plt.legend(loc='upper right')
plt.show()
#Hemoglobin
positives = std_df_positive['Hemoglobin']
negatives = std_df_negative['Hemoglobin']

plt.hist([positives, negatives], label=['positives', 'negatives'], stacked=True)
plt.legend(loc='upper right')
plt.show()
#Leukocytes
positives = std_df_positive['Leukocytes']
negatives = std_df_negative['Leukocytes']

plt.hist([positives, negatives], label=['positives', 'negatives'], stacked=True)
plt.legend(loc='upper right')
plt.show()
#Lymphocytes
positives = std_df_positive['Lymphocytes']
negatives = std_df_negative['Lymphocytes']

plt.hist([positives, negatives], label=['positives', 'negatives'], stacked=True)
plt.legend(loc='upper right')
plt.show()
#Neutrophils
positives = std_df_positive['Neutrophils']
negatives = std_df_negative['Neutrophils']

plt.hist([positives, negatives], label=['positives', 'negatives'], stacked=True)
plt.legend(loc='upper right')
plt.show()
#Basophils
positives = std_df_positive['Basophils']
negatives = std_df_negative['Basophils']

plt.hist([positives, negatives], label=['positives', 'negatives'], stacked=True)
plt.legend(loc='upper right')
plt.show()
#Monocytes
positives = std_df_positive['Monocytes']
negatives = std_df_negative['Monocytes']

plt.hist([positives, negatives], label=['positives', 'negatives'], stacked=True)
plt.legend(loc='upper right')
plt.show()
#Urea
positives = std_df_positive['Urea']
negatives = std_df_negative['Urea']

plt.hist([positives, negatives], label=['positives', 'negatives'], stacked=True)
plt.legend(loc='upper right')
plt.show()
#Potassium
positives = std_df_positive['Potassium']
negatives = std_df_negative['Potassium']

plt.hist([positives, negatives], label=['positives', 'negatives'], stacked=True)
plt.legend(loc='upper right')
plt.show()
#Sodium
positives = std_df_positive['Sodium']
negatives = std_df_negative['Sodium']

plt.hist([positives, negatives], label=['positives', 'negatives'], stacked=True)
plt.legend(loc='upper right')
plt.show()
# CREATING AN ANALYSIS BASED ON DISTANCES FOR THE  VARIABLES:
selected_variables = ['Hematocrit','Hemoglobin','Leukocytes','Lymphocytes','Neutrophils','Basophils','Monocytes','Urea','Potassium','Sodium']

positives = std_df_positive[selected_variables]
negatives = std_df_negative[selected_variables]

distance_matrix_positives = pd.DataFrame(
    squareform(pdist(positives, metric='euclidean')),
    columns = positives.index,
    index = positives.index
)

distance_matrix_negatives = pd.DataFrame(
    squareform(pdist(negatives, metric='euclidean')),
    columns = negatives.index,
    index = negatives.index
)

distance_matrix_negatives.describe()
def get_mean_of_quartil(matrix, quartil):
    sum_q = 0
    mean = 0
    countRows = 0 
    
    for column in matrix.columns:
        sum_q  += matrix[column].quantile(quartil)
        countRows += 1
    
    mean = sum_q/countRows
    return mean

q1_mean_negative = get_mean_of_quartil(distance_matrix_negatives, 0.25)
q1_mean_negative
q3_mean_negative  = get_mean_of_quartil(distance_matrix_negatives, 0.75)
q3_mean_negative 
q1_mean_positive = get_mean_of_quartil(distance_matrix_positives, 0.25)
q1_mean_positive
q3_mean_positive = get_mean_of_quartil(distance_matrix_positives, 0.75)
q3_mean_positive
#aDJACENCY MATRIX

adj_matrix_negatives = distance_matrix_negatives.applymap(lambda x: 1 if x >= q1_mean_negative and x <= q3_mean_negative  else 0)
adj_matrix_positives = distance_matrix_positives.applymap(lambda x: 1 if x >= q1_mean_negative and x <= q3_mean_negative  else 0)
#Defining functions to get graph nodes and edges 

def get_nodes(matrix):
    nodes = []
    for i in range(len(matrix.values)):
            nodes.append((i))
    return nodes

def get_edges(matrix):
    edges = []
    for i in range(len(matrix.values)):
        for j in range(len(matrix.values)):
            if(i != j and matrix.values[i][j] == 1):
                edges.append((i,j))
    return edges
#GRAPH POSITIVES (GREEN) AND NEGATIVES (RED)

G = nx.Graph() 
plt.figure(figsize =(20, 30)) 

G.add_nodes_from(get_nodes(adj_matrix_positives)) 
G.add_edges_from(get_edges(adj_matrix_positives)) 

G.add_nodes_from(get_nodes(adj_matrix_negatives)) 
G.add_edges_from(get_edges(adj_matrix_negatives)) 

plt.subplot(311) 
nx.draw_networkx(G)

#AS IS POSSIBLE TO SEE BELOW, IT WAS NOT FOUND EXPRESSIVE DIFFERENCES AMONG POSITIVE AND NEGATIVE GROUP FOR THE CONSIDERED VARIABLES
