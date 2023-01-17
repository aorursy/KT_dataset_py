from sklearn import neighbors
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
#Dataset
data = pd.read_csv('../input/train_data.csv')
#X= np.array([['AAAAAAA'],['AAATTAA'], ['TTTTTTT'],['TTTAAAA'],['TTTTAAA'],['TATATAT']]) 
## class 0 if #A>#T, class 1 if #A<#T
#y= [0,0,1,0,1,1]
data.head()
# Example of string kernel
# compare poisition-wise 2 sequences(X & Y) and return similarity score.
def equal_elements(s1,s2):
    score = 0
    for i in range(len(s1)):
        score += (s1[i] == s2[i])*1 # This is an unoptimized way to do this. 
    return score

equal_elements("STRING","KERNEL")
clf = SVC(kernel=equal_elements)
clf.fit(data['DNA_sequence'],data['RPOD']) # this producecs an error
data.iloc[1,0]
# We use only a small fraction of the data for demonstration purposes,
# be aware that the equal_elements function is unoptimized and will 
# take some time if executed on the whole dataset

size = 12
not_so_good_string_kernel = np.zeros((size, size))
for row in range(size):
    for column in range(size):
        not_so_good_string_kernel[row,column] = equal_elements(data.iloc[row, 0],data.iloc[column, 0])
not_so_good_string_kernel
def compose_kernel(row_idxs, col_idxs):
    row_idxs = np.array(row_idxs).astype(np.int)
    col_idxs = np.array(col_idxs).astype(np.int)
    select_kernel = np.zeros((len(row_idxs),len(col_idxs)))
    for i, row_idx in enumerate(row_idxs):
        for j, col_idx in enumerate(col_idxs):
            select_kernel[i,j] = not_so_good_string_kernel[row_idx,col_idx]  # Change to custom distance kernel
    
    return select_kernel

compose_kernel([5,2,3,1],[5,2,3,1]) # random example
y = data['RPOD'].values
X_train_idx, X_test_idx, y_train, y_test = train_test_split(np.arange(size),y[:size], test_size=4) # OR USE KFoldStratified()
X_train_idx, X_test_idx, y_train, y_test
# KERNEL used for training
compose_kernel(X_train_idx, X_train_idx) # Distances between the training sequences
# KERNEL used for predictions
compose_kernel(X_train_idx, X_test_idx) # Distances between the training sequences and the testing sequences
clf= SVC(kernel=compose_kernel)
clf.fit(X_train_idx.reshape(-1,1), y_train) # reshape X_train_idx to be 2D
pred = clf.predict(X_test_idx.reshape(-1,1))
print(pred, y_test)