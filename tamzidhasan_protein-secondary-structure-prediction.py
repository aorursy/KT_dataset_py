import numpy as np # linear algebra
#find the path of input file 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#create container for data
primary_data = []
secondary_data = []

#providing data to the container from the input file (dataset)
with open('/kaggle/input/RS126.data.txt', 'r') as f:
    for count, line in enumerate(f, start=1):
        if count % 2 == 0:                           #Even line of dataset is secondary structure
            secondary_data.append(line.strip())
        else:                                        #Odd line of dataset is primary structure
            primary_data.append(line.strip())
primary_data  #primary structure data container
secondary_data     #secondary structure data container
for row in range(len(secondary_data)):
    secondary_lenth = len(secondary_data[row])
    primary_lenth = len(primary_data[row])
    
    if(secondary_lenth != primary_lenth):
        print("(",row,") Secondary_Structure ->", secondary_data[row]," Primary_Structure -> ",primary_data[row])
primary_data.pop(109)
secondary_data.pop(109)

secondary_count = 0
primary_count = 0
for row in range(len(secondary_data)):
    secondary_lenth = len(secondary_data[row])
    primary_lenth = len(primary_data[row])
    secondary_count = secondary_count + secondary_lenth
    primary_count = primary_count + primary_lenth
    if(secondary_lenth != primary_lenth):
        print("(",row,") Secondary_Structure ->", secondary_data[row]," Primary_Structure -> ",primary_data[row])
        
print("count of secondary structure : ",secondary_count)
print("count of primary structure : ",primary_count)
def split(sequence): 
    return [char for char in sequence]
primary_split = []
secondary_split = []
for row in range(len(secondary_data)):
    primary_split.append(split(primary_data[row]))
    secondary_split.append(split(secondary_data[row]))
    
#primary_split
def orthogonal_primary(arg):
    switch = {
        'A' : np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),  # 20 amino acids
        'C' : np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        'E' : np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        'D' : np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        'G' : np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        'F' : np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        'I' : np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        'H' : np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]),
        'K' : np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),
        'M' : np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),
        'L' : np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]),
        'N' : np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
        'Q' : np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]),
        'P' : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]),
        'S' : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]),
        'R' : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
        'T' : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]),
        'W' : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]),
        'V' : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]),
        'Y' : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
    }
    
    return switch.get(arg)

def orthogonal_secondary(arg):
    switch = {
        'H' : 0,                    # H= alpha helix
        'C' : 1,                    # C= coil
        'E' : 2                     # E= bita sheets
    }
    
    return switch.get(arg)
for row in range(len(primary_split)):  
    sequence = primary_split[row]
    for col in range(len(sequence)):
        #print(sequence[col])
        sequence[col] = orthogonal_primary(sequence[col])
        #print(sequence[col])
primary_split
for row in range(len(secondary_split)):  
    sequenceS = secondary_split[row]
    for col in range(len(sequenceS)):
        sequenceS[col] = orthogonal_secondary(sequenceS[col])
secondary_split
def target(data_list):
    Y = []
    for i in range(len(data_list)):
        for j  in range(len(data_list[i])):
            Y.append(data_list[i][j])
    return Y
y_label = target(secondary_split)
y_label
len(y_label)
def window_padding_data(size, sequence):
    num = int(size/2)
    #print("initial :",sequence[0])
    #print("")
    zeros = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for i in range(len(sequence)):
        for j in range(num):
            sequence[i].append(zeros)
            sequence[i].insert(0, zeros)
            #print(sequence[i])
            #print("")
            
    X = []
    temp = []

    for k in range(len(sequence)):
        #print(sequence[k])
        for l in range(len(sequence[k])-(size-1)):
            temp = sequence[k][l:l+size]
           # print(temp)
            X.append(temp)
            temp = []

    return X
X = window_padding_data(15,primary_split)
len(X)
X = np.array(X)
y_label = np.array(y_label)
X = X.reshape(22594, 15*20)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size = 0.20)
from sklearn.svm import SVC
from sklearn.metrics import classification_report
svc = SVC(kernel='rbf', gamma = 0.1, C=1.5)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
y_true = y_test

print(classification_report(y_true,y_pred))