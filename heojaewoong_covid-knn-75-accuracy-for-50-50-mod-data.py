import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
df = pd.read_excel('../input/mod-data/dataset.xlsx', sheet_name='All')
df.head()
df.groupby("SARS-Cov-2 exam result").count()
df['SARS-Cov-2 exam result'].value_counts().plot(kind='pie', autopct='%.2f%%')
# Will delete columns including object(text) value
df.dtypes
#Define the 'Y' array including target labels : positive = 1 , negative = 0

Y = df['SARS-Cov-2 exam result']
Y = np.array([1 if status=="positive" else 0 for status in Y])

#  One-hot encoding :  *specific columns* 
#  *specific columns*
#      = columns that consist of text values (such as 'detected', 'not detected')
#      = columns that indicate the disease-infection except COVID-19
# detected = [1,0,0], not_detected = [0,0,1], missing value = [0,1,0]


A = df['Respiratory Syncytial Virus']
A = np.array([(1,0,0) if status=="detected" else (0,0,1) if status == "not_detected" else (0,1,0) for status in A])


B = df['Influenza A']
B = np.array([(1,0,0) if status=="detected" else (0,0,1) if status == "not_detected" else (0,1,0) for status in B])

C = df['Influenza B']
C = np.array([(1,0,0) if status=="detected" else (0,0,1) if status == "not_detected" else (0,1,0) for status in C])

D = df['CoronavirusNL63']
D = np.array([(1,0,0) if status=="detected" else (0,0,1) if status == "not_detected" else (0,1,0) for status in D])


E = df['Rhinovirus/Enterovirus']
E = np.array([(1,0,0) if status=="detected" else (0,0,1) if status == "not_detected" else (0,1,0) for status in E])

F = df['Coronavirus HKU1']
F = np.array([(1,0,0) if status=="detected" else (0,0,1) if status == "not_detected" else (0,1,0) for status in F])


G = df['Parainfluenza 3']
G = np.array([(1,0,0) if status=="detected" else (0,0,1) if status == "not_detected" else (0,1,0) for status in G])


H = df['Adenovirus']
H = np.array([(1,0,0) if status=="detected" else (0,0,1) if status == "not_detected" else (0,1,0) for status in H])


I = df['Parainfluenza 4']
I = np.array([(1,0,0) if status=="detected" else (0,0,1) if status == "not_detected" else (0,1,0) for status in I])


J = df['Coronavirus229E']
J = np.array([(1,0,0) if status=="detected" else (0,0,1) if status == "not_detected" else (0,1,0) for status in J])


K = df['CoronavirusOC43']
K = np.array([(1,0,0) if status=="detected" else (0,0,1) if status == "not_detected" else (0,1,0) for status in K])


L = df['Inf A H1N1 2009']
L = np.array([(1,0,0) if status=="detected" else (0,0,1) if status == "not_detected" else (0,1,0) for status in L])


M = df['Metapneumovirus'] 
M = np.array([(1,0,0) if status=="detected" else (0,0,1) if status == "not_detected" else (0,1,0) for status in M])


N = df['Influenza B, rapid test']
N = np.array([(1,0,0) if status=="positive" else (0,0,1) if status == "negative" else (0,1,0) for status in N])

O = df['Influenza A, rapid test']
O = np.array([(1,0,0) if status=="positive" else (0,0,1) if status == "negative" else (0,1,0) for status in O])


P = df['Strepto A']
P = np.array([(1,0,0) if status=="positive" else (0,0,1) if status == "negative" else (0,1,0) for status in P])


# One-hot encoding causes feature-increase (curse of dimensionality)

A = np.reshape(A, (1065,3))
B = np.reshape(B, (1065,3))
C = np.reshape(C, (1065,3))
D = np.reshape(D, (1065,3))
E = np.reshape(E, (1065,3))
F = np.reshape(F, (1065,3))
G = np.reshape(G, (1065,3))
H = np.reshape(H, (1065,3))
I = np.reshape(I, (1065,3))
J = np.reshape(J, (1065,3))
K = np.reshape(K, (1065,3))
L = np.reshape(L, (1065,3))
M = np.reshape(M, (1065,3))
N = np.reshape(N, (1065,3))
O = np.reshape(O, (1065,3))
P = np.reshape(P, (1065,3))


# this numerated dataset will be combined with the training dataset (dataframe excluding object columns)

a = np.concatenate((A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P), axis = 1)
print(a.shape)
print(a)

# Remove useless column
# Remove Y dataset (target label/ target variable)

df = df.drop(columns=['Patient ID'])
df = df.drop(columns=['SARS-Cov-2 exam result'])
# Remove the columns & rows lack of enough data value

df = df.dropna(axis=1, how='all')   #Delete coloum when all of values are null
df = df.dropna(axis=0, how='all')   #Delete row when all of values are null
df = df.dropna(thresh=2)            #Maintain iff the row has at least 2 non-null value


# The number of features decreased (110  --> 104)
df.head()
# Remove the columes including 'object' (text values such as detected, not detected)
# will combine the 'df' with One-hot encoded dataset: the entire dataset consists of numeric data 

df = df.select_dtypes(exclude=['object'])
print(df.shape)
df.dtypes

# Replace the missing values with mean value (=0)

X = df
X = np.nan_to_num(X.to_numpy())

#X = df.to_numpy()
print(X.shape)

X = np.reshape(X, (1065,69))

print(X)

# Combine the one-hot encoded 'a' array with 'X' array 

X = np.concatenate((a, X), axis=1)
print(X.shape)
X.dtype
X
X.shape
#Split the dataframe into Training data and Test data 
#6:4 ratio results in the optimal performance 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=45)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_train)

# Reshape the patient data into rows
print(X_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0], -1))
print(X_train.shape)
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

print(X_train)
# Split training data into 3 groups of data 
# Each group of data has to be the validation dataset in order to ingnore outliers (for optimal k value)

num_folds = 3
k_choices = [1, 2, 3, 4, 5, 6,  7, 8,  9, 10, 11, 15, 19, 20, 30, 40,  50, 75, 100]

X_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}

for k in k_choices:
    k_to_accuracies[k] = []
    for i in range(num_folds): #split training data into 3 groups of data 
        # prepare training data for the current fold
        X_train_fold = np.concatenate([ fold for j, fold in enumerate(X_train_folds) if i != j ]) #merge except X_train_folds[i]
                                                                                                    #426 x 117
        y_train_fold = np.concatenate([ fold for j, fold in enumerate(y_train_folds) if i != j ])

        
        J = X_train_folds[i] #0th fold, 1st fold, 2nd fold
        num_X_train_folds = J.shape[0]              #validation/ test (213)
        num_X_train_fold = X_train_fold.shape[0]    #train            (426)
       

        # #i'th fold become the validation set/ calculate the distance values/ dists = 213 x 426
        dists = np.reshape(np.sum(X_train_folds[i]**2, axis=1), [num_X_train_folds,1]) + np.sum(X_train_fold**2, axis=1) \
            - 2 * np.matmul(X_train_folds[i], X_train_fold.T)
        dists = np.sqrt(dists)

        
            
        #num_val = dists.shape[0]  #213
        #y_pred = np.zeros(num_val) #213row coloum vector 0,1,0,1,..
        
       # U = y_train_fold    #validation_X dataset (213x117)
       
        num_val = dists.shape[0]
        y_pred = []
        y_pred = np.zeros(num_val)
        U = y_train_fold          #426
        for c in range(num_val): #213 
        
            
            closest_y = []
            closest_y = U[np.argsort(dists[c])][0:k] #returns indices of Y label mapped with the shortest distance value 

            
            y_pred[c] = np.bincount(closest_y).argmax() #what is the most drawn Y label value?
       
        
          
        num_correct = np.sum(y_pred  == y_train_folds[i]) 
        accuracy = float(num_correct) / num_val
          
        k_to_accuracies[k].append(accuracy)

                   
# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:

        print('k = %d, accuracy = %f' % (k, accuracy))              
q = [0.755869,0.723005,0.774648]
i = [0.680751,0.652582, 0.661972]
w = [0.638498,0.615023,0.605634]
z = [0.619718,0.600939,0.610329]
r = [0.638498, 0.615023,0.624413]
t = [0.704225,0.690141,0.657277]
x = [0.690141,0.643192,0.629108]
s = [0.56338, 0.549296,0.516432]

q = np.average(q)
i = np.average(i)
w = np.average(w)
z = np.average(z)
r = np.average(r)
t = np.average(t)
x = np.average(x)
s = np.average(s)

  #print('k = %d, accuracy = %f' % (k, accuracy))

print(q)
print('k = 2', 'accuracy = %f' %(q))
print('k = 3', 'accuracy = %f' %(i))
print('k = 5', 'accuracy = %f' %(w))
print('k = 7', 'accuracy = %f' %(z))
print('k = 15', 'accuracy = %f' %(r))
print('k = 30', 'accuracy = %f' %(t))
print('k = 50', 'accuracy = %f' %(x))
print('k = 100', 'accuracy = %f' %(s))

# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()
best_k = k_choices[accuracies_mean.argmax()]
print(best_k)


num_X_test = X_test.shape[0]    
num_X_train = X_train.shape[0]              

        #i'th fold become the validation set/ calculate the distance values
dists = np.reshape(np.sum(X_test**2, axis=1), [num_X_test,1]) + np.sum(X_train**2, axis=1) \
            - 2 * np.matmul(X_test, X_train.T)
dists = np.sqrt(dists)

        
            
num_test = dists.shape[0]
y_pred = []
y_pred = np.zeros(num_test)
U = y_train
for c in range(num_test):
    closest_y = []
    closest_y = U[np.argsort(dists[c])][0:best_k] 

            
    y_pred[c] = np.bincount(closest_y).argmax()
    
       
       
  
          
num_correct = np.sum(y_pred  == y_test) 
accuracy = float(num_correct) / num_test
          
k_to_accuracies[k].append(accuracy)


print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
