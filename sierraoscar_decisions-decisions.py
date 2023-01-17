import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

import pandas as pd



rng = np.random.RandomState(100)



datum = pd.read_csv("../input/creditcard.csv")

print("Data Shape is ",datum.shape)

data = datum.values

msec = int(data[:,0].max())
print("\nTotal listed seconds dataset is ", msec)

print("The total seconds in a day is 24 * 3600 = 86,400")

print("The data set will be split into day 1 and 2 due to classifier memory limitations")



d1 = np.where(data[:,0] == 86400); d2 = int(d1[0]); d3 = len(data)

print("Day 1 ends at index ",d2,"dataset is at index",d3)

data1 = data[0:d2]; data2 = data[d2:d3]



x2 = ([1, 6, 8, 13, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])

xy = x2
m1 = np.where(data1[:,30] == 0)

norm1 = data1[m1];

norm1 = np.delete(norm1,30,1)

norm1 = np.delete(norm1,xy,1)

y1 = norm1.shape[0]

norm1a = np.ones((y1,1), dtype = np.float16)



m2 = np.where(data2[:,30] == 0)

norm2 = data2[m2];

norm2 = np.delete(norm2,30,1)

norm2 = np.delete(norm2,xy,1)

y2 = norm2.shape[0]

norm2a = np.ones((y2,1), dtype = np.float16)



n1 = np.where(data1[:,30] == 1)

out1 = data1[n1];

out1 = np.delete(out1,30,1)

out1 = np.delete(out1,xy,1)

y1 = out1.shape[0]

out1a = np.ones((y1,1), dtype = np.float16)#set outliers output to 1 (not(0)

out1a = -1 * out1a#reset outliers output to -1



n2 = np.where(data2[:,30] == 1)

out2 = data2[n2];

out2 = np.delete(out2,30,1)

out2 = np.delete(out2,xy,1)

y2 = out2.shape[0]

out2a = np.ones((y2,1), dtype = np.float16)#set outliers output to 1 (not(0)

out2a = -1 * out2a#reset outliers output to -1



data1 = np.delete(data1,30,1); data1 = np.delete(data1,xy,1)

data2 = np.delete(data2,30,1); data2 = np.delete(data2,xy,1)



print("\nDay 1 transactions - Legitimate Transactions - Fraudulent Transactions")

print(data1.shape, norm1a.shape, out1a.shape)

print("Day 2 transactions - Legitimate Transactions - Fraudulent Transactions")

print(data2.shape, norm2a.shape, out2a.shape)
ct = 0.11
print("\nRunning Isolation Classifier with ", ct,"contamination")

clf = IsolationForest(max_samples=120000,  contamination= ct, random_state=rng)



clf.fit(data1)#2

nm1 = clf.predict(norm1); ot1 = clf.predict(out1)

nm1a = nm1[nm1 == -1].size; ot1a = ot1[ot1 == 1].size



clf.fit(data2)#2

nm2 = clf.predict(norm2); ot2 = clf.predict(out2)

nm2a = nm2[nm2 == -1].size; ot2a = ot2[ot2 == 1].size





print("\nDay 1 data")

print("Legitimate transactions - Fraudulent Transactions")

print(len(norm1), len(out1))

print("Legitimate transactions classed as fraudulent - ",nm1a)

print("Fraudulent transcations classed as legitimate ", ot1a)

print('\nLegitimate transactions accuracy is ',accuracy_score(nm1,norm1a))

print('Fraudulent transactions (Outliers) accuracy is ',accuracy_score(ot1,out1a))



print("\nDay 2 data")

print("Legitimate transactions - Fraudulent Transactions")

print(len(norm2), len(out2))

print("Legitimate transactions classed as fraudulent - ",nm2a)

print("Fraudulent transcations classed as legitimate ", ot2a)

print('\nLegitimate transactions accuracy is ',accuracy_score(nm2,norm2a))

print('Fraudulent transactions (Outliers) accuracy is ',accuracy_score(ot2,out2a))
ct = 0.011
print("\nRunning Isolation Classifier with ", ct,"contamination")

clf = IsolationForest(max_samples=120000,  contamination= ct, random_state=rng)



clf.fit(data1)#2

nm1 = clf.predict(norm1); ot1 = clf.predict(out1)

nm1a = nm1[nm1 == -1].size; ot1a = ot1[ot1 == 1].size



clf.fit(data2)#2

nm2 = clf.predict(norm2); ot2 = clf.predict(out2)

nm2a = nm2[nm2 == -1].size; ot2a = ot2[ot2 == 1].size



print("\nDay 1 data")

print("Legitimate transactions - Fraudulent Transactions")

print(len(norm1), len(out1))

print("Legitimate transactions classed as fraudulent - ",nm1a)

print("Fraudulent transcations classed as legitimate ", ot1a)

print('Legitimate transactions accuracy is ',accuracy_score(nm1,norm1a))

print('Fraudulent transactions (Outliers) accuracy is ',accuracy_score(ot1,out1a))



print("\nDay 2 data")

print("Legitimate transactions - Fraudulent Transactions")

print(len(norm2), len(out2))

print("Legitimate transactions classed as fraudulent - ",nm2a)

print("Fraudulent transcations classed as legitimate ", ot2a)

print('Legitimate transactions accuracy is ',accuracy_score(nm2,norm2a))

print('Fraudulent transactions (Outliers) accuracy is ',accuracy_score(ot2,out2a))

print("\nRandom Forest ")

ln = 0.75

ln1 = int(ln * len(data))

dattn = data[:ln1]

dattt = data[ln1:]



trainip = dattn[:,:-1]; trainop = dattn[:,-1]

testip = dattt[:,:-1]; testop = dattt[:,-1]



x1 = ([1, 6, 8, 13, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])



trainip = np.delete(trainip,x1,1)

testip = np.delete(testip,x1,1)

print("\nInitial dataset ", data.shape)

print("Features indices removed are ", x1)

print("Training set shape + Output, Test set shape + output")

print(trainip.shape, trainop.shape, testip.shape, testop.shape)



rf = RandomForestClassifier(max_features=10, max_depth=10,

                            min_samples_split=4,n_estimators=200) # initialize



#Locate the indices of test set that is non-fraud

x = np.where(testop[:] == 0)

iner = testip[x]

iner1 = np.zeros((len(iner),1), dtype = np.float16)

#print(len(x), len(iner), len(iner1))



#Locate the indices of test set that is fraudulent

x = np.where(testop[:] == 1)

outer = testip[x]

outer1 = np.ones((len(outer),1), dtype = np.float16)





rf.fit(trainip, trainop)#2

answ = rf.predict(testip)

print('Test set Accuracy is ',accuracy_score(testop,answ))



answn = rf.predict(iner)

print("\nTest set legitimate transactions is ", len(iner),"Accuracy is ",accuracy_score(iner1,answn))



answo = rf.predict(outer)

print("Test set fraudulent transactions (outliers) is ", len(outer),"Accuracy is ",accuracy_score(outer1,answo))
