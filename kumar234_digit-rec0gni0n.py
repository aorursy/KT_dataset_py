import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import seaborn as sns
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()
train.shape
test.shape
train.describe()
train.tail()
test.head()
train.info()
train.label.value_counts()
train['label'[:30]]
plt.hist(train['label'],color='blue')
plt.title("Frequency of no. of data")
plt.xlabel("Numbers")
plt.ylabel("Frequency")
plt.show()
label_train = train['label']
train  = train.drop('label',axis=1)
train.head()
#data normalisssation

train = train/255
test = test/255
from sklearn.cross_validation import train_test_split
X_train ,X_test ,y_train ,y_test = train_test_split(train , label_train,train_size=0.8,random_state=42)
from sklearn import decomposition
##PCA 
pca = decomposition.PCA(n_components=200)   # find first 200PCs
pca.fit(X_train)
plt.plot(pca.explained_variance_ratio_)
plt.ylabel('% of variance explained')
plt.show()


#plot reaches asymptote at around 100, which is optimal number of PCs to use

## PCA decomposition with optimal number of PCs
#decompose train data
pca = decomposition.PCA(n_components=100)
pca.fit(X_train)

PCtrain = pca.transform(X_train)
PCtest = pca.transform(X_test)

#decompose test data
PCtest = pca.transform(test)
X_train = PCtrain
X_test = PCtest
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn import svm, metrics
import csv
clf = SVC()
clf.fit(X_train,y_train)
predicted = clf.predict(X_test)
expected = y_test
print(predicted[0:30])
output_label = clf.predict(PCtest)
print(predicted)
print(expected[:30])
output = pd.DataFrame(output_label,columns = ['Label'])
output.reset_index(inplace=True)
output['index'] = output['index'] + 1
output.rename(columns={'index': 'ImageId'}, inplace=True)
output.to_csv('output_digit.csv', index=False)
output.head()