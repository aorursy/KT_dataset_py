%%time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting
import seaborn as sns
%matplotlib inline

%%time
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.head(5)
train.shape
train.columns
sns.countplot(train['label'])
x_train=(train.ix[:,1:].values).astype('float32')
y_train=(train.ix[:,0].values).astype('int32')

# preview the images first
plt.figure(figsize=(12,10))
x,y=10,3
for i in range(30):  
    plt.subplot(y, x, i+1)
    plt.imshow(x_train[i].reshape((28,28)),interpolation='nearest')
plt.show()
test=(test.values).astype('float32')
test.shape

# preview the images first
plt.figure(figsize=(12,10))
x,y=10,3
for i in range(30):  
    plt.subplot(y, x, i+1)
    plt.imshow(test[i].reshape((28,28)),interpolation='nearest')
plt.show()
%%time
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier()
clf.fit(x_train, y_train)
output=clf.predict(test)
d={"ImageId": list(range(1,len(output)+1)),
                         "Label": output}

sub=pd.DataFrame(d)
sub.to_csv('Submission_digit.csv',index=False)
