import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
traindata = pd.read_csv("../input/train.csv")
testdata = pd.read_csv("../input/test.csv")
traindata.head(10)
testdata.head(10)
from sklearn.tree import DecisionTreeClassifier
traindata.shape
x = traindata.loc[0] #Locating the column headers
x.shape
np.sqrt(785-1) #Splitting the colums by 28,28. '-1' corresponds excluding the index column
#Training Dataset
traindata = pd.read_csv("../input/train.csv").as_matrix()
clf = DecisionTreeClassifier()

xtrain = traindata[0:21000,1:]
train_label=traindata[0:21000,0]

clf.fit(xtrain,train_label)

#Testing Dataset
xtest = traindata[21000:,1:]
actual_label = traindata[21000:,0]
d = xtest[5]
d.shape = (28,28)
plt.imshow(255-d,cmap= 'gray')
plt.show()
plt.imshow(255-d,cmap= 'gray')
print (clf.predict( [xtest[5]] ))
plt.show()
p=clf.predict(xtest)

count=0
for i in range(0,21000):
    count+=1 if p[i]==actual_label[i] else 0
print("Accuracy=", (count/21000)*100)    
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
%matplotlib inline
data = pd.read_csv("../input/train.csv")
data.head()
a = data.iloc[0,1:].values #consider the 0th row with all the columns from pixel0 to the end.
#Lets reshape it by 28,28 2D matrices and the type must be uint8 for plotting
a = a.reshape(28,28).astype('uint8')
plt.imshow(a) #--->for 0th row, the label is 1
#similarly, let change the value of a.
a = data.iloc[3,1:].values
a = a.reshape(28,28).astype('uint8')
plt.imshow(a)
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y,test_size=0.2, random_state=42)
print("x_train",x_train)
print("x_test",x_test)
print("y_train",y_train)
print("y_test",y_test)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)
pred =rf.predict(x_test)
pred
count=0
s = y_test.values
s
for i in range(len(pred)):
     if pred[i]==s[i]:
        count=count+1
count

len(pred)
(count/len(pred)) * 100
print('accuracy of training set: {}'.format(rf.score(x_train, y_train)))
print('accuracy of validation set: {}'.format(rf.score(x_test,  y_test)))
cross_val_score(rf, x_train, y_train)
submission = pd.DataFrame()
submission['Label'] = rf.predict(x_test)
submission.index += 1
submission.index.name = 'ImageId'
submission.to_csv('./submission.csv')