import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

data_dir = '../input/'
trainrow =500
def load_data(data_dir,trainrow):
    train = pd.read_csv(data_dir+'train.csv')
    print(train.shape)
    test = pd.read_csv(data_dir+'test.csv').values
    origin_X_train = train.values[0:trainrow,1:]
    origin_y_train = train.values[0:trainrow, 0]
    return origin_X_train,origin_y_train,test
origin_X_train,origin_y_train,test = load_data(data_dir,trainrow)

row = 8 
print(origin_y_train[row])
plt.imshow(origin_X_train[row].reshape((28,28)))
plt.show()
from sklearn.model_selection import train_test_split
X_train,X_vali,y_train, y_vali = train_test_split(origin_X_train,
                                                 origin_y_train,
                                                 test_size= 0.2,
                                                 random_state=0)
X_train.shape,X_vali.shape,y_train.shape, y_vali.shape
class knn():
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, num, k=3):
        dataSet = X_train
        labels = y_train
      
        dataSetSize = dataSet.shape[0]
        
        # np.tile: 重复数组若干次
        # a = np.array([0, 1, 2])
        # np.tile(a, (2, 2))
        # array([[0, 1, 2, 0, 1, 2],
        #        [0, 1, 2, 0, 1, 2]])

        diffMat = np.tile(X,(dataSetSize,1)) - dataSet
        sqDiffMat = diffMat**2
        sumDiffMat = sqDiffMat.sum(axis=1)
        distances = sumDiffMat**0.5
        sortedDistances = distances.argsort()
        
        # np.argsort: return idx of elements after sorting in ascending order
        # x = np.array([3, 1, 2])
        # np.argsort(x)
        # array([1, 2, 0])
        
        classCount = {}
        
        for i in range(k):
            vote = labels[sortedDistances[i]]
            classCount[vote] = classCount.get(vote,0) + 1
        max = 0
        ans = 0
        for k,v in classCount.items():
            if(v>max):
                ans = k
                max = v
#         print("test #"+ str(num+1) + " prediction is " + str(ans)
        return(ans)
#argsort
a=np.array([1,10,4,6,74,3,2,4,6,43,2])
np.argsort(a)
from sklearn.metrics import accuracy_score

classifier = knn()
classifier.train(X_train, y_train)
from sklearn.metrics import accuracy_score

classifier = knn()
classifier.train(X_train, y_train)

max = 0
ans_k = 0

for k in range(1, 4):
    print ('when k = ' + str(k) + ', start training')
    predictions = np.zeros(len(y_vali))
    for i in range(X_vali.shape[0]):
        if i % 500 == 0:
            print("Computing  " + str(i+1) + "/" + str(int(len(X_vali))) + "...")
        output = classifier.predict(X_vali[i], i, k)
        predictions[i] = output
    
#     print(k, predictions)
#     predictions.shape
    accuracy = accuracy_score(y_vali, predictions)
    print ('k = '+ str(k) , ' accuracy =' + str(accuracy))
    if max < accuracy:
        ans_k = k
        max = accuracy
k = 3
Origin_y_test = test[:300] # remove this line for full test
predictions = np.zeros(Origin_y_test.shape[0])
for i in tqdm(range(Origin_y_test.shape[0])):
    predictions[i] = classifier.predict(Origin_y_test[i], i, k)
row =1
print(predictions[row])
plt.imshow(Origin_y_test[row].reshape((28,28)))
print(len(predictions))
out_file=open("predictions.csv","w")
out_file.write("ImageID, label\n")
for i in range(len(predictions)):
    out_file.write(str(i+1)+","+"str(int(predictions[i]))+\n")
out_file.close()
