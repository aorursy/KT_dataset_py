import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

data_dir = "../input/"

# load csv files to numpy arrays
def load_data(data_dir, train_row):
    train = pd.read_csv(data_dir + "train.csv")
    print(train.shape)
#     print(train.head())
#     print(train.shape)
#     print(train.head())
#     print(train[0:])
    X_train = train.values[0:train_row,1:] # 取下标为1-784的列（pixel0 - pixel783）
    y_train = train.values[0:train_row,0] # 取下标为0的列 (label)
    
    
    Pred_test = pd.read_csv(data_dir + "test.csv").values  
#     print(Pred_test.shape)
#     print(pd.read_csv(data_dir + "test.csv").head())
    return X_train, y_train, Pred_test

train_row = 5000    # partially take the data, the max = 42000
Origin_X_train, Origin_y_train, Origin_y_test = load_data(data_dir, train_row)


print(Origin_X_train.shape, Origin_y_train.shape, Origin_y_test.shape)
import matplotlib
import matplotlib.pyplot as plt
row = 3

# print (X_train[row].reshape((28, 28)))

print (Origin_y_train[row])

plt.imshow(Origin_X_train[row].reshape((28, 28)))
plt.show()
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
rows = 4

print(classes)
for y, cls in enumerate(classes):
    idxs = np.nonzero([i == y for i in Origin_y_train])
    idxs = np.random.choice(idxs[0], rows)
    for i , idx in enumerate(idxs):
        plt_idx = i * len(classes) + y + 1
        plt.subplot(rows, len(classes), plt_idx)
        plt.imshow(Origin_X_train[idx].reshape((28, 28)))
        plt.axis("off")
        if i == 0:
            plt.title(cls)
        

plt.show()
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(Origin_X_train,
                                                   Origin_y_train,
                                                   test_size = 0.2,
                                                   random_state = 0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.metrics import accuracy_score

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
        diffMat = np.tile(X,(dataSetSize,1)) - dataSet
        sqDiffMat = diffMat**2
        sumDiffMat = sqDiffMat.sum(axis=1)
        distances = sumDiffMat**0.5
        sortedDistances = distances.argsort()
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


classifier = knn()
classifier.train(X_train, y_train)


# clf = KNeighborsClassifier(n_neighbors=5)
# clf.fit(trainX, trainY)
# predictions = clf.predict(validX)
# accuracy = accuracy_score(validY, predictions)
# print("the accuracy of kNN is : %f" % accuracy)

max = 0
ans_k = 0

for k in range(1, 4):
    print ('when k = ' + str(k) + ', start training')
    predictions = np.zeros(len(y_test))
    for i in range(X_test.shape[0]):
        if i % 500 == 0:
            print("Computing  " + str(i+1) + "/" + str(int(len(X_test))) + "...")
        output = classifier.predict(X_test[i], i, k)
        predictions[i] = output
    
#     print(k, predictions)
#     predictions.shape
    accuracy = accuracy_score(y_test, predictions)
    print ('k = '+ str(k) , ' accuracy =' + str(accuracy))
    if max < accuracy:
        ans_k = k
        max = accuracy
        
    
print(y_test)
print(predictions)
k = 3
predictions = np.zeros(Origin_y_test.shape[0])
for i in range(Origin_y_test.shape[0]):
    if i % 500 ==0:
        print("Computing  " + str(i+1) + "/" + str(int(len(Origin_y_test))) + "...")
    predictions[i] = classifier.predict(Origin_y_test[i], i, k)

print (predictions[200])
plt.imshow(Origin_y_test[200].reshape((28, 28)))
print(len(predictions))
out_file = open("predictions.csv", "w")
out_file.write("ImageId,Label\n")
for i in range(len(predictions)):
    out_file.write(str(i+1) + "," + str(int(predictions[i])) + "\n")
out_file.close()
