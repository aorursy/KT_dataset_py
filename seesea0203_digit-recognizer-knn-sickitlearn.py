import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
data_dir = '../input/'
# load csv files to ndarray
def load_data(data_dir, train_row):
    train = pd.read_csv(data_dir + 'train.csv')
    print(train.shape)
    X_train = train.values[:train_row, 1:]
    y_train = train.values[:train_row, 0]
    
    X_test = pd.read_csv(data_dir + 'test.csv').values
    print(X_test.shape)
    return X_train, y_train, X_test

train_row = 5000 # If you want to use more traning data, increase value of train_row
Origin_Xtrain, Origin_ytrain, OriginXtest = load_data(data_dir, train_row)
    
%matplotlib inline
n = 3 # show the n th image

print(Origin_ytrain[n])
plt.imshow(Origin_Xtrain[n].reshape(28,28))
rows = 5 # Choose how many rows you want to show
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print(classes)

for y, c in enumerate(classes):
    idxs = np.random.choice(np.where([i == y for i in Origin_ytrain])[0], rows) # get the index where the row has the label - number 'y'
    for i, idx in enumerate(idxs):
        plt_idx = i * len(classes) + y + 1 # define the position of the image in subplot
        plt.subplot(rows, len(classes), plt_idx)
        plt.imshow(Origin_Xtrain[idx].reshape(28,28))
        plt.axis('off')
        if i == 0:
            plt.title(c)
        
    
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(Origin_Xtrain,
                                                 Origin_ytrain,
                                                 test_size=0.2,
                                                 random_state=0)
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

scores = []

for k in range(1,10):
    print('k = ' + str(k) + ' begin:')
    start = time.time()
    knn = KNeighborsClassifier(k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    scores.append(accuracy_score(y_val, y_pred))
    print('Accuracy: '+ str(scores[k-1]))
    end = time.time()
    print(classification_report(y_val, y_pred))
    print(confusion_matrix(y_val, y_pred))
    print('Time consuming: ' + str(end - start) + ' s.')
print(scores)
plt.plot(range(1,10), scores)
plt.xlabel('K-value')
plt.ylabel('Accuracy')
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(Origin_Xtrain, Origin_ytrain)
y_test = knn.predict(OriginXtest)
trows = 10 # Choose how many rows you want to show
print(classes)

for y, c in enumerate(classes):
    idxs = np.random.choice(np.where([i == y for i in y_test])[0], trows) # get the index where the row has the label - number 'y'
    for i, idx in enumerate(idxs):
        plt_idx = i * len(classes) + y + 1 # define the position of the image in subplot
        plt.subplot(trows, len(classes), plt_idx)
        plt.imshow(OriginXtest[idx].reshape(28,28))
        plt.axis('off')
        if i == 0:
            plt.title(c)

#pd.DataFrame(zip(list(range(1,len(y_test)+1)),y_test), columns=['ImageId', 'Label'])
pd.DataFrame({'ImageId': list(range(1, len(y_test)+1)), 'Label': y_test}).to_csv('Digit_Recogniser_Result.csv', index=False, header=True)