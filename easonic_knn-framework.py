import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

import matplotlib.pyplot as plt # 画图常用库



import time



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

%matplotlib inline
def load_data_set():

    train = pd.read_csv('../input/train.csv')

    print('Train set shape: {}'.format(train.shape))

    test = pd.read_csv('../input/test.csv')

    print('Test set shape: {}'.format(test.shape))

    return train, test

    

train, test = load_data_set()
num_row2train = int(np.ceil(train.shape[0]/10000)*10000/10 if train.shape[0]>40000 else train.shape[0]>40000)

print(num_row2train)

X_train = train.values[:num_row2train, 1:]

y_train = train.values[:num_row2train, 0]



n = int(np.random.randint(num_row2train, size=1))

# print(n, type(n))

plt.title(y_train[n])

plt.imshow(X_train[n].reshape(28, 28))



y_train
nums = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

print(nums)



num_row = 5



plt.figure(figsize=(12,6))

for y, num in enumerate(nums):

    idxs = np.nonzero([i == y for i in y_train])

    idxs = np.random.choice(idxs[0], num_row)

    for i , idx in enumerate(idxs):

        plt_idx = i * len(nums) + y + 1

        plt.subplot(num_row, len(nums), plt_idx)

        plt.imshow(X_train[idx].reshape((28, 28)))

        plt.axis("off")

        if i == 0:

            plt.title(num)

    
xtrain, xtest, ytrain, ytest = train_test_split(X_train, y_train, test_size=0.2 ,random_state=42)

print('xtrain.shape{}, xtest.shape{}, ytrain.shape{}, ytest.shape{}'.format(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))
k_range = range(1, 9)

scores = []



# 这个地方通过枚举所有的k值来取找到最好的k值预测数据

for k in k_range:

    print("k{} started:".format(k))

    start = time.time()

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(xtrain,ytrain)

    ypred = knn.predict(xtest)

    accuracy = accuracy_score(ytest,ypred)

    scores.append(accuracy)

    end = time.time()

    print(classification_report(ytest, ypred))  

    print(confusion_matrix(ytest, ypred))  

    

    print("k{} completed, duration:{}".format(k, end-start))
plt.plot(k_range, scores)
knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train,y_train)

y_pred = knn.predict(test[:300])
print (y_pred[0:15])



plt.figure(figsize=(12,12))

for i in range(0,16):

    plt.subplot(4, 4, i+1)

    plt.title('Predit: {}'.format(y_pred[i]))

    plt.imshow(test.values[i].reshape((28, 28)))

    plt.axis('off')
print(len(ypred))

pd.DataFrame({"ImageId": list(range(1,len(ypred)+1)),"Label": ypred}).to_csv('Digit_Recogniser_Result.csv', index=False,header=True)
rslt=pd.read_csv('Digit_Recogniser_Result.csv',index_col=False)

rslt.head()