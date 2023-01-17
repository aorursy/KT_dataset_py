import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
def load_data(data_dir):
    train_data=pd.read_csv(data_dir+'train.csv',header=0)
    test_data=pd.read_csv(data_dir+'test.csv', header=0)
    train_data=train_data.values
    test_data=test_data.values
    trainX_org=train_data[:,1:]
    trainY_org=train_data[:,0]
    return trainX_org, trainY_org, test_data
trainX_org, trainY_org, testX_org = load_data("../input/")
print(trainX_org.shape, trainY_org.shape,testX_org.shape)
nrows=10
cols=[i for i in range(0,10)]
for i in cols:
    sample_idx=np.nonzero(trainY_org==i)
    sample_idx=np.random.choice(sample_idx[0],nrows)
    for idx,s in enumerate(sample_idx):
        plt.subplot(nrows, len(cols), idx*len(cols)+i+1)
        plt.imshow(trainX_org[s,:].reshape(28,28))
        plt.axis("off")
        if idx==0:
            plt.title(str(i))
plt.show()
from sklearn.model_selection import train_test_split
trainX,valX,trainY,valY=train_test_split(trainX_org, trainY_org, test_size=0.2, random_state=1)
print(trainX.shape, trainY.shape, valX.shape, valY.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
krange=range(1,10)
scores=[]
for k in krange:
    print("starting k="+str(k)+":")
    start=time.time()
    knn=KNeighborsClassifier(n_neighbors=k)
    print("training.....")
    knn.fit(trainX, trainY)
    print("predicting.....")
    ypred=knn.predict(valX)
    scores.append(accuracy_score(valY, ypred))
    print(confusion_matrix(valY, ypred))
    print(classification_report(valY, ypred))
    print("done. Time= "+str(time.time()-start)+ "s.")
    
    
plt.plot(krange,scores)
plt.xlabel('k')
plt.ylabel('accuracy_scores')
plt.show()
k=3;
knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(trainX_org, trainY_org)
testy_pred=knn.predict(testX_org)
nr=10
nc=range(0,10)
for i in nc:
    idx=np.nonzero(testy_pred==i)
    idx=np.random.choice(idx[0],nr)
    for _i, _idx in enumerate(idx):
        plt.subplot(nr,len(nc),_i*len(nc)+i+1)
        plt.imshow(testX_org[_idx,:].reshape(28,28))
        plt.axis('off')
        if _i==0:
            plt.title(str(i))

plt.show()
dataout=pd.DataFrame({'ImageId':list(range(1,len(testy_pred)+1)), 'Label': testy_pred})
dataout.to_csv('Digital_Recogniser_Result.csv', index=False, header=True)
