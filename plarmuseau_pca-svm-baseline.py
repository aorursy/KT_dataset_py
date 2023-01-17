import pandas as pd

import numpy as np

from sklearn.decomposition import PCA

from sklearn import svm

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.multiclass import OneVsRestClassifier



train = pd.read_csv("../input/Kannada-MNIST/train.csv")

test  = pd.read_csv("../input/Kannada-MNIST/test.csv")

submission  = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")





train_x = train.values[:,1:]

train_y = train.values[:,0]

test_x = test.values[:,1:]

totaal=np.concatenate((train_x,test_x),axis=0)

train_bwx=np.concatenate( (totaal,256-totaal),axis=1)

print(train_bwx.shape)



for xi in range (65,70,5):  #25 74%  65 99%

    pca = PCA(n_components=xi/100,whiten=True)

    trainp = pca.fit_transform(train_bwx) #train_x)

    #testp = pca.transform(test_bwx) #test_x)



    svc=OneVsRestClassifier(svm.SVC())

    #svc = svm.SVC(kernel='rbf',C=9)

    svc.fit(trainp[:len(train_x)], train_y)

    predt= svc.predict(trainp[:len(train_x)])

    print(xi,classification_report(train_y, predt) )

    #print('score',np.mean(predt,train_y))

    print(confusion_matrix(predt,train_y) )

        

preds = svc.predict(trainp[len(train_x):])

submission['label'] = preds

submission.to_csv('submission.csv', index=False)