import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

print(os.listdir("../input/leapgestrecog/leapGestRecog"))



# Any results you write to the current directory are saved as output.
import os

import numpy as np

from PIL import Image

import matplotlib.image as mpimg 

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import IPython.display

path='../input/leapgestrecog/leapGestRecog'

folders=os.listdir(path)

folders=set(folders)



import codecs

import json





different_classes=os.listdir(path+'/'+'00')

different_classes=set(different_classes)









print("The different classes that exist in this dataset are:")

print(different_classes,sep='\n')
x=[]

z=[]

y=[]#converting the image to black and white

threshold=200

import cv2





for i in folders:

    print('***',i,'***')

    subject=path+'/'+i

    subdir=os.listdir(subject)

    subdir=set(subdir)

    for j in subdir:

        print(j)

        images=os.listdir(subject+'/'+j)

        for k in images:

            results=dict()

            results['y']=j.split('_')[0]

            img = cv2.imread(subject+'/'+j+'/'+k,0)

            img=cv2.resize(img,(int(160),int(60)))

            

            ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            imgD=np.asarray(img,dtype=np.float64)

            z.append(imgD)

            imgf=np.asarray(imgf,dtype=np.float64)

            x.append(imgf)

            y.append(int(j.split('_')[0]))

            results['x']=imgf



print(list(set(y)))

        

# import pandas as pd 

# df=pd.DataFrame({'x':x,'y':y})

# df.to_csv('results.csv',index=False)
#sample black and white image from each class

l = []

list_names = []

for i in range(10):

    l.append(0)

for i in range(len(x)):

    if(l[y[i] - 1] == 0):

        l[y[i] - 1] = i

        if(len(np.unique(l)) == 10):

            break

for i in range(len(l)):

    %matplotlib inline

    print("Class Label: " + str(i + 1))

    plt.imshow(np.asarray(z[l[i]]), cmap  =cm.gray)

    plt.show()

    plt.imshow(np.asarray(x[l[i]]), cmap = cm.gray)     

    plt.show()
x=np.array(x)

y=np.array(y)

y = y.reshape(len(x), 1)

print(x.shape)

print(y.shape)

print(max(y),min(y))
x_data = x.reshape((len(x), 60, 160, 1))



x_data/=255

x_data=list(x_data)

for i in range(len(x_data)):

    x_data[i]=x_data[i].flatten()
from sklearn.decomposition import PCA

pca = PCA(n_components=4)

x_data=np.array(x_data)

x_data=pca.fit_transform(x_data)

print(pca.explained_variance_ratio_)  

print(pca.singular_values_)  
from sklearn.model_selection import train_test_split

x_train,x_further,y_train,y_further = train_test_split(x_data,y,test_size = 0.2)

from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()  

scaler.fit(x_train)



X_train = scaler.transform(x_train)  

X_test = scaler.transform(x_further)  
from sklearn.linear_model import SGDClassifier

sgd= SGDClassifier(loss='modified_huber',shuffle=True,random_state=101)

sgd.fit(X_train, y_train)

y_pred_sgd = sgd.predict(X_test) 

y_train_score_sgd=sgd.predict(X_train)

from sklearn.metrics import accuracy_score

print("accuracy of the model is:\nTest ", accuracy_score(y_further, y_pred_sgd, normalize=True, sample_weight=None))

print('Train',accuracy_score(y_train, y_train_score_sgd, normalize=True, sample_weight=None))
from sklearn.neighbors import KNeighborsClassifier  

classifier = KNeighborsClassifier(n_neighbors=5)  

classifier.fit(X_train, y_train)  

y_pred_knn = classifier.predict(X_test)  
y_train_score_knn=classifier.predict(X_train)
from sklearn.metrics import accuracy_score

print("accuracy of the model is:\nTest ", accuracy_score(y_further, y_pred_knn, normalize=True, sample_weight=None))

print('Train',accuracy_score(y_train, y_train_score_knn, normalize=True, sample_weight=None))
from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=10)

clf = clf.fit(X_train, y_train)
y_pred_dt=clf.predict(X_test)

y_train_score_dt=clf.predict(X_train)
print("accuracy of the model is:\nTest ", accuracy_score(y_further, y_pred_dt, normalize=True, sample_weight=None))

print('Train',accuracy_score(y_train, y_train_score_dt, normalize=True, sample_weight=None))
from sklearn.ensemble import RandomForestClassifier

clf_rft = RandomForestClassifier(n_estimators=100, max_depth=15,random_state=0)

clf_rft = clf_rft.fit(X_train, y_train)

y_pred_rft=clf_rft.predict(X_test)

y_train_score_rft=clf_rft.predict(X_train)
print("accuracy of the model is:\nTest ", accuracy_score(y_further, y_pred_rft, normalize=True, sample_weight=None))

print('Train',accuracy_score(y_train, y_train_score_rft, normalize=True, sample_weight=None))




from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression(solver = 'lbfgs')

logistic.fit(X_train, y_train)

y_pred_logistic=logistic.predict(X_test)

y_train_score_logistic=logistic.predict(X_train)

print("accuracy of the model is:\nTest ", accuracy_score(y_further, y_pred_logistic, normalize=True, sample_weight=None))

print('Train',accuracy_score(y_train, y_train_score_logistic, normalize=True, sample_weight=None))
from sklearn.naive_bayes import GaussianNB 

gnb = GaussianNB() 

gnb.fit(X_train, y_train)

y_pred_gnb=gnb.predict(X_test)

y_train_score_gnb=gnb.predict(X_train)

print("accuracy of the model is:\nTest ", accuracy_score(y_further, y_pred_gnb, normalize=True, sample_weight=None))

print('Train',accuracy_score(y_train, y_train_score_gnb, normalize=True, sample_weight=None))
from sklearn.neural_network import MLPClassifier



ann_clf = MLPClassifier()

ann_clf.fit(X_train, y_train)

y_pred_ann=ann_clf.predict(X_test)

y_train_score_ann=ann_clf.predict(X_train)

print("accuracy of the model is:\nTest ", accuracy_score(y_further, y_pred_ann, normalize=True, sample_weight=None))

print('Train',accuracy_score(y_train, y_train_score_ann, normalize=True, sample_weight=None))
from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingClassifier

gdc_model= GradientBoostingClassifier(learning_rate=0.01,random_state=1)

gdc_model.fit(x_train, y_train)

y_pred_gdc=gdc_model.predict(X_test)

y_train_score_gdc=gdc_model.predict(X_train)

print("accuracy of the model is:\nTest ", accuracy_score(y_further, y_pred_gdc, normalize=True, sample_weight=None))

print('Train',accuracy_score(y_train, y_train_score_gdc, normalize=True, sample_weight=None))
from sklearn.svm import SVC 

svm_model_rbf = SVC(kernel = 'rbf', C = 10,probability=True).fit(X_train, y_train) 

y_pred_svm=svm_model_rbf.predict(X_test)

y_train_score_svm=svm_model_rbf.predict(X_train)

print("accuracy of the model is:\nTest ", accuracy_score(y_further, y_pred_svm, normalize=True, sample_weight=None))

print('Train',accuracy_score(y_train, y_train_score_svm, normalize=True, sample_weight=None))
from sklearn.ensemble import VotingClassifier

model = VotingClassifier(estimators=[('logistic',logistic),('nb',gnb),('gdc',gdc_model),('ann',ann_clf),('clf_rft',clf_rft),('dt',clf),('sv',svm_model_rbf),('knn',classifier),('stochastic',sgd)],voting='soft')

model.fit(X_train, y_train)

y_pred=model.predict(X_test)

y_train_score=model.predict(X_train)

print("accuracy of the model is:\nTest ", accuracy_score(y_further, y_pred, normalize=True, sample_weight=None))

print('Train',accuracy_score(y_train, y_train_score, normalize=True, sample_weight=None))
stacking_xtest=[[y_pred_svm[i],y_pred[i],y_pred_ann[i],y_pred_rft[i],y_pred_dt[i],y_pred_knn[i],y_pred_sgd[i],y_pred_logistic[i],y_pred_gnb[i],y_pred_gdc[i]] for i in range(len(X_test))]

stacking_xtrain=[[y_train_score_svm[i],y_train_score[i],y_train_score_ann[i],y_train_score_rft[i],y_train_score_dt[i],y_train_score_knn[i],y_train_score_sgd[i],y_train_score_logistic[i],y_train_score_gnb[i],y_train_score_gdc[i]] for i in range(len(X_train))]





ann_stacking = MLPClassifier()

ann_stacking.fit(stacking_xtrain, y_train)

y_pred_stacking=ann_stacking.predict(stacking_xtest)

y_train_score_stacking=ann_stacking.predict(stacking_xtrain)

print("accuracy of the model is:\nTest ", accuracy_score(y_further, y_pred_stacking, normalize=True, sample_weight=None))

print('Train',accuracy_score(y_train, y_train_score_stacking, normalize=True, sample_weight=None))



names=['Stochastic Gradient Classifier','K Nearest Neighbour','Decision Tree','Random Forest Tree','Logistic Regression','Naive Bayes','Artificial Neural Network','Gradient Descent Classifier','Support Vector Machine','Voting','Stacking']

all_models=[y_pred_sgd,y_pred_knn,y_pred_dt,y_pred_rft,y_pred_logistic,y_pred_gnb,y_pred_ann,y_pred_gdc,y_pred_svm,y_pred,y_pred_stacking]

all_training=[y_train_score_sgd,y_train_score_knn,y_train_score_dt,y_train_score_rft,y_train_score_logistic,y_train_score_gnb]

all_training+=[y_train_score_ann,y_train_score_gdc,y_train_score_svm,y_train_score,y_train_score_stacking]

testing_accuracy=[]

training_accuracy=[]

for i in all_models:

    testing_accuracy.append(accuracy_score(y_further, i, normalize=True, sample_weight=None))

for i in all_training:

    training_accuracy.append(accuracy_score(y_train, i, normalize=True, sample_weight=None))

import pandas as pd

df=pd.DataFrame({'Names':names,'Training Accuracy':training_accuracy,'Testing Accuracy':testing_accuracy})

df=df[['Names','Training Accuracy','Testing Accuracy']]

df