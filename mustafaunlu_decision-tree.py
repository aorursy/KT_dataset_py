import pandas as pd # verinin organizasyonu için

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate

from sklearn import svm

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier



#Grafik çizdirme kütüphanesi

import matplotlib.pyplot as plt



import os #Sistem 

import warnings #uyarılar

print(os.listdir("../input/"))

warnings.filterwarnings("ignore")
import pandas as pd

data = pd.read_csv("../input/cancer/data.csv", header =None)

data.head()
dataorj = data.copy()

feature_names = dataorj.drop([0,32,1],axis=1).values[0].tolist()



data = data.drop([0],axis=0)

Y=data[1].map({'M':1,'B':0})

data = data.drop([0,32,1],axis=1)





dataset = data.values

X = dataset[:,0:31]

X = X.astype('float64')
data.head()
for i in range(1,31):

    X[:,(i-1):i] = (X[:,(i-1):i]-np.min(X[:,(i-1):i]))/(np.max(X[:,(i-1):i])

    -np.min(X[:,(i-1):i]))
x_train, x_test, y_train, y_test = train_test_split(X, Y, 

      test_size=0.3, random_state=1)



data_name = "wisconsin"

def calculate2(cm):

    TP = cm[0,0]

    TN = cm[1,1]

    FP = cm[0,1]

    FN = cm[1,0]

    

    if TP+FN == 0:

        FN = 0.000001

    if TN+FP == 0:

        FP = 0.000001

    accuracy = (TP+TN)/(TP+TN+FP+FN)

    print("accuracy: ",accuracy)

    sensitivity = TP/(TP+FN)

    print("sensitivity: ",sensitivity)

    specificity = TN/(TN+FP)

    print("specificity: ",specificity)

    

    matrisim=[["accuracy: ",accuracy],["sensitivity: ",sensitivity],

          ["specificity: ",specificity]]

    return matrisim

    

def training(model,name,n1,n2,n3):

    print("--------------------")

    print("MODEL : ",str(name))

    print("--------------------")

    fig=plt.gcf()

    fig.set_size_inches(10,5)

    plt.subplot(n1,n2,n3)

    plt.title('train')

    model.fit(x_train,y_train)

    y_pred0=cross_val_predict(model,x_train,y_train,cv=10)

    cm=confusion_matrix(y_train,y_pred0)

    sns.heatmap(cm,annot=True,fmt="d")

        

    print("\naccuracy_score: "+str(metrics.accuracy_score(y_train, y_pred0)))

    print("*")

    plt.subplot(n1,n2,n3+1)

    plt.title('test')

    model.fit(x_test,y_test)

    y_pred00=cross_val_predict(model,x_test,y_test,cv=10)

    cm2=confusion_matrix(y_test,y_pred00)

    sns.heatmap(cm2,annot=True,fmt="d")



    plt.subplot(n1,n2,n3+2)

    plt.title('validation all')

    model.fit(X,Y)

    y_pred2=cross_val_predict(model,X,Y,cv=10)

    conf_mat2=confusion_matrix(Y,y_pred2)

    sns.heatmap(conf_mat2,annot=True,fmt="d")

    # plt.show()



    a=str(data_name)+str(name)+'.png'

    fig.savefig(a,dpi=100)





    cv1 = cross_validate(model, x_train, y_train, cv=10)

    cv2 = cross_validate(model, x_test, y_test, cv=10)

    cv3 = cross_validate(model, X, Y, cv=10)



    print('train '+str(name)+'accuracy is: ',cv1['test_score'].mean())

    print('test '+str(name)+' accuracy is: ',cv2['test_score'].mean())

    print('validation all'+str(name)+'accuracy is: ',cv3['test_score'].mean())

    print('')

    

    matris1 = calculate2(cm)

    matris2 = calculate2(cm2)

    matris3 = calculate2(conf_mat2)

    

    return matris1,matris2,matris3
randomforest=RandomForestClassifier(n_estimators=20,oob_score=False,random_state=43)

knn=KNeighborsClassifier(n_neighbors=4)

log_class=LogisticRegression()  

clf = DecisionTreeClassifier()
matris1,matris2,matris3=training(randomforest,"randomForest",4,3,1)

matris11,matris22,matris33=training(knn,"knn",4,3,4)

matris111,matris222,matris333=training(log_class,"logisticReg",4,3,7)

matris1111,matris2222,matris3333=training(clf,"clf",4,3,10)

plt.show()
import graphviz

from sklearn.tree import export_graphviz



targ_names = ['Yes','No']



data = export_graphviz(clf,out_file=None,feature_names=feature_names,class_names=targ_names,   

                         filled=True, rounded=True,  

                         special_characters=True)

graph = graphviz.Source(data)

graph