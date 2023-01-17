from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784',version=1)  
import numpy as np

from sklearn.model_selection import train_test_split



X,y = mnist["data"],mnist["target"].astype(np.int) 

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=50000)



# For quick calculation, reduce the size of data set

X_train, y_train = X_train[:5000], y_train[:5000]

X_test, y_test = X_train[:2500], y_train[:2500]



print("X_train :",len(X_train))

print("y_train :",len(y_train))

print("X_test  :",len(X_test))

print("y_test  :",len(y_test))
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt





def report(model) :

    y_pred     = cross_val_predict(model, X_test, y_test, cv=3)     

    clf_report = classification_report(y_test,y_pred, output_dict =True)

    accuracy   = clf_report["accuracy"]                # accuracy

    precision  = clf_report['macro avg']['precision']  # precision

    recall     = clf_report['macro avg']['recall']     # recall

    confusion  = confusion_matrix(y_test, y_pred)      # confusion_matrix

    print("accuracy : ",accuracy)

    print("precision :", precision)

    print("recall :",recall)

    print("Confusion Matrix :\n"+str(confusion))

    

    

def show_auc(y_true,y_score):

    fpr, tpr, _ = roc_curve(y_true, y_score)

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')    

    plt.legend(loc="lower right")



    

def Pretreatment_ROC(model, X, y):

    y_prob = model.predict_proba(X)

    y_pred = cross_val_predict(model, X, y)

    y_true = np.array(y == y_pred)

    y_score = np.array([y_prob[i][int(y[i])] for i in range(len(y))])

    return y_true, y_score





def Print_clf(clf):

    clf.fit(X_train,y_train)

    report(clf)



    y_true, y_score = Pretreatment_ROC(clf,X_test,y_test)

    show_auc(y_true,y_score)
from sklearn.neighbors    import KNeighborsClassifier

from sklearn.ensemble     import ExtraTreesClassifier

from sklearn.ensemble     import RandomForestClassifier

from sklearn.ensemble     import BaggingClassifier

from sklearn.ensemble     import AdaBoostClassifier

from sklearn.ensemble     import VotingClassifier



rnd = 1234 #random_state



knn_clf = KNeighborsClassifier(n_neighbors=15)

ext_clf = ExtraTreesClassifier(n_estimators=20,random_state=rnd)

rdf_clf = RandomForestClassifier(n_estimators=10, random_state=rnd)



bag_clf = BaggingClassifier(

    ExtraTreesClassifier(n_estimators=20,random_state=rnd), 

    n_jobs=-1,

    n_estimators=5,

    random_state=12

)



ada_clf = AdaBoostClassifier(

    ExtraTreesClassifier(n_estimators=20,random_state=rnd),

    n_estimators=50,

    learning_rate=0.2, 

    algorithm="SAMME.R", 

    random_state=12

)





vot_clf = VotingClassifier(

    estimators= [        

        ("ext_clf",ext_clf),

        ("rdf_clf",rdf_clf),

        ("bag_clf",bag_clf),

        ("ada_clf",ada_clf)

    ] , voting='soft'

)
Print_clf(knn_clf)
Print_clf(ext_clf)
Print_clf(rdf_clf)
Print_clf(bag_clf)
Print_clf(ada_clf)
Print_clf(vot_clf)
import matplotlib

import matplotlib.pyplot as plt

import random



def show(y, img_ord,img_pca):

    index = random.randint(1,100)

    image_ord = img_ord[index].reshape(28, 28)

    image_rcd = img_pca[index].reshape(28, 28)



    plt.figure(figsize=(7, 4))

    pos = 121

    for img in [image_ord, image_rcd] :

        plt.subplot(pos)

        plt.title(f"y = {y[index]}",fontsize = 15)

        plt.imshow(img, cmap = matplotlib.cm.binary,interpolation="nearest")

        plt.axis("off"); pos += 1    

    plt.tight_layout()

    
from sklearn.decomposition import PCA



pca = PCA(n_components=40, whiten=True)



train_reduced = pca.fit_transform(X_train)

train_recovered = pca.inverse_transform(train_reduced)



test_reduced = pca.fit_transform(X_test)

test_recovered = pca.inverse_transform(test_reduced)



show(y_train,X_train,train_recovered)





def Print_clf_PCA(clf):

    print("ord dimention :",X_train.shape[1])

    print("pca dimention : ",test_reduced.shape[1])



    clf.fit(train_reduced,y_train)

    report(clf)



    y_true, y_score = Pretreatment_ROC(clf,test_reduced,y_test)

    show_auc(y_true,y_score)
Print_clf_PCA(knn_clf)
Print_clf_PCA(ext_clf)
Print_clf_PCA(rdf_clf)
Print_clf_PCA(bag_clf)
Print_clf_PCA(ada_clf)
Print_clf_PCA(vot_clf)