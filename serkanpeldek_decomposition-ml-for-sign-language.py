import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import cv2



from sklearn.decomposition import PCA, KernelPCA

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score, KFold

from sklearn.model_selection import GridSearchCV

from sklearn import metrics



import os

import warnings

warnings.filterwarnings("ignore")

print(os.listdir("../input"))
X=np.load('../input/Sign-language-digits-dataset/X.npy')

y=np.load('../input/Sign-language-digits-dataset/Y.npy')
print("X shape:",X.shape)

print("y shape:",y.shape)

random_choice=np.random.randint(X.shape[0])

print("y[{}]:{}".format(random_choice, y[random_choice]))
y_new=list()

for target in y:

    y_new.append(np.argmax(target))

y=np.array(y_new)
print("X shape:",X.shape)

print("y shape:",y.shape)

print("y[{}]:{}".format(random_choice, y[random_choice]))
sample_per_class=np.unique(y, return_counts=True)

 

for sign, number_of_sample in zip(sample_per_class[0], sample_per_class[1]):

    print("{} sign has {} samples.".format(sign, number_of_sample))
def show_image_classes(image, label, n=10):

    fig, axarr=plt.subplots(nrows=n, ncols=n, figsize=(18, 18))

    axarr=axarr.flatten()

    plt_id=0

    start_index=0

    for sign in range(10):

        sign_indexes=np.where(label==sign)[0]

        for i in range(n):



            image_index=sign_indexes[i]

            axarr[plt_id].imshow(image[image_index], cmap='gray')

            axarr[plt_id].set_xticks([])

            axarr[plt_id].set_yticks([])

            axarr[plt_id].set_title("Sign :{}".format(sign))

            plt_id=plt_id+1

    plt.suptitle("{} Sample for Each Classes".format(n))

    plt.show()
show_image_classes(image=X, label=y)
label_map={0:9,1:0, 2:7, 3:6, 4:1, 5:8, 6:4, 7:3, 8:2, 9:5}

y_new=list()

for s in y:

    y_new.append(label_map[s])

y=np.array(y_new)
show_image_classes(image=X, label=y)
X_normal=X.reshape(X.shape[0],X.shape[1]*X.shape[2])
X_train, X_test, y_train, y_test=train_test_split(X_normal, y,

                                                 stratify=y,

                                                 test_size=0.3,

                                                 random_state=42)
pca=PCA(n_components=2)

pca.fit(X_train)

X_train_pca=pca.transform(X_train)

X_test_pca=pca.transform(X_test)

print("PCA transform performed...")
number_of_class=10



fig=plt.figure(figsize=(10,8))

ax=fig.add_subplot(1,1,1)

scatter=ax.scatter(X_train_pca[:,0],

            X_train_pca[:,1], 

            c=y_train,

            s=10,

           cmap=plt.get_cmap('jet', number_of_class)

          )



ax.set_xlabel("First Principle Component")

ax.set_ylabel("Second Principle Component")

ax.set_title("PCA projection of {} classes".format(number_of_class))



fig.colorbar(scatter)
pca=PCA()

pca.fit(X_normal)



plt.figure(1, figsize=(12,8))



plt.plot(pca.explained_variance_, linewidth=2)

 

plt.xlabel('Components')

plt.ylabel('Explained Variaces')

plt.xticks([50,100, 250, 500, 1000, 2050])

plt.show()
determined_n_components=100
models=[LinearDiscriminantAnalysis(),

       LogisticRegression(multi_class="auto", solver="liblinear"),

       RandomForestClassifier(n_estimators=100),

       KNeighborsClassifier(n_neighbors=5),

       DecisionTreeClassifier(),

       SVC(gamma="scale")]
class Clf_Helper:

    

    def __init__(self, X, y, n_components):

        self.train_test_split(X,y)

        self.n_components=n_components

        self.transform()

        

    

    def train_test_split(self, X, y):

        if np.max(X[0])>1:

            X=X/255.0

            print("Data scaled...")

            

        self.X_train, self.X_test, self.y_train, self.y_test=train_test_split(X, y,

                                                 stratify=y,

                                                 test_size=0.3,

                                                 random_state=42)

    def transform(self):

        self.pca_transform()

        

    def pca_transform(self):

        pca=PCA(n_components=self.n_components)

        pca.fit(self.X_train)

        self.X_train=pca.transform(self.X_train)

        self.X_test=pca.transform(self.X_test)

        print("PCA transform performed...")

    

    def best_model(self, models, show_metrics=False):

        print("INFO: Finding Accuracy Best Classifier...", end="\n\n")

        best_clf=None

        best_acc=0

        for clf in models:

            clf.fit(self.X_train, self.y_train)

            y_pred=clf.predict(self.X_test)

            acc=metrics.accuracy_score(self.y_test, y_pred)

            print(clf.__class__.__name__, end=" ")

            print("Accuracy:{:.3f}".format(acc))



            if best_acc<acc:

                best_acc=acc

                best_clf=clf

                best_y_pred=y_pred

        

        print("Best Classifier:{}".format(best_clf.__class__.__name__))

        if show_metrics:

            self.metrics(y_true=self.y_test, y_pred=best_y_pred)

    

    def cv_best_model(self, models, show_metrics=False):

        print("INFO: Finding Cross Validated Accuracy Best Classifier...", end="\n\n")

        kfold=KFold(n_splits=5,  shuffle=True, random_state=0)

        best_clf=None

        best_acc=0

        for clf in models:

            cv_scores=cross_val_score(clf, self.X_train, self.y_train, cv=kfold)

            print(clf.__class__.__name__, end=" ")

            cv_mean_acc=cv_scores.mean()

            print("CV Mean Accuracy:{:.3f}".format(cv_mean_acc))

            if best_acc<cv_mean_acc:

                  best_acc=cv_mean_acc

                  best_clf=clf

                

        print("CV Best Classifier:{}".format(best_clf.__class__.__name__))

        if show_metrics:

            y_pred = best_clf.predict(self.X_test)

            self.metrics(y_true=self.y_test, y_pred=y_pred)

        

        return best_clf

    

    def grid_searc_cv_for_best_model(self, model, params, show_metrics=False):

        kfold=KFold(n_splits=3,  shuffle=True, random_state=0)

        grid_search_cv=GridSearchCV(SVC(), params, cv=kfold, scoring="accuracy")

        grid_search_cv.fit(self.X_train, self.y_train)

        y_pred=grid_search_cv.predict(self.X_test)

        print("Best pamameters for {}:{}".format(model.__class__.__name__, 

                                              grid_search_cv.best_params_))

        print("Accuracy:{:.3f}".format(metrics.accuracy_score(self.y_test, y_pred)))

        if show_metrics:

            self.metrics(y_true=self.y_test, y_pred=y_pred)

    

    def metrics(self, y_true, y_pred):

        print("Accuracy:{:.3f}".format(metrics.accuracy_score(y_true, y_pred)))

        print("Confusion Matrix:\n{}".format(metrics.confusion_matrix(y_true, y_pred)))

        print("Classification Report:\n{}".format(metrics.classification_report(y_true, y_pred)))
clf_helper=Clf_Helper(X_normal, y, determined_n_components)
clf_helper.best_model(models)
cv_best_clf=clf_helper.cv_best_model(models)
parameters={'gamma': [1, 1e-1, 1e-2, 1e-3],

                     'C': [1, 10, 100, 1000]}

clf_helper.grid_searc_cv_for_best_model(cv_best_clf, parameters)

hist=cv2.calcHist(images=[X[0]],

                 channels=[0],

                 mask=None,

                 histSize=[256],

                 ranges=[0,1])
plt.figure(figsize=(6,2))

plt.subplot(121)

plt.plot(hist, color="black")



plt.subplot(122)

plt.imshow(X[0],cmap="gray")

plt.xticks([])

plt.yticks([])



plt.show()
image= X[0].copy()



image=np.uint8(image*255)



plt.figure(figsize=(12,2))

plt.subplot(141)

plt.imshow(image, cmap="gray")

plt.subplot(142)

plt.plot(hist, color="black")



eq_image=cv2.equalizeHist(image)

eq_hist=cv2.calcHist(images=[eq_image],

                 channels=[0],

                 mask=None,

                 histSize=[256],

                 ranges=[0,256])

plt.subplot(143)

eq_image=eq_image/255.0

plt.imshow(eq_image, cmap="gray")

plt.subplot(144)

plt.plot(eq_hist, color="black")

X_eq=list()

count=0

for image in X.copy():

    image=np.uint8(image*255)

    eq_image=cv2.equalizeHist(image)

    eq_image/255.0

    X_eq.append(eq_image)

    count=count+1

    



X_eq=np.array(X_eq)

print("Histogram equalized performed...")

    
show_image_classes(image=X_eq, label=y)
X_eq=X_eq.reshape(X_eq.shape[0],X.shape[1]*X.shape[2])
clf_helper=Clf_Helper(X_eq, y, determined_n_components)
clf_helper.best_model(models)
cv_best_clf=clf_helper.cv_best_model(models)
parameters={'gamma': [1, 1e-1, 1e-2, 1e-3],

                     'C': [1, 10, 100, 1000]}

clf_helper.grid_searc_cv_for_best_model(cv_best_clf, parameters)
class Extended_Clf_Helper(Clf_Helper):

    

    def __init__(self, X, y, n_components, transform_type, kernel):

        

        self.transform_type=transform_type

        self.kernel=kernel

        Clf_Helper.__init__(self, X, y, n_components)

        

    

    def transform(self):

        if self.transform_type=="pca":

            self.pca_transform()

        elif self.transform_type=="kernel_pca":

            self.kernel_pca_transform()

        else:

            raise ValueError("Bad transform choice!!!")

    

    def kernel_pca_transform(self):

        kernel_pca=KernelPCA(n_components=self.n_components, kernel=self.kernel)

        kernel_pca.fit(self.X_train)

        self.X_train=kernel_pca.transform(self.X_train)

        self.X_test=kernel_pca.transform(self.X_test)

        print("KernelPCA(kernel={}) transform performed...".format(self.kernel))

extended_clf_helper=Extended_Clf_Helper(X_eq, y, determined_n_components,"kernel_pca","linear")
cv_best_clf=extended_clf_helper.best_model(models)
extended_clf_helper.grid_searc_cv_for_best_model(cv_best_clf, parameters)
from sklearn.pipeline import Pipeline
work_flow=[]

work_flow.append(('PCA',PCA(n_components=100)))

work_flow.append(("SVC", SVC(C=10, gamma=0.01)))



X_train, X_test, y_train, y_test=train_test_split(X_normal, y, 

                                                    test_size=0.3, random_state=42)

model=Pipeline(work_flow)

model.fit(X_train, y_train)

y_pred=model.predict(X_test)

acc=metrics.accuracy_score(y_test, y_pred)

cm=metrics.confusion_matrix(y_test, y_pred)

cr=metrics.classification_report(y_test, y_pred)

print("Accuracy score:{:.3f}".format(acc))

print("Confusion Matrix:\n{}".format(cm))

print("Classification Report:\n{}".format(cr))
 

print(" Many thanks for reading and upvoting the kernel\n"*10)