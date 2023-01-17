#Gerekli Kütüphanelerin Yüklenmesi Yapılıyor

import numpy as np # linear algebra

import pandas as pd # Veri işleme

#Görselleştirme

import matplotlib.pyplot as plt

import seaborn as sns

#Makine Öğrenmesi Gereksimleri

# Ön işleme için

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Model seçimi için

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

# Makine öğrenmesi modelleri

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

#Sistem Kütüphaneleri

import os

import warnings

print(os.listdir("../input"))
#Uyarıları kapatılıyor

warnings.filterwarnings("ignore")

print("Uyarılar kapatıldı")
#Üç sınıf içeren veri seti

dataset_3c=pd.read_csv("../input/column_3C_weka.csv")

#Veri setinin ilk beş örneği(satırı) listelenecek. n varsayılan olarak 5'tir.  

dataset_3c.head()
#İki sınıf içeren veri seti

dataset_2c=pd.read_csv("../input/column_2C_weka.csv")

#Veri setinin ilk beş örneği(satırı) listelenecek.

dataset_2c.head(n=3)
dataset_3c.info()
def discrete_univariate(dataset, discrete_feature):

    fig, axarr=plt.subplots(nrows=1,ncols=2, figsize=(8,5))

      

    dataset[discrete_feature].value_counts().plot(kind="bar",ax=axarr[0])

    dataset[discrete_feature].value_counts().plot.pie(autopct="%1.1f%%",ax=axarr[1])

        

    plt.tight_layout()

    plt.show()
discrete_univariate(dataset=dataset_3c, discrete_feature="class")
def continuous_univariate(dataset, continuos_feature):

    fig, ax=plt.subplots(nrows=2,ncols=3, figsize=(12,8))



    ax=ax.flatten()



    "pandas çizim fonksiyonları kullanılıyor"

    dataset_3c[continuos_feature].plot.hist(density=True,ax=ax[0])

    dataset_3c[continuos_feature].plot.kde(ax=ax[1])

    dataset_3c[continuos_feature].plot.hist(density=True,ax=ax[2])

    dataset_3c[continuos_feature].plot.kde(ax=ax[2])



    "Seaborn çizim fonksiyonları kullanılıyor"

    sns.distplot(a=dataset_3c[continuos_feature], kde=False, ax=ax[3])

    sns.distplot(a=dataset_3c[continuos_feature], hist=False, ax=ax[4])

    sns.distplot(a=dataset_3c[continuos_feature], ax=ax[5]) 
continuous_univariate(dataset=dataset_3c, continuos_feature="pelvic_incidence")
dataset_3c.hist(bins=10, density=True, figsize=(15,8))

plt.show()
fig, ax=plt.subplots(nrows=2, ncols=3,figsize=(12,8))

ax=ax.flatten()

col_names=dataset_3c.drop('class', axis=1).columns.values



for i,col_name in enumerate(col_names):

    sns.distplot(a=dataset_3c[col_name], ax=ax[i])
def plot_categorical(dataset, categorical_feature, rows, cols):

    fig, axarr=plt.subplots(nrows=2,ncols=4, figsize=(15,10))

    features=dataset.columns.values[:-1]

    

    counter=0

    #sns.countplot(x=categorical_feature, data=dataset, ax=axarr[0,0])

    dataset['class'].value_counts().plot.bar(ax=axarr[0,0])

    dataset['class'].value_counts().plot.pie(autopct="%1.1f%%",ax=axarr[0,1])

    for i in range(rows):

        for j in range(cols):

            feature=features[counter]

            if (i==0 and j==0) or (i==0 and j==1):

                continue

            else:

                sns.swarmplot(x=categorical_feature,y=feature,

                             

                            data=dataset, 

                            ax=axarr[i, j])

            counter=counter+1

            if counter>=len(features):

                break

    

    plt.tight_layout()

    plt.show()
plot_categorical(dataset=dataset_3c, categorical_feature="class", rows=2, cols=4)
plot_categorical(dataset=dataset_2c, categorical_feature="class", rows=2, cols=4)
#Sınıflar ile özelliklerin ikilişkisini gösteren pairplot

sns.pairplot(dataset_2c, hue="class", markers=["o", "s"])
#Sınıflar ile özelliklerin ikilişkisini gösteren pairplot

sns.pairplot(dataset_3c, hue="class", markers=["o", "s",'D'])
corr=dataset_3c.corr()
fig, ax=plt.subplots(1,1,figsize=(12,8))

sns.heatmap(corr,annot=True, linewidth=.5, ax=ax)
#Sınıflar ile özelliklerin ikilişkisini gösteren farklı grafikler;

#boxplot, violinplot, pointplot, barplot

def draw_multivarient_plot(dataset, rows, cols, plot_type):

    

    assert plot_type in ['violin', "box", "point", "bar"],"We dont have such as plot type:{}".format(plot_type)

    column_names=dataset.columns.values

    number_of_column=len(column_names)

    fig, axarr=plt.subplots(rows,cols, figsize=(22,16))

    

    counter=0

    for i in range(rows):

        for j in range(cols):

            if 'violin' in plot_type:

                sns.violinplot(x='class', y=column_names[counter],data=dataset, ax=axarr[i][j])

            elif 'box'in plot_type :

                sns.boxplot(x='class', y=column_names[counter],data=dataset, ax=axarr[i][j])

            elif 'point' in plot_type:

                sns.pointplot(x='class',y=column_names[counter],data=dataset, ax=axarr[i][j])

            elif 'bar' in plot_type:

                sns.barplot(x='class',y=column_names[counter],data=dataset, ax=axarr[i][j])

                

            counter+=1

            if counter==(number_of_column-1,):

                break
draw_multivarient_plot(dataset=dataset_2c, rows=2, cols=3,plot_type="violin")
draw_multivarient_plot(dataset=dataset_2c, rows=2, cols=3,plot_type="point")
draw_multivarient_plot(dataset=dataset_2c, rows=2, cols=3,plot_type="box")
draw_multivarient_plot(dataset=dataset_2c, rows=2, cols=3,plot_type="bar")
draw_multivarient_plot(dataset=dataset_3c, rows=2, cols=3,plot_type="violin")
draw_multivarient_plot(dataset=dataset_3c, rows=2, cols=3,plot_type="point")
draw_multivarient_plot(dataset=dataset_3c, rows=2, cols=3,plot_type="box")
draw_multivarient_plot(dataset=dataset_3c, rows=2, cols=3,plot_type="bar")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def get_models():

    models=[]

    models.append(("LR",LogisticRegression()))

    models.append(("LDA", LinearDiscriminantAnalysis()))

    models.append(("NB",GaussianNB()))

    models.append(("KNN",KNeighborsClassifier(n_neighbors = 3)))

    models.append(("DT",DecisionTreeClassifier()))

    models.append(("SVM rbf",SVC()))

    models.append(("SVM linear",SVC(kernel='linear')))

    

    return models
def get_X_and_y(dataset, target_name):

    X=dataset.drop(target_name, axis=1)

    y=dataset[target_name]



    X=StandardScaler().fit_transform(X)

    

    labelEncoder=LabelEncoder()

    y=labelEncoder.fit_transform(y)

    

    return X, y
def accuracy_scores_for_various_ml_models(X, y):

    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)

    models=get_models()

    for name, model in models:

        model.fit(X_train, y_train)

        score=model.score(X_test, y_test)

        

        print("{} accuracy:{:.2f}".format(name, score))
print("dataset_2c Veri Seti İçin Başarı Oranları")

X, y=get_X_and_y(dataset=dataset_2c, target_name='class')

accuracy_scores_for_various_ml_models(X, y)
print("dataset_3c Veri Seti İçin Başarı Oranları")

X, y=get_X_and_y(dataset=dataset_3c, target_name='class')

accuracy_scores_for_various_ml_models(X, y)


def cross_validation_scores_for_various_ml_models(X_cv, y_cv):

    print("Çapraz Doğrulama Başarı Oranları".upper())

    models=get_models()

    results=[]

    names= []



    for name, model in models:

        kfold=KFold(n_splits=5,random_state=22)

        cv_result=cross_val_score(model,X_cv, y_cv, cv=kfold,scoring="accuracy")

        names.append(name)

        results.append(cv_result)

        print("{} modelinin çapraz doğrulaması yapıldı, başarı oranı:{:0.2f}".format(name, cv_result.mean()))





def cross_validate(dataset, target_name):

    X, y=get_X_and_y(dataset, target_name)



    cross_validation_scores_for_various_ml_models(X, y)
cross_validate(dataset=dataset_2c, target_name="class")
def MY_SVM_GridSearch(dataset, target_name):

    X=dataset.drop(target_name, axis=1)

    y=dataset[target_name]

    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)

    best_score=0

    gammas=[0.001, 0.01, 0.1, 1, 10, 100]

    Cs=[0.001, 0.01, 0.1, 1, 10, 100]

    kernels=['rbf', 'linear']

    

    for gamma in gammas:

        for C in Cs:

            for kernel in kernels:

                svm=SVC(kernel=kernel,gamma=gamma, C=C)

                svm.fit(X_train, y_train)

                

                score=svm.score(X_test, y_test)



                if score>best_score:

                    y_pred=svm.predict(X_test)

                    best_score=score

                    best_params={'kernel':kernel, 'C':C, 'gamma':gamma}

        

    print("best score:",best_score)

    print("best params:",best_params)

    print("classification reports:\n",classification_report(y_test, y_pred))
MY_SVM_GridSearch(dataset=dataset_2c, target_name='class')
cross_validate(dataset=dataset_3c, target_name="class")
MY_SVM_GridSearch(dataset=dataset_3c, target_name='class')
scv_params=[{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]



grid_search=GridSearchCV(SVC(), scv_params, cv=5)
X, y=get_X_and_y(dataset=dataset_2c, target_name='class')

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)

grid_search.fit(X_train, y_train)



print("İki Sınıf İçin sklearn GridSearchCV Sonuçları")

print()

print("best params:{}".format(grid_search.best_params_))



print()

test_means=grid_search.cv_results_['mean_test_score']

print("ortalama test sonucu:{:.2f}".format(np.mean(test_means)))



print()

y_pred=grid_search.predict(X_test)

print("en iyi parametre sonucu:{:.2f}".format(accuracy_score(y_test, y_pred)))

print("classification report:\n{}".format(classification_report(y_test, y_pred)))

X, y=get_X_and_y(dataset=dataset_3c, target_name='class')

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)

grid_search.fit(X_train, y_train)



print("Üç Sınıf İçin sklearn GridSearchCV Sonuçları")

print()

print("best params:{}".format(grid_search.best_params_))



print()

test_means=grid_search.cv_results_['mean_test_score']

print("ortalama test sonucu:{:.2f}".format(np.mean(test_means)))



print()

y_pred=grid_search.predict(X_test)

print("en iyi parametre sonucu:{:.2f}".format(accuracy_score(y_test, y_pred)))

print("classification report:\n{}".format(classification_report(y_test, y_pred)))
