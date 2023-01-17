# Gerekli Kütüphaneleri Yüklenmesi Yapılıyor

import numpy as np # linear algebra

import pandas as pd # veri işleme



#Görselleştirme Kütüphaneleri

import seaborn as sns

import matplotlib.pyplot as plt



#Makine öğrenmesi gereçleri

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score



#Makine öğrenmesi algoritmaları

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



#Performans metrikleri

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report



import imblearn

from imblearn.over_sampling import SMOTE

#Sistem kütüphaneleri

import os

import warnings



# Çıktılarda karmaşıklığa sebep olduğu için uyarılırı iptal ediyoruz

warnings.filterwarnings("ignore")

print(os.listdir("../input"))
#Veri setinin yüklemesi yapılıyor 

dataset=pd.read_csv("../input/winequality-red.csv")

dataset.head()
dataset.info()
#Kaç farklı kalite puanı olduğunu öğrenelim

print("Kalite puanları:",dataset['quality'].unique())
#Herbir kalite puanından kaçtane örnek olduğunu görelim

print(dataset['quality'].value_counts())
#kalite puanlarını pie grafikle göstrelim

plt.figure(1, figsize=(8,8))

dataset['quality'].value_counts().plot.pie(autopct="%1.1f%%")
dataset.describe().T
selected_features=['residual sugar', 'total sulfur dioxide', 'sulphates',

                   'alcohol', 'volatile acidity', 'quality']

dataset_selected_features=dataset[selected_features]
condition1=(dataset_selected_features['quality']==3)|(dataset_selected_features['quality']==4)

condition2=(dataset_selected_features['quality']==5)|(dataset_selected_features['quality']==6)

condition3=(dataset_selected_features['quality']==7)|(dataset_selected_features['quality']==8)

level_34=round(dataset_selected_features[condition1].describe(),2)

level_56=round(dataset_selected_features[condition2].describe(),2)

level_78=round(dataset_selected_features[condition3].describe(),2)
level_all=pd.concat([level_34,level_56, level_78],

                    axis=1, 

                    keys=['Levels:3,4','Levels:5,6','Levels:7,8',])

level_all.T
#Özelliklerin kalite puanları ile ilişkisini göstermek için kullanılacak

#çizim türleri

def draw_multivarient_plot(dataset, rows, cols, plot_type):

    """

    dataset: Veri seti

    rows: Satır sayısı

    cols: sütün sayısı

    plot_type: Çizdirilecek grafik türü

    """

    

    #Veri setindeki sütünların isimleri alınıyor

    column_names=dataset.columns.values

    #Kaç tane sütün olduğu bulunuyor

    number_of_column=len(column_names)

    

    #Satır*sütün boyutlarında alt grafik içeren

    #matris oluşturuluyor. Matrisin genişliği:22 yüksekliği:16

    fig, axarr=plt.subplots(rows,cols, figsize=(22,16))



    counter=0# Çizimi yapılacak özelliğin column_names listesindeki indeks değerini tutuyor

    for i in range(rows):

        for j in range(cols):

            """

            i: satır numarasını tutuyor

            j: sütün numarasını tutuyor

            axarr[i][j]: Çizilen grafigin grafik matrisindeki yerini belirliyor

            """

            if 'violin' in plot_type:

                sns.violinplot(x='quality', y=column_names[counter],data=dataset, ax=axarr[i][j])

            elif 'box'in plot_type :

                sns.boxplot(x='quality', y=column_names[counter],data=dataset, ax=axarr[i][j])

            elif 'point' in plot_type:

                sns.pointplot(x='quality',y=column_names[counter],data=dataset, ax=axarr[i][j])

            elif 'bar' in plot_type:

                sns.barplot(x='quality',y=column_names[counter],data=dataset, ax=axarr[i][j])

                

            counter+=1

            if counter==(number_of_column-1,):

                break
#Box Plot türünde grafik çizdiriliyor

draw_multivarient_plot(dataset,4,3,"box")
#Violin Plot türünde grafik çizdiriliyor

draw_multivarient_plot(dataset,4,3,"violin")
#Point Plot türünde grafik çizdiriliyor

draw_multivarient_plot(dataset,4,3,"pointplot")
#Bar Plot türünde grafik çizdiriliyor

draw_multivarient_plot(dataset,4,3,"bar")


def get_models():

    models=[]

    models.append(("LR",LogisticRegression()))

    models.append(("NB",GaussianNB()))

    models.append(("KNN",KNeighborsClassifier()))

    models.append(("DT",DecisionTreeClassifier()))

    models.append(("SVM rbf",SVC()))

    models.append(("SVM linear",SVC(kernel='linear')))

    models.append(('LDA', LinearDiscriminantAnalysis()))

    

    return models



def cross_validation_scores_for_various_ml_models(X_cv, y_cv):

    print("Çapraz Doğrulama Başarı Oranları".upper())

    models=get_models()





    results=[]

    names= []



    for name, model in models:

        kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=22)

        cv_result=cross_val_score(model,X_cv, y_cv, cv=kfold,scoring="accuracy")

        names.append(name)

        results.append(cv_result)

        print("{} modelinin çapraz doğrulaması yapıldı, başarı oranı:{:0.2f}".format(name, cv_result.mean()))



   
dataset_temp=dataset.copy(deep=True)

X=dataset.drop('quality', axis=1)

y=dataset['quality']



X=StandardScaler().fit_transform(X)

cross_validation_scores_for_various_ml_models(X, y)
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
y_frame=pd.DataFrame()

y_frame['kalite seviyesi']=y_train

y_frame.groupby(['kalite seviyesi']).size().plot.bar(figsize=(8,4),

                                                     title="Herbir Kalite Seviyesinin Eğitim Kümesindeki Dağılımı")
y_frame=pd.DataFrame()

y_frame['kalite seviyesi']=y_test

y_frame.groupby(['kalite seviyesi']).size().plot.bar(figsize=(8,4),title="Herbir Kalite Seviyesinin Test Kümesindeki Dağılımı")
def SVM_GridSearch(X_train, X_test, y_train, y_test):

    best_score=0

    gammas=[0.001, 0.01, 0.1, 1, 10, 100]

    Cs=[0.001, 0.01, 0.1, 1, 10, 100]

    

    for gamma in gammas:

        for C in Cs:

            svm=SVC(kernel='rbf',gamma=gamma, C=C)

            svm.fit(X_train, y_train)

            

            

            score=svm.score(X_test, y_test)

            

            if score>best_score:

                y_pred=svm.predict(X_test)

                best_score=score

                best_params={'C':C, 'gamma':gamma}

        

    print("best score:",best_score)

    print("best params:",best_params)

    print("classification reports:\n",classification_report(y_test, y_pred))
SVM_GridSearch(X_train, X_test, y_train, y_test)
dataset_temp.loc[(dataset_temp['quality']==3),'quality']=1

dataset_temp.loc[(dataset_temp['quality']==4),'quality']=1



dataset_temp.loc[(dataset_temp['quality']==5),'quality']=2

dataset_temp.loc[(dataset_temp['quality']==6),'quality']=2



dataset_temp.loc[(dataset_temp['quality']==7),'quality']=3

dataset_temp.loc[(dataset_temp['quality']==8),'quality']=3
dataset_temp['quality'].value_counts()
#Box Plot türünde grafik çizdiriliyor

draw_multivarient_plot(dataset_temp,4,3,"box")
#Violin Plot türünde grafik çizdiriliyor

draw_multivarient_plot(dataset_temp,4,3,"violin")
#Point Plot türünde grafik çizdiriliyor

draw_multivarient_plot(dataset_temp,4,3,"point")
#Bar Plot türünde grafik çizdiriliyor

draw_multivarient_plot(dataset_temp,4,3,"bar")
X_temp=dataset_temp.drop('quality', axis=1)

y_temp=dataset_temp['quality']

X_temp=StandardScaler().fit_transform(X_temp)



X_train_temp, X_test_temp, y_train_temp, y_test_temp=train_test_split(X_temp, 

                                                                      y_temp,

                                                                      stratify=y_temp,

                                                                      test_size=0.3,

                                                                      random_state=42)





cross_validation_scores_for_various_ml_models(X_temp, y_temp)
SVM_GridSearch(X_train_temp, X_test_temp, y_train_temp, y_test_temp)




print('Az örnekleri çoğaltmadan önce')

print('X_train_temp.shape:', X_train_temp.shape)

print('y_train_temp.shape:', y_train_temp.shape)

smote = SMOTE()

X_train_temp, y_train_temp = smote.fit_resample(X_train_temp, y_train_temp)

print('Az örnekleri çoğalttıktan sonra')

print('X_train_temp.shape:', X_train_temp.shape)

print('y_train_temp.shape:', y_train_temp.shape)



SVM_GridSearch(X_train_temp, X_test_temp, y_train_temp, y_test_temp)