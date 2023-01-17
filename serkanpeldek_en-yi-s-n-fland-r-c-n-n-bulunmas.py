#Gerekli Kütüphaneler Geliştirme Ortamına Dahil Ediliyor

import numpy as np # linear algebra

import pandas as pd # Veri işleme



# Visiualization tools

import matplotlib.pyplot as plt

import seaborn as sns



#Machine Learning tools

#Önişleme 

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler



#Model Seçimi

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



#Makine Öğrenmesi Modelleri

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

#Metrikler

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report



from sklearn.externals import joblib



#Sistem Kütüphaneleri

import os

print(os.listdir("../input"))
import warnings

#Sonuçların okunmasını zorlaştırdığı için uyarıları kapatıyoruz

warnings.filterwarnings("ignore")

print("Uyarılar Kapatıldı")
#veri setini pandas DataFrame olarak yüklüyoruz

dataset=pd.read_csv('../input/indian_liver_patient.csv')



#veri setine ait ilk beş satır; 

dataset.head()
# veri setindeki sayısal özelliklere ait özet istatistiksel bilgiler

# Gender özelliği sayısal olmayan değerler içerdiği için,istatistiksel verileri yoktur

dataset.describe().T
dataset.info()
#eksik veriler 'Albumin_and_Globulin_Ratio' sütununun ortalaması ile doldurulu

dataset['Albumin_and_Globulin_Ratio'].fillna(dataset['Albumin_and_Globulin_Ratio'].mean(), inplace=True)

dataset.info()
#'Dataset' sütünun adı 'target' olarak değitiriliyor

dataset.rename(columns={'Dataset':'target'},inplace=True)

dataset.head()
fig, axes=plt.subplots(nrows=2, ncols=2, figsize=(15,10))



sns.countplot(x="target", data=dataset, ax=axes[0,0])

sns.countplot(x="target", hue="Gender", data=dataset, ax=axes[0,1])

sns.countplot(y="target", data=dataset, ax=axes[1,0])

sns.countplot(y="target", hue="Gender", data=dataset, ax=axes[1,1])



liver_patients, not_liver_patinets=dataset['target'].value_counts()

print("Karaciğer hastası sayısı:{}, \nKaraciğer hastası olmayanların sayısı:{}".\

      format(liver_patients,not_liver_patinets))
target_counts=dataset['target'].value_counts().values

gender_counts=dataset['Gender'].value_counts().values



fig1, axes=plt.subplots(nrows=1, ncols=2,figsize=(10,5))

fig1.suptitle("Teşhis ve Cinsiyet Yüzdeleri")



target_sizes=dataset.groupby('target').size()

axes[0].pie(

    x=target_counts,

    labels=['patient({})'.format(target_sizes[1]),'not patient({})'.format(target_sizes[2])],

    autopct='%1.1f%%'

)

axes[0].set_title("Hasta Teşhis Yüzdeleri")



gender_sizes=dataset.groupby('Gender').size()

axes[1].pie(

    x=gender_counts, 

    labels=['male({})'.format(gender_sizes['Male']), 'female({})'.format(gender_sizes['Female'])], 

    autopct="%1.1f%%"

)

axes[1].set_title("Hastaların Cinsiyet Yüzdeleri")
def plot_categorical_dist(dataset, categorical_feature, rows, cols, plot_type):

    fig, axarr=plt.subplots(nrows=rows,ncols=cols, figsize=(15,10))

    features=dataset.columns.values[:-1]

    

    counter=0

    

    for i in range(rows):

        for j in range(cols):

            feature=features[counter]

            if "swarm" in plot_type:

                sns.swarmplot(x=categorical_feature,y=feature, data=dataset, ax=axarr[i, j])

            elif "box" in plot_type:

                sns.boxplot(x=categorical_feature, y=feature, data=dataset,ax=axarr[i,j])

            elif "violin" in plot_type:

                sns.violinplot(x=categorical_feature, y=feature, data=dataset,ax=axarr[i,j])

            counter=counter+1

            if counter>=len(features):

                break

    

    plt.tight_layout()

    plt.show()
plot_categorical_dist(dataset=dataset, 

                      categorical_feature='target',

                      rows=3, 

                      cols=4,

                     plot_type="swarm")
plot_categorical_dist(dataset=dataset, 

                      categorical_feature='target',

                      rows=3, 

                      cols=4,

                     plot_type="box")
plot_categorical_dist(dataset=dataset, 

                      categorical_feature='target',

                      rows=3, 

                      cols=4,

                     plot_type="violin")
dataset=pd.get_dummies(dataset)
#dataset.hist(figsize=(10,12))



def draw_hist(dataset, rows, cols):

    fig, axes=plt.subplots(nrows=rows, ncols=cols, figsize=(15,12))

    names=dataset.columns.values

    counter=0

    for i in range(rows):

        for j in range(cols):

            if counter>=len(names):

                break

            name=names[counter]

            sns.distplot(a=dataset[name], ax=axes[i,j])

            

            counter+=1
draw_hist(dataset=dataset, rows=3, cols=4)
selected_pair_cols=['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 

           'Alamine_Aminotransferase','Total_Protiens','target' ]

sns.pairplot(data=dataset[selected_pair_cols], hue="target", kind='reg')
corr_matrix=dataset.corr()

fig, ax = plt.subplots(figsize=(12,12))

sns.heatmap(corr_matrix,annot=True,linewidths=.5, ax=ax)
def show_corr(dataset, target_name, n_most=None):

    if n_most is None:

        n_most=len(dataset.columns.values)-1

    corr_matrix=dataset.corr().abs()

    

    most_correlated_features=corr_matrix[target_name].sort_values(ascending=False).drop(target_name)

       

    most_correlated_feature_names=most_correlated_features.index.values

    

    fig, ax=plt.subplots(figsize=(15,5))

    plt.xticks(rotation="90")

    sns.barplot(x=most_correlated_feature_names, y=most_correlated_features)

    plt.title("{} İle En Yüksek Korelasyona Sahip Özellikler".format(target_name))
show_corr(dataset=dataset, target_name='target')
corr_matrix=dataset.corr().abs()

sorted_corr=(corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

                 .stack()

                 .sort_values(ascending=False))

sorted_corr.head()
x_vars=['Total_Bilirubin', 'Alamine_Aminotransferase', 'Total_Protiens', 'Albumin']

y_vars=['Direct_Bilirubin', 'Aspartate_Aminotransferase', 'Albumin', 'Albumin_and_Globulin_Ratio']



fig, axarr=plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

axarr=axarr.flatten()

for i in range(len(x_vars)):

    sns.scatterplot(data=dataset, x=x_vars[i], y=y_vars[i], ax=axarr[i])
#Veri seti data ve target olarak ayrıştırılır

X=dataset.drop('target', axis=1) #data

y=dataset['target'] # target
#Data kısmı normalleştirilir

scaler=MinMaxScaler()

scaled_values=scaler.fit_transform(X)

X.loc[:,:]=scaled_values
#Eğitim ve test kümelerine ayrıştırılır

X_train, X_test, y_train, y_test=train_test_split(X,y, stratify=y, test_size=0.3,random_state=42)
models=[]

models.append(("LR",LogisticRegression()))

models.append(("NB",GaussianNB()))

models.append(("KNN",KNeighborsClassifier(n_neighbors=5)))

models.append(("DT",DecisionTreeClassifier()))

models.append(("SVM",SVC()))

models.append(('LDA', LinearDiscriminantAnalysis()))


for name, model in models:

    

    clf=model



    clf.fit(X_train, y_train)



    y_pred=clf.predict(X_test)

    print(10*"=","{} için Sonuçlar".format(name).upper(),10*"=")

    print("Başarı oranı:{:0.2f}".format(accuracy_score(y_test, y_pred)))

    print("Karışıklık Matrisi:\n{}".format(confusion_matrix(y_test, y_pred)))

    print("Sınıflandırma Raporu:\n{}".format(classification_report(y_test,y_pred)))

    print(30*"=")
import mglearn
mglearn.plots.plot_cross_validation()
for name, model in models:

    kfold=KFold(n_splits=5, random_state=42)

    cv_result=cross_val_score(model, X, y, cv=kfold, scoring="accuracy")

    print("{} modelinin çağraz doğrulama sonucu:{:0.2f}".format(name,cv_result.mean()))
params_clfs=list()



svm_params=[

    {'kernel':['rbf'], 'gamma':[1e-3, 1e-4]},

    {'kernel':['linear'], 'C':[1, 10, 100, 1000]}       

]

params_clfs.append((SVC(),svm_params))





lr_params= {'penalty':['l1', 'l2'], 'C':np.logspace(0, 4, 10)}

params_clfs.append((LogisticRegression(),lr_params))



clf=DecisionTreeClassifier()

dt_params={'max_features': ['auto', 'sqrt', 'log2'],

          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15],

          'min_samples_leaf':[1],

          'random_state':[123]}

params_clfs.append((DecisionTreeClassifier(),dt_params))
for clf, param in params_clfs:

    

    grid_search=GridSearchCV(clf, param, cv=5)

    grid_search.fit(X_train, y_train)

    print(80*"*")

    print("{} İçin sklearn GridSearchCV Sonuçları".format(clf.__class__.__name__))

    print("best params:{}".format(grid_search.best_params_))

    test_means=grid_search.cv_results_['mean_test_score']

    print("ortalama test sonucu:{:.2f}".format(np.mean(test_means)))

    y_pred=grid_search.predict(X_test)

    print("en iyi parametre sonucu:{:.2f}".format(accuracy_score(y_test, y_pred)))

    print("Karışıklık matrisi:\n{}".format(confusion_matrix(y_test, y_pred)))

    print("Sınıflandırma Raporu:\n{}".format(classification_report(y_test, y_pred)))

    print(80*"*")
from yellowbrick.classifier import ClassificationReport
best_clf=LogisticRegression(C= 21.544346900318832, penalty='l1')

best_clf.fit(X_train, y_train)

y_pred=best_clf.predict(X_test)

print("Baları oranı:{:.2f}".format(accuracy_score(y_test, y_pred)))

print("Karışıklık matrisi:\n{}".format(confusion_matrix(y_test, y_pred)))

print("Sınıflandırma Raporu:\n{}".format(classification_report(y_test, y_pred)))
visualizer = ClassificationReport(best_clf, classes=['patient','not patient'], support=True)



visualizer.fit(X_train, y_train)  # Fit the visualizer and the model

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

g = visualizer.poof()  
from yellowbrick.classifier import ROCAUC
fig, ax=plt.subplots(1,1,figsize=(12,8))

roc_auc=ROCAUC(best_clf, ax=ax)

roc_auc.fit(X_train, y_train)

roc_auc.score(X_test, y_test)



roc_auc.poof()
work_flows_std = list()

work_flows_std.append(('standardize', StandardScaler()))

work_flows_std.append(('logReg', LogisticRegression(C= 21.544346900318832, penalty='l1')))

model_std = Pipeline(work_flows_std)

model_std.fit(X_train, y_train)

y_pred=model_std.predict(X_test)



print("Başarı oranı:{:.2f}".format(accuracy_score(y_test, y_pred)))

print("Karışıklık matrisi:\n{}".format(confusion_matrix(y_test, y_pred)))

print("Sınıflandırma Raporu:\n{}".format(classification_report(y_test, y_pred)))
work_flows_pca = list()

work_flows_pca.append(('pca', PCA(n_components=7)))

work_flows_pca.append(('logReg', LogisticRegression(C= 21.544346900318832, penalty='l1')))

model_pca = Pipeline(work_flows_pca)

model_pca.fit(X_train, y_train)

y_pred=model_pca.predict(X_test)
print("Başarı oranı:{:.2f}".format(accuracy_score(y_test, y_pred)))

print("Karışıklık matrisi:\n{}".format(confusion_matrix(y_test, y_pred)))

print("Sınıflandırma Raporu:\n{}".format(classification_report(y_test, y_pred)))
print(os.listdir(".."))
dosya_adi="../working/model_pca.pickle"

joblib.dump(model_pca,dosya_adi)
loaded_model=joblib.load('../working/model_pca.pickle')
y_pred=loaded_model.predict(X_test)

print("Başarı oranı:{:.2f}".format(accuracy_score(y_test, y_pred)))

print("Karışıklık matrisi:\n{}".format(confusion_matrix(y_test, y_pred)))

print("Sınıflandırma Raporu:\n{}".format(classification_report(y_test, y_pred)))