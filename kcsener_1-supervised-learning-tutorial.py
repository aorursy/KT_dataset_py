# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
"""

Irıs data seti yükledik:

"""

iris = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
"""

sepal_length, sepal_width, petal_length, petal_width are features (predictor variables)



species (setosa, versicolor, virginica) are target variables



"""



iris.head()
"""

bunu belirtmek için target variable = y, predictor variables = X olarak ayırmamız gerekiyor.



"""



X = iris.drop('species', axis=1).values  

y = iris['species'].values



"""

.values diyerek modelin istediği formata (array) soktuk. 



"""
"""

Basic Idea of KNN: k-closest data point'e bakarak unlabeled data point'in label'ını tahmin etme



"""
#model importing



from sklearn.neighbors import KNeighborsClassifier #classifier ı import ettik.



#calling classifier with a variable



knn = KNeighborsClassifier(n_neighbors=6) #classifier'ı knn variable'ına atadık. 
#train-test-split



#modele yerleştirilen datanın bir kısmı ile train yapılırken diğer bir kısmı ile de bu train

#test edilir. yani, örneğin, datanın %70'i ile optimum weight'ler belirlenir, kalan %30luk kısım ise bu modelin

#doğruluğu test edilir.



from sklearn.model_selection import train_test_split



"""

X_train: train data

X_test: test data

y_train: training labels

y_test: test labels



X:feature data, 

y:target, 

test_size:train-test oranı, 

random_state:aynı sonucu verir denediğimiz randomstate Id'si.

"""

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=21) # X ve y'yi kullanarak train-test yaptık.

#.fit() 



knn.fit(X_train, y_train) #train data ile fit(yani train) yaptık. bu şu demek, datanın %70lik kısmının feature ve target(elimizde bilgi mevcut)

#valuelarını kullanarak modeli oluşturduk. 
#.predict()



y_pred = knn.predict(X_test) #X_test (yani kalan %30luk feature datası) nı kullanarak bu feature dataya karşılık gelen prediction'ları oluşturduk

#buna da y_pred dedik. elimizde şu an y_pred ve y_test(gerçek değerler) mevcut. yani artık modelin performansını ölçebiliriz.b
#.score()



knn.score(X_test, y_test) #score() ile modelin performansını ölçeriz. X_test'i verip, ondan elde edilen tahmin ile, y_test(gerçek değer)i kıyaslar.

"""

Overfitting - Underfitting: finding optimum k-value



burada, manual olarak belirlenmesi gereken n_neighbors(k) parametresi mevcut. (hyperparameter)

k değeri küçük olursa overfitting yapmış olabiliriz.

büyüdükçe ise underfiting yapmış olabiliriz. 

optimum k değerini bulmamız gerekir!



"""



# Setup arrays to store train and test accuracies

neighbors = np.arange(1, 20)

train_accuracy = np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



# Loop over different values of k

for i, k in enumerate(neighbors):

    # Setup a k-NN Classifier with k neighbors: knn

    knn = KNeighborsClassifier(n_neighbors=k)



    # Fit the classifier to the training data

    knn.fit(X_train, y_train)

    

    #Compute accuracy on the training set

    train_accuracy[i] = knn.score(X_train, y_train)



    #Compute accuracy on the testing set

    test_accuracy[i] = knn.score(X_test, y_test)



# Generate plot

plt.title('k-NN: Varying Number of Neighbors')

plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')

plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.show()



#grafiğe göre, testing acuracy 2-3-4-5 aynı sonucu veriyor, sonra düşüş var. n=5 ideal seçim diyebiliriz.
"""

İsmi regression olabilir, ama classification'da kullanılır.



Mekanizmayı özetlemek gerekirse, 

Log Reg bize bir p,probability verir. p>0.5 ise target=1, p<0.5 ise target=0 diye sonuçlanır.

log reg bize linear decision boundary sağlar. 



"""
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
logreg = LogisticRegression()
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
"""

yukarıda bahsettiğimiz p-value default 0.5, ancak bu değeri değiştirirsek ne olur:

mesela, p=0.0 olursa, herşeyi 1 tahmin eder, p=1.0 olursa da herşeyi 0 tahmin edecektir.

bu p-value'ya threshold deriz, bu threshold'un değişiminin etkisini incelediğimiz curve'e ise

ROC (Receiver Operating Characteristic) Curve deriz.

Ancak, ROC curve binary classification için geçerlidir. bu yüzden heart disease dataseti kullanacağız:



"""
df = pd.read_csv('../input/heart-disease-uci/heart.csv')
df.head()
X = df.drop('target', axis=1).values  

y = df['target'].values
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
logreg = LogisticRegression()
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
from sklearn.metrics import roc_curve
y_pred_prob = logreg.predict_proba(X_test)[:,1] #predict_proba 2 column döner, 1.si index olduğu için 2.columnı seçtik.
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob) #fpr:false pos.rate, tpr: true positive rate
#grafiği çizelim:

plt.plot([0,1], [0,1], 'k--')

plt.plot(fpr, tpr, label= 'Log Reg')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('LogReg ROC Curve')

plt.show()
""" Another metric for classification models.



ROC Curve'un altındaki alan (Area Under the ROC Curve - AUC) ne kadar fazla ise model o kadar iyi.



"""
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_prob)
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc') #scoring ile metodu belirledik.
print(cv_scores)
"""

Accuracy scores her zaman bize doğru bilgi vermez. çünkü;

örneğin, spam - not spam örneğinde, %99 real email, %1 spam iken hepsine real derse %99 doğru sonucu vermiş olur,

accuracy %99 olur. ama, bu bize modelin doğruluğu ile ilgili bilgiden ziyade, datanın içeriğinden kaynaklı bilgi vermiş olur.

yani, data imbalanced'tır. 



Confusion Matrix:

    Classification problemlerinde kullanılır. CM nedir:

    eğer email spam'se ve spam olarak tahmin edildiyse True Positive

    eğer email spam'se ve real olarak tahmin edildiyse False Negative

    eğer email real'se ve real olarak tahmin edildiyse True Negative

    eğer email real'se ve spam olarak tahmin edildiyse False Positive'dir.

    

    burada spam aradığımız için spam'e positive demiş olduk. bu bize bağlı.

    

    accuracy = (tp+tn) / (tp+tn+fp+fn)

    precision = tp / (tp+fp)

    recall (sensitivity) = tp / (tp+fn)

    f1-score = 2 * precision * recall / (precision + recall)

    

    high preision: not many real emails predicted as spam

    high recall: predicted most spam emails correctly.

    



"""
"""

yukarıda açıkladık zaten CM'i.

"""

#model import:

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

#yukarıdaki KNN modelini kullanarak Confusion Matrix'i örneklendirelim:

X = iris.drop('species', axis=1).values  

y = iris['species'].values
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=8) 
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=21)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred)) #-->>> ilk arguman true labellardan, ikinci argüman da predictionlardan oluştu.
print(classification_report(y_test, y_pred))
boston = pd.read_csv('../input/bostoncsv/Boston.csv')
boston.drop('Unnamed: 0', axis=1, inplace=True)

boston.head()



#target value medv'dir, diğerleri feature(predictor variable)
# X ve y yi belirleyelim:



X = boston.drop('medv', axis=1).values 

y = boston['medv'].values
#linear regression, y=a+bx. more dimensional olunca y = a+bx1+cx2... 

#amaç, bu line öyle bir line olsun ki, residuals'ı(error unction)(loss or cost function) 

#(data point'lerin line'dan uzaklıkları(nın kareleri)) toplamı minimum olsun.

#y target, x feature, a ve b de bizim öğrenmek istediğimiz parametreler.

#elde ettiğimiz, residualsın karelerini minimize eden functiona da OLS (ORDİNARY Least Squares) denir.
#sadece 1 column'ı alarak başlayalım; (one dimensional linear regression)

X_rooms = X[:,5]
#shape'lere bakalım:



X_rooms.shape
y.shape
#shape'leri istenen formata getirelim:



y = y.reshape(-1,1)
#istenen shape:



y.shape
X_rooms = X_rooms.reshape(-1,1)
X_rooms.shape



#tamamdır!..
#bir grafikle iki variable'ı görelim:



plt.scatter(X_rooms, y)

plt.ylabel('value of house')

plt.xlabel('nr of rooms')
#artık regression'a geldik:



from sklearn import linear_model #linear model'ı import ettik.

reg = linear_model.LinearRegression() #modeli reg variable'ına atadık.
reg.fit(X_rooms, y) #X_rooms ve y'ye göre fit ettik.
#fitting line'ı grafik olarak göstermek için,

prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1,1)

plt.scatter(X_rooms, y)

plt.plot(prediction_space, reg.predict(prediction_space), color='blue')
#ŞİMDİ, tüm datayı kullanarak more dimensional linear regression yapalım:



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=21)
reg_all = linear_model.LinearRegression()



reg_all.fit(X_train, y_train)



y_pred = reg_all.predict(X_test)



reg_all.score(X_test, y_test)
"""

Model performansı datanın nasıl split edildiğiyle alakalı olabilir.

Bu yüzden, bulduğumuz model performans score aldatıcı olabilir.

Bu sorunu çözmek için cross validation metodu kullanılır.



İşleyişi şu şekildedir:

test_size=0.2 ise, CV ile, farklı 0.2'lik gruplara ayırarak modeli test eder.

önce ilk 0.2'yi test olarak alır, sonra 2. 0.2lik kısmı...

bu şekilde 5 farklı train-test grubuna göre performans belirlenir.

bu, 5-fold CV olarak addedilir. 



k-fold CV 'deki k değerinin optimumunu bulmamız gerekir. (k büyük olursa işlem süresi artar, 

k küçük olursa da geçerliliğini ölçemeyiz)



"""
#import CV:

from sklearn.model_selection import cross_val_score 
#üstteki dataya göre 5-fold cv:

cv_results = cross_val_score(reg_all, X, y, cv=5) 
#sonuçlar:

cv_results
# 5 farklı sonucun ortalaması:

np.mean(cv_results)
"""

Linear Regression ile loss function'ı minimize etmektir fitting line'ın amacı.

y=a1x1+a2x2+... 'daki a1,a2,ai lerin seçimi yapılır buna göre. 

buradaki coefficient'ların fazla büyük seçimi overfitting getirir.

multi dimensional linear regression'ı düşünürsek bu durum sağlıklı sonuç almayı engeller.

bunu aşmak için large coefficientları penalize eden bir yapı ile loss function kontrol edilebilir.

İşte buna regularization deniyor.



"""
"""

Regularized Regression Types:



1. Ridge Regression 



(Loss function = OLS loss function + alfa* sum of squared values of each coefficient) 



(bu şekilde, Loss function coefficientlerin karelerinin toplamlarının alfa ile çarpımı kadar artmıştır. 

amaç ikinci kısmın da min olmasını sağlamaktır.)



(alpha, hyperparameter'dır.)



2. Lasso Regression



(Loss function = OLS loss function + alfa* sum of absolute values of each coefficient) !!!Tek fark bu!!!



!! Lasso regression, important feature seçimi yapılmakta kullanılır. Çünkü, küçük coefficientleri 0'a küçültme eğilimindedir.

0'a yuvarlanmamış feature'lar important olarak tanımlanabilir.



"""
#importing

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1, normalize=True) #modeli oluşturduk. normalize=True tüm variable'lar aynı scale'da olması için.
ridge.fit(X_train, y_train) #train ettik
ridge_pred= ridge.predict(X_test)
ridge.score(X_test, y_test)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)
#yukarıda bahsettiğimiz, önemsiz feature'ları 0 yapıp önemli olanları belirleme işlemi:

lasso_coef = lasso.fit(X, y).coef_
lasso_coef
"""

Modeldeki hyperparameterları belirterek grid search yaparak hyperparameterların optimum değerini bulmamızı sağlar.

"""
iris = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
X = iris.drop('species', axis=1).values  

y = iris['species'].values
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arange(1,50)} #dict içinde '' içinde grid edilecek parametreyi yazıp, karşılığında da hangi değerlerin 

#deneneceğini yazdık.
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5) #this return the grid search object. fit etmeliyiz
knn_cv.fit(X,y)
knn_cv.best_params_
knn_cv.best_score_
"""

GridSearchCV'den farkı, çoklu hyperparameter belirlemede calculation yoğunluğu olduğu zaman kullanılmasıdır.

Tüm değerler denenmez, bir prob.dist kullanılarak hesaplama yapılır.



Bu uygulamayı yapmak için RandomForest kullanalım:

"""
from scipy.stats import randint

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import RandomizedSearchCV
# Setup the parameters and distributions to sample from: param_dist

param_dist = {"max_depth": [3, None],

              "min_samples_leaf": randint(1, 9),

              "criterion": ["gini", "entropy"]}
# Instantiate a Decision Tree classifier: tree

tree = DecisionTreeClassifier()
# Instantiate the RandomizedSearchCV object: tree_cv

tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
# Fit it to the data

tree_cv.fit(X, y)
# Print the tuned parameters and score

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))

print("Best score is {}".format(tree_cv.best_score_))
"""

categorical data type için, 







"""
df = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
df.head()
df_species = pd.get_dummies(df)
df_species.head()
"""

3'ünden 2si değilse diğeridir. bu yüzden kullandığımız bir parametre var.

"""
df_species2 = pd.get_dummies(df, drop_first=True)
df_species2.head()
"""

bazen, missing value olarak görülmez, 0, '', vs vs olur.

bu durumda, hem nan value^ları smartly doldurmak için Imputer object kullanabiliriz.



"""
df = pd.read_csv('../input/diabetescsv/diabetes.csv')
df.head() #mesela burada, insulin = 0, bu not possible
df.Insulin.replace(0, np.nan, inplace=True) #TÜM 0 ları nan value yaptık.

df.SkinThickness.replace(0, np.nan, inplace=True)

df.BMI.replace(0, np.nan, inplace=True)
df.head()
df.info()
# df = df.dropna() # missing value olan tüm row'ları drop ettik.
# df.shape #ancak, datanın yarısını kaybettik. unacceptible
from sklearn.linear_model import LogisticRegression



#nan value'ların yerine başka şeyler koyabiliriz stratejiye bağlı olarak:

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #missing_values = 'Nan' olarak gösterimiş demek. axis=0 ile de sadece o column'a baktık.
X = df.drop('Outcome', axis=1).values  

y = df['Outcome'].values
imp.fit(X)
X = imp.transform(X)
"""

Imputer'ın yaptığı işi yapar, üstüne bir de model çalıştırır.

"""
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

logreg = LogisticRegression()
steps = [('imputation', imp), 

         ('logistic_regression', logreg)] #imputer modeli ve uygulanacak reg modelini steps e yazdık
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=21)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
pipeline.score(X_test, y_test)
"""

Farklı feature'lar için farklı range'de değerlerin olması, ML modellerinin çalışmasını etkileyecektir.

Eğer scaling(normalizing) yapmazsak bazı feature'lar modelde daha ağırlıklı değerlendirilecek, 

sonuçlar yanıltıcı olacaktır.



"""
iris = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
iris.head()
X =iris.drop('species', axis=1).values  

y = iris['species'].values
from sklearn.preprocessing import scale
X_scaled = scale(X)
np.mean(X), np.std(X)
np.mean(X_scaled), np.std(X_scaled)
""" burada da, pipeline ile hem imputation, hem scaling, hem model çalışması,

    ardından da gridsearchcv ile hyperparameter optimization bir arada.

    

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Imputer

from sklearn.pipeline import Pipeline



# Setup the pipeline steps: steps

steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),

         ('scaler', StandardScaler()),

         ('elasticnet', ElasticNet())]



# Create the pipeline: pipeline 

pipeline = Pipeline(steps)



# Specify the hyperparameter space

parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)



# Create the GridSearchCV object: gm_cv

gm_cv = GridSearchCV(pipeline, parameters)



# Fit to the training set

gm_cv.fit(X_train, y_train)



# Compute and print the metrics

print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))



"""