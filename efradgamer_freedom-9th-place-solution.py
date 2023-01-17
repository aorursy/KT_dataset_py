
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/datmin-joints-2020/train_data.csv')
test = pd.read_csv('/kaggle/input/datmin-joints-2020/test_data.csv')
submission = pd.read_csv('/kaggle/input/datmin-joints-2020/sample_submission.csv')

# Keanehan nilai dari setiap kolom
kolom_huruf = [('word-1','a','588184'), ('word-5','4+F2185','11h','a'), ('word-8','a','`16'),('word-9','`','`29'), ('word-10','`10'), ('word-11','\\'),('word-13','a','`27'),
              ('word-14','`3'),('word-17','a','`4'),('word-18','\\','a'),('word-19','`155','`363','`361', '`51','`764','`84','994821','536457','`61'),
              ('word-22','31415669','`21','`11'),('word-24','`2'), ('word-27','`2','['), ('word-28','a'), ('word-30','a'),('word-33','`10','`18','a'),('word-35','`3'),
              ('word-38','412334','`10'),('word-39','`1'),('word-40','a')]

'''
Dapat diambil kesimpulan jika dikelompokkan, nilai unique dari setiap kesalahan input adalah :
1. 'a' --> dibikin nan atau nilai tertentu ?
2. '`' --> Paling banyak, nanti di hilangkan maka itu nilai yang asli
3. '\\' --> sedikit, fix NaN
4.  '[' --> sedikit, fix NaN
5. 31415669 --> kasus angka sebanyak ini, dijadikan nan atau nilai tertentu ?
6. 4+F2185 --> Fix NaN


'''

# Membuat dictionary, dimana keynya berupa nomor kolom, dan valuenya adalah nilai kolom_huruf
# Ini digunakan untuk looping
kolom_huruf = {1:kolom_huruf[0],5:kolom_huruf[1],8:kolom_huruf[2],9:kolom_huruf[3],10:kolom_huruf[4],11:kolom_huruf[5],13:kolom_huruf[6],14:kolom_huruf[7],17:kolom_huruf[8],
              18:kolom_huruf[9],19:kolom_huruf[10], 22:kolom_huruf[11], 24:kolom_huruf[12], 27:kolom_huruf[13], 28:kolom_huruf[14], 30:kolom_huruf[15], 33:kolom_huruf[16],
              35:kolom_huruf[17],38:kolom_huruf[18],39:kolom_huruf[19],40:kolom_huruf[20]}
# Membuat fungsi mengganti huruf atau benerin nilai yang inputnya salah
def ganti_huruf(kolom,huruf):
    
    nomor_row = 0
    for i in range(train.shape[0]):
        if train.iloc[i,kolom] == huruf:
            # Menyimpan nomor row untuk mengetahui row keberapa dimana a itu berada
            #print('nomor_kolom :',i)
            nomor_row = i 
 
    # Test apakah row tersebut bener atau gak
   # print('testing apakah bener:',train.iloc[nomor_row,kolom])
    
    # Ganti huruf tersebut menjadi nan atau hilangkan tanda '`'
    if kolom == 9:
        if '`' in huruf:
            train.iloc[nomor_row,kolom] = np.nan
            
    elif 'a' in huruf:
        train.iloc[nomor_row,kolom] = 0
        
    elif  '`' in huruf:
        train.iloc[nomor_row,kolom] = huruf[1:] 
        
    elif 'h' in huruf:
        train.iloc[nomor_row,kolom] = huruf[:-1] # untuk nangkep word-5 --> '11h'
    
    else:
        #print(kolom,huruf)
        train.iloc[nomor_row,kolom] = np.nan
   # print('Berhasil !!!')
# Looping fungsi ganti_huruf untuk setiap kolom_huruf !
for key,value in kolom_huruf.items():
    
    # Dibuat error excepition karena nanti akan terjadi IndexError !
    try :
        for j in range(len(value)):
            
            # key --> kolom, value --> huruf
            ganti_huruf(key,value[j+1])
            
    except:
        pass
# Kolom 8 dan 13, entah kenapa tidak berhasil untuk mengganti huruf a diatas ! jadi dilakukan secara terpisah dari looping
ganti_huruf(8,'a')
ganti_huruf(13,'a')
# Missing Values
train = train.fillna(0)
kolom = ['word-1', 'word-2', 'word-3', 'word-4', 'word-5', 'word-6',
       'word-7', 'word-8', 'word-9', 'word-10', 'word-11', 'word-12',
       'word-13', 'word-14', 'word-15', 'word-16', 'word-17', 'word-18',
       'word-19', 'word-20', 'word-21', 'word-22', 'word-23', 'word-24',
       'word-25', 'word-26', 'word-27', 'word-28', 'word-29', 'word-30',
       'word-31', 'word-32', 'word-33', 'word-34', 'word-35', 'word-36',
       'word-37', 'word-38', 'word-39', 'word-40']

# Mengganti tipe data object ke float
for i in kolom:

    train[i] = train[i].astype('int')
    
train.info()
# Tambahan data error ! 
train.loc[32, 'word-4'] = 1 # eror karena -1
train.loc[24, 'word-25'] = 9 # error karena -9
train.loc[3220, 'word-36'] = 0 # error karena nilainya sangat besar, contoh : 1295321
test = test.fillna(0)
train_ = train.drop(columns = 'Result')
full = pd.concat([train_,test])
full = full.drop(columns = 'id')
full.head()
full1 = full.copy()

X_train_full1 = full1.iloc[:3620] 
X_test_full1 = full1.iloc[3620:] 

X1 = X_train_full1
y1 = train.Result
X1.head()
full2 = full.copy()

from array import array
tf = 1/full2.sum(axis=1)[:,None]
full2 = full2*tf

idf = np.log(full2.shape[0]/ (full2>0).sum())
full2 = full2*idf

full2 = full2.fillna(0)

X_train_full2 = full2.iloc[:3620] 
X_test_full2 = full2.iloc[3620:] 

X2 = X_train_full2
y2 = train.Result
X2.head()
from numpy import array
from sklearn.decomposition import PCA
full3 = full.copy()

binary = list()
for i in range(len(full3)):
    bi = list()
    for j in full3.iloc[i,]:
        if j != 0:
            bi.append(1)
        else: bi.append(0)
    binary.append(bi)
    
ha = pd.DataFrame(binary, columns = full.columns)

for i in ha.columns:
        full3[i+'binary'] = ha[i]

X_train_full3 = full3.iloc[:3620] 
X_test_full3 = full3.iloc[3620:] 

X3 = X_train_full3
y3 = train.Result

X3.head()
full4 = full.copy()
std_words = list()
for j in range(len(full4)): 
    std_words.append(full4.iloc[j].std())
    
std_words = pd.Series(std_words)


jumlah_kata = list()
for i in range(len(full4)):
    jumlah_kata.append(full4.iloc[i,].sum())
               
Total_Kata = pd.Series(jumlah_kata)

unique_words = list()
jumlah_unique = 0
for j in range(len(full4)):
    for key,i in full4.iloc[j].items():
        if key != 'Result':
            if i == 1:
                jumlah_unique += 1
    unique_words.append(jumlah_unique)
    jumlah_unique = 0
    
unique_words = pd.Series(unique_words)
    
absence_words = list()
jumlah_absence = 0
for j in range(len(full4)):
    for key,i in full4.iloc[j].items():
        if key != 'Result':
            if i == 0:
                jumlah_absence += 1
    absence_words.append(jumlah_absence)
    jumlah_absence = 0
absence_words = pd.Series(absence_words)

from array import array
tf = 1/full4.sum(axis=1)[:,None]
full4 = full4*tf

idf = np.log(full4.shape[0]/ (full4>0).sum())
full4 = full4*idf

full4 = full4.fillna(0)
full4['Kata_Unik'] = unique_words
full4['Persentase_Unik_vs_TotalKata'] = (pd.Series(unique_words).multiply(100))/pd.Series(Total_Kata)
full4['Std_Kata'] = std_words
full4['Absen_Kata'] = absence_words
full4['Total_Kata'] = Total_Kata
full4['multiply_18_34'] = full['word-18'].multiply(full['word-34'])
full4['multiply_15_33'] = full['word-15'].multiply(full['word-33'])
full4 = full4.fillna(0)
X_train_full4 = full4.iloc[:3620] 
X_test_full4 = full4.iloc[3620:] 

X4 = X_train_full4
y4 = train.Result
X4.head()
full5 = full.copy()

jumlah_kata = list()
for i in range(len(full5)):
    jumlah_kata.append(full5.iloc[i,].sum())
               
Total_Kata = pd.Series(jumlah_kata)

full5['multiply_18_34'] = full4['word-18'].multiply(full4['word-34'])
full5['Total_Kata'] = Total_Kata
full5['multiply_15_33'] = full['word-15'].multiply(full['word-33'])
full5 = full5.fillna(0)
X_train_full5 = full5.iloc[:3620] 
X_test_full5 = full5.iloc[3620:] 

X5 = X_train_full5
y5 = train.Result
X5.head()

from sklearn.model_selection import StratifiedKFold

# Some useful parameters which will come in handy later on
ntrain = full.iloc[:3620].shape[0]
ntest = full.iloc[3620:].shape[0]
SEED = 0 # for reproducibility
NFOLDS = 3 # set folds for out-of-fold prediction
skf = StratifiedKFold(n_splits=NFOLDS, random_state=SEED)
# clf --> adalah model yang ingin digunakan
# x_tr --> adalah tipe dataset yang ingin digunakan, seluruh data_train
# y_tr --> adalah variabel target
# x_test --> adalah test set untuk di lakukan prediksi, lalu di submission
def get_oof(clf, x_tr, y_tr, x_test, deeplearning = False, deep = False):
    count = 0
    
    # BUAT VECTOR UNTUK MENYIMPAN HASIL CV 
    oof_train = np.zeros((ntrain,))
    oof_train_ = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    oof_y_valid = np.zeros((ntrain,))
    
    # Cross validasi
    for train_index, test_index in skf.split(x_tr,y_tr):
        X_train, X_valid = x_tr.iloc[train_index], x_tr.iloc[test_index] 
        y_train, y_valid = y_tr.iloc[train_index], y_tr.iloc[test_index]

        if deeplearning == True:
            clf.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
            clf.fit(X_train,y_train,batch_size=25,epochs=10,validation_split=0.3,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
            oof_train[test_index] = clf.predict(X_valid)[:,0]
            oof_test_skf[count,:] = clf.predict(x_test)[:,0]
        elif deeplearning == True & deep == True:
            clf.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

            clf.fit(X_train,y_train, batch_size=16, epochs=30,validation_data=(X_valid, y_valid),callbacks=[es, rlr],verbose=1)
            oof_train[test_index] = clf.predict(X_valid)[:,0]
            oof_test_skf[count,:] = clf.predict(x_test)[:,0]
            
        else:
            clf.fit(X_train, y_train)
            oof_train[test_index] = clf.predict_proba(X_valid)[:,1]
            oof_test_skf[count,:] = clf.predict_proba(x_test)[:,1]
            oof_train_[test_index] = clf.predict(X_valid)
            oof_y_valid[test_index] = y_valid
            
            if count > 1:
                oof_y_pred = pd.Series(oof_train_.ravel())
                tn, fp, fn, tp = confusion_matrix(oof_y_valid, oof_y_pred).ravel()
                #Metric yang digunakan
                categorization_accuracy = (tp+tn)/(tp+tn+fp+fn)
                print(categorization_accuracy)
        count+=1
        
   
    oof_test[:] = oof_test_skf.mean(axis=0)
    # Output adalah matriks yg berisikan nilai prediksi
    # oof_train --> isinya hasil prediksi dari seluruh cross validation
    # oof_test --> isinya hasil prediksi data test (x_test)
    # oof_y_valid --> isinya adalah seluruh target y yang asli saat di cross validation
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1), oof_y_valid
y = train.Result
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from catboost import CatBoostClassifier

ext = ExtraTreesClassifier()
svc = SVC(kernel='sigmoid', gamma=1.0, probability=True)
lgr = LogisticRegression()
nb = MultinomialNB()
xgb = XGBClassifier(n_estimators=250,scale_pos_weight =  (y.shape[0]-y.sum()) / y.sum())
lgbm = LGBMClassifier(n_estimators = 250, scale_pos_weight=(y.shape[0]-y.sum()) / y.sum())

cat = CatBoostClassifier()
lda = discriminant_analysis.LinearDiscriminantAnalysis()
qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
dct = tree.DecisionTreeClassifier()
nsvc =svm.NuSVC(probability=True)
lsvc =  svm.LinearSVC()
knc = neighbors.KNeighborsClassifier()
ber =  naive_bayes.BernoulliNB()
gau =  naive_bayes.GaussianNB()
pac = linear_model.PassiveAggressiveClassifier()
rid =   linear_model.RidgeClassifierCV()
sgdc =  linear_model.SGDClassifier()
per = linear_model.Perceptron()
gaup = gaussian_process.GaussianProcessClassifier()
gbc = ensemble.GradientBoostingClassifier()
rf = ensemble.RandomForestClassifier()
ada = ensemble.AdaBoostClassifier()
bag = ensemble.BaggingClassifier()

# Mengambil hasil prediksi untuk data_train, data_test per model

print('catboost score:')
xgb_oof_train191, xgb_oof_test191,oof_y_valid191 = get_oof(bag,X_train_full1, y, X_test_full1) # Bagging
xgb_oof_train192, xgb_oof_test192,oof_y_valid192 = get_oof(bag,X_train_full2, y, X_test_full2) # Bagging
xgb_oof_train193, xgb_oof_test193,oof_y_valid193 = get_oof(bag,X_train_full3, y, X_test_full3) # Bagging
xgb_oof_train194, xgb_oof_test194,oof_y_valid194 = get_oof(bag,X_train_full4, y, X_test_full4) # Bagging
xgb_oof_train195, xgb_oof_test195,oof_y_valid195 = get_oof(bag,X_train_full5, y, X_test_full5) # Bagging


print('xgbg score:')
xgb_oof_train11, xgb_oof_test11,oof_y_valid11 = get_oof(xgb,X_train_full1, y, X_test_full1) # Xtreme Gradient Boost
xgb_oof_train12, xgb_oof_test12,oof_y_valid12 = get_oof(xgb,X_train_full2, y, X_test_full2) # Xtreme Gradient Boost
xgb_oof_train13, xgb_oof_test13,oof_y_valid13 = get_oof(xgb,X_train_full3, y, X_test_full3) # Xtreme Gradient Boost
xgb_oof_train14, xgb_oof_test14,oof_y_valid14 = get_oof(xgb,X_train_full4, y, X_test_full4) # Xtreme Gradient Boost
xgb_oof_train15, xgb_oof_test15,oof_y_valid15 = get_oof(xgb,X_train_full5, y, X_test_full5) # Xtreme Gradient Boost

print('lgbm score:')
xgb_oof_train21, xgb_oof_test21,oof_y_valid21 = get_oof(lgbm,X_train_full1, y, X_test_full1) # LGBM
xgb_oof_train22, xgb_oof_test22,oof_y_valid22 = get_oof(lgbm,X_train_full2, y, X_test_full2) # LGBM
xgb_oof_train23, xgb_oof_test23,oof_y_valid23 = get_oof(lgbm,X_train_full3, y, X_test_full3) # LGBM
xgb_oof_train24, xgb_oof_test24,oof_y_valid24 = get_oof(lgbm,X_train_full4, y, X_test_full4) # LGBM
xgb_oof_train25, xgb_oof_test25,oof_y_valid25 = get_oof(lgbm,X_train_full5, y, X_test_full5) # LGBM

print('multinomial score:')
xgb_oof_train31, xgb_oof_test31,oof_y_valid31 = get_oof(nb,X_train_full1, y, X_test_full1) # MultinomialNB
xgb_oof_train32, xgb_oof_test32,oof_y_valid32 = get_oof(nb,X_train_full2, y, X_test_full2) # MultinomialNB
# xgb_oof_train33, xgb_oof_test33,oof_y_valid33 = get_oof(nb,X_train_full3, y, X_test_full3) # MultinomialNB
xgb_oof_train34, xgb_oof_test34,oof_y_valid34 = get_oof(nb,X_train_full4, y, X_test_full4) # MultinomialNB
xgb_oof_train35, xgb_oof_test35,oof_y_valid35 = get_oof(nb,X_train_full5, y, X_test_full5) # MultinomialNB

print('Logistic regression score:')
xgb_oof_train41, xgb_oof_test41,oof_y_valid41 = get_oof(lgr,X_train_full1, y, X_test_full1) # LogisticRegression
xgb_oof_train42, xgb_oof_test42,oof_y_valid42 = get_oof(lgr,X_train_full2, y, X_test_full2) # LogisticRegression
xgb_oof_train43, xgb_oof_test43,oof_y_valid43 = get_oof(lgr,X_train_full3, y, X_test_full3) # LogisticRegression
xgb_oof_train44, xgb_oof_test44,oof_y_valid44 = get_oof(lgr,X_train_full4, y, X_test_full4) # LogisticRegression
xgb_oof_train45, xgb_oof_test45,oof_y_valid45 = get_oof(lgr,X_train_full5, y, X_test_full5) # LogisticRegression

print('svm score:')
xgb_oof_train51, xgb_oof_test51,oof_y_valid51 = get_oof(svc,X_train_full1, y, X_test_full1) # SVM
xgb_oof_train52, xgb_oof_test52,oof_y_valid52 = get_oof(svc,X_train_full2, y, X_test_full2) # SVM
xgb_oof_train53, xgb_oof_test53,oof_y_valid53 = get_oof(svc,X_train_full3, y, X_test_full3) # SVM
xgb_oof_train54, xgb_oof_test54,oof_y_valid54 = get_oof(svc,X_train_full4, y, X_test_full4) # SVM
xgb_oof_train55, xgb_oof_test55,oof_y_valid55 = get_oof(svc,X_train_full5, y, X_test_full5) # SVM

print('extra tree:')
xgb_oof_train61, xgb_oof_test61,oof_y_valid61 = get_oof(ext,X_train_full1, y, X_test_full1) # ExtraTree
xgb_oof_train62, xgb_oof_test62,oof_y_valid62 = get_oof(ext,X_train_full2, y, X_test_full2) # ExtraTree
xgb_oof_train63, xgb_oof_test63,oof_y_valid63 = get_oof(ext,X_train_full3, y, X_test_full3) # ExtraTree
xgb_oof_train64, xgb_oof_test64,oof_y_valid64 = get_oof(ext,X_train_full4, y, X_test_full4) # ExtraTree
xgb_oof_train65, xgb_oof_test65,oof_y_valid65 = get_oof(ext,X_train_full5, y, X_test_full5) # ExtraTree

print('lda score:')
xgb_oof_train71, xgb_oof_test71,oof_y_valid71 = get_oof(lda,X_train_full1, y, X_test_full1) # LinearDiscriminantAnalysis
xgb_oof_train72, xgb_oof_test72,oof_y_valid72 = get_oof(lda,X_train_full2, y, X_test_full2) # LinearDiscriminantAnalysis
xgb_oof_train73, xgb_oof_test73,oof_y_valid73 = get_oof(lda,X_train_full3, y, X_test_full3) # LinearDiscriminantAnalysis
xgb_oof_train74, xgb_oof_test74,oof_y_valid74 = get_oof(lda,X_train_full4, y, X_test_full4) # LinearDiscriminantAnalysis
xgb_oof_train75, xgb_oof_test75,oof_y_valid75 = get_oof(lda,X_train_full5, y, X_test_full5) # LinearDiscriminantAnalysis

print('qda score:')
xgb_oof_train81, xgb_oof_test81,oof_y_valid81 = get_oof(qda,X_train_full1, y, X_test_full1) # QuadraticDiscriminantAnalysis
xgb_oof_train82, xgb_oof_test82,oof_y_valid82 = get_oof(qda,X_train_full2, y, X_test_full2) # QuadraticDiscriminantAnalysis
xgb_oof_train83, xgb_oof_test83,oof_y_valid83 = get_oof(qda,X_train_full3, y, X_test_full3) # QuadraticDiscriminantAnalysis
xgb_oof_train84, xgb_oof_test84,oof_y_valid84 = get_oof(qda,X_train_full4, y, X_test_full4) # QuadraticDiscriminantAnalysis
xgb_oof_train85, xgb_oof_test85,oof_y_valid85 = get_oof(qda,X_train_full5, y, X_test_full5) # QuadraticDiscriminantAnalysis

print('decision tree score:')
xgb_oof_train91, xgb_oof_test91,oof_y_valid91 = get_oof(dct,X_train_full1, y, X_test_full1) # DecisionTree
xgb_oof_train92, xgb_oof_test92,oof_y_valid92 = get_oof(dct,X_train_full2, y, X_test_full2) # DecisionTree
xgb_oof_train93, xgb_oof_test93,oof_y_valid93 = get_oof(dct,X_train_full3, y, X_test_full3) # DecisionTree
xgb_oof_train94, xgb_oof_test94,oof_y_valid94 = get_oof(dct,X_train_full4, y, X_test_full4) # DecisionTree
xgb_oof_train95, xgb_oof_test95,oof_y_valid95 = get_oof(dct,X_train_full5, y, X_test_full5) # DecisionTree


print('nusvc score:')
xgb_oof_train101, xgb_oof_test101,oof_y_valid101 = get_oof(nsvc,X_train_full1, y, X_test_full1) # NuSVC
xgb_oof_train102, xgb_oof_test102,oof_y_valid102 = get_oof(nsvc,X_train_full2, y, X_test_full2) # NuSVC
xgb_oof_train103, xgb_oof_test103,oof_y_valid103 = get_oof(nsvc,X_train_full3, y, X_test_full3) # NuSVC
xgb_oof_train104, xgb_oof_test104,oof_y_valid104 = get_oof(nsvc,X_train_full4, y, X_test_full4) # NuSVC
xgb_oof_train105, xgb_oof_test105,oof_y_valid105 = get_oof(nsvc,X_train_full5, y, X_test_full5) # NuSVC

print('kneighbors score:')
xgb_oof_train111, xgb_oof_test111,oof_y_valid111 = get_oof(knc,X_train_full1, y, X_test_full1) # KNeighborsClassifier
xgb_oof_train112, xgb_oof_test112,oof_y_valid112 = get_oof(knc,X_train_full2, y, X_test_full2) # KNeighborsClassifier
xgb_oof_train113, xgb_oof_test113,oof_y_valid113 = get_oof(knc,X_train_full3, y, X_test_full3) # KNeighborsClassifier
xgb_oof_train114, xgb_oof_test114,oof_y_valid114 = get_oof(knc,X_train_full4, y, X_test_full4) # KNeighborsClassifier
xgb_oof_train115, xgb_oof_test115,oof_y_valid115 = get_oof(knc,X_train_full5, y, X_test_full5) # KNeighborsClassifier

print('bernouli score:')
xgb_oof_train121, xgb_oof_test121,oof_y_valid121 = get_oof(ber,X_train_full1, y, X_test_full1) # BernoulliNB
xgb_oof_train122, xgb_oof_test122,oof_y_valid122 = get_oof(ber,X_train_full2, y, X_test_full2) # BernoulliNB
xgb_oof_train123, xgb_oof_test123,oof_y_valid123 = get_oof(ber,X_train_full3, y, X_test_full3) # BernoulliNB
xgb_oof_train124, xgb_oof_test124,oof_y_valid124 = get_oof(ber,X_train_full4, y, X_test_full4) # BernoulliNB
xgb_oof_train125, xgb_oof_test125,oof_y_valid125 = get_oof(ber,X_train_full5, y, X_test_full5) # BernoulliNB

print('gaussian score:')
xgb_oof_train131, xgb_oof_test131,oof_y_valid131 = get_oof(gau,X_train_full1, y, X_test_full1) # GaussianNB
xgb_oof_train132, xgb_oof_test132,oof_y_valid132 = get_oof(gau,X_train_full2, y, X_test_full2) # GaussianNB
xgb_oof_train133, xgb_oof_test133,oof_y_valid133 = get_oof(gau,X_train_full3, y, X_test_full3) # GaussianNB
xgb_oof_train134, xgb_oof_test134,oof_y_valid134 = get_oof(gau,X_train_full4, y, X_test_full4) # GaussianNB
xgb_oof_train135, xgb_oof_test135,oof_y_valid135 = get_oof(gau,X_train_full5, y, X_test_full5) # GaussianNB

print('gausian process score:')
xgb_oof_train141, xgb_oof_test141,oof_y_valid141 = get_oof(gaup,X_train_full1, y, X_test_full1) # GaussianProcess
xgb_oof_train142, xgb_oof_test142,oof_y_valid142 = get_oof(gaup,X_train_full2, y, X_test_full2) # GaussianProcess
xgb_oof_train143, xgb_oof_test143,oof_y_valid143 = get_oof(gaup,X_train_full3, y, X_test_full3) # GaussianProcess
xgb_oof_train144, xgb_oof_test144,oof_y_valid144 = get_oof(gaup,X_train_full4, y, X_test_full4) # GaussianProcess
xgb_oof_train145, xgb_oof_test145,oof_y_valid145 = get_oof(gaup,X_train_full5, y, X_test_full5) # GaussianProcess

print('gradient boosting score:')
xgb_oof_train151, xgb_oof_test151,oof_y_valid151 = get_oof(gbc,X_train_full1, y, X_test_full1) # GradientBoosting
xgb_oof_train152, xgb_oof_test152,oof_y_valid152 = get_oof(gbc,X_train_full2, y, X_test_full2) # GradientBoosting
xgb_oof_train153, xgb_oof_test153,oof_y_valid153 = get_oof(gbc,X_train_full3, y, X_test_full3) # GradientBoosting
xgb_oof_train154, xgb_oof_test154,oof_y_valid154 = get_oof(gbc,X_train_full4, y, X_test_full4) # GradientBoosting
xgb_oof_train155, xgb_oof_test155,oof_y_valid155 = get_oof(gbc,X_train_full5, y, X_test_full5) # GradientBoosting

print('random foreset score:')
xgb_oof_train161, xgb_oof_test161,oof_y_valid161 = get_oof(rf,X_train_full1, y, X_test_full1) # RandomForest
xgb_oof_train162, xgb_oof_test162,oof_y_valid162 = get_oof(rf,X_train_full2, y, X_test_full2) # RandomForest
xgb_oof_train163, xgb_oof_test163,oof_y_valid163 = get_oof(rf,X_train_full3, y, X_test_full3) # RandomForest
xgb_oof_train164, xgb_oof_test164,oof_y_valid164 = get_oof(rf,X_train_full4, y, X_test_full4) # RandomForest
xgb_oof_train165, xgb_oof_test165,oof_y_valid165 = get_oof(rf,X_train_full5, y, X_test_full5) # RandomForest

print('ada boost score:')
xgb_oof_train171, xgb_oof_test171,oof_y_valid171 = get_oof(ada,X_train_full1, y, X_test_full1) # AdaBoost
xgb_oof_train172, xgb_oof_test172,oof_y_valid172 = get_oof(ada,X_train_full2, y, X_test_full2) # AdaBoost
xgb_oof_train173, xgb_oof_test173,oof_y_valid173 = get_oof(ada,X_train_full3, y, X_test_full3) # AdaBoost
xgb_oof_train174, xgb_oof_test174,oof_y_valid174 = get_oof(ada,X_train_full4, y, X_test_full4) # AdaBoost
xgb_oof_train175, xgb_oof_test175,oof_y_valid175 = get_oof(ada,X_train_full5, y, X_test_full5) # AdaBoost

print('baggin score:')
xgb_oof_train181, xgb_oof_test181,oof_y_valid181 = get_oof(bag,X_train_full1, y, X_test_full1) # Bagging
xgb_oof_train182, xgb_oof_test182,oof_y_valid182 = get_oof(bag,X_train_full2, y, X_test_full2) # Bagging
xgb_oof_train183, xgb_oof_test183,oof_y_valid183 = get_oof(bag,X_train_full3, y, X_test_full3) # Bagging
xgb_oof_train184, xgb_oof_test184,oof_y_valid184 = get_oof(bag,X_train_full4, y, X_test_full4) # Bagging
xgb_oof_train185, xgb_oof_test185,oof_y_valid185 = get_oof(bag,X_train_full5, y, X_test_full5) # Bagging

# Membuat dataframe dari seluruh hasil prediksi data train
base_predictions_train = pd.DataFrame( {

    'Model11': xgb_oof_train11.ravel(),
    'Model12': xgb_oof_train12.ravel(),
    'Model13': xgb_oof_train13.ravel(),
    'Model14': xgb_oof_train14.ravel(),
    'Model15': xgb_oof_train15.ravel(),

    'Model21': xgb_oof_train21.ravel(),
    'Model22': xgb_oof_train22.ravel(),
    'Model23': xgb_oof_train23.ravel(),
    'Model24': xgb_oof_train24.ravel(),
    'Model25': xgb_oof_train25.ravel(),
    
    'Model31': xgb_oof_train31.ravel(),
    'Model32': xgb_oof_train32.ravel(),
#     'Model33': xgb_oof_train33.ravel(),
    'Model34': xgb_oof_train34.ravel(),
    'Model35': xgb_oof_train35.ravel(),
    
    'Model41': xgb_oof_train41.ravel(),
    'Model42': xgb_oof_train42.ravel(),
    'Model43': xgb_oof_train43.ravel(),
    'Model44': xgb_oof_train44.ravel(),
    'Model45': xgb_oof_train45.ravel(),
    
    
    'Model51': xgb_oof_train51.ravel(),
    'Model52': xgb_oof_train52.ravel(),
    'Model53': xgb_oof_train53.ravel(),
    'Model54': xgb_oof_train54.ravel(),
    'Model55': xgb_oof_train55.ravel(),
    
    'Model61': xgb_oof_train61.ravel(),
    'Model62': xgb_oof_train62.ravel(),
    'Model63': xgb_oof_train63.ravel(),
    'Model64': xgb_oof_train64.ravel(),
    'Model65': xgb_oof_train65.ravel(),
    
    
    'Model71': xgb_oof_train71.ravel(),
    'Model72': xgb_oof_train72.ravel(),
    'Model73': xgb_oof_train73.ravel(),
    'Model74': xgb_oof_train74.ravel(),
    'Model75': xgb_oof_train75.ravel(),
   
    'Model81': xgb_oof_train81.ravel(),
    'Model82': xgb_oof_train82.ravel(),
    'Model83': xgb_oof_train83.ravel(),
    'Model84': xgb_oof_train84.ravel(),
    'Model85': xgb_oof_train85.ravel(),
    
    'Model91': xgb_oof_train91.ravel(),
    'Model92': xgb_oof_train92.ravel(),
    'Model93': xgb_oof_train93.ravel(),
    'Model94': xgb_oof_train94.ravel(),
    'Model95': xgb_oof_train95.ravel(),
    
    'Model101': xgb_oof_train101.ravel(),
    'Model102': xgb_oof_train102.ravel(),
    'Model103': xgb_oof_train103.ravel(),
    'Model104': xgb_oof_train104.ravel(),
    'Model105': xgb_oof_train105.ravel(),
    
    
    'Model111': xgb_oof_train111.ravel(),
    'Model112': xgb_oof_train112.ravel(),
    'Model113': xgb_oof_train113.ravel(),
    'Model114': xgb_oof_train114.ravel(),
    'Model115': xgb_oof_train115.ravel(),
    
    'Model121': xgb_oof_train121.ravel(),
    'Model122': xgb_oof_train122.ravel(),
    'Model123': xgb_oof_train123.ravel(),
    'Model124': xgb_oof_train124.ravel(),
    'Model125': xgb_oof_train125.ravel(),
    
    
    'Model131': xgb_oof_train131.ravel(),
    'Model132': xgb_oof_train132.ravel(),
    'Model133': xgb_oof_train133.ravel(),
    'Model134': xgb_oof_train134.ravel(),
    'Model135': xgb_oof_train135.ravel(),
    
    
    'Model141': xgb_oof_train141.ravel(),
    'Model142': xgb_oof_train142.ravel(),
    'Model143': xgb_oof_train143.ravel(),
    'Model144': xgb_oof_train144.ravel(),
    'Model145': xgb_oof_train145.ravel(),
    
    'Model151': xgb_oof_train151.ravel(),
    'Model152': xgb_oof_train152.ravel(),
    'Model153': xgb_oof_train153.ravel(),
    'Model154': xgb_oof_train154.ravel(),
    'Model155': xgb_oof_train155.ravel(),
    
    'Model161': xgb_oof_train161.ravel(),
    'Model162': xgb_oof_train162.ravel(),
    'Model163': xgb_oof_train163.ravel(),
    'Model164': xgb_oof_train164.ravel(),
    'Model165': xgb_oof_train165.ravel(),
    
    'Model171': xgb_oof_train171.ravel(),
    'Model172': xgb_oof_train172.ravel(),
    'Model173': xgb_oof_train173.ravel(),
    'Model174': xgb_oof_train174.ravel(),
    'Model175': xgb_oof_train175.ravel(),
    
    'Model181': xgb_oof_train181.ravel(),
    'Model182': xgb_oof_train182.ravel(),
    'Model183': xgb_oof_train183.ravel(),
    'Model184': xgb_oof_train184.ravel(),
    'Model185': xgb_oof_train185.ravel(),
    
    'Model191': xgb_oof_train191.ravel(),
    'Model192': xgb_oof_train192.ravel(),
    'Model193': xgb_oof_train193.ravel(),
    'Model194': xgb_oof_train194.ravel(),
    'Model195': xgb_oof_train195.ravel(),
    })
base_predictions_train.head()
# membuat kolom hasil prediksi dari data_train (seluruh model)
x_train = np.concatenate((  
    xgb_oof_train11,
    xgb_oof_train12,
    xgb_oof_train13,
    xgb_oof_train14,
    xgb_oof_train15,

    xgb_oof_train21,
    xgb_oof_train22,
    xgb_oof_train23,
    xgb_oof_train24,
    xgb_oof_train25,
    
    xgb_oof_train31,
    xgb_oof_train32,
    xgb_oof_train34,
    xgb_oof_train35,
    
    xgb_oof_train41,
    xgb_oof_train42,
    xgb_oof_train43,
    xgb_oof_train44,
    xgb_oof_train45,
    
    
    xgb_oof_train51,
    xgb_oof_train52,
    xgb_oof_train53,
    xgb_oof_train54,
    xgb_oof_train55,
    
    xgb_oof_train61,
    xgb_oof_train62,
    xgb_oof_train63,
    xgb_oof_train64,
    xgb_oof_train65,
    
    
    xgb_oof_train71,
    xgb_oof_train72,
    xgb_oof_train73,
    xgb_oof_train74,
    xgb_oof_train75,
   
    xgb_oof_train81,
    xgb_oof_train82,
    xgb_oof_train83,
    xgb_oof_train84,
    xgb_oof_train85,
    
    xgb_oof_train91,
    xgb_oof_train92,
    xgb_oof_train93,
    xgb_oof_train94,
    xgb_oof_train95,
    
    xgb_oof_train101,
    xgb_oof_train102,
    xgb_oof_train103,
    xgb_oof_train104,
    xgb_oof_train105,
    
    
    xgb_oof_train111,
    xgb_oof_train112,
    xgb_oof_train113,
    xgb_oof_train114,
    xgb_oof_train115,
    
    xgb_oof_train121,
    xgb_oof_train122,
    xgb_oof_train123,
    xgb_oof_train124,
    xgb_oof_train125,
    
    
    xgb_oof_train131,
    xgb_oof_train132,
    xgb_oof_train133,
    xgb_oof_train134,
    xgb_oof_train135,
    
    
    xgb_oof_train141,
    xgb_oof_train142,
    xgb_oof_train143,
    xgb_oof_train144,
    xgb_oof_train145,
    
    xgb_oof_train151,
    xgb_oof_train152,
    xgb_oof_train153,
    xgb_oof_train154,
    xgb_oof_train155,
    
    xgb_oof_train161,
    xgb_oof_train162,
    xgb_oof_train163,
    xgb_oof_train164,
    xgb_oof_train165,
    
    xgb_oof_train171,
    xgb_oof_train172,
    xgb_oof_train173,
    xgb_oof_train174,
    xgb_oof_train175,
    
    xgb_oof_train181,
    xgb_oof_train182,
    xgb_oof_train183,
    xgb_oof_train184,
    xgb_oof_train185,
    
    xgb_oof_train191,
    xgb_oof_train192,
    xgb_oof_train193,
    xgb_oof_train194,
    xgb_oof_train195), axis=1)
    
   

# membuat kolom hasil prediksi dari data_Test (seluruh model)
x_test = np.concatenate((xgb_oof_test11,
    xgb_oof_test12,
    xgb_oof_test13,
    xgb_oof_test14,
    xgb_oof_test15,

    xgb_oof_test21,
    xgb_oof_test22,
    xgb_oof_test23,
    xgb_oof_test24,
    xgb_oof_test25,
    
    xgb_oof_test31,
    xgb_oof_test32,

    xgb_oof_test34,
    xgb_oof_test35,
    
    xgb_oof_test41,
    xgb_oof_test42,
    xgb_oof_test43,
    xgb_oof_test44,
    xgb_oof_test45,
    
    
    xgb_oof_test51,
    xgb_oof_test52,
    xgb_oof_test53,
    xgb_oof_test54,
    xgb_oof_test55,
    
    xgb_oof_test61,
    xgb_oof_test62,
    xgb_oof_test63,
    xgb_oof_test64,
    xgb_oof_test65,
    
    
    xgb_oof_test71,
    xgb_oof_test72,
    xgb_oof_test73,
    xgb_oof_test74,
    xgb_oof_test75,
   
    xgb_oof_test81,
    xgb_oof_test82,
    xgb_oof_test83,
    xgb_oof_test84,
    xgb_oof_test85,
    
    xgb_oof_test91,
    xgb_oof_test92,
    xgb_oof_test93,
    xgb_oof_test94,
    xgb_oof_test95,
    
    xgb_oof_test101,
    xgb_oof_test102,
    xgb_oof_test103,
    xgb_oof_test104,
    xgb_oof_test105,
    
    
    xgb_oof_test111,
    xgb_oof_test112,
    xgb_oof_test113,
    xgb_oof_test114,
    xgb_oof_test115,
    
    xgb_oof_test121,
    xgb_oof_test122,
    xgb_oof_test123,
    xgb_oof_test124,
    xgb_oof_test125,
    
    
    xgb_oof_test131,
    xgb_oof_test132,
    xgb_oof_test133,
    xgb_oof_test134,
    xgb_oof_test135,
    
    
    xgb_oof_test141,
    xgb_oof_test142,
    xgb_oof_test143,
    xgb_oof_test144,
    xgb_oof_test145,
    
    xgb_oof_test151,
    xgb_oof_test152,
    xgb_oof_test153,
    xgb_oof_test154,
    xgb_oof_test155,
    
    xgb_oof_test161,
    xgb_oof_test162,
    xgb_oof_test163,
    xgb_oof_test164,
    xgb_oof_test165,
    
    xgb_oof_test171,
    xgb_oof_test172,
    xgb_oof_test173,
    xgb_oof_test174,
    xgb_oof_test175,
    
    xgb_oof_test181,
    xgb_oof_test182,
    xgb_oof_test183,
    xgb_oof_test184,
    xgb_oof_test185,
    
    xgb_oof_test191,
    xgb_oof_test192,
    xgb_oof_test193,
    xgb_oof_test194,
    xgb_oof_test195,
    ), axis=1)
# meta_model, model ini digunakan untuk training data hasil prediksi kita --> x_train
gbm = XGBClassifier( n_estimators= 250,max_depth= 4,min_child_weight= 2,gamma=0.9, subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',
                    nthread= -1,scale_pos_weight =  (y.shape[0]-y.sum()) / y.sum()) # default parameter :learning_rate = 0.02,#gamma=1,
gbm.fit(x_train, y)

# lalu hasil model tersebut dilakukan prediksi ke x_test
predictions2 = gbm.predict(x_test)
submit_stacking1= pd.DataFrame(test.iloc[:,0])
submit_stacking1['Result'] = predictions2
submit_stacking1.head()
submit_stacking1.Result.value_counts()
submit_stacking2= pd.DataFrame(test.iloc[:,0])
submit_stacking2['Result'] = predictions2
submit_stacking2.head()
submit_stacking2.Result.value_counts()
submit_stacking3= pd.DataFrame(test.iloc[:,0])
submit_stacking3['Result'] = predictions2
submit_stacking3.head()
submit_stacking3.Result.value_counts()
fi = pd.DataFrame()
fi['R1'] = submit_stacking1.Result
fi['R2'] = submit_stacking2.Result
fi['R3'] = submit_stacking3.Result
final = list()
from statistics import mode
for i in range(len(fi)):
    final_bin.append(mode(fi.iloc[i,]))
submit = pd.DataFrame(test.iloc[:,0])
submit['Result'] = hasil['final']
submit.head()
submit.Result.value_counts()
from IPython.display import HTML
import pandas as pd
import numpy as np
submit.to_csv('final_ensemble_bin.csv', index=False)

def create_download_link(title = "Download CSV file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe which was saved with .to_csv method
create_download_link(filename='final_ensemble_bin.csv')