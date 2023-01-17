#imports
import csv
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.preprocessing import Imputer
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#reading datasets
train = pd.read_csv("../input/trainecsv/train.csv")
test = pd.read_csv("../input/iust-nba-rookies/test.csv")

# print(train.info())
#print(test.info())
del train['Name']
del train['PlayerID']

del test['Name']
del test['PlayerID']

#Adding new features
# train['a'] = train['GP']*train['MIN']
# train['b'] = train['GP']*train['PTS']
# train['c'] = 3*train['3P Made']
# train['d'] = train['GP']*train['FGM']
# train['e'] = train['GP']*train['FGA']
# print(train.info())


# test['a'] = test['GP']*test['MIN']
# test['b'] = test['GP']*test['PTS']
# test['c'] = 3*test['3P Made']
# test['d'] = test['GP']*test['FGM']
# test['e'] = test['GP']*test['FGA']

#print(test.info())

#extracting labels
train_labels = train['TARGET_5Yrs']
train_labels=train_labels.as_matrix()

del train['TARGET_5Yrs']

#print(len(train_labels))

#Handling missing values
imputer_train = Imputer(missing_values='NaN', strategy='mean', axis=0).fit(train)
train= imputer_train.transform(train)

imputer_test = Imputer(missing_values='NaN', strategy='mean', axis=0).fit(test)
test= imputer_test.transform(test)

#Denoising data
#_______________________unigue______________
# trainData_unique = np.unique(train,axis=0)
# print(len(train))
# print("_____________________")
# print(len(trainData_unique))


#DATA preporcessing

#standardizing
std_scale = preprocessing.StandardScaler().fit(train)
train_std = std_scale.transform(train)
test_std = std_scale.transform(test)


# #PCA
# pca_std = PCA(n_components=10).fit(train_std)
# train_stdwPCA = pca_std.transform(train_std)
# test_stdwPCA = pca_std.transform(test_std)

#normalize
train_normalized = preprocessing.normalize(train_std, norm='l2')
test_normalized = preprocessing.normalize(test_std, norm='l2')



#feature selection
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# select = SelectKBest(chi2, k=6)
# train = select.fit_transform(train, train_labels)
# test = select.transform(test)

# from sklearn.feature_selection import VarianceThreshold
# sel = VarianceThreshold(threshold=0.1)
# selFeature=sel.fit_transform(train)
# #linearSVM (1)

# linearSVM_clf = svm.SVC(kernel='linear', C=1).fit(train_normalized,train_labels)
# #acc1=cross_val_score(clf, train_normalized, train_labels, cv=20, scoring='accuracy')

# trainpred=linearSVM_clf.predict(train_normalized)
# testpred=linearSVM_clf.predict(test_normalized)

# print(metrics.accuracy_score(train_labels, trainpred))

# # print(acc1)
# # print(np.mean(acc1))


# #results.append(clf.predict(test_normalized))

# #rbfSVM (2)
# rbfSVM_clf = svm.SVC(kernel='rbf', C=1).fit(train_normalized,train_labels)

# #rbfSVM_acc=cross_val_score(rbfSVM_clf, train_normalized, train_labels, cv=20, scoring='accuracy')

# trainpred=rbfSVM_clf.predict(train_normalized)
# testpred=rbfSVM_clf.predict(test_normalized)

# print(metrics.accuracy_score(train_labels, trainpred))

# # print(rbfSVM_acc)
# # print(np.mean(rbfSVM_acc))

# #results.append(rbfSVM_clf.predict(test_normalized))

# #KNN (3)
from sklearn.neighbors import KNeighborsClassifier
# #find best k for knn

# accs=[]
# ks=[]
# for k in range (1,50):
#     Tknn=KNeighborsClassifier(n_neighbors=k)
#     acc=cross_val_score(Tknn, train_normalized, train_labels, cv=10, scoring='accuracy')
#     accs.append(acc.mean())
#     ks.append(k)

# print('Best K value in KNN with Max Accuracy is :',(accs.index(max(accs))+1))
# print('Best Accuracy : ', max(accs))

# best_k = accs.index(max(accs))+1


#use best K for knn
knn=KNeighborsClassifier(n_neighbors=2).fit(train_normalized,train_labels)
#acc2=cross_val_score(knn, train_normalized, train_labels, cv=10, scoring='accuracy')

trainpred=knn.predict(train_normalized)
#testpred=knn.predict(test_normalized)

#print(metrics.accuracy_score(train_labels, trainpred))

# print(acc2)
# print(np.mean(acc2))

#results.append(knn.predict(test_normalized))
# print(knn.get_params().keys())


#MLP (4)
from sklearn.neural_network import MLPClassifier

MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(25, 10), random_state=1).fit(train_normalized,train_labels)

trainpred=MLP.predict(train_normalized)
#testpred=MLP.predict(test_normalized)

#print(metrics.accuracy_score(train_labels, trainpred))
#acc3=cross_val_score(MLP, train_normalized, train_labels, cv=10, scoring='accuracy')

# print(acc3)
# print(np.mean(acc3))

#results.append(MLP.predict(test_normalized))
# print(MLP.get_params().keys())

# #logReg (5)
# from sklearn.linear_model import LogisticRegression

# logReg= LogisticRegression().fit(train_normalized,train_labels)

# trainpred=logReg.predict(train_normalized)
# testpred=logReg.predict(test_normalized)

# print(metrics.accuracy_score(train_labels, trainpred))

# # logreg_acc=cross_val_score(logReg,train_normalized,train_labels,cv=10,scoring='accuracy')

# # print(logreg_acc)
# # print(np.mean(logreg_acc))

# #results.append(logReg.predict(test_normalized))
# #NearestCentroid (6)

# from sklearn.neighbors.nearest_centroid import NearestCentroid

# NC_clf = NearestCentroid()
# NC_clf.fit(train_normalized, train_labels)

# trainpred=NC_clf.predict(train_normalized)
# testpred=NC_clf.predict(test_normalized)

# print(metrics.accuracy_score(train_labels, trainpred))

# # NC_acc=cross_val_score(NC_clf,train_normalized,train_labels,cv=10,scoring='accuracy')

# # print(logreg_acc)
# # print(np.mean(logreg_acc))

# #results.append(logReg.predict(test_normalized))
#GradientBoosting (7)
from sklearn.ensemble import GradientBoostingClassifier

GBC_clf = GradientBoostingClassifier(n_estimators=2000, learning_rate=0.008, max_depth=1, random_state=1).fit(train_normalized, train_labels)

trainpred=GBC_clf.predict(train_normalized)
#testpred=GBC_clf.predict(test_normalized)

#print(metrics.accuracy_score(train_labels, trainpred))

# GBC_acc=cross_val_score(GBC_clf,train_normalized,train_labels,cv=20,scoring='accuracy')

# print(GBC_acc)
# print(np.mean(GBC_acc))
#randomForest (8)
from sklearn.ensemble import RandomForestClassifier

random_forest_clf = RandomForestClassifier(n_estimators=100).fit(train_normalized,train_labels)
#acc_random_forest = cross_val_score(random_forest_clf, train, train_labels, cv=10, scoring='accuracy')

trainpred=random_forest_clf.predict(train_normalized)
#testpred=random_forest_clf.predict(test_normalized)

#print(metrics.accuracy_score(train_labels, trainpred))
# print(acc_random_forest)
# print(np.mean(acc_random_forest))

#results.append(random_forest_clf.predict(test_normalized))
#DecisionTree (9)
from sklearn.tree import DecisionTreeClassifier

DT_clf = DecisionTreeClassifier(max_depth=15, min_samples_split=3,random_state=6)
DT_clf.fit(train_normalized,train_labels)

trainpred=DT_clf.predict(train_normalized)
#testpred=DT_clf.predict(test_normalized)

#print(metrics.accuracy_score(train_labels, trainpred))

# DT_acc = cross_val_score(DT_clf, train_normalized, train_labels, cv=20, scoring='accuracy')

# print(DT_acc)
# print(np.mean(DT_acc))

# ExtraTreesClassifier (10)
from sklearn.ensemble import ExtraTreesClassifier

ET_clf = ExtraTreesClassifier(n_estimators=30, max_depth=12,min_samples_split=3, random_state=0)
ET_clf.fit(train_normalized,train_labels)

trainpred=ET_clf.predict(train_normalized)
#testpred=ET_clf.predict(test_normalized)

#print(metrics.accuracy_score(train_labels, trainpred))
# ET_acc = cross_val_score(ET_clf, train_normalized, train_labels, cv=20, scoring='accuracy')

# print(ET_acc)
# print(np.mean(ET_acc))

# #SGD (11)
# from sklearn.linear_model import SGDClassifier

# sgd_clf = SGDClassifier(loss="hinge", penalty="l2").fit(train_normalized,train_labels)

# trainpred=clf.predict(train_normalized)

# print(metrics.accuracy_score(train_labels, trainpred))
# results.append(sgd_clf.predict(test_normalized))

#AdaBoost Classifier (12)
from sklearn.ensemble import AdaBoostClassifier

AdB_clf = AdaBoostClassifier(n_estimators=450)

AdB_clf.fit(train_normalized,train_labels)

trainpred=AdB_clf.predict(train_normalized)
#testpred=AdB_clf.predict(test_normalized)

#print(metrics.accuracy_score(train_labels, trainpred))
# AdB_acc = cross_val_score(AdB_clf, train_normalized, train_labels, cv=20, scoring='accuracy')

# print(AdB_acc)
# print(np.mean(AdB_acc))
# #LDA (13)
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA_clf = LinearDiscriminantAnalysis().fit(train_normalized,train_labels)

# LDA_clf.fit(train_normalized,train_labels)

# trainpred=LDA_clf.predict(train_normalized)
# testpred=LDA_clf.predict(test_normalized)

# print(metrics.accuracy_score(train_labels, trainpred))

# # #LDA_acc = cross_val_score(LDA_clf, train_normalized, train_labels, cv=10, scoring='accuracy')

# # print(LDA_acc)
# # print(np.mean(LDA_acc))

# # resultLDA = LDA_clf.predict(test_normalized)
# #GaussianNB (14)
# from sklearn.naive_bayes import GaussianNB

# GNB_clf = GaussianNB()
# GNB_clf.fit(train_normalized,train_labels)

# trainpred=GNB_clf.predict(train_normalized)
# testpred=GNB_clf.predict(test_normalized)

# print(metrics.accuracy_score(train_labels, trainpred))
# # #GNB_acc = cross_val_score(GNB_clf, train_normalized, train_labels, cv=10, scoring='accuracy')

# # print(GNB_acc)
# # print(np.mean(GNB_acc))
#voting 
from sklearn.ensemble import VotingClassifier
# from sklearn.model_selection import GridSearchCV

ens_clf=VotingClassifier(estimators=[('kn', knn), ('ml', MLP),('gbc', GBC_clf), ('rf', random_forest_clf), 
                                     ('dt', DT_clf), ('et', ET_clf), ('adb', AdB_clf)],
                                      voting='soft', weights=[1, 2, 3, 5, 5, 5, 4])


# grid = GridSearchCV(estimator=ens_clf,  cv=5)

ens_clf.fit(train_normalized,train_labels)

#ens_acc = cross_val_score(ens_clf, train_normalized, train_labels, cv=10, scoring='accuracy')


trainpredEns=ens_clf.predict(train_normalized)
#print(metrics.accuracy_score(train_labels, trainpredEns))

# print (ens_acc)
# print(np.mean(ens_acc))

print("ENSDone")
#predicting results

result=ens_clf.predict(test_normalized)

cols = { 'PlayerID': [i+901 for i in range(440)] , 'TARGET_5Yrs': result }
submission = pd.DataFrame(cols)


submission.to_csv("submission.csv", index=False)

print(submission.info())
print (submission)
print("done")
