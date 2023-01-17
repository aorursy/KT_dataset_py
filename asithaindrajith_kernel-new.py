from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors.nearest_centroid import NearestCentroid

from sklearn.decomposition import LatenDiricheltAllocation



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import math as ma



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



Trainset = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 

Trainset = Trainset.dropna()



sampleset = pd.read_csv("/kaggle/input/hackstat2k19/sample_submisison.csv", header = 0) 

sampleset = sampleset.dropna()



Trainset1 = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 

Trainset1 = Trainset1.dropna()



dummies1 = pd.get_dummies(Trainset1.Month)

merged1 = pd.concat([Trainset1,dummies1],axis = 'columns')



dummies2 = pd.get_dummies(merged1.VisitorType)

merged2 = pd.concat([merged1,dummies2],axis = 'columns')



dummies3 = pd.get_dummies(merged2.OperatingSystems)

#dummies3 = dummies3.drop([1], axis = 'columns')

merged3 = pd.concat([merged2,dummies3],axis = 'columns')



dummies4 = pd.get_dummies(merged3.Browser)

#dummies4 = dummies4.drop([9], axis = 'columns')

merged4 = pd.concat([merged3,dummies4],axis = 'columns')



dummies5 = pd.get_dummies(merged4.Province)

#dummies5 = dummies5.drop([9], axis = 'columns')

merged5 = pd.concat([merged4,dummies5],axis = 'columns')



y = merged5['Revenue']

Trainset = merged5.drop(['Month','VisitorType','Weekend','OperatingSystems','Browser','Province','Revenue'], axis = 'columns')



xset = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")

xset = xset.dropna()



xset1 = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")

xset1 = xset1.dropna()



xdummies1 = pd.get_dummies(xset1.Month)

xmerged1 = pd.concat([xset1,xdummies1],axis = 'columns')



xdummies2 = pd.get_dummies(xmerged1.VisitorType)

xmerged2 = pd.concat([xmerged1,xdummies2],axis = 'columns')



xdummies3 = pd.get_dummies(xmerged2.OperatingSystems)

#xdummies3 = xdummies3.drop([1], axis = 'columns')

xmerged3 = pd.concat([xmerged2,xdummies3],axis = 'columns')



xdummies4 = pd.get_dummies(xmerged3.Browser)

#xdummies4 = xdummies4.drop([9], axis = 'columns')

xmerged4 = pd.concat([xmerged3,xdummies4],axis = 'columns')



xdummies5 = pd.get_dummies(xmerged4.Province)

#xdummies5 = dummies5.drop([9], axis = 'columns')

xmerged5 = pd.concat([xmerged4,xdummies5],axis = 'columns')



xset = xmerged5.drop(['ID','Month','VisitorType','Weekend','OperatingSystems','Browser','Province'], axis = 'columns')



#Trainset['is_train'] = np.random.uniform(0,1,len(Trainset)) <= 0.75

print(Trainset)



Train, Test = Trainset,xset



print('Number of Training examples', len(Train))

print('Number of Testing examples', len(Test))



features = Test.columns[:52]

print(features)

y



id_ = sampleset['ID']

#new_ = id_.DataFarame(id_)

clf = RandomForestClassifier(n_jobs = 2, random_state = 0)

clf.fit(Train[features],y)



clf.predict(Test[features])



clf.predict_proba(Test[features])



preds = clf.predict(Test[features])



print("Random forest = ", preds)

#print(tpe(new_))

#print(type(preds))

YArray= id_.as_matrix(columns=None)

#print (YArray)



df = pd.DataFrame({"ID" : YArray, "Revenue" : preds})

df.to_csv("submission.csv", index=False)



#########################################################################################################

#from sklearn import datasets

#iris = datasets.load_iris()

#nb = MultinomialNB()

#nbclassifier = nb.fit(Train[features], y)

#y_pred_nb = nbclassifier.predict(Test[features])

#>>> print("Number of mislabeled points out of a total %d points : %d"

#...       % (iris.data.shape[0],(iris.target != y_pred).sum()))

#Number of mislabeled points out of a total 150 points : 6

#print("Gaussian Naive Bayes = ", y_pred_gnb)

#########################################################################################################



#ncclf = NearestCentroid()

#ncclf.fit(Train[features], y)

#y_pred_nc = ncclf.predict(Test[features])

#print("nb = ", y_pred_nb)



from sklearn.linear_model import SGDClassifier

sgdclf = SGDClassifier()

sgdclf.fit(Train[features], y) 

y_pred_sgd = sgdclf.predict(Test[features])  

print("sgd = ", y_pred_sgd)

#########################################################################################################



from sklearn import tree

#iris = load_iris()

treeclf = tree.DecisionTreeClassifier()

treeclf = treeclf.fit(Train[features], y)

y_pred_tree = treeclf.predict(Test[features])

print("Decision Tree classifier = ",y_pred_tree)



#########################################################################################################



svmclf = svm.SVC(gamma='scale')

svmclf.fit(Train[features], y)  

y_pred_svm = svmclf.predict(Test[features])

print("svm classifier = ",y_pred_svm)



########################################################################################################



knnclassifier = KNeighborsClassifier()

knnclassifier.fit(Train[features], y) 

y_pred_knn = knnclassifier.predict(Test[features])

print("KNN classifier = ",y_pred_knn)



#########################################################################################################

df = pd.DataFrame({"ID" : YArray, "Random" : preds, "sgd" : y_pred_sgd,"tree" : y_pred_tree, "svm" : y_pred_svm, "knn" : y_pred_knn})

df.to_csv("submission1.csv", index=False)

df

#svclassifier = SVC(kernel='linear')

#svclassifier.fit(Train[features], y)

#y_pred = svclassifier.predict(Test)

#print("SVC classifier = ",y_pred)

count = 0

pred_list = []

for i in range(len(y_pred_knn)):

    count = 0

    c_list = [preds[i],y_pred_sgd[i],y_pred_tree[i],y_pred_svm[i],y_pred_knn[i]]

    for j in range(5):

        if c_list[j] == 1:

            count += 1

    if count >= 3:

        pred_list += [1]

    else:

        pred_list += [preds[i]]

print(len(pred_list))

o = pd.DataFrame(pred_list) 



pred_= o.as_matrix(columns=None)

df = pd.DataFrame({"ID" : YArray, "Revenue" : pred_list})

df.to_csv("submission_new2.csv", index=False)
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors.nearest_centroid import NearestCentroid

from sklearn.linear_model import SGDClassifier

from sklearn import tree



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math as ma



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



Trainset1 = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 

Trainset1 = Trainset1.dropna()



#Trainset  = Trainset1.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], value=[0,2,166,0,307,21,54,69,72,97,641,183])

#Trainset  = Trainset.replace(to_replace=['Returning_Visitor', 'New_Visitor' , 'Other'], value=[1242, 366, 11])

#Trainset  = Trainset.replace(to_replace=[True,False], value=[1,2])

dummies1 = pd.get_dummies(Trainset1.Month)

merged1 = pd.concat([Trainset1,dummies1],axis = 'columns')



dummies2 = pd.get_dummies(merged1.VisitorType)

merged2 = pd.concat([merged1,dummies2],axis = 'columns')



dummies3 = pd.get_dummies(merged2.OperatingSystems)

#dummies3 = dummies3.drop([1], axis = 'columns')

merged3 = pd.concat([merged2,dummies3],axis = 'columns')



dummies4 = pd.get_dummies(merged3.Browser)

#dummies4 = dummies4.drop([1], axis = 'columns')

merged4 = pd.concat([merged3,dummies4],axis = 'columns')



dummies5 = pd.get_dummies(merged4.Province)

#dummies5 = dummies5.drop([9], axis = 'columns')

merged5 = pd.concat([merged4,dummies5],axis = 'columns')



#dummies3 = pd.get_dummies(merged2.OperatingSystems)

#dummies3 = merged3.drop(['Month','VisitorType','Weekend','Feb','Other','OperatingSystems',1], axis = 'columns')

#merged3 = pd.concat([merged2,dummies3],axis = 'columns')



Trainset = merged5.drop(['Month','VisitorType','Weekend','OperatingSystems','Browser','Province'], axis = 'columns')



print(Trainset)

xset = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")

xset = xset.dropna()



Trainset['is_train'] = np.random.uniform(0,1,len(Trainset)) <= 0.75

print(Trainset['is_train'])



Train, Test = Trainset[Trainset['is_train'] == True], Trainset[Trainset['is_train'] == False]

y = Train['Revenue']



Train = Train.drop(['Revenue'], axis = 'columns')

o = Test['Revenue']

Test = Test.drop(['Revenue'], axis = 'columns')

print('Number of Training examples', len(Train))

print('Number of Testing examples', len(Test))



features = Test.columns[:54]

print(features)



y



clf = tree.DecisionTreeClassifier()

clf.fit(Train[features],y)



#clf.predict(Test[features])



#clf.predict_proba(Test[features])



preds = clf.predict(Test[features])



print(pd.crosstab(o, preds, rownames = ['Actual Revenue'], colnames = ['Predicted Revenue']))



i = pd.crosstab(o, preds, rownames = ['Actual Revenue'], colnames = ['Predicted Revenue'])

print((i[0][0] + i[1][1])/len(Test[features]))

#clf = RandomForestClassifier(n_estimators=17, max_depth=None,min_samples_split=2, random_state=0)

#scores = cross_val_score(clf, X, y, cv=5)

#scores.mean() 
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors.nearest_centroid import NearestCentroid

from sklearn import svm

from sklearn.naive_bayes import ComplementNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors.nearest_centroid import NearestCentroid

from sklearn.tree import DecisionTreeClassifier

from sklearn.cluster import KMeans

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



#import xgboost as xgb



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import math as ma



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



Trainset = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 

Trainset = Trainset.dropna()



Trainset  = Trainset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], value=[0,2,166,0,307,21,54,69,72,97,641,183])

Trainset  = Trainset.replace(to_replace=['Returning_Visitor', 'New_Visitor' , 'Other'], value=[1242, 366, 11])

#Trainset  = Trainset.replace(to_replace=[True,False], value=[1,2])



sampleset = pd.read_csv("/kaggle/input/hackstat2k19/sample_submisison.csv", header = 0) 

sampleset = sampleset.dropna()



#Trainset1 = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 

#Trainset1 = Trainset1.dropna()



#dummies1 = pd.get_dummies(Trainset1.Month)

#merged1 = pd.concat([Trainset1,dummies1],axis = 'columns')



#dummies2 = pd.get_dummies(merged1.VisitorType)

#merged2 = pd.concat([merged1,dummies2],axis = 'columns')



#dummies3 = pd.get_dummies(merged2.OperatingSystems)

#dummies3 = dummies3.drop([1], axis = 'columns')

#merged3 = pd.concat([merged2,dummies3],axis = 'columns')



#dummies4 = pd.get_dummies(merged3.Browser)

#dummies4 = dummies4.drop([9], axis = 'columns')

#merged4 = pd.concat([merged3,dummies4],axis = 'columns')



#dummies5 = pd.get_dummies(merged4.Province)

#dummies5 = dummies5.drop([9], axis = 'columns')

#merged5 = pd.concat([merged4,dummies5],axis = 'columns')

y = Trainset['Revenue']

#Trainset = merged5.drop(['Month','VisitorType','Weekend','OperatingSystems','Browser','Province','Revenue'], axis = 'columns')



xset = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")

xset = xset.dropna()



xset  = xset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], value=[0,2,166,0,307,21,54,69,72,97,641,183])

xset  = xset.replace(to_replace=['Returning_Visitor', 'New_Visitor' , 'Other'], value=[1242, 366, 11])

#xset  = xset.replace(to_replace=[True,False], value=[1,2])



#xset1 = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")

#xset1 = xset1.dropna()



#xdummies1 = pd.get_dummies(xset1.Month)

#xmerged1 = pd.concat([xset1,xdummies1],axis = 'columns')



#xdummies2 = pd.get_dummies(xmerged1.VisitorType)

#xmerged2 = pd.concat([xmerged1,xdummies2],axis = 'columns')



#xdummies3 = pd.get_dummies(xmerged2.OperatingSystems)

#xdummies3 = xdummies3.drop([1], axis = 'columns')

#xmerged3 = pd.concat([xmerged2,xdummies3],axis = 'columns')



#xdummies4 = pd.get_dummies(xmerged3.Browser)

#xdummies4 = xdummies4.drop([9], axis = 'columns')

#xmerged4 = pd.concat([xmerged3,xdummies4],axis = 'columns')



#xdummies5 = pd.get_dummies(xmerged4.Province)

#xdummies5 = dummies5.drop([9], axis = 'columns')

#xmerged5 = pd.concat([xmerged4,xdummies5],axis = 'columns')



#xset = xmerged5.drop(['ID','Month','VisitorType','Weekend','OperatingSystems','Browser','Province'], axis = 'columns')

xset = xset.drop(['ID'], axis = 'columns')

#Trainset['is_train'] = np.random.uniform(0,1,len(Trainset)) <= 0.75

print(Trainset)



Train, Test = Trainset,xset



print('Number of Training examples', len(Train))

print('Number of Testing examples', len(Test))



#features = Test.columns[15:17]

features = Test.columns[:54]

print(features)

y



id_ = sampleset['ID']

#new_ = id_.DataFarame(id_)

clf = RandomForestClassifier(n_jobs = 2, random_state = 0)

clf.fit(Train[features],y)



clf.predict(Test[features])



clf.predict_proba(Test[features])



preds = clf.predict(Test[features])



print("Random forest = ", preds)

#print(tpe(new_))

#print(type(preds))

YArray= id_.as_matrix(columns=None)

#print (YArray)



df = pd.DataFrame({"ID" : YArray, "Revenue" : preds})

#df.to_csv("submission_withfive.csv", index=False)



#########################################################################################################

#from sklearn import datasets

#iris = datasets.load_iris()

#nb = MultinomialNB()

#nbclassifier = nb.fit(Train[features], y)

#y_pred_nb = nbclassifier.predict(Test[features])

#>>> print("Number of mislabeled points out of a total %d points : %d"

#...       % (iris.data.shape[0],(iris.target != y_pred).sum()))

#Number of mislabeled points out of a total 150 points : 6

#print("Gaussian Naive Bayes = ", y_pred_gnb)

#########################################################################################################



#ncclf = NearestCentroid()

#ncclf.fit(Train[features], y)

#y_pred_nc = ncclf.predict(Test[features])

#print("nb = ", y_pred_nb)



from sklearn.linear_model import SGDClassifier

sgdclf = SGDClassifier()

sgdclf.fit(Train[features], y) 

y_pred_sgd = sgdclf.predict(Test[features])  

print("sgd = ", y_pred_sgd)

#########################################################################################################



from sklearn import tree

#iris = load_iris()

treeclf = tree.DecisionTreeClassifier()

treeclf = treeclf.fit(Train[features], y)

y_pred_tree = treeclf.predict(Test[features])

print("Decision Tree classifier = ",y_pred_tree)



#########################################################################################################



#svmclf = svm.SVC()

#svmclf.fit(Train[features], y)  

#y_pred_svm = svmclf.predict(Test[features])

#print("svm classifier = ",y_pred_svm)



cclassifier = LinearDiscriminantAnalysis()

cclassifier.fit(Train[features], y) 

y_pred_svm = cclassifier.predict(Test[features])

print("c classifier = ",y_pred_svm)

########################################################################################################



knnclassifier = KNeighborsClassifier()

knnclassifier.fit(Train[features], y) 

y_pred_knn = knnclassifier.predict(Test[features])

print("KNN classifier = ",y_pred_knn)



#########################################################################################################

df = pd.DataFrame({"ID" : YArray, "Random" : preds, "sgd" : y_pred_sgd,"tree" : y_pred_tree, "c" : y_pred_svm, "knn" : y_pred_knn})

df.to_csv("submission_neeeew.csv", index=False)

df

#svclassifier = SVC(kernel='linear')

#svclassifier.fit(Train[features], y)

#y_pred = svclassifier.predict(Test)

#print("SVC classifier = ",y_pred)

count = 0

pred_list = []

for i in range(len(y_pred_knn)):

    count = 0

    c_list = [preds[i],y_pred_sgd[i],y_pred_tree[i],y_pred_svm[i],y_pred_knn[i]]

    for j in range(5):

        if c_list[j] == 1:

            count += 1

    if count >= 3:

        pred_list += [1]

    else:

        pred_list += [0]

print(len(pred_list))

o = pd.DataFrame(pred_list) 



pred_= o.as_matrix(columns=None)

df = pd.DataFrame({"ID" : YArray, "Revenue" : pred_list})

df.to_csv("submission_v2.csv", index=False)
from sklearn.ensemble import RandomForestClassifier

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import math as masampleset = pd.read_csv("/kaggle/input/hackstat2k19/sample_submisison.csv", header = 0) 



Trainset = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv") 



xset = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")



y = Trainset['Revenue']

Trainset = Trainset.drop(['Revenue'], axis = 'columns')



id_ = xset['ID']

xset = xset.drop(['ID'], axis = 'columns')



clf = RandomForestClassifier(n_jobs = 2, random_state = 0)

clf.fit(Trainset,y)



preds = clf.predict(Test[features])



print("Random forest = ", preds)

YArray = id_.as_matrix(columns=None)



df = pd.DataFrame({"ID" : YArray, "Revenue" : preds})

print(df)

df.to_csv("submission_today.csv", index=False)
                            #     USING THREE RANDOM FOREST CLASSIFIERS



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn.ensemble import RandomForestClassifier



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



Trainset = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv") 

Trainset = Trainset.dropna()



xset = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")

xset = xset.dropna()



dummies1 = pd.get_dummies(Trainset.Month)

merged1 = pd.concat([Trainset,dummies1],axis = 'columns')



dummies2 = pd.get_dummies(merged1.VisitorType)

merged2 = pd.concat([merged1,dummies2],axis = 'columns')



dummies3 = pd.get_dummies(merged2.OperatingSystems)

merged3 = pd.concat([merged2,dummies3],axis = 'columns')



dummies4 = pd.get_dummies(merged3.Browser)

dummies4 = dummies4.drop([9], axis = 'columns')

merged4 = pd.concat([merged3,dummies4],axis = 'columns')



dummies5 = pd.get_dummies(merged4.Province)

merged5 = pd.concat([merged4,dummies5],axis = 'columns')



dummies6 = pd.get_dummies(merged5.Weekend)

merged6 = pd.concat([merged5,dummies6],axis = 'columns')



y = merged6['Revenue']



Trainset = merged6.drop(['Month','VisitorType','Weekend','OperatingSystems','Browser','Province','Revenue'], axis = 'columns')

#Trainset = pd.concat([Trainset,y],axis = 'columns')



xdummies1 = pd.get_dummies(xset.Month)

xmerged1 = pd.concat([xset,xdummies1],axis = 'columns')



xdummies2 = pd.get_dummies(xmerged1.VisitorType)

xmerged2 = pd.concat([xmerged1,xdummies2],axis = 'columns')



xdummies3 = pd.get_dummies(xmerged2.OperatingSystems)

xmerged3 = pd.concat([xmerged2,xdummies3],axis = 'columns')



xdummies4 = pd.get_dummies(xmerged3.Browser)

xmerged4 = pd.concat([xmerged3,xdummies4],axis = 'columns')



xdummies5 = pd.get_dummies(xmerged4.Province)

xmerged5 = pd.concat([xmerged4,xdummies5],axis = 'columns')



xdummies6 = pd.get_dummies(xmerged5.Weekend)

xmerged6 = pd.concat([xmerged5,xdummies6],axis = 'columns')



id_ = xset['ID']

xset = xmerged6.drop(['ID','Month','VisitorType','Weekend','OperatingSystems','Browser','Province'], axis = 'columns')



print(Trainset)

print(xset)



Test = xset

Train = Trainset



print('Number of Training examples', len(Train))

print('Number of Testing examples', len(Test))



features1 = Train.columns[:55]

print(features1)



features2 = Train.columns[:55]

print(features2)



clf1 = RandomForestClassifier()

clf1.fit(Train[features1],y)



clf2 = RandomForestClassifier()

clf2.fit(Train[features2],y)



train_preds1 = clf1.predict(Train[features1])

train_preds2 = clf2.predict(Train[features2])



train_preds1 = pd.DataFrame({"Revenue1" :train_preds1})

train_preds2 = pd.DataFrame({"Revenue2" :train_preds2})



test_preds1 = clf1.predict(Test[features1])

test_preds2 = clf2.predict(Test[features2])



test_preds1 = pd.DataFrame({"Revenue1" :test_preds1})

test_preds2 = pd.DataFrame({"Revenue2" :test_preds2})



Train_two_dfs = pd.concat([Trainset,train_preds1],axis = 'columns')

Train_two_dfs = pd.concat([Train_two_dfs,train_preds2],axis = 'columns')

Train_two_dfs = pd.concat([Train_two_dfs,y],axis = 'columns')

Train_two_dfs = Train_two_dfs.dropna()



y_new = Train_two_dfs['Revenue']



Test_two_dfs = pd.concat([Test,test_preds1],axis = 'columns')

Test_two_dfs = pd.concat([Test_two_dfs,test_preds2],axis = 'columns')

Test_two_dfs = Test_two_dfs.dropna()



print(len(Train_two_dfs))

print(len(Test_two_dfs))

print(len(y))

features3 = Train_two_dfs.columns[[55,56]]

print(features3)



clf3 = RandomForestClassifier(n_estimators = 10, n_jobs = 2, random_state = 0)

clf3.fit(Train_two_dfs[features3],y_new)



preds = clf3.predict(Test_two_dfs[features3])



YArray= id_.as_matrix(columns=None)

print (YArray)



df = pd.DataFrame({"ID" : YArray, "Revenue" : preds})

df.to_csv("submission_with_three_randomforest.csv", index=False)    