from __future__ import division

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn import metrics

from sklearn.metrics import accuracy_score, recall_score

from math import log

from sklearn.feature_selection import mutual_info_classif

from sklearn.feature_extraction.text import CountVectorizer

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

import statistics

from sklearn.model_selection import GridSearchCV 

from sklearn.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import classification_report

from scipy.cluster.hierarchy import dendrogram

from sklearn.metrics import confusion_matrix



from sklearn.feature_selection import SelectFromModel

from imblearn.metrics import specificity_score

from imblearn.metrics import sensitivity_score

from imblearn.metrics import geometric_mean_score





from sklearn.feature_selection import RFE



clf_rf_2 = RandomForestClassifier(random_state=43)   

rfe = RFE(estimator=clf_rf_2, n_features_to_select=16, step=1)





from sklearn.svm import SVC

svm=SVC(kernel = 'linear',C=10, gamma=1000, probability = True)

svmm=SVC(random_state=1, probability=True)

#svm=SVC(probability=True)



#clf = AdaBoostClassifier(SVC(probability=True,kernel='linear'),n_estimators=50, learning_rate=1.0, algorithm='SAMME')



from sklearn.model_selection import train_test_split







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
import pandas as pd

wbc = pd.read_csv("../input/wbc.csv")

wbc=wbc.astype(float)
wbc.tail()


# bcs = pd.DataFrame(preprocessing.scale(wbc.ix[:,0:9]))

# bcs.columns = list(wbc.ix[:,0:9].columns)

# bcs['diagnosis'] = wbc['diagnosis']
wbc.drop(["Sample code number"],axis=1,inplace=True)
# bcs.drop(["Sample code number"],axis=1,inplace=True)
wbc.diagnosis = [1 if each == 4 else -1 for each in  wbc.diagnosis]
M = wbc[wbc.diagnosis==1]

B = wbc[wbc.diagnosis==-1]
B
fig = plt.figure()

ax = fig.add_subplot(111)

scatter = ax.scatter(wbc['Clump Thickness'],wbc['Bland Chromatin'],

                     c=wbc['diagnosis'])

ax.set_title('Class Distribution')

ax.set_xlabel('Clump Thickness')

ax.set_ylabel('Bland Chromatin')

plt.colorbar(scatter)
# bcs.diagnosis = [1 if each == 4 else -1 for each in  bcs.diagnosis]
y=wbc.diagnosis.values

x_data=wbc.drop(["diagnosis"],axis=1)
wbc.diagnosis
sum=0

c=0



for i in range(wbc['Bare Nuclei'].size):

    if(wbc['Bare Nuclei'].iloc[i]!=0 ):

        c=c+1

        sum+=float(wbc['Bare Nuclei'].iloc[i])

        

        

bn_mean=sum/c

print(c,bn_mean)               
for i in range(wbc['Bare Nuclei'].size):

    if(wbc['Bare Nuclei'].iloc[i]==0 ):

        wbc['Bare Nuclei'].iloc[i]=bn_mean


# y=wbc.diagnosis.values

# x_data=wbc.drop(["diagnosis"],axis=1)
# clustering = AgglomerativeClustering(n_clusters=5).fit(x)

# x_data['clabel']=clustering.labels_

# x_data
clustering = AgglomerativeClustering(n_clusters=5).fit(x_data)

x_data['clabel']=clustering.labels_

xx_data = x_data.copy()

x=(xx_data - np.min(xx_data))/(np.max(xx_data)-np.min(xx_data))



clustering = AgglomerativeClustering(n_clusters=6).fit(x)

x_data['clabel']=clustering.labels_

x_data
# def plot_dendrogram(model, **kwargs):

#     # Create linkage matrix and then plot the dendrogram



#     # create the counts of samples under each node

#     counts = np.zeros(model.children_.shape[0])

#     n_samples = len(model.labels_)

#     for i, merge in enumerate(model.children_):

#         current_count = 0

#         for child_idx in merge:

#             if child_idx < n_samples:

#                 current_count += 1  # leaf node

#             else:

#                 current_count += counts[child_idx - n_samples]

#         counts[i] = current_count



#     linkage_matrix = np.column_stack([model.children_, model.distances_,

#                                       counts]).astype(float)

# clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(x_data)

# x_data['clabel']=clustering.labels_
# plt.title('Hierarchical Clustering Dendrogram')

# # plot the top three levels of the dendrogram

# plot_dendrogram(clustering, truncate_mode='level', p=3)

# plt.xlabel("Number of points in node (or index of point if no parenthesis).")

# plt.show()
# clustering = KMeans(n_clusters=5).fit(x_data)

# x_data['clabel']=clustering.labels_

# xx_data = x_data.copy()

# x=(xx_data - np.min(xx_data))/(np.max(xx_data)-np.min(xx_data))



# clustering = KMeans(n_clusters=6).fit(x)

# x_data['clabel']=clustering.labels_
# kmeans = KMeans(n_clusters=2, random_state=42).fit(x)

# x_data['clabel']=kmeans.labels_

# x_data
#x=x_data.copy()

x= (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
a=[2,3,4,5]

a
x
drop_list = ['Uniformity of Cell Size','Marginal Adhesion','Single Epithelial Cell Size','Bland Chromatin','Normal Nucleoli']

x = x.drop(drop_list,axis = 1 )

#drop(drop_list,axis = 1 )

#copy()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
param_grid = {'C': [.001,0.1, 1, 10, 100, 1000],  

              'gamma': [1000,100,10,1, 0.1, 0.01, 0.001, 0.0001], 

              'kernel': ['linear']} 



grid = GridSearchCV(svm, param_grid, refit = True, verbose = False)

grid.fit(x, y) 

# print best parameter after tuning 

print(grid.best_params_) 

  

# print how our model looks after hyper-parameter tuning 

print(grid.best_estimator_) 


    

skf =StratifiedKFold(n_splits=10,random_state=42)

skf.get_n_splits(x)



print(skf)
scores=[None]*10

i=0

for train_index, test_index in skf.split(x_data,y):

    #print("TRAIN:", train_index, "TEST:", test_index)

    x_train, x_test = x.iloc[train_index], x.iloc[test_index]

    

#     clustering = AgglomerativeClustering(n_clusters=5).fit(x_train)

#     x_train['clabel']=clustering.labels_

#     xx_data = x_train.copy()

#     x=(xx_data - np.min(xx_data))/(np.max(xx_data)-np.min(xx_data))



#     clustering = AgglomerativeClustering(n_clusters=6).fit(x)

#     x_train['clabel']=clustering.labels_

#     x_train= (x_train - np.min(x_train))/(np.max(x_train)-np.min(x_train))





    y_train, y_test = y[train_index], y[test_index]

    svm.fit(x_train,y_train)

    scores[i]=svm.score(x_test,y_test)

    i=i+1

    

    

i=0



scores
print(statistics.mean(scores))

print(statistics.stdev(scores))

kf = KFold(n_splits=10,random_state=42)

kf.get_n_splits(x)



print(kf)
scores=[None]*10

i=0

for train_index, test_index in kf.split(x):

    #print("TRAIN:", train_index, "TEST:", test_index)

    x_train, x_test = x.iloc[train_index], x.iloc[test_index]

#     kmeans = KMeans(n_clusters=5, random_state=42).fit(x_train)

#     x_train['clabel']=kmeans.labels_

#     kmeanss = KMeans(n_clusters=5, random_state=42).fit(x_test)

#     x_test['clabel']=kmeanss.labels_

     

    y_train, y_test = y[train_index], y[test_index]

    svm.fit(x_train,y_train)

    scores[i]=svm.score(x_test,y_test)

    i=i+1

    

    

i=0



scores
print(statistics.mean(scores))

print(statistics.stdev(scores))

 #x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
# scores=cross_val_score(svm, x, y, cv=10)

# scores
#  print((scores.mean(), scores.std() * 2))
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
# svm.fit(x,y)

# #print("print accuracy of svm alg: ", svm.score(x_test,y_test))

x
# model = SelectFromModel(svm, prefit=True)

# xx_new = model.transform(x)

# xx_new.shape

# model.get_support()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
svmm.fit(x_train,y_train)

print("print accuracy of svm alg: ", svmm.score(x_test,y_test))

scores=[None]*10

sp_scores=[None]*10

sn_scores=[None]*10

gm_scores=[None]*10

tn_scores=[None]*10



ii=0

for train_index, test_index in kf.split(x,y):

    #print("TRAIN:", train_index, "TEST:", test_index)

    x_train, x_test = x.iloc[train_index], x.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    row_count=len(x_train.axes[0])

    weight_arr1 = [None]*row_count



    i=0

    for i in range(row_count):

        weight_arr1[i]=1/row_count





    x_new=x_train.copy()

    x_new['diagnosis']=y_train

    x_new.insert(3,'weight',weight_arr1)



    #print(x_new)



    

    l=list(y_train)

    n=l.count(-1)

    p=l.count(1)

    k=min(n,p)



    ap=[None]*p

    an=[None]*n



    j=0

    s=[0]*(y_train.size)



    alphaa=[0]*5

    kk=0



    for i in range(5):



        x_new.set_index('diagnosis' , inplace=True)

        df_plus = x_new.loc[1]

        df_minus = x_new.loc[-1]





        #print(x_new)



        a=0

        for a in range(p):

            ap[a]=1



        df_plus['diagnosis']=ap



        b=0

        for b in range(n):

            an[b]=-1



        df_minus['diagnosis']=an



        g_plus = df_plus.nlargest(k, ['weight'])

        #print(g_plus)



        g_minus = df_minus.nlargest(k, ['weight'])

        g_total = pd.concat([g_plus,g_minus],ignore_index=False)

        g_total

        

#        print(g_total['weight'])

    

        yy=g_total.diagnosis.values

        xx=g_total.drop(["diagnosis","weight"],axis=1)





        svm.fit(xx,yy)

        yy_pred = svm.predict(x_train)





        sum_weight=0

        pp=0

        for pp in range(yy_pred.size):

            if(y_train[pp]!=yy_pred[pp]):

                sum_weight+=x_new['weight'].iloc[pp]





#         print(sum_weight)



        if(sum_weight!=0):

            alpha=(1/2)*np.log((1-sum_weight)/sum_weight)

        else:

            alpha=0





        alphaa[kk]=alpha

        kk=kk+1



    

        r=0

        for r in range(y_train.size):

            s[r]+=alpha*yy_pred[r]



        t=0

        for t in range(y_train.size):

             x_new['weight'].iloc[t]=np.exp((-1)*s[t]*(y_train[t]))



        x_new['weight']=x_new['weight'].div(x_new['weight'].sum())





        x_new['diagnosis']=y_train

#         print(x_new['weight'])

#         j=j+1

#         print(j)



    y_pred=svm.predict(x_test)

    yn_pred=svm.predict(x_train)

        

    scores[ii]=metrics.accuracy_score(y_test,y_pred)

    tn_scores[ii]=metrics.accuracy_score(y_train,yn_pred)

    

    gm_scores[ii]=geometric_mean_score(y_test, y_pred, average='macro')

    sp_scores[ii]=specificity_score(y_test, y_pred, average='macro')

    sn_scores[ii]=sensitivity_score(y_test, y_pred, average='macro')

    

    ii=ii+1





ii=0



scores

    

    
print('Test accuracy',statistics.mean(scores))

print('Standard Deviation',statistics.stdev(scores))
print('Train accuracy %',statistics.mean(tn_scores))

print('Standard Deviation',statistics.stdev(tn_scores))
# print(statistics.mean(gm_scores))

# print(statistics.stdev(gm_scores))
# print(statistics.mean(sp_scores))

# print(statistics.stdev(sp_scores))
# print(statistics.mean(sn_scores))

# print(statistics.stdev(sn_scores))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
row_count=len(x_train.axes[0])



weight_arr1 = [None]*row_count

 

i=0

for i in range(row_count):

    weight_arr1[i]=1/row_count

    



x_new=x_train.copy()

x_new['diagnosis']=y_train

x_new.insert(3,'weight',weight_arr1)



#print(x_new)



    

l=list(y_train)

n=l.count(-1)

p=l.count(1)

k=min(n,p)

    

ap=[None]*p

an=[None]*n

    

j=0

s=[0]*(y_train.size)



alphaa=[0]*5

kk=0



for i in range(5):

    

    x_new.set_index('diagnosis' , inplace=True)

    df_plus = x_new.loc[1]

    df_minus = x_new.loc[-1]

    

    

    #print(x_new)

    

    a=0

    for a in range(p):

        ap[a]=1



    df_plus['diagnosis']=ap

    

    b=0

    for b in range(n):

        an[b]=-1

    

    df_minus['diagnosis']=an

    

    g_plus = df_plus.nlargest(k, ['weight'])

    #print(g_plus)

    

    g_minus = df_minus.nlargest(k, ['weight'])

    g_total = pd.concat([g_plus,g_minus],ignore_index=False)

    g_total

    

#     print(g_total)

    yy=g_total.diagnosis.values

    xx=g_total.drop(["diagnosis","weight"],axis=1)

    

    

    svm.fit(xx,yy)

    yy_pred = svm.predict(x_train)

    

#     rfe = rfe.fit(xx, yy)

#     yy_pred = rfe.predict(x_train)

    

    

    sum_weight=0

    pp=0

    for pp in range(yy_pred.size):

        if(y_train[pp]!=yy_pred[pp]):

            sum_weight+=x_new['weight'].iloc[pp]

        

    

    print(sum_weight)

    

    if(sum_weight!=0):

        alpha=(1/2)*np.log((1-sum_weight)/sum_weight)

    else:

        alpha=0

        

        

    alphaa[kk]=alpha

    kk=kk+1

    

    

    r=0

    for r in range(y_train.size):

        s[r]+=alpha*yy_pred[r]

    

    t=0

    for t in range(y_train.size):

         x_new['weight'].iloc[t]=np.exp((-1)*s[t]*(y_train[t]))

            

    x_new['weight']=x_new['weight'].div(x_new['weight'].sum())

        

    

       

#adaboost        

        

#          if(y_train[i]!=yy_pred[i]):

                                              

#                 x_new['weight'].iloc[i]= (x_new['weight'].iloc[i])*(np.exp(alpha))

#                 #(np.exp((-1)*((alpha*yy_pred[i])*y_train[i])))

                 

#          else:

#             x_new['weight'].iloc[i]= (x_new['weight'].iloc[i])*(np.exp((-1)*alpha))

                 

       

        

        

        #/(np.min(x_new['weight']))

    

    #x_new=(x_new - np.min(x_new))/(np.max(x_new)-np.min(x_new))

    

    x_new['diagnosis']=y_train

    #print(x_new['weight'])

    j=j+1

    print(j)

    

        

    

    #yy_pred_prob = svm.predict_proba(x_train)[:,0]

    #yy_pred_prob

    

    

    
y_pred = svm.predict(x_test)

#rfe.predict(x_test)

#





y_pred

#alphaa
f=[0]*y_pred.size

yyy_pred=[0]*y_pred.size







for i in range(5):

    for j in range(y_pred.size):

        f[j]=f[j]+y_pred[j]*alphaa[i]

        



        

for j in range(y_pred.size):

    if(f[j]<0):

        yyy_pred[j]=-1

    else:

        yyy_pred[j]=1
# scores=cross_val_score(svm, x, y, cv=10)

# scores
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# scores=[None]*10

# i=0

# for train_index, test_index in skf.split(x,y):

#     #print("TRAIN:", train_index, "TEST:", test_index)

#     x_train, x_test = x.iloc[train_index], x.iloc[test_index]

# #     kmeans = AgglomerativeClustering(n_clusters=5).fit(x_train)

# #     x_train['clabel']=kmeans.labels_

# #     kmeanss = AgglomerativeClustering(n_clusters=5).fit(x_test)

# #     x_test['clabel']=kmeanss.labels_

  

#     y_train, y_test = y[train_index], y[test_index]

#     #svm.fit(x_train,y_train)

#     scores[i]=svm.score(x_test,y_test)

#     i=i+1

    

    

# i=0



# scores
# print(statistics.mean(scores))

# print(statistics.stdev(scores))
#test

from sklearn import metrics

print("print accuracy of svm alg: ", metrics.accuracy_score(y_test,yyy_pred))



print("print accuracy of svm alg: ", metrics.accuracy_score(y_test,y_pred))



print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))

print(classification_report(y_test, y_pred))





print("print accuracy of svm alg: ", svm.score(x_test,y_test))

print("print accuracy of svm alg: ", svm.score(x_train,y_train))

print("print accuracy of svm alg: ", svm.score(xx,yy))

print("\n",confusion_matrix(y_test, y_pred))