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



# Any results you write to the current directory are saved as output.
df
                            #     USING ONE RANDOM FOREST CLASSIFIER



from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

from matplotlib import pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import math as masampleset = pd.read_csv("/kaggle/input/hackstat2k19/sample_submisison.csv", header = 0) 



Trainset1 = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv") 

Trainset = Trainset.dropna()



xset = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")

xset = xset.dropna()



dummies1 = pd.get_dummies(Trainset1.Month)

merged1 = pd.concat([Trainset1,dummies1],axis = 'columns')



dummies2 = pd.get_dummies(merged1.VisitorType)

merged2 = pd.concat([merged1,dummies2],axis = 'columns')



dummies3 = pd.get_dummies(merged2.OperatingSystems)

#dummies3 = dummies3.drop([1], axis = 'columns')

merged3 = pd.concat([merged2,dummies3],axis = 'columns')



dummies4 = pd.get_dummies(merged3.Browser)

dummies4 = dummies4.drop([9], axis = 'columns')

merged4 = pd.concat([merged3,dummies4],axis = 'columns')



dummies5 = pd.get_dummies(merged4.Province)

#dummies5 = dummies5.drop([9], axis = 'columns')

merged5 = pd.concat([merged4,dummies5],axis = 'columns')



dummies6 = pd.get_dummies(merged5.Weekend)

merged6 = pd.concat([merged5,dummies6],axis = 'columns')



y = merged6['Revenue']



Trainset = merged6.drop(['Month','VisitorType','Weekend','OperatingSystems','Browser','Province','Revenue'], axis = 'columns')

Trainset = pd.concat([Trainset,y],axis = 'columns')

Trainset = Trainset.dropna()



y = Trainset['Revenue']



features = Trainset.columns[:55]



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



xset = xmerged6.drop(['Month','VisitorType','Weekend','OperatingSystems','Browser','Province'], axis = 'columns')

print(xset)



id_ = xset['ID']

xset = xset.drop(['ID'], axis = 'columns')



clf = RandomForestClassifier(n_jobs = 2, random_state = 0)

clf.fit(Trainset[features],y)



preds = clf.predict(Test[features])



print("Random forest = ", preds)

YArray = id_.as_matrix(columns=None)



df = pd.DataFrame({"ID" : YArray, "Revenue" : preds})

print(df)

df.to_csv("submission_random.csv", index=False)



corr = Trainset.corr()



plt.figure(figsize=(40,40)) 

# plot the heatmap

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns, vmin = -1, vmax =1, center = 0, cmap = sns.diverging_palette(0,220,n=200),square = True)
                            #     CHECK ACCURACY USING ONE RANDOM FOREST CLASSIFIER    

from sklearn.ensemble import RandomForestClassifier

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math as ma



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



Trainset = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 



dummies1 = pd.get_dummies(Trainset1.Month)

merged1 = pd.concat([Trainset,dummies1],axis = 'columns')



dummies2 = pd.get_dummies(merged1.VisitorType)

merged2 = pd.concat([merged1,dummies2],axis = 'columns')



dummies3 = pd.get_dummies(merged2.OperatingSystems)

merged3 = pd.concat([merged2,dummies3],axis = 'columns')



dummies4 = pd.get_dummies(merged3.Browser)

merged4 = pd.concat([merged3,dummies4],axis = 'columns')



dummies5 = pd.get_dummies(merged4.Province)

merged5 = pd.concat([merged4,dummies5],axis = 'columns')



Trainset = merged5.drop(['Month','VisitorType','OperatingSystems','Browser','Province'], axis = 'columns')

Trainset = Trainset.dropna()



print(Trainset)



xset = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")

xset = xset.dropna()



Trainset['is_train'] = np.random.uniform(0,1,len(Trainset)) <= 0.75

print(Trainset['is_train'])



Train, Test = Trainset[Trainset['is_train'] == True], Trainset[Trainset['is_train'] == False]



y = Train['Revenue']

#Train = Train.drop(['Revenue'], axis = 'columns')

Actual_revenue = Test['Revenue']

Test = Test.drop(['Revenue'], axis = 'columns')



print('Number of Training examples', len(Train))

print('Number of Testing examples', len(Test))



features = Test.columns[:55]

print(features)



clf = RandomForestClassifier(n_estimators = 54, n_jobs = 2, random_state = 0)

clf.fit(Train[features],y)



preds = clf.predict(Test[features])

print(pd.crosstab(Actual_revenue, preds, rownames = ['Actual Revenue'], colnames = ['Predicted Revenue']))



confution_matrix = pd.crosstab(Actual_revenue, preds, rownames = ['Actual Revenue'], colnames = ['Predicted Revenue'])

print("Accuracy:",(confution_matrix[0][0] + confution_matrix[1][1])/len(Test[features]))
                            #     FIVE CLASSIFIER WITHOUT ONE HOT ENCODING

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn import tree

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



Trainset = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 

Trainset = Trainset.dropna()



Trainset  = Trainset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], value=[0,2,166,0,307,21,54,69,72,97,641,183])

Trainset  = Trainset.replace(to_replace=['Returning_Visitor', 'New_Visitor' , 'Other'], value=[1242, 366, 11])



y = Trainset['Revenue']



xset = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")

xset = xset.dropna()



xset  = xset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], value=[0,2,166,0,307,21,54,69,72,97,641,183])

xset  = xset.replace(to_replace=['Returning_Visitor', 'New_Visitor' , 'Other'], value=[1242, 366, 11])



id_ = xset['ID']

xset = xset.drop(['ID'], axis = 'columns')



print(Trainset)



Train, Test = Trainset,xset



print('Number of Training examples', len(Train))

print('Number of Testing examples', len(Test))



features = Test.columns[:54]

print(features)



YArray= id_.as_matrix(columns=None)

#########################################################################################################



#Random Forest

clf = RandomForestClassifier(n_jobs = 2, random_state = 0)

clf.fit(Train[features],y)

y_pred_random = clf.predict(Test[features])

print("Random forest = ", y_pred_random)



#########################################################################################################



#MLP

mlpclf = MLPClassifier()

mlpclf.fit(Train[features], y) 

y_pred_mlp = mlpclf.predict(Test[features])  

print("mlp = ", y_pred_mlp)

#########################################################################################################



#Decision Tree

treeclf = tree.DecisionTreeClassifier()

treeclf = treeclf.fit(Train[features], y)

y_pred_tree = treeclf.predict(Test[features])

print("Decision Tree classifier = ",y_pred_tree)



#########################################################################################################



#LDA

lda_classifier = LinearDiscriminantAnalysis()

lda_classifier.fit(Train[features], y) 

y_pred_lda = lda_classifier.predict(Test[features])

print("LDA classifier = ",y_pred_lda)

########################################################################################################



#KNN 

knnclassifier = KNeighborsClassifier()

knnclassifier.fit(Train[features], y) 

y_pred_knn = knnclassifier.predict(Test[features])

print("KNN classifier = ",y_pred_knn)



#########################################################################################################

df = pd.DataFrame({"ID" : YArray, "Random" : y_pred_random, "mlp" : y_pred_mlp,"tree" : y_pred_tree, "lda" : y_pred_lda, "knn" : y_pred_knn})

df.to_csv("submission_five_classifiers.csv", index=False)



count = 0

pred_list = []

for i in range(len(y_pred_knn)):

    count = 0

    c_list = [y_pred_random[i],y_pred_mlp[i],y_pred_tree[i],y_pred_lda[i],y_pred_knn[i]]

    for j in range(5):

        if c_list[j] == 1:

            count += 1

    if count >= 3:

        pred_list += [1]

    else:

        pred_list += [0]



df = pd.DataFrame({"ID" : YArray, "Revenue" : pred_list})

df.to_csv("submission_with_five.csv", index=False)
df
                            #     LOGISTIC REGRESSION

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math as ma



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



Trainset = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 

Trainset = Trainset.dropna()



Trainset  = Trainset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], value=[0,2,166,0,307,21,54,69,72,97,641,183])

Trainset  = Trainset.replace(to_replace=['Returning_Visitor', 'New_Visitor' , 'Other'], value=[1242, 366, 11])



xset = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")

xset = xset.dropna()



xset  = xset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], value=[0,2,166,0,307,21,54,69,72,97,641,183])

xset  = xset.replace(to_replace=['Returning_Visitor', 'New_Visitor' , 'Other'], value=[1242, 366, 11])



print(Trainset.shape)

print(list(Trainset.columns))

print(xset.shape)



xArray = Trainset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]].values

yArray = Trainset.iloc[:, 17].values



print(xArray)

print(type(xArray[0][9]))

print(len(yArray))

m = 10466

M =[]

count = 0

xsetArray = xset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]].values



print(xsetArray)



alpha = 0.05 



class LogisticRegression:    

    def gradient_decent(self, x, y):

        w = np.ones((np.shape(x)[1],1))

        print (type(x))

        wARR = np.mat(w)

        print (type(wARR))

        print (type(y))

        for i in range(m):

            w = w - alpha * x.transpose() * (self.logistic(x * w) - y)

        return w



    def classify(self, x, w):

        prob = self.logistic(sum(x * w))

        classification = 0

        if prob >  0.5: 

            classification = 1

            #myData = 1

        #else:

          #  myData = 0

        #myData

        return classification

    def logistic(self,ws):

        return 1.0/(1 + np.exp(-ws))



logisticRegression = LogisticRegression()

wArray = logisticRegression.gradient_decent(np.mat(xArray), np.mat(yArray).transpose())

print ("close to window proceed")

#handle_command_line()

score = xArray



for p in range(m):

    if logisticRegression.classify([float(score[p][0]), float(score[p][1]),float(score[p][2]), float(score[p][3]),float(score[p][4]), float(score[p][5]),float(score[p][6]),float(score[p][7]),float(score[p][8]),float(score[p][9]),float(score[p][10]), float(score[p][11]),float(score[p][12]),float(score[p][13]),float(score[p][14]),float(score[p][15])], wArray) == yArray[p]:

        count+= 1                          

    #print(logisticRegression.classify([float(score[p][0]), float(score[p][1]),float(score[p][2]), float(score[p][3]),float(score[p][4]), float(score[p][5]),float(score[p][6]), float(score[p][7])], wArray))

    M += [(p+1,logisticRegression.classify([float(score[p][0]), float(score[p][1]),float(score[p][2]), float(score[p][3]),float(score[p][4]), float(score[p][5]),float(score[p][6]),float(score[p][7]),float(score[p][8]),float(score[p][9]),float(score[p][10]), float(score[p][11]),float(score[p][12]),float(score[p][13]),float(score[p][14]),float(score[p][15])], wArray))]



print("count=",count)

print(M)    

dfObj = pd.DataFrame(M, columns = ["ID","Revenue"]) 

dfObj.to_csv('submission_logistic.csv', index = False)
dfObj