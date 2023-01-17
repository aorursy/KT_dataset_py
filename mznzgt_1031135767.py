# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
breastData = pd.read_csv('../input/wisconsin_breast_cancer.csv')

breastData.head()
median_nuclei = breastData['nuclei'].median()

breastData['nuclei'].fillna(median_nuclei, inplace=True)
from sklearn.model_selection import train_test_split

x = breastData.iloc[:,1:10]

y = breastData.iloc[:,10]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1337316)

print(len(X_train), len(X_test))



allresult=[]

allname=[]

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_score



sgd=SGDClassifier(random_state=1337316,max_iter=1000,tol=1e-3)



for i in range(1,512):

    tmp=int("{0:b}".format(i))

    result=[]

   

    for x in range(9):

        if tmp%10==1:

            if(x==0):

               

                result.append('thickness')

            elif (x==1):

            

                result.append('size')

            elif (x==2):

             

                result.append('shape')

            elif (x==3):

                

                result.append('adhesion')

            elif (x==4):

               

                result.append('single')

            elif (x==5):

                

                result.append('nuclei')

            elif (x==6):

               

                result.append('chromatin')

            elif (x==7):

              

                result.append('nucleoli')

            elif(x==8):

               

                result.append('mitosis')

        tmp=tmp//10

    

    if len(result)==1:

        continue

    elif len(result)==2:

        X_trainNew1=X_train.loc[:,[result[0],result[1]]]   

    elif len(result)==3:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2]]]

    elif len(result)==4:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3]]]

    elif len(result)==5:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4]]]

    elif len(result)==6:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5]]]

    elif len(result)==7:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6]]]

    elif len(result)==8:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7]]]

    elif len(result)==9:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8]]]

           

    display_score=cross_val_score(sgd,X_trainNew1,y_train,cv=10,scoring="accuracy")

    display_score.mean()

    

    print(*result)

    print(display_score.mean())

    allname.append(result)

    allresult.append(display_score.mean())

    

    

    



    

    
print("Best subset for train data")

print(*allname[allresult.index(max(allresult))])

print(max(allresult))

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score

allresult1=[]

allname1=[]

for i in range(1,512):

    tmp=int("{0:b}".format(i))

    result=[]

   

    for x in range(9):

        if tmp%10==1:

            if(x==0):

               

                result.append('thickness')

            elif (x==1):

            

                result.append('size')

            elif (x==2):

             

                result.append('shape')

            elif (x==3):

                

                result.append('adhesion')

            elif (x==4):

               

                result.append('single')

            elif (x==5):

                

                result.append('nuclei')

            elif (x==6):

               

                result.append('chromatin')

            elif (x==7):

              

                result.append('nucleoli')

            elif(x==8):

               

                result.append('mitosis')

        tmp=tmp//10

    

    if len(result)==1:

        continue 

        #X_trainNew1=X_train.loc[:,result[0]]

    elif len(result)==2:

        X_testNew=X_test.loc[:,[result[0],result[1]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1]]]

    elif len(result)==3:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2]]]

    elif len(result)==4:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3]]]

    elif len(result)==5:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3],result[4]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4]]]

    elif len(result)==6:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5]]]

    elif len(result)==7:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6]]]

    elif len(result)==8:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7]]]

    elif len(result)==9:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8]]]

    

    sgd.fit(X_trainNew1, y_train)

    d=accuracy_score(sgd.predict(X_testNew),y_test)

    print(d)

    print(*result)

    allname1.append(result)

    allresult1.append(d)

    

    
print("Accaurcy for the best subset in train data for sgd classifier")

print(allresult1[allname1.index(allname[allresult.index(max(allresult))])])



print("The best subset for test data for sgd classifier")

print(*allname1[allresult1.index(max(allresult1))])

print(max(allresult1))
import seaborn as sns

cv_tupels = [[f,'cv',x] for f,x in zip(range(511),allresult)]

test_tupels = [[f,'test',x] for f,x in zip(range(511),allresult1)]

results = pd.DataFrame(cv_tupels+test_tupels,columns=['fold','algo','rmse'])

sns.scatterplot(data=results,x='fold',y='rmse',hue='algo')



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

forest_data=RandomForestClassifier(random_state=1337316,n_estimators=30)

allresult2=[]

allname2=[]

for i in range(1,512):

    tmp=int("{0:b}".format(i))

    result=[]

   

    for x in range(9):

        if tmp%10==1:

            if(x==0):

               

                result.append('thickness')

            elif (x==1):

            

                result.append('size')

            elif (x==2):

             

                result.append('shape')

            elif (x==3):

                

                result.append('adhesion')

            elif (x==4):

               

                result.append('single')

            elif (x==5):

                

                result.append('nuclei')

            elif (x==6):

               

                result.append('chromatin')

            elif (x==7):

              

                result.append('nucleoli')

            elif(x==8):

               

                result.append('mitosis')

        tmp=tmp//10

    

    if len(result)==1:

        continue

    elif len(result)==2:

        X_trainNew1=X_train.loc[:,[result[0],result[1]]]   

    elif len(result)==3:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2]]]

    elif len(result)==4:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3]]]

    elif len(result)==5:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4]]]

    elif len(result)==6:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5]]]

    elif len(result)==7:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6]]]

    elif len(result)==8:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7]]]

    elif len(result)==9:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8]]]

        

    display_score=cross_val_score(forest_data,X_trainNew1,y_train,cv=10,scoring="accuracy")

    display_score.mean()

    

    print(*result)

    print(display_score.mean())

    allname2.append(result)

    allresult2.append(display_score.mean())
print("Best accuracy for random forest's train subset")

print(*allname2[allresult2.index(max(allresult2))])

print(max(allresult2))
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

allresult3=[]

allname3=[]



for i in range(1,512):

    tmp=int("{0:b}".format(i))

    result=[]

   

    for x in range(9):

        if tmp%10==1:

            if(x==0):

               

                result.append('thickness')

            elif (x==1):

            

                result.append('size')

            elif (x==2):

             

                result.append('shape')

            elif (x==3):

                

                result.append('adhesion')

            elif (x==4):

               

                result.append('single')

            elif (x==5):

                

                result.append('nuclei')

            elif (x==6):

               

                result.append('chromatin')

            elif (x==7):

              

                result.append('nucleoli')

            elif(x==8):

               

                result.append('mitosis')

        tmp=tmp//10

    

    if len(result)==1:

        continue 

    elif len(result)==2:

        X_testNew=X_test.loc[:,[result[0],result[1]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1]]]

    elif len(result)==3:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2]]]

    elif len(result)==4:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3]]]

    elif len(result)==5:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3],result[4]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4]]]

    elif len(result)==6:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5]]]

    elif len(result)==7:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6]]]

    elif len(result)==8:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7]]]

    elif len(result)==9:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8]]]

    

    forest_data.fit(X_trainNew1, y_train)

    d=accuracy_score(forest_data.predict(X_testNew),y_test)

    print(d)

    print(*result)

    allname3.append(result)

    allresult3.append(d)
print("The best subset for random forest's train data in test data")

print(allresult3[allname3.index(allname2[allresult2.index(max(allresult2))])])

print("---------")

print("The best subset for random forest's test data")

print(*allname3[allresult3.index(max(allresult3))])

print(max(allresult3))
import seaborn as sns

cv_tupels = [[f,'cv',x] for f,x in zip(range(511),allresult2)]

test_tupels = [[f,'test',x] for f,x in zip(range(511),allresult3)]

results = pd.DataFrame(cv_tupels+test_tupels,columns=['fold','algo','rmse'])

sns.scatterplot(data=results,x='fold',y='rmse',hue='algo')
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score

GNB_data=GaussianNB()

allresult4=[]

allname4=[]

for i in range(1,512):

    tmp=int("{0:b}".format(i))

    result=[]

   

    for x in range(9):

        if tmp%10==1:

            if(x==0):

               

                result.append('thickness')

            elif (x==1):

            

                result.append('size')

            elif (x==2):

             

                result.append('shape')

            elif (x==3):

                

                result.append('adhesion')

            elif (x==4):

               

                result.append('single')

            elif (x==5):

                

                result.append('nuclei')

            elif (x==6):

               

                result.append('chromatin')

            elif (x==7):

              

                result.append('nucleoli')

            elif(x==8):

               

                result.append('mitosis')

        tmp=tmp//10

    

    if len(result)==1:

        continue

    elif len(result)==2:

        X_trainNew1=X_train.loc[:,[result[0],result[1]]]   

    elif len(result)==3:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2]]]

    elif len(result)==4:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3]]]

    elif len(result)==5:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4]]]

    elif len(result)==6:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5]]]

    elif len(result)==7:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6]]]

    elif len(result)==8:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7]]]

    elif len(result)==9:

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8]]]

        

    display_score=cross_val_score(GNB_data,X_trainNew1,y_train,cv=10,scoring="accuracy")

    display_score.mean()

    

    print(*result)

    print(display_score.mean())

    allname4.append(result)

    allresult4.append(display_score.mean())
print("best subset for GaussianNB's train subset ")

print(*allname4[allresult4.index(max(allresult4))])

print(max(allresult4))
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

allresult5=[]

allname5=[]



for i in range(1,512):

    tmp=int("{0:b}".format(i))

    result=[]

   

    for x in range(9):

        if tmp%10==1:

            if(x==0):

               

                result.append('thickness')

            elif (x==1):

            

                result.append('size')

            elif (x==2):

             

                result.append('shape')

            elif (x==3):

                

                result.append('adhesion')

            elif (x==4):

               

                result.append('single')

            elif (x==5):

                

                result.append('nuclei')

            elif (x==6):

               

                result.append('chromatin')

            elif (x==7):

              

                result.append('nucleoli')

            elif(x==8):

               

                result.append('mitosis')

        tmp=tmp//10

    

    if len(result)==1:

        continue 

        #X_trainNew1=X_train.loc[:,result[0]]

    elif len(result)==2:

        X_testNew=X_test.loc[:,[result[0],result[1]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1]]]

    elif len(result)==3:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2]]]

    elif len(result)==4:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3]]]

    elif len(result)==5:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3],result[4]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4]]]

    elif len(result)==6:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5]]]

    elif len(result)==7:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6]]]

    elif len(result)==8:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7]]]

    elif len(result)==9:

        X_testNew=X_test.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8]]]

        X_trainNew1=X_train.loc[:,[result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8]]]

    

    GNB_data.fit(X_trainNew1, y_train)

    d=accuracy_score(GNB_data.predict(X_testNew),y_test)

    

    print(*result)

    print(d)

    allname5.append(result)

    allresult5.append(d)
print("The best subset for GaussianNB's train data in test data")

print(allresult5[allname5.index(allname4[allresult4.index(max(allresult4))])])

print("---------")

print("The best subset for GaussianNB's test data")

print(*allname5[allresult5.index(max(allresult5))])

print(max(allresult5))
import seaborn as sns

cv_tupels = [[f,'cv',x] for f,x in zip(range(511),allresult4)]

test_tupels = [[f,'test',x] for f,x in zip(range(511),allresult5)]

results = pd.DataFrame(cv_tupels+test_tupels,columns=['fold','algo','rmse'])

sns.scatterplot(data=results,x='fold',y='rmse',hue='algo')
from sklearn.model_selection import cross_val_predict

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

import sklearn.metrics as metrics

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import roc_auc_score



sgd1 = SGDClassifier(random_state=1337316)

a = breastData.iloc[:,1:10]

b = breastData.iloc[:,10]

sgd_scores=cross_val_predict(sgd1,a,b,cv=10,method="decision_function")

print(sgd_scores)





fpr,tpr,thresholds = roc_curve(y, sgd_scores)

fpr,tpr,thresholds







forest_clf = RandomForestClassifier(random_state=1337316)

randomForest_scores=cross_val_predict(forest_clf,a,b,cv=10,method="predict_proba")

print(randomForest_scores)



y_scores_forest = randomForest_scores[:,1]

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y, y_scores_forest)

fpr_forest, tpr_forest, thresholds_forest















gnb = GaussianNB()

gnb_scores=cross_val_predict(gnb,a,b,cv=10,method="predict_proba")



y_scores_gnb = gnb_scores[:,1]

fpr_gnb, tpr_gnb, thresholds_gnb = roc_curve(y, y_scores_gnb)

fpr_gnb,tpr_gnb,thresholds_gnb



def plot_roc_curve(fpr_forest, tpr_forest, label=None):

    plt.plot(fpr_forest, tpr_forest, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([0, 1, 0, 1])                                   

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16) 

    plt.grid(True)

    

plot_roc_curve(fpr_forest, tpr_forest, "Random Forest" )

plot_roc_curve(fpr_gnb, tpr_gnb, "GaussianNB" )

plot_roc_curve(fpr,tpr,"SGD")

plt.legend(loc="lower right")







print("This is the auc for sgd:")

auc_sgd=roc_auc_score(y,sgd_scores)

print(auc_sgd)

print("----------------")

print("This is the auc for Random_Forest:")

auc_forest=roc_auc_score(y,y_scores_forest)

print(auc_forest)

print("----------------")

print("This is the auc for GaussianNB:")

auc_gnb=roc_auc_score(y,y_scores_gnb)

print(auc_gnb)




