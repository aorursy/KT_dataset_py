import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
dataset=pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv", encoding='latin-1')

dataset.dropna(how="any", inplace=True, axis=1)

dataset.columns = ['label', 'sms']

# labels=dataset['labels']

# message=dataset['sms']
dataset.head()
dataset['label'].value_counts()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
y_data=dataset['label'].values
y_data=le.fit_transform(y_data)
print(y_data)
print(dataset['sms'])
import re

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
sw=set(stopwords.words('english'))

ps=PorterStemmer()
def cleantext(sample):

    sample=sample.lower()

    sample=sample.replace("<br /><br />"," ")

    sample=re.sub("[^a-zA-Z]+"," ",sample)

    

    sample=sample.split(" ")

    sample=[ps.stem(s) for s in sample if s not in sw] # stemming and removing stopwords

    

    sample=" ".join(sample)

    

    return sample
cleantext(dataset['sms'][0])
dataset['sms'][0]
# Apply clean text function to each sms



dataset['cleanedmessage']=dataset['sms'].apply(cleantext)
corpus=dataset['cleanedmessage'].values
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
#CountVectorizer transformer from the sklearn.feature_extraction model has its own internal tokenization

#and normalization methods



cv=CountVectorizer(max_df=0.5,max_features=50000)
x_data=cv.fit_transform(corpus)
x_data.shape
print(x_data[0])
# Assign weights to every word in the vocab using tf-idf

tfidf=TfidfTransformer()
x_data=tfidf.fit_transform(x_data)
print(x_data[0])
x_data.shape
y_data.shape
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.3,random_state=42)
#Function for plotting confusion matrix



import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)

    #plt.figure(figsize=[10,10])

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 verticalalignment="center",

                 color="blue" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()
# FUNCTION TO CALCULATE TRUE POSITIVE, TRUE NEGATIVE ,FALSE POSITIVE AND FALSE NEGATIVE 



def perf_measure(y_actual, y_hat):

    y_actual=np.array(y_actual)

    y_hat=np.array(y_hat)

    TP = 0

    FP = 0

    TN = 0

    FN = 0



    for i in range(len(y_hat)): 

        if y_actual[i]==y_hat[i] and y_hat[i]==1:

           TP += 1

        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:

           FP += 1

        if y_actual[i]==y_hat[i]==0:

           TN += 1

        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:

           FN += 1



    return(TP, FP, TN, FN)
from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier



from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score
clf_lr= LogisticRegression(solver='liblinear', penalty='l1')

clf_lr.fit(X_train, y_train)

pred_lr=clf_lr.predict(X_test)
clf_lr.score(X_test,y_test)
print(classification_report(y_test,pred_lr))
# VISUALIZNG CONFUSION MATRIX



cnf_matrix_lr=confusion_matrix(y_test,pred_lr)

#print(cnf_matrix_lr)

plot_confusion_matrix(cnf_matrix_lr,[0,1],normalize=False,title="Confusion Matrix")
# PLOTTING AUC-ROC CURVE



probs_lr= clf_lr.predict_proba(X_test)

probs_lr=probs_lr[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test,probs_lr)

plt.title("AUC-ROC curve--LR",color="green",fontsize=20)

plt.xlabel("False positive rate")

plt.ylabel("True positive rate")

plt.plot(fpr,tpr,linewidth=2, markersize=12)

plt.show()
clf_mnb=MultinomialNB(alpha=0.2)



clf_mnb.fit(X_train,y_train)

pred_mnb=clf_mnb.predict(X_test)

acc_mnb=clf_mnb.score(X_test,y_test)



#acc=accuracy_score(y_test,pred)

print("Accuracy : ",acc_mnb)
print(classification_report(y_test,pred_mnb))
# VISUALIZNG CONFUSION MATRIX



cnf_matrix_mnb=confusion_matrix(y_test,pred_mnb)

#print(cnf_matrix_mnb)

plot_confusion_matrix(cnf_matrix_mnb,[0,1],normalize=False,title="Confusion Matrix")
# PLOTTING AUC-ROC CURVE



probs_mnb= clf_mnb.predict_proba(X_test)

probs_mnb=probs_mnb[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test,probs_mnb)

plt.title("AUC-ROC curve--MNB",color="green",fontsize=20)

plt.xlabel("False positive rate")

plt.ylabel("True positive rate")

plt.plot(fpr,tpr,linewidth=2, markersize=12)

plt.show()
clf_knn= KNeighborsClassifier(n_neighbors=49)

clf_knn.fit(X_train,y_train)



pred_knn=clf_knn.predict(X_test)

acc_knn=clf_knn.score(X_test,y_test)



print("Accuracy : ",acc_knn)
print(classification_report(y_test,pred_knn))
# VISUALIZNG CONFUSION MATRIX



cnf_matrix_knn=confusion_matrix(y_test,pred_knn)

#print(cnf_matrix_knn)

plot_confusion_matrix(cnf_matrix_knn,[0,1],normalize=False,title="Confusion Matrix")
# PLOTTING AUC-ROC CURVE



probs_knn= clf_knn.predict_proba(X_test)

probs_knn=probs_knn[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test,probs_knn)

plt.title("AUC-ROC curve--KNN",color="green",fontsize=20)

plt.xlabel("False positive rate")

plt.ylabel("True positive rate")

plt.plot(fpr,tpr,linewidth=2, markersize=12)

plt.show()
clf_svm = svm.SVC(kernel='sigmoid', gamma=1.0,probability=True)



clf_svm.fit(X_train,y_train)

pred_svm=clf_svm.predict(X_test)

acc_svm=clf_svm.score(X_test,y_test)



print("Accuracy : ",acc_svm)
print(classification_report(y_test,pred_svm))
# VISUALIZNG CONFUSION MATRIX



cnf_matrix_svm=confusion_matrix(y_test,pred_svm)

#print(cnf_matrix_svm)

plot_confusion_matrix(cnf_matrix_svm,[0,1],normalize=False,title="Confusion Matrix")
# PLOTTING AUC-ROC CURVE



probs_svm= clf_svm.predict_proba(X_test)

probs_svm=probs_svm[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test,probs_svm)

plt.title("AUC-ROC curve--SVM",color="green",fontsize=20)

plt.xlabel("False positive rate")

plt.ylabel("True positive rate")

plt.plot(fpr,tpr,linewidth=2, markersize=12)

plt.show()
clf_dtc=DecisionTreeClassifier(random_state=0)



clf_dtc.fit(X_train,y_train)

pred_dtc=clf_dtc.predict(X_test)

acc_dtc=clf_dtc.score(X_test,y_test)



print("Accuracy : ",acc_dtc)
print(classification_report(y_test,pred_dtc))
# VISUALIZNG CONFUSION MATRIX



cnf_matrix_dtc=confusion_matrix(y_test,pred_dtc)

#print(cnf_matrix_dtc)

plot_confusion_matrix(cnf_matrix_dtc,[0,1],normalize=False,title="Confusion Matrix")
# PLOTTING AUC-ROC CURVE



probs_dtc= clf_dtc.predict_proba(X_test)

probs_dtc=probs_dtc[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test,probs_dtc)

plt.title("AUC-ROC curve--DTC",color="green",fontsize=20)

plt.xlabel("False positive rate")

plt.ylabel("True positive rate")

plt.plot(fpr,tpr,linewidth=2, markersize=12)

plt.show()
clf_rf= RandomForestClassifier(n_estimators=31, random_state=111)



clf_rf.fit(X_train,y_train)

pred_rf=clf_rf.predict(X_test)

acc_rf=clf_rf.score(X_test,y_test)



print("Accuracy : ",acc_rf)
print(classification_report(y_test,pred_rf))
# VISUALIZNG CONFUSION MATRIX



cnf_matrix_rf=confusion_matrix(y_test,pred_rf)

#print(cnf_matrix_rf)

plot_confusion_matrix(cnf_matrix_rf,[0,1],normalize=False,title="Confusion Matrix")
# PLOTTING AUC-ROC CURVE



probs_rf= clf_rf.predict_proba(X_test)

probs_rf=probs_rf[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test,probs_rf)



plt.title("AUC-ROC curve--RandomForest",color="green",fontsize=20)

plt.xlabel("False positive rate")

plt.ylabel("True positive rate")

plt.plot(fpr,tpr,linewidth=2, markersize=12)

plt.show()
clf_adb=AdaBoostClassifier(n_estimators=62, random_state=111)



clf_adb.fit(X_train,y_train)

pred_adb=clf_adb.predict(X_test)

acc_adb=clf_adb.score(X_test,y_test)



print("Accuracy : ",acc_adb)
print(classification_report(y_test,pred_adb))
# VISUALIZNG CONFUSION MATRIX



cnf_matrix_adb=confusion_matrix(y_test,pred_adb)

#print(cnf_matrix_adb)

plot_confusion_matrix(cnf_matrix_adb,[0,1],normalize=False,title="Confusion Matrix")
# PLOTTING AUC-ROC CURVE



probs_adb= clf_adb.predict_proba(X_test)

probs_adb=probs_adb[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test,probs_adb)



plt.title("AUC-ROC curve--AdaBoost",color="green",fontsize=20)

plt.xlabel("False positive rate")

plt.ylabel("True positive rate")

plt.plot(fpr,tpr,linewidth=2, markersize=12)

plt.show()
from keras import models

from keras.layers import Dense
model=models.Sequential()

model.add(Dense(16,activation='relu',input_shape=(X_train.shape[1],)))

model.add(Dense(16,activation='relu'))

model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=['accuracy'])
hist=model.fit(X_train,y_train,batch_size=128,epochs=100)
pred_mlp=model.predict(X_test)

pred_mlp[pred_mlp>=0.5]=1

pred_mlp[pred_mlp<0.5]=0

print(pred_mlp)
acc_mlp=accuracy_score(pred_mlp,y_test)

print(acc_mlp)
print(classification_report(y_test,pred_mlp))
# VISUALIZNG CONFUSION MATRIX



cnf_matrix_mlp=confusion_matrix(y_test,pred_mlp)

#print(cnf_matrix_mlp)

plot_confusion_matrix(cnf_matrix_mlp,[0,1],normalize=False,title="Confusion Matrix")
# PLOTTING AUC-ROC CURVE



probs_mlp= model.predict_proba(X_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test,probs_mlp)

plt.title("AUC-ROC curve--MLP",color="green",fontsize=20)

plt.xlabel("False positive rate")

plt.ylabel("True positive rate")

plt.plot(fpr,tpr,linewidth=2, markersize=12)

plt.show()
classifiers=[]



classifiers.append(('LogisticRegression',clf_lr))

classifiers.append(('MNB',clf_mnb))

classifiers.append(('KNN',clf_knn))

classifiers.append(('SVM',clf_svm))

classifiers.append(('Desicion Tree',clf_dtc))

classifiers.append(('Random Forest',clf_rf))

classifiers.append(('AdaBoost',clf_adb))

classifiers.append(('MLP',model))
result=[]

cnf_matric_parameter=[]

for i,v in classifiers:

    if i=='MLP':

        pred=v.predict(X_test)

        pred[pred>=0.5]=1

        pred[pred<0.5]=0

        print(pred)

        acc=accuracy_score(y_test,pred)

        precision = precision_score(y_test,pred)

        recall=recall_score(y_test, pred)

        f_measure=f1_score(y_test,pred)

        result.append((i,acc,precision,recall,f_measure))

        

        TP,FP,TN,FN=perf_measure(y_test,pred)

        cnf_matric_parameter.append((i,TP,FP,TN,FN))

        continue

        

    

    pred=v.predict(X_test)

    acc=accuracy_score(y_test,pred)

    precision = precision_score(y_test,pred)

    recall=recall_score(y_test, pred)

    #print(precision)

    f_measure=f1_score(y_test,pred)

    result.append((i,acc,precision,recall,f_measure))

    

    TP,FP,TN,FN=perf_measure(y_test,pred)

    cnf_matric_parameter.append((i,TP,FP,TN,FN))
column_names=['Algorithm','Accuracy','Precision','Recall','F-measure']

df1=pd.DataFrame(result,columns=column_names)

print(df1)
df1.plot(kind='bar', ylim=(0.65,1.0), figsize=(15,6), align='center', colormap="Accent")

plt.xticks(np.arange(8), df1['Algorithm'],fontsize=15)

plt.ylabel('Score',fontsize=20)

plt.title('Distribution by Classifier',fontsize=20)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=20)
column_names=['Algorithm','True_Pos','False_Pos','True_Neg','False_Neg']

df2=pd.DataFrame(cnf_matric_parameter,columns=column_names)

print(df2)
## save the result as a csv file to the disk



df1.to_csv("result_spam_sms_det.csv",index=True)

df2.to_csv("cnf_matrix_parameter.csv",index=True)
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import mean_squared_error

import math
EPSILON = 1e-10

def rae(actual: np.ndarray, predicted: np.ndarray):

    """ Relative Absolute Error (aka Approximation Error) """

    return np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual - np.mean(actual))) + EPSILON)



def rrse(actual: np.ndarray, predicted: np.ndarray):

    """ Root Relative Squared Error """

    return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))
performance_metrics=[]

for i,v in classifiers:

    if i=='MLP':

        pred=v.predict(X_test)

        pred[pred>=0.5]=1

        pred[pred<0.5]=0

        pred=pred.reshape(-1)

        #print(y_test.shape,pred.shape)

        

        mae=mean_absolute_error(y_test,pred)

        mcc=matthews_corrcoef(y_test,pred)

        mse=mean_squared_error(y_test,pred)

        rmse = math.sqrt(mse)

        rrsError=rrse(y_test,pred)

        raError=rae(y_test,pred)

        performance_metrics.append((i,mcc,mae,rmse,rrsError,raError))

        continue

        

    

    pred=v.predict(X_test)

    #print(y_test.shape,pred.shape)

    mae=mean_absolute_error(y_test,pred)

    mcc=matthews_corrcoef(y_test, pred)

    mse=mean_squared_error(y_test,pred)

    rmse = math.sqrt(mse)

    rrsError=rrse(y_test,pred)

    raError=rae(y_test,pred)

    performance_metrics.append((i,mcc,mae,rmse,rrsError,raError))
# corr_coef == Matthews correlation coefficient



column_names=['Algorithm','corr_coef','MAE','RMSE','RRSE','RAE']

df3=pd.DataFrame(performance_metrics,columns=column_names)

print(df3)
# saving the performance merics to the disk



df2.to_csv("performance_metrics.csv",index=True)