#%% load library

import os

import time

import sklearn

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter

from scipy.stats import ttest_rel

from keras.preprocessing import image

from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

#Extract input from files

train_image = []

train_label = []



names_class_file=['cat', 'car', 'fruit', 'dog', 'person', 'flower', 'motorbike', 'airplane']

file='../input/natural-images/natural_images/'



#Extract picture features 

for j in range(len(names_class_file)):

    name_class_file=os.listdir(file+names_class_file[j])

    for i in range(len(name_class_file)):

        img = image.load_img(file+names_class_file[j]+'/'+name_class_file[i], target_size=(28,28,1), color_mode = "grayscale")

        img=np.array(img)

        img=img.reshape((1,784))

        train_image.append(img[0])

        train_label.append(j)  

        

data_x = np.array(train_image)/255 

data_y=np.array(train_label)



#Verify dimensions

print('Data Dimension:')

print('Number of Records:', data_x.shape[0])

print('Number of Features:', data_x.shape[1])

np.unique(data_y)

print(dict(Counter(data_y)))



#verify data integrity

summary=pd.DataFrame(dict(Counter(data_y)),index=[0])

summary.columns=names_class_file

print(summary)

# plot a historgram to show the distribution of data

plt.bar(range(len(list(summary.loc[0]))), list(summary.loc[0]),align = "center",color = "steelblue",alpha = 0.6)

plt.ylabel("Count")

plt.ylim([0,1100])

plt.xticks(range(8),names_class_file)

plt.title('class distribution')

for x,y in enumerate(list(summary.loc[0])):

    plt.text(x,y+3,'%s' %round(y,1),ha='center')

plt.show()   

    
#plot pir chart of dataset

plt.figure(figsize=(6,6))

explode = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] 

colors=['turquoise','greenyellow','darkgray','orange','cornflowerblue','tomato','y','violet'] 

  

plt.axes(aspect='equal')

 

plt.xlim(0,4)

plt.ylim(0,4)

 

plt.pie(x =  list(summary.iloc[0]), 

        explode=explode,

        labels=names_class_file,

        colors=colors,pctdistance=0.85, 



        autopct='%.1f%%',textprops = {'fontsize':12, 'color':'k','weight' : 'normal'})  



plt.pie(x =  list(summary.iloc[0]), radius=0.6,colors = 'w')

plt.xticks(())

plt.yticks(())

plt.title('dataset')

plt.show()
#reduce the dimension by PCA and determine the number of principal components for classsification



pca = PCA(n_components=784)

pca.fit(data_x)

#print(pca.explained_variance_ratio_)



total_explained_list=[]

g=0

for i in range(len(pca.explained_variance_ratio_)):

    total_explained=sum(pca.explained_variance_ratio_[:i])

    total_explained_list.append(total_explained)

    if total_explained>0.9 and g==0:

        components=i

        g=1

    else:

        pass



print('Number of the principal components that will be used for classification:', components)



plt.plot(total_explained_list)

plt.vlines(components, 0, 1,color="red")

plt.title('The percentage(accumulated) of variance that can be explained by principal components')

plt.xlabel('Number of principal components')

plt.ylabel('Percentage of variance')

plt.savefig('PCA.png')

plt.show()
# reduce the training set and test set by using first 213 Principal Components

pca = PCA(n_components=components)

pca.fit(data_x)

reduced_x=pca.fit_transform(data_x)

X_train, X_test, y_train, y_test = train_test_split(reduced_x, data_y, random_state=42, test_size=0.2)

print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
# plot confusion matrix to show the resuluts of classification.

def plot_confusion_matrix(cm, labels_name, title):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]   

    plt.imshow(cm, interpolation='nearest')    

    plt.title(title)    

    plt.colorbar()

    num_local = np.array(range(len(labels_name)))    

    plt.xticks(num_local, labels_name, rotation=0)  

    plt.yticks(num_local, labels_name)   

    plt.ylabel('True')    

    plt.xlabel('Predicted')

# model 1: knn

from sklearn.neighbors import KNeighborsClassifier

model_knn=KNeighborsClassifier() 



n_neighbors=list(range(2,21))

parameters_knn = [

    {

        'n_neighbors': n_neighbors,

    }

]

clf = GridSearchCV(model_knn, parameters_knn, cv=5, n_jobs=8)

clf.fit(X_train, y_train)

result_knn=clf.cv_results_

y_pred_knn=clf.predict(X_test)



# model function 

print("\n model: \n",clf)

print("\n best estimator: ",clf.best_estimator_)

print("\n best parameters: ",clf.best_params_)



# accuracy of model knn

accu_knn=clf.best_score_ # accuracy

print("\n best accuracy=","%.1f%%"  %(clf.best_score_*100))



# classification report

class_report_knn=sklearn.metrics.classification_report(y_test, y_pred_knn,digits=2, target_names=['cat', 'car', 'fruit', 'dog', 'person', 'flower', 'motorbike', 'airplane'],output_dict=True)

print("\n classification report \n:",sklearn.metrics.classification_report(y_test, y_pred_knn,digits=2, target_names=['cat', 'car', 'fruit', 'dog', 'person', 'flower', 'motorbike', 'airplane']))



# F1 score

print("\n F1 score=",sklearn.metrics.f1_score(y_test, y_pred_knn, labels=None, average='macro',pos_label=1, sample_weight=None))



# confusion matrix

print("\n confusion_matrix: \n",confusion_matrix(y_test, y_pred_knn))

plt.figure(figsize=(9,8))

cm=confusion_matrix(y_test, y_pred_knn)

plot_confusion_matrix(cm,names_class_file ,"confusion_matrix")



# plot accuracy vs parameters

plt.figure(figsize=(15,8))

plt.plot(n_neighbors,list(result_knn["mean_test_score"]),label="KNN",marker='o', mec='r', mfc='w')

for x, y in zip(n_neighbors, list(result_knn["mean_test_score"])):

    plt.text(x, y+0.001, "%.1f%%"  %(y*100), ha='center', va='bottom', fontsize=10.5)

plt.xlabel("K")

plt.ylabel("Accuracy")

plt.title("Model knn vs k")

plt.legend()  

plt.yticks([x/100 for x in list(range(50,70,5))])

#%% logistic regression

from sklearn.linear_model import LogisticRegression

model_LogisticRegression = LogisticRegression(multi_class="multinomial",solver="newton-cg")



parameter_lr1=[x/10 for x in list(range(1,10,1))]

parameters_lr = [{   "C" : parameter_lr1    }]



clf = GridSearchCV(model_LogisticRegression, parameters_lr, cv=5, n_jobs=8)

clf.fit(X_train, y_train)

result_lr=clf.cv_results_

y_pred_lr=clf.predict(X_test)



# model function 

print("\n model: \n",clf)

print("\n best estimator: ",clf.best_estimator_)

print("\n best parameters: ",clf.best_params_)



# accuracy of model lr

accu_lr=clf.best_score_ # accuracy

print("\n best accuracy=","%.1f%%"  %(clf.best_score_*100))



# classification report

class_report_lr=sklearn.metrics.classification_report(y_test, y_pred_lr,digits=2, target_names=['cat', 'car', 'fruit', 'dog', 'person', 'flower', 'motorbike', 'airplane'],output_dict=True)

print("\n classification report \n:",sklearn.metrics.classification_report(y_test, y_pred_lr,digits=2, target_names=['cat', 'car', 'fruit', 'dog', 'person', 'flower', 'motorbike', 'airplane']))



# F1 score

print("\n F1 score=",sklearn.metrics.f1_score(y_test, y_pred_lr, labels=None, average='macro',pos_label=1, sample_weight=None))



# confusion matrix

print("\n confusion_matrix: \n",confusion_matrix(y_test, y_pred_lr))

plt.figure(figsize=(9,8))

cm=confusion_matrix(y_test, y_pred_lr)

plot_confusion_matrix(cm,names_class_file ,"confusion_matrix")



# plot accuracy vs parameters

plt.figure(figsize=(15,8))

plt.plot(parameter_lr1,list(result_lr["mean_test_score"]),label="LR",marker='o', mec='r', mfc='w')

for x, y in zip(parameter_lr1, list(result_lr["mean_test_score"])):

    plt.text(x, y+0.001, "%.1f%%"  %(y*100), ha='center', va='bottom', fontsize=10.5)

plt.xlabel("C")

plt.ylabel("Accuracy")

plt.title("Model LR vs C")

plt.legend()  

plt.yticks([x/100 for x in list(range(50,80,5))])

#%% randomforest

from sklearn.ensemble import RandomForestClassifier

model_randomforest=RandomForestClassifier()



parameter_rf1=list(range(500,2500,500))

parameter_rf2=list(range(10,200,50))

parameters_rf =[

        { 

             'n_estimators':parameter_rf1,

             'max_features':parameter_rf2

             }

        ]

clf = GridSearchCV(model_randomforest,parameters_rf,cv=2)

clf.fit(X_train, y_train)

result_rf=clf.cv_results_

y_pred_rf=clf.predict(X_test)



# model function 

print("\n model: \n",clf)

print("\n best estimator: ",clf.best_estimator_)

print("\n best parameters: ",clf.best_params_)



# accuracy of model rf

accu_rf=clf.best_score_ # accuracy

print("\n best accuracy=","%.1f%%"  %(clf.best_score_*100))



# classification report

class_report_rf=sklearn.metrics.classification_report(y_test, y_pred_rf,digits=2, target_names=['cat', 'car', 'fruit', 'dog', 'person', 'flower', 'motorbike', 'airplane'],output_dict=True)

print("\n classification report \n:",sklearn.metrics.classification_report(y_test, y_pred_rf,digits=2, target_names=['cat', 'car', 'fruit', 'dog', 'person', 'flower', 'motorbike', 'airplane']))



# F1 score

print("\n F1 score=",sklearn.metrics.f1_score(y_test, y_pred_rf, labels=None, average='macro',pos_label=1, sample_weight=None))



# confusion matrix

print("\n confusion_matrix: \n",confusion_matrix(y_test, y_pred_rf))

plt.figure(figsize=(9,8))

cm=confusion_matrix(y_test, y_pred_rf)

plot_confusion_matrix(cm,names_class_file ,"confusion_matrix")



# plot accuracy vs parameters

result_rf_dic=dict()

for i in  range(len(parameter_rf1)*len(parameter_rf2)):

  if result_rf['params'][i].get("max_features") not in result_rf_dic:

    result_rf_dic[result_rf['params'][i].get("max_features")] = [list(result_rf["mean_test_score"])[i]]

  else:

    result_rf_dic[result_rf['params'][i].get("max_features")].append(list(result_rf["mean_test_score"])[i])



plt.figure(figsize=(15,8))

for g in range(len(parameter_rf2)):

    plt.plot(parameter_rf1,result_rf_dic[parameter_rf2[g]],label="max_features={}".format(parameter_rf2[g]),marker='o', mec='r', mfc='w')

    for x, y in zip(parameter_rf1, result_rf_dic[parameter_rf2[g]]):

        plt.text(x, y+0.001, "%.1f%%"  %(y*100), ha='center', va='bottom', fontsize=10.5)

    plt.xlabel("value of n_estimators ")

    plt.ylabel("crossValidation accuracy")

    plt.title("Model randomforest ")

    plt.legend()  

    plt.yticks([x/100 for x in list(range(70,85,5))])

from sklearn.svm import SVC

svc = SVC()

parameter_svm_c=[1,2,3,4,5,6,7,8,9,10]

parameter_svm_gamma= [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]



parameters_svm = [

    {

        'C': parameter_svm_c,

        'gamma': parameter_svm_gamma,

        'kernel': ['rbf']

    },

    {

        'C':parameter_svm_c,

        'kernel': ['linear']

    }

]

clf = GridSearchCV(svc, parameters_svm, cv=2)

clf.fit(X_train, y_train)

result_svm=clf.cv_results_

y_pred_svm=clf.predict(X_test)





# model function 

print("\n model: \n",clf)

print("\n best estimator: ",clf.best_estimator_)

print("\n best parameters: ",clf.best_params_)



# accuracy of model rf

accu_svm=clf.best_score_ # accuracy

print("\n best accuracy=","%.1f%%"  %(clf.best_score_*100))



# classification report

class_report_svm=sklearn.metrics.classification_report(y_test, y_pred_svm,digits=2, target_names=['cat', 'car', 'fruit', 'dog', 'person', 'flower', 'motorbike', 'airplane'],output_dict=True)

print("\n classification report \n:",sklearn.metrics.classification_report(y_test, y_pred_svm,digits=2, target_names=['cat', 'car', 'fruit', 'dog', 'person', 'flower', 'motorbike', 'airplane']))



# F1 score

print("\n F1 score=",sklearn.metrics.f1_score(y_test, y_pred_svm, labels=None, average='macro',pos_label=1, sample_weight=None))



# confusion matrix

print("\n confusion_matrix: \n",confusion_matrix(y_test, y_pred_svm))

plt.figure(figsize=(9,8))

cm=confusion_matrix(y_test, y_pred_svm)

plot_confusion_matrix(cm,names_class_file ,"confusion_matrix")



# plot accuracy vs parameters

result_svm_dic=dict()

for i in  range(len(parameter_svm_c)*len(parameter_svm_gamma)):

  if result_svm['params'][i].get("gamma") not in result_svm_dic:

    result_svm_dic[result_svm['params'][i].get("gamma")] = [list(result_svm["mean_test_score"])[i]]

  else:

    result_svm_dic[result_svm['params'][i].get("gamma")].append(list(result_svm["mean_test_score"])[i])

    

plt.figure(figsize=(15,8))

for g in range(len(parameter_svm_gamma)):

    plt.plot(parameter_svm_c,result_svm_dic[parameter_svm_gamma[g]],label="gamma={}".format(parameter_svm_gamma[g]),marker='o', mec='r', mfc='w')

    for x, y in zip(parameter_svm_c, result_svm_dic[parameter_svm_gamma[g]]):

        plt.text(x, y+0.001, "%.1f%%"  %(y*100), ha='center', va='bottom', fontsize=10.5)

    plt.xlabel("value of Cost for SVM")

    plt.ylabel("crossValidation accuracy")

    plt.title("Model SVM vs cost & gamma")

    plt.legend()  

    plt.yticks([x/100 for x in list(range(70,85,5))])
# comparison of model accuracy

model_list = [accu_knn,accu_lr,accu_rf,accu_svm]

plt.bar(["knn","lr","rf","svm"], model_list,color='rgbk')

for x, y in zip(["knn","lr","rf","svm"], model_list):

    plt.text(x, y+0.001, "%.1f%%"  %(y*100), ha='center', va='bottom', fontsize=10.5)

plt.yticks([x/100 for x in list(range(0,105,5))])

plt.show()


evaluation_method_dic=dict({"precision":0,"recall":1,"f1":2})

def evaluation(evaluation_method):

    df_classification_report=pd.DataFrame()

    for model in [class_report_rf,class_report_knn,class_report_lr,class_report_svm]:

        df_classification_report=df_classification_report.append( pd.DataFrame(model).iloc[evaluation_method_dic[evaluation_method],[0,1,2,3,4,5,6,7,10]] )

    df_classification_report.index=["KNN","LR","RF","SVM"]

    

    print(df_classification_report.T)

    df_classification_report.T.plot.bar(alpha=0.7,rot=0,legend=False)

    plt.title("comparision of {}".format(evaluation_method))

    plt.legend(bbox_to_anchor=(1.05,1.0),borderaxespad = -0.2)  

    plt.yticks([x/100 for x in list(range(0,105,10))])

    

    print(df_classification_report)

    df_classification_report.plot.bar(alpha=0.7,rot=0,legend=False)

    plt.title("comparision of {}".format(evaluation_method))

    plt.legend(bbox_to_anchor=(1.05,1.0),borderaxespad = -0.2)  

    plt.yticks([x/100 for x in list(range(0,105,10))])

    

    return df_classification_report



# compraison of model recall rate

class_report_recall=evaluation("recall")

# compraison of model precision

class_report_precision=evaluation("precision")

# compraison of model F1 Score

class_report_f1=evaluation("f1")

# T test for classification results

def ttest(table):

    ttesttable=pd.DataFrame(np.zeros(shape=(len(class_report_recall),len(class_report_recall))),index=class_report_recall.index,columns=class_report_recall.index)

    for each in [(0,1),(0,2),(0,3),(1, 2), (1, 3), (2, 3)]:

        # calculate p value

        print(each,table.index[each[0]],table.index[each[1]])

        x=table.iloc[each[0]]

        y=table.iloc[each[1]]

        p=ttest_rel(x, y)

        ttesttable.iloc[each]= p[1]

        ttesttable=ttesttable.replace(0,np.nan)

        print(p)

    print(ttesttable)   

    return ttesttable



# T test for recall rates between metrics

recall_pvalue=ttest(class_report_recall)
# T test for recall rates between metrics

f1_pvalue=ttest(class_report_f1)
# T test for precision between metrics

precision_pvalue=ttest(class_report_precision)