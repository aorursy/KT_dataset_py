"""

Importing Modules

"""

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import matplotlib as mp
"""

Importing Machine Learning Modules

"""

from sklearn.metrics import cohen_kappa_score as MC

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold

from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.metrics import f1_score

from sklearn.naive_bayes import GaussianNB

from sklearn import svm, datasets

from sklearn import linear_model

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

from sklearn.model_selection import cross_val_predict

from sklearn import metrics 
"""

Importing the Dataset

"""

data = pd.read_csv("../input/predicting-a-pulsar-star/pulsar_stars.csv")
data.head() # First 5 features of the dataset
sns.pairplot(data=data,

             palette=['red','orange'],

             hue="target_class",

             vars=[" Mean of the integrated profile",

                   " Excess kurtosis of the integrated profile",

                   " Skewness of the integrated profile",

                   " Mean of the DM-SNR curve",

                   " Excess kurtosis of the DM-SNR curve",

                   " Skewness of the DM-SNR curve"])



plt.suptitle("Pair Plot of HTRU2 Dataset ",fontsize=16)



plt.tight_layout()

plt.show() 
"""

Number of each target class in dataset using various charts

"""

Not_Pulsar_num = len(data[data['target_class'] == 0]) # Data where class isn't a Pulsar

Pulsar_num  = len(data[data['target_class'] == 1]) # Data where class is a Pulsar

num_list = [Not_Pulsar_num,Pulsar_num]

class_names = ['Not a Pulsar Star','Pulsar Star'] # names of classes used later



fig1, ax1 = plt.subplots()

ax1.bar(class_names,num_list, width=0.6,align='center',color=['red', 'orange'])

ax1.set_title("Bar Chart of each Target Class in Dataset");



fig2, ax2 = plt.subplots()

ax2.pie(num_list,labels=class_names, autopct='%1.1f%%',shadow=False, startangle=90,colors = ['red', 'orange'])

ax2.set_title("Percentage of each Target Class in Dataset");
columns = [' Mean of the integrated profile',

       ' Standard deviation of the integrated profile',

       ' Excess kurtosis of the integrated profile',

       ' Skewness of the integrated profile', ' Mean of the DM-SNR curve',

       ' Standard deviation of the DM-SNR curve',

       ' Excess kurtosis of the DM-SNR curve', ' Skewness of the DM-SNR curve']



color_range  = ["g","r","m","b","c","y","orange","k"] # colours used for each density plot



"""

looping through column list and colour range list

"""

for i in range(len(columns)): 

    sns.distplot(data[columns[i]],color = color_range[i],hist = True) # Distribution Plot for each Feature in Dataset

    plt.ylabel("Density")

    plt.show()
"""

All features where target class isn't a Pulsar

""" 

not_pulsar_data = data[data['target_class'] == 0]

"""

All features where target class is a Pulsar

"""

pulsar_data = data[data['target_class'] == 1]





Excess_not_pulsar = not_pulsar_data[' Excess kurtosis of the integrated profile']

Mean_not_pulsar = not_pulsar_data[' Mean of the integrated profile']

Stdev_not_pulsar = not_pulsar_data[' Standard deviation of the integrated profile']

skew_not_pulsar = not_pulsar_data[' Skewness of the integrated profile']





Excess_pulsar = pulsar_data[' Excess kurtosis of the integrated profile']

Mean_pulsar = pulsar_data[' Mean of the integrated profile']

Stdev_pulsar = pulsar_data[' Standard deviation of the integrated profile']

skew_pulsar = pulsar_data[' Skewness of the integrated profile']
"""

Plot of Standard deviation of the Integrated Profile vs Mean of the Integrated Profile

"""

plt.figure(figsize=(12, 6))



ax3 = plt.subplot(121)

ax3.scatter(Mean_not_pulsar,Stdev_not_pulsar,marker = '.',label = "Not a Pulsar",color = 'red')

ax3.scatter(Mean_pulsar,Stdev_pulsar,marker = '.', label = "Pulsar",color = 'orange')

ax3.legend(loc = 'best')

ax3.set_ylabel("Standard deviation of the Integrated Profile")

ax3.set_xlabel("Mean of the Integrated Profile")



"""

Plot of Skewness of the Integrated Profile vs Excess kurtosis of the Integrated Profile

"""

ax4 = plt.subplot(122)

ax4.scatter(skew_not_pulsar,Excess_not_pulsar,marker = '.',label = "Not a Pulsar",color = 'red')

ax4.scatter(skew_pulsar,Excess_pulsar,marker = '.', label = "Pulsar",color = 'orange')

ax4.legend(loc = 'best')

ax4.set_xlabel("Excess kurtosis of the integrated profile")

ax4.set_ylabel("Skewness of the integrated profile");
DM_SNR_Excess_not_pulsar = not_pulsar_data[' Excess kurtosis of the DM-SNR curve']

DM_SNR_Mean_not_pulsar = not_pulsar_data[' Mean of the DM-SNR curve']

DM_SNR_Stdev_not_pulsar = not_pulsar_data[' Standard deviation of the DM-SNR curve']

DM_SNR_skew_not_pulsar = not_pulsar_data[' Skewness of the DM-SNR curve']





DM_SNR_Excess_pulsar = pulsar_data[' Excess kurtosis of the DM-SNR curve']

DM_SNR_Mean_pulsar = pulsar_data[' Mean of the DM-SNR curve']

DM_SNR_Stdev_pulsar = pulsar_data[' Standard deviation of the DM-SNR curve']

DM_SNR_skew_pulsar = pulsar_data[' Skewness of the DM-SNR curve']

"""

Plot of Standard deviation of the DM-SNR curve vs Mean of the DM-SNR curve

"""



plt.figure(figsize=(12, 6))



ax5 = plt.subplot(121)

ax5.scatter(DM_SNR_Mean_not_pulsar,DM_SNR_Stdev_not_pulsar,marker = '.',label = "Not a Pulsar",color = 'red')

ax5.scatter(DM_SNR_Mean_pulsar,DM_SNR_Stdev_pulsar,marker = '.', label = "Pulsar",color = 'orange')

ax5.legend(loc = 'best')

ax5.set_ylabel("Standard deviation of the DM-SNR curve")

ax5.set_xlabel("Mean of the DM-SNR curve")



"""

Plot of Standard deviation of the Skewness vs Mean of the Excess kurtosis

"""

ax6 = plt.subplot(122)

ax6.scatter(DM_SNR_skew_not_pulsar,DM_SNR_Excess_not_pulsar,marker = '.',label = "Not a Pulsar",color = 'red')

ax6.scatter(DM_SNR_skew_pulsar,DM_SNR_Excess_pulsar,marker = '.', label = "Pulsar",color = 'orange')

ax6.legend(loc = 'best')

ax6.set_xlabel("Excess kurtosis of the DM-SNR curve")

ax6.set_ylabel("Skewness of the DM-SNR curve");
"""

CLASSIFICATION APPROACHES 

"""



"""

Column representing the different pulsar classes

"""

classes = data["target_class"]



"""

dropping the class column

"""

data = data.drop('target_class', axis=1)



"""

Splitting the Data for Testing and Training

"""

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=0.25, stratify=classes,random_state=4)
"""

Random Forest Classifier

"""

from sklearn.ensemble import RandomForestClassifier 



rf = RandomForestClassifier(max_depth=3,n_estimators = 100)

rf.fit(X_train,y_train)

y_pred_rf = rf.predict(X_test)



print(classification_report(y_test,y_pred_rf, target_names = class_names))



cm_rf = confusion_matrix(y_test, y_pred_rf) # Confusion Matrix



'''

Plotting Heat Map for Confusion Matrix

'''

sns.set(font_scale=1.2)

sns.heatmap(cm_rf, linewidths=0.5, cmap=sns.light_palette((0.4, 0.4, 0.7),n_colors=10000), annot=True)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.title("Confusion Matrix for RF Classifier");
"""

Decision Tree Classifier

"""

from sklearn import tree



trees = tree.DecisionTreeClassifier(max_depth = 3)

trees = trees.fit(X_train, y_train) 

y_pred_tree = trees.predict(X_test) 



print(classification_report(y_test,y_pred_tree, target_names = class_names))



cm_DT = confusion_matrix(y_test, y_pred_tree) # Confusion Matrix



'''

Plotting Heat Map for Confusion Matrix

'''

sns.set(font_scale=1.2)

sns.heatmap(cm_DT, linewidths=0.5, cmap=sns.light_palette((0.4, 0.4, 0.7),n_colors=10000), annot=True)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.title("Confusion Matrix for DT Classifier");
"""

k-Nearest Neighbor

"""

from sklearn.neighbors import KNeighborsClassifier 



knn = KNeighborsClassifier(n_neighbors = 3)

knn = knn.fit(X_train,y_train)

y_pred_knn = knn.predict(X_test)



print(classification_report(y_test,y_pred_knn, target_names = class_names))





cm_knn = confusion_matrix(y_test, y_pred_knn) # Confusion Matrix



'''

Plotting Heat Map for Confusion Matrix

'''

sns.set(font_scale=1.2)

sns.heatmap(cm_knn, linewidths=0.5, cmap=sns.light_palette((0.4, 0.4, 0.7),n_colors=10000), annot=True)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.title("Confusion Matrix for kNN Classifier");
"""

Linear Discriminant Analysis

"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 



lda=LDA(n_components=None)

fit = lda.fit(X_train,y_train) # fitting LDA to dataset

y_pred_lda=lda.predict(X_test) # predicting with LDA 



print(classification_report(y_test,y_pred_lda, target_names = class_names))



cm_lda = confusion_matrix(y_test, y_pred_lda) 



sns.set(font_scale=1.2)

sns.heatmap(cm_lda, linewidths=0.5, cmap=sns.light_palette((0.4, 0.4, 0.7),n_colors=10000), annot=True)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.title("Confusion Matrix for LDA Classifier");
"""

Gaussian Naive Bayes

"""

gnb = GaussianNB()

gnb.fit(X_train,y_train)

y_pred_gnb = gnb.predict(X_test)



print(classification_report(y_test,y_pred_gnb, target_names = class_names))



cm_gnb = confusion_matrix(y_test, y_pred_gnb) 



sns.set(font_scale=1.2)

sns.heatmap(cm_gnb, linewidths=0.5, cmap=sns.light_palette((0.4, 0.4, 0.7),n_colors=10000), annot=True)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.title("Confusion Matrix for GNB Classifier");
"""

Evaluating Feature Importance using Tree Based Models

"""

plt.figure(figsize=(20, 10))



ax7 = plt.subplot(121)

ax8 = plt.subplot(122)



ax7.barh(data.columns,trees.feature_importances_,align='center', height=0.2,label="DT",color = 'red')

ax7.set_title("Feature Importance Coefficient");

ax7.legend(loc = 'best');



ax8.barh(data.columns,rf.feature_importances_,align='center', height=0.2,label="RF",color = 'orange')

ax8.set_title("Feature Importance Coefficient");

ax8.legend(loc = 'best');



plt.tight_layout()

"""

Function for solving ROC Curve for any algorithm

"""

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



def ROC(Algorithm,X_train,y_train,X_test,y_test,graph_label):

    

    # predict probabilities

    Alg_probs = Algorithm.predict_proba(X_test)

    # keep probabilities for the positive outcome only

    Alg_probs = Alg_probs[:,1]

    # calculate scores

    Alg_score = roc_auc_score(y_test, Alg_probs)

    # calculate roc curves

    Alg_fpr, Alg_tpr, _ = roc_curve(y_test, Alg_probs)

    

    plt.plot(Alg_fpr, Alg_tpr, label = graph_label+": "+str(round(Alg_score,2)))

    # axis labels

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.legend(loc = 'best')

    

"""

Plotting the ROC Curve for each Classifier used

"""

plt.figure(figsize=(12, 6)) # setting the size of the figure



ROC(rf,X_train,y_train,X_test,y_test,"RF")

ROC(trees,X_train,y_train,X_test,y_test,"DT")

ROC(knn,X_train,y_train,X_test,y_test,"kNN")

ROC(lda,X_train,y_train,X_test,y_test,"LDA")

ROC(gnb,X_train,y_train,X_test,y_test,"GNB")



plt.plot([1,0],[1,0],color = 'black',linestyle = '--', label = 'No Skill: 0.5')

plt.legend(loc = 'best')



plt.title("ROC Curves for each machine learning algorithm");
"""

Function for plotting the Precision Recall Curve for each Classifier used

"""

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc



def Pre_Recall(Algorithm,X_train,y_train,X_test,y_test,graph_label):

    # predict probabilities

    Alg_probs = Algorithm.predict_proba(X_test)

    # keep probabilities for the positive outcome only

    Alg_probs = Alg_probs[:,1]

    # calculate scores

    Alg_score = roc_auc_score(y_test, Alg_probs)

    # calculate roc curves

    Alg_precision, Alg_recall, _ = precision_recall_curve(y_test, Alg_probs)

    auc_score = auc(Alg_recall, Alg_precision)

    plt.plot(Alg_recall,Alg_precision, label = graph_label + str(round(auc_score,2)))

    # axis labels

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.legend(loc = 'best')
"""

Plotting the Precision Recall Curve for each Classifier used

"""

plt.figure(figsize=(12, 6)) # setting the size of the figure



Pre_Recall(rf,X_train,y_train,X_test,y_test,"RF: ")

Pre_Recall(trees,X_train,y_train,X_test,y_test,"DT: ")

Pre_Recall(knn,X_train,y_train,X_test,y_test,"kNN: ")

Pre_Recall(lda,X_train,y_train,X_test,y_test,"LDA: ")

Pre_Recall(gnb,X_train,y_train,X_test,y_test,"GNB: ")



plt.title("Precision Recall Curve for each machine learning algorithm");