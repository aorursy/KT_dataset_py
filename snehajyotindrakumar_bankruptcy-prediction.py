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


#Polish Bankrupcy data analysis



import warnings



warnings.filterwarnings("ignore")



warnings.filterwarnings("ignore", category=DeprecationWarning)



# Basic Libraries for Data organization, Statistical operations and Plotting



import numpy as np



import pandas as pd



# For loading .arff files



from scipy.io import arff







from sklearn.preprocessing import Imputer







# Library imbalanced-learn to deal with the data imbalance. To use SMOTE oversampling



from imblearn.over_sampling import SMOTE 







# Importing classification models







from sklearn.svm import SVC



from sklearn.ensemble import RandomForestClassifier



from sklearn.linear_model import LogisticRegression



from imblearn.ensemble import BalancedBaggingClassifier



from sklearn.tree import DecisionTreeClassifier



from sklearn.naive_bayes import GaussianNB



from sklearn.linear_model import Perceptron



from sklearn.decomposition import PCA



from sklearn.preprocessing import StandardScaler



from sklearn.ensemble import GradientBoostingClassifier



from sklearn.metrics import mean_squared_error



from math import sqrt



from sklearn import metrics



from matplotlib import pyplot as plt



from sklearn.metrics import roc_curve, roc_auc_score











import random







from sklearn.metrics import f1_score



from sklearn.metrics import accuracy_score



from sklearn.metrics import precision_score



from sklearn.metrics import recall_score



from sklearn.metrics import classification_report



from sklearn.model_selection import train_test_split



from sklearn.metrics import confusion_matrix



from sklearn.metrics import roc_curve



from sklearn.metrics import precision_recall_curve



from scipy.io import arff















# Gaussian Naive Bayes classifier



Gaussian = GaussianNB()



# Logistic Regression classifier



LogisticReg = LogisticRegression(penalty = 'l1', random_state = 0)



# Decision Tree Classifier



DecisionTree = DecisionTreeClassifier(random_state=42)



# Random Forest Classifier



RandomForest = RandomForestClassifier(n_estimators = 5, criterion = 'entropy')



# SVM



svm = SVC(kernel='rbf', C=10, random_state=0, probability = True)



# Balanced Bagging Classifier



BalancedBagging = BalancedBaggingClassifier(base_estimator = RandomForestClassifier(criterion='entropy'), n_estimators = 5, bootstrap = True)



# neural 



# Perceptron



Ppn = Perceptron(max_iter=10, eta0=0.01, random_state=0)



# Grdient Boosting



GradientBoosting = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 2, random_state = 0)







models = [Gaussian,LogisticReg,DecisionTree,RandomForest,svm,BalancedBagging,Ppn,GradientBoosting] # xgb_classifier,







# perform data preprocessing and modeling



def perform_data_processing_modeling(models,df,year):



    



    df.drop_duplicates(inplace=True)



    df.dropna(thresh=50,inplace=True)



    imr=Imputer(missing_values='NaN', strategy='median', axis=0)



    imr=imr.fit(df)



    imputed_data=imr.transform(df.values)



    imputed_data_df=pd.DataFrame(imputed_data)



    df=pd.DataFrame(imputed_data_df.values,columns=["Attr1","Attr2","Attr3","Attr4","Attr5","Attr6","Attr7","Attr8","Attr9","Attr10","Attr11","Attr12","Attr13","Attr14","Attr15","Attr16","Attr17","Attr18","Attr19","Attr20","Attr21","Attr22","Attr23","Attr24","Attr25","Attr26","Attr27","Attr28","Attr29","Attr30","Attr31","Attr32","Attr33","Attr34","Attr35","Attr36","Attr37","Attr38","Attr39","Attr40","Attr41","Attr42","Attr43","Attr44","Attr45","Attr46","Attr47","Attr48","Attr49","Attr50","Attr51","Attr52","Attr53","Attr54","Attr55","Attr56","Attr57","Attr58","Attr59","Attr60","Attr61","Attr62","Attr63","Attr64","Class"])







    lf = pd.DataFrame(columns=["Data Year","F1SCORE", "ACCURACY", "RMSE", "Recalls","Precisions","MAE"])



    X = df.iloc[:,:-1]



    y = df.iloc[:,-1]



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



    sc=StandardScaler()



    sc.fit(X_train)



    X_train_std=sc.transform(X_train)



    X_test_std=sc.transform(X_test)



    sm = SMOTE(random_state=2)



    X_train_res, y_train_res = sm.fit_sample(X_train_std, y_train.ravel())



    



    pca=PCA(.95)



    pca.fit(X_train_std)







    X_train_std_pca=pca.transform(X_train_res)







    X_test_std_pca=pca.transform(X_test_std)







    X_train_final=pd.DataFrame(X_train_std_pca)







    X_test_final=pd.DataFrame(X_test_std_pca)



    cov_mat=np.cov(X_train_std.T)







    



    eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)







    tot=sum(eigen_vals)







    var_exp=[(i/tot) for i in sorted(eigen_vals, reverse=True)]







    cum_var_exp=np.cumsum(var_exp)











    plt.bar(range(1,65),var_exp,alpha=0.5,align='center',label='individual explained variance')







    plt.step(range(1,65),cum_var_exp,where='mid',label='cummulative explained variance')







    plt.ylabel('Explained Variance ratio')







    plt.xlabel('Principal components')







    plt.legend(loc='best')







    plt.tight_layout()







    plt.show()











    for i in range(len(models)):



        



        clf = models[i]







        clf = clf.fit(X_train_std_pca, y_train_res)



        y_test_predicted = clf.predict(X_test_final)



        f1=f1_score(y_test, y_test_predicted, average='weighted')



        acc = accuracy_score(y_test,y_test_predicted)



        rms = sqrt(mean_squared_error(y_test, y_test_predicted))



        recalls = recall_score(y_test, y_test_predicted)



        precisions = precision_score(y_test, y_test_predicted)



        mae = metrics.mean_absolute_error(y_test, y_test_predicted)



        if clf!=Ppn:



            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, clf.predict_proba(X_test_final)[:,1])



        



        



            list1 = [year,f1,acc,rms,recalls,precisions,mae]



            lf.loc[i] = list1



            plt.clf()



            plt.plot(false_positive_rate, true_positive_rate)



            plt.xlabel('FPR')



            plt.ylabel('TPR')



            plt.title('ROC curve-' + year + '-' + str(clf))



            plt.show()







        else:



            list1 = [year,f1,acc,rms,recalls,precisions,mae]



            lf.loc[i] = list1



    



    lf.index = ("GaussianNB", "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "SVM", "BalancedBaggingClassifier", "Perceptron", "GradientBoostingClassifier")



    #lf.to_csv('Model_Results.csv')



    print(lf)



    with open('Model_Results.csv', 'a') as f:



        lf.to_csv(f, header=True)



    f.close()



    return lf







#importing data sets



d1= arff.loadarff('../input/1year.arff')



df1 = pd.DataFrame(d1[0])



d2= arff.loadarff('../input/2year.arff')



df2 = pd.DataFrame(d2[0])



d3= arff.loadarff('../input/3year.arff')



df3 = pd.DataFrame(d3[0])



d4= arff.loadarff('../input/4year.arff')



df4 = pd.DataFrame(d4[0])



d5= arff.loadarff('../input/5year.arff')



df5 = pd.DataFrame(d5[0])



frames = [df1, df2, df3, df4, df5]



df_all = pd.concat(frames)



   







perform_data_processing_modeling(models, df1,"Year1")



perform_data_processing_modeling(models, df2,"Year2")



#perform_data_processing_modeling(models, df3,"Year3")



#perform_data_processing_modeling(models, df4,"Year4")



#perform_data_processing_modeling(models, df5,"Year5")



#perform_data_processing_modeling(models, df_all,"YearAll")




