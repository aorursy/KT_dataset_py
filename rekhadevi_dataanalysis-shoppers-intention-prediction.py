# Importing necessary Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from matplotlib import style
#Reading Dataset

df= pd.read_csv('../input/online-shoppers-intention/online_shoppers_intention.csv')

df.head()
#Checking datatypes

df.dtypes
#Checking null values

df.isnull().any()
df_final=df.dropna() #Removing Null items from out dataset



rw,col=df_final.shape 



print(f"We have final dataset with {rw} rows and {col} columns.")
revenue= df_final['Revenue'].value_counts()

#style.use('classic')

sns.set_style("darkgrid")

sns.set_context("talk")

plt.figure(figsize=(8,8))

x=revenue.index

y=revenue.values

outside=(0,0.1)

plt.pie(y,labels=x,autopct="%1.1f%%",startangle=90,explode=outside,shadow=True)

plt.title('Revenue')

plt.legend(loc='upper right')
# People visiting on Weekends and Visitor Types

#Weekends

plt.figure(figsize=(10,8))

plt.subplot(2,1,1)

sns.countplot(x=df_final['Weekend'],palette='twilight')



# Visitor Types

plt.figure(figsize=(10,8))

plt.subplot(2,1,2)

sns.countplot(x=df_final['VisitorType'])
month_wise_Count= df_final['Month'].value_counts().sort_values(ascending=False)

month_wise_Count
# Plotting Months



sns.set_style("ticks")

sns.set_context("notebook")

plt.figure(figsize=(10,8))

x=month_wise_Count.index

y=month_wise_Count.values

outside=(0.05,0,0,0,0,0,0,0,0,0)

plt.pie(y,labels=x,autopct="%1.1f%%",startangle=90,explode=outside,pctdistance=0.8,shadow=True,labeldistance=1.1)

plt.title('Month-wise Customers')

plt.legend(loc='best')
# Plotting OperatingSystems,Browser,Region and TrafficType



plt.figure(figsize=(10,9))



#Operating Systems

plt.subplot(2,2,1)

sns.countplot(x=df_final['OperatingSystems'],palette='twilight')



# Browser

plt.subplot(2,2,2)

sns.countplot(x=df_final['Browser'],palette="cubehelix")



# Regions

plt.subplot(2,2,3)

sns.countplot(x=df_final['Region'],palette="winter")



# Regions

plt.subplot(2,2,4)

sns.countplot(x=df_final['TrafficType'],palette="RdGy")
# Weekends/VisitorType vs Revenue

sns.set_style("ticks")

sns.set_context("talk")

df1 = pd.crosstab(df_final['Weekend'], df_final['Revenue'])

df1.plot(kind='bar',stacked=True,colormap='RdBu')

plt.title('Weekend vs Revenue', fontsize = 30)
# Visitor Types

sns.set_style("ticks")

sns.set_context("notebook")

df2 = pd.crosstab(df_final['VisitorType'], df_final['Revenue'])

df2.plot(kind='bar',stacked=True)

plt.title('Visitor Type vs Revenue', fontsize = 25)
# Traffic Type vs Revenue

sns.set_style("ticks")

sns.set_context("talk")

#plt.figure(figsize=(5,5))

df3 = pd.crosstab(df_final['TrafficType'], df_final['Revenue'])

df3.plot(kind='bar',stacked=True,colormap="Wistia")

plt.title('Traffic Type vs Revenue', fontsize = 25)
# Region vs Revenue

sns.set_style("ticks")

sns.set_context("talk")

#plt.figure(figsize=(5,5))

df4 = pd.crosstab(df_final['Region'], df_final['Revenue'])

df4.plot(kind='bar',stacked=True,colormap="nipy_spectral")

plt.title('Region Type vs Revenue', fontsize = 25)
# Browser vs Revenue

sns.set_style("ticks")

sns.set_context("talk")

#plt.figure(figsize=(5,5))

df4 = pd.crosstab(df_final['Browser'], df_final['Revenue'])

df4.plot(kind='bar',stacked=True,colormap="Wistia")

plt.title('Browser vs Revenue', fontsize = 25)
# Operating System vs Revenue

sns.set_style("ticks")

sns.set_context("talk")

#plt.figure(figsize=(5,5))

df5 = pd.crosstab(df_final['OperatingSystems'], df_final['Revenue'])

df5.plot(kind='bar',stacked=True,colormap="Set1")

plt.title('Operating System vs Revenue', fontsize = 25)
df_final.columns
sns.set_style("whitegrid")

sns.set_context("talk")

plt.figure(figsize=(10,8))

distcols=df_final[['Administrative', 'Administrative_Duration', 'Informational','Informational_Duration', 'ProductRelated', 'ProductRelated_Duration','BounceRates', 'ExitRates', 'PageValues', 'SpecialDay','Revenue']]

sns.heatmap(distcols.corr(),cmap="icefire")
# Distribution of columns w.r.t Revenue

sns.set_style("whitegrid")

sns.set_context("paper")

plt.figure(figsize=(9,9))

distcols1=df_final[['ProductRelated', 'ProductRelated_Duration','BounceRates','ExitRates','Revenue']]

sns.pairplot(distcols1,hue='Revenue')
# Checking distribution of Sepcial Days

sns.set_style("whitegrid")

sns.set_context("talk")

plt.figure(figsize=(7,5))

sns.distplot(df_final['SpecialDay'],kde=False,bins=15)
df_final.dtypes
#Converting Object datatypes into numeric to make machine readable

from sklearn.preprocessing import LabelEncoder

lbenc=LabelEncoder()

df_final['Month']= lbenc.fit_transform(df_final['Month'])

df_final['VisitorType']= lbenc.fit_transform(df_final['VisitorType'])
#df_final.dtypes
x=df_final.loc[:,df_final.columns!='Revenue'] #Independent Variables

x

y=df_final.loc[:,'Revenue'] #Dependent Variable
# Train_test Data Split

from sklearn.model_selection import train_test_split

x_trn,x_tst,y_trn,y_tst= train_test_split(x,y,test_size=.20,random_state=1)

print("x_trn",x_trn)

print("\n x_tst",x_tst)

print("\n \n y_trn",y_trn)

print("\n y_tst",y_tst)
#Applying Logistic Regression Algorithm

from sklearn.linear_model import LogisticRegression

lgr= LogisticRegression()



lgr.fit(x_trn,y_trn) #Training



predic_y1= lgr.predict(x_tst) #Predicting y values of x test based on training

predic_y1



# Checking accuracy

from sklearn.metrics import accuracy_score

logistic_accuracy=accuracy_score(y_tst,predic_y1)*100

print(f'\n Accuracy of Logistic Regression {logistic_accuracy}%.')
# checking no. of wrong results

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_tst,predic_y1))
from sklearn.neighbors import KNeighborsClassifier

kn=KNeighborsClassifier()

kn.fit(x_trn,y_trn)

predic_y2=kn.predict(x_tst)#Output Prediction

predic_y2



#Checking Accuracy score

KNN_accuracy=accuracy_score(y_tst,predic_y2)*100

KNN_accuracy



print(f'Accuracy of KNN algorithm is {KNN_accuracy}%.')



#Accurate values

confusion_matrix(y_tst,predic_y2)
#Applying Algorithm

from sklearn.svm import SVC

svc=SVC()

svc.fit(x_trn,y_trn)



predic_y3=svc.predict(x_tst)

print("Predicting output y:",predic_y3)



#Checking accuracy

from sklearn.metrics import accuracy_score

SVM_accuracy=accuracy_score(y_tst,predic_y3)*100

print(f'Accuracy score with dafault parameters of SVC is {SVM_accuracy}%.')
# rbf kernel with C and gamma values

from sklearn.svm import SVC

svc1=SVC(kernel="rbf",C=1.0,gamma=0.1,random_state=3)

svc1.fit(x_trn,y_trn)

predict_y6=svc1.predict(x_tst)

print("Output Prediction",predict_y6)



#Checking accuracy

SVM_accuracy_rbf= accuracy_score(y_tst,predict_y6)*100

print(f'Accuracy score with rbf kernelC,gamma values is {SVM_accuracy_rbf}%.')
# Sigmoid kernel

svc2=SVC(kernel='sigmoid')

svc2.fit(x_trn,y_trn)

predic_y9=svc2.predict(x_tst)

print("Output Prediction",predic_y9)



#Checking accuracy

SVM_accuracy_sig= accuracy_score(y_tst,predic_y9)*100

print(f'Accuracy score with Sigmoid kernel and C,gamma values:{SVM_accuracy_sig}%.')
# Checking best SVM Algorithm

SVM_Algorithms = ["SVM with Default Parameters","SVM with rbf Kernel","SVM with Sigmoid Kernel"]

Accuracy_Score=[SVM_accuracy,SVM_accuracy_rbf,SVM_accuracy_sig]



Best_SVM_Algorithm= pd.DataFrame({" SVM Algorithm":SVM_Algorithms,"Accuracy Score":Accuracy_Score})

Best_SVM_Algorithm
from sklearn.tree import DecisionTreeClassifier

dt= DecisionTreeClassifier(criterion='entropy',max_depth=17,random_state=3)

print("Decision Tree with parameters: Criteria-Entropy\n",dt.fit(x_trn,y_trn))



predic_y10= dt.predict(x_tst)

predic_y10



#Accuracy Check

from sklearn.metrics import accuracy_score

DT_accuracy= accuracy_score(y_tst,predic_y10)*100

print(f'\n Accuracy Score with Decision Tree Algoritm is {DT_accuracy}%.')
x.columns
#Plotting tree

from sklearn import tree

from sklearn.tree import export_graphviz 

data_feature= ['Administrative', 'Administrative_Duration', 'Informational','Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',

'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month','OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType',

'Weekend']

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(dt,out_file=None, feature_names=data_feature, filled = True,rounded=True))

display(SVG(graph.pipe(format='svg')))
# Applying Algorithm

from sklearn.ensemble import RandomForestClassifier

rnd_clf= RandomForestClassifier(n_estimators=10, random_state=3, max_depth=10, criterion = 'entropy')

rnd_clf.fit(x_trn,y_trn)



#Predicting Output

predic_y11= rnd_clf.predict(x_tst)

predic_y11



#Checking accuracy

RNDFrst_accuracy= accuracy_score(y_tst,predic_y11)*100

print(f'\n Accuracy Score with Decision Tree Algoritm is {RNDFrst_accuracy}%.')
#Plotting Random Forest Tree

estimators=rnd_clf.estimators_[5] # gives 5 decision trees

data_feature= ['Administrative', 'Administrative_Duration', 'Informational','Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',

'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month','OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType',

'Weekend']

from sklearn import tree

from graphviz import Source

from IPython.display import SVG  #SVG format

from IPython.display import display 



graph = Source(tree.export_graphviz(estimators, out_file=None,feature_names=data_feature,filled = True))

display(SVG(graph.pipe(format='svg')))
from sklearn.naive_bayes import GaussianNB

gauss= GaussianNB()

gauss.fit(x_trn,y_trn)



predict_y12= gauss.predict(x_tst) 

predict_y12



#Checking Accuracy

GaussNB_accuracy= accuracy_score(y_tst,predict_y12)*100

print(f'\n Accuracy Score with Decision Tree Algoritm is {GaussNB_accuracy}%.')
#Let us compare accuracy of each Algrithm

Algorithms = ["Logistic Regression"," K-NN Algorithm ","SVM ","Decision Tree","Random Forest","Gaussian NB"]

Accuracy_Score=[logistic_accuracy,KNN_accuracy,SVM_accuracy_rbf,DT_accuracy,RNDFrst_accuracy,GaussNB_accuracy]



#Checking Accuracies of various Algorithms

Algorithms_Accuracy= pd.DataFrame({" Algorithms":Algorithms,"Accuracy Score":Accuracy_Score})

Algorithms_Accuracy
Algorithms_Accuracy['Accuracy Score'].mean()
from sklearn.preprocessing import MinMaxScaler

minmax_sclr= MinMaxScaler()

x_trn1= minmax_sclr.fit_transform(x_trn) #Scaling x-train and test data

x_tst1=minmax_sclr.transform(x_tst)
#Applying Logistic Regression Algorithm

lgr_m= LogisticRegression()

lgr_m.fit(x_trn1,y_trn) #Training



#Predicting output

predic_y1_m= lgr_m.predict(x_tst1) #Predicting y values of x test based on training

predic_y1_m



# Checking accuracy

logistic_accuracy_m=accuracy_score(y_tst,predic_y1_m)*100

print(f'\n Accuracy of Logistic Regression using Min-Max Scaling is {logistic_accuracy_m}%.')



# checking no. of wrong results

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_tst,predic_y1_m))
# K-NN algorithm

kn_m=KNeighborsClassifier()

kn_m.fit(x_trn1,y_trn)

predic_y2_m=kn_m.predict(x_tst1)#Output Prediction

predic_y2_m



#Checking Accuracy score

KNN_accuracy_m=accuracy_score(y_tst,predic_y2_m)*100

KNN_accuracy_m



print(f'Accuracy of KNN algorithm using Min-Max Scaler is {KNN_accuracy_m}%.')



#Accurate values

confusion_matrix(y_tst,predic_y2_m)
# SVM algorithm with rbf kernel(C and gamma values)

svc1_m=SVC(kernel="rbf",C=1.0,gamma=0.1,random_state=1)

svc1_m.fit(x_trn1,y_trn)

predict_y6_m=svc1_m.predict(x_tst1)

print("Output Prediction",predict_y6_m)



#Checking accuracy

SVM_accuracy_rbf_m= accuracy_score(y_tst,predict_y6_m)*100

print(f'Accuracy score of SVM with rbf kernel(C,gamma values)is {SVM_accuracy_rbf_m}%.')
#Decision Tree Algorithm

dt_m= DecisionTreeClassifier(criterion='entropy',max_depth=17,random_state=3)

print("Decision Tree with parameters: Criteria-Entropy\n",dt_m.fit(x_trn1,y_trn))



predic_y10_m= dt_m.predict(x_tst1)

predic_y10_m



#Accuracy Check

DT_accuracy_m= accuracy_score(y_tst,predic_y10_m)*100

print(f'\n Accuracy Score with Decision Tree Algoritm using Min-Max Scaler is {DT_accuracy_m}%.')
# Applying Random Forest Algorithm

rnd_clf_m= RandomForestClassifier(n_estimators=10, random_state=3, max_depth=17, criterion = 'entropy')

rnd_clf_m.fit(x_trn1,y_trn)



#Predicting Output

predic_y11_m= rnd_clf_m.predict(x_tst1)

predic_y11_m



#Checking accuracy

RNDFrst_accuracy_m= accuracy_score(y_tst,predic_y11_m)*100

print(f'\n Accuracy Score with Decision Tree Algoritm with Min-Max Scaler is {RNDFrst_accuracy_m}%.')
# Gaussian Naive Bayes' Algorithm

gauss_m= GaussianNB()

gauss_m.fit(x_trn1,y_trn)



predict_y12_m= gauss_m.predict(x_tst1) 

predict_y12_m



#Checking Accuracy

GaussNB_accuracy_m= accuracy_score(y_tst,predict_y12_m)*100

print(f'\n Accuracy Score with Decision Tree Algoritm is {GaussNB_accuracy_m}%.')
#Comparing accuracy of each Algorithm

Algorithms1 = ["Logistic Regression"," K-NN Algorithm ","SVM ","Decision Tree","Random Forest","Gaussian NB"]

Accuracy_Score1=[logistic_accuracy_m,KNN_accuracy_m,SVM_accuracy_rbf_m,DT_accuracy_m,RNDFrst_accuracy_m,GaussNB_accuracy_m]



#Checking Accuracies of various Algorithms

Algorithms_Accuracy1= pd.DataFrame({" Algorithms":Algorithms1,"Accuracy Score":Accuracy_Score1})

Algorithms_Accuracy1



Algorithms_Accuracy1['Accuracy Score'].mean()
#Standard Scaling

from sklearn.preprocessing import StandardScaler

std_sclr = StandardScaler()

x_trn2 = std_sclr.fit_transform(x_trn)

x_tst2 = std_sclr.transform(x_tst)



#print("x_trn2:\n ",x_trn2)

#print("x_tst2:\n ",x_tst2)



#Dimenstionality Reduction Using PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=10)

x_trn_P=pca.fit_transform(x_trn2)

x_tst_P= pca.transform(x_tst2)

print("Old dimension of training dataset:",x_trn2.shape)

print("Reduced Dimension of training datset:",x_trn_P.shape)
print(pca.explained_variance_ratio_)
#Applying Logistic Regression Algorithm

lgr_p= LogisticRegression()

lgr_p.fit(x_trn_P,y_trn) #Training



#Predicting output

predic_y1_P= lgr_p.predict(x_tst_P) #Predicting y values of x test based on training

predic_y1_P



# Checking accuracy

logistic_accuracy_P=accuracy_score(y_tst,predic_y1_P)*100

print(f'\n Accuracy of Logistic Regression using PCA is {logistic_accuracy_P}%.')



# checking no. of wrong results

print(confusion_matrix(y_tst,predic_y1_P))
# K-NN algorithm

kn_p=KNeighborsClassifier()

kn_p.fit(x_trn_P,y_trn)

predic_y2_P=kn_p.predict(x_tst_P)#Output Prediction

predic_y2_P



#Checking Accuracy score

KNN_accuracy_P=accuracy_score(y_tst,predic_y2_P)*100

KNN_accuracy_P



print(f'Accuracy of KNN algorithm using PCA is {KNN_accuracy_P}%.')



#Accurate values

confusion_matrix(y_tst,predic_y2_P)
# SVM algorithm with rbf kernel(C and gamma values)

svc1_p=SVC(kernel="rbf",C=1.0,gamma=0.1,random_state=2)

svc1_p.fit(x_trn_P,y_trn)

predict_y6_P=svc1_p.predict(x_tst_P)

print("Output Prediction",predict_y6_P)



#Checking accuracy

SVM_accuracy_rbf_P= accuracy_score(y_tst,predict_y6_P)*100

print(f'\n Accuracy score of SVM with rbf kernel(C,gamma values) using PCA is {SVM_accuracy_rbf_P}%.')
#Decision Tree Algorithm

dt_p= DecisionTreeClassifier(criterion='entropy',max_depth=17,random_state=3)

print("Decision Tree with parameters: Criteria-Entropy\n",dt_p.fit(x_trn_P,y_trn))



predic_y10_P= dt_p.predict(x_tst_P)

predic_y10_P



#Accuracy Check

DT_accuracy_P= accuracy_score(y_tst,predic_y10_P)*100

print(f'\n Accuracy Score with Decision Tree Algoritm using PCA is {DT_accuracy_P}%.')
# Applying Random Forest Algorithm

rnd_clf_p= RandomForestClassifier(n_estimators=10, random_state=3, max_depth=17, criterion = 'entropy')

rnd_clf_p.fit(x_trn_P,y_trn)



#Predicting Output

predic_y11_P= rnd_clf_p.predict(x_tst_P)

predic_y11_P



#Checking accuracy

RNDFrst_accuracy_P= accuracy_score(y_tst,predic_y11_P)*100

print(f'\n Accuracy Score with Decision Tree Algoritm with PCA is {RNDFrst_accuracy_P}%.')
# Gaussian Naive Bayes' Algorithm

gauss_p= GaussianNB()

gauss_p.fit(x_trn_P,y_trn)



predict_y12_P= gauss_p.predict(x_tst_P) 

predict_y12_P



#Checking Accuracy

GaussNB_accuracy_P= accuracy_score(y_tst,predict_y12_P)*100

print(f'\n Accuracy Score with Decision Tree Algoritm using PCA is {GaussNB_accuracy_P}%.')
#Comparing accuracy of each Algorithm using PCA

Algorithms2 = ["Logistic Regression"," K-NN Algorithm ","SVM ","Decision Tree","Random Forest","Gaussian NB"]

Accuracy_Score2=[logistic_accuracy_P,KNN_accuracy_P,SVM_accuracy_rbf_P,DT_accuracy_P,RNDFrst_accuracy_P,GaussNB_accuracy_P]



#Checking Accuracies of various Algorithms using PCA

Algorithms_Accuracy2= pd.DataFrame({" Algorithms":Algorithms2,"Accuracy Score":Accuracy_Score2})

Algorithms_Accuracy2
Algorithms_Accuracy2['Accuracy Score'].mean()
#Let us combine all results in one Dataframe and plot in a line graph for best understanding.

Algo_vs_Accuracy= pd.DataFrame({'Algorithm': Algorithms,'Accuracy score(without Scaling)':Accuracy_Score,'Accuracy score(Min-Max Scaling)':Accuracy_Score1,'Accuracy score(using PCA)':Accuracy_Score2})

Algo_vs_Accuracy
#Plotting the results

from matplotlib import style

style.use('seaborn-deep')

sns.set_context("talk")

plt.figure(figsize=(22,10))

plt.plot(Algorithms,Accuracy_Score,'yo--',label='Without Scaling',linewidth=4, markersize=12)

plt.plot(Algorithms,Accuracy_Score1,'ro-',label='Using Min-Max Scaling',linewidth=4, markersize=12)

plt.plot(Algorithms,Accuracy_Score2,'bo-',label='Using PCA',linewidth=4, markersize=12)

plt.legend()

plt.title("Algorithms vs Accuracy Scores",fontsize=25)