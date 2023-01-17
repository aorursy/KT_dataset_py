import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

from sklearn import metrics

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve

import cufflinks as cf

import plotly.plotly as py

import plotly.tools as tls

import plotly.graph_objs as go

from scipy import stats

from IPython.display import IFrame
Breastcancer=pd.read_csv('../input/Breastcancer_edited.csv')

Breastcancer.head()
Breastcancer.columns=(['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Diagnosis'])

Breastcancer.head()
Breastcancer.describe()
#Contructing distribution plots

sns.set_color_codes(palette='dark')

f, axes = plt.subplots(3,3, figsize=(20, 12))

f.suptitle("Distribution of the variables", fontsize=15)

b1=sns.kdeplot(Breastcancer['Clump Thickness'], shade=True, color='m',ax=axes[0,0])

b2=sns.kdeplot(Breastcancer['Uniformity of Cell Size'], shade=True, color='m', ax=axes[0,1])

b3=sns.kdeplot(Breastcancer['Uniformity of Cell Shape'], shade=True,color='m', ax=axes[1,0])

b4=sns.kdeplot( Breastcancer["Marginal Adhesion"],shade=True, color='m', ax=axes[1,1])

b5=sns.kdeplot( Breastcancer["Single Epithelial Cell Size"],shade=True, color='m', ax=axes[0,2])

b6=sns.kdeplot( Breastcancer["Bare Nuclei"],shade=True, color='m', ax=axes[1,2])

b7=sns.kdeplot( Breastcancer["Bland Chromatin"],shade=True, color='m', ax=axes[2,0])

b8=sns.kdeplot( Breastcancer["Normal Nucleoli"],shade=True, color='m', ax=axes[2,1])

b9=sns.kdeplot( Breastcancer["Mitoses"],shade=True, color='m', ax=axes[2,2])
r=Breastcancer.drop('Diagnosis',axis=1)

c1=sns.catplot(data=r, orient="v", kind="violin", height=8, aspect=2, palette='husl')
# Compute the correlation matrix

corr=Breastcancer.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool);mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))

f.suptitle("Correlation Matrix", fontsize=15)





# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

p=sns.heatmap(corr, mask=mask, cmap=cmap, center=0,square=True, linewidths=.4, cbar_kws={"shrink": .5})
IFrame('https://plot.ly/~rahul350rg/6/', width=850, height=800)
X = Breastcancer.drop('Diagnosis',axis=1).values

y = Breastcancer['Diagnosis'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42, stratify=y)

#Setup arrays to store training and test accuracies

neighbors = np.arange(1,60)

train_accuracy =np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



for i,k in enumerate(neighbors):

    #Setup a knn classifier with k neighbors

    knn = KNeighborsClassifier(n_neighbors=k)

    

    #Fit the model

    knn.fit(X_train, y_train)

    

    #Accuracy of the training set

    train_accuracy[i] = knn.score(X_train, y_train)

    

    #Accuracy of the test set

    test_accuracy[i] = knn.score(X_test, y_test) 

    

#Plot

sns.set_style('whitegrid')

plt.figure(figsize=(12,10))

plt.title('Finding the value of K optimal accuracy', size=15)

plt.plot(neighbors, test_accuracy, label='Testing Accuracy',color='purple')

plt.plot(neighbors, train_accuracy, label='Training accuracy', color='grey')

plt.legend()

plt.xlabel('Number of neighbors (K)',fontsize=15)

plt.ylabel('Accuracy',fontsize=15)

plt.savefig('K-NN.pdf')
#Setup a knn classifier with k neighbors

knn = KNeighborsClassifier(n_neighbors=24)

#Fit the model

knn.fit(X_train,y_train)
# Printing the accurcy of the model

Ac=knn.score(X_test,y_test)

print ('Accuracy of the model:',Ac)
#Predict and print

y_pred = knn.predict(X_test)

[TN,FP],[FN,TP]=confusion_matrix(y_test,y_pred)

print('TP:',TP)

print('TN:',TN)

print('FN:',FN)

print('FP:',FP)
Contigency_table = pd.DataFrame(np.array([[FN,TN], [TP,FP]]))

Contigency_table.columns=['Actual Malignant', 'Actual Benign']

Contigency_table.index=(['Tested Benign','Tested Malignant'])
Contigency_table
#Defining the measures of performance 

Accuracy=(TP+TN)/(TP+TN+FP+FN)

Sensitivity= TP/(TP+FN)

Specificity= TN/(TN+FP)

PPV=TP/(TP+FP)

NPV=TN/(TN+FN)

LRp=Sensitivity/(1-Specificity)

LRn=(1-Sensitivity)/Specificity



#Creating a table

Measures_of_test_performace= {'TP':[TP],'TN':[TN],'FP':[FP],'FN':[FN],

                              'Sensitivity': [Sensitivity], 

                              'Specificity': [Specificity],

                              'PPV':[PPV],

                              'NPV':[NPV],

                              'LR+':[LRp],

                              'LR−':[LRn],

                              'Accuracy':[Accuracy]}

Measures_of_test_performace=pd.DataFrame(data=Measures_of_test_performace)

Measures_of_test_performace.index=(['Vaules'])
Measures_of_test_performace.transpose()
print(classification_report(y_test,y_pred))
y_pred_proba = knn.predict_proba(X_test)[:,1]

auc = metrics.roc_auc_score(y_test, y_pred_proba)



sns.set_style('white')

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10,8))



plt.title('ROC curve: KNN,K=24')

plt.plot(fpr,tpr,label="AUC ="+str(auc),color='purple')

plt.plot([0,1],[0,1],'k--',color='black')

plt.xlabel('1-Specificity',fontsize=15, color='black')

plt.ylabel('Sensitivity',fontsize=15)

plt.legend(loc=4)







plt.savefig('KNN ROC.pdf')
Measures_of_test_performace
Breastcancer.groupby(['Diagnosis']).mean()
Breastcancer.describe()