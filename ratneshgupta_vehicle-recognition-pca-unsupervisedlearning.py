#import the necessary libraries
import os

import warnings
warnings.filterwarnings('ignore')

#import the necessary libraries
import numpy as np
import pandas as pd

#Importing libraries for visulization

import matplotlib.pyplot as plt
import seaborn as sns

#Library for Data Pre-processing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#Traditional Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#Decision Tree and other Ensemble Techniques
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier

#Library for Model Evaluation 
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve

#Other Libraries
from collections import Counter
from scipy import stats
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA
from scipy.stats import zscore
#load the csv file and make the data frame
vehicle_df = pd.read_csv('/kaggle/input/vehicle/vehicle.csv')
#display the first 5 rows of dataframe
vehicle_df.head()
print("The dataframe has {} rows and {} columns".format(vehicle_df.shape[0],vehicle_df.shape[1]))
#display the information of dataframe
vehicle_df.info()
#display in each column how many null values are there
vehicle_df.apply(lambda x: sum(x.isnull()))
#display 5 point summary of dataframe
#vehicle_df.describe().transpose()
vehicle_df.describe().T
sns.pairplot(vehicle_df,diag_kind='kde', hue='class')
plt.show()
#Corelation Matrix of attributes 
vehicle_df.corr()
#Function for Null values treatment

def null_values(base_dataset):
    print("Shape of DataFrame before null treatment",base_dataset.shape)
    print("Null values count before treatment")
    print("===================================")
    print(base_dataset.isna().sum(),"\n")
    ## null value percentage     
    null_value_table=(base_dataset.isna().sum()/base_dataset.shape[0])*100
    ## null value percentage beyond threshold drop , else treat the columns    
    retained_columns=null_value_table[null_value_table<30].index
    # if any variable as null value greater than input(like 30% of the data) value than those variable are consider as drop
    drop_columns=null_value_table[null_value_table>30].index
    base_dataset.drop(drop_columns,axis=1,inplace=True)
    len(base_dataset.isna().sum().index)
    #cont=base_dataset.describe().columns
    cont=[col for col in base_dataset.select_dtypes(np.number).columns ]
    cat=[i for i in base_dataset.columns if i not in base_dataset.describe().columns]
    for i in cat:
        base_dataset[i].fillna(base_dataset[i].value_counts().index[0],inplace=True)
    for i in cont:
        base_dataset[i].fillna(base_dataset[i].mean(),inplace=True)
    print("Null values counts after treatment")
    print("===================================")
    print(base_dataset.isna().sum())
    print("\nShape of DataFrame after null treatment",base_dataset.shape)
null_values(vehicle_df)
#display 5 point summary of new dataframe
#vehicle_df.describe().transpose()
vehicle_df.describe().T
#display the shape of dataframe
print("Shape of dataframe after missing values treatment:",vehicle_df.shape)
#Distribution of data

vehicle_df.hist( figsize=(15,15), color='red')
plt.show()
num_features=[col for col in vehicle_df.select_dtypes(np.number).columns ]

plt.figure(figsize=(20,20))
for i,col in enumerate(num_features,start=1):
    plt.subplot(5,4,i);
    sns.distplot(vehicle_df[col])
plt.show()
num_features=[col for col in vehicle_df.select_dtypes(np.number).columns ]

plt.figure(figsize=(20,20))
for i,col in enumerate(num_features,start=1):
    plt.subplot(5,4,i);
    sns.boxplot(vehicle_df[col]);
plt.show()

num_features=[col for col in vehicle_df.select_dtypes(np.number).columns ]

plt.figure(figsize=(20,20))
for i,col in enumerate(num_features,start=1):
    plt.subplot(5,4,i);
    sns.boxplot(vehicle_df['class'],vehicle_df[col]);
plt.show()
vehicle_df.skew()
def outliers_transform_with_drop_record(base_dataset):
    num_features=[col for col in base_dataset.select_dtypes(np.number).columns ]
    print("Outliers in Dataset before Treatment")
    print("====================================")
    for i,cols in enumerate(num_features,start=1):
        x = base_dataset[cols]
        qr3, qr1=np.percentile(x, [75,25])
        iqr=qr3-qr1
        utv=qr3+(1.5*(iqr))
        ltv=qr1-(1.5*(iqr))
        count=(base_dataset[base_dataset[cols]>utv][cols].count())+(base_dataset[base_dataset[cols]<ltv][cols].count()) 
        print("Column ",cols,"\t has ",count," outliers")
        
    for i,cols in enumerate(num_features,start=1):
        x = base_dataset[cols]
        qr3, qr1=np.percentile(x, [75,25])
        iqr=qr3-qr1
        utv=qr3+(1.5*(iqr))
        ltv=qr1-(1.5*(iqr))
        for p in x:
            if p <ltv or p>utv:
                base_dataset.drop(base_dataset[base_dataset[cols]>utv].index, axis=0, inplace=True)
                base_dataset.drop(base_dataset[base_dataset[cols]<ltv].index, axis=0, inplace=True)
    
    print("\nOutliers in Dataset after Treatment")
    print("====================================")
    for i,cols in enumerate(num_features,start=1):
        x = base_dataset[cols]
        qr3, qr1=np.percentile(x, [75,25])
        iqr=qr3-qr1
        utv=qr3+(1.5*(iqr))
        ltv=qr1-(1.5*(iqr))
        count=(base_dataset[base_dataset[cols]>utv][cols].count())+(base_dataset[base_dataset[cols]<ltv][cols].count()) 
        print("Column ",cols,"\t has ",count," outliers")
#outliers_transform_with_drop_record(vehicle_df)
def outliers_transform_with_replace_mean(base_dataset):
    num_features=[col for col in base_dataset.select_dtypes(np.number).columns ]
    print("Outliers in Dataset before Treatment")
    print("====================================")
    for i,cols in enumerate(num_features,start=1):
        x = base_dataset[cols]
        qr3, qr1=np.percentile(x, [75,25])
        iqr=qr3-qr1
        utv=qr3+(1.5*(iqr))
        ltv=qr1-(1.5*(iqr))
        count=(base_dataset[base_dataset[cols]>utv][cols].count())+(base_dataset[base_dataset[cols]<ltv][cols].count()) 
        print("Column ",cols,"\t has ",count," outliers")
        
    for i,cols in enumerate(num_features,start=1):
        x = base_dataset[cols]
        qr3, qr1=np.percentile(x, [75,25])
        iqr=qr3-qr1
        utv=qr3+(1.5*(iqr))
        ltv=qr1-(1.5*(iqr))
        y=[]
        for p in x:
            if p <ltv or p>utv:
                y.append(np.mean(x))
            else:
                y.append(p)
        base_dataset[cols]=y
                
    print("\nOutliers in Dataset after Treatment")
    print("====================================")
    for i,cols in enumerate(num_features,start=1):
        x = base_dataset[cols]
        qr3, qr1=np.percentile(x, [75,25])
        iqr=qr3-qr1
        utv=qr3+(1.5*(iqr))
        ltv=qr1-(1.5*(iqr))
        count=(base_dataset[base_dataset[cols]>utv][cols].count())+(base_dataset[base_dataset[cols]<ltv][cols].count()) 
        print("Column ",cols,"\t has ",count," outliers")
outliers_transform_with_replace_mean(vehicle_df)
#display how many are car,bus,van. 
new_vehicle_df['class'].value_counts()
sns.countplot(new_vehicle_df['class'])
plt.show()
#find the correlation between independent variables
plt.figure(figsize=(20,5))
sns.heatmap(vehicle_df.corr(),annot=True)
plt.show()
corr = vehicle_df.drop('class', axis=1).corr() # We already examined SalePrice correlations
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);
vehicle_df.replace({'car':0,'bus':1,'van':2},inplace=True)
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        print("Train Result:\n=============")
        print(f"accuracy score: {accuracy_score(y_train, pred):.4f}\n")
        #print(f"Classification Report: \n \tPrecision: {precision_score(y_train, pred,average=None)}\n\tRecall Score: {recall_score(y_train, pred,average=None)}\n\tF1 score: {f1_score(y_train, pred,average=None)}\n")
        print(f"Confusion Matrix:\n=================\n {confusion_matrix(y_train, clf.predict(X_train))}\n")
        print("Classification Report:\n======================\n",classification_report(y_train, pred))
        
    elif train==False:
        pred = clf.predict(X_test)
        print("Test Result:\n============")        
        print(f"accuracy score: {accuracy_score(y_test, pred)}\n")
        #print(f"Classification Report: \n \tPrecision: {precision_score(y_test, pred,average=None)}\n\tRecall Score: {recall_score(y_test, pred,average=None)}\n\tF1 score: {f1_score(y_test, pred,average=None)}\n")
        print(f"Confusion Matrix:\n===============\n {confusion_matrix(y_test, pred)}\n")
        print("Classification Report:\n======================\n",classification_report(y_test, pred))
#now separate the dataframe into dependent and independent variables
X = vehicle_df.drop('class',axis=1)
Y = vehicle_df['class']
print("shape of X :", X.shape)
print("shape of Y :", Y.shape)
from sklearn.model_selection import cross_val_score, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=5)
from sklearn.svm import SVC

lsvm = SVC(kernel='linear')
lsvm.fit(X_train, y_train)

print_score(lsvm, X_train, y_train, X_test, y_test, train=True)
print_score(lsvm, X_train, y_train, X_test, y_test, train=False)


lsvm_accuracy=accuracy_score(y_test, lsvm.predict(X_test))
from sklearn.svm import SVC

psvm = SVC(kernel='poly', degree=2, gamma='auto')
psvm.fit(X_train, y_train)

print_score(psvm, X_train, y_train, X_test, y_test, train=True)
print_score(psvm, X_train, y_train, X_test, y_test, train=False)

lsvm_accuracy=accuracy_score(y_test, psvm.predict(X_test))
from sklearn.svm import SVC

rsvm = SVC(kernel='rbf', gamma=1)
rsvm.fit(X_train, y_train)

print_score(rsvm, X_train, y_train, X_test, y_test, train=True)
print_score(rsvm, X_train, y_train, X_test, y_test, train=False)

rsvm_accuracy=accuracy_score(y_test, rsvm.predict(X_test))
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
X_std = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std, Y, test_size=0.3, random_state=5)
print("=======================Linear Kernel SVM==========================")

from sklearn.svm import SVC

lsvm = SVC(kernel='linear')
lsvm.fit(X_train, y_train)

print_score(lsvm, X_train, y_train, X_test, y_test, train=True)
print_score(lsvm, X_train, y_train, X_test, y_test, train=False)

lsvm_accuracy=accuracy_score(y_test, lsvm.predict(X_test))

print("=======================Polynomial Kernel SVM==========================")
from sklearn.svm import SVC

psvm = SVC(kernel='poly', degree=2, gamma='auto')
psvm.fit(X_train, y_train)

print_score(psvm, X_train, y_train, X_test, y_test, train=True)
print_score(psvm, X_train, y_train, X_test, y_test, train=False)

psvm_accuracy=accuracy_score(y_test, psvm.predict(X_test))

print("=======================Radial Kernel SVM==========================")
from sklearn.svm import SVC

rsvm = SVC(kernel='rbf', gamma=1)
rsvm.fit(X_train, y_train)

print_score(rsvm, X_train, y_train, X_test, y_test, train=True)
print_score(rsvm, X_train, y_train, X_test, y_test, train=False)

rsvm_accuracy=accuracy_score(y_test, rsvm.predict(X_test))

result = pd.DataFrame({'Model' : ['SVM Linear', 'SVM Polynomial', 'SVM Redial'], 
                       'Test Accuracy' : [lsvm_accuracy, psvm_accuracy, rsvm_accuracy],
                      })
result
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 0.5, 1, 10, 100], 
              'gamma': [1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001], 
              'kernel': ['rbf', 'poly', 'linear']} 

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5, iid=True)

grid.fit(X_train, y_train)

print_score(grid, X_train, y_train, X_test, y_test, train=True)
print_score(grid, X_train, y_train, X_test, y_test, train=False)
from sklearn.model_selection import KFold, cross_val_score


kfold = KFold(n_splits= 10, random_state = 1)

#instantiate the object
svc = SVC(kernel='linear') 


#now we will train the model with raw data

results = cross_val_score(estimator = svc, X = X_train, y = y_train, cv = kfold)

print(results,"\n")

print("Accuracy: %0.2f (+/- %0.2f)" % (results.mean()*100, results.std()*100 * 2))

kf_accuracy=results.mean()
from sklearn.model_selection import RepeatedKFold

X = vehicle_df.drop('class',axis=1).values
y = vehicle_df['class'].values

accuracies = []
#lr = LogisticRegression(random_state = 1)
svc = SVC(kernel='linear') 

rkf = RepeatedKFold(n_splits = 10, n_repeats= 3, random_state = 1)

for train_index, test_index in rkf.split(X):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    svc.fit(X_train, y_train)
    accuracies.append(accuracy_score(y_test, svc.predict(X_test)))

print(np.round(accuracies, 3),"\n")

print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(accuracies)*100, np.std(accuracies)*100 * 2))

rkf_accuracy=np.mean(accuracies)
result = pd.DataFrame({'Model' : ['Linear SVM', 'Linear SVM K-Fold', 'Linear SVM Repeated K-Fold'], 
                       'Accuracy' : [lsvm_accuracy, kf_accuracy, rkf_accuracy],
                      })
result
#now sclaed the features attribute and replace the target attribute values with number
X = vehicle_df.drop('class',axis=1)
y = vehicle_df['class']

X_scaled = X.apply(zscore)
#make the covariance matrix and we have 18 independent features so aur covariance matrix is 18*18 matrix
cov_matrix = np.cov(X_scaled,rowvar=False)
print("cov_matrix shape:",cov_matrix.shape)
print("Covariance_matrix",cov_matrix)
#now with the help of above covariance matrix we will find eigen value and eigen vectors
pca = PCA(n_components=18)
pca.fit(X_scaled)
#display explained variance ratio
pca_to_learn_variance.explained_variance_ratio_
#display explained variance
pca_to_learn_variance.explained_variance_
#display principal components
pca_to_learn_variance.components_
plt.bar(list(range(1,19)),pca_to_learn_variance.explained_variance_ratio_)
plt.xlabel("eigen value/components")
plt.ylabel("variation explained")
plt.show()
plt.step(list(range(1,19)),np.cumsum(pca_to_learn_variance.explained_variance_ratio_))
plt.xlabel("eigen value/components")
plt.ylabel("cummalative of variation explained")
plt.show()
#use first 8 principal components
pca_8c = PCA(n_components=8)
pca_8c.fit(X_scaled)
#transform the raw data which is in 18 dimension into 8 new dimension with pca
X_scaled_pca_8c = pca_8c.transform(X_scaled)
#display the shape of new_vehicle_df_pca_independent_attr
X_scaled_pca_8c.shape
#now split the data into 80:20 ratio
rawdata_X_train,rawdata_X_test,rawdata_y_train,rawdata_y_test = train_test_split(X_scaled,Y,test_size=0.20,random_state=1)
pca_X_train,pca_X_test,pca_y_train,pca_y_test = train_test_split(X_scaled_pca_8c,Y,test_size=0.20,random_state=1)
print("shape of rawdata_X_train",rawdata_X_train.shape)
print("shape of rawdata_y_train",rawdata_y_train.shape)
print("shape of rawdata_X_test",rawdata_X_test.shape)
print("shape of rawdata_y_test",rawdata_y_test.shape)
print("--------------------------------------------")
print("shape of pca_X_train",pca_X_train.shape)
print("shape of pca_y_train",pca_y_train.shape)
print("shape of pca_X_test",pca_X_test.shape)
print("shape of pca_y_test",pca_y_test.shape)
from sklearn.model_selection import KFold, cross_val_score


kfold = KFold(n_splits= 10, random_state = 1)

svc = SVC() #instantiate the object

#now we will train the model with raw data

results = cross_val_score(estimator = svc, X = rawdata_X_train, y = rawdata_y_train, cv = kfold)

print(results,"\n")

print("Accuracy: %0.2f (+/- %0.2f)" % (results.mean()*100, results.std()*100 * 2))

sns.boxplot(results)
plt.show()
svc.fit(rawdata_X_train,rawdata_y_train)

print("Raw Data Training Accuracy :\t ", svc.score(rawdata_X_train, rawdata_y_train))

raw_train_accuracy=svc.score(rawdata_X_train, rawdata_y_train)

#Scoring the model on test_data
print("Raw Data Testing Accuracy :\t  ",  svc.score(rawdata_X_test, rawdata_y_test))

raw_test_accuracy=svc.score(rawdata_X_test, rawdata_y_test)

y_pred = svc.predict(rawdata_X_test)
print(classification_report(rawdata_y_test, svc.predict(rawdata_X_test)))
#now fit the model on pca data with new dimension

from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits= 10, random_state = 1)

svc = SVC() #instantiate the object

#now train the model with pca data with new dimension

pca_results = cross_val_score(estimator = svc, X = pca_X_train, y = pca_y_train, cv = kfold)

print(pca_results,"\n")

print("Accuracy: %0.2f (+/- %0.2f)" % (pca_results.mean()*100, pca_results.std()*100 * 2))

sns.boxplot(pca_results)
plt.show()
svc.fit(pca_X_train,pca_y_train)

print("PCA data Training Accuracy :\t ", svc.score(pca_X_train, pca_y_train))

pca_train_accuracy=svc.score(pca_X_train, pca_y_train)

#Scoring the model on test_data
print("PCA data Testing Accuracy :\t  ",  svc.score(pca_X_test, pca_y_test))

pca_test_accuracy=svc.score(pca_X_test, pca_y_test)

print(classification_report(pca_y_test, svc.predict(pca_X_test)))
#display confusion matrix of both models
print("Confusion matrix with raw data(18 dimension)\n",confusion_matrix(rawdata_y_test,rawdata_y_predict))
print("Confusion matrix with pca data(8 dimension)\n",confusion_matrix(pca_y_test,pca_y_predict))
result = pd.DataFrame({'TrainTest' : ['raw_train_accuracy', 'raw_test_accuracy', 'pca_train_accuracy','pca_test_accuracy'], 
                       'Accuracy' : [raw_train_accuracy,raw_test_accuracy, pca_train_accuracy, pca_test_accuracy],
                      })
result
#drop the columns
X_scaled.drop(['max.length_rectangularity','scaled_radius_of_gyration','skewness_about.2','scatter_ratio','elongatedness','pr.axis_rectangularity','scaled_variance','scaled_variance.1'],axis=1,inplace=True)
#display the shape of new dataframe
X_scaled.shape
dropcolumn_X_train,dropcolumn_X_test,dropcolumn_y_train,dropcolumn_y_test = train_test_split(X_scaled,Y,test_size=0.20,random_state=1)
print("shape of dropcolumn_X_train",dropcolumn_X_train.shape)
print("shape of dropcolumn_y_train",dropcolumn_y_train.shape)
print("shape of dropcolumn_X_test",dropcolumn_X_test.shape)
print("shape of dropcolumn_y_test",dropcolumn_y_test.shape)
#fit the model on dropcolumn_X_train,dropcolumn_y_train
svc.fit(dropcolumn_X_train,dropcolumn_y_train)
#predict the y value
dropcolumn_y_predict = svc.predict(dropcolumn_X_test)
#display the accuracy score and confusion matrix
print("Accuracy score with dropcolumn data(10 dimension)",accuracy_score(dropcolumn_y_test,dropcolumn_y_predict))
print("Confusion matrix with dropcolumn data(10 dimension)\n",confusion_matrix(dropcolumn_y_test,dropcolumn_y_predict))
