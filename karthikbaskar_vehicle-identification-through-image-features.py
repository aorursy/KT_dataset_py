import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt
df=pd.read_csv('../input/vehicle/vehicle.csv')
df.head()
df.info()
print("The Number of Rows in our dataset:{} & Number of columns:{}".format(df.shape[0],df.shape[1]))
df.isna().sum()
df=df.fillna(df.median())
df.describe().T
columns=list(df)

df[columns].hist(stacked=True,density=True, bins=100,color='Orange', figsize=(16,30), layout=(10,3)); 
grp=df.groupby('class')['class'].count()

grp.plot.pie(shadow=True, startangle=120,autopct='%.2f')
df_car=df[df['class']=='car']

df_van=df[df['class']=='van']

df_bus=df[df['class']=='bus']
plt.scatter(df_bus['circularity'],np.zeros_like(df_bus['circularity']),marker='s',color='Red',alpha=0.5)

plt.scatter(df_car['circularity'],np.zeros_like(df_car['circularity']),marker='|',color='blue',alpha=0.8)

plt.scatter(df_van['circularity'],np.zeros_like(df_van['circularity']),marker='o',color='yellow',alpha=1)

plt.xlabel('Circularity')

plt.show()

plt.scatter(df_bus['distance_circularity'],np.zeros_like(df_bus['distance_circularity']),marker='s',color='Red',)

plt.scatter(df_car['distance_circularity'],np.zeros_like(df_car['distance_circularity']),marker='|',color='blue',alpha=0.4)

plt.scatter(df_van['distance_circularity'],np.zeros_like(df_van['distance_circularity']),marker='o',color='yellow',alpha=0.4)

plt.xlabel('distance_circularity')

plt.show()

plt.scatter(df_bus['hollows_ratio'],np.zeros_like(df_bus['hollows_ratio']),marker='s',color='Red',)

plt.scatter(df_car['hollows_ratio'],np.zeros_like(df_car['hollows_ratio']),marker='|',color='blue',alpha=0.4)

plt.scatter(df_van['hollows_ratio'],np.zeros_like(df_van['hollows_ratio']),marker='d',color='green',alpha=0.4)

plt.xlabel('hollows_ratio')

plt.show()
plt.figure(figsize=(12,8))

sns.distplot(df_bus['elongatedness'],kde=True,color='r',hist=False,label="Bus")

sns.distplot(df_car['elongatedness'],kde=True,color='G',hist=False,label="Car")

sns.distplot(df_van['elongatedness'],kde=True,color='B',hist=False,label="Van")

plt.legend()

plt.title("elongatedness Distribution")
plt.figure(figsize=(12,8))

sns.distplot(df_bus['max.length_rectangularity'],kde=True,color='r',hist=False,label="Bus")

sns.distplot(df_car['max.length_rectangularity'],kde=True,color='G',hist=False,label="Car")

sns.distplot(df_van['max.length_rectangularity'],kde=True,color='B',hist=False,label="Van")

plt.legend()

plt.title("max.length_rectangularity Distribution")
plt.figure(figsize=(8,6))

sns.scatterplot(df['scaled_radius_of_gyration'],df['scaled_variance'],hue=df['class'],markers='+')
#Print the corelation between columns in tabular format 

#core=df.corr()

#print(core)

#Using Pair plot to visualize the corelation

sns.pairplot(df,hue='class');
plt.figure(figsize=(18,12))

sns.boxplot(data=df) 

plt.xticks(rotation=45)
df.skew()
plt.subplots(figsize=(15,15))

sns.heatmap(df.corr(),annot=True)
Cor_Matrix=df.corr().abs()

Cor_Matrix

upper_tri = Cor_Matrix.where(np.triu(np.ones(Cor_Matrix.shape),k=1).astype(np.bool))

#print(upper_tri)

to_drop =[column for column in upper_tri.columns if any(upper_tri[column] > 0.93)]



print("These columns can be dropped as they are redundant:",to_drop[0:6])
df1=df.drop(['elongatedness','pr.axis_rectangularity','max.length_rectangularity','scaled_variance','scaled_variance.1'],axis=1)

print("The Number of Rows in our dataset :{} & Number of columns after remving multicollinearity:{}".format(df1.shape[0],df1.shape[1]))

plt.figure(figsize=(10,12))

sns.boxplot(data=df1) 

plt.xticks(rotation=45)
#IQR Calculation 



Q1=df1.quantile(0.25)

Q3=df1.quantile(0.75)

IQR= Q3 - Q1



IQR
df1 = df1[~((df1 < (Q1 - 1.5 * IQR)) |(df1 > (Q3 + 1.5 * IQR))).any(axis=1)]

df1.shape
plt.figure(figsize=(10,12))

sns.boxplot(data=df1) 

plt.xticks(rotation=45)
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,RocCurveDisplay

from sklearn.preprocessing import StandardScaler



X_MNMX=df1.drop(['class'],axis=1)

Y_MNMX=df1['class']

#X_MNMX=X_MNMX.apply(zscore)

X_Train,X_Test,Y_Train,Y_Test=train_test_split(X_MNMX,Y_MNMX,test_size=0.3,random_state=23)

sc=StandardScaler()

X_Train=sc.fit_transform(X_Train)

X_Test=sc.transform(X_Test)





SVM1=SVC(C=1.0,kernel='rbf')

SVM1.fit(X_Train,Y_Train)

#print(SVM1.score(X_Train,Y_Train))

#print(SVM1.score(X_Test,Y_Test))

PRED_SVM_M1=SVM1.predict(X_Test)

CM_SVM_M1=confusion_matrix(Y_Test,PRED_SVM_M1)

#print(CM_SVM_M1)



print("#####################Classification Report & Accuracy SCore#####################")

print("----SVM Model ----")

print("Model Score on Training data with selected features :{}".format(SVM1.score(X_Train,Y_Train) * 100))

print("Model Score on Testing  data with selected features :{}".format(SVM1.score(X_Test,Y_Test) * 100))

print("Accuracy Score of SVM on Test Data:{}".format(accuracy_score(Y_Test,PRED_SVM_M1)*100))

print(classification_report(Y_Test,PRED_SVM_M1))

sns.heatmap(CM_SVM_M1,annot=True,xticklabels=True,yticklabels=True,fmt='g',linewidths=.5,cmap='tab20c_r')

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()

# Applying k-Fold Cross Validation 

from sklearn.model_selection import KFold,cross_val_score

Folds=10

seed=23

kfold=KFold(shuffle=True,n_splits=Folds,random_state=seed)

accuracies = cross_val_score(estimator = SVM1, X = X_MNMX, y = Y_MNMX, cv = kfold) 

accuracies

print("K Fold score mean:{}".format(accuracies.mean()*100))

print("K Fold score standard deviation:{}".format(accuracies.std()*100))
#Import PCA from sklearn

from sklearn.decomposition import PCA

from scipy.stats import zscore
# Dropping those columns which had high correlation that we found during EDA

df=df.drop(['elongatedness','pr.axis_rectangularity','max.length_rectangularity','scaled_variance','scaled_variance.1'],axis=1)
#Scale and split data 

X_PCA=df.drop(['class'],axis=1)

Y_PCA=df['class']

X_PCA=X_PCA.apply(zscore)



PC=PCA(n_components=10,random_state=23)

PC_DF=PC.fit_transform(X_PCA)

#X_Train_PCA=PC.transform(X_PCA_Train)
covMatrix = np.cov(X_PCA,rowvar=False)

plt.subplots(figsize=(7,7))

sns.heatmap(covMatrix,annot=True,cmap='afmhot_r')
print("##############Eigen Values##############")

print(PC.explained_variance_)

print("##############Eigen Vectors##############")

print(PC.components_)
plt.bar(list(range(1,11)),PC.explained_variance_ratio_, align='center')

plt.ylabel('Variation explained')

plt.xlabel('eigen Value')

plt.show()
plt.step(list(range(1,11)),np.cumsum(PC.explained_variance_ratio_), where='mid')

plt.ylabel('Cum of variation explained')

plt.xlabel('eigen Value')

plt.show()
print('Explained variation per principal component: {}'.format(PC.explained_variance_ratio_))

P_Components=PC.explained_variance_ratio_

print("The Ideal number of components that could explain:{}% of variance in data is 7".format(np.sum(P_Components[0:7])*100))
PCA_DF=pd.DataFrame(data=PC_DF,columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'])



PCA_DF['Y']=Y_PCA

PCA_DF
sns.pairplot(PCA_DF,diag_kind='kde');
X_PCA_DF=PCA_DF.drop(['Y'],axis=1)

Y_PCA_DF=PCA_DF['Y']
X_PCA_Train,X_PCA_Test,Y_PCA_Train,Y_PCA_Test=train_test_split(X_PCA_DF,Y_PCA_DF,test_size=0.3,random_state=23)
PCA_Final=PCA(n_components=7,random_state=23)
SVM3=SVC(C=1.0,kernel='rbf')
SVM3.fit(X_PCA_Train,Y_PCA_Train)
PRED_PCA=SVM3.predict(X_PCA_Test)
CM_SVM_PCA=confusion_matrix(Y_PCA_Test,PRED_PCA)
print("#####################Classification Report & Accuracy SCore of PCA Data  on SVM#####################")

print("----SVM Model ----")

print("Model Score on Training data with PCA features :{}".format(SVM3.score(X_PCA_Train,Y_PCA_Train) * 100))

print("Model Score on Testing  data with PCA features :{}".format(SVM3.score(X_PCA_Test,Y_PCA_Test) * 100))

print("Accuracy Score of SVM on Test Data:{}".format(accuracy_score(Y_PCA_Test,PRED_PCA)*100))

print(classification_report(Y_PCA_Test,PRED_PCA))

sns.heatmap(CM_SVM_PCA,annot=True,xticklabels=True,yticklabels=True,linewidths=.5,cmap='tab20c_r',fmt='g')

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()
# Applying k-Fold Cross Validation 

from sklearn.model_selection import cross_val_score,KFold

kfold_pca=KFold(shuffle=True,n_splits=10,random_state=23)

accuracies = cross_val_score(estimator = SVM3, X = X_PCA_DF, y = Y_PCA_DF, cv = kfold_pca) 

accuracies

print("K Fold score mean:{}".format(accuracies.mean()*100))

print("K Fold score standard deviation:{}".format(accuracies.std()*100))