import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 

df = pd.read_csv("/kaggle/input/vehicle/vehicle-1.csv")
df.head(20)
df.info()
df.shape
df['class'] = df['class'].astype('category')

df.isna().sum()
df =  df.fillna(df.median())
df.isna().sum()

print(df.describe().T.shape)
df.describe().T
sns.pairplot(df)

def plotDistribution(df,rows,columns,plot):

    fig, axs = plt.subplots(rows, columns,figsize=(40, 20))

    ctr = 0
    for i in range(rows):
        for j in range(columns):
            try:
              #print(ctr)
              plot(df.iloc[:,ctr],ax=axs[i, j])
            except Exception as ex:
               print('Exception: ', ex)
               print("Column index not found:",ctr)
            ctr = ctr +1


plotDistribution(df,3,6,sns.distplot)

plotDistribution(df,3,6,sns.boxplot)

corr = df.corr()


fig, axs = plt.subplots(figsize=(20,20))         # Sample figsize in inches
sns.heatmap(corr, annot=True, linewidths=.8, ax=axs)
def findCorrelations(value,threshold):
    if((abs(value) > threshold) and value < 1) :
         return value
        
    
import math
def replaceNansAndNones(value):
    
    if(value == None or math.isnan(value)):
        return 0;
    else:
        return value
correlations = corr.applymap(lambda x: findCorrelations(x,0.85)).applymap(lambda x: replaceNansAndNones(x))
correlations

fig, axs = plt.subplots(figsize=(20,20))         # Sample figsize in inches
sns.heatmap(correlations, annot=True, linewidths=.8, ax=axs)

df.columns
df_dropped = df.drop(['scatter_ratio',
 'pr.axis_rectangularity',
 'scaled_variance',
 'scaled_variance.1',
'max.length_rectangularity',
 'scaled_radius_of_gyration',
 'skewness_about.2',
'elongatedness',
],axis=1)

fig, axs = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(df_dropped.corr(), annot=True, linewidths=.8, ax=axs)
df_dropped['class'].value_counts()
df_dropped['class']= LabelEncoder().fit_transform(df_dropped['class']) 
df['class']= LabelEncoder().fit_transform(df['class']) 

df_dropped['class'].value_counts()
# Splitting the data into independent and dependent variables

y = df_dropped["class"]
X = df_dropped.drop(["class"],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


train_df = X_train
train_df['class'] = y_train
train_df['class'].value_counts()
def balanceClasses(train_df):

    train_df_min_1 = train_df[train_df["class"] == 0] 
    train_df_min_2 = train_df[train_df["class"] == 2] 
    train_df_maj = train_df[train_df["class"] == 1]
    train_df_min_1_upsampled = skl.utils.resample(train_df_min_1,n_samples=296,random_state=1);
    train_df_min_2_upsampled = skl.utils.resample(train_df_min_2,n_samples=296,random_state=1);

    train_df_upsampled = pd.concat([train_df_maj,train_df_min_1_upsampled,train_df_min_2_upsampled])
    return train_df_upsampled
    
train_df_upsampled = balanceClasses(train_df)
y_train = train_df_upsampled["class"]
X_train = train_df_upsampled.drop(["class"],axis=1)
y_train.value_counts()
def getOulierPecentage(df):
    
    for columnName in df.columns:
        if(columnName == 'class'):
         continue
        try:
            featureValues = df[columnName]
            q1 = featureValues.quantile(0.25)
            q3 = featureValues.quantile(0.75)
            iqr = q3-q1
            upperlimit = q3+(1.5*iqr)
            lowerlimit = q1-(1.5*iqr)
            print("ColumnName:",columnName)
            
            outlierCount = featureValues.loc[(featureValues < lowerlimit) | (featureValues > upperlimit)].count()

            outlierPercentage = (outlierCount/featureValues.count())*100
            print("Outlier Percentage:",outlierPercentage, "%");
            
           
        except Exception as ex:
           
            print(ex)
getOulierPecentage(X_train)
def ReplaceOutliersWithMedian(df):
    
    for columnName in df.columns:
        if(columnName == 'class'):
         continue
        try:
            featureValues = df[columnName]
            q1 = featureValues.quantile(0.25)
            q3 = featureValues.quantile(0.75)
            iqr = q3-q1
            upperlimit = q3+(1.5*iqr)
            lowerlimit = q1-(1.5*iqr)
          
            median = featureValues.median()
            featureValues.loc[(featureValues < lowerlimit) | (featureValues > upperlimit)] = median
           
            
            outlierCount = featureValues.loc[(featureValues < lowerlimit) | (featureValues > upperlimit)].count()
            outlierPercentage = (outlierCount/featureValues.count())*100
          
            
           
        except Exception as ex:
           
            print(ex)
    return df
from sklearn.linear_model import LogisticRegression as logisticRegressor
import sklearn.metrics as skmetrics
import sklearn.metrics as skmetrics
from  sklearn.neighbors import KNeighborsClassifier as knnClassifier
from  sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
def fit_and_predict(algorithm,hyperparameter, X_train,y_train,X_test,y_test,scale=True):

     if(scale):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.fit_transform(X_test)
  
     if (hyperparameter != None):
      model = GridSearchCV(algorithm,hyperparameter).fit(X_train,y_train)
     else:
      model = algorithm.fit(X_train,y_train)
     y_pred = model.predict(X_test)
     test_score = skmetrics.accuracy_score(y_test,y_pred)
        
     
     accuracy = test_score*100

     return model,accuracy

def switchData(dataType,algorithm,hyperparameter,X_train,y_train,X_test,y_test):

    if(dataType == 'Scaled_Treated'):
       return fit_and_predict(algorithm,hyperparameter,ReplaceOutliersWithMedian(X_train),y_train,X_test,y_test,scale=True)
        
    elif (dataType == 'Scaled_NotTreated'):
       return fit_and_predict(algorithm,hyperparameter,X_train,y_train,X_test,y_test,scale=True)
        
    elif (dataType == 'NotScaled_Treated'):
       return fit_and_predict(algorithm,hyperparameter,ReplaceOutliersWithMedian(X_train),y_train,X_test,y_test,scale=False)
        
    elif (dataType == 'NotScaled_NotTreated'):
       return fit_and_predict(algorithm,hyperparameter,X_train,y_train,X_test,y_test,scale=False)
     

    


accuracy_df = pd.DataFrame()

dataTypes = pd.Series(data=['Scaled_Treated'])


def makeSeries(algorithm,hyperparameter,X_train,y_train,X_test,y_test):
        acc = []
        for dtype in dataTypes:
            model,accuracy = switchData(dtype,algorithm,hyperparameter,X_train,y_train,X_test,y_test)
            acc.append(accuracy)
        return pd.Series(data=acc)
            
 
  
      
algorithm = SVC();

hyperparameter = {'C': [0.01,0.1,1,10,100],
                  'gamma':[0.01,0.1,1,10,100]
                 }
                   

accuracy_df['Feature_Elimination'] = makeSeries(algorithm,hyperparameter, X_train,y_train,X_test,y_test)

print("Accuracy value on the test data is",accuracy_df['Feature_Elimination'][0])

df.shape


y = df["class"]
X = df.drop(["class"],axis=1)

from sklearn.decomposition import PCA

scaler = StandardScaler()
X_Scaled_for_PCA = scaler.fit_transform(X)




pca = PCA()
pca.fit(X_Scaled_for_PCA)

explained_var = pca.explained_variance_ratio_
cum_explained_var = pca.explained_variance_ratio_.cumsum();
print(cum_explained_var)


plt.figure(figsize=(20, 10))

plt.bar(range(18),explained_var, align='center',
        label='Individual Explained Variance')
plt.step(range(18), cum_explained_var,
         label='Cumulative Explained Variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Number of Principal components')
plt.legend(loc='best')

pca_final = PCA(n_components=7)
X_PCA= pd.DataFrame(data=pca_final.fit_transform(X_Scaled_for_PCA))

X_PCA

X_train, X_test, y_train, y_test = train_test_split(X_PCA, y, test_size=0.3, random_state=5)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)

print(y_test.shape)

X_train.columns
train_df = X_train

train_df['class'] = y_train
train_df['class'].value_counts()
train_df_upsampled = balanceClasses(train_df)
y_train = train_df_upsampled["class"]
X_train = train_df_upsampled.drop(["class"],axis=1)

y_train.value_counts()


X_train.shape
algorithm = SVC();

hyperparameter = {'C': [0.01,0.1,1,10,100],
                  'gamma':[0.01,0.1,1,10,100]
                 }
   
accuracy_df['Feature_Extraction_PCA_7'] = makeSeries(algorithm,hyperparameter, X_train,y_train,X_test,y_test)

print("The accuracy score after PCA is ",accuracy_df['Feature_Extraction_PCA_7'][0])
accuracy_df.T

ax = accuracy_df.T.plot.bar()
ax.get_legend().remove()
ax.set_xlabel("Feature Engineering Approach")
ax.set_ylabel("Accuracy score in percentage")
pca_final = PCA(n_components=11)
X_PCA= pd.DataFrame(data=pca_final.fit_transform(X_Scaled_for_PCA))

X_PCA

X_train, X_test, y_train, y_test = train_test_split(X_PCA, y, test_size=0.3, random_state=5)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)

print(y_test.shape)

X_train.columns
train_df = X_train

train_df['class'] = y_train
train_df['class'].value_counts()
train_df_upsampled = balanceClasses(train_df)
y_train = train_df_upsampled["class"]
X_train = train_df_upsampled.drop(["class"],axis=1)

y_train.value_counts()


X_train.shape
algorithm = SVC();

hyperparameter = {'C': [0.01,0.1,1,10,100],
                  'gamma':[0.01,0.1,1,10,100]
                 }
   
accuracy_df['Feature_Extraction_PCA_11'] = makeSeries(algorithm,hyperparameter, X_train,y_train,X_test,y_test)

print("The accuracy score after PCA is ",accuracy_df['Feature_Extraction_PCA_11'][0])
ax = accuracy_df.T.plot.bar()
ax.get_legend().remove()
ax.set_xlabel("Feature Engineering Approach")
ax.set_ylabel("Accuracy score in percentage")