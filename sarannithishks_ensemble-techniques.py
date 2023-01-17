# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/parkinsons-data-set/parkinsons.data")
df.head()
df.info()
df.shape
df.isna().sum()
print(df.describe().T.shape)
df.describe().T
df.drop('name',axis=1,inplace=True)
# sns.pairplot(df)

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


plotDistribution(df,4,6,sns.distplot)

plotDistribution(df,4,6,sns.boxplot)

corr = df.corr()
corr

fig, axs = plt.subplots(figsize=(20,20))         # Sample figsize in inches
sns.heatmap(corr, annot=True, linewidths=.8, ax=axs)

    
df.columns
df_dropped = df.drop(['MDVP:Jitter(Abs)',
 'MDVP:RAP',
 'MDVP:PPQ',
 'Jitter:DDP',
'MDVP:Shimmer(dB)',
 'Shimmer:APQ3',
 'Shimmer:APQ5',

 'MDVP:APQ',
 'Shimmer:DDA',
'spread1','NHR'],axis=1)
df_dropped.columns
fig, axs = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(df_dropped.corr(), annot=True, linewidths=.8, ax=axs)
df_dropped['status'].value_counts()
df_dropped_min = df_dropped[df_dropped["status"] == 0] 
df_dropped_maj = df_dropped[df_dropped["status"] == 1]
df_dropped_min_upsampled = skl.utils.resample(df_dropped_min,n_samples=147,random_state=1);

df_dropped_upsampled = pd.concat([df_dropped_maj,df_dropped_min_upsampled])

df_dropped_upsampled["status"].value_counts()
# Splitting the data into independent and dependent variables

y = df_dropped["status"]
X = df_dropped.drop(["status"],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)

print(y_test.shape)
y_train.value_counts()
def getOulierCount(df):
    
    for columnName in df.columns:
        if(columnName == 'status'):
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
            print("Outlier Percentage:",outlierPercentage);
            
           
        except Exception as ex:
           
            print(ex)
getOulierCount(X_train)
def ReplaceOutliersWithMedian(df):
    
    for columnName in df.columns:
        if(columnName == 'status'):
         continue
        try:
            featureValues = df[columnName]
            q1 = featureValues.quantile(0.25)
            q3 = featureValues.quantile(0.75)
            iqr = q3-q1
            upperlimit = q3+(1.5*iqr)
            lowerlimit = q1-(1.5*iqr)
            #print("ColumnName:",columnName)
            median = featureValues.median()
            featureValues.loc[(featureValues < lowerlimit) | (featureValues > upperlimit)] = median
            #featureValues.fillna(median,inplace=True)
            
            outlierCount = featureValues.loc[(featureValues < lowerlimit) | (featureValues > upperlimit)].count()
            outlierPercentage = (outlierCount/featureValues.count())*100
            #print("Outlier Percentage:",outlierPercentage);
            
           
        except Exception as ex:
           
            print(ex)
    return df


plotDistribution(X_train,2,6,sns.boxplot)
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
     #print("Accuracy score on test data:",accuracy)
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
     

    
dataTypes = pd.Series(data=['Scaled_Treated','Scaled_NotTreated','NotScaled_Treated','NotScaled_NotTreated'])

accuracy_df = pd.DataFrame(columns=['dataType','Logistic','KNN','NB','SVM','Stacking','Bagging','AdaBoost','GradBoost','RandomForest'])
                           
                           
                           
                           
                           
                           
accuracy_df['dataType'] = dataTypes


def makeSeries(algorithm,hyperparameter,X_train,y_train,X_test,y_test):
        acc = []
        for dtype in dataTypes:
            model,accuracy = switchData(dtype,algorithm,hyperparameter,X_train,y_train,X_test,y_test)
            acc.append(accuracy)
        return pd.Series(data=acc)
            
 
  
      
algorithm = logisticRegressor(max_iter=10000,random_state=1);

hyperparameter = {'solver' : ['newton-cg', 'lbfgs','liblinear', 'sag', 'saga']}

#get_accuracy(algorithm,X_train,y_train)
accuracy_df['Logistic'] = makeSeries(algorithm,hyperparameter, X_train,y_train,X_test,y_test)



algorithm = knnClassifier();

hyperparameter = {'algorithm' : ['ball_tree', 'kd_tree', 'brute'],
                   'n_neighbors': np.arange(5 , 20 , 2),
                   'metric': ['euclidean','manhattan']}

#get_accuracy(algorithm,X_train,y_train)

#knnModel ,y_pred_knn = fit_and_predict(algorithm,hyperparameter,X_train,y_train,X_test,y_test,scale=False)
#knnModel_scaled ,y_pred_knn_scaled = fit_and_predict(algorithm,hyperparameter,X_train,y_train,X_test,y_test,scale=True)

accuracy_df['KNN'] = makeSeries(algorithm,hyperparameter, X_train,y_train,X_test,y_test)

algorithm =GaussianNB() 
# NBModel,y_pred_NB = fit_and_predict(GaussianNB(),None,X_train,y_train,X_test,y_test,scale=False)
# NBModel_scaled,y_pred_NB_scaled = fit_and_predict(GaussianNB(),None,X_train,y_train,X_test,y_test,scale=True)

accuracy_df['NB'] = makeSeries(algorithm,None, X_train,y_train,X_test,y_test)

algorithm = SVC();

hyperparameter = {'C': [0.01,0.1,1,10,100],
                  'gamma':[0.01,0.1,1,10,100]
                 }
                   

#get_accuracy(algorithm,X_train,y_train)

#SVCModel ,y_pred_svc = fit_and_predict(algorithm,hyperparameter,X_train,y_train,X_test,y_test,scale=False)
#SVCModel_scaled ,y_pred_svc_scaled= fit_and_predict(algorithm,hyperparameter,X_train,y_train,X_test,y_test,scale=True)

accuracy_df['SVM'] = makeSeries(algorithm,hyperparameter, X_train,y_train,X_test,y_test)





estimators = [
    ('lr', logisticRegressor(max_iter=10000, random_state=1, solver='liblinear')),
    ('knn',knnClassifier(metric='euclidean', n_neighbors=9)),
    ('nb',SVC(C=100, gamma=0.01))
 ]

final_estimator = logisticRegressor()

algorithm = StackingClassifier(estimators,final_estimator);



#StackingModel ,y_pred_svc = fit_and_predict(algorithm,hyperparameter,X_train,y_train,X_test,y_test,scale=True)

accuracy_df['Stacking'] = makeSeries(algorithm,None, X_train,y_train,X_test,y_test)
#print(algorithm.fit(X_train,y_train).score(X_test,y_test))


algorithm = BaggingClassifier(random_state=1);

#get_accuracy(algorithm,X_train,y_train)

#baggingModel ,y_pred_bag = fit_and_predict(algorithm,None,X_train,y_train,X_test,y_test,scale=True)
#SVCModel_scaled ,y_pred_svc_scaled= fit_and_predict(algorithm,hyperparameter,X_train,y_train,X_test,y_test,scale=True)
accuracy_df['Bagging'] = makeSeries(algorithm,None, X_train,y_train,X_test,y_test)

algorithm = RandomForestClassifier(random_state=1);

#get_accuracy(algorithm,X_train,y_train)

#rForest ,y_pred_bag = fit_and_predict(algorithm,None,X_train,y_train,X_test,y_test,scale=True)
#SVCModel_scaled ,y_pred_svc_scaled= fit_and_predict(algorithm,hyperparameter,X_train,y_train,X_test,y_test,scale=True)
accuracy_df['RandomForest'] = makeSeries(algorithm,None, X_train,y_train,X_test,y_test)
algorithm = AdaBoostClassifier(random_state=1);

#get_accuracy(algorithm,X_train,y_train)

#adaBoost ,y_pred_bag = fit_and_predict(algorithm,None,X_train,y_train,X_test,y_test,scale=True)
#SVCModel_scaled ,y_pred_svc_scaled= fit_and_predict(algorithm,hyperparameter,X_train,y_train,X_test,y_test,scale=True)
accuracy_df['AdaBoost'] = makeSeries(algorithm,None, X_train,y_train,X_test,y_test)
algorithm = GradientBoostingClassifier(random_state=1);

#get_accuracy(algorithm,X_train,y_train)

#gradBoost ,y_pred_bag = fit_and_predict(algorithm,None,X_train,y_train,X_test,y_test,scale=True)
#SVCModel_scaled ,y_pred_svc_scaled= fit_and_predict(algorithm,hyperparameter,X_train,y_train,X_test,y_test,scale=True)
accuracy_df['GradBoost'] = makeSeries(algorithm,None, X_train,y_train,X_test,y_test)
accuracy_df
accuracy_df.max()
#sns.pointplot(x=accuracy_df["dataType"],y=accuracy_df["Logistic"])

fig, axs = plt.subplots(1, 1,figsize=(15, 8))

for column in accuracy_df.columns.drop("dataType"):
    axs.plot(accuracy_df["dataType"],accuracy_df[column],label = column )


plt.title("Accuracy Comparison", fontsize = 20)
plt.xlabel("Date Treatment Parameter", fontsize = 15)
plt.ylabel("Accuracy Percentage", fontsize = 15)

axs.legend(loc='upper right',bbox_to_anchor=(1.15,1))
plt.show()


