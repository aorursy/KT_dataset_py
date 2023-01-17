import numpy as np   
import pandas as pd    
import matplotlib.pyplot as plt 
%matplotlib inline 
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import time
data = pd.read_csv('vehicle.csv')
data.head()
data.shape
data.info()
# Check number of not a number values
data.isna().sum()
data[data['radius_ratio'].isna() == True]
data['class'].value_counts()
# Creating a copy of dataframe to have class variable with us before operating on NaN values
import copy
data_copy = copy.deepcopy(data)
# Convert class variable to values
from sklearn.preprocessing import LabelEncoder
data_copy["class"] = LabelEncoder().fit_transform(data_copy["class"])
data_copy.head()
data_copy.describe().transpose()
# Observation: as outliers present, replace nan values with median
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer = imputer.fit(data_copy)
data_copy = pd.DataFrame(np.array(imputer.transform(data_copy)),columns=data_copy.columns)
data_copy
# Understanding the attributes - Find relationship between different
# attributes (Independent variables) and choose carefully which all
# attributes have to be a part of the analysis and why
# 2. Relationship between var
data_copy.corr()
correlation_matrix = data_copy.corr()
fig,ax = plt.subplots(figsize=(20,20)) 
sns.heatmap(correlation_matrix, annot=True, ax=ax)  
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns) 
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns) 
# For multicollinearity lets remove columns having corr of 95% pr.axis_rectangularity,scaled_variance,scaled_variance.1
# 99 % corelation between scatter_ratio and scaled_variance
# There are Attributes which are directly not related to class but can not remove these columns directly as they can be useful while doing PCA
data_copy.drop(['pr.axis_rectangularity','scaled_variance','scaled_variance.1'],axis=1,inplace=True)
data_copy.shape
# draw pair plot which wil lhelp to get idea of covariance matrix 
sns.pairplot(data_copy,diag_kind='kde',size=4)
X=data_copy.drop('class',axis=1)
Y=data_copy['class']
# Lets scale our data
X_Scaled=X.apply(zscore)
X_Scaled.head()
def get_SVM_Accuracy(X):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    model = SVC()
    start_time = time.time() 
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    return model.score(X_train,y_train)*100,model.score(X_test,y_test)*100,elapsed_time
Before_PCA_train_acc,Before_PCA_test_acc,time_taken = get_SVM_Accuracy(X_Scaled)
print(Before_PCA_train_acc)
print(Before_PCA_test_acc)
print(time_taken)
data_copy.shape
#Apply k fold cross validation
#using default genral practice 10 kfolds
# Common k fold function
def get_kFold_Results(X):
    Y = data_copy['class']
    num_folds = 10
    seed = 7
    kfold = KFold(n_splits=num_folds, random_state=seed)
    model = SVC()
    results = cross_val_score(model, X, Y, cv=kfold)
    print(results)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100.0, results.std()*100.0))
    return results
data_copy.head()
beforePCACV_results = get_kFold_Results(X_Scaled)
#Apply PCA
covMatrix = np.cov(X_Scaled,rowvar=False)
print(covMatrix)
pca = PCA()
pca.fit(X_Scaled)
# eigen value
print(pca.explained_variance_)
# eigen ratio
print(pca.explained_variance_ratio_)
plt.bar(list(range(1,16)),pca.explained_variance_ratio_,alpha=0.5, align='center')
plt.ylabel('Variation explained')
plt.xlabel('eigen Value')
plt.show()
plt.step(list(range(1,16)),np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Cum of variation explained')
plt.xlabel('eigen Value')
plt.show()
# Ploting 
plt.figure(figsize=(10 , 5))
plt.bar(range(1, pca.explained_variance_ratio_.size + 1), pca.explained_variance_ratio_, alpha = 0.5, align = 'center', label = 'Individual explained variance')
plt.step(range(1, pca.explained_variance_ratio_.size + 1), np.cumsum(pca.explained_variance_ratio_), where='mid', label = 'Cumulative explained variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()
pca7 = PCA(n_components=7)
pca7.fit(X_Scaled)
X_Scaled_PCA = pca7.transform(X_Scaled)
X_Scaled_PCA
After_PCA_train_acc,After_PCA_test_acc,time_taken = get_SVM_Accuracy(X_Scaled_PCA)
print(After_PCA_train_acc)
print(After_PCA_test_acc)
print(time_taken)
Before_PCA_train_acc,Before_PCA_test_acc,time_taken = get_SVM_Accuracy(X_Scaled)
print(Before_PCA_train_acc)
print(Before_PCA_test_acc)
print(time_taken)
After_PCA_train_acc,After_PCA_test_acc,time_taken = get_SVM_Accuracy(X_Scaled_PCA)
print(After_PCA_train_acc)
print(After_PCA_test_acc)
print(time_taken)
beforePCACV_results = get_kFold_Results(X_Scaled)
afterPCACV_results = get_kFold_Results(X_Scaled_PCA)
# Summary:
# After PCA we can see drop in accuracy as we loose information(from attributes 6 to 7 Principal components)
# Although there is no significant gain in
# Computation time as well 
# before PCA time taken is 0.0267 
# After PCA tie taken in fitting the SVC model 0.0209
# with confidance interval of 95 % we can say that with using PCA acc 91.48% with std deviation 2.44% will range 
# from 86.6 to  96.36
