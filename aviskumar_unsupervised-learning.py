import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.preprocessing import Imputer



from sklearn.model_selection import train_test_split

from sklearn import preprocessing



from sklearn import metrics

from sklearn.metrics import roc_curve,roc_auc_score



from sklearn import svm



from sklearn import model_selection



from sklearn.model_selection import RandomizedSearchCV

from pprint import pprint
import os

os.listdir('../input')
data=pd.read_csv('../input/vehicle2/vehicle-2.csv')

data.head()
data.info()
#We see that many columns has null values in it. So we need to deal with them.
data.describe()
#Replacing null values in all columns with mean/median
data['circularity'].isnull().sum()



data['circularity'].fillna(data['circularity'].mean(),inplace=True)
data['distance_circularity'].isnull().sum()



data['distance_circularity'].fillna(data['distance_circularity'].mean(),inplace=True)
data['radius_ratio'].isnull().sum()



data['radius_ratio'].fillna(data['radius_ratio'].mean(),inplace=True)
data['pr.axis_aspect_ratio'].isnull().sum()



data['pr.axis_aspect_ratio'].fillna(data['pr.axis_aspect_ratio'].mean(),inplace=True)
data['scatter_ratio'].isnull().sum()



data['scatter_ratio'].fillna(data['scatter_ratio'].mean(),inplace=True)
data['elongatedness'].isnull().sum()



data['elongatedness'].fillna(data['elongatedness'].mean(),inplace=True)
data['pr.axis_rectangularity'].isnull().sum()



data['pr.axis_rectangularity'].fillna(data['pr.axis_rectangularity'].mean(),inplace=True)
data['scaled_variance'].isnull().sum()



data['scaled_variance'].fillna(data['scaled_variance'].mean(),inplace=True)
data['scaled_variance.1'].isnull().sum()



data['scaled_variance.1'].fillna(data['scaled_variance.1'].median(),inplace=True)
data['scaled_radius_of_gyration'].isnull().sum()



data['scaled_radius_of_gyration'].fillna(data['scaled_radius_of_gyration'].mean(),inplace=True)
data['scaled_radius_of_gyration.1'].isnull().sum()



data['scaled_radius_of_gyration.1'].fillna(data['scaled_radius_of_gyration.1'].mean(),inplace=True)
data['skewness_about'].isnull().sum()



data['skewness_about'].fillna(data['skewness_about'].mean(),inplace=True)
data['skewness_about.1'].isnull().sum()



data['skewness_about.1'].fillna(data['skewness_about.1'].mean(),inplace=True)
data['skewness_about.2'].isnull().sum()



data['skewness_about.2'].fillna(data['skewness_about.2'].mean(),inplace=True)
data.isnull().sum()
plt.figure(figsize=(30,40))

plt.subplot(6,3,1)

sns.boxplot(data['compactness'])

plt.subplot(6,3,2)

sns.boxplot(data['circularity'])

plt.subplot(6,3,3)

sns.boxplot(data['distance_circularity'])

plt.subplot(6,3,4)

sns.boxplot(data['pr.axis_aspect_ratio'])

plt.subplot(6,3,5)

sns.boxplot(data['max.length_aspect_ratio'])

plt.subplot(6,3,6)

sns.boxplot(data['scatter_ratio'])

plt.subplot(6,3,7)

sns.boxplot(data['elongatedness'])

plt.subplot(6,3,8)

sns.boxplot(data['pr.axis_rectangularity'])

plt.subplot(6,3,9)

sns.boxplot(data['max.length_rectangularity'])

plt.subplot(6,3,10)

sns.boxplot(data['scaled_variance'])

plt.subplot(6,3,11)

sns.boxplot(data['scaled_radius_of_gyration'])

plt.subplot(6,3,12)

sns.boxplot(data['scaled_variance.1'])

plt.subplot(6,3,13)

sns.boxplot(data['scaled_radius_of_gyration.1'])

plt.subplot(6,3,14)

sns.boxplot(data['skewness_about'])

plt.subplot(6,3,15)

sns.boxplot(data['skewness_about.1'])

plt.subplot(6,3,16)

sns.boxplot(data['skewness_about.2'])

plt.subplot(6,3,17)

sns.boxplot(data['hollows_ratio'])
data.boxplot(figsize=(35,20))
#We see that the columns pr.axis_aspect_ratio,max.length_aspect_ratio,scaled_radius_of_gyration.1,skewness_about,radius_ratio are largely affected by the outliers.
pd.crosstab(data['pr.axis_aspect_ratio'],data['class'])
#We see that from 76 it follows same pattern,



data['pr.axis_aspect_ratio']=np.where(data['pr.axis_aspect_ratio']>76,76,data['pr.axis_aspect_ratio'])
pd.crosstab(data['max.length_aspect_ratio'],data['class'])
#We see that from 19 it follows same pattern,



data['max.length_aspect_ratio']=np.where(data['max.length_aspect_ratio']>19,19,data['max.length_aspect_ratio'])
pd.crosstab(data['scaled_radius_of_gyration.1'],data['class'])
#We see that from 89 it follows same pattern,



data['scaled_radius_of_gyration.1']=np.where(data['scaled_radius_of_gyration.1']>89,89,data['scaled_radius_of_gyration.1'])
pd.crosstab(data['skewness_about'],data['class'])
#We see that from 17 it follows same pattern,



data['skewness_about']=np.where(data['skewness_about']>17,17,data['skewness_about'])
pd.crosstab(data['radius_ratio'],data['class'])
#We see that from 235 it follows same pattern,



data['radius_ratio']=np.where(data['radius_ratio']>235,235,data['radius_ratio'])
#Now we have done some tings to overcome the outliers, Lets see the box plot now,



data.boxplot(figsize=(40,15))





#The outliers has been handled to some extend.
sns.pairplot(data,diag_kind='kde')
corr=data.corr()

plt.figure(figsize=(20,10))

plt.subplot(1,1,1)

sns.heatmap(corr,annot=True)
#Splitting of Independent and Dependent variables



y=data['class']

X=data.drop(columns='class')
#Standardization of Data



def standardization(X_train,X_test):

    scaler=preprocessing.StandardScaler()

    X_train=scaler.fit_transform(X_train)

    X_test=scaler.transform(X_test)

    return X_train,X_test
#SVM



def svm_fun(X,y):

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)

    X_train,X_test=standardization(X_train,X_test)

    

    clf = svm.SVC(gamma=0.025,C=3)

    #when C increases Marigin shrinks

    # gamma is a measure of influence of a data point. It is inverse of distance of influence. C is complexity of the model

    # lower C value creates simple hyper surface while higher C creates complex surface



    clf.fit(X_train,y_train)

    svm_pred=clf.predict(X_test)

    svm_score=clf.score(X_test,y_test)

    print("The KNN model prediction is " + str(svm_score*100) + "%")

    

    print("The confusion matrix is ")

    print(metrics.confusion_matrix(y_test,svm_pred))

    print("the Classification report is")

    print(metrics.classification_report(y_test,svm_pred))

    #roc=roc_auc_score(y_test, svm_pred)

    #print("ROC value for svm model is "+ str(roc*100) + "%")
#SVM



svm_fun(X,y)
#Splitting of Independent and Dependent variables



y_pcm=data['class']

X_pcm=data.drop(columns='class')
# We transform (centralize) the entire X (independent variable data) to zscores through transformation. We will create the PCA dimensions

# on this distribution. 



#Covariance is done only on Independent variables

sc = preprocessing.StandardScaler()

X_std =  sc.fit_transform(X_pcm)          

cov_matrix = np.cov(X_std.T)

print('Covariance Matrix \n%s', cov_matrix)
#The dimensions are rotated



eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print('Eigen Vectors \n%s', eigenvectors)

print('\n Eigen Values \n%s', eigenvalues)
# Step 3 (continued): Sort eigenvalues in descending order



# Make a set of (eigenvalue, eigenvector) pairs

eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]



# Sort the (eigenvalue, eigenvector) pairs from highest to lowest with respect to eigenvalue

eig_pairs.sort()



#Desc sort

eig_pairs.reverse()

print(eig_pairs)



# Extract the descending ordered eigenvalues and eigenvectors

eigvalues_sorted = [eig_pairs[index][0] for index in range(len(eigenvalues))]

eigvectors_sorted = [eig_pairs[index][1] for index in range(len(eigenvalues))]



# Let's confirm our sorting worked, print out eigenvalues

print('Eigenvalues in descending order: \n%s' %eigvalues_sorted)
tot = sum(eigenvalues)

var_explained = [(i / tot) for i in sorted(eigenvalues, reverse=True)]  # an array of variance explained by each 

# eigen vector... there will be 8 entries as there are 8 eigen vectors)

cum_var_exp = np.cumsum(var_explained)  # an array of cumulative variance. There will be 8 entries with 8 th entry 

# cumulative reaching almost 100%
#1,19 depends on covariance matrix - count



plt.figure(figsize=(20,10))

plt.bar(range(1,19), var_explained, alpha=0.5, align='center', label='individual explained variance')

plt.step(range(1,19),cum_var_exp, where= 'mid', label='cumulative explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc = 'best')

plt.show()
# P_reduce represents reduced mathematical space....



#From the above graph we see that 10 to 16 would be good.So lets try with that

#After trying with the above range we see that 13 would be perfect,



P_reduce = np.array(eigvectors_sorted[0:13])   # Reducing from 18 to 13 dimension space



X_std_13D = np.dot(X_std,P_reduce.T)   # projecting original data into principal component dimensions



Proj_data_df = pd.DataFrame(X_std_13D)  # converting array to dataframe for pairplot
#Let us check it visually



sns.pairplot(Proj_data_df, diag_kind='kde') 
#Calling SVM function using PCM data



svm_fun(Proj_data_df, y_pcm)
  #RandomizedSearchCV - SVM

#Implement Hyperparameter



def hyper_params_svm(X,y):

#gamma is a parameter for non linear hyperplanes. The higher the gamma value it tries to exactly fit the training data set

    gammas = [0.1, 1, 10, 100]

#kernel parameters selects the type of hyperplane used to separate the data. Using ‘linear’ will use a linear hyperplane 

        #(a line in the case of 2D data). ‘rbf’ and ‘poly’ uses a non linear hyper-plane

    kernels  = ['linear', 'rbf', 'poly']

#C is the penalty parameter of the error term. It controls the trade off between smooth decision boundary and classifying the training points correctly.

    cs = [0.1, 1, 10, 100, 1000]

#degree is a parameter used when kernel is set to ‘poly’. It’s basically the degree of the polynomial used to find the hyperplane to split the data.

    degrees = [0, 1, 2, 3, 4, 5, 6]



# Create the random grid

    random_grid = {'gamma': gammas,

                   'kernel': kernels,

                   'C': cs,

                   'degree': degrees}



    pprint(random_grid)

    return random_grid



def randomizedsearch_svm(X,y):

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)

    X_train,X_test=standardization(X_train,X_test)

# Use the random grid to search for best hyperparameters

# First create the base model to tune

    svm_obj = svm.SVC(random_state=1)

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

    rf_random = RandomizedSearchCV(estimator = svm_obj, param_distributions = hyper_params_svm(X,y), n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    rf_random.fit(X_train, y_train)

    print("Best Hyper Parameters:",rf_random.best_params_)

    

    pred=rf_random.predict(X_test)

    score=rf_random.score(X_test,y_test)

    print("The model prediction is " + str(score*100) + "%")

    print("The confusion matrix is ")

    print(metrics.confusion_matrix(y_test, pred))

    print("the Classification report is")

    print(metrics.classification_report(y_test, pred))
#Calling Randmized search for SVM



randomizedsearch_svm(Proj_data_df, y_pcm)