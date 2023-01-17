#let us start by importing the relevant libraries



%matplotlib inline

import warnings

import seaborn as sns

warnings.filterwarnings('ignore')

#import the necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score,confusion_matrix, classification_report,roc_auc_score

from scipy.stats import zscore

from sklearn.model_selection import train_test_split

vehdf= pd.read_csv("../input/vehicle-2.csv")

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also

#print(vehdf)

vehdf.head(200)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder 

le = LabelEncoder() 

columns = vehdf.columns

#Let's Label Encode our class variable: 

print(columns)

vehdf['class'] = le.fit_transform(vehdf['class'])

vehdf.shape
vehdf.info()
from sklearn.impute import SimpleImputer



newdf = vehdf.copy()



X = newdf.iloc[:,0:19] #separting all numercial independent attribute

#y = vehdf.iloc[:,18] #seprarting class attribute. 

#imputer = SimpleImputer()

imputer = SimpleImputer(missing_values=np.nan, strategy='median', verbose=1)

#fill missing values with mean column values

transformed_values = imputer.fit_transform(X)

column = X.columns

print(column)

newdf = pd.DataFrame(transformed_values, columns = column )

newdf.describe()





print("Original null value count:", vehdf.isnull().sum())

print("\n\nCount after we impiuted the NaN value: ", newdf.isnull().sum())
newdf.describe().T
newdf.shape
plt.style.use('seaborn-whitegrid')



newdf.hist(bins=20, figsize=(60,40), color='lightblue', edgecolor = 'red')

plt.show()




#Let us use seaborn distplot to analyze the distribution of our columns and see the skewness in attributes

f, ax = plt.subplots(1, 6, figsize=(30,5))

vis1 = sns.distplot(newdf["scaled_variance.1"],bins=10, ax= ax[0])

vis2 = sns.distplot(newdf["scaled_variance"],bins=10, ax=ax[1])

vis3 = sns.distplot(newdf["skewness_about.1"],bins=10, ax= ax[2])

vis4 = sns.distplot(newdf["skewness_about"],bins=10, ax=ax[3])

vis6 = sns.distplot(newdf["scatter_ratio"],bins=10, ax=ax[5])



f.savefig('subplot.png')



  
skewValue = newdf.skew()

print("skewValue of dataframe attributes: ", skewValue)
#Summary View of all attribute , The we will look into all the boxplot individually to trace out outliers



ax = sns.boxplot(data=newdf, orient="h")

plt.figure(figsize= (20,15))

plt.subplot(3,3,1)

sns.boxplot(x= newdf['pr.axis_aspect_ratio'], color='orange')



plt.subplot(3,3,2)

sns.boxplot(x= newdf.skewness_about, color='purple')



plt.subplot(3,3,3)

sns.boxplot(x= newdf.scaled_variance, color='brown')



plt.show()



plt.figure(figsize= (20,15))

plt.subplot(3,3,1)

sns.boxplot(x= newdf['radius_ratio'], color='red')



plt.subplot(3,3,2)

sns.boxplot(x= newdf['scaled_radius_of_gyration.1'], color='lightblue')



plt.subplot(3,3,3)

sns.boxplot(x= newdf['scaled_variance.1'], color='yellow')



plt.show()
plt.figure(figsize= (20,15))

plt.subplot(3,3,1)

sns.boxplot(x= newdf['max.length_aspect_ratio'], color='green')



plt.subplot(3,3,2)

sns.boxplot(x= newdf['skewness_about.1'], color='grey')





plt.show()
newdf.shape

from scipy.stats import iqr



Q1 = newdf.quantile(0.25)

Q3 = newdf.quantile(0.75)

IQR = Q3 - Q1

print(IQR)





cleandf = newdf[~((newdf < (Q1 - 1.5 * IQR)) |(newdf > (Q3 + 1.5 * IQR))).any(axis=1)]

cleandf.shape
plt.figure(figsize= (20,15))

plt.subplot(8,8,1)

sns.boxplot(x= cleandf['pr.axis_aspect_ratio'], color='orange')



plt.subplot(8,8,2)

sns.boxplot(x= cleandf.skewness_about, color='purple')



plt.subplot(8,8,3)

sns.boxplot(x= cleandf.scaled_variance, color='brown')

plt.subplot(8,8,4)

sns.boxplot(x= cleandf['radius_ratio'], color='red')



plt.subplot(8,8,5)

sns.boxplot(x= cleandf['scaled_radius_of_gyration.1'], color='lightblue')



plt.subplot(8,8,6)

sns.boxplot(x= cleandf['scaled_variance.1'], color='yellow')



plt.subplot(8,8,7)

sns.boxplot(x= cleandf['max.length_aspect_ratio'], color='lightblue')



plt.subplot(8,8,8)

sns.boxplot(x= cleandf['skewness_about.1'], color='pink')



plt.show()

def correlation_heatmap(dataframe,l,w):

    #correlations = dataframe.corr()

    correlation = dataframe.corr()

    plt.figure(figsize=(l,w))

    sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')

    plt.title('Correlation between different fearures')

    plt.show();

    

# Let's Drop Class column and see the correlation Matrix & Pairplot Before using this dataframe for PCA as PCA should only be perfromed on independent attribute

cleandf= newdf.drop('class', axis=1)

#print("After Dropping: ", cleandf)

correlation_heatmap(cleandf, 30,15)
sns.pairplot(cleandf, diag_kind="kde")
#display how many are car,bus,van. 

newdf['class'].value_counts()



splitscaledf = newdf.copy()

sns.countplot(newdf['class'])

plt.show()
#now separate the dataframe into dependent and independent variables

#X1= newdf.drop('class',axis=1)

#y1 = newdf['class']

#print("shape of new_vehicle_df_independent_attr::",X.shape)

#print("shape of new_vehicle_df_dependent_attr::",y.shape)



X = newdf.iloc[:,0:18].values

y = newdf.iloc[:,18].values



X
from sklearn.preprocessing import StandardScaler

#We transform (centralize) the entire X (independent variable data) to normalize it using standardscalar through transformation. We will create the PCA dimensions

# on this distribution. 

sc = StandardScaler()

X_std =  sc.fit_transform(X)          



cov_matrix = np.cov(X_std.T)

print("cov_matrix shape:",cov_matrix.shape)

print("Covariance_matrix",cov_matrix)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print('Eigen Vectors \n%s', eigenvectors)

print('\n Eigen Values \n%s', eigenvalues)


# Make a set of (eigenvalue, eigenvector) pairs:



eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]



# Sort the (eigenvalue, eigenvector) pairs from highest to lowest with respect to eigenvalue

eig_pairs.sort()



eig_pairs.reverse()

print(eig_pairs)



# Extract the descending ordered eigenvalues and eigenvectors

eigvalues_sorted = [eig_pairs[index][0] for index in range(len(eigenvalues))]

eigvectors_sorted = [eig_pairs[index][1] for index in range(len(eigenvalues))]



# Let's confirm our sorting worked, print out eigenvalues

print('Eigenvalues in descending order: \n%s' %eigvalues_sorted)
tot = sum(eigenvalues)

var_explained = [(i / tot) for i in sorted(eigenvalues, reverse=True)]  # an array of variance explained by each 

# eigen vector... there will be 18 entries as there are 18 eigen vectors)

cum_var_exp = np.cumsum(var_explained)  # an array of cumulative variance. There will be 18 entries with 18 th entry 

# cumulative reaching almost 100%


plt.bar(range(1,19), var_explained, alpha=0.5, align='center', label='individual explained variance')

plt.step(range(1,19),cum_var_exp, where= 'mid', label='cumulative explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc = 'best')

plt.show()
# P_reduce represents reduced mathematical space....



P_reduce = np.array(eigvectors_sorted[0:8])   # Reducing from 8 to 4 dimension space



X_std_8D = np.dot(X_std,P_reduce.T)   # projecting original data into principal component dimensions



reduced_pca = pd.DataFrame(X_std_8D)  # converting array to dataframe for pairplot



reduced_pca


sns.pairplot(reduced_pca, diag_kind='kde') 

#sns.pairplot(reduced_pca1, diag_kind='kde') 
#now split the data into 70:30 ratio



#orginal Data

Orig_X_train,Orig_X_test,Orig_y_train,Orig_y_test = train_test_split(X_std,y,test_size=0.30,random_state=1)



#PCA Data

pca_X_train,pca_X_test,pca_y_train,pca_y_test = train_test_split(reduced_pca,y,test_size=0.30,random_state=1)

#pca_X_train,pca_X_test,pca_y_train,pca_y_test = train_test_split(reduced_pca1,y,test_size=0.30,random_state=1)


svc = SVC() #instantiate the object

#fit the model on orighinal raw data

svc.fit(Orig_X_train,Orig_y_train)
#predict the y value

Orig_y_predict = svc.predict(Orig_X_test)



#now fit the model on pca data with new dimension

svc1 = SVC() #instantiate the object

svc1.fit(pca_X_train,pca_y_train)



#predict the y value

pca_y_predict = svc1.predict(pca_X_test)
#display accuracy score of both models



print("Model Score On Original Data ",svc.score(Orig_X_test, Orig_y_test))

print("Model Score On Reduced PCA Dimension ",svc1.score(pca_X_test, pca_y_test))



print("Before PCA On Original 18 Dimension",accuracy_score(Orig_y_test,Orig_y_predict))

print("After PCA(On 8 dimension)",accuracy_score(pca_y_test,pca_y_predict))
# Calculate Confusion Matrix & PLot To Visualize it



def draw_confmatrix(y_test, yhat, str1, str2, str3, datatype ):

    #Make predictions and evalute

    #model_pred = fit_test_model(model,X_train, y_train, X_test)

    cm = confusion_matrix( y_test, yhat, [0,1,2] )

    print("Confusion Matrix For :", "\n",datatype,cm )

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = [str1, str2,str3] , yticklabels = [str1, str2,str3] )

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()

    



draw_confmatrix(Orig_y_test, Orig_y_predict,"Van ", "Car ", "Bus", "Original Data Set" )



draw_confmatrix(pca_y_test, pca_y_predict,"Van ", "Car ", "Bus", "For Reduced Dimensions Using PCA ")



#Classification Report Of Model built on Raw Data

print("Classification Report For Raw Data:", "\n", classification_report(Orig_y_test,Orig_y_predict))



#Classification Report Of Model built on Principal Components:



print("Classification Report For PCA:","\n", classification_report(pca_y_test,pca_y_predict))
splitscaledf.head(850)
splitscale_X = splitscaledf.iloc[:,0:18].values

splitscale_y = splitscaledf.iloc[:,18].values



print("Indpendent Variable X",splitscale_X )

print("Class Variable y",splitscale_y )
#splitting the data in train and test sets into 70:30 Ratio



SplitScale_X_train, SplitScale_X_test, SplitScale_y_train, SplitScale_y_test = train_test_split(splitscale_X,splitscale_y, test_size = 0.3, random_state = 10)
ssx_train_sd = StandardScaler().fit_transform(SplitScale_X_train)

ssx_test_sd = StandardScaler().fit_transform(SplitScale_X_test)



print(len(ssx_train_sd))

print(len(ssx_test_sd))

# generating the covariance matrix and the eigen values for the PCA analysis

cov_matrix_1 = np.cov(ssx_train_sd.T) # the relevanat covariance matrix

print('Covariance Matrix \n%s', (cov_matrix_1))



#generating the eigen values and the eigen vectors

e_vals, e_vecs = np.linalg.eig(cov_matrix_1)

print('Eigenvectors \n%s' %(e_vecs))

print('\nEigenvalues \n%s' %e_vals)
# Step 3 (continued): Sort eigenvalues in descending order



# Make a set of (eigenvalue, eigenvector) pairs

eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]



# Sort the (eigenvalue, eigenvector) pairs from highest to lowest with respect to eigenvalue

eig_pairs.sort()



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
plt.bar(range(1,19), var_explained, alpha=0.5, align='center', label='individual explained variance')

plt.step(range(1,19),cum_var_exp, where= 'mid', label='cumulative explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc = 'best')

plt.show()
# P_reduce represents reduced mathematical space....



P_reduce_1 = np.array(eigvectors_sorted[0:8])   # Reducing from 8 to 4 dimension space



X_train_std_pca = np.dot(ssx_train_sd,P_reduce_1.T)   # projecting original data into principal component dimensions



X_test_std_pca = np.dot(ssx_test_sd,P_reduce_1.T) 

#Proj_data_df_new = pd.DataFrame(X_std_8D_1) 



print(X_train_std_pca)

print(X_test_std_pca)



Projected_df_train = pd.DataFrame(X_train_std_pca)

Projected_df_test = pd.DataFrame(X_test_std_pca)
sns.pairplot(Projected_df_train, diag_kind='kde')

### Pairplot Analysis : On Test PCA Data Set
sns.pairplot(Projected_df_test, diag_kind='kde')
ssx_train_sd.shape, P_reduce_1.T.shape, X_train_std_pca.shape, X_test_std_pca.shape
clf1 = SVC()

clf1.fit(ssx_train_sd, SplitScale_y_train)

print ('Before PCA score', clf1.score(ssx_test_sd, SplitScale_y_test))



clf2 = SVC()

clf2.fit(X_train_std_pca, SplitScale_y_train)

print ('After PCA score', clf2.score(X_test_std_pca, SplitScale_y_test))





#print("Before PCA On Original 18 Dimension",accuracy_score(Orig_y_test,Orig_y_predict))

#print("After PCA(On 8 dimension)",accuracy_score(pca_y_test,pca_y_predict))



#predict the y value

pca_yhat_predict= clf2.predict(X_test_std_pca)



#orginal data yhat value

orig_yhat_predict = clf1.predict(ssx_test_sd)



print("Before PCA On Original 18 Dimension",accuracy_score(SplitScale_y_test,orig_yhat_predict))

print("After PCA(On 8 dimension)",accuracy_score(SplitScale_y_test,pca_yhat_predict))


draw_confmatrix(SplitScale_y_test, orig_yhat_predict,"Van ", "Car ", "Bus", "Original Data Set" )



draw_confmatrix(SplitScale_y_test, pca_yhat_predict,"Van ", "Car ", "Bus", "For Reduced Dimensions Using PCA ")



#Classification Report Of Model built on Raw Data

print("Classification Report For Raw Data:", "\n", classification_report(SplitScale_y_test,orig_yhat_predict))



#Classification Report Of Model built on Principal Components:



print("Classification Report For PCA:","\n", classification_report(SplitScale_y_test,pca_yhat_predict))
import itertools



def classifiers_hypertune(name,rf,param_grid,x_train_scaled,y_train,x_test_scaled,y_test,CV):

    CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=CV, verbose= 1, n_jobs =-1 )

    CV_rf.fit(x_train_scaled, y_train)

    

    y_pred_train = CV_rf.predict(x_train_scaled)

    y_pred_test = CV_rf.predict(x_test_scaled)

    

    print('Best Score: ', CV_rf.best_score_)

    print('Best Params: ', CV_rf.best_params_)

    

    

    

    #Classification Report

    print(name+" Classification Report: ")

    print(classification_report(y_test, y_pred_test))

    

   

    #Confusion Matrix for test data

    draw_confmatrix(y_test, y_pred_test,"Van", "Car", "Bus", "Original Data Set" )

    print("SVM Accuracy Score:",round(accuracy_score(y_test, y_pred_test),2)*100)

    


#Training on SVM Classifier

from sklearn.model_selection import GridSearchCV

svmc = SVC()



#Let's See What all parameters one can tweak 

print("SVM Parameters:", svmc.get_params())



# Create the parameter grid based on the results of random search 

param_grid = [

  {'C': [0.01, 0.05, 0.5, 1], 'kernel': ['linear']},

  {'C': [0.01, 0.05, 0.5, 1],  'kernel': ['rbf']},

 ]



param_grid_1 = [

  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},

  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},

 ]
classifiers_hypertune("Support Vector Classifier", svmc, param_grid,X_train_std_pca, SplitScale_y_train, X_test_std_pca, SplitScale_y_test,10)
classifiers_hypertune("Support Vector Classifier", svmc, param_grid,ssx_train_sd, SplitScale_y_train, ssx_test_sd, SplitScale_y_test,10)
classifiers_hypertune("Support Vector Classifier_iterarion2", svmc, param_grid_1,X_train_std_pca, SplitScale_y_train, X_test_std_pca, SplitScale_y_test,10)
classifiers_hypertune("Support Vector Classifier", svmc, param_grid_1,ssx_train_sd, SplitScale_y_train, ssx_test_sd, SplitScale_y_test,10)
from sklearn.linear_model import LogisticRegression



model = LogisticRegression()



model.fit(ssx_train_sd, SplitScale_y_train)

print ('Before PCA score', model.score(ssx_test_sd, SplitScale_y_test))



model.fit(X_train_std_pca, SplitScale_y_train)

print ('After PCA score', model.score(X_test_std_pca, SplitScale_y_test))



from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()



nb.fit(ssx_train_sd, SplitScale_y_train)

print ('Before PCA score', nb.score(ssx_test_sd, SplitScale_y_test))



nb.fit(X_train_std_pca, SplitScale_y_train)

print ('After PCA score', nb.score(X_test_std_pca, SplitScale_y_test))

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(criterion = 'entropy' )



dt_model.fit(ssx_train_sd, SplitScale_y_train)

print ('Before PCA score', dt_model.score(ssx_test_sd, SplitScale_y_test))



dt_model.fit(X_train_std_pca, SplitScale_y_train)

print ('After PCA score', dt_model.score(X_test_std_pca, SplitScale_y_test))
