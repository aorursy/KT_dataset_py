"""

   Author : Kenil Shah

   Github: Data-Science-Analytics/Datasets/UCLA Graduate Admission Prediction/Graduate_Admission_Prediction.ipynb

   

"""   

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import cross_val_score



%matplotlib inline
dataset = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')

dataset.head() 
dataset.info()
dataset.drop('Serial No.',axis = 1,inplace = True)

dataset.columns = ['GRE', 'TOEFL', 'University Rating','SOP','LOR','CGPA','Research','Chance of Admit']
plt.rcParams['figure.figsize'] = 15,10

dataset['GRE'].plot(kind = 'kde')

dataset['GRE'].plot(kind = 'hist',density = True,color = 'g',alpha = 0.25)

plt.xlabel('GRE Score',fontsize = 20)

plt.ylabel('Frequency',fontsize = 20)

plt.title('Histogram and Distribution Plot of GRE Score',fontsize = 20)

plt.text(308, 0.013, 'Average Score %d' %(int(np.mean(dataset['GRE']))), fontsize=15)

plt.show()
dataset[dataset['Chance of Admit'] > 0.75]['GRE'].plot(kind = 'hist',x = 'GRE',color = 'g',alpha = 0.25)

plt.text(320, 20, 'Average Score %d' %(int(np.mean(dataset[dataset['Chance of Admit'] > 0.75]['GRE']))), fontsize=15)

plt.xlabel('GRE Score',fontsize = 20)

plt.ylabel('Frequency',fontsize = 20)

plt.title('Histogram and Distribution Plot of GRE Score with Chances higher than 75%',fontsize = 20)

plt.show()
# add SNS plot here with hue = 'Research'

plt.rcParams['figure.figsize'] = 10,20

sns.lmplot(x = 'GRE' , y = 'Chance of Admit',hue = 'Research',data = dataset,fit_reg = False)

plt.axvline(x = 300,ymin = 0,ymax = 1)

plt.axvline(x = 320,ymin = 0,ymax = 1,color = 'orange')

plt.axhline(y = 0.6,color = 'green')

plt.title('Impact of Research')

plt.show()
plt.rcParams['figure.figsize'] = 15,10

dataset['TOEFL'].plot(kind = 'kde')

dataset['TOEFL'].plot(kind = 'hist',density = True,color = 'r',alpha = 0.25)

plt.text(100, 0.02, 'Average Score %d' %(int(np.mean(dataset['TOEFL']))), fontsize=15)

plt.xlabel('TOEFL Score',fontsize = 20)

plt.ylabel('Frequency',fontsize = 20)

plt.title('Histogram and Distribution Plot of TOEFL Score',fontsize = 20)

plt.show()
dataset[dataset['Chance of Admit'] > 0.75]['TOEFL'].plot(kind = 'hist',x = 'TOEFL',color = 'r',alpha = 0.25)

plt.text(110, 15, 'Average Score %d' %(int(np.mean(dataset[dataset['Chance of Admit'] > 0.75]['TOEFL']))), fontsize=15)

plt.xlabel('TOEFL Score',fontsize = 20)

plt.ylabel('Frequency',fontsize = 20)

plt.title('Histogram and Distribution Plot of TOEFL Score with chances higher than 75%',fontsize = 20)

plt.show()
# add SNS plot here with hue = 'Research'

plt.rcParams['figure.figsize'] = 10,20

sns.lmplot(x = 'TOEFL' , y = 'Chance of Admit',hue = 'Research',data = dataset,fit_reg = False)

plt.axvline(x = 100,ymin = 0,ymax = 1)

plt.axvline(x = 110,ymin = 0,ymax = 1,color = 'orange')

plt.axhline(y = 0.6,color = 'green')

plt.title('Impact of Research')

plt.show()
dataset['University Rating'].unique() # We have 5 different Ratings by the University
plt.rcParams['figure.figsize'] = 15,5

sns.swarmplot(x = 'University Rating', y = 'Chance of Admit', hue = 'Research',data = dataset)

plt.title('Impact of Research')
dataset[(dataset['University Rating'] >= 4) & (dataset['Research'] == 1)].sort_values(by = ['Chance of Admit']).head(5)
plt.rcParams['figure.figsize'] = 10,5

dataset['CGPA'].plot(kind = 'kde')

dataset['CGPA'].plot(kind = 'hist',density = True,color = 'y',alpha = 0.25)

plt.text(7.88, 0.2, 'Average CGPA %d' %(int(np.mean(dataset['CGPA']))), fontsize=15)

plt.xlabel('CGPA',fontsize = 20)

plt.ylabel('Frequency',fontsize = 20)

plt.title('Histogram and Distribution Plot of CGPA',fontsize = 20)

plt.show()
dataset[dataset['Chance of Admit'] > 0.75]['CGPA'].plot(kind = 'hist',x = 'CGPA',color = 'y',alpha = 0.25)

plt.text(8.75, 20, 'Average Score %d' %(int(np.mean(dataset[dataset['Chance of Admit'] > 0.75]['CGPA']))), fontsize=15)

plt.xlabel('CGPA',fontsize = 20)

plt.ylabel('Frequency',fontsize = 20)

plt.xlim(8,10)

plt.title('Histogram and Distribution Plot of CGPA with chances higher than 75%',fontsize = 20)

plt.show()
# add SNS plot here with hue = 'Research'

plt.rcParams['figure.figsize'] = 10,20

sns.lmplot(x = 'CGPA' , y = 'Chance of Admit',hue = 'Research',data = dataset,fit_reg = False)

plt.axvline(x = 7,ymin = 0,ymax = 1)

plt.axvline(x = 9,ymin = 0,ymax = 1,color = 'orange')

plt.axhline(y = 0.6,color = 'green')

plt.title('Impact of Research')

plt.show()
print('Average SOP :', int(np.mean(dataset['SOP'])))

print('Average LOR :', int(np.mean(dataset['LOR'])))

plt.rcParams['figure.figsize'] = 10,5



dataset['SOP'].plot(kind = 'kde')

dataset['LOR'].plot(kind = 'kde')

plt.legend(['SOP','LOR'])

#dataset['CGPA'].plot(kind = 'hist',density = True,color = 'y',alpha = 0.25)
# Swarmplot for SOP and LOR values with hue Reasearch and y Chance of Admit

plt.rcParams['figure.figsize'] = 15,5

sns.swarmplot(x = 'SOP', y = 'Chance of Admit', hue = 'Research',data = dataset)

plt.title('Impact of Research')

plt.rcParams['figure.figsize'] = 15,5

sns.swarmplot(x = 'LOR', y = 'Chance of Admit', hue = 'Research',data = dataset)

plt.title('Impact of Research')

sns.pairplot(dataset,vars = ['GRE','TOEFL','University Rating','SOP','LOR','CGPA','Chance of Admit'],

             kind = 'reg',diag_kind = 'kde',palette="husl")
#Dividing it into Independent and Dependent Variables



X = dataset.iloc[:,:-1].values # Independent Variables

Y = dataset.iloc[:,7].values # Dependent Variables
#Splitting it into train and test dataset

train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2,random_state = 0) 

# Splitting it into 400 train and 100 test data set

model_simple = LinearRegression()

model_simple.fit(train_X,train_Y)
pred = model_simple.predict(test_X)
print('Mean Square Error is: ', mean_squared_error(test_Y,pred))

print('Model Accuracy Score : ',r2_score(test_Y,pred))
plt.scatter(model_simple.predict(train_X),model_simple.predict(train_X) - train_Y, c = 'b')

plt.hlines(y = 0,xmin = min(model_simple.predict(train_X)),xmax = max(model_simple.predict(train_X)))
#Splitting it into train and test dataset

train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2,random_state = 0) 

# Splitting it into 400 train and 100 test data set
polynomial = PolynomialFeatures(degree = 2)   # Degree 2

polynomial_x = polynomial.fit_transform(train_X)

polynomial.fit(polynomial_x, train_Y)



polynomial_3 = PolynomialFeatures(degree = 3)   # Degree 3

polynomial_x_3 = polynomial_3.fit_transform(train_X)

polynomial_3.fit(polynomial_x_3, train_Y)
model_poly = LinearRegression()  # Degree 2

model_poly.fit(polynomial_x,train_Y)



model_poly_3 = LinearRegression() # Degree 3

model_poly_3.fit(polynomial_x_3,train_Y)
pred_2 = model_poly.predict(polynomial.fit_transform(test_X)) # Degree 2

pred_3 = model_poly_3.predict(polynomial_3.fit_transform(test_X)) # Degree 3
print('Mean Square Error for Polynomial degree 2 is: ', mean_squared_error(test_Y,pred_2))

print('Model Accuracy Score for Polynomial degree 2 is : ',r2_score(test_Y,pred_2))

print('Mean Square Error for Polynomial degree 3 is: ', mean_squared_error(test_Y,pred_3))

print('Model Accuracy Score for Polynomial degree 3 is : ',r2_score(test_Y,pred_3))
plt.figure(figsize=(15,5))

x = np.arange(1,50)

plt.plot(x,test_Y[:49], '-o', label='Actual',color ='green')

plt.plot(x, pred[:49], ':o', label='Predicted',color = 'red',linewidth = 1)

plt.plot(x, pred_3[:49], ':x', label='Predicted',color = 'blue',linewidth = 1)

plt.legend();
#Splitting it into train and test dataset

train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2,random_state = 0) 

# Splitting it into 400 train and 100 test data set
model_svr_rbf = SVR(kernel = 'rbf',C = 10,gamma = 0.01) # Gaussian Kernel

model_svr_rbf.fit(train_X,train_Y)



model_svr_linear = SVR(kernel = 'linear', C = 1) #Linear kernel

model_svr_linear.fit(train_X,train_Y)
pred_svr_rbf = model_svr_rbf.predict(test_X)

pred_svr_linear = model_svr_linear.predict(test_X)
print('Mean Square Error for Gaussian(Radial) kernel is: ', mean_squared_error(test_Y,pred_svr_rbf))

print('Model Accuracy Score for Gaussian(Radial) kernel is : ',r2_score(test_Y,pred_svr_rbf))

print('Mean Square Error for Linear kernel is: ', mean_squared_error(test_Y,pred_svr_linear))

print('Model Accuracy Score for Linear kernel is : ',r2_score(test_Y,pred_svr_linear))
plt.figure(figsize=(15,5))

x = np.arange(1,50)

plt.plot(x,test_Y[:49], '-o', label='Actual',color ='green')

plt.plot(x, pred_svr_rbf[:49], ':o', label='Predicted',color = 'red',linewidth = 1)

plt.plot(x, pred_svr_linear[:49], ':x', label='Predicted',color = 'blue',linewidth = 1)

plt.legend();
#Splitting it into train and test dataset

train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2,random_state = 0) 

# Splitting it into 400 train and 100 test data set
model_dtree = DecisionTreeRegressor(random_state = 0,max_depth = 5,max_features = 5,min_samples_split = 10)

model_dtree.fit(train_X,train_Y)
pred_dtree = model_dtree.predict(test_X)
print('Mean Square Error : ', mean_squared_error(test_Y,pred_dtree))

print('Model Accuracy Score : ',r2_score(test_Y,pred_dtree))
plt.figure(figsize=(15,5))

x = np.arange(1,50)

plt.plot(x,test_Y[:49], '-o', label='Actual',color ='green')

plt.plot(x, pred_dtree[:49], ':o', label='Predicted',color = 'red',linewidth = 2)

plt.legend();
#Splitting it into train and test dataset

train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2,random_state = 0) 

# Splitting it into 400 train and 100 test data set
model_rforest = RandomForestRegressor(n_estimators = 500,random_state = 0,max_depth = 7

                                      ,max_features = 5,min_samples_split = 10)

model_rforest.fit(train_X,train_Y)
pred_rforest = model_rforest.predict(test_X)
print('Mean Square Error : ', mean_squared_error(test_Y,pred_rforest))

print('Model Accuracy Score : ',r2_score(test_Y,pred_rforest))
plt.figure(figsize=(15,5))

x = np.arange(1,50)

plt.plot(x,test_Y[:49], '-o', label='Actual',color ='green')

plt.plot(x, pred_rforest[:49], ':o', label='Predicted',color = 'red',linewidth = 2)

plt.legend();
#Splitting it into train and test dataset

train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2,random_state = 0) 

# Splitting it into 400 train and 100 test data set
pca = PCA(n_components = None)

train_X_pca = pca.fit_transform(train_X)

test_X_pca = pca.fit(test_X)

explained_variance = pca.explained_variance_ratio_

for x in explained_variance:

    print(round(x,2))
pca = PCA(n_components = 4)

train_X_pca = pca.fit_transform(train_X)

test_X_pca = pca.transform(test_X)

model_simple_pca = LinearRegression()

model_simple_pca.fit(train_X_pca,train_Y)
pred_pca = model_simple_pca.predict(test_X_pca)
print('Mean Square Error : ', mean_squared_error(test_Y,pred_pca))

print('Model Accuracy Score : ',r2_score(test_Y,pred_pca))
plt.figure(figsize=(15,5))

x = np.arange(1,50)

plt.plot(x,test_Y[:49], '-o', label='Actual',color ='green')

plt.plot(x, pred_pca[:49], ':o', label='Predicted',color = 'red',linewidth = 2)

plt.legend();
#Splitting it into train and test dataset

train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2,random_state = 0) 

# Splitting it into 400 train and 100 test data set
model_knn = KNeighborsRegressor(n_neighbors =10, metric = 'minkowski' , p = 2)

model_knn.fit(train_X,train_Y)
pred_knn = model_knn.predict(test_X)
print('Mean Square Error : ', mean_squared_error(test_Y,pred_knn))

print('Model Accuracy Score : ',r2_score(test_Y,pred_knn))
plt.figure(figsize=(15,5))

x = np.arange(1,50)

plt.plot(x,test_Y[:49], '-o', label='Actual',color ='green')

plt.plot(x, pred_knn[:49], ':o', label='Predicted',color = 'red',linewidth = 2)

plt.legend();
cv = cross_val_score(estimator = model_rforest,X = X,y = Y,cv = 10)  # 10 parts
for accuracy in cv:

    print(accuracy)    
print('Accuracy mean:',cv.mean())

print('Accuracy Standard Deviation:',cv.std())
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()

train_X = minmax.fit_transform(train_X)

test_X = minmax.transform(test_X)
model_simple_normalized = LinearRegression()

model_simple_normalized.fit(train_X,train_Y)

pred_normalized = model_simple_normalized.predict(test_X)

print('Mean Square Error is: ', mean_squared_error(test_Y,pred_normalized))

print('Model Accuracy Score : ',r2_score(test_Y,pred_normalized))
model_rforest_normalized = RandomForestRegressor(n_estimators = 500,random_state = 0,max_depth = 7

                                      ,max_features = 5,min_samples_split = 10)

model_rforest_normalized.fit(train_X,train_Y)

pred_rforest_normalized = model_rforest_normalized.predict(test_X)

print('Mean Square Error is: ', mean_squared_error(test_Y,pred_rforest_normalized))

print('Model Accuracy Score : ',r2_score(test_Y,pred_rforest_normalized))
index = ['Linear','Polynomial_2','Polynomial_3','SupportVectorGuassin','SupportVectorLinear',

         'DecisionTree','RandomForest','PCR','KNearest','RandomForestNormalized','LinearNormalized']

mse = [[mean_squared_error(test_Y,pred),r2_score(test_Y,pred)],

       [mean_squared_error(test_Y,pred_2),r2_score(test_Y,pred_2)],

       [mean_squared_error(test_Y,pred_3),r2_score(test_Y,pred_3)],

       [mean_squared_error(test_Y,pred_svr_rbf),r2_score(test_Y,pred_svr_rbf)],

       [mean_squared_error(test_Y,pred_svr_linear),r2_score(test_Y,pred_svr_linear)],

       [mean_squared_error(test_Y,pred_dtree),r2_score(test_Y,pred_dtree)],

       [mean_squared_error(test_Y,pred_rforest),r2_score(test_Y,pred_rforest)],

       [mean_squared_error(test_Y,pred_pca),r2_score(test_Y,pred_pca)],

       [mean_squared_error(test_Y,pred_knn),r2_score(test_Y,pred_knn)],

       [mean_squared_error(test_Y,pred_rforest_normalized),r2_score(test_Y,pred_rforest_normalized)],

       [mean_squared_error(test_Y,pred_normalized),r2_score(test_Y,pred_normalized)]]

data = pd.DataFrame(data = mse,index = index)

data.columns = ['MSE','Accuracy']

data.sort_values(by = 'Accuracy',ascending = False)
