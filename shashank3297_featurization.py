import numpy as np

import pandas as pd

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot

import matplotlib.pyplot as plt

import itertools



from scipy.stats import zscore

from scipy.stats import pearsonr



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor)

from sklearn import svm

from sklearn.svm import SVR

from sklearn import metrics

from sklearn.ensemble import VotingRegressor

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score



from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing

from sklearn.cluster import KMeans

from sklearn.utils import resample
data=pd.read_csv("/kaggle/input/featurization-model/concrete_details.csv")
data.head()
print(data.shape,"\n")

data.info()
data.describe().T
data.isnull().sum()
plt.rcParams.update({'font.size': 30})

plt.style.use('seaborn-whitegrid')



data.hist(bins=20, figsize=(60,40), color='lightblue', edgecolor = 'red',layout=(3,3))

plt.show()

plt.rcParams.update({'font.size': 10})
#Let us use seaborn distplot to analyze the distribution of our columns and see the skewness in attributes

f, ax = plt.subplots(1, 5, figsize=(30,5))



vis1 = sns.distplot(data["cement"],bins=10, ax=ax[0])

vis2 = sns.distplot(data["coarseagg"],bins=10, ax= ax[1])

vis3 = sns.distplot(data["fineagg"],bins=10, ax=ax[2])

vis4 = sns.distplot(data["strength"],bins=10, ax= ax[3])

vis5 = sns.distplot(data["water"],bins=10, ax=ax[4])



f.savefig('subplot.png')
f, ax = plt.subplots(1, 4, figsize=(30,5))



vis1 = sns.distplot(data["age"],bins=10, ax=ax[0])

vis2 = sns.distplot(data["ash"],bins=10, ax= ax[1])

vis3 = sns.distplot(data["slag"],bins=10, ax=ax[2])

vis4 = sns.distplot(data["superplastic"],bins=10, ax= ax[3])



f.savefig('subplot.png')
skewValue = data.skew()

print("skewValue of dataframe attributes:\n", skewValue)
#Summary View of all attribute , Then we will look into all the boxplot individually to trace out outliers



ax = sns.boxplot(data=data, orient="h")

Q1 = data.quantile(0.25)

Q3 = data.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
def outlierCount(aSeries):

    

    q1 = aSeries.quantile(0.25)

    q3 = aSeries.quantile(0.75)

   

    iqr = q3-q1 #Interquartile range

    fence_low  = q1-1.5*iqr

    fence_high = q3+1.5*iqr

    outliers_low = aSeries[(aSeries < fence_low)]

    outliers_high= aSeries[(aSeries > fence_high)]

    return outliers_low.count()>0 or outliers_high.count()>0
outlier=[]

for col in data.columns:

    if(outlierCount(data[col])):

        outlier.append(col)

        

#######

print("Outlier columns are : \n",outlier)
plt.figure(figsize= (20,15))



plt.subplot(3,3,1)

sns.boxplot(x= data['slag'], color='red')



plt.subplot(3,3,2)

sns.boxplot(x= data['water'], color='orange')



plt.subplot(3,3,3)

sns.boxplot(x= data['superplastic'], color='green')



plt.show()



plt.figure(figsize= (20,15))



plt.subplot(3,3,1)

sns.boxplot(x= data['fineagg'], color='brown')



plt.subplot(3,3,2)

sns.boxplot(x= data['age'], color='yellow')



plt.subplot(3,3,3)

sns.boxplot(x= data['strength'], color='lightblue')



plt.show()
#Function returning lower outliers and higher outliers for every attribute.



def fetchOutliers(aSeries):

    

    q1 = aSeries.quantile(0.25)

    q3 = aSeries.quantile(0.75)

   

    iqr = q3-q1 #Interquartile range

    fence_low  = q1-1.5*iqr

    fence_high = q3+1.5*iqr

    outliers_low = aSeries[(aSeries < fence_low)]

    outliers_high= aSeries[(aSeries > fence_high)]

    

    return outliers_low,outliers_high
for col in data.columns:

    if(col!="age" and col!="strength"):

        low,high=fetchOutliers(data[col])

        if(low.count()>0):

            data.loc[low.index,[col]]=data[col].quantile(0.25)-1.5*(data[col].quantile(0.75)-data[col].quantile(0.25))

        if(high.count()>0):

            data.loc[high.index,[col]]=data[col].quantile(0.75)+1.5*(data[col].quantile(0.75)-data[col].quantile(0.25))
outlier=[]

for col in data.columns:

    if(outlierCount(data[col])):

        outlier.append(col)

        

#######

print("Outlier columns are : \n",outlier)
# Heatmap

#Correlation Matrix

corr = data.corr() # correlation matrix

lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix

mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap



plt.figure(figsize = (10,5))  # setting the figure size

sns.set_style(style = 'white')  # Setting it to white so that we do not see the grid lines

sns.heatmap(lower_triangle, center=0, cmap= 'Blues', annot= True, xticklabels = corr.index, yticklabels = corr.columns,

            cbar= False, linewidths= 1, mask = mask)   # Do Heatmap

plt.xticks(rotation = 30)   # Aesthetic purposes

plt.yticks(rotation = 0)   # Aesthetic purposes

plt.show()
sns.pairplot(data,diag_kind="kde")
sns.jointplot(x="cement",y="strength",data=data,kind='hex')
sns.scatterplot(x="slag",y="strength",data=data)
sns.scatterplot(x="ash",y="strength",data=data)
sns.lmplot(x="water",y="strength",data=data)
sns.jointplot(x="superplastic",y="strength",data=data,kind='hex',color='b')
sns.lmplot(x="coarseagg",y="strength",data=data)
sns.jointplot(x="fineagg",y="strength",data=data,kind='hex',color='g')
sns.barplot(x="age",y="strength",data=data)
plt.figure(figsize=(12,25))



for i,j in itertools.zip_longest(data.columns[0:-1],range(len(data.columns)-1)):

    

    plt.subplot(3,3,j+1)

    ax = sns.swarmplot( x = data[i],y=data["strength"],color="orange")

    ax.set_facecolor("k")

    ax.set_xlabel(i)

    ax.set_ylabel("strength")

    ax.set_title(i,color="navy")

    plt.subplots_adjust(wspace = .3)
#Scaling the features



concrete_df_z = data.apply(zscore)

concrete_df_z=pd.DataFrame(concrete_df_z,columns=data.columns)
#Splitting the data into independent and dependent attributes



#independent and dependent variables

X=concrete_df_z.iloc[:,0:8]

y = concrete_df_z.iloc[:,8]
# Split X and y into training and test set in 70:30 ratio



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1)

dt_model = DecisionTreeRegressor()

dt_model.fit(X_train , y_train)
#printing the feature importance

print('Feature importances: \n',pd.DataFrame(dt_model.feature_importances_,columns=['Imp'],index=X_train.columns))
y_pred = dt_model.predict(X_test)

# performance on train data

print('Performance on training data using DT:',dt_model.score(X_train,y_train))

# performance on test data

print('Performance on testing data using DT:',dt_model.score(X_test,y_test))

#Evaluate the model using accuracy

acc_DT=metrics.r2_score(y_test, y_pred)

print('Accuracy DT: ',acc_DT)

print('MSE: ',metrics.mean_squared_error(y_test, y_pred))
sns.set(style="darkgrid", color_codes=True)   

with sns.axes_style("white"):

    sns.jointplot(x=y_test, y=y_pred, stat_func=pearsonr,kind="reg", color="k");
#Store the accuracy results for each model in a dataframe for final comparison

results = pd.DataFrame({'Method':['Decision Tree'], 'accuracy': acc_DT},index={'1'})

results = results[['Method', 'accuracy']]

results
num_folds = 18

seed = 77

kfold = KFold(n_splits=num_folds, random_state=seed)

results1 = cross_val_score(dt_model,X, y, cv=kfold)

accuracy=np.mean(abs(results1))

print('Average accuracy: ',accuracy)

print('Standard Deviation: ',results1.std())
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['Decision Tree k fold'], 'accuracy': [accuracy]},index={'2'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
concrete_df_z.info()
#Create a copy of the dataset

concrete_df2=concrete_df_z.copy()

#independent and dependent variable

X = concrete_df2.drop( ['strength','ash','coarseagg','fineagg'] , axis=1)

y = concrete_df2['strength']

# Split X and y into training and test set in 70:30 ratio

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1)
dt_model = DecisionTreeRegressor()

dt_model.fit(X_train , y_train)
#printing the feature importance

print('Feature importances: \n',pd.DataFrame(dt_model.feature_importances_,columns=['Imp'],index=X_train.columns))
y_pred = dt_model.predict(X_test)

# performance on train data

print('Performance on training data using DT:',dt_model.score(X_train,y_train))

# performance on test data

print('Performance on testing data using DT:',dt_model.score(X_test,y_test))

#Evaluate the model using accuracy

acc_DT=metrics.r2_score(y_test, y_pred)

print('Accuracy DT: ',acc_DT)
sns.set(style="darkgrid", color_codes=True)   

with sns.axes_style("white"):

    sns.jointplot(x=y_test, y=y_pred, stat_func=pearsonr,kind="reg", color="k");
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['Decision Tree2'], 'accuracy': [acc_DT]},index={'3'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
#independent and dependent variables

X=concrete_df_z.iloc[:,0:8]

y = concrete_df_z.iloc[:,8]

# Split X and y into training and test set in 70:30 ratio

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1)
# Regularizing the Decision tree classifier and fitting the model

reg_dt_model = DecisionTreeRegressor( max_depth = 4,random_state=1,min_samples_leaf=5)

reg_dt_model.fit(X_train, y_train)
print (pd.DataFrame(reg_dt_model.feature_importances_, columns = ["Imp"], index = X_train.columns))

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from IPython.display import Image  

#Install the below libraries in python.

import graphviz

import pydot

bank_df=concrete_df_z

xvar = bank_df.drop('strength', axis=1)

feature_cols = xvar.columns
dot_data = StringIO()

export_graphviz(reg_dt_model, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True,feature_names = feature_cols,class_names=['0','1'])

(graph,) = pydot.graph_from_dot_data(dot_data.getvalue())  

graph.write_png('concrete_pruned.png')

Image(graph.create_png())
y_pred = reg_dt_model.predict(X_test)

# performance on train data

print('Performance on training data using DT:',reg_dt_model.score(X_train,y_train))

# performance on test data

print('Performance on testing data using DT:',reg_dt_model.score(X_test,y_test))

#Evaluate the model using accuracy

acc_RDT=metrics.r2_score(y_test, y_pred)

print('Accuracy DT: ',acc_RDT)

print('MSE: ',metrics.mean_squared_error(y_test, y_pred))
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['Pruned Decision Tree'], 'accuracy': [acc_RDT]},index={'4'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
num_folds = 18

seed = 77

kfold = KFold(n_splits=num_folds, random_state=seed)

results1 = cross_val_score(reg_dt_model,X, y, cv=kfold)

accuracy=np.mean(abs(results1))

print('Average accuracy: ',accuracy)

print('Standard Deviation: ',results1.std())
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['Pruned Decision Tree k fold'], 'accuracy': [accuracy]},index={'5'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
#Create a copy of the dataset

concrete_df3=concrete_df_z.copy()
#independent and dependent variable

X = concrete_df3.drop( ['strength','ash','coarseagg','fineagg'] , axis=1)

y = concrete_df3['strength']

# Split X and y into training and test set in 70:30 ratio

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1)
# Regularizing the Decision tree classifier and fitting the model

reg_dt_model = DecisionTreeRegressor( max_depth = 4,random_state=1,min_samples_leaf=5)

reg_dt_model.fit(X_train, y_train)
y_pred = reg_dt_model.predict(X_test)

# performance on train data

print('Performance on training data using DT:',reg_dt_model.score(X_train,y_train))

# performance on test data

print('Performance on testing data using DT:',reg_dt_model.score(X_test,y_test))

#Evaluate the model using accuracy

acc_RDT=metrics.r2_score(y_test, y_pred)

print('Accuracy DT: ',acc_RDT)

print('MSE: ',metrics.mean_squared_error(y_test, y_pred))
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['Pruned Decision Tree2'], 'accuracy': [acc_RDT]},index={'6'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
cluster_range = range( 1, 15 )  

cluster_errors = []

for num_clusters in cluster_range:

    clusters = KMeans( num_clusters, n_init = 5)

    clusters.fit(data)

    labels = clusters.labels_

    centroids = clusters.cluster_centers_

    cluster_errors.append( clusters.inertia_ )

clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

clusters_df[0:15]
# Elbow plot

plt.figure(figsize=(12,6))

plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
# k=6

cluster = KMeans( n_clusters = 6, random_state = 2354 )

cluster.fit(concrete_df_z)
# Creating a new column "GROUP" which will hold the cluster id of each record

prediction=cluster.predict(concrete_df_z)

concrete_df_z["GROUP"] = prediction     

# Creating a mirror copy for later re-use instead of building repeatedly

concrete_df_z_copy = concrete_df_z.copy(deep = True)
centroids = cluster.cluster_centers_

centroids
centroid_df = pd.DataFrame(centroids, columns = list(data) )

centroid_df
## Instead of interpreting the neumerical values of the centroids, let us do a visual analysis by converting the 

## centroids and the data in the cluster into box plots.

concrete_df_z.boxplot(by = 'GROUP',  layout=(3,3), figsize=(15, 10))
#independent and dependent variables

X=concrete_df_z.iloc[:,0:8]

y = concrete_df_z.iloc[:,8]

# Split X and y into training and test set in 70:30 ratio

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1)
model=RandomForestRegressor()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# performance on train data

print('Performance on training data using RFR:',model.score(X_train,y_train))

# performance on test data

print('Performance on testing data using RFR:',model.score(X_test,y_test))

#Evaluate the model using accuracy

acc_RFR=metrics.r2_score(y_test, y_pred)

print('Accuracy DT: ',acc_RFR)

print('MSE: ',metrics.mean_squared_error(y_test, y_pred))
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['Random Forest Regressor'], 'accuracy': [acc_RFR]},index={'7'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
num_folds = 20

seed = 77

kfold = KFold(n_splits=num_folds, random_state=seed)

results1 = cross_val_score(model,X, y, cv=kfold)

accuracy=np.mean(abs(results1))

print('Average accuracy: ',accuracy)

print('Standard Deviation: ',results1.std())
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['Random Forest Regressor k fold'], 'accuracy': [accuracy]},index={'8'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
model=GradientBoostingRegressor()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# performance on train data

print('Performance on training data using GBR:',model.score(X_train,y_train))

# performance on test data

print('Performance on testing data using GBR:',model.score(X_test,y_test))

#Evaluate the model using accuracy

acc_GBR=metrics.r2_score(y_test, y_pred)

print('Accuracy DT: ',acc_GBR)

print('MSE: ',metrics.mean_squared_error(y_test, y_pred))
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['Gradient Boost Regressor'], 'accuracy': [acc_GBR]},index={'9'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
num_folds = 20

seed = 77

kfold = KFold(n_splits=num_folds, random_state=seed)

results1 = cross_val_score(model,X, y, cv=kfold)

accuracy=np.mean(abs(results1))

print('Average accuracy: ',accuracy)

print('Standard Deviation: ',results1.std())
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['Gradient Boost Regressor k fold'], 'accuracy': [accuracy]},index={'10'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
model=AdaBoostRegressor()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# performance on train data

print('Performance on training data using GBR:',model.score(X_train,y_train))

# performance on test data

print('Performance on testing data using GBR:',model.score(X_test,y_test))

#Evaluate the model using accuracy

acc_ABR=metrics.r2_score(y_test, y_pred)

print('Accuracy DT: ',acc_ABR)

print('MSE: ',metrics.mean_squared_error(y_test, y_pred))
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['Ada Boosting Regressor'], 'accuracy': [acc_ABR]},index={'11'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
num_folds = 18

seed = 77

kfold = KFold(n_splits=num_folds, random_state=seed)

results1 = cross_val_score(model,X, y, cv=kfold)

accuracy=np.mean(abs(results1))

print('Average accuracy: ',accuracy)

print('Standard Deviation: ',results1.std())
tempResultsDf = pd.DataFrame({'Method':['Ada Boosting Regressor k fold'], 'accuracy': [accuracy]},index={'12'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
model=BaggingRegressor()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# performance on train data

print('Performance on training data using GBR:',model.score(X_train,y_train))

# performance on test data

print('Performance on testing data using GBR:',model.score(X_test,y_test))

#Evaluate the model using accuracy

acc_BR=metrics.r2_score(y_test, y_pred)

print('Accuracy DT: ',acc_BR)

print('MSE: ',metrics.mean_squared_error(y_test, y_pred))
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['Bagging Regressor'], 'accuracy': [acc_BR]},index={'13'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
num_folds = 20

seed = 77

kfold = KFold(n_splits=num_folds, random_state=seed)

results1 = cross_val_score(model,X, y, cv=kfold)

accuracy=np.mean(abs(results1))

print('Average accuracy: ',accuracy)

print('Standard Deviation: ',results1.std())
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['Bagging Regressor k fold'], 'accuracy': [accuracy]},index={'14'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
error=[]

for i in range(1,30):

    knn = KNeighborsRegressor(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error.append(np.mean(pred_i!=y_test))
plt.figure(figsize=(12,6))

plt.plot(range(1,30),error,color='red', linestyle='dashed',marker='o',markerfacecolor='blue',markersize=10)

plt.title('Error Rate K Value')

plt.xlabel('K Value')

plt.ylabel('Mean error')
#k=3

model = KNeighborsRegressor(n_neighbors=3)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# performance on train data

print('Performance on training data using KNNR:',model.score(X_train,y_train))

# performance on test data

print('Performance on testing data using KNNR:',model.score(X_test,y_test))

#Evaluate the model using accuracy

acc_K=metrics.r2_score(y_test, y_pred)

print('Accuracy KNNR: ',acc_K)

print('MSE: ',metrics.mean_squared_error(y_test, y_pred))
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['KNN Regressor'], 'accuracy': [acc_K]},index={'15'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
num_folds = 30

seed = 77

kfold = KFold(n_splits=num_folds, random_state=seed)

results1 = cross_val_score(model,X, y, cv=kfold)

accuracy=np.mean(abs(results1))

print('Average accuracy: ',accuracy)

print('Standard Deviation: ',results1.std())
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['KNN Regressor k fold'], 'accuracy': [accuracy]},index={'16'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
model = SVR(kernel='linear')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# performance on train data

print('Performance on training data using SVR:',model.score(X_train,y_train))

# performance on test data

print('Performance on testing data using SVR:',model.score(X_test,y_test))

#Evaluate the model using accuracy

acc_S=metrics.r2_score(y_test, y_pred)

print('Accuracy SVR: ',acc_S)

print('MSE: ',metrics.mean_squared_error(y_test, y_pred))
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['Support Vector Regressor'], 'accuracy': [acc_S]},index={'17'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
num_folds = 10

seed = 77

kfold = KFold(n_splits=num_folds, random_state=seed)

results1 = cross_val_score(model,X, y, cv=kfold)

accuracy=np.mean(abs(results1))

print('Average accuracy: ',accuracy)

print('Standard Deviation: ',results1.std())
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['SVR k fold'], 'accuracy': [accuracy]},index={'18'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
#Multiple model Ensemble

LR=LinearRegression()

KN=KNeighborsRegressor(n_neighbors=3)

SVM=svm.SVR(kernel='linear')


evc=VotingRegressor(estimators=[('LR',LR),('KN',KN),('SVM',SVM)])

evc.fit(X_train, y_train)
y_pred = evc.predict(X_test)

# performance on train data

print('Performance on training data using ensemble:',evc.score(X_train,y_train))

# performance on test data

print('Performance on testing data using ensemble:',evc.score(X_test,y_test))

#Evaluate the model using accuracy

acc_E=metrics.r2_score(y_test, y_pred)

print('Accuracy ensemble: ',acc_E)

print('MSE: ',metrics.mean_squared_error(y_test, y_pred))
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['Ensemble'], 'accuracy': [acc_E]},index={'19'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results

num_folds = 10

seed = 77

kfold = KFold(n_splits=num_folds, random_state=seed)

results1 = cross_val_score(evc,X, y, cv=kfold)

accuracy=np.mean(abs(results1))

print('Average accuracy: ',accuracy)

print('Standard Deviation: ',results1.std())
#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['Ensemble k fold'], 'accuracy': [accuracy]},index={'20'})

results = pd.concat([results, tempResultsDf])

results = results[['Method', 'accuracy']]

results
concrete_XY = X.join(y)
values = concrete_XY.values

# Number of bootstrap samples to create

n_iterations = 1000        

# size of a bootstrap sample

n_size = int(len(concrete_df_z) * 1)    



# run bootstrap

# empty list that will hold the scores for each bootstrap iteration

stats = list()   

for i in range(n_iterations):

    # prepare train and test sets

    train = resample(values, n_samples=n_size)  # Sampling with replacement 

    test = np.array([x for x in values if x.tolist() not in train.tolist()])# picking rest of the data not considered in sample

    

    

     # fit model

    gbmTree = GradientBoostingRegressor(n_estimators=50)

    # fit against independent variables and corresponding target values

    gbmTree.fit(train[:,:-1], train[:,-1]) 

    # Take the target column for all rows in test set



    y_test = test[:,-1]    

    # evaluate model

    # predict based on independent variables in the test data

    score = gbmTree.score(test[:, :-1] , y_test)

    predictions = gbmTree.predict(test[:, :-1])  



    stats.append(score)
# plot scores



pyplot.hist(stats)

pyplot.show()

# confidence intervals

alpha = 0.95                             # for 95% confidence 

p = ((1.0-alpha)/2.0) * 100              # tail regions on right and left .25 on each side indicated by P value (border)

lower = max(0.0, np.percentile(stats, p))  

p = (alpha+((1.0-alpha)/2.0)) * 100

upper = min(1.0, np.percentile(stats, p))

print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
values = concrete_XY.values

# Number of bootstrap samples to create

n_iterations = 1000

# size of a bootstrap sample

n_size = int(len(concrete_df_z) * 1)    



# run bootstrap

# empty list that will hold the scores for each bootstrap iteration

stats = list()   

for i in range(n_iterations):

    # prepare train and test sets

    train = resample(values, n_samples=n_size)  # Sampling with replacement 

    test = np.array([x for x in values if x.tolist() not in train.tolist()])  # picking rest of the data not considered in sample

    

    

     # fit model

    rfTree = RandomForestRegressor(n_estimators=100)

    # fit against independent variables and corresponding target values

    rfTree.fit(train[:,:-1], train[:,-1]) 

    # Take the target column for all rows in test set



    y_test = test[:,-1]    

    # evaluate model

    # predict based on independent variables in the test data

    score = rfTree.score(test[:, :-1] , y_test)

    predictions = rfTree.predict(test[:, :-1])  



    stats.append(score)
# plot scores



pyplot.hist(stats)

pyplot.show()

# confidence intervals

alpha = 0.95                             # for 95% confidence 

p = ((1.0-alpha)/2.0) * 100              # tail regions on right and left .25 on each side indicated by P value (border)

lower = max(0.0, np.percentile(stats, p))  

p = (alpha+((1.0-alpha)/2.0)) * 100

upper = min(1.0, np.percentile(stats, p))

print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))