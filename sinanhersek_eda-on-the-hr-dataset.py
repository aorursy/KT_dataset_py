import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

#my first Kaggle notebook so check if things are ruuning on track :D
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"));
#create a data frame where all data is numberical: convert categorical variables to numerical
dataFrNumeric = pd.read_csv("../input/HR_comma_sep.csv");
dataFrNumeric['sales numeric']=0 
dataFrNumeric['salary numeric']=0 

#hard code the conversion of the salary categorical variable to a numerical variable
dataFrNumeric.loc[(dataFrNumeric['salary'] == 'low'),'salary numeric']=1 
dataFrNumeric.loc[(dataFrNumeric['salary'] == 'medium'),'salary numeric']=2
dataFrNumeric.loc[(dataFrNumeric['salary'] == 'high'),'salary numeric']=3 

#new feature projects worked on per year
dataFrNumeric['projects_per_year'] = dataFrNumeric['number_project']/dataFrNumeric['time_spend_company']  

#hard code the conversion of the sales categorical variable to a numerical variable
dataFrNumeric.loc[(dataFrNumeric['sales'] == 'hr'),'sales numeric']=1 
dataFrNumeric.loc[(dataFrNumeric['sales'] == 'sales'),'sales numeric']=2 
dataFrNumeric.loc[(dataFrNumeric['sales'] == 'support'),'sales numeric']=3 
dataFrNumeric.loc[(dataFrNumeric['sales'] == 'accounting'),'sales numeric']=4 
dataFrNumeric.loc[(dataFrNumeric['sales'] == 'marketing'),'sales numeric']=5 
dataFrNumeric.loc[(dataFrNumeric['sales'] == 'IT'),'sales numeric']=6 
dataFrNumeric.loc[(dataFrNumeric['sales'] == 'technical'),'sales numeric']=7 
dataFrNumeric.loc[(dataFrNumeric['sales'] == 'management'),'sales numeric']=8 
dataFrNumeric.loc[(dataFrNumeric['sales'] == 'product_mng'),'sales numeric']=9 
dataFrNumeric.loc[(dataFrNumeric['sales'] == 'product_mng'),'sales numeric']=10 
#produce kernel density estimate plots and histograms to look at each feature
fig = plt.figure()
ax=sns.kdeplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 0),'satisfaction_level'] , color='b')
ax=sns.kdeplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 1),'satisfaction_level'] , color='r')
fig = plt.figure()
ax=sns.kdeplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 0),'last_evaluation'] , color='b')
ax=sns.kdeplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 1),'last_evaluation'] , color='r')
fig = plt.figure()
ax=sns.distplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 0),'number_project'] , color='b',kde=False, bins=range(0,8,1) ,norm_hist=True)
ax=sns.distplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 1),'number_project'] , color='r',kde=False, bins=range(0,8,1),norm_hist=True)
fig = plt.figure()
ax=sns.kdeplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 0),'projects_per_year'] , color='b')
ax=sns.kdeplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 1),'projects_per_year'] , color='r')
fig = plt.figure()
ax=sns.kdeplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 0),'average_montly_hours'] , color='b')
ax=sns.kdeplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 1),'average_montly_hours'] , color='r')
fig = plt.figure()
ax=sns.distplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 0),'time_spend_company'] , color='b',kde=False, bins=range(0,11,1),norm_hist=True)
ax=sns.distplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 1),'time_spend_company'] , color='r',kde=False, bins=range(0,11,1),norm_hist=True)
fig = plt.figure()
ax=sns.distplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 0),'Work_accident'] , color='b',kde=False,bins=[-0.5,0.5,1.5],norm_hist=True)
ax=sns.distplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 1),'Work_accident'] , color='r',kde=False,bins=[-0.5,0.5,1.5],norm_hist=True)
fig = plt.figure()
ax=sns.distplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 0),'promotion_last_5years'] , color='b',kde=False,bins=[-0.5,0.5,1.5],norm_hist=True)
ax=sns.distplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 1),'promotion_last_5years'] , color='r',kde=False,bins=[-0.5,0.5,1.5],norm_hist=True)
fig = plt.figure()
ax=sns.distplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 0),'sales numeric'] , color='b',kde=False,bins=range(0,11,1),norm_hist=True)
ax=sns.distplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 1),'sales numeric'] , color='r',kde=False,bins=range(0,11,1),norm_hist=True)
fig = plt.figure()
ax=sns.distplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 0),'salary numeric'] , color='b',kde=False,bins=[0.5,1.5,2.5,3.5],norm_hist=True)
ax=sns.distplot(dataFrNumeric.loc[(dataFrNumeric['left'] == 1),'salary numeric'] , color='r',kde=False,bins=[0.5,1.5,2.5,3.5],norm_hist=True)
from sklearn import preprocessing

#one-hot encode the sales column
salesNumericVector = (dataFrNumeric['sales numeric'].as_matrix()).reshape(-1,1)
enc = preprocessing.OneHotEncoder()
enc.fit(salesNumericVector)  
salesOneHot= enc.transform(salesNumericVector).toarray()


#one-hot encode the salary column
salaryNumericVector = (dataFrNumeric['salary numeric'].as_matrix()).reshape(-1,1)
enc.fit(salaryNumericVector)  
salaryOneHot= enc.transform(salaryNumericVector).toarray()

#convert data frame into a data matrix
dataMatrixTemp =dataFrNumeric[['satisfaction_level','last_evaluation', 'number_project' 
                               , 'average_montly_hours' , 'time_spend_company', 'Work_accident', 
                               'promotion_last_5years' , 'projects_per_year' ]].as_matrix()

#get the labels (did the employee leave ? )
labels = dataFrNumeric['left']

#concatanate the one hot encoded salary and sales features to the data matrix
dataMatrix = np.concatenate((dataMatrixTemp, salesOneHot , salaryOneHot) , axis=1)
from sklearn import manifold
from sklearn.preprocessing import Imputer


#seperate data matrices for the people who left and who did not leave
X_1=np.squeeze(dataMatrix[np.where(labels==1),:])
X_0=np.squeeze(dataMatrix[np.where(labels==0),:])
salesNumericVector_0 = salesNumericVector[np.where(labels==0)]
salesNumericVector_1 = salesNumericVector[np.where(labels==1)]
salaryNumericVector_0 =salaryNumericVector[np.where(labels==0)]
salaryNumericVector_1 =salaryNumericVector[np.where(labels==1)]

#random shuffle
numberOfDataPointsToVisualize = 2000;
shuffledIndices_0 = np.random.randint(0,X_0.shape[0]-1,numberOfDataPointsToVisualize)
shuffledIndices_1 = np.random.randint(0,X_1.shape[0]-1,numberOfDataPointsToVisualize)
salesNumericVector_0 = salesNumericVector_0[shuffledIndices_0]
salesNumericVector_1 = salesNumericVector_1[shuffledIndices_1]
salaryNumericVector_0 = salaryNumericVector_0[shuffledIndices_0]
salaryNumericVector_1 = salaryNumericVector_1[shuffledIndices_1]
X_0=X_0[shuffledIndices_0, :]
X_1=X_1[shuffledIndices_1, :]
y_0=np.zeros(numberOfDataPointsToVisualize)
y_1=np.zeros(numberOfDataPointsToVisualize)+1

#concatanate the matrices and vectors for employees who left and are still there
X=np.concatenate((X_0, X_1), axis=0)
y=np.concatenate((y_0, y_1), axis=0)
salesNumericVector=np.concatenate((salesNumericVector_0, salesNumericVector_1), axis=0)
salaryNumericVector=np.concatenate((salaryNumericVector_0, salaryNumericVector_1), axis=0)

#handle NaNs
imp = Imputer(missing_values='NaN' , strategy = 'mean' , axis = 0)
X = imp.fit_transform(X)

#Standardize data set
X_original = X
mean_X =np.mean(X , axis=0);
std_X =np.std(X , axis=0);
X  = np.divide((X- mean_X) , std_X)

#Apply TSNE to data set
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X[ : , :]);
#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=y , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: left')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=salesNumericVector , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Department ')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=salaryNumericVector , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Salary')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=X_original[:,0] , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Satisfaction')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=X_original[:,1] , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Last Evaluation ')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=X_original[:,2] , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Number of Projects ')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=X_original[:,3] , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Average Monthly Hours ')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=X_original[:,4] , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Time Spent Working ')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=X_original[:,5] , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Work Accident ')
plt.show()


#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=X_original[:,6] , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Promotion ')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=X_original[:,7] , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Projects Per Year ')
plt.show()
#Apply TSNE to data set
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X[ : , [0,1,2,3,4,5,6,7,18,19,20] ]);
#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=y , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: left')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=salesNumericVector , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Department ')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=salaryNumericVector , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Salary')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=X_original[:,0] , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Satisfaction')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=X_original[:,1] , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Last Evaluation ')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=X_original[:,2] , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Number of Projects ')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=X_original[:,3] , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Average Monthly Hours ')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=X_original[:,4] , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Time Spent Working ')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=X_original[:,5] , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Work Accident ')
plt.show()


#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=X_original[:,6] , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Promotion ')
plt.show()

#visualize data
fig = plt.figure();
plt.scatter(X_tsne[:,0], X_tsne[:,1] , c=X_original[:,7] , cmap = "jet_r"  ,  s = [80 for n in range(len(y))]);
plt.xlabel('1');
plt.ylabel('2');
plt.colorbar()
fig.suptitle('TSNE Visualization: Projects Per Year ')
plt.show()
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score

#set up a pipeline where the data matrix is first standardized so that each column has zero mean and unit variance
#than a random forest classifier is fit
pipeline_steps = [('scaler', preprocessing.StandardScaler()), ('clf', RandomForestClassifier())]
pipe = Pipeline(pipeline_steps)

#Set the number of trees in the forest to 500, use a large number of trees
pipe.set_params(clf__n_estimators=500)

#Run 10 fold cross validation and print the cross-validation accuracy
#do not use the features: 'Work_accident,'promotion_last_5years' and the department of each employee
accuracy_CV = cross_val_score(pipe, X=dataMatrix[:,[0,1,2,3,4,7,18,19,20]], y=labels, cv=10)
print('The 10 fold cross-validation accuracy score of the classifier is: '+ str(accuracy_CV.mean()))