import pandas as pd # for data handling

import numpy as np # for data manipulation 

import sklearn as sk

from matplotlib import pyplot as plt # for plotting

from sklearn.preprocessing import LabelEncoder # For encoding class variables

from sklearn.model_selection import train_test_split # for train and test split

from sklearn.svm import SVC # to built svm model

from sklearn import svm # inherits other SVM objects

from sklearn import metrics # to calculate classifiers accuracy

from sklearn.model_selection import cross_val_score # to perform cross validation

from sklearn.preprocessing import StandardScaler # to perform standardization

from sklearn.model_selection import GridSearchCV # to perform grid search for all classifiers

from sklearn import tree # to perform decision tree classification

from sklearn import neighbors # to perform knn

from sklearn import naive_bayes # to perform Naive Bayes

from sklearn.metrics import classification_report # produce classifier reports

from sklearn.ensemble import RandomForestClassifier # to perform ensemble bagging - random forest

from sklearn.ensemble import AdaBoostClassifier # to perform ensemble boosting

from sklearn.metrics import roc_curve, auc # to plot ROC Curve

% matplotlib inline
%pwd
%ls
# Reding the data as pandas dataframe

data_raw = pd.read_csv('../input/voice.csv',sep=',')

data_raw.shape
# Verifying if all records are read 

data_raw.head(3)
# having the headers handy

columns = data_raw.columns

print(columns)
# Data type

df = pd.DataFrame(data_raw.dtypes,columns=['Data Type'])

df = df.reset_index()

df.columns = ['Attribute Name','Data Type']

df
# Checking for any missing values in data and other junk values if any

if data_raw.isnull() is True:

    print('There are missing records')

else:

    print('No missing records')
# let us seperate the independent and dependent variables seperately

data_x = data_raw[columns[0:20]].copy()

data_y = data_raw[columns[-1]].copy()

print('Independent var: \n',data_x.head(3),'\n')

print('Dependent var: \n',data_y.head(3))
# encoding the target variable from categorical values to binary form

encode_obj = LabelEncoder()

data_y = encode_obj.fit_transform(data_y)

print('sample values of target values:\n',data_y[0:3])
# Let us do a 80-20 split

test_x_train,test_x_test,test_y_train,test_y_test = train_test_split(data_x,data_y,train_size=0.8,test_size=0.2,random_state=1)
nbclf = naive_bayes.GaussianNB()

nbclf = nbclf.fit(test_x_train, test_y_train)

nbpreds_test = nbclf.predict(test_x_test)

print('Accuracy obtained from train-test split on training data is:',nbclf.score(test_x_train, test_y_train))

print('Accuracy obtained from train-test split on testing data is:',nbclf.score(test_x_test, test_y_test))
test_eval_result = cross_val_score(nbclf, data_x, data_y, cv=10, scoring='accuracy')

print('Accuracy obtained from 10-fold cross validation on actual raw data is:',test_eval_result.mean())
### plotting the independent variables

plt.subplot(221)

plt.hist(data_x['meanfreq'])

plt.subplot(222)

plt.hist(data_x['sd'])

plt.subplot(223)

plt.hist(data_x['median'])

plt.subplot(224)

plt.hist(data_x['Q25'])
plt.subplot(221)

plt.hist(data_x['Q75'])

plt.subplot(222)

plt.hist(data_x['IQR'])

plt.subplot(223)

plt.hist(data_x['skew'])

plt.subplot(224)

plt.hist(data_x['kurt'])
print('Mean and Median value for Q75 is: ',[data_x.Q75.mean(), data_x.Q75.median()])

print('Mean and Median value for IQR is: ',[data_x.IQR.mean(), data_x.IQR.median()])
plt.subplot(221)

plt.hist(data_x['sp.ent'])

plt.subplot(222)

plt.hist(data_x['sfm'])

plt.subplot(223)

plt.hist(data_x['mode'])

plt.subplot(224)

plt.hist(data_x['centroid'])
print('Mean and Median value for Mode is: ',[data_x['mode'].mean(), data_x['mode'].median()])
plt.subplot(221)

plt.hist(data_x['meanfun'])

plt.subplot(222)

plt.hist(data_x['minfun'])

plt.subplot(223)

plt.hist(data_x['maxfun'])

plt.subplot(224)

plt.hist(data_x['meandom'])
plt.subplot(221)

plt.hist(data_x['mindom'])

plt.subplot(222)

plt.hist(data_x['maxdom'])

plt.subplot(223)

plt.hist(data_x['dfrange'])

plt.subplot(224)

plt.hist(data_x['modindx'])
# let us do a descriptive statistics

means = data_x.describe().loc['mean']

medians = data_x.describe().loc['50%']

pd.DataFrame([means,medians], index=['mean','median'])
# Distribution of target variables

print(pd.Series(data_y).value_counts())

pd.Series(data_y).value_counts().plot(kind='bar', title='Bar graph of Number of male and female users')
# Actual Raw Data size

data_raw.shape
# Filtering ouliers from male category

male_funFreq_outlier_index = data_raw[((data_raw['meanfun'] < 0.085) | (data_raw['meanfun'] > 0.180)) & 

                                      (data_raw['label'] == 'male')].index

male_funFreq_outlier_index = list(male_funFreq_outlier_index)

data_raw[((data_raw['meanfun'] < 0.085) | (data_raw['meanfun'] > 0.180)) & (data_raw['label'] == 'male')].shape
# Filtering ouliers from female category

female_funFreq_outlier_index = data_raw[((data_raw['meanfun'] < 0.165) | (data_raw['meanfun'] > 0.255)) & 

                                        (data_raw['label'] == 'female')].index

female_funFreq_outlier_index = list(female_funFreq_outlier_index)

data_raw[((data_raw['meanfun'] < 0.165) | (data_raw['meanfun'] > 0.255)) & (data_raw['label'] == 'female')].shape
index_to_remove = male_funFreq_outlier_index + female_funFreq_outlier_index

len(index_to_remove)
# Thus, we need to remove 710 rows from both data_x and data_y using the index obtained from above filters

# Preparing final dataset for model building

data_x = data_x.drop(index_to_remove,axis=0)

data_x.shape
# Target dataset

data_y = pd.Series(data_y).drop(index_to_remove,axis=0)

data_y.shape
# Distribution of target variables after cleanup

print(data_y.value_counts())

data_y.value_counts().plot(kind='bar', title='Target variable after cleanup (1/0=Male/Female)')
# Z-score Normalization

z_score_norm = lambda colname: (data_x[colname]- data_x[colname].mean())/(data_x[colname].std())

min_max_norm = lambda colname: (data_x[colname]- data_x[colname].min())/(data_x[colname].max()-data_x[colname].min())
data_x1 = data_x.copy()

data_x1['z_meanfreq'] = z_score_norm('meanfreq')

data_x1['z_median'] = z_score_norm('median')

data_x1['z_Q25'] = z_score_norm('Q25')

data_x1['z_Q75'] = z_score_norm('Q75')

data_x1['Norm_IQR'] = min_max_norm('IQR')
# Lets now drop the original column from data_x as we have these as backup in data_raw dataframe

data_x1 = data_x1.drop(['meanfreq','median','Q25','Q75','IQR'],axis=1)
data_x1.head(3)
# Plotting the normalized columns

# we could see that z-score norm variables have mean 0 and standard deviation 1

# And the min-max norm varibales value are confined between 0-1 and stays positive

plt.subplot(231)

plt.hist(data_x1['z_meanfreq'])

plt.subplot(232)

plt.hist(data_x1['z_median'])

plt.subplot(233)

plt.hist(data_x1['z_Q25'])

plt.subplot(234)

plt.hist(data_x1['z_Q75'])

plt.subplot(235)

plt.hist(data_x1['Norm_IQR'])
# let us see the correlation in data

corr_mat = data_x1.corr()

corr_mat
for names in corr_mat.index:

    if len(corr_mat[(corr_mat.loc[names] > 0.9) & (corr_mat.loc[names].index != names)].index) > 0:

        print('column', names,' correlates strongly with: ',corr_mat[(corr_mat.loc[names] > 0.9) & 

                                                                     (corr_mat.loc[names].index != names)].index)
corr_df = pd.DataFrame([{'Column Name':'skew', 'Correlated with':'kurt'},

                        {'Column Name':'kurt', 'Correlated with':'skew'},

                        {'Column Name':'centroid', 'Correlated with':['z_meanfreq', 'z_median', 'z_Q25']},

                        {'Column Name':'maxdom', 'Correlated with':['dfrange']},

                        {'Column Name':'dfrange', 'Correlated with':['maxdom']},

                        {'Column Name':'z_meanfreq', 'Correlated with':['centroid', 'z_median', 'z_Q25']},

                        {'Column Name':'z_median', 'Correlated with':['centroid', 'z_meanfreq']},

                        {'Column Name':'z_Q25', 'Correlated with':['centroid', 'z_meanfreq']},

                        ])

corr_df
# Thus we see high correlation exist between above variables, 

# thus let us create a dataset by removing variables that create high Variance Inflation Factor

# Thus, removing kurt, Centroid, dfrange, z_meanfreq

data_x2 = data_x1.drop(['kurt', 'centroid', 'dfrange', 'z_meanfreq'],axis=1).copy()

data_x2.head(3)
# let me not do any dimentionality reduction and do z-score normalization on all independent variables

xDataStdardized = StandardScaler()

xDataStdardized.fit(data_x)

data_x3 = xDataStdardized.transform(data_x).copy()
columns[0:20]
data_x3 = pd.DataFrame(data_x3, columns=columns[0:20])

data_x3.head(3)
# Let us do a 80-20 split on raw dataset

data_x_train,data_x_test,data_y_train,data_y_test = train_test_split(data_x,data_y,train_size=0.8,test_size=0.2,random_state=1)
# let us do a 80-20 split on dimention reduced dataset too

data_x2_train,data_x2_test,data_y2_train,data_y2_test=train_test_split(data_x2,data_y,train_size=0.8,test_size=0.2,random_state=1)
# let us do a 80-20 split on raw dataset which was only normalized

data_x3_train,data_x3_test,data_y3_train,data_y3_test=train_test_split(data_x3,data_y,train_size=0.8,test_size=0.2,random_state=1)
# let us check the size

data_x_train.shape
data_x_test.shape
data_y_train.shape
# let is cross check the size of dimention reduced data set too 

data_x2_train.shape
data_x2_test.shape
# let is cross check the size of normalized raw data set too 

data_x3_train.shape
data_x3_test.shape
# defining the Naive Bayes object

nbclf = naive_bayes.GaussianNB()
# lets do a 10 fold Cross validation to make sure the accuracy obtained above

nbclf = nbclf.fit(data_x_train, data_y_train)

nbpreds_test = nbclf.predict(data_x_test)

nb_eval_result1 = cross_val_score(nbclf, data_x, data_y, cv=10, scoring='accuracy')

print('Mean accuracy with 10 fold cross validation on Naive Bayes with treated data: ',nb_eval_result1.mean())
# lets do a 10 fold Cross validation to make sure the accuracy obtained above

nbclf = nbclf.fit(data_x2_train, data_y2_train)

nbpreds_test = nbclf.predict(data_x2_test)

nb_eval_result2 = cross_val_score(nbclf, data_x2, data_y, cv=10, scoring='accuracy')

print('Mean accuracy with 10 fold cross validation on Naive Bayes with dimention reduced data: ',nb_eval_result2.mean())
# lets do a 10 fold Cross validation to make sure the accuracy obtained above

nbclf = nbclf.fit(data_x3_train, data_y3_train)

nbpreds_test = nbclf.predict(data_x3_test)

nb_eval_result3 = cross_val_score(nbclf, data_x3, data_y, cv=10, scoring='accuracy')

print('Mean accuracy with 10 fold cross validation on Naive Bayes with Normalized data: ',nb_eval_result3.mean())
validation_result = pd.DataFrame([{'Dataset':'Partially Normalized', 'Accuracy':nb_eval_result1.mean()},

                                    {'Dataset':'Dimention Reduced', 'Accuracy':nb_eval_result2.mean()},

                                    {'Dataset':'Completely Normalized', 'Accuracy':nb_eval_result3.mean()}], 

                                 columns=['Dataset','Accuracy'])

validation_result
def funct_svm(kernal_type,xTrain,yTrain,xTest,yTest):

    svm_obj=SVC(kernel=kernal_type)

    svm_obj.fit(xTrain,yTrain)

    yPredicted=svm_obj.predict(xTest)

    print('Accuracy Score of',kernal_type,'Kernal SVM is:',metrics.accuracy_score(yTest,yPredicted))

    return metrics.accuracy_score(yTest,yPredicted)
# Partially normlized dataset

%timeit 10

PN_linear_result = funct_svm('linear',data_x_train,data_y_train,data_x_test,data_y_test)
# Dimention reduced dataset

%timeit 10

DR_linear_result = funct_svm('linear',data_x2_train,data_y2_train,data_x2_test,data_y2_test)
# Completely normalized dataset

%timeit 10

CN_linear_result = funct_svm('linear',data_x3_train,data_y3_train,data_x3_test,data_y3_test)
linear_kernal_result = pd.DataFrame([{'Dataset':'Partially Normalized', 'Accuracy':PN_linear_result},

                                    {'Dataset':'Dimention Reduced', 'Accuracy':DR_linear_result},

                                    {'Dataset':'Completely Normalized', 'Accuracy':CN_linear_result}], columns=['Dataset','Accuracy'])

linear_kernal_result
# Partially normlized dataset

%timeit 10

PN_rbf_result = funct_svm('rbf',data_x_train,data_y_train,data_x_test,data_y_test)
# Dimention reduced dataset

%timeit 10

DR_rbf_result = funct_svm('rbf',data_x2_train,data_y2_train,data_x2_test,data_y2_test)
# Completely normalized dataset

%timeit 10

CN_rbf_result = funct_svm('rbf',data_x3_train,data_y3_train,data_x3_test,data_y3_test)
gausian_kernal_result = pd.DataFrame([{'Dataset':'Partially Normalized', 'Accuracy':PN_rbf_result},

                                    {'Dataset':'Dimention Reduced', 'Accuracy':DR_rbf_result},

                                    {'Dataset':'Completely Normalized', 'Accuracy':CN_rbf_result}], columns=['Dataset','Accuracy'])

gausian_kernal_result
# Partially normlized dataset

%timeit 10

PN_poly_result = funct_svm('poly',data_x_train,data_y_train,data_x_test,data_y_test)
# Dimentione reduced dataset

%timeit 10

DR_poly_result = funct_svm('poly',data_x2_train,data_y2_train,data_x2_test,data_y2_test)
# Completely normalized dataset

%timeit 10

CN_poly_result = funct_svm('poly',data_x3_train,data_y3_train,data_x3_test,data_y3_test)
poly_kernal_result = pd.DataFrame([{'Dataset':'Partially Normalized', 'Accuracy': PN_poly_result},

                                    {'Dataset':'Dimention Reduced', 'Accuracy':DR_poly_result},

                                    {'Dataset':'Completely Normalized', 'Accuracy':CN_poly_result}], columns=['Dataset','Accuracy'])

poly_kernal_result
# Partially normlized dataset

%timeit 10

PN_sigmoid_result = funct_svm('sigmoid',data_x_train,data_y_train,data_x_test,data_y_test)
# Dimentione reduced dataset

%timeit 10

DR_sigmoid_result = funct_svm('sigmoid',data_x2_train,data_y2_train,data_x2_test,data_y2_test)
# Completely normalized dataset

%timeit 10

CN_sigmoid_result = funct_svm('sigmoid',data_x3_train,data_y3_train,data_x3_test,data_y3_test)
sigmoid_kernal_result = pd.DataFrame([{'Dataset':'Partially Normalized', 'Accuracy':PN_sigmoid_result},

                                    {'Dataset':'Dimention Reduced', 'Accuracy':DR_sigmoid_result},

                                    {'Dataset':'Completely Normalized', 'Accuracy':CN_sigmoid_result}], columns=['Dataset','Accuracy'])

sigmoid_kernal_result
kernal_result = pd.DataFrame([{'Dataset':'Completely Normalized','Kernal':'Linear', 'Accuracy':CN_linear_result},

                            {'Dataset':'Completely Normalized','Kernal':'Gaussian', 'Accuracy':CN_rbf_result},

                            {'Dataset':'Completely Normalized','Kernal':'Polynomial', 'Accuracy':CN_poly_result}, 

                            {'Dataset':'Completely Normalized','Kernal':'Sigmoidal', 'Accuracy':CN_sigmoid_result}], 

                             columns=['Dataset','Kernal','Accuracy'])

kernal_result
def funct_svm_cv(kernal_type,xData,yData,k,eval_param):

    svm_obj=SVC(kernel=kernal_type)

    eval_result = cross_val_score(svm_obj, xData, yData, cv=k, scoring=eval_param)

    print(eval_param,'of each fold is:',eval_result)

    print('Mean accuracy with 10 fold cross validation for',kernal_type,' kernal SVM is: ',eval_result.mean())

    return eval_result.mean()
# Partially normlized dataset

%timeit 10

PN_CV_linear_result = funct_svm_cv('linear',data_x,data_y,10,'accuracy')
# Dimentione reduced dataset

%timeit 10

DR_CV_linear_result = funct_svm_cv('linear',data_x2,data_y,10,'accuracy')
# Completely normalized dataset

%timeit 10

CN_CV_linear_result = funct_svm_cv('linear',data_x3,data_y,10,'accuracy')
cv_linear_kernal_result = pd.DataFrame([{'Dataset':'Partially Normalized', 'Accuracy':PN_CV_linear_result},

                                    {'Dataset':'Dimention Reduced', 'Accuracy':DR_CV_linear_result},

                                    {'Dataset':'Completely Normalized', 'Accuracy':CN_CV_linear_result}], columns=['Dataset','Accuracy'])

cv_linear_kernal_result
# Partially normlized dataset

%timeit 10

PN_CV_rbf_result = funct_svm_cv('rbf',data_x,data_y,10,'accuracy')
# Dimentione reduced dataset

%timeit 10

DR_CV_rbf_result = funct_svm_cv('rbf',data_x2,data_y,10,'accuracy')
# Completely normalized dataset

%timeit 10

CN_CV_rbf_result = funct_svm_cv('rbf',data_x3,data_y,10,'accuracy')
cv_rbf_kernal_result = pd.DataFrame([{'Dataset':'Partially Normalized', 'Accuracy':PN_CV_rbf_result},

                                    {'Dataset':'Dimention Reduced', 'Accuracy':DR_CV_rbf_result},

                                    {'Dataset':'Completely Normalized', 'Accuracy':CN_CV_rbf_result}], columns=['Dataset','Accuracy'])

cv_rbf_kernal_result
# Partially normlized dataset

%timeit 10

PN_CV_sigmoid_result = funct_svm_cv('sigmoid',data_x,data_y,10,'accuracy')
# Dimentione reduced dataset

%timeit 10

DR_CV_sigmoid_result = funct_svm_cv('sigmoid',data_x2,data_y,10,'accuracy')
# Completely normalized dataset

%timeit 10

CN_CV_sigmoid_result = funct_svm_cv('sigmoid',data_x3,data_y,10,'accuracy')
cv_sigmoid_kernal_result = pd.DataFrame([{'Dataset':'Partially Normalized', 'Accuracy':PN_CV_sigmoid_result},

                                    {'Dataset':'Dimention Reduced', 'Accuracy':DR_CV_sigmoid_result},

                                    {'Dataset':'Completely Normalized', 'Accuracy':CN_CV_sigmoid_result}], columns=['Dataset','Accuracy'])

cv_sigmoid_kernal_result
cv_kernal_result = pd.DataFrame([{'Dataset':'Completely Normalized','Kernal':'Linear', 'Accuracy':CN_CV_linear_result},

                            {'Dataset':'Completely Normalized','Kernal':'Gaussian', 'Accuracy':CN_CV_rbf_result},

                            {'Dataset':'Completely Normalized','Kernal':'Polynomial', 'Accuracy':CN_poly_result}, 

                            {'Dataset':'Completely Normalized','Kernal':'Sigmoidal', 'Accuracy':CN_CV_sigmoid_result}], 

                             columns=['Dataset','Kernal','Accuracy'])

cv_kernal_result
# penality parameter C is 1.0 by default in sklearn

# I would like to experiment it with multiple margins in range of c from 1 to 10

def funct_tune_svm(kernal_type,margin_val,xData,yData,k,eval_param):

    if(kernal_type=='linear'):

        svm_obj=SVC(kernel=kernal_type,C=margin_val)

    elif(kernal_type=='rbf'):

        svm_obj=SVC(kernel=kernal_type,gamma=margin_val)

    elif(kernal_type=='poly'):

        svm_obj=SVC(kernel=kernal_type,degree=margin_val) 

    eval_result = cross_val_score(svm_obj, xData, yData, cv=k, scoring=eval_param)

    return eval_result.mean()
# Completely normlized dataset

accu_list = list()

for c in np.arange(0.1,10,0.5):

    result = funct_tune_svm('linear',c,data_x3,data_y,5,'accuracy')

    accu_list.append(result)
C_values=np.arange(0.1,10,0.5)

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(C_values,accu_list)

plt.xticks(np.arange(0.1,10,0.5))

plt.xlabel('Value of C for SVC')

plt.ylabel('Cross-Validated Accuracy')
tuning_linear_svm = pd.DataFrame(columns=['Penality Parameter C', 'Accuracy'])

tuning_linear_svm['Penality Parameter C'] = np.arange(0.1,10,0.5)

tuning_linear_svm['Accuracy'] = accu_list

tuning_linear_svm
# Completely normlized dataset

accu_list = list()

for c in np.arange(0.1,10,1):

    result = funct_tune_svm('rbf',c,data_x3,data_y,5,'accuracy')

    accu_list.append(result)
C_values=list(range(0,10))

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(C_values,accu_list)

plt.xticks(np.arange(0,10,1))

plt.xlabel('Value of Gamma for SVC')

plt.ylabel('Cross-Validated Accuracy')
tuning_rbf_svm = pd.DataFrame(columns=['Parameter Gamma', 'Accuracy'])

tuning_rbf_svm['Parameter Gamma'] = np.arange(0.1,10,1)

tuning_rbf_svm['Accuracy'] = accu_list
tuning_rbf_svm
# Doing further tradeoff

accu_list = list()

for c in np.arange(0.001,0.01,0.001):

    result = funct_tune_svm('rbf',c,data_x3,data_y,5,'accuracy')

    accu_list.append(result)



C_values=list(np.arange(0.001,0.01,0.001))

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(C_values,accu_list)

plt.xticks(np.arange(0.001,0.01,0.001))

plt.xlabel('Value of Gamma for SVC')

plt.ylabel('Cross-Validated Accuracy')
tuning_rbf_svm = pd.DataFrame(columns=['Parameter Gamma', 'Accuracy'])

tuning_rbf_svm['Parameter Gamma'] = np.arange(0.001,0.01,0.001)

tuning_rbf_svm['Accuracy'] = accu_list

tuning_rbf_svm
# Completely normlized dataset

accu_list = list()

for c in np.arange(0.1,10,1):

    result = funct_tune_svm('poly',c,data_x3,data_y,5,'accuracy')

    accu_list.append(result)
np.arange(0.1,10,1)
C_values=list(np.arange(0.1,10,1))

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(C_values,accu_list)

plt.xticks(np.arange(0.1,10,1))

plt.xlabel('Value of C for SVC')

plt.ylabel('Cross-Validated Accuracy')
tuning_poly_svm = pd.DataFrame(columns=['Parameter Degree', 'Accuracy'])

tuning_poly_svm['Parameter Degree'] = np.arange(0.1,10,1)

tuning_poly_svm['Accuracy'] = accu_list

tuning_poly_svm
# Now performing SVM by taking hyperparameter C=0.1 and kernel as linear

svc=SVC(kernel='linear',C=0.6)

scores = cross_val_score(svc, data_x3, data_y, cv=10, scoring='accuracy')

print(scores.mean())
# With rbf gamma value = 0.01

svc= SVC(kernel='rbf',gamma=0.005)

svc.fit(data_x3_train,data_y3_train)

y_predict=svc.predict(data_x3_test)

metrics.accuracy_score(data_y3_test,y_predict)
# performing grid search with different tuning parameters

svm_obj= SVC()

grid_parameters = {

 'C': [0.1,0.6,1.1,1.6] , 'kernel': ['linear'],

 'C': [0.1,0.6,1.1,1.6] , 'gamma': [0.002,0.003,0.004,0.005], 'kernel': ['rbf'],

 'degree': [1,2,3] ,'gamma':[0.002,0.003,0.004,0.005], 'C':[0.1,0.6,1.1,1.6] , 'kernel':['poly']

                   }

model_svm = GridSearchCV(svm_obj, grid_parameters,cv=10,scoring='accuracy')

model_svm.fit(data_x3_train, data_y3_train)

print(model_svm.best_score_)

print(model_svm.best_params_)

y_pred= model_svm.predict(data_x3_test)
svm_performance = metrics.accuracy_score(y_pred,data_y3_test)

svm_performance
gridSearch_kernal_result = pd.DataFrame([{'kernel': 'poly', 'gamma': 0.005, 'degree': 1, 'C': 1.6}],

                                       columns=['kernel','C','gamma','degree'])

gridSearch_kernal_result
# Scatter plot with strong correlation - not useful much to represnt the distribution wrt kernal boundries

plt.scatter(data_raw['meanfreq'],data_raw['centroid'])
# Scatter plot with weak correlation - not useful much to represnt the distribution wrt kernal boundries

plt.scatter(data_raw['modindx'],data_raw['minfun'])
# Scatter plot with moderate correlation - useful much to represnt the distribution wrt kernal boundries

plt.scatter(data_raw['dfrange'],data_raw['centroid'])
# Scatter plot with moderate negative correlation - useful much to represnt the distribution wrt kernal boundries

plt.scatter(data_raw['meanfun'],data_raw['sp.ent'])
# import some data to play with

X = data_x3[['meanfun','sp.ent']].copy()

X = np.array(X)

y = np.array(data_y)



# fit the model, don't regularize for illustration purposes

clf = SVC(kernel='poly', degree=1.1, gamma = 0.05,C=1.6)

clf.fit(X, y)



# title for the plots

title = ('SVC with poly kernel(with degree=1.1 & gamma=0.05 & C=1.6)')



plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)



# plot the decision function

ax = plt.gca()

xlim = ax.get_xlim()

ylim = ax.get_ylim()



# create grid to evaluate model

xx = np.linspace(xlim[0], xlim[1], 30)

yy = np.linspace(ylim[0], ylim[1], 30)

YY, XX = np.meshgrid(yy, xx)

xy = np.vstack([XX.ravel(), YY.ravel()]).T

Z = clf.decision_function(xy).reshape(XX.shape)



# plot decision boundary and margins

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,

           linestyles=['--', '-', '--'])

# plot support vectors

ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,

           linewidth=1, facecolors='none')

ax.set_xlabel('meanfun')

ax.set_ylabel('sp.ent')

ax.set_title(title)

plt.show()
def make_meshgrid(x, y, h=.02):

    """Create a mesh of points to plot in



    Parameters

    ----------

    x: data to base x-axis meshgrid on

    y: data to base y-axis meshgrid on

    h: stepsize for meshgrid, optional



    Returns

    -------

    xx, yy : ndarray

    """

    x_min, x_max = x.min() - 1, x.max() + 1

    y_min, y_max = y.min() - 1, y.max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                         np.arange(y_min, y_max, h))

    return xx, yy





def plot_contours(ax, clf, xx, yy, **params):

    """Plot the decision boundaries for a classifier.



    Parameters

    ----------

    ax: matplotlib axes object

    clf: a classifier

    xx: meshgrid ndarray

    yy: meshgrid ndarray

    params: dictionary of params to pass to contourf, optional

    """

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    out = ax.contourf(xx, yy, Z, **params)

    return out



# import some data to play with

X = data_x3[['meanfun','sp.ent']].copy()

X = np.array(X)

y = np.array(data_y)



C = 1.6  # SVM regularization parameter

models = (SVC(kernel='linear', C=C),

          svm.LinearSVC(C=C),

          SVC(kernel='rbf', gamma=0.005, C=C),

          SVC(kernel='poly', degree=1, gamma=0.005, C=C))

models = (clf.fit(X, y) for clf in models)



# title for the plots

titles = ('SVC with linear kernel (C=1.6)',

          'LinearSVC (linear kernel)',

          'RBF kernel(gamma=0.005)',

          'Polynomial (degree 1)')



# Set-up 2x2 grid for plotting.

fig, sub = plt.subplots(2, 2)

plt.subplots_adjust(wspace=0.4, hspace=0.4)



X0, X1 = X[:, 0], X[:, 1]

xx, yy = make_meshgrid(X0, X1)



for clf, title, ax in zip(models, titles, sub.flatten()):

    plot_contours(ax, clf, xx, yy,

                  cmap=plt.cm.coolwarm, alpha=0.8)

    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    ax.set_xlim(xx.min(), xx.max())

    ax.set_ylim(yy.min(), yy.max())

    ax.set_xlabel('meanfun')

    ax.set_ylabel('sp.ent')

    ax.set_xticks(())

    ax.set_yticks(())

    ax.set_title(title)



plt.show()
dt = tree.DecisionTreeClassifier()

parameters = {

    'criterion': ['entropy','gini'],

    'max_depth': np.linspace(1, 20, 10),

    #'min_samples_leaf': np.linspace(1, 30, 15),

    #'min_samples_split': np.linspace(2, 20, 10)

}

gs = GridSearchCV(dt, parameters, verbose=0, cv=5)

gs.fit(data_x3_train, data_y3_train)

gs.best_params_, gs.best_score_
def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True):

    y_pred = clf.predict(X)   

    if show_accuracy:

         print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)),"\n")

    if show_classification_report:

        print("Classification report")

        print(metrics.classification_report(y, y_pred),"\n")

      

    if show_confussion_matrix:

        print("Confussion matrix")

        print(metrics.confusion_matrix(y, y_pred),"\n")
dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=7)

dt.fit(data_x3_train, data_y3_train)

measure_performance(data_x3_test, data_y3_test, dt, show_confussion_matrix=False, show_classification_report=True)
dt_performance = dt.score(data_x3_test, data_y3_test)

dt_performance
# lets do a 10 fold Cross validation to make sure the accuracy obtained above

dt_eval_result = cross_val_score(dt, data_x3, data_y, cv=10, scoring='accuracy')

print('Mean accuracy with 10 fold cross validation for Decision tree is: ',dt_eval_result.mean())
n_neighbors = 5

knnclf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

knnclf.fit(data_x3_train, data_y3_train)
knnpreds_test = knnclf.predict(data_x3_test)
print(knnclf.score(data_x3_test, data_y3_test))
print(classification_report(data_y3_test, knnpreds_test))
knn_performance = knnclf.score(data_x3_test, data_y3_test)
# lets do a 10 fold Cross validation to make sure the accuracy obtained above

knn_eval_result = cross_val_score(knnclf, data_x3, data_y, cv=10, scoring='accuracy')

print('Mean accuracy with 10 fold cross validation for KNN is: ',knn_eval_result.mean())
final_resutls = pd.DataFrame(columns=['Classifier Name', 'Performance in terms of Accuracy'])
final_resutls['Classifier Name'] = ['SVM','Decision Tree','KNN','Naive Bayes']

final_resutls['Performance in terms of Accuracy'] = [svm_performance, dt_eval_result.mean(), 

                                                     knn_eval_result.mean(),nb_eval_result3.mean()]
final_resutls
final_resutls.plot.line(x=final_resutls['Classifier Name'])
# Applying Random forest to improve the decision tree model

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='entropy',max_depth=7)

rf_model = rf.fit(data_x3_train, data_y3_train)
rfpreds_test = rf_model.predict(data_x3_test)

rf_performance = rf_model.score(data_x3_test, data_y3_test)
print(rf_performance)
# lets do a 10 fold Cross validation to make sure the accuracy obtained above

rf_eval_result = cross_val_score(rf_model, data_x3, data_y, cv=10, scoring='accuracy')

print('Mean accuracy with 10 fold cross validation for KNN is: ',rf_eval_result.mean())
# adaboost

adaBoost = AdaBoostClassifier()

adaboost_model = adaBoost.fit(data_x3_train, data_y3_train)
adboostpreds_test = adaboost_model.predict(data_x3_test)

adaboost_performance = adaboost_model.score(data_x3_test, data_y3_test)
print(adaboost_performance)
# lets do a 10 fold Cross validation to make sure the accuracy obtained above

adaboost_eval_result = cross_val_score(adaboost_model, data_x3, data_y, cv=10, scoring='accuracy')

print('Mean accuracy with 10 fold cross validation for KNN is: ',adaboost_eval_result.mean())
final_report = pd.DataFrame(columns=['Classifier Name', 'Performance in terms of Accuracy'])

final_report['Classifier Name'] = ['SVM','AdaBoost','Random Forest','Decision Tree','KNN','Naive Bayes']

final_report['Performance in terms of Accuracy'] = [svm_performance, adaboost_eval_result.mean(),rf_eval_result.mean(),

                                                    dt_eval_result.mean(),

                                                    knn_eval_result.mean(),nb_eval_result3.mean()]

final_report
final_report.plot.line(x=final_report['Classifier Name'])
# Building the ROC Curve for the final SVM Kernal model

final_model = SVC(kernel='poly', C=1.6, gamma=0.005, degree=1)

print('Final Model Detail:\n',final_model)

final_model_score = final_model.fit(data_x3_train, data_y3_train).decision_function(data_x3_test)

# CV Accuracy 

final_eval_result = cross_val_score(final_model, data_x3, data_y, cv=10, scoring='accuracy')

print('\nAccuracy obtained from final model with 10 fold CV:\n',final_eval_result.mean())

# ROC measure

fpr, tpr, _ = roc_curve(data_y3_test,final_model_score)

roc_auc= auc(fpr, tpr)

print('\nROC Computed Area Under Curve:\n',roc_auc)
plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic for Best SVM model')

plt.legend(loc="lower right")

plt.show()