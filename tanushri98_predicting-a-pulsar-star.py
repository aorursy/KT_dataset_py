import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np   
import pandas as pd    
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix,plot_confusion_matrix

#Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn import svm
ps_train = pd.read_csv("/kaggle/input/predicting-pulsar-starintermediate/pulsar_data_train.csv")
ps_test = pd.read_csv("/kaggle/input/predicting-pulsar-starintermediate/pulsar_data_test.csv")
ps_train.head(10) #train set
ps_test.head(10) #test set
print("Train dataset shape: ",ps_train.shape)
print("Test dataset shape: ",ps_test.shape)
print("Proportion of training data: %.2f" % (ps_train.shape[0]/(ps_train.shape[0] + ps_test.shape[0])*100), '%')
print("Proportion of test data: %.2f" % (ps_test.shape[0]/(ps_train.shape[0] + ps_test.shape[0])*100), '%')
ps_train.info() #train set
ps_test.info() #test set
ps_train.describe() #train set
ps_test.describe() #test set
print('For Train set')
for feature in ps_train.columns:
    print('Missing values in feature ' + str(feature) + ' : ' + str(len(ps_train[ps_train[feature].isnull() == True])))

print('\n')
print('For Test set')
for feature in ps_test.columns:
    print('Missing values in feature ' + str(feature) + ' : ' + str(len(ps_test[ps_test[feature].isnull() == True])))
print('There are total 12528 observations\n')
print("Total missing values for train set:")
print('Excess kurtosis of the integrated profile: ', 1735)
print('Standard deviation of the DM-SNR curve: ', 1178)
print('Skewness of the DM-SNR curve: ', 625 ,"\n")

print("Total % of missing values for train set:")
print('Excess kurtosis of the integrated profile: ', round(((1735/12528)*100),2),'%')
print('Standard deviation of the DM-SNR curve: ', round(((1178/12528)*100),2),'%')
print('Skewness of the DM-SNR curve:: ', round(((625/12528)*100),2),'%')
print('There are total 5370 observations\n')
print("Total missing values for test set:")
print('Excess kurtosis of the integrated profile: ', 767)
print('Standard deviation of the DM-SNR curve: ', 524)
print('Skewness of the DM-SNR curve: ', 244 ,"\n")

print("Total % of missing values for test set:")
print('Excess kurtosis of the integrated profile: ', round(((767/5370)*100),2),'%')
print('Standard deviation of the DM-SNR curve: ', round(((524/5370)*100),2),'%')
print('Skewness of the DM-SNR curve:: ', round(((244/5370)*100),2),'%')
print("Duplicate rows in training data: ", ps_train.duplicated().sum())
print("Duplicate rows in test data: ", ps_test.duplicated().sum())
ps_train.target_class.value_counts(1)
ps_train['target_class'].value_counts()
sns.countplot(ps_train.target_class)
plt.figure(figsize=(15,10))
plt.title('With Outliers',fontsize=16)
ps_train.boxplot(vert=0)
# Number of outliers
Q1 = ps_train.quantile(0.25)
Q3 = ps_train.quantile(0.75)
IQR = Q3 - Q1
lower_range= Q1-(1.5 * IQR)
upper_range= Q3+(1.5 * IQR)
print('Number of Outliers:')
((ps_train < (lower_range)) | (ps_train > (upper_range))).sum()
print('Percentage of Outliers:')
((((ps_train < (lower_range)) | (ps_train > (upper_range))).sum())/17898)*100
fig, axes = plt.subplots(nrows=8,ncols=2,  figsize=(15, 40))
fig.subplots_adjust(hspace = .4, wspace=.2)
for i in range(0,len(ps_train.columns)-1):
  sns.distplot(ps_train[ps_train.columns[i]], ax=axes[i][0]).set_title("Hisotogram of" + ps_train.columns[i])
  sns.boxplot(ps_train[ps_train.columns[i]], ax=axes[i][1]).set_title("Boxplot of" + ps_train.columns[i])
ps_train_pairplot = ps_train.copy() #creating deep copy
ps_train_pairplot = ps_train_pairplot.rename(columns={' Mean of the integrated profile': 'Mean IP', ' Standard deviation of the integrated profile': 'SD IP',
                                  ' Excess kurtosis of the integrated profile': 'EK IP',' Skewness of the integrated profile': 'Skewness IP',
                                  ' Mean of the DM-SNR curve':'Mean DM-SNR',' Standard deviation of the DM-SNR curve': 'SD DM-SNR',
                                  ' Excess kurtosis of the DM-SNR curve': 'EK DM-SNR', ' Skewness of the DM-SNR curve': 'Skewness DM-SNR'})
# Changing column names for better pairplot visualization
sns.pairplot(data = ps_train_pairplot,hue = 'target_class',corner = True) #,height = 3,aspect = 1.2
plt.figure(figsize=(12,7))
sns.heatmap(ps_train.corr(), annot=True, fmt='.2f', cmap='Blues',mask=np.triu(ps_train.corr(),+1))
ps_train_1 = ps_train[ps_train.target_class == 1] #creating a dataset for only true pulsars for EDA
ps_train_0 = ps_train[ps_train.target_class == 0] #creating a dataset for only non pulsars for EDA
ps_train_1.shape
ps_train_1.info()
ps_train_1.isnull().sum()
print("Total % of missing values:")
print('Excess kurtosis of the integrated profile: ', round(((158/1153)*100),2),'%')
print('Standard deviation of the DM-SNR curve: ', round(((105/1153)*100),2),'%')
print('Skewness of the DM-SNR curve:: ', round(((62/1153)*100),2),'%')
ps_train_1.describe()
fig, axes = plt.subplots(nrows=8,ncols=2,  figsize=(15, 40))
fig.subplots_adjust(hspace = .4, wspace=.2)
for i in range(0,len(ps_train_1.columns)-1):
  sns.distplot(ps_train_1[ps_train_1.columns[i]], ax=axes[i][0]).set_title("Hisotogram of" + ps_train_1.columns[i])
  sns.boxplot(ps_train_1[ps_train_1.columns[i]], ax=axes[i][1]).set_title("Boxplot of" + ps_train_1.columns[i])
ps_train_0.info()
ps_train_0.isnull().sum()
print("Total % of missing values:")
print('Excess kurtosis of the integrated profile: ', round(((1577/11375)*100),2),'%')
print('Standard deviation of the DM-SNR curve: ', round(((1073/11375)*100),2),'%')
print('Skewness of the DM-SNR curve:: ', round(((563/11375)*100),2),'%')
ps_train_0.describe()
fig, axes = plt.subplots(nrows=8,ncols=2,  figsize=(15, 40))
fig.subplots_adjust(hspace = .4, wspace=.2)
for i in range(0,len(ps_train_0.columns)-1):
  sns.distplot(ps_train_0[ps_train_0.columns[i]], ax=axes[i][0]).set_title("Hisotogram of" + ps_train_0.columns[i])
  sns.boxplot(ps_train_0[ps_train_0.columns[i]], ax=axes[i][1]).set_title("Boxplot of" + ps_train_0.columns[i])
fig=plt.figure(figsize=(24,12))
for i in range(0,len(ps_train.columns)-1):
    ax=fig.add_subplot(2,4,i+1).set_title(ps_train.columns[i])
    sns.boxplot(x = 'target_class', y = ps_train.columns[i], data = ps_train)
    plt.grid()
print('There are total 12528 observations\n')
print("Total missing values for train set:")
print('Excess kurtosis of the integrated profile: ', 1735)
print('Standard deviation of the DM-SNR curve: ', 1178)
print('Skewness of the DM-SNR curve: ', 625 ,"\n")

print("Total % of missing values for train set:")
print('Excess kurtosis of the integrated profile: ', round(((1735/12528)*100),2),'%')
print('Standard deviation of the DM-SNR curve: ', round(((1178/12528)*100),2),'%')
print('Skewness of the DM-SNR curve:: ', round(((625/12528)*100),2),'%')
ps_train_median = ps_train.fillna(value=ps_train[[' Excess kurtosis of the integrated profile',' Standard deviation of the DM-SNR curve',' Skewness of the DM-SNR curve']].median())
ps_train_median.isnull().sum() #null values successfully imputed
ps_train_median.describe()
ps_train.describe() #of original train set
ps_train_LR = ps_train.copy() #new dataframe for linear regression method
# Preparing data for modelling

#dataframe with only sets of required columns with no missing values

# For Excess kurtosis of the integrated profile
df1 = ps_train.dropna(axis = 0, subset = [' Excess kurtosis of the integrated profile',' Skewness of the integrated profile']) 
df1 = df1.loc[:,[' Excess kurtosis of the integrated profile',' Skewness of the integrated profile']] 
df1_miss = pd.DataFrame(ps_train[' Skewness of the integrated profile'][ps_train[' Excess kurtosis of the integrated profile'].isnull()])


# For Standard deviation of the DM-SNR curve
df2 = ps_train.dropna(axis = 0, subset = [' Standard deviation of the DM-SNR curve',' Mean of the DM-SNR curve'])
df2 = df2.loc[:,[' Standard deviation of the DM-SNR curve',' Mean of the DM-SNR curve']]
df2_miss = pd.DataFrame(ps_train[' Mean of the DM-SNR curve'][ps_train[' Standard deviation of the DM-SNR curve'].isnull()])

# For Skewness of the DM-SNR curve
df3 = ps_train.dropna(axis = 0, subset = [' Skewness of the DM-SNR curve',' Excess kurtosis of the DM-SNR curve'])
df3 = df3.loc[:,[' Skewness of the DM-SNR curve',' Excess kurtosis of the DM-SNR curve']]
df3_miss = pd.DataFrame(ps_train[' Excess kurtosis of the DM-SNR curve'][ps_train[' Skewness of the DM-SNR curve'].isnull()])
#Creating X and Y variables for each of them

# For Excess kurtosis of the integrated profile
X1 = df1[[' Skewness of the integrated profile']]
y1 = df1[' Excess kurtosis of the integrated profile']   # to be predicted

# For Standard deviation of the DM-SNR curve
X2 = df2[[' Mean of the DM-SNR curve']]
y2 = df2[' Standard deviation of the DM-SNR curve']

# For Skewness of the DM-SNR curve
X3 = df3[[' Excess kurtosis of the DM-SNR curve']]
y3 = df3[' Skewness of the DM-SNR curve']
# Importing required libraries for model building
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

#Creating train test split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.30 , random_state=1)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.30 , random_state=1)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.30 , random_state=1)

# Fitting in the model
lm1,lm2,lm3 = LinearRegression().fit(X1_train, y1_train),LinearRegression().fit(X2_train, y2_train),LinearRegression().fit(X3_train, y3_train)
# Finding the coefficient and intercepts in each case m = coefficient, c = intercept
m1,m2,m3 = lm1.coef_[0],lm2.coef_[0],lm3.coef_[0]
c1,c2,c3 = lm1.intercept_,lm2.intercept_,lm3.intercept_
# Creating dummy columns in dataset where if there is a null value, value is 0 otherwise 1
ps_train_LR['value1'] = ps_train[' Excess kurtosis of the integrated profile'].map(lambda x : 0 if np.isnan(x) else 1)
ps_train_LR['value2'] = ps_train[' Standard deviation of the DM-SNR curve'].map(lambda x : 0 if np.isnan(x) else 1)
ps_train_LR['value3'] = ps_train[' Skewness of the DM-SNR curve'].map(lambda x : 0 if np.isnan(x) else 1)
# Using y = mx + c method to predict missing values and imputing at that location
for i in range(0,len(ps_train_LR)-1):
    if(ps_train_LR.value1[i] == 0):
        ps_train_LR[' Excess kurtosis of the integrated profile'][i] = c1 + m1 * ps_train_LR[' Skewness of the integrated profile'][i] 
    if(ps_train_LR.value2[i] == 0):
        ps_train_LR[' Standard deviation of the DM-SNR curve'][i] = c2 + m2 * ps_train_LR[' Mean of the DM-SNR curve'][i] 
    if(ps_train_LR.value3[i] == 0):
        ps_train_LR[' Skewness of the DM-SNR curve'][i] = c3 + m3 * ps_train_LR[' Excess kurtosis of the DM-SNR curve'][i] 
ps_train_LR.drop(['value1', 'value2','value3'], axis = 1,inplace = True) # dropping dummy variables as not required anymore
ps_train_LR.isnull().sum()
ps_train_LR.describe()
ps_train.describe() # original dataset
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=3)

X = ps_train.drop('target_class', axis=1)
y = ps_train[['target_class']]

ps_train_KNN = imputer.fit_transform(X)

ps_train_KNN =   pd.DataFrame(data=ps_train_KNN,columns=X.columns)
ps_train_KNN['target_class'] = ps_train['target_class']
ps_train_KNN.isnull().sum()
ps_train_KNN.describe()
ps_train.describe()
print('Percentage of Outliers:')
((((ps_train < (lower_range)) | (ps_train > (upper_range))).sum())/17898)*100
ps_train_out = ps_train_KNN.copy() #creating dataframe without outliers using missing value treated dataframe

def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range
for column in ps_train_out.loc[:,:' Skewness of the DM-SNR curve'].columns: #I didn't took the target class here
    if ps_train_out[column].dtype != 'object':
        lr,ur=remove_outlier(ps_train_out[column])
        ps_train_out[column]=np.where(ps_train_out[column]>ur,ur,ps_train_out[column])
        ps_train_out[column]=np.where(ps_train_out[column]<lr,lr,ps_train_out[column])
plt.figure(figsize=(10,8))
plt.title('Without Outliers',fontsize=16)
ps_train_out.boxplot(vert=0)
# Correlation plot after outlier treatment
plt.figure(figsize=(12,8))
sns.heatmap(ps_train_out.corr(),annot=True, cmap='Blues',mask=np.triu(ps_train_out.corr(),+1))
plt.figure(figsize=(12,8))
sns.heatmap(ps_train_out.drop(' Standard deviation of the DM-SNR curve',axis=1).corr(),annot=True, cmap='Blues',mask=np.triu(ps_train_out.drop([' Standard deviation of the DM-SNR curve'],axis=1).corr(),+1))
plt.figure(figsize=(12,8))
sns.heatmap(ps_train_out.drop([' Standard deviation of the DM-SNR curve',' Excess kurtosis of the DM-SNR curve'],axis=1).corr(),annot=True, cmap='Blues',mask=np.triu(ps_train_out.drop([' Standard deviation of the DM-SNR curve',' Excess kurtosis of the DM-SNR curve'],axis=1).corr(),+1))
# VIF of all variables
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = ps_train_out.drop('target_class', axis=1)
vif = [variance_inflation_factor(X.values, ix) for ix in range(X.shape[1])] 
i=0
for column in X.columns:
  print (column ,"--->",  vif[i])
  i = i+1
# VIF after removing Excess kurtosis of the DM-SNR curve and Standard deviation of the DM-SNR curve
X1 = ps_train_out.drop(['target_class',' Excess kurtosis of the DM-SNR curve',' Standard deviation of the DM-SNR curve'], axis=1)

i=0
for column in X1.columns:
  print (column ,"--->",  vif[i])
  i = i+1
X = ps_train_out.drop('target_class',axis=1)
y = ps_train_out[['target_class']]
from sklearn.model_selection import train_test_split

# Keeping test size as 30% of dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler() 
# Scaling the data
X_trains = ss.fit_transform(X_train)
X_tests = ss.transform(X_test)
# Building base Model
DT = DecisionTreeClassifier(random_state=1) #random state given for consistency in results
DT.fit(X_trains, y_train)
# Classification Report
print('Classification Report of the training data:\n\n',metrics.classification_report(y_train, DT.predict(X_trains)),'\n')
print('Classification Report of the test data:\n\n',metrics.classification_report(y_test, DT.predict(X_tests)),'\n')
# Mean 10 fold cross validation scores for train and test set
print('Train set CV scores: %0.4f'%np.mean(cross_val_score(DT, X_train, y_train, cv=10)),'\n')
print('Test set CV scores: %0.4f'%np.mean(cross_val_score(DT, X_test, y_test, cv=10)))
DT_param_random = [    
    {'splitter' : ['best', 'random'],
     'max_features' : list(range(1,X_train.shape[1])),
     'max_depth' : np.linspace(1, 32, 32, endpoint=True),
     'min_samples_leaf' : randint(1, 1000), # 1-3% of length of dataset
     'min_samples_split' : randint(300, 5000), # approx 3 times the min_samples_leaf
     "criterion": ["gini", "entropy"]
    }
]

DT_random1 = RandomizedSearchCV(DT, param_distributions = DT_param_random, cv = 5, verbose=True, n_jobs=-1)
DT_random1.fit(X_trains, y_train)
# Checking the best estimator values
DT_random1.best_estimator_
# Best Random Model
DT_random = DecisionTreeClassifier(max_depth=5.0,max_features=4,min_samples_leaf=525, min_samples_split=2593,criterion='entropy',random_state=1)
DT_random.fit(X_trains, y_train)
# Classification Report
print('Classification Report of the training data:\n\n',metrics.classification_report(y_train, DT_random.predict(X_trains)),'\n')
print('Classification Report of the test data:\n\n',metrics.classification_report(y_test, DT_random.predict(X_tests)),'\n')
# Mean 10 fold cross validation scores for train and test set
print('Train set CV scores: %0.4f'%np.mean(cross_val_score(DT_random, X_train, y_train, cv=10)),'\n')
print('Test set CV scores: %0.4f'%np.mean(cross_val_score(DT_random, X_test, y_test, cv=10)))
DT_param_grid = [    
    {'splitter' : ['best', 'random'],
     'max_features' : [4,5,6],
     'max_depth' : [20,30,40],
     'min_samples_leaf' : [100,200,300], # 1-3% of total dataset
     'min_samples_split' : [300, 400, 500], # approx 3 times the min_samples_leaf
     "criterion": ["gini", "entropy"]
    }
]

DT_grid1 = GridSearchCV(DT, param_grid = DT_param_grid, cv = 5, verbose=False, n_jobs=-1)
DT_grid1.fit(X_trains, y_train) 
DT_grid1.best_estimator_
# Best Grid Model
DT_grid = DecisionTreeClassifier(max_depth=20,max_features=4,min_samples_leaf=100, min_samples_split=300,criterion='entropy',random_state=1)
DT_grid.fit(X_trains, y_train)
# Classification Report
print('Classification Report of the training data:\n\n',metrics.classification_report(y_train, DT_grid.predict(X_trains)),'\n')
print('Classification Report of the test data:\n\n',metrics.classification_report(y_test, DT_grid.predict(X_tests)),'\n')
# Mean 10 fold cross validation scores for train and test set
print('Train set CV scores: %0.4f'%np.mean(cross_val_score(DT_grid, X_train, y_train, cv=10)),'\n')
print('Test set CV scores: %0.4f'%np.mean(cross_val_score(DT_grid, X_test, y_test, cv=10)))
# Prediction for final model
DT_train_predict = DT_grid.predict(X_trains)
DT_test_predict = DT_grid.predict(X_tests)

# Probability Prediction for final model
DT_prob_train = DT_grid.predict_proba(X_trains)
DT_prob_test = DT_grid.predict_proba(X_tests)

# AUC score for final model
DT_train_auc = metrics.roc_auc_score(y_train,DT_prob_train[:,1])
DT_test_auc = metrics.roc_auc_score(y_test,DT_prob_test[:,1])
# Classification report in a dataframe of final model
DT_df_train=pd.DataFrame(classification_report(y_train, DT_train_predict,output_dict=True)).transpose()
DT_df_test=pd.DataFrame(classification_report(y_test, DT_test_predict,output_dict=True)).transpose()
# Building base Model
RF = RandomForestClassifier(random_state=1) 
RF.fit(X_trains, y_train)
# Classification Report
print('Classification Report of the training data:\n\n',metrics.classification_report(y_train, RF.predict(X_trains)),'\n')
print('Classification Report of the test data:\n\n',metrics.classification_report(y_test, RF.predict(X_tests)),'\n')
# Mean 10 fold cross validation scores for train and test set
print('Train set CV scores: %0.4f'%np.mean(cross_val_score(RF, X_train, y_train, cv=10)),'\n')
print('Test set CV scores: %0.4f'%np.mean(cross_val_score(RF, X_test, y_test, cv=10)))
RF_param_random = [    
    {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
     'max_features' : ['auto', 'sqrt'],
     'max_depth' : np.linspace(1, 32, 32, endpoint=True),
     'min_samples_leaf' : randint(1, 300), # 1-3% of length of dataset
     'min_samples_split' : randint(300, 3000), # approx 3 times the min_samples_leaf
     'bootstrap': [True, False]
    }
]

RF_random1 = RandomizedSearchCV(RF, param_distributions = RF_param_random, cv = 5, n_jobs=-1)
RF_random1.fit(X_trains, y_train)
RF_random1.best_estimator_
# Best Random Model
RF_random = RandomForestClassifier(max_depth=20.0,max_features='sqrt',min_samples_leaf=66, min_samples_split=511,criterion='gini',n_estimators=1400,random_state=1)
RF_random.fit(X_trains, y_train)
# Classification Report
print('Classification Report of the training data:\n\n',metrics.classification_report(y_train, RF_random.predict(X_trains)),'\n')
print('Classification Report of the test data:\n\n',metrics.classification_report(y_test, RF_random.predict(X_tests)),'\n')
# Mean 10 fold cross validation scores for train and test set
print('Train set CV scores: %0.4f'%np.mean(cross_val_score(RF_random, X_train, y_train, cv=10)),'\n')
print('Test set CV scores: %0.4f'%np.mean(cross_val_score(RF_random, X_test, y_test, cv=10)))
RF_param_grid = [    
    {'n_estimators': [300,400],
     'max_features' : [3,4],
     'max_depth' : [15,20],
     'min_samples_leaf' : [100,200], # 1-3% of length of dataset
     'min_samples_split' : [300,400] # approx 3 times the min_samples_leaf
    }
]

RF_grid1 = GridSearchCV(RF, param_grid = RF_param_grid, cv = 3, n_jobs=-1)
RF_grid1.fit(X_trains, y_train) 
best_estimator_
# Best Grid Model
RF_grid = RandomForestClassifier(max_depth=15,max_features=3,min_samples_leaf=100, min_samples_split=300,criterion='gini',n_estimators=300,random_state=1)
RF_grid.fit(X_trains, y_train)
# Classification Report
print('Classification Report of the training data:\n\n',metrics.classification_report(y_train, RF_grid.predict(X_trains)),'\n')
print('Classification Report of the test data:\n\n',metrics.classification_report(y_test, RF_grid.predict(X_tests)),'\n')
# Mean 10 fold cross validation scores for train and test set
print('Train set CV scores: %0.4f'%np.mean(cross_val_score(RF_grid, X_train, y_train, cv=10)),'\n')
print('Test set CV scores: %0.4f'%np.mean(cross_val_score(RF_grid, X_test, y_test, cv=10)))
# Prediction for final model
RF_train_predict = RF_random.predict(X_trains)
RF_test_predict = RF_random.predict(X_tests)

# Probability Prediction for final model
RF_prob_train = RF_random.predict_proba(X_trains)
RF_prob_test = RF_random.predict_proba(X_tests)

# AUC score for final model
RF_train_auc = metrics.roc_auc_score(y_train,RF_prob_train[:,1])
RF_test_auc = metrics.roc_auc_score(y_test,RF_prob_test[:,1])
# Classification report in a dataframe of final model
RF_df_train=pd.DataFrame(classification_report(y_train, RF_train_predict,output_dict=True)).transpose()
RF_df_test=pd.DataFrame(classification_report(y_test, RF_test_predict,output_dict=True)).transpose()
# Building base model
LR = LogisticRegression(random_state = 1)
LR.fit(X_trains, y_train)
# Classification Report
print('Classification Report of the training data:\n\n',metrics.classification_report(y_train, LR.predict(X_trains)),'\n')
print('Classification Report of the test data:\n\n',metrics.classification_report(y_test, LR.predict(X_tests)),'\n')
# Mean 10 fold cross validation scores for train and test set
print('Train set CV scores: %0.4f'%np.mean(cross_val_score(LR, X_train, y_train, cv=10)),'\n')
print('Test set CV scores: %0.4f'%np.mean(cross_val_score(LR, X_test, y_test, cv=10)))
LR_param_random = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]

LR_random1 = RandomizedSearchCV(LR, param_distributions = LR_param_random, cv = 5, n_jobs=-1)
LR_random1.fit(X_trains,y_train)
LR_random1.best_estimator_
# Best Random model
LR_random = LogisticRegression(C=1.623776739188721,penalty='none', solver='newton-cg',max_iter=100, random_state = 1)
LR_random.fit(X_trains,y_train)
# Classification Report
print('Classification Report of the training data:\n\n',metrics.classification_report(y_train, LR_random.predict(X_trains)),'\n')
print('Classification Report of the test data:\n\n',metrics.classification_report(y_test, LR_random.predict(X_tests)),'\n')
# Mean 10 fold cross validation scores for train and test set
print('Train set CV scores: %0.4f'%np.mean(cross_val_score(LR_random, X_train, y_train, cv=10)),'\n')
print('Test set CV scores: %0.4f'%np.mean(cross_val_score(LR_random, X_test, y_test, cv=10)))
LR_param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : [0.00001],
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]

LR_grid1 = GridSearchCV(LR, param_grid = LR_param_grid, cv = 5, n_jobs=-1)
LR_grid1.fit(X_trains, y_train) 
LR_grid1.best_estimator_
# Best Grid model
LR_grid = LogisticRegression(C=1e-05,penalty='none', solver='lbfgs',max_iter=100, random_state = 1)
LR_grid.fit(X_trains,y_train)
# Classification Report
print('Classification Report of the training data:\n\n',metrics.classification_report(y_train, LR_grid.predict(X_trains)),'\n')
print('Classification Report of the test data:\n\n',metrics.classification_report(y_test, LR_grid.predict(X_tests)),'\n')
# Mean 10 fold cross validation scores for train and test set
print('Train set CV scores: %0.4f'%np.mean(cross_val_score(LR_grid, X_train, y_train, cv=10)),'\n')
print('Test set CV scores: %0.4f'%np.mean(cross_val_score(LR_grid, X_test, y_test, cv=10)))
# Prediction for final model
LR_train_predict = LR_random.predict(X_trains)
LR_test_predict = LR_random.predict(X_tests)

# Probability Prediction for final model
LR_prob_train = LR_random.predict_proba(X_trains)
LR_prob_test = LR_random.predict_proba(X_tests)

# AUC score for final model
LR_train_auc = metrics.roc_auc_score(y_train,LR_prob_train[:,1])
LR_test_auc = metrics.roc_auc_score(y_test,LR_prob_test[:,1])
# Classification report in a dataframe of final model
LR_df_train=pd.DataFrame(classification_report(y_train, LR_train_predict,output_dict=True)).transpose()
LR_df_test=pd.DataFrame(classification_report(y_test, LR_test_predict,output_dict=True)).transpose()
# Building base Model
NB = GaussianNB()
NB.fit(X_trains, y_train)
# Classification Report
print('Classification Report of the training data:\n\n',metrics.classification_report(y_train, NB.predict(X_trains)),'\n')
print('Classification Report of the test data:\n\n',metrics.classification_report(y_test, NB.predict(X_tests)),'\n')
# Mean 10 fold cross validation scores for train and test set
print('Train set CV scores: %0.4f'%np.mean(cross_val_score(NB, X_trains, y_train, cv=10)),'\n')
print('Test set CV scores: %0.4f'%np.mean(cross_val_score(NB, X_tests, y_test, cv=10)))
# Prediction for final model
NB_train_predict = NB.predict(X_trains)
NB_test_predict = NB.predict(X_tests)

# Probability Prediction for final model
NB_prob_train = NB.predict_proba(X_trains)
NB_prob_test = NB.predict_proba(X_tests)

# AUC score for final model
NB_train_auc = metrics.roc_auc_score(y_train,NB_prob_train[:,1])
NB_test_auc = metrics.roc_auc_score(y_test,NB_prob_test[:,1])
# Classification report in a dataframe of final model
NB_df_train=pd.DataFrame(classification_report(y_train, NB_train_predict,output_dict=True)).transpose()
NB_df_test=pd.DataFrame(classification_report(y_test, NB_test_predict,output_dict=True)).transpose()
# Building base Model
XGB=xgb.XGBClassifier(random_state=1)
XGB.fit(X_trains, y_train)
# Classification Report
print('Classification Report of the training data:\n\n',metrics.classification_report(y_train, XGB.predict(X_trains)),'\n')
print('Classification Report of the test data:\n\n',metrics.classification_report(y_test, XGB.predict(X_tests)),'\n')
# Mean 10 fold cross validation scores for train and test set
print('Train set CV scores: %0.4f'%np.mean(cross_val_score(XGB, X_trains, y_train, cv=10)),'\n')
print('Test set CV scores: %0.4f'%np.mean(cross_val_score(XGB, X_tests, y_test, cv=10)))
XGB_param_random = [    
    {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
     'max_depth' : [int(x) for x in np.linspace(1, 32, 32, endpoint=True)],
     'subsample': np.linspace(start = 0.00001, stop = 0.9, num = 20),
      'gamma': np.linspace(start = 0.00001, stop = 0.9, num = 20),
      'colsample_bytree': [0.2,0.4,0.6,0.8,1.0],
     'learning_rate': np.linspace(start = 0.00001, stop = 0.1, num = 20)
    }
]

XGB_random1 = RandomizedSearchCV(XGB, param_distributions = XGB_param_random, cv = 5, n_jobs=-1)
XGB_random1.fit(X_trains, y_train)
XGB_random1.best_estimator_
# Best Random model
XGB_random = xgb.XGBClassifier(colsample_bytree=1.0,gamma=0.09474578947368421,n_estimators=1400,max_depth=15,subsample=0.14211368421052634, learning_rate=0.005272631578947368, random_state=1)
XGB_random.fit(X_trains, y_train)
# Classification Report
print('Classification Report of the training data:\n\n',metrics.classification_report(y_train, XGB_random.predict(X_trains)),'\n')
print('Classification Report of the test data:\n\n',metrics.classification_report(y_test, XGB_random.predict(X_tests)),'\n')
# Mean 10 fold cross validation scores for train and test set
print('Train set CV scores: %0.4f'%np.mean(cross_val_score(XGB_random, X_train, y_train, cv=10)),'\n')
print('Test set CV scores: %0.4f'%np.mean(cross_val_score(XGB_random, X_test, y_test, cv=10)))
# Prediction for final model
XGB_train_predict = XGB.predict(X_trains)
XGB_test_predict = XGB.predict(X_tests)

# Probability Prediction for final model
XGB_prob_train = XGB.predict_proba(X_trains)
XGB_prob_test = XGB.predict_proba(X_tests)

# AUC score for final model
XGB_train_auc = metrics.roc_auc_score(y_train,XGB_prob_train[:,1])
XGB_test_auc = metrics.roc_auc_score(y_test,XGB_prob_test[:,1])
# Classification report in a dataframe of final model
XGB_df_train=pd.DataFrame(classification_report(y_train, XGB_train_predict,output_dict=True)).transpose()
XGB_df_test=pd.DataFrame(classification_report(y_test, XGB_test_predict,output_dict=True)).transpose()
# Building base Model
SVM = svm.SVC(random_state=1).fit(X_trains, y_train)
# Classification Report
print('Classification Report of the training data:\n\n',metrics.classification_report(y_train, SVM.predict(X_trains)),'\n')
print('Classification Report of the test data:\n\n',metrics.classification_report(y_test, SVM.predict(X_tests)),'\n')
# Mean 10 fold cross validation scores for train and test set
print('Train set CV scores: %0.4f'%np.mean(cross_val_score(SVM, X_trains, y_train, cv=10)),'\n')
print('Test set CV scores: %0.4f'%np.mean(cross_val_score(SVM, X_tests, y_test, cv=10)))
SVM_param_random = {'C': [0.01,0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf'],
              'tol':[0.01,0.001,0.0001],
              'degree': [2,3,4,5]}  #,'linear','poly', 'sigmoid', 'precomputed'

SVM_random1 = RandomizedSearchCV(SVM, param_distributions = SVM_param_random, cv = 5, n_jobs=-1)
SVM_random1.fit(X_trains, y_train) 
SVM_random1.best_estimator_
# Best Random model
SVM_random = svm.SVC(C=1,degree=2, gamma=0.1, kernel='rbf',tol=0.001, random_state=1)
SVM_random.fit(X_trains, y_train)
# Classification Report
print('Classification Report of the training data:\n\n',metrics.classification_report(y_train, SVM_random.predict(X_trains)),'\n')
print('Classification Report of the test data:\n\n',metrics.classification_report(y_test, SVM_random.predict(X_tests)),'\n')
# Mean 10 fold cross validation scores for train and test set
print('Train set CV scores: %0.4f'%np.mean(cross_val_score(SVM_random, X_train, y_train, cv=10)),'\n')
print('Test set CV scores: %0.4f'%np.mean(cross_val_score(SVM_random, X_test, y_test, cv=10)))
SVM_param_grid = {'C': [0.01,0.1, 1, 10],  
              'gamma': [0.09, 0.1, 0.2, 0.001], 
              'kernel': ['rbf'],
              'tol':[0.001,0.0001],
              'degree':[2,3]}

SVM_grid1 = GridSearchCV(SVM, param_grid = SVM_param_grid, cv = 5, n_jobs=-1)
SVM_grid1.fit(X_trains, y_train) 
SVM_grid1.best_estimator_
# Best Random model
SVM_grid = svm.SVC(C=10,degree=2, gamma=0.09, kernel='rbf',tol=0.001, random_state=1)
SVM_grid.fit(X_trains, y_train)
# Classification Report
print('Classification Report of the training data:\n\n',metrics.classification_report(y_train, SVM_grid.predict(X_trains)),'\n')
print('Classification Report of the test data:\n\n',metrics.classification_report(y_test, SVM_grid.predict(X_tests)),'\n')
# Mean 10 fold cross validation scores for train and test set
print('Train set CV scores: %0.4f'%np.mean(cross_val_score(SVM_grid, X_train, y_train, cv=10)),'\n')
print('Test set CV scores: %0.4f'%np.mean(cross_val_score(SVM_grid, X_test, y_test, cv=10)))
# Prediction for final model
SVM_train_predict = SVM.predict(X_trains)
SVM_test_predict = SVM.predict(X_tests)
# Classification report in a dataframe of final model
SVM_df_train=pd.DataFrame(classification_report(y_train, SVM_train_predict,output_dict=True)).transpose()
SVM_df_test=pd.DataFrame(classification_report(y_test, SVM_test_predict,output_dict=True)).transpose()
models = {"DT_grid":DT_grid, "RF_random":RF_random, "LR_random":LR_random, "NB":NB, "XGB":XGB, "SVM":SVM}
model = [DT_grid, RF_random, LR_random, NB, XGB, SVM]
model_name = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'Naive Bayes', 'XGBoost', 'SVM']
fig, axes = plt.subplots(nrows=2,ncols=6,  figsize=(25, 20))
fig.subplots_adjust(wspace=0.8)
idx = 0
for m,n in zip(model,model_name):
  plot_confusion_matrix(m,X_trains,y_train,cmap='Greys',display_labels=['Non Pulsars','Pulsars'],values_format = '.0f', ax=axes[0][idx]);
  axes[0][idx].set_title("Train of " + n)
  plot_confusion_matrix(m,X_tests,y_test,cmap='Greys',display_labels=['Non Pulsars','Pulsars'],values_format = '.0f', ax=axes[1][idx]);
  axes[1][idx].set_title("Test of " + n)
  idx=idx+1
# Creating a dataframe with 'Accuracy', 'AUC', 'Recall','Precision','F1 Score' values for all models

index=['Accuracy', 'AUC', 'Recall','Precision','F1 Score']
data = pd.DataFrame({'DT Train':[DT_df_train.loc["accuracy"][0],DT_train_auc,DT_df_train.loc["1.0"][1],DT_df_train.loc["1.0"][0],DT_df_train.loc["1.0"][2]],
                     'DT Test':[DT_df_test.loc["accuracy"][0],DT_test_auc,DT_df_test.loc["1.0"][1],DT_df_test.loc["1.0"][0],DT_df_test.loc["1.0"][2]],
                     'RF Train':[RF_df_train.loc["accuracy"][0],RF_train_auc,RF_df_train.loc["1.0"][1],RF_df_train.loc["1.0"][0],RF_df_train.loc["1.0"][2]],
                     'RF Test':[RF_df_test.loc["accuracy"][0],RF_test_auc,RF_df_test.loc["1.0"][1],RF_df_test.loc["1.0"][0],RF_df_test.loc["1.0"][2]],
                     'LR Train':[LR_df_train.loc["accuracy"][0],LR_train_auc,LR_df_train.loc["1.0"][1],LR_df_train.loc["1.0"][0],LR_df_train.loc["1.0"][2]],
                     'LR Test':[LR_df_test.loc["accuracy"][0],LR_test_auc,LR_df_test.loc["1.0"][1],LR_df_test.loc["1.0"][0],LR_df_test.loc["1.0"][2]],
                     'NB Train':[NB_df_train.loc["accuracy"][0],NB_train_auc,NB_df_train.loc["1.0"][1],NB_df_train.loc["1.0"][0],NB_df_train.loc["1.0"][2]],
                     'NB Test':[NB_df_test.loc["accuracy"][0],NB_test_auc,NB_df_test.loc["1.0"][1],NB_df_test.loc["1.0"][0],NB_df_test.loc["1.0"][2]],
                     'XGB Train':[XGB_df_train.loc["accuracy"][0],XGB_train_auc,XGB_df_train.loc["1.0"][1],XGB_df_train.loc["1.0"][0],XGB_df_train.loc["1.0"][2]],
                     'XGB Test':[XGB_df_test.loc["accuracy"][0],XGB_test_auc,XGB_df_test.loc["1.0"][1],XGB_df_test.loc["1.0"][0],XGB_df_test.loc["1.0"][2]],
                     'SVM Train':[SVM_df_train.loc["accuracy"][0],0,SVM_df_train.loc["1.0"][1],SVM_df_train.loc["1.0"][0],SVM_df_train.loc["1.0"][2]],
                     'SVM Test':[SVM_df_test.loc["accuracy"][0],0,SVM_df_test.loc["1.0"][1],SVM_df_test.loc["1.0"][0],SVM_df_test.loc["1.0"][2]]
                     },index=index)
data = round(data,3)
data
data1 = data.T
data1['Model'] = data1.index 
data1 = data1.reset_index()
# Accuracy of Train
plt.figure(figsize = (15,10))
graph = sns.barplot(x = 'Model', y = 'Accuracy',data = data1.iloc[[0,2,4,6,8,10]])

for p in graph.patches:
        graph.annotate('{:.3f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
# Accuracy of Test
plt.figure(figsize = (15,10))
graph = sns.barplot(x = 'Model', y = 'Accuracy',data = data1.iloc[[1,3,5,7,9,11]])

for p in graph.patches:
        graph.annotate('{:.3f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
# F1-score of Train
plt.figure(figsize = (15,10))
graph = sns.barplot(x = 'Model', y = 'F1 Score',data = data1.iloc[[0,2,4,6,8,10]])

for p in graph.patches:
        graph.annotate('{:.3f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
# F1-score of Test
plt.figure(figsize = (15,10))
graph = sns.barplot(x = 'Model', y = 'F1 Score',data = data1.iloc[[1,3,5,7,9,11]])

for p in graph.patches:
        graph.annotate('{:.3f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
# AUC of Train
plt.figure(figsize = (15,10))
graph = sns.barplot(x = 'Model', y = 'AUC',data = data1.iloc[[0,2,4,6,8]])

for p in graph.patches:
        graph.annotate('{:.3f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
# AUC of Test
plt.figure(figsize = (15,10))
graph = sns.barplot(x = 'Model', y = 'AUC',data = data1.iloc[[1,3,5,7,9]])

for p in graph.patches:
        graph.annotate('{:.3f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
DT_train_fpr, DT_train_tpr, thresholds = metrics.roc_curve(y_train,DT_prob_train[:,1])
DT_test_fpr, DT_test_tpr, thresholds = metrics.roc_curve(y_test,DT_prob_test[:,1])
RF_train_fpr, RF_train_tpr, thresholds = metrics.roc_curve(y_train,RF_prob_train[:,1])
RF_test_fpr, RF_test_tpr, thresholds = metrics.roc_curve(y_test,RF_prob_test[:,1])
LR_train_fpr, LR_train_tpr, thresholds = metrics.roc_curve(y_train,LR_prob_train[:,1])
LR_test_fpr, LR_test_tpr, thresholds = metrics.roc_curve(y_test,LR_prob_test[:,1])
NB_train_fpr, NB_train_tpr, thresholds = metrics.roc_curve(y_train,NB_prob_train[:,1])
NB_test_fpr, NB_test_tpr, thresholds = metrics.roc_curve(y_test,NB_prob_test[:,1])
XGB_train_fpr, XGB_train_tpr, thresholds = metrics.roc_curve(y_train,XGB_prob_train[:,1])
XGB_test_fpr, XGB_test_tpr, thresholds = metrics.roc_curve(y_test,XGB_prob_test[:,1])
plt.figure(figsize=(10,5))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(DT_train_fpr,DT_train_tpr,color='crimson',label="DT")
plt.plot(RF_train_fpr,RF_train_tpr,color='black',label="RF")
plt.plot(LR_train_fpr, LR_train_tpr,color='r',label="LR")
plt.plot(NB_train_fpr,NB_train_tpr,color='c',label="NB")
plt.plot(XGB_train_fpr,XGB_train_tpr,color='gray',label="XGB")

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Training Data')
plt.legend(loc=0)
plt.figure(figsize=(10,5))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(DT_test_fpr,DT_test_tpr,color='crimson',label="DT")
plt.plot(RF_test_fpr,RF_test_tpr,color='black',label="RF")
plt.plot(LR_test_fpr, LR_test_tpr,color='r',label="LR")
plt.plot(NB_test_fpr,NB_test_tpr,color='c',label="NB")
plt.plot(XGB_test_fpr,XGB_test_tpr,color='gray',label="XGB")

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Testing Data')
plt.legend(loc=0)