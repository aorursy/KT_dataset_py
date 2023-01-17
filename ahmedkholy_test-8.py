import  numpy as np 
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns; sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier 

from sklearn.metrics import mean_squared_error
from math import sqrt
data=pd.read_csv('../input/bankmarketing/bank_marketing_dataset.csv',header=None ,low_memory=False  )
data.columns=data.iloc[0]
data=data.drop(0)
print(data.head(10))
print(data.isnull().values.any(),'\n')
print('\n',data.info(),'\n' ,data.describe())    
le = preprocessing.LabelEncoder()
for col in data.columns:
       data[col] = le.fit_transform(data[col])
print(data.head(10))
subdata=data.loc[:,['age' ,'day','y' ]]
fig, ax = plt.subplots(figsize=(10,10))   

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f' ,vmin=0, vmax=1 );


X=data.drop(columns=['age','marital','education','default','balance','month','campaign','y'],axis=1)
y=data['y']
#Standard Scaler for Data
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler=MinMaxScaler(feature_range=(0,10))
# X = scaler.fit_transform(X)
X_scaled = pd.DataFrame(scaler.fit_transform(X),columns = X.columns)
X_scaled.head(10)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=44, shuffle =True)
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)
SelectedModel = SVC(gamma='auto')
SelectedParameters = {'kernel':('linear', 'rbf'), 'C':[1,2,3,4,5]}


GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters, cv = 2,return_train_score=True)
GridSearchModel.fit(X_train, y_train)
sorted(GridSearchModel.cv_results_.keys())
GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]

print('All Results are :\n', GridSearchResults )
print('===========================================')
print('Best Score is :', GridSearchModel.best_score_)
print('===========================================')
print('Best Parameters are :', GridSearchModel.best_params_)
print('===========================================')
print('Best Estimator is :', GridSearchModel.best_estimator_)
SVCModel =  GridSearchModel.best_estimator_
SVCModel.fit(X_train, y_train)
print('SVCModel Train Score is : ' , SVCModel.score(X_train, y_train))
print('SVCModel Test Score is : ' , SVCModel.score(X_test, y_test))
print('----------------------------------------------------')


y_pred = SVCModel.predict(X_test)
print('Predicted Value for SVCModel is : ' , y_pred[:10])
SelectedModel = LogisticRegression(penalty='l2' , solver='sag',random_state=33)
SelectedParameters = {'C':[1,2,3,4,5]}


GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters, cv = 4,return_train_score=True)
GridSearchModel.fit(X_train, y_train)
sorted(GridSearchModel.cv_results_.keys())
GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]
print('All Results are :\n', GridSearchResults )
print('===========================================')
print('Best Score is :', GridSearchModel.best_score_)
print('===========================================')
print('Best Parameters are :', GridSearchModel.best_params_)
print('===========================================')
print('Best Estimator is :', GridSearchModel.best_estimator_)

DTModel_ = DecisionTreeClassifier(criterion = 'entropy',max_depth=3,random_state = 33)
GaussianNBModel_ = GaussianNB()
BernoulliNBModel_ = BernoulliNB(alpha = 0.1)
MultinomialNBModel_= MultinomialNB(alpha = 0.1)
SGDModel_ = SGDClassifier(loss='log', penalty='l2', max_iter=10000, tol=1e-5)
#loading Voting Classifier
VotingClassifierModel = VotingClassifier(estimators=[('DTModel',DTModel_),('GaussianNBModel',GaussianNBModel_),
                                                     ('BernoulliNBModel',BernoulliNBModel_),
                                                     ('MultinomialNBModel',MultinomialNBModel_),
                                                     ('SGDModel',SGDModel_)], voting='hard')
VotingClassifierModel.fit(X_train, y_train)
#Calculating Details
print('VotingClassifierModel Train Score is : ' , VotingClassifierModel.score(X_train, y_train))
print('VotingClassifierModel Test Score is : ' , VotingClassifierModel.score(X_test, y_test))
print('----------------------------------------------------')
test=X_test.join(y_test )
final_result = SVCModel.predict(X_test)
final_result

final_result =GridSearchModel.predict(X_test)
rms = sqrt(mean_squared_error(y_test, final_result))
rms