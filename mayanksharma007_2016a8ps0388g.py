#from google.colab import drive

#drive.mount('/content/drive/')

#ls "/content/drive/My Drive/ML Labs/Evaluative_Lab_1"
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

#df=pd.read_csv("/content/drive/My Drive/ML Labs/Evaluative_Lab_1/train.csv")

df=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")

df.head()

df.info()
df.describe()

df.isnull().sum()
df.fillna(value=df.mean(),inplace=True)

df.isnull().sum()

df.isnull().any().any()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_error



corr = df.corr()

print(corr['rating'])

sns.heatmap(corr)
print(df.var())
#from sklearn.decomposition import PCA

#X_std = StandardScaler().fit_transform(df)

#df1 = PCA(n_components=8)

#pca.fit_transform()

#df1.head()
#df.drop(['C', 'D'], axis = 1)

df=df.drop(['id'],axis=1)

df.head(10)

numerical_features = ['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11']

categorical_features = ['type']

X = df[numerical_features+categorical_features]

y = df["rating"]
df['type'].unique()

X=pd.get_dummies(X,['type'])

print(X.head())

X=X.drop(['type_old'],axis=1)



X['feature3']=np.log(X['feature3'])

X['feature5']=np.log(X['feature5'])

X.head()
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.33,random_state=42)  #Checkout what does random_state do

#TODO

from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

#df_row = pd.concat([df1, df2])

X_train=pd.concat([X_train,X_val])

y_train=pd.concat([y_train,y_val])

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])

#X_val[numerical_features] = scaler.transform(X_val[numerical_features])  



# It is important to scale tain and val data separately because val is supposed to be unseen data on which we test our models. If we scale them together, data from val set will also be considered while calculating mean, median, IQR, etc



X_train[numerical_features].head()
X_train.info()
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

#TODO

clf = ExtraTreesRegressor()        #Initialize the classifier object





parameters = {'n_estimators':[3048],'min_samples_leaf':[1],'max_features':['auto','sqrt'],'min_samples_split':[2],'bootstrap':[False]}    #Dictionary of parameters



scorer = make_scorer(mean_squared_error, greater_is_better = False)#Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf,parameters,scoring=scorer,n_jobs=-1)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

#grid_obj = RandomizedSearchCV(clf,parameters,scoring=scorer,n_jobs=-1)

grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train



best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



#unoptimized_predictions = (clf.fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions

#optimized_predictions = best_clf.predict(X_val)        #Same, but use the best estimator



#acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

#acc_opd = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model



#print("Accuracy score on unoptimized model:{}".format(acc_unop))

#print("Accuracy score on optimized model:{}".format(acc_op))
print(best_clf)
from sklearn.metrics import mean_squared_error

#for i in range(len(optimized_predictions)):

#  optimized_predictions[i]=round(optimized_predictions[i])

#mse = mean_squared_error(y_val, optimized_predictions)

#opt3=[]

#print(np.sqrt(mse))

#for i in range(len(optimized_predictions)):

#  opt3.append(round(optimized_predictions[i]))

#mse2=mean_squared_error(y_val,opt3)

#print(np.sqrt(mse2))

#from sklearn.metrics import mean_squared_error

#for i in range(len(optimized_predictions)):

#  optimized_predictions[i]=round(optimized_predictions[i])

#for i in range(len(opt3)):

#  if opt3[i]<0.6:

#    opt3[i]=0

#  elif opt3[i]>5.3:

#    opt3[i]=6

#  else:

#    opt3[i]=round(opt3[i])

#

#mse = mean_squared_error(y_val, opt3)

#np.sqrt(mse)

#print(np.unique(opt3))

df4=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")

df4.fillna(value=df4.mean(),inplace=True)

df4.isnull().sum()

t=df4['id']

df4=df4.drop(['id'],axis=1)

df4.head(10)

numerical_features = ['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11']

categorical_features = ['type']

X_test = df4[numerical_features+categorical_features]

X_test=pd.get_dummies(X_test,['type'])

X_test=X_test.drop(['type_old'],axis=1)

X_test['feature3']=np.log(X_test['feature3'])

X_test['feature5']=np.log(X_test['feature5'])

X_test[numerical_features] = scaler.transform(X_test[numerical_features])



#optimized_predictions = best_clf.predict(X_val)

#opt=best_clf.predict(X_test)

X_test.isnull().sum()
opt=best_clf.predict(X_test)

print(opt)



df4.info()

print(np.max(opt))

print(len(opt))

for i in range(len(opt)):

  opt[i]=round(opt[i])


df3=pd.DataFrame({"id":t,"rating":opt})

#df3.to_csv('/content/drive/My Drive/ML Labs/Evaluative_Lab_1/final26.csv',index=False)

df3.to_csv('final26.csv',index=False)

#df = pd.DataFrame({"Index":[i+1 for i in range(len(label))],"Sentiment":label})

#df.to_csv('/content/drive/My Drive/Colab Notebooks/evaluation/sol130.csv',index=False)-v2
#X_test.head()
import pandas as pd

import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error, make_scorer

from scipy.stats import skew

from IPython.display import display

import matplotlib.pyplot as plt

import seaborn as sns

#def rmse_cv_train(model):

#    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))

#    return(rmse)#

#

#def rmse_cv_test(model):

#    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))

#    return(rmse)

#elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],

#                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 

#                                    0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 

#                          max_iter = 50000, cv = 10)

#elasticNet.fit(X_train, y_train)

#alpha = elasticNet.alpha_

#ratio = elasticNet.l1_ratio_

#print("Best l1_ratio :", ratio)

#print("Best alpha :", alpha )

#

#print("Try again for more precision with l1_ratio centered around " + str(ratio))

#elasticNet = ElasticNetCV(l1_ratio = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],

#                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 

#                          max_iter = 50000, cv = 10)

#elasticNet.fit(X_train, y_train)

#if (elasticNet.l1_ratio_ > 1):

#    elasticNet.l1_ratio_ = 1    

#alpha = elasticNet.alpha_

#ratio = elasticNet.l1_ratio_

#print("Best l1_ratio :", ratio)

#print("Best alpha :", alpha )

#

#

#print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) + 

#      " and alpha centered around " + str(alpha))

#elasticNet = ElasticNetCV(l1_ratio = ratio,

#                          alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9, 

#                                    alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, 

#                                    alpha * 1.35, alpha * 1.4], 

#                          max_iter = 50000, cv = 10)

#elasticNet.fit(X_train, y_train)

#if (elasticNet.l1_ratio_ > 1):

#    elasticNet.l1_ratio_ = 1    

#alpha = elasticNet.alpha_

#ratio = elasticNet.l1_ratio_

#print("Best l1_ratio :", ratio)

#print("Best alpha :", alpha )

#

#

#print("ElasticNet RMSE on Training set :", rmse_cv_train(elasticNet).mean())

#print("ElasticNet RMSE on Test set :", rmse_cv_test(elasticNet).mean())

#y_train_ela = elasticNet.predict(X_train)

#y_test_ela = elasticNet.predict(X_test)

#mse = mean_squared_error(y_train_ela, y_train)

#np.sqrt(mse)
#df3=pd.DataFrame({"id":df2['id'],"rating":y_test_ela})

#df3.to_csv('/content/drive/My Drive/ML Labs/Evaluative_Lab_1/m2.csv',index=False)
#Model for final25.csv

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

#import warnings



warnings.filterwarnings('ignore')
df=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")

df.head()
df.info()
df.describe()
df.isnull().sum()
df.fillna(value=df.mean(),inplace=True)

df.isnull().sum()
df.isnull().any().any()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_error



corr = df.corr()

print(corr['rating'])

sns.heatmap(corr)
print(df.var())
#from sklearn.decomposition import PCA

#X_std = StandardScaler().fit_transform(df)

#df1 = PCA(n_components=8)

#pca.fit_transform()

#df1.head()
#df.drop(['C', 'D'], axis = 1)

df=df.drop(['id'],axis=1)

df.head(10)
numerical_features = ['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11']

categorical_features = ['type']

X = df[numerical_features+categorical_features]

y = df["rating"]
df['type'].unique()

X=pd.get_dummies(X,['type'])

print(X.head())

X=X.drop(['type_old'],axis=1)

X.head()

X['feature3']=np.log(X['feature3'])

X['feature5']=np.log(X['feature5'])
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.33,random_state=42)  #Checkout what does random_state do

from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

#df_row = pd.concat([df1, df2])

X_train=pd.concat([X_train,X_val])

y_train=pd.concat([y_train,y_val])

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])

#X_val[numerical_features] = scaler.transform(X_val[numerical_features])  



# It is important to scale tain and val data separately because val is supposed to be unseen data on which we test our models. If we scale them together, data from val set will also be considered while calculating mean, median, IQR, etc



X_train[numerical_features].head()
X_train.info()
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

#TODO

clf = ExtraTreesRegressor()        #Initialize the classifier object





parameters = {'n_estimators':[3048],'min_samples_leaf':[1],'max_features':['sqrt'],'max_depth':[75],'min_samples_split':[2],'bootstrap':[False]}    #Dictionary of parameters



scorer = make_scorer(mean_squared_error, greater_is_better = False)#Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf,parameters,scoring=scorer,n_jobs=-1)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

#grid_obj = RandomizedSearchCV(clf,parameters,scoring=scorer,n_jobs=-1)

grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train



best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



#unoptimized_predictions = (clf.fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions

#optimized_predictions = best_clf.predict(X_val)        #Same, but use the best estimator



#acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

#acc_opd = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model



#print("Accuracy score on unoptimized model:{}".format(acc_unop))

#print("Accuracy score on optimized model:{}".format(acc_op))
print(best_clf)
from sklearn.metrics import mean_squared_error

#for i in range(len(optimized_predictions)):

#  optimized_predictions[i]=round(optimized_predictions[i])

#mse = mean_squared_error(y_val, optimized_predictions)

#opt3=[]

#print(np.sqrt(mse))

#for i in range(len(optimized_predictions)):

#  opt3.append(round(optimized_predictions[i]))

#mse2=mean_squared_error(y_val,opt3)

#print(np.sqrt(mse2))
from sklearn.metrics import mean_squared_error

#for i in range(len(optimized_predictions)):

#  optimized_predictions[i]=round(optimized_predictions[i])

#for i in range(len(opt3)):

#  if opt3[i]<0.6:

#    opt3[i]=0

#  elif opt3[i]>5.3:

#    opt3[i]=6

#  else:

#    opt3[i]=round(opt3[i])

#

#mse = mean_squared_error(y_val, opt3)

#np.sqrt(mse)

#print(np.unique(opt3))
df4=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")

df4.fillna(value=df4.mean(),inplace=True)

df4.isnull().sum()
t=df4['id']

df4=df4.drop(['id'],axis=1)

df4.head(10)
numerical_features = ['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11']

categorical_features = ['type']

X_test = df4[numerical_features+categorical_features]

X_test=pd.get_dummies(X_test,['type'])

X_test=X_test.drop(['type_old'],axis=1)

X_test['feature3']=np.log(X_test['feature3'])

X_test['feature5']=np.log(X_test['feature5'])

X_test[numerical_features] = scaler.transform(X_test[numerical_features])



#optimized_predictions = best_clf.predict(X_val)

#opt=best_clf.predict(X_test)

X_test.isnull().sum()
opt=best_clf.predict(X_test)

print(opt)
df4.info()

print(np.max(opt))
print(len(opt))

for i in range(len(opt)):

  opt[i]=round(opt[i])
df3=pd.DataFrame({"id":t,"rating":opt})

#df3.to_csv('/content/drive/My Drive/ML Labs/Evaluative_Lab_1/final26.csv',index=False)

df3.to_csv('final25.csv',index=False)