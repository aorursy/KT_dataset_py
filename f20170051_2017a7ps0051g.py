import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

df.head()
df.info()
df.shape
df.describe()
df.describe(include='object')
df_dtype_nunique = pd.concat([df.dtypes, df.nunique()],axis=1)

df_dtype_nunique.columns = ["dtype","unique"]

df_dtype_nunique
missing_count = df.isnull().sum()

missing_count[missing_count>0]
df.fillna(value=df.mean(),inplace=True)
df.isnull().any().any()
#df['feature8'] = df['feature8'] + 1  ## shifting feature8's min from 0 to 1
# for i in range(1,12):

#        df['feature'+str(i)] = np.log(df['feature'+str(i)])
df.corr()
corr_mat=df.corr(method='pearson')

plt.figure(figsize=(15,10))

sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
sns.regplot(x=df['feature6'],y=df['rating'],data=df)
sns.boxplot(x=df['type'], y=df['rating'], data = df)
sns.barplot(x=df['rating'].value_counts().index, y=df['rating'].value_counts())
df = pd.get_dummies(data=df,columns=['type'])
df_train = df.sample(frac=0.95,random_state=100) ## 1 for final submission

df_test = df.drop(df_train.index)
from sklearn.utils import resample
df_train['rating'].value_counts()
df_test['rating'].value_counts()
df_majority = df_train[df_train['rating'].isin([2,3])]

df_minority4 = df_train[df_train['rating']==4]

df_minority15 = df_train[df_train['rating'].isin([1,5])]

df_minority06 = df_train[df_train['rating'].isin([0,6])]
df_minority_upsampled4 = resample(df_minority4, replace=True, n_samples=1000, random_state=1) 
df_minority_upsampled15 = resample(df_minority15, replace = True, n_samples=1200, random_state=1)
df_minority_upsampled06 = resample(df_minority06, replace = True, n_samples=500, random_state=1)
df_upsampled = pd.concat([df_majority,df_minority_upsampled06,df_minority_upsampled15,df_minority_upsampled4])
df_upsampled['rating'].value_counts()
sns.barplot(x=df_upsampled['rating'].value_counts().index, y=df_upsampled['rating'].value_counts())
#df_train = df_upsampled #Comment to turn off
#X = data.drop(['id', 'feature1', 'feature2', 'feature4','feature8', 'feature9', 'type', 'feature10', 'feature11', 'rating'],axis=1)



X_train = df_train.drop(['id','rating'],axis=1)

y_train = df_train['rating']

X_val = df_test.drop(['id','rating'],axis=1)

y_val = df_test['rating']
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
from sklearn.decomposition import PCA

pca = PCA().fit(X_train)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlim(0,7,1)

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import BayesianRidge

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import ElasticNetCV

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, RandomForestRegressor, RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor

from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.metrics import mean_squared_error
# reg1 = ElasticNetCV(l1_ratio = [.1, .5, .7, .9, .95, .99, 1]).fit(X_train,y_train)

# reg2 = LinearRegression().fit(X_train,y_train)

# reg3 = Ridge().fit(X_train,y_train)

# reg4 = Lasso().fit(X_train,y_train)

# reg5 = ElasticNet().fit(X_train,y_train)

# reg6 = BayesianRidge().fit(X_train,y_train)

# reg7 = KNeighborsRegressor().fit(X_train,y_train)

# reg8 = DecisionTreeRegressor().fit(X_train,y_train)

# reg9 = GradientBoostingClassifier().fit(X_train,y_train)

# reg10 = GradientBoostingRegressor().fit(X_train,y_train)

# reg11 = AdaBoostRegressor().fit(X_train,y_train)

# reg12 = LogisticRegression().fit(X_train,y_train)

# reg13 = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)

# reg14 = RandomForestRegressor(n_estimators=500).fit(X_train,y_train)

# reg15 = MLPClassifier().fit(X_train,y_train)

# reg16 = MLPRegressor().fit(X_train,y_train)

reg17 = ExtraTreesRegressor(n_estimators=500, max_depth=50).fit(X_train,y_train)

# reg18 = ExtraTreesClassifier(n_estimators=900,min_samples_leaf=1,max_depth=None).fit(X_train,y_train)

# reg19 = GaussianNB().fit(X_train,y_train)

# reg20 = GaussianProcessRegressor.fit(X_train,y_train)
y_pred1 = reg17.predict(X_train)

#y_pred2 = reg18.predict(X_train)

#y_pred3 = reg19.predict(X_train)



rmse1 = np.sqrt(mean_squared_error(y_pred1,y_train))

#rmse2 = np.sqrt(mean_squared_error(y_pred2,y_train))

#rmse3 = np.sqrt(mean_squared_error(y_pred3,y_train))
print(rmse1,sep="\n") #train rmse
y_pred1 = reg17.predict(X_val)

#y_pred2 = reg2.predict(X_val)

#y_pred3 = reg3.predict(X_val)



rmse1 = np.sqrt(mean_squared_error(np.round(y_pred1),y_val))

#rmse2 = np.sqrt(mean_squared_error(np.round(y_pred2),y_val))

#rmse3 = np.sqrt(mean_squared_error(np.round(y_pred3),y_val))
print(rmse1,sep="\n") #val rmse
from sklearn.model_selection import GridSearchCV

 

param_grid = {

    'min_samples_split':[4,2,3],

    'min_samples_leaf': [1,2,3],

    'max_depth': [50,100,None],

    'n_estimators': [500]

}



rf = ExtraTreesRegressor()



grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
grid_search.best_params_
reg1 = ExtraTreesRegressor(min_samples_leaf=1,min_samples_split=2,n_estimators=1000).fit(X_train,y_train)

reg2 = ExtraTreesRegressor(n_estimators=350,max_depth=100).fit(X_train,y_train)

reg3 = ExtraTreesRegressor(n_estimators=515,max_depth=51).fit(X_train,y_train)
from sklearn.ensemble import VotingRegressor
regGod = VotingRegressor(estimators=[('et1', reg1) , ('et2', reg2), ('et3', reg3)])
regGod.fit(X_train,y_train)
y_god = regGod.predict(X_val)

rmse_god = (np.sqrt(mean_squared_error(np.round(y_god),y_val)))

print(rmse_god) # Val rmse
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(accuracy_score(y_val,np.round(y_god)))

print(confusion_matrix(y_val,np.round(y_god)))

print(classification_report(y_val,np.round(y_god))) 
'''

Didn't work well



reg20 = MLPClassifier(warm_start=True, learning_rate_init=0.00001, hidden_layer_sizes=[50, 50])

import pickle

rmses = []

for i in range(2500):

    reg20.fit(X_train, y_train)

    y_pred = reg20.predict(X_val)

    rmse20 = np.sqrt(mean_squared_error(np.round(y_pred),y_val))

    print(i, rmse20)

    if i>0 and rmse20<np.min(rmses):

        pickle.dump(reg20, open("best_mlp_model.p", "wb"))

        print("Saving best model in epoch", i, ". RMSE =", rmse20)

    rmses.append(rmse20)

plt.plot(np.arange(0, 2500), rmses)

'''
df1 = pd.DataFrame({'Actual': y_val, 'Predicted': np.round(y_pred1)})

df1head = df1.head(20)
df1head.plot(kind='bar',figsize=(10,8))

plt.show()
df_test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
df_test.describe()
missing_count = df_test.isnull().sum()

missing_count[missing_count>0]
df_test.fillna(value=df_test.mean(),inplace=True)
#df_test['feature8'] = df_test['feature8'] + 1
# for i in range(1,12):

#        df_test['feature'+str(i)] = np.log(df_test['feature'+str(i)])
df_test = pd.get_dummies(data=df_test,columns=['type'])
X_test = df_test.drop(['id'],axis=1)
X_test_scaled = scaler.transform(X_test)
X_test_scaled[0] ##Checking same number of columns
X_train[0]
y_test = reg3.predict(X_test_scaled)
y_test
y_test = np.round(y_test)
y_test
np.unique(y_test)
submission = pd.concat([df_test['id'],pd.Series(y_test)],axis=1)

submission.columns = ['id','rating']

submission.head()
submission['rating'] = submission['rating'].astype(int)
submission.to_csv('/kaggle/input/eval-lab-1-f464-v2/submission.csv',index=False)
submission['rating'].value_counts()/len(submission)
df['rating'].value_counts()/len(df)
import pickle

pickle.dump(reg1, open("/kaggle/input/eval-lab-1-f464-v2/best_model.p", "wb")) #reg5 = ExtraTreesRegressor(n_estimators=500,max_depth=50).fit(X_train,y_train)