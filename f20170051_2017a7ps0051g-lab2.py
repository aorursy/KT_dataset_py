import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

df.head(10)
df.info()
df.shape
df.describe()
df_dtype_nunique = pd.concat([df.dtypes, df.nunique()],axis=1)

df_dtype_nunique.columns = ["dtype","unique"]

df_dtype_nunique
df.isnull().any().any()
#df.corr()
corr_mat=df.corr(method='pearson')

plt.figure(figsize=(15,10))

sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
#sns.regplot(x=df['chem_6'],y=df['class'],data=df)
data = df.copy()
X = data.drop(['class'],axis=1)

y = data['class']
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_scaled = scaler.fit_transform(X)

X_scaled
# from sklearn.decomposition import PCA

# pca = PCA().fit(X_scaled)

# plt.plot(np.cumsum(pca.explained_variance_ratio_))

# plt.xlim(0,7,1)

# plt.xlabel('Number of components')

# plt.ylabel('Cumulative explained variance')
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import BayesianRidge

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import ElasticNetCV

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, RandomForestRegressor, RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor

from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb 

import xgboost as xgb

from sklearn.ensemble import VotingRegressor

from sklearn.ensemble import VotingClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X_scaled,y,test_size=0.15,random_state=4)
X_train = X_scaled

y_train = y
reg1 = ElasticNetCV(l1_ratio = [.1, .5, .7, .9, .95, .99, 1]).fit(X_train,y_train)

reg2 = LinearRegression().fit(X_train,y_train)

reg3 = Ridge().fit(X_train,y_train)

reg4 = Lasso().fit(X_train,y_train)

reg5 = ElasticNet().fit(X_train,y_train)

reg6 = BayesianRidge().fit(X_train,y_train)

reg7 = KNeighborsRegressor().fit(X_train,y_train)

reg8 = DecisionTreeRegressor().fit(X_train,y_train)

reg9 = GradientBoostingClassifier().fit(X_train,y_train)

reg10 = GradientBoostingRegressor().fit(X_train,y_train)

reg11 = AdaBoostRegressor().fit(X_train,y_train)

reg12 = LogisticRegression().fit(X_train,y_train)

reg13 = RandomForestClassifier().fit(X_train,y_train)

reg14 = RandomForestRegressor().fit(X_train,y_train)

reg15 = MLPClassifier().fit(X_train,y_train)

reg16 = MLPRegressor().fit(X_train,y_train)

reg17 = ExtraTreesRegressor().fit(X_train,y_train)

reg18 = ExtraTreesClassifier(max_depth = 100,

 min_samples_leaf = 1,

 min_samples_split = 2,

 n_estimators = 50).fit(X_train,y_train)

reg19 = GaussianNB().fit(X_train,y_train)

reg20 = xgb.XGBClassifier().fit(X_train,y_train)

reg21 = xgb.XGBRegressor().fit(X_train,y_train)

reg22 = lgb.LGBMClassifier().fit(X_train,y_train)

reg23 = lgb.LGBMRegressor().fit(X_train,y_train)

reg24 = xgb.XGBClassifier(objective="multi:softmax",n_estimators=200).fit(X_train,y_train)

reg25 = KNeighborsClassifier().fit(X_train,y_train)

reg26 = SVC().fit(X_train,y_train)

reg27 = GaussianProcessClassifier(1.0 * RBF(1.0)).fit(X_train,y_train)
y_pred1 = reg1.predict(X_train)

y_pred2 = reg2.predict(X_train)

y_pred3 = reg3.predict(X_train)

y_pred4 = reg4.predict(X_train)

y_pred5 = reg5.predict(X_train)

y_pred6 = reg6.predict(X_train)

y_pred7 = reg7.predict(X_train)

y_pred8 = reg8.predict(X_train)

y_pred9 = reg9.predict(X_train)

y_pred10 = reg10.predict(X_train)

y_pred11 = reg11.predict(X_train)

y_pred12 = reg12.predict(X_train)

y_pred13 = reg13.predict(X_train)

y_pred14 = reg14.predict(X_train)

y_pred15 = reg15.predict(X_train)

y_pred16 = reg16.predict(X_train)

y_pred17 = reg17.predict(X_train)

y_pred18 = reg18.predict(X_train)

y_pred19 = reg19.predict(X_train)

y_pred20 = reg20.predict(X_train)

y_pred21 = reg21.predict(X_train)

y_pred22 = reg22.predict(X_train)

y_pred23 = reg23.predict(X_train)

y_pred24 = reg24.predict(X_train)

y_pred25 = reg25.predict(X_train)

y_pred26 = reg26.predict(X_train)

y_pred27 = reg27.predict(X_train)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(accuracy_score(y_train,np.round(y_pred1)))

# print(accuracy_score(y_train,np.round(y_pred2)))

# print(accuracy_score(y_train,np.round(y_pred3)))

# print(accuracy_score(y_train,np.round(y_pred4)))

# print(accuracy_score(y_train,np.round(y_pred5)))

# print(accuracy_score(y_train,np.round(y_pred6)))

# print(accuracy_score(y_train,np.round(y_pred7)))

# print(accuracy_score(y_train,np.round(y_pred8)))

print("9: " + str(accuracy_score(y_train,np.round(y_pred9))))

print("10: " + str(accuracy_score(y_train,np.round(y_pred10))))

# print(accuracy_score(y_train,np.round(y_pred11)))

print("12: " + str(accuracy_score(y_train,np.round(y_pred12))))

print("13: " + str(accuracy_score(y_train,np.round(y_pred13))))

print("14: " + str(accuracy_score(y_train,np.round(y_pred14))))

print("15: " + str(accuracy_score(y_train,np.round(y_pred15))))

#print("16: " + str(accuracy_score(y_train,np.round(y_pred16))))

print("17: " + str(accuracy_score(y_train,np.round(y_pred17))))

print("18: " + str(accuracy_score(y_train,np.round(y_pred18))))

#print("19: " + str(accuracy_score(y_train,np.round(y_pred19))))

print("20: " + str(accuracy_score(y_train,np.round(y_pred20))))

print("21: " + str(accuracy_score(y_train,np.round(y_pred21))))

print("22: " + str(accuracy_score(y_train,np.round(y_pred22))))

print("23: " + str(accuracy_score(y_train,np.round(y_pred23))))

print("24: " + str(accuracy_score(y_train,np.round(y_pred24))))

print("25: " + str(accuracy_score(y_train,np.round(y_pred25))))

print("26: " + str(accuracy_score(y_train,np.round(y_pred26))))

print("27: " + str(accuracy_score(y_train,np.round(y_pred27))))
y_pred1 = reg1.predict(X_val)

y_pred2 = reg2.predict(X_val)

y_pred3 = reg3.predict(X_val)

y_pred4 = reg4.predict(X_val)

y_pred5 = reg5.predict(X_val)

y_pred6 = reg6.predict(X_val)

y_pred7 = reg7.predict(X_val)

y_pred8 = reg8.predict(X_val)

y_pred9 = reg9.predict(X_val)

y_pred10 = reg10.predict(X_val)

y_pred11 = reg11.predict(X_val)

y_pred12 = reg12.predict(X_val)

y_pred13 = reg13.predict(X_val)

y_pred14 = reg14.predict(X_val)

y_pred15 = reg15.predict(X_val)

y_pred16 = reg16.predict(X_val)

y_pred17 = reg17.predict(X_val)

y_pred18 = reg18.predict(X_val)

y_pred19 = reg19.predict(X_val)

y_pred20 = reg20.predict(X_val)

y_pred21 = reg21.predict(X_val)

y_pred22 = reg22.predict(X_val)

y_pred23 = reg23.predict(X_val)

y_pred24 = reg24.predict(X_val)

y_pred25 = reg25.predict(X_val)

y_pred26 = reg26.predict(X_val)

y_pred27 = reg27.predict(X_val)
# print(accuracy_score(y_val,np.round(y_pred1)))

# print(accuracy_score(y_val,np.round(y_pred2)))

# print(accuracy_score(y_val,np.round(y_pred3)))

# print(accuracy_score(y_val,np.round(y_pred4)))

# print(accuracy_score(y_val,np.round(y_pred5)))

# print(accuracy_score(y_val,np.round(y_pred6)))

# print(accuracy_score(y_val,np.round(y_pred7)))

# print(accuracy_score(y_val,np.round(y_pred8)))

print("9: " + str(accuracy_score(y_val,np.round(y_pred9))))

print("10: " + str(accuracy_score(y_val,np.round(y_pred10))))

# print(accuracy_score(y_val,np.round(y_pred11)))

print("12: " + str(accuracy_score(y_val,np.round(y_pred12))))

print("13: " + str(accuracy_score(y_val,np.round(y_pred13))))

print("14: " + str(accuracy_score(y_val,np.round(y_pred14))))

print("15: " + str(accuracy_score(y_val,np.round(y_pred15))))

#print("16: " + str(accuracy_score(y_val,np.round(y_pred16))))

print("17: " + str(accuracy_score(y_val,np.round(y_pred17))))

print("18: " + str(accuracy_score(y_val,np.round(y_pred18))))

# print(accuracy_score(y_val,np.round(y_pred19)))

#print("19: " + str(accuracy_score(y_train,np.round(y_pred19))))

print("20: " + str(accuracy_score(y_val,np.round(y_pred20))))

print("21: " + str(accuracy_score(y_val,np.round(y_pred21))))

print("22: " + str(accuracy_score(y_val,np.round(y_pred22))))

print("23: " + str(accuracy_score(y_val,np.round(y_pred23))))

print("24: " + str(accuracy_score(y_val,np.round(y_pred24))))

print("25: " + str(accuracy_score(y_val,np.round(y_pred25))))

print("26: " + str(accuracy_score(y_val,np.round(y_pred26))))

print("27: " + str(accuracy_score(y_val,np.round(y_pred27))))
regET = ExtraTreesClassifier().fit(X_train,y_train)
# "learning_rate": [0.1, 0.01, 0.001],

#                "gamma" : [0.01, 0.1, 0.3, 1],

#                "max_depth": [2, 4, 10],

#                "colsample_bytree": [0.3, 0.6, 0.8, 1.0],

#                "subsample": [0.2, 0.4, 0.6],

#                "reg_alpha": [0, 0.5, 1],

#                "reg_lambda": [1, 2, 4.5],

#                "min_child_weight": [1, 3, 5],

param_grid = {

    'min_samples_split':[4,2,3],

    'min_samples_leaf': [1,2,3],

    'max_depth': [50,100,None],

    'n_estimators': [10,50,100,250,500]

}

grid_search = GridSearchCV(estimator = regET, param_grid = param_grid, cv = 3, n_jobs=-1, verbose=2,

                           return_train_score=True)
grid_search.fit(X_train, y_train)
grid_search.best_params_
regGod = VotingClassifier(estimators=[('GradientBoostingClassifier', reg9) , ('XGB', reg20), ('LGBM', reg22)])
regGod.fit(X_train,y_train)
y_god = regGod.predict(X_val)

print("Voting accuracy " + str(accuracy_score(y_val,np.round(y_god))))
df_test = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')
df_test.describe()
X_test = df_test

#.drop(['chem_2','chem_7'],axis=1)
X_test_scaled = scaler.transform(X_test)
y_test = reg22.predict(X_test_scaled)
y_test
y_test = np.round(y_test)
y_test
np.unique(y_test)
submission = pd.concat([df_test['id'],pd.Series(y_test)],axis=1)

submission.columns = ['id','class']

submission.head()
submission['class'] = submission['class'].astype(int)
submission.to_csv('submission.csv',index=False)
submission['class'].value_counts()/len(submission)
df['class'].value_counts()/len(df)