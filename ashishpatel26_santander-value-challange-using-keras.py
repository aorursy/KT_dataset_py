import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')

print("Training Set:")
n_train_data=len(train_df)
n_train_features=train_df.shape[1]

print("Number of Records: {}".format(n_train_data))
print("Number of Features:{}".format(n_train_features))

print ("\nTesting set:")
n_test_data  = len(test_df)
n_test_features = test_df.shape[1]
print ("Number of Records: {}".format(n_test_data))
print ("Number of Features: {}".format(n_test_features))
train_df.head(10)
dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
print("Total Train Features with NaN Values = " + str(train_df.columns[train_df.isnull().sum() != 0].size))
print("\nTotal Test Features with NaN Values = " + str(test_df.columns[test_df.isnull().sum() != 0].size))
unique_df = train_df.nunique().reset_index()  ## check number of distinct observations in each column
unique_df.columns = ["col_name", "unique_count"]
constant_df = unique_df[unique_df["unique_count"]==1] ## if the number of distinct observation in each column is 1 then this column has constant value
constant_df.shape
train_df.drop(constant_df.col_name.tolist(),axis=1,inplace=True) ## Drop 256 columns with constant values
train_df.shape
train_df=train_df.T.drop_duplicates().T
train_df.shape
X_train=train_df.drop(['ID','target'],axis=1)
y_train=np.log1p(train_df['target'].values.astype(int))

X_test = test_df.drop(constant_df.col_name.tolist() + ["ID"], axis=1)
X_test.shape
feat_labels=list(X_train)
clf_gb = GradientBoostingRegressor(random_state = 42)
clf_gb.fit(X_train, y_train)
print(clf_gb)
feat_importances = pd.Series(clf_gb.feature_importances_, index=feat_labels)
feat_importances = feat_importances.nlargest(25)
plt.figure(figsize=(16,15))
feat_importances.plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()
print(pd.Series(clf_gb.feature_importances_, index=X_train.columns).nlargest(10))
pd.Series(clf_gb.feature_importances_, index=X_train.columns).nlargest(200).sum()
clf_rf = RandomForestRegressor(random_state = 42)
clf_rf.fit(X_train, y_train)
print(clf_rf)
feat_importances_rf = pd.Series(clf_rf.feature_importances_, index=feat_labels)
feat_importances_rf = feat_importances_rf.nlargest(25)
plt.figure(figsize=(16,15))
feat_importances_rf.plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()
print(pd.Series(clf_rf.feature_importances_, index=X_train.columns).nlargest(10))
pd.Series(clf_rf.feature_importances_, index=X_train.columns).nlargest(200).sum()
s1 = pd.Series(clf_gb.feature_importances_, index=X_train.columns).nlargest(10).index
s2 = pd.Series(clf_rf.feature_importances_, index=X_train.columns).nlargest(10).index
common_features = pd.Series(list(set(s1).intersection(set(s2)))).values

print(common_features)
pd.Series(clf_gb.feature_importances_, index=X_train.columns)[common_features].sum()
pd.Series(clf_rf.feature_importances_, index=X_train.columns)[common_features].sum()
from sklearn.decomposition import PCA
model_PCA=PCA(n_components=3)
model_PCA.fit(X_train)
transformed=model_PCA.transform(X_train)
print(transformed)
common_features=np.append(common_features,'target')
common_features
train_df1=train_df[common_features]

X_train1=train_df1.drop(['target'],axis=1)
y_train1=np.log1p(train_df1['target'].values.astype(int))
y_train1
from sklearn.model_selection import train_test_split
X_train_1, X_test_1,y_train_1,y_test_1=train_test_split(X_train1,y_train1,test_size=0.4,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_1=sc.fit_transform(X_train_1)
X_test_1=sc.transform(X_test_1)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
def baseline_model():
    model=Sequential()
    model.add(Dense(units=6,kernel_initializer='normal',activation='relu',input_dim=6))
    model.add(Dense(units=3,kernel_initializer='normal',activation='relu'))
    model.add(Dense(units=1,kernel_initializer='normal'))
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model
    
    
   # model.fit(X_train_1,y_train_1,batch_size=10,epochs=100)
seed=7
np.random.seed(seed)
estimator=KerasRegressor(build_fn=baseline_model,epochs=100,batch_size=5,verbose=1)
kfold=KFold(n_splits=3,random_state=seed)
results=cross_val_score(estimator,X_train1,y_train1,cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=3, random_state=seed)
results = cross_val_score(pipeline, X_train1,y_train1, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
model=Sequential()
model.add(Dense(units=6,kernel_initializer='normal',activation='relu',input_dim=6))
model.add(Dense(units=3,kernel_initializer='normal',activation='relu'))
model.add(Dense(units=1,kernel_initializer='normal'))
model.compile(optimizer='adam',loss='mean_squared_error')
prediction=model.predict(X_train1)
prediction
y_train1
plt.plot(prediction)
plt.plot(y_train1)
plt.show()
# result = pd.DataFrame({'ID':ID,'target':submissions})
sub = pd.read_csv("../input/sample_submission.csv")
sub.shape
test_df.shape
