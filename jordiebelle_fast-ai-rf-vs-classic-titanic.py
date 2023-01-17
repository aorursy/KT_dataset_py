!pip install -q fastai==0.7.0
# https://github.com/Kaggle/docker-python/issues/315

from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
df_raw = pd.read_csv('../input/train.csv')
train_cats(df_raw)
df,y,nas = proc_df(df_raw,'Survived')
m = RandomForestClassifier(n_jobs=-1)
m.fit(df,y)
m.score(df,y) #mean accuracy given RF **Classifer**
X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.33,random_state=42)
m = RandomForestClassifier(n_jobs=-1)
m.fit(X_train,y_train)
m.score(X_train,y_train), m.score(X_valid,y_valid)
m = RandomForestClassifier(n_jobs=-1)

param_grid = { 
    'n_estimators': [10,20,30,40],
    'max_features': [1,0.5,'sqrt','log2'],
    'min_samples_leaf': [1,3,5,10,25]
}
cv_m = GridSearchCV(estimator=m,param_grid=param_grid)
cv_m.fit(X_train,y_train)
m = cv_m.best_estimator_
cv_m.best_params_
m.fit(X_train,y_train)
m.score(X_train,y_train), m.score(X_valid,y_valid)
m = cv_m.best_estimator_
m.fit(df,y)
raw_test_df = pd.read_csv('../input/test.csv')
apply_cats(raw_test_df,df_raw)
test_df,test_y,test_nas = proc_df(raw_test_df)
test_df.drop('Fare_na',axis=1,inplace=True) # added
test_preds = m.predict(test_df)
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_preds
})
submission.to_csv('submission.csv',index=False)