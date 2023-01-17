# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,HuberRegressor,LinearRegression,LogisticRegression,SGDRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()
del df['Serial No.']
df.head()
%pylab inline
sns.distplot(df['Chance of Admit '])
pylab.hist(df['GRE Score'])
pylab.xlabel('GRE Score')
pylab.hist(df['TOEFL Score'])
pylab.xlabel('TOEFL Score')
sns.relplot(x="GRE Score", y="TOEFL Score", kind="line", ci="sd", data=df)
sns.relplot(x="GRE Score", y="CGPA", kind="line", ci="sd",data=df)
fig = sns.lmplot(x="CGPA", y="LOR ", data=df, hue="Research")
plt.title("GRE Score vs CGPA")
plt.show()
X = df.drop(['Chance of Admit '], axis=1)
y = df['Chance of Admit ']
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, shuffle=False)
kfolds = 4 
split = KFold(n_splits=kfolds, shuffle=True, random_state=42)
base_models = [("DT_model",DecisionTreeRegressor(random_state=42)),
               ("RF_model", RandomForestRegressor(random_state=42,n_jobs=-1)),
               ("LR_model", LinearRegression(n_jobs=-1)),
               ("KN_model", KNeighborsRegressor(n_jobs=-1)),
              ("SVR_model",SVR()),
              ('ABR_model',AdaBoostRegressor(random_state=42)),
              ('GBR_model',GradientBoostingRegressor(random_state=42)),
              ('XGB_model',XGBRegressor(random_state=42,n_jobs=-1))]
for name,model in base_models:
    clf = model
    cv_results = cross_val_score(clf, 
                                 X, y, 
                                 cv=split,
                                 scoring="neg_mean_absolute_error",
                                 n_jobs=-1
                                 )
    min_score = round(min(cv_results), 4)
    max_score = round(max(cv_results), 4)
    mean_score = round(np.mean(cv_results), 4)
    std_dev = round(np.std(cv_results), 4)
    print(f"{name} absolute error: {mean_score} +/- {std_dev} (std) min: {min_score}, max: {max_score}")
def GetScaledModel(nameOfScaler):
    scaler = StandardScaler()
    pipelines = []
    
    pipelines.append((nameOfScaler+'DT_model'  , Pipeline([('Scaler', scaler),("DT_model",DecisionTreeRegressor())])))
    pipelines.append((nameOfScaler+'RF_model' , Pipeline([('Scaler', scaler),("RF_model", RandomForestRegressor())])))
    pipelines.append((nameOfScaler+'LR_model' , Pipeline([('Scaler', scaler), ("LR_model", LinearRegression())])))
    pipelines.append((nameOfScaler+'KN_model', Pipeline([('Scaler', scaler),("KN_model", KNeighborsRegressor())])))
    pipelines.append((nameOfScaler+'SVR_model'  , Pipeline([('Scaler', scaler),("SVR_model",SVR())])))
    pipelines.append((nameOfScaler+'ABR_model'  , Pipeline([('Scaler', scaler),('ABR_model',AdaBoostRegressor())])))
    pipelines.append((nameOfScaler+'GBR_model' , Pipeline([('Scaler', scaler),('GBR_model',GradientBoostingRegressor())])  ))
    pipelines.append((nameOfScaler+'XGB_model'  , Pipeline([('Scaler', scaler),('XGB_model',XGBRegressor(random_state=42,n_jobs=-1))]) ))
    return pipelines
def BasedLine2(X_train, y_train,models):
    num_folds = 10
    scoring="neg_mean_absolute_error"

    results = []
    names = []
    for name, model in models:
        split = KFold(n_splits=kfolds, shuffle=True, random_state=42)
        cv_results = cross_val_score(model, X_train, y_train, cv=split, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        
    return names, results
def ScoreDataFrame(names,results):
    def floatingDecimals(f_val, dec=3):
        prc = "{:."+str(dec)+"f}" 
    
        return float(prc.format(f_val))

    scores = []
    for r in results:
        scores.append(floatingDecimals(r.mean(),4))

    scoreDataFrame = pd.DataFrame({'Model':names, 'Score': scores})
    return scoreDataFrame
names,results = BasedLine2(X_train, y_train,base_models)
basedLineScore = ScoreDataFrame(names,results)
basedLineScore
models = GetScaledModel('standard')
names,results = BasedLine2(X_train, y_train,models)
scaledScoreStandard = ScoreDataFrame(names,results)
compareModels = pd.concat([basedLineScore,
                           scaledScoreStandard], axis=1)
compareModels
lr =  LinearRegression()
lr.fit(X_train,y_train)
predict = lr.predict(X_test)
mean_absolute_error(y_test,predict)
r2_score(y_test,predict)