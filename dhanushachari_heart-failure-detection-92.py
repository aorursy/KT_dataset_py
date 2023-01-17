import pandas as pd

df=pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head(10)
df.shape


df.describe()
df['DEATH_EVENT'].value_counts()
#checking for missing values
df.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns

#checking correlation
df.corr()
#heatmap for corr
sns.heatmap(df.corr(),cmap="YlGnBu")
#handling Outliers using percentile method

df.columns
mini=df['creatinine_phosphokinase'].quantile(0.02)
df[df['creatinine_phosphokinase']<mini]
maxi=df['creatinine_phosphokinase'].quantile(.978)
df[df['creatinine_phosphokinase']>maxi]

df2=df[(df['creatinine_phosphokinase']>=mini) & (df['creatinine_phosphokinase']<maxi)]
df2.sample(10)
df2.describe()
#handling outliers using Z score or standard ditribution
sns.distplot(df2['platelets'])
mean=df2['platelets'].mean()
sd=df2['platelets'].std()
df2['zscore']=(df2.platelets-mean)/sd
df2.shape
df2 =df2[(df2['zscore']<3) & (df2['zscore']>-3)]
sns.distplot(df2['platelets'])
df2.shape

plt.hist(df['serum_sodium'])
sns.boxplot(y='serum_sodium',data=df2)
plt.boxplot(df['serum_sodium'])
Q1=df2['serum_sodium'].quantile(0.25)
Q3=df2['serum_sodium'].quantile(0.75)
iqr=Q3-Q1
lowerlimit=Q1-1.5*iqr
upperlimit=Q3+1.5*iqr

df2=df2[(df2['serum_sodium']>lowerlimit) & (df2['serum_sodium']<upperlimit)]
df2

X=df.iloc[:,:12]
y=df['DEATH_EVENT']
from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(X,y)
imp=pd.Series(data=model.feature_importances_,index=X.columns)
imp.nlargest(5).plot(kind ='bar')
sns.jointplot('time','ejection_fraction',data=df2,kind ='reg')
sns.pairplot(df2,hue='DEATH_EVENT')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,random_state=0)
y_train
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
#Randomized search CV


n_estimators=[int(i) for i in np.linspace(100,1200,12)]
max_features=['auto','sqrt','log2']
max_depth=[int(i) for i in np.linspace(5,30,6)]
min_samples_leaf=[1,2,5,10]
#create random_grid
random_grid={'n_estimators':n_estimators,
            'max_features':max_features,
            'max_depth':max_depth,
            'min_samples_leaf':min_samples_leaf}
random_grid
rf=RandomForestClassifier()
random=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=10,cv=5,verbose=2,scoring='roc_auc',random_state=42,return_train_score=True,n_jobs=1)
random.fit(X_train,y_train)

random.best_params_
random.best_score_
random.score(X_test,y_test)
plt.figure()
plt.plot(df2[df2['DEATH_EVENT']==1]['time'],df2[df2['DEATH_EVENT']==1]['platelets'],'ro',label='yes')
plt.plot(df2[df2['DEATH_EVENT']==0]['time'],df2[df2['DEATH_EVENT']==0]['platelets'],'bo',label='no')
plt.legend()
y_pred=random.predict(X_test)
y_pred
sns.distplot(y_pred-y_test,bins=5)

#we get a very narrow normal distribution whuch shows are model is performimg very well
plt.scatter(y_pred,y_test)
import pickle

with open('random_forest_classifier.pkl','wb') as file:
    pickle.dump(random,file) #dumps the model as pickle file