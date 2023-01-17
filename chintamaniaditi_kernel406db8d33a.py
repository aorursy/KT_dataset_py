import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix,classification_report
data=pd.read_csv('../input/indian-liver-patient-records/indian_liver_patient.csv')
data.head()
data.rename({'Dataset':'Liver_disorder'},axis=1,inplace=True)
data.isnull().sum()
data.shape
data.dropna().shape
data['Albumin_and_Globulin_Ratio'].unique()
data[data['Albumin_and_Globulin_Ratio'].isnull()]
data.dropna(inplace=True)
data['Liver_disorder'].value_counts()
data.shape
data.head()
sns.pairplot(data)
data.corr()
data['Gender']=data['Gender'].replace({'Male':0,'Female':1})
data['Gender'].value_counts()
data.info()
data['Gender']=data['Gender'].astype('category')
#no_of_MF_with_disorder=data[data['Liver_disorder']==1].groupby('Gender')[['Gender']].count()
#no_of_MF_with_disorder
tab=pd.crosstab(data.Gender,data.Liver_disorder)

print(tab)
#tot=data.groupby('Gender')[['Gender']].count()
#tot
tot_male=tab.iloc[1].sum()
male_with_disorder=tab.iloc[1,0]
tot_female=tab.iloc[0].sum()
female_with_disorder=tab.iloc[0,0]
p1=male_with_disorder/tot_male
p2=female_with_disorder/tot_female
p=(male_with_disorder+female_with_disorder)/data.shape[0]
Zstats=(p1-p2)/(np.sqrt(p*(1-p)*((1/tot_male)+(1/tot_female))))
Zstats
import scipy.stats as stats

stats.norm.cdf(Zstats)
cor=data.corr()
cor_target=abs(cor['Liver_disorder'])
Pearson_Coeff=pd.DataFrame(cor_target)

print(Pearson_Coeff)
data.info()
data.drop('Total_Protiens',axis=1,inplace=True)
d=data.drop('Liver_disorder',axis=1)
features = "+" .join(d)

#y, X = dmatrices('annual_inc ~' + features, df, return_type='dataframe')
features
from   patsy                                import dmatrices

from   statsmodels.stats.outliers_influence import variance_inflation_factor

y,X=dmatrices('Liver_disorder~'+features,data,return_type='dataframe')
Vif=pd.DataFrame()
Vif['features']=X.columns

Vif['VIF factor']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]



print(Vif)
LData=data.drop('Direct_Bilirubin',axis=1)
LData.head()
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

penalty=['l2','l1']

multi_class=['ovr', 'auto']

X=LData.drop('Liver_disorder',axis=1)

y=LData[['Liver_disorder']]

model=LogisticRegression()

grid=GridSearchCV(estimator=model,cv=3,param_grid=dict(penalty=penalty,multi_class=multi_class))

grid.fit(X,y)
print(grid.best_params_)
print('Recall:',grid.best_score_)

print('Accuracy:',grid.best_score_)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.20,random_state=2)
y_train.shape
model1=LogisticRegression()

model1.fit(X_train, y_train)
preds= model1.predict(X_test)
print(accuracy_score(y_test,preds))