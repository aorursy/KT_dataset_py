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
df=pd.read_csv('/kaggle/input/heart-disease-prediction-using-logistic-regression/framingham.csv')
df.info()
df.describe(include='all')
df.isnull().sum()
data=df#.drop(['education'],axis=1)
data.dropna(axis=0,inplace=True)

data.describe(include='all')
category=['male','currentSmoker','BPMeds','prevalentStroke','prevalentHyp','diabetes','education']
cols=list(data.columns)
cols.remove('TenYearCHD')
numeric=list(set(cols)-set(category))
for i in category:
    print("-------"+i+"-------")
    contingency_table=pd.crosstab(data[i],data["TenYearCHD"])
    #print('contingency_table :-\n',contingency_table)

    Observed_Values = contingency_table.values 
    #print("Observed Values :-\n",Observed_Values)

    #Expected Values
    import scipy.stats
    b=scipy.stats.chi2_contingency(contingency_table)
    Expected_Values = b[3]
    #print("Expected Values :-\n",Expected_Values)

    deg_freedom=b[2]
    print("Degree of Freedom:-",deg_freedom)

    from scipy.stats import chi2
    chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
    chi_square_statistic=chi_square[0]+chi_square[1]
    print("chi-square statistic:-",chi_square_statistic)

    p_value=1-chi2.cdf(x=chi_square_statistic,df=deg_freedom)
    print('p-value:',p_value)

    if p_value<=0.05:
        print("There is a relationship")
    else:
        print("There is no relationship ")
from scipy import stats
for i in numeric:
    print("------"+i+"--------")
    df_anova = data[["TenYearCHD",i]]

    grps = pd.unique(df_anova["TenYearCHD"].values)
    d_data = {grp:df_anova[i][df_anova["TenYearCHD"] == grp] for grp in grps}

    F, p = stats.f_oneway(d_data[0], d_data[1])
    print("p-value for significance is: ", round(p,4))
    if p<0.05:
        print("reject null hypothesis")
    else:
        print("accept null hypothesis")
from statsmodels.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
x=data[cols]
x = add_constant(x)
y=data.TenYearCHD
model=Logit(y,x)
result=model.fit()
result.summary()
cols=data.columns[:-1]
print(cols)

def back_feature_elem (data_frame,dep_var,col_list):
    while len(col_list)>0 :
        x=data[col_list]
        x = add_constant(x)
        y=data.TenYearCHD
        model=Logit(y,x)
        result=model.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.05):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)
    
result=back_feature_elem(data,data.TenYearCHD,cols)
result.summary()
print(np.exp(result.params))
#lets get the confusion matrix based on statsmodel

conf_sm=pd.DataFrame(result.pred_table())
conf_sm.columns=['predicted 0','predicted 1']
conf_sm=conf_sm.rename(index={0:'Actual 0',1:'Actual 1'})
conf_sm
#train split

import sklearn
new_features=data[['age','male','cigsPerDay','totChol','sysBP','glucose','TenYearCHD']]
x=new_features.iloc[:,:-1]
y=new_features.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)


from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test,y_pred)

conf=pd.DataFrame(confusion_matrix)
conf.columns=['predicted 0','predicted 1']
conf=conf.rename(index={0:'Actual 0',1:'Actual 1'})
print(conf)

print("model accuracy:")
print(sklearn.metrics.accuracy_score(y_test,y_pred))
x=data.drop(['TenYearCHD','currentSmoker','heartRate'],axis=1)
x=add_constant(x)
y=data.TenYearCHD
model=Logit(y,x)
result=model.fit(disp=0)
result.summary()
x=data.drop(['TenYearCHD','currentSmoker','heartRate','glucose','prevalentStroke','education','BPMeds','prevalentHyp','diabetes','diaBP','BMI'],axis=1)
x=add_constant(x)
y=data.TenYearCHD
model=Logit(y,x)
result=model.fit(disp=0)
result.summary()
#train split

import sklearn
new_features=data[['age','male','cigsPerDay','totChol','sysBP','TenYearCHD']]
x=new_features.iloc[:,:-1]
y=new_features.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)
#print(x_test)
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)

from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test,y_pred)

conf=pd.DataFrame(confusion_matrix)
conf.columns=['predicted 0','predicted 1']
conf=conf.rename(index={0:'Actual 0',1:'Actual 1'})
print(conf)

print("model accuracy:")
print(sklearn.metrics.accuracy_score(y_test,y_pred))