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
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df
df.info()
df.isnull().sum(axis=0)
df.describe()
df.isnull().mean()
df.Pclass.value_counts()
df.Embarked.value_counts()
df = df.drop('Cabin', axis = 1)
df.info()
df.isnull().mean()
df = df[~np.isnan(df['Age'])]
df.Embarked.isnull().value_counts()
df.dropna(axis = 1, how= 'any', inplace = True)
df.head()
df = df.drop(['Name','Ticket'], axis = 1)
df.info()
cor = df.corr()   
cor
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
sns.heatmap(cor, cmap= 'YlGnBu', annot = True)

pd_dumm = pd.get_dummies(df['Pclass'], drop_first = True, prefix = 'pclass')
pd_dumm
df = pd.concat([df,pd_dumm], axis =1)
df1 = pd.get_dummies(df['Sex'], drop_first=True)
df = pd.concat([df,df1], axis =1)
df.head()
df = df.drop(['Sex','Pclass'], 1)
df.head()
df.info()
df.Survived.value_counts()
df2 = df[['Survived', 'Age', 'SibSp', 'Parch', 'Fare']]
cor1= df2.corr()
plt.figure(figsize= (10,10))
sns.heatmap(cor1, cmap= 'YlGnBu', annot = True)
df['PassengerId'] = df['PassengerId'].astype(object)
df.info()
from sklearn.model_selection import train_test_split

x = df.drop(['PassengerId','Survived'], 1)
x.head()
y= df['Survived']
y.head()
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size= .7, test_size = .3, random_state= 100 )
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train.head()
plt.figure(figsize = (20,10))        
sns.heatmap(df.corr(),annot = True)
plt.show()
df = df.drop('pclass_3',1)
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE
rfe = RFE(logreg, 6)
rfe= rfe.fit(x_train,y_train)
logm1 = sm.GLM(y_train,(sm.add_constant(x_train)), family = sm.families.Binomial())
logm1.fit().summary()
rfe.support_
list(zip(x_train.columns, rfe.support_, rfe.ranking_))
x_train1 = x_train.drop('Fare',1)
m2 = sm.GLM(y_train,(sm.add_constant(x_train1)),family =sm.families.Binomial())
res = m2.fit()
res.summary()
x_train2 = x_train1.drop('Parch',1)
m3 = sm.GLM(y_train,(sm.add_constant(x_train2)),family =sm.families.Binomial())
res = m3.fit()
res.summary()
y_train_pred = res.predict(sm.add_constant(x_train2))
y_train_pred
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Survived':y_train.values, 'Surv_Prob':y_train_pred})
y_train_pred_final['PassengerId'] = y_train.index
y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Surv_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()
from sklearn import metrics
confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted )
print(confusion)
print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.predicted))
from statsmodels.stats.outliers_influence import variance_inflation_factor

a = x_train.drop(['Parch'], 1)

a.columns.value_counts()
vif = pd.DataFrame()
vif['Features'] = x_train2.columns
vif['VIF'] = [variance_inflation_factor(x_train2.values, i) for i in range(x_train2.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Survived, y_train_pred_final.predicted, drop_intermediate = False )
draw_roc(y_train_pred_final.Survived, y_train_pred_final.predicted)
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Surv_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix


num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()
y_train_pred_final['final_pred'] = y_train_pred_final.Surv_Prob.map(lambda x: 1 if x > 0.4 else 0)
y_train_pred_final.head()
con = metrics.confusion_matrix(y_train_pred_final.final_pred,y_train_pred_final.predicted)

print(con)

x_test= x_test.drop(['Parch','Fare'], 1)


x_test.info()
x_test_sm = sm.add_constant(x_test)
y_test_pred = res.predict(x_test_sm)
y_test_pred[:10]
y_pred_1 = pd.DataFrame(y_test_pred)
y_pred_1.head()
y_pred_1= pd.DataFrame(y_test_pred)
y_pred_1.head()
y_test_df = pd.DataFrame(y_test)
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
y_pred_final = y_pred_final.rename(columns = {0: 'Survive_pred'})
y_pred_final['final_pred']= y_pred_final['Survive_pred'].map(lambda x : 1 if x > 0.4 else 0)
y_pred_final
metrics.accuracy_score(y_pred_final.Survived,y_pred_final.final_pred)
met = metrics.confusion_matrix(y_pred_final.Survived,y_pred_final.final_pred) 
met
