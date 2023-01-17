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
import warnings
warnings.filterwarnings('ignore')
titanic = pd.read_csv('/kaggle/input/titanic/train.csv')
titanic.head()
## checking the head of our data set
titanic.info()
## checking info of all columns
titanic.shape
## checking shape of data set
titanic.describe()
## statistical information about numerical variable
round(100*(titanic.isnull().sum()/len(titanic)),2)
## checking missing value percentage in all columns
titanic.drop('Cabin',axis=1,inplace=True)
## cabin almost have 77% of missing values hence remove this column from data set
age_median = titanic['Age'].median(skipna=True)
titanic['Age'].fillna(age_median,inplace=True)
## as there is 19% of missing values in age column hence it is not a good idea to remove this row wise or column wise hence impute those missing values with the median of age 

titanic = titanic[titanic['Embarked'].isnull()!=True]
## as embarked has a very small amount of missing values hence remove those rows which have missing values in embarked column 

titanic.shape
## checking shape after removing null values
titanic_dub = titanic.copy()
## creating copy of the data frame to check duplicate values
titanic_dub.shape
## comparing shapes of two data frames
titanic.shape
## shape of original data frame
import seaborn as sns
import matplotlib.pyplot as plt
## importing libraries for data visualitation
plt.figure(figsize=(15,5), dpi=80)
plt.subplot(1,4,1)
sns.boxplot(y=titanic['Age'])
plt.title("Outliers in 'Age'")

plt.subplot(1,4,2)
ax = sns.boxplot(y=titanic['Fare'])
ax.set_yscale('log')
plt.title("Outliers in 'Fare'")

plt.subplot(1,4,3)
sns.boxplot(y=titanic['SibSp'])
plt.title("Outliers in 'SibSp'")


plt.subplot(1,4,4)
sns.boxplot(y=titanic['Parch'])
plt.title("Outliers in 'Parch'")
#ax.set_yscale('log')
plt.tight_layout()
plt.show()

## plotting all four variables to check for outliers
## it clearly shows that all four variables has some outliers


sns.catplot(x="SibSp", col = 'Survived', data=titanic, kind = 'count', palette='pastel')
sns.catplot(x="Parch", col = 'Survived', data=titanic, kind = 'count', palette='pastel')
plt.tight_layout()
plt.show()

## plotting of sibsp and parch in basis of survived and not survived
def alone(x):
    if (x['SibSp']+x['Parch']>0):
        return (1)
    else:
        return (0)
titanic['Alone'] = titanic.apply(alone,axis=1)
## creating a function to make one variable which tells us whether a person is single or accompanied by some on the ship
sns.catplot(x="Alone", col = 'Survived', data=titanic, kind = 'count', palette='pastel')
plt.show()
## drop parch and sibsp
titanic = titanic.drop(['Parch','SibSp'],axis=1)
titanic.head()

sns.distplot(titanic['Fare'])
plt.show()
titanic['Fare'] = titanic['Fare'].map(lambda x: np.log(x) if x>0 else 0)
## converting fare into a logarithmic scale
sns.distplot(titanic['Fare'])
plt.show()
## again check the distribution of fare 
sns.catplot(x="Sex", y="Survived", col="Pclass", data=titanic, saturation=.5, kind="bar", ci=None, aspect=0.8, palette='deep')
sns.catplot(x="Sex", y="Survived", col="Embarked", data=titanic, saturation=.5, kind="bar", ci=None, aspect=0.8, palette='deep')
plt.show()

## plotting of survive on basis of pclass
survived_0 = titanic[titanic['Survived']==0]
survived_1 = titanic[titanic['Survived']==1]
## divided our dataset into survived or not survived to check the distribution of age in both the cases 
survived_0.shape
## checking shape of the data set that contains the data of passengers who not survived
survived_1.shape
## checking shape of the data set that contains the data of passengers who survived
sns.distplot(survived_0['Age'])
plt.show()
## checking distribution of age in not survived data set
sns.distplot(survived_1['Age'])
plt.show()
## checking distribution of age in survived dataset
sns.boxplot(x='Survived',y='Fare',data=titanic)
plt.show()
## checking survival rate on basis of fare
Pclass_dummy = pd.get_dummies(titanic['Pclass'],prefix='Pclass',drop_first=True)
Pclass_dummy.head()
## creating dummy variables for pclass


## joing dummy variables
titanic = pd.concat([titanic,Pclass_dummy],axis=1)
titanic.head()
titanic.drop('Pclass',axis=1,inplace=True)
## as there is no use of pclass after joining the columns that contains dummy variables  for pclass
Embarked_dummy = pd.get_dummies(titanic['Embarked'],drop_first=True)
Embarked_dummy.head()
## creating dummy variables for embarked and dropping first column
titanic = pd.concat([titanic,Embarked_dummy],axis=1)
titanic.drop('Embarked',axis=1,inplace=True)
## joining dummy variables
titanic.head()
## checking head of the data set after joining dummy variables
def sex_map(x):
    if x == 'male':
        return (1)
    elif x == 'female':
        return (0)
titanic['Sex'] = titanic['Sex'].apply(lambda x:sex_map(x))

## creating function for convert sex into binary values
from sklearn.preprocessing import StandardScaler
## import libraries for scaling data
scaler = StandardScaler()
cols = ['Age','Fare']
titanic[cols] = scaler.fit_transform(titanic[cols])
titanic.head()

## using standardization method of scaling for age and fare variables
titanic.drop(['Name','Ticket'],axis=1,inplace=True)
## dropping name and ticket column
titanic.head()
## checking head after converting all values
titanic.set_index('PassengerId')
## set index as passengerid

## creating heatmap for checking corelations of variables
sns.heatmap(titanic.corr(),annot=True)
plt.show()
titanic.drop(['Pclass_2','Q'],axis=1,inplace=True)
## removing highly co related dummy variables pclass_2 and q
sns.heatmap(titanic.corr(),annot=True)
plt.show()
## again checking co relations of variables
y_train = titanic.pop('Survived')
X_train = titanic
## divided train data into x and y as independent and dependent variable
X_train = titanic[['Sex','Age','Fare','Alone','Pclass_3','S']]
X_train.head()
## selectig all columns insted of passengerid for our x
## checking head of x after that
import statsmodels.api as sm
## import stats model to build our first model
logm1 = sm.GLM(y_train,sm.add_constant(X_train),family = sm.families.Binomial())
res1 = logm1.fit()
res1.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
## removing alone as it has high p value 
X_train.drop('Alone',axis=1,inplace=True)
logm2 = sm.GLM(y_train,sm.add_constant(X_train),family = sm.families.Binomial())
res2 = logm2.fit()
res2.summary()
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train.drop('Fare',axis=1,inplace=True)
logm3 = sm.GLM(y_train,sm.add_constant(X_train),family = sm.families.Binomial())
res3 = logm3.fit()
res3.summary()
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train.columns
y_train_pred = res3.predict(sm.add_constant(X_train))

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
final_pred = pd.DataFrame({'Survived':y_train.values,'Survived_prob':y_train_pred})
final_pred['PassengerId'] = np.arange(1,len(final_pred)+1)
final_pred.head()

final_pred.info()
final_pred['predicted'] = final_pred['Survived_prob'].apply(lambda x: 1 if x>0.5 else 0)
final_pred.head()
from sklearn import metrics
confusion = metrics.confusion_matrix(final_pred.Survived,final_pred.predicted)
print(confusion)
metrics.accuracy_score(final_pred.Survived,final_pred.predicted)
## lets define all values of confusion matrix
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]
TP = confusion[1,1]
## lets calculate sensitivity
TP/float(TP+FN)
## lets calculate specificity
TN/float(TN+FP)
## false positive rate 
FP/ float(TN+FP)
## positive predictive value
TP / float(TP+FP)
## negative predictive value 
TN / float(TN+ FN)
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
fpr, tpr, thresholds = metrics.roc_curve( final_pred.Survived, final_pred.Survived_prob, drop_intermediate = False )
draw_roc(final_pred.Survived, final_pred.Survived_prob)
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    final_pred[i]= final_pred.Survived_prob.map(lambda x: 1 if x > i else 0)
final_pred.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(final_pred.Survived, final_pred[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()
final_pred['final_predicted'] = final_pred['Survived_prob'].apply(lambda x: 1 if x>0.3 else 0)
final_pred.head()
final_confusion = metrics.confusion_matrix(final_pred.Survived,final_pred.final_predicted)
print(final_confusion)
metrics.accuracy_score(final_pred.Survived,final_pred.final_predicted)
## lets define all values of confusion matrix
TN = final_confusion[0,0]
FP = final_confusion[0,1]
FN = final_confusion[1,0]
TP = final_confusion[1,1]
## lets calculate sensitivity
TP/float(TP+FN)
## lets calculate specificity
TN/float(TN+FP)
# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))
# Positive predictive value 
print (TP / float(TP+FP))
# Negative predictive value
print (TN / float(TN+ FN))
titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')
titanic_test.head()
titanic_test.info()
titanic_test['Sex'] = titanic_test['Sex'].apply(lambda x:sex_map(x))
titanic_test.head()
titanic_test.drop(['Name','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)
titanic_test.head()
Pclass = pd.get_dummies(titanic_test['Pclass'],prefix = 'Pclass')
Pclass.head()
titanic_test = pd.concat([titanic_test,Pclass],axis=1)

titanic_test.head()
Embarked = pd.get_dummies(titanic_test['Embarked'])
titanic_test = pd.concat([titanic_test,Embarked],axis=1)
titanic_test.head()
titanic_test.drop(['Pclass','Embarked','Pclass_1','Pclass_2','C','Q'],axis=1,inplace=True)
titanic_test[['Age','Fare']] = scaler.transform(titanic_test[['Age','Fare']])
titanic_test.drop('Fare',axis=1,inplace=True)
age_median = titanic_test['Age'].median(skipna=True)
titanic_test['Age'].fillna(age_median,inplace=True)
titanic_test.info()
X_test = titanic_test[['Sex', 'Age', 'Pclass_3', 'S']]
X_test.columns
y_test_pred = res3.predict(sm.add_constant(X_test))
y_test_pred.head()
test_final = pd.DataFrame({'PassengerId': titanic_test.PassengerId,'Survived_prob':y_test_pred.values})
test_final.head()
test_final['Survived'] = test_final['Survived_prob'].apply(lambda x:1 if x>0.3 else 0)
test_final.head()
test_final.drop('Survived_prob',axis=1,inplace = True)
test_final.to_csv("prediction_titanic.csv",index=False)
final_pred.head()
ks_stat_check = final_pred.iloc[ : ,[1,14]]
ks_stat_check.shape
## using function for calculate ks statistics
def ks(data=None,target=None, prob=None):
    data['target0'] = 1 - data[target]
    data['bucket'] = pd.qcut(data[prob], 10)
    grouped = data.groupby('bucket', as_index = False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['events']   = grouped.sum()[target]
    kstable['nonevents'] = grouped.sum()['target0']
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    kstable['event_rate'] = (kstable.events / data[target].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_eventrate']=(kstable.events / data[target].sum()).cumsum()
    kstable['cum_noneventrate']=(kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['KS'] = np.round(kstable['cum_eventrate']-kstable['cum_noneventrate'], 3) * 100

    #Formating
    kstable['cum_eventrate']= kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate']= kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    kstable.index = range(1,11)
    kstable.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 9)
    print(kstable)
    
    #Display KS
    from colorama import Fore
    print(Fore.RED + "KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
    return(kstable)
mydf = ks(data=ks_stat_check,target="final_predicted", prob="Survived_prob")

