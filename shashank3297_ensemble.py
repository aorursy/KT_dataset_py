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

import matplotlib.pyplot as plot

import seaborn as sns

%matplotlib inline

sns.set(style="ticks")
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn import metrics

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score

%matplotlib inline
bank=pd.read_csv("/kaggle/input/bankfull/bank-full.csv")

bank.head()
bank.shape
bank.dtypes
bank.isnull().any()
bank.describe().T
no,yes=bank.Target.value_counts()

bank.Target.value_counts()
bank['Target'].value_counts(normalize=True)
sns.boxplot(bank.age)
# Quartiles

print('1º Quartile: ', bank['age'].quantile(q = 0.25))

print('2º Quartile: ', bank['age'].quantile(q = 0.50))

print('3º Quartile: ', bank['age'].quantile(q = 0.75))

print('4º Quartile: ', bank['age'].quantile(q = 1.00))

#Calculate the outliers:

  # Interquartile range, IQR = Q3 - Q1

  # lower 1.5*IQR whisker = Q1 - 1.5 * IQR 

  # Upper 1.5*IQR whisker = Q3 + 1.5 * IQR

    

print('Ages above: ', bank['age'].quantile(q = 0.75) + 

                      1.5*(bank['age'].quantile(q = 0.75) - bank['age'].quantile(q = 0.25)), 'are outliers')
print('Numerber of outliers: ', bank[bank['age'] > 70.5]['age'].count())

print('Number of rows: ', len(bank))

#Outliers in %

print('Outliers are:', round(bank[bank['age'] > 70.5]['age'].count()*100/len(bank),2), '%')
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))

sns.boxplot(x = 'duration', data = bank, orient = 'v', ax = ax1)

ax1.tick_params(labelsize=10)



sns.distplot(bank['duration'], ax = ax2)

sns.despine(ax = ax2)

ax2.set_xlabel('Call Duration', fontsize=10)

ax2.set_ylabel('Occurence', fontsize=10)

ax2.set_title('Duration x Ocucurence', fontsize=10)

ax2.tick_params(labelsize=10)



plt.subplots_adjust(wspace=0.5)

plt.tight_layout() 
print("Max duration  call in minutes:  ", round((bank['duration'].max()/60),1))

print("Min duration  call in minutes:   ", round((bank['duration'].min()/60),1))

print("Mean duration call in minutes:   ", round((bank['duration'].mean()/60),1))

print("STD dev of duration  call in minutes:   ", round((bank['duration'].std()/60),1))

# Std close to the mean means that the data values are close to the mean 
# Quartiles

print('1º Quartile: ', bank['duration'].quantile(q = 0.25))

print('2º Quartile: ', bank['duration'].quantile(q = 0.50))

print('3º Quartile: ', bank['duration'].quantile(q = 0.75))

print('4º Quartile: ', bank['duration'].quantile(q = 1.00))

#Calculate the outliers:

  # Interquartile range, IQR = Q3 - Q1

  # lower 1.5*IQR whisker = Q1 - 1.5 * IQR 

  # Upper 1.5*IQR whisker = Q3 + 1.5 * IQR

    

print('Duration calls above: ', bank['duration'].quantile(q = 0.75) + 

                      1.5*(bank['duration'].quantile(q = 0.75) - bank['duration'].quantile(q = 0.25)), 'are outliers')
print('Numerber of outliers: ', bank[bank['duration'] > 644.5]['duration'].count())

print('Number of rows: ', len(bank))

#Outliers in %

print('Outliers are:', round(bank[bank['duration'] > 644.5]['duration'].count()*100/len(bank),2), '%')
# Look, if the call duration is iqual to 0, then is obviously that this person didn't subscribed, 

# THIS LINES NEED TO BE DELETED LATER 

bank[(bank['duration'] == 0)]
bank=bank[(bank['duration'] != 0)]
#histograms from the pair plots

sns.pairplot(bank)
sns.heatmap(bank.corr(),annot=True)
bank['job'].value_counts()
sns.countplot(bank['marital'])
plt.figure(figsize=(12,5))

sns.countplot(bank['education'])
plt.figure(figsize=(12,5))

sns.countplot(bank['default'])
sns.countplot(bank['housing'])
sns.countplot(bank['loan'])
sns.countplot(bank['contact'])
sns.countplot(bank['poutcome'])
sns.countplot(bank['Target'])
#Group numerical variables by mean for the classes of Y variable

np.round(bank.groupby(["Target"]).mean() ,1)
pd.crosstab(bank['job'], bank['Target'], normalize='index').sort_values(by='yes',ascending=False )
pd.crosstab(bank['marital'], bank['Target'], normalize='index').sort_values(by='yes',ascending=False )
pd.crosstab(bank['education'], bank['Target'], normalize='index').sort_values(by='yes',ascending=False )
print(pd.crosstab(bank['default'], bank['Target'], normalize='index').sort_values(by='yes',ascending=False ))

print(bank['default'].value_counts(normalize=True))
pd.crosstab(bank['housing'], bank['Target'], normalize='index').sort_values(by='yes',ascending=False )
pd.crosstab(bank['loan'], bank['Target'], normalize='index').sort_values(by='yes',ascending=False )
pd.crosstab(bank['contact'], bank['Target'], normalize='index').sort_values(by='yes',ascending=False )
pd.crosstab(bank['month'], bank['Target'], normalize='index').sort_values(by='yes',ascending=False )
#Binning:

def binning(col, cut_points, labels=None):

  #Define min and max values:

  minval = col.min()

  maxval = col.max()



  #create list by adding min and max to cut_points

  break_points = [minval] + cut_points + [maxval]



  #if no labels provided, use default labels 0 ... (n-1)

  if not labels:

    labels = range(len(cut_points)+1)



  #Binning using cut function of pandas

  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)

  return colBin
#Binning campaign

cut_points = [2,3,4]

labels = ["<=2","3","4",">4"]

bank['campaign_range'] = binning(bank['campaign'], cut_points, labels)

bank['campaign_range'].value_counts()
bank.drop(['campaign'], axis=1, inplace=True)

bank.columns
#function to creat group of ages, this helps because we have 78 differente values here

def age(dataframe):

    dataframe.loc[dataframe['age'] <= 32, 'age'] = 1

    dataframe.loc[(dataframe['age'] > 32) & (dataframe['age'] <= 47), 'age'] = 2

    dataframe.loc[(dataframe['age'] > 47) & (dataframe['age'] <= 70), 'age'] = 3

    dataframe.loc[(dataframe['age'] > 70) & (dataframe['age'] <= 98), 'age'] = 4

           

    return dataframe



age(bank);
def duration(data):



    data.loc[data['duration'] <= 102, 'duration'] = 1

    data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration']    = 2

    data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration']   = 3

    data.loc[(data['duration'] > 319) & (data['duration'] <= 644.5), 'duration'] = 4

    data.loc[data['duration']  > 644.5, 'duration'] = 5



    return data

duration(bank);
X = bank.drop("Target" , axis=1)

y = bank["Target"]   # select all rows and the 17 th column which is the classification "Yes", "No"

X = pd.get_dummies(X,drop_first=True)

y=y.replace(['yes','no'],[1,0])
test_size = 0.30 # taking 70:30 training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
dtree = DecisionTreeClassifier(criterion='gini',random_state=1) #criterion = entopy, gini

dtree.fit(X_train, y_train)

dtreepred = dtree.predict(X_test)



cm=confusion_matrix(y_test, dtreepred)

print(cm)

print(round(accuracy_score(y_test, dtreepred),2)*100)

sns.heatmap(cm,annot=True,fmt='g',yticklabels=['actual 0','actual 1'],xticklabels=['predict 0','predict 1'])
#classification Metrics

DTREECV = (cross_val_score(dtree, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())

DTREECV_Recall=recall_score(y_test, dtreepred)

DTREECV_F1=f1_score(y_test, dtreepred)

DTREECV_Pre=precision_score(y_test,dtreepred)
rfc = RandomForestClassifier(n_estimators = 200)

rfc.fit(X_train, y_train)

rfcpred = rfc.predict(X_test)

cm=confusion_matrix(y_test, rfcpred )

print(cm)

print(round(accuracy_score(y_test, rfcpred),2)*100)

sns.heatmap(cm,annot=True,fmt='g',yticklabels=['actual 0','actual 1'],xticklabels=['predict 0','predict 1'])
#classification Metrics

RFCCV_Recall=recall_score(y_test, rfcpred)

RFCCV_Pre=precision_score(y_test, rfcpred)

RFCCV_F1=f1_score(y_test, rfcpred)

RFCCV = (cross_val_score(rfc, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
bgcl = BaggingClassifier(base_estimator=dtree, n_estimators=200,random_state=1)

bgcl = bgcl.fit(X_train, y_train)
y_predict = bgcl.predict(X_test)

print(bgcl.score(X_test , y_test))

print("Accuracy score is ",round(accuracy_score(y_test, rfcpred),2)*100)

cm=confusion_matrix(y_test, y_predict)

print(cm)

sns.heatmap(cm,annot=True,fmt='g',yticklabels=['actual 0','actual 1'],xticklabels=['predict 0','predict 1'])
#classification Metrics

BAGCV= (cross_val_score(bgcl, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())

BAGCV_Recall=recall_score(y_test, y_predict)

BAGCV_Pre=precision_score(y_test, y_predict)

BAGCV_F1=f1_score(y_test, y_predict)
abcl = AdaBoostClassifier(n_estimators=200, random_state=1)

abcl = abcl.fit(X_train, y_train)
y_predict = abcl.predict(X_test)

print(abcl.score(X_test , y_test))



cm=confusion_matrix(y_test, y_predict)

print(cm)

sns.heatmap(cm,annot=True,fmt='g',yticklabels=['actual 0','actual 1'],xticklabels=['predict 0','predict 1'])
#classification Metrics

ADACV= (cross_val_score(abcl, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())

ADACV_Recall=recall_score(y_test, y_predict)

ADACV_Pre=precision_score(y_test, y_predict)

ADACV_F1=f1_score(y_test, y_predict)
gbk = GradientBoostingClassifier()

gbk.fit(X_train, y_train)

gbkpred = gbk.predict(X_test)

cm=confusion_matrix(y_test, gbkpred )

print(cm)

print(round(accuracy_score(y_test, gbkpred),2)*100)

sns.heatmap(cm,annot=True,fmt='g',yticklabels=['actual 0','actual 1'],xticklabels=['predict 0','predict 1'])
#classification Metrics

GBKCV = (cross_val_score(gbk, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())

GBKCV_Recall=recall_score(y_test, gbkpred)

GBKCV_Pre=precision_score(y_test, gbkpred)

GBKCV_F1=f1_score(y_test, gbkpred)
#Cross value scores

models = pd.DataFrame({

                'Models': ['Random Forest Classifier', 'Decision Tree Classifier','Bagging','Ada Boosting','Gradient Boosting'],

                'Score':  [RFCCV, DTREECV,BAGCV,ADACV ,GBKCV]})



models.sort_values(by='Score', ascending=False)
#Precision value scores

models = pd.DataFrame({

                'Models': ['Random Forest Classifier', 'Decision Tree Classifier','Bagging','Ada Boosting','Gradient Boosting'],

                'Score':  [RFCCV_Pre, DTREECV_Pre,BAGCV_Pre,ADACV_Pre ,GBKCV_Pre]})



models.sort_values(by='Score', ascending=False)
#F1 Scores

models = pd.DataFrame({

                'Models': ['Random Forest Classifier', 'Decision Tree Classifier','Bagging','Ada Boosting','Gradient Boosting'],

                'Score':  [RFCCV_F1, DTREECV_F1,BAGCV_F1,ADACV_F1 ,GBKCV_F1]})



models.sort_values(by='Score', ascending=False)
#Recall scores

models = pd.DataFrame({

                'Models': ['Random Forest Classifier', 'Decision Tree Classifier','Bagging','Ada Boosting','Gradient Boosting'],

                'Score':  [RFCCV_Recall, DTREECV_Recall,BAGCV_Recall,ADACV_Recall ,GBKCV_Recall]})



models.sort_values(by='Score', ascending=False)