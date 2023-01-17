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
#Suressing warnings
import warnings
warnings.filterwarnings('ignore')
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#import the placement data set
placement_data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
placement_data.head()
#dimension of data frame
placement_data.shape
#statistical aspect of dataframe
placement_data.describe()
placement_data.info()
placement_data.isnull().sum()
placement_data['salary'].describe()
placement_data['status'].value_counts()
placement_data['salary'].fillna("0",inplace=True)
placement_data
#lets check the salary column
placement_data['salary']=placement_data['salary'].astype('float')
#map values with 0 and 1
placement_data['status'] = placement_data['status'].replace({'Placed':1,'Not Placed':0}).astype('object')
#counting values
placement_data['status'].value_counts()
#map values with 0 and 1
#placement_data['gender'] = placement_data['gender'].replace({'M':0,'F':1}).astype('object')
#counting values
placement_data['gender'].value_counts()
placement_data['gender'].value_counts()
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
colors = ['gold','lightcoral']
explode = (0,0)
placement_data.gender.value_counts().plot.pie(explode=explode, colors=colors, shadow=True, autopct='%1.1f%%')
plt.title('gender')
plt.subplot(1,2,2)
sns.countplot('gender',hue='status',data=placement_data)
plt.show()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(x='gender',y='salary',data=placement_data)
plt.subplot(1,2,2)
sns.barplot(x='gender',y='salary',data=placement_data)

plt.show()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(x='salary',y='ssc_b',data=placement_data)
plt.subplot(1,2,2)
sns.countplot(x='status',hue='ssc_b',data=placement_data)

plt.show()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(x='salary',y='hsc_b',data=placement_data)
plt.subplot(1,2,2)
sns.countplot(x='status',hue='hsc_b',data=placement_data)

plt.show()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.violinplot(x='salary',y='degree_t',data=placement_data)
plt.subplot(1,2,2)
sns.countplot(x='status',hue='degree_t',data=placement_data)

plt.show()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(x='salary',y='specialisation',data=placement_data)
plt.subplot(1,2,2)
sns.countplot(x='status',hue='specialisation',data=placement_data)

plt.show()
#findout pearson correlation between 10th & 12th marks student.

from scipy.stats import pearsonr
corr,_ = pearsonr(placement_data['ssc_p'], placement_data['hsc_p'])
print('pearsons correlation: %.3f' % corr)
sns.regplot(x='ssc_p',y='hsc_p',data=placement_data)
plt.figure(figsize=(10,10))
sns.pairplot(placement_data)
plt.show()
plt.figure(figsize=(10,10))
sns.heatmap(placement_data.corr(),annot=True)
plt.figure(figsize=(10,10))


sns.catplot(x='status',y='ssc_p',hue='gender',kind='swarm', data=placement_data)
sns.catplot(x='status', y='hsc_p', hue='gender', kind='swarm', data=placement_data)
sns.catplot(x='status', y='degree_p', hue='gender',kind='swarm', data=placement_data)

plt.show()
#creating dummy variable for some categorical column and dropping the first one.
dummy1 = pd.get_dummies(placement_data[['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation',]],drop_first=True)

#adding result to master data frame
placement_data = pd.concat([placement_data,dummy1],axis=1)

#now check the dataframe
placement_data.head()
# For status, we would change the data type to int from object.

placement_data['status'] = placement_data['status'].astype('int')
placement_data['status'].value_counts()
#In our analysis there is no use of salary, hence dropping it.

placement_data.drop(['salary'],axis=1,inplace=True)
placement_data.head()
#Dropping all other repeated column

placement_data.drop(['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation'],axis=1,inplace=True)
placement_data.head()
#dropping serial number too.
placement_data.drop(['sl_no'],axis=1,inplace=True)
placement_data.head()
print('Final shape of data:', placement_data.shape)
print('\n Final column of data:\n',placement_data.columns)
X = placement_data.loc[:,['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'gender_M',
       'ssc_b_Others', 'hsc_b_Others', 'hsc_s_Commerce', 'hsc_s_Science',
       'degree_t_Others', 'degree_t_Sci&Tech', 'workex_Yes',
       'specialisation_Mkt&HR']]
y= placement_data.loc[:,['status']]
#plot heatmap

plt.figure(figsize=(10,10))
sns.heatmap(X.corr(),annot=True,cmap='RdYlGn')
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_test_pred = logreg.predict(X_test)
from sklearn import metrics

#confusion metrics
confusion = metrics.confusion_matrix(y_test,y_test_pred)
print(confusion)
#plot confusion matrix

import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_test,y_test_pred)

print('Logistic model accuracy score:',metrics.accuracy_score(y_test,y_test_pred))
TP = confusion[1,1] #True Positive
TN = confusion[0,0] #True negative
FP = confusion[0,1] #False Positive
FN = confusion[1,0] #False negative
#sensitivity

TP/float(TP+FN)
#specificity

TN/float(TN+FP)
#false positive rate

FP/float(FP+TN)
#positive predicted value

TP/float(TP+FP)
#negative predicted value

TN/float(TN+FN)
confusion[1,1]/(confusion[0,1]+confusion[1,1])
confusion[1,1]/(confusion[1,0]+confusion[1,1])
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve(area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None
probs = logreg.predict_proba(X_test)
preds = probs[:,1]

fpr, tpr, thresholds = metrics.roc_curve( y_test, preds, drop_intermediate = False )
draw_roc(y_test, preds)