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


df=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

df

df.describe()

df.info()

import matplotlib.pyplot as plt

import seaborn as sns

sns.pairplot(df)
plt.show()

df.columns

plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'Outcome', y = 'Pregnancies', data = df)
plt.subplot(2,3,2)
sns.boxplot(x = 'Outcome', y = 'Glucose', data = df)
plt.subplot(2,3,3)
sns.boxplot(x = 'Outcome', y = 'BloodPressure', data = df)
plt.subplot(2,3,4)
sns.boxplot(x = 'Outcome', y = 'SkinThickness', data = df)
plt.subplot(2,3,5)
sns.boxplot(x = 'Outcome', y = 'Insulin', data = df)
plt.subplot(2,3,6)
sns.boxplot(x = 'Outcome', y = 'BMI', data = df)
# plt.subplot(2,3)
# sns.boxplot(x='Outcome',y='DiabetesPedigreeFunction',data=df)
plt.show()


df.isnull().sum()

df.isna().sum()

(df==0).sum()

def plotUnivariateAnalysis(column, xLabel, typeVar, order):
    plotType = 'hist'
    if plotType == 'hist':
        plt.figure(figsize=[15,6])
        x = df[df.Outcome==0][column];
        y = df[df.Outcome==1][column];
        plt.hist([x,y])
        plt.xlabel(xLabel,fontsize=15)
        plt.ylabel('Frequency of people',fontsize=15)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.show()
        
        if typeVar == 'categorical':
            plt.figure(figsize=[15,6])
            if order == '':
                order = pd.unique(df[column])
            sns.barplot(x=column, y='Outcome', data=df, estimator=np.mean, order=order)
            plt.ylabel('norm(no. people)',fontsize=15)
            plt.xlabel(xLabel,fontsize=15)
            plt.show()
        
        if typeVar == 'numerical':
            plt.figure(figsize=[15,6])
            sns.boxplot(x='Outcome', y=column, data=df)
            plt.ylabel(xLabel,fontsize=15)
            plt.xlabel('D status',fontsize=15)
            plt.show()


plotUnivariateAnalysis('Pregnancies', ' Pregnancies', 'numerical','' )

plotUnivariateAnalysis('Age', ' Age', 'numerical','' )
# wit age betweeb 20-30-40 heigher rate of being dibetic 

df.Pregnancies.unique()

df.Age.unique()

df.columns

bins = [20,30,40,50,60,70]
binees=[5,10,15]
df['binned'] = pd.cut(df['Age'], bins)
df['preg_bineed']=pd.cut(df['Pregnancies'],binees)
df.binned
plt.style.use('ggplot')

df.groupby(['Outcome','preg_bineed','binned'])\
      .binned.count().unstack().plot.bar(legend=True)

plt.show()
#highest rate of people getting the problem are of age 40-50 and number of pregnancies done by them are between 5-10.


#check for outliers
df.describe(percentiles=[.25, .5, .75, .90, .95, .99])

df.isnull().sum()

round(100*(df.isnull().sum()/len(df.index)), 2)

df.Outcome.unique()

from sklearn.model_selection import train_test_split

X=df.drop(['Outcome','preg_bineed','binned'],axis=1)
X.head()

y = df['Outcome']

y.head()

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']] = scaler.fit_transform(X_train[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']])

X_train.head()

outcome = (sum(df['Outcome'])/len(df['Outcome'].index))*100
outcome

# Importing matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(df.corr(),annot = True)
plt.show()

import statsmodels.api as sm

# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE
rfe = RFE(logreg, 15)             # running RFE with 13 variables as output
rfe = rfe.fit(X_train, y_train)

list(zip(X_train.columns, rfe.support_, rfe.ranking_))

from statsmodels.stats.outliers_influence import variance_inflation_factor

col = X_train.columns[rfe.support_]

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

col = col.drop('SkinThickness', 1)
col

# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()

col = col.drop('Insulin', 1)
col

# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()

col = col.drop('BloodPressure', 1)
col

# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()

y_train_pred = res.predict(X_train_sm).values.reshape(-1)

y_train_pred[:10]

y_train_pred_final = pd.DataFrame({'outcomess':y_train.values, 'outcome_prob':y_train_pred})
y_train_pred_final['id'] = y_train.index
y_train_pred_final.head()

y_train_pred_final['predicted'] = y_train_pred_final.outcome_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()

from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.outcomess, y_train_pred_final.predicted )
print(confusion)

print(metrics.accuracy_score(y_train_pred_final.outcomess, y_train_pred_final.predicted))

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

fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.outcomess, y_train_pred_final.outcome_prob, drop_intermediate = False )

draw_roc(y_train_pred_final.outcomess, y_train_pred_final.outcome_prob)

# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.outcome_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()

# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.outcomess, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)

# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()

y_train_pred_final['final_predicted'] = y_train_pred_final.outcome_prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()

# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.outcomess, y_train_pred_final.final_predicted)

confusion2 = metrics.confusion_matrix(y_train_pred_final.outcomess, y_train_pred_final.final_predicted )
confusion2

from sklearn.metrics import precision_recall_curve

y_train_pred_final.outcomess, y_train_pred_final.predicted

p, r, thresholds = precision_recall_curve(y_train_pred_final.outcomess, y_train_pred_final.outcome_prob)

plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()

X_test[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']] = scaler.transform(X_test[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']])

X_test = X_test[col]
X_test.head()

X_test_sm = sm.add_constant(X_test)

y_test_pred = res.predict(X_test_sm)

y_test_pred[:10]

# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)

# Let's see the head
y_pred_1.head()

# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)

# Putting CustID to index
y_test_df['ID'] = y_test_df.index

# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)

# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)

y_pred_final.head()

# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'outcomes_prob'})

# Let's see the head of y_pred_final
y_pred_final.head()

y_pred_final['final_predicted'] = y_pred_final.outcomes_prob.map(lambda x: 1 if x > 0.42 else 0)

# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Outcome, y_pred_final.final_predicted)


