import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

pd.set_option('display.max_columns',None)
import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.tools as tls

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

import squarify
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
prima=pd.read_csv("/kaggle/input/prima-diabetes/prima_diabetes.csv")
prima.shape
prima.head()
list(prima.columns)
prima.describe()
## Data Exploration

#plt.figure(figsize=(12,5))

print(prima.corr()['Outcome'])

sns.heatmap(prima.corr(),annot=True)
prima.isnull().sum()
prima[prima['Glucose']==0]
prima[prima['BloodPressure']==0]
prima[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = prima[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
prima.isnull().sum()
# Define missing plot to detect all missing values in dataset

def missing_plot(dataset, key) :

    null_feat = pd.DataFrame(len(dataset[key]) - dataset.isnull().sum(), columns = ['Count'])

    percentage_null = pd.DataFrame((len(dataset[key]) - (len(dataset[key]) - dataset.isnull().sum()))/len(dataset[key])*100, columns = ['Count'])

    percentage_null = percentage_null.round(2)



    trace = go.Bar(x = null_feat.index, y = null_feat['Count'] ,opacity = 0.8, text = percentage_null['Count'],  textposition = 'auto',marker=dict(color = '#7EC0EE',

            line=dict(color='#000000',width=1.5)))



    layout = dict(title =  "Missing Values (count & %)")



    fig = dict(data = [trace], layout=layout)

    py.iplot(fig)
# Plotting 

missing_plot(prima, 'Outcome')
prima['Glucose'].median()
prima['Glucose'].mean()
# patient who is suffering with dIABETES Will have more Glucose level.

# patient who is not suffering with dIABETES Will have less Glucose level.
prima.groupby('Outcome').agg({'Glucose':'median'}).reset_index()
def median_target(var):   

    temp = prima[prima[var].notnull()]

    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()

    return temp
median_target('Insulin')
prima.loc[(prima['Outcome'] == 0 ) & (prima['Insulin'].isnull()), 'Insulin'] = 102.5

prima.loc[(prima['Outcome'] == 1 ) & (prima['Insulin'].isnull()), 'Insulin'] = 169.5
median_target('Glucose')
prima.loc[(prima['Outcome'] == 0 ) & (prima['Glucose'].isnull()), 'Glucose'] = 107.0

prima.loc[(prima['Outcome'] == 1 ) & (prima['Glucose'].isnull()), 'Glucose'] = 140.0
median_target('SkinThickness')
prima.loc[(prima['Outcome'] == 0 ) & (prima['SkinThickness'].isnull()), 'SkinThickness'] = 27.0

prima.loc[(prima['Outcome'] == 1 ) & (prima['SkinThickness'].isnull()), 'SkinThickness'] = 32.0
median_target('BloodPressure')
prima.loc[(prima['Outcome'] == 0 ) & (prima['BloodPressure'].isnull()), 'BloodPressure'] = 70.0

prima.loc[(prima['Outcome'] == 1 ) & (prima['BloodPressure'].isnull()), 'BloodPressure'] = 74.5
median_target('BMI')
prima.loc[(prima['Outcome'] == 0 ) & (prima['BMI'].isnull()), 'BMI'] = 30.1

prima.loc[(prima['Outcome'] == 1 ) & (prima['BMI'].isnull()), 'BMI'] = 34.3
missing_plot(prima, 'Outcome')
plt.style.use('ggplot') # Using ggplot2 style visuals 

f, ax = plt.subplots(figsize=(11, 15))

ax.set_facecolor('#fafafa')

ax.set(xlim=(-.05, 200))

plt.ylabel('Variables')

plt.title("Overview Data Set")

ax = sns.boxplot(data = prima, 

  orient = 'h', 

  palette = 'Set2')
median_target('Glucose')
prima.head()
sns.distplot(prima['Age'])
# Standard Scaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

var=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

prima[var] = scaler.fit_transform(prima[var])

prima.head()
labels = "Diabetes", "Non Diabetes"

plt.title('Diabetes Status')

plt.ylabel('Condition')

prima['Outcome'].value_counts().plot.pie(explode = [0, 0.25], autopct = '%1.2f%%',

                                                shadow = True, labels = labels)
prima.head()
y=prima['Outcome']

X=prima.drop('Outcome',axis=1)
# Splitting the data into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
y_train
X_train.shape
X_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logreg=LogisticRegression()

logreg.fit(X_train,y_train)

print(logreg.intercept_)

print(logreg.coef_)

print(metrics.accuracy_score(y_train,logreg.predict(X_train)))
print(logreg.intercept_)
print(logreg.coef_)
metrics.accuracy_score(y_train,logreg.predict(X_train))
import statsmodels.api as sm

model = sm.GLM(y_train,(sm.add_constant(X_train)),family=sm.families.Binomial()).fit()

print(model.summary())
from sklearn import metrics

y_train_pred=model.predict(sm.add_constant(X_train))

y_train_pred=round(y_train_pred).astype('int')

metrics.accuracy_score(y_train,y_train_pred)
X_test.head()
y_test_pred=model.predict(sm.add_constant(X_test))
PredictedOutcome=pd.DataFrame(y_test_pred,columns=['PredicbtedOutcome'])
PredictedOutcome.head()
Test=pd.concat([X_test,PredictedOutcome],axis=1)
Test.describe()
Test.head()
Test['Outcome']=Test['PredicbtedOutcome'].apply(lambda x: 1 if x > 0.71 else 0)
Test.head()
Test.shape
sns.countplot(Test['Outcome'])
labels = "Diabetes", "Non Diabetes"

plt.title('Diabetes Status')

plt.ylabel('Condition')

Test['Outcome'].value_counts().plot.pie(explode = [0, 0.25], autopct = '%1.2f%%',

                                                shadow = True, labels = labels)
from sklearn import metrics

metrics.confusion_matrix(y_test,Test['Outcome'])
TN= 135

TP=38

FN=43

FP=15
Accuracy= (TP+TN)/(TP+FN+FP+TN)  

print(Accuracy)
Precision = TP / (TP+FP) 

print(Precision)
Recall = TP/(TP+FN) 

print(Recall)
Specificity = TN / (TN+FP)

print(Specificity)
F1_Score = 2 * Precision * Recall / (Precision + Recall)

print(F1_Score)
## False Positive Rate or Fall Out or Probability of False Alarm

FPR= FP / (FP+ TN )

print(FPR)
# False Negative Rate or Miss Rate:

FNR = FN/(FN+TP)

print(FNR)
Prevalence=   (TP + FN) / (TP + TN + FP + FN) 

print(Prevalence)
FPR,TPR,thresholds=metrics.roc_curve(y_test,Test['Outcome'])
plt.plot(FPR,TPR)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel("FPR  or (1- Specificity)")

plt.ylabel("TPR or Sensitivity")

plt.title("ROC - Receiver Operating Characteristics")
AUC=metrics.accuracy_score(y_test,Test['Outcome'])

print(AUC)
# ROC on Train data

FPR,TPR,thresholds=metrics.roc_curve(y_train,y_train_pred)

plt.plot(FPR,TPR)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel("FPR  or (1- Specificity)")

plt.ylabel("TPR or Sensitivity")

plt.title("ROC - Receiver Operating Characteristics")
AUC=metrics.accuracy_score(y_train,y_train_pred)

print(AUC)
Test.head()
numbers = [float(x)/10 for x in range(11) ]

for i in numbers:

    Test[i]=Test.PredicbtedOutcome.map(lambda x: 1 if x > i else 0)

Test.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(Test['Outcome'], Test[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()

Test['final_predicted'] = Test.PredicbtedOutcome.map( lambda x: 1 if x > 0.7 else 0)

Test.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_test, Test['final_predicted'])
from sklearn.metrics import precision_recall_curve

p, r, thresholds = precision_recall_curve(y_test, Test['final_predicted'])

plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()