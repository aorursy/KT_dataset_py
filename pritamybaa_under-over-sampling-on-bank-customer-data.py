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

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.offline as po

import plotly.graph_objs as go



import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/bank-customer-churn-modeling/Churn_Modelling.csv")

df.drop(['RowNumber','CustomerId'], axis=1, inplace=True)

df.head()
df.isnull().sum()
df.dtypes
df['Exited'].value_counts()
sns.countplot(df['Exited'])
df['Exited'].value_counts().keys().tolist(), df['Exited'].value_counts().values
# visulaize the Exited data using plotly

plot_by_exited_labels = df['Exited'].value_counts().keys().tolist()

plot_by_exited_values = df['Exited'].value_counts().values.tolist()



data = [

    go.Pie(labels=plot_by_exited_labels,

          values=plot_by_exited_values,

          hole=.6)

]



plot_layout = go.Layout(dict(title="Customer Churn"))



fig = go.Figure(data=data, layout=plot_layout)

po.iplot(fig)
df.groupby("Exited").mean()
categorical_feat = ['Geography','Tenure','NumOfProducts',"HasCrCard","HasCrCard","IsActiveMember"]

for cat in categorical_feat:

    print(cat," : ")

    print(df[cat].value_counts())

    sns.countplot(df[cat])

    plt.show() 
print(df['Gender'].value_counts()/df.shape[0])

sns.countplot(df['Gender'])

plt.show() 
# Gender vs Exited

# total percentage exited in by each gender

plot_by_gender = df.groupby('Gender')['Exited'].mean().reset_index()

print(plot_by_gender)



plot_data = [

    go.Bar(

    x = plot_by_gender['Gender'],

    y = plot_by_gender['Exited'])

]

plot_layout = go.Layout(dict(title="% of Exited customers in each gender"))

fig = go.Figure(data = plot_data, layout=plot_layout)

po.iplot(fig)
print(df['Geography'].value_counts())

sns.countplot(df['Geography'])

plt.show() 
# Geography vs Exited in percentage

# total percentage of Exited customer percentage in each Geography

plot_by_geo = df.groupby('Geography')['Exited'].mean().reset_index()

print(plot_by_geo)



plot_data = [

    go.Bar(

    x = plot_by_geo['Geography'],

    y = plot_by_geo["Exited"])

]

plot_layout = go.Layout(dict(title="Percentage of CUstomer Exited in each Geography"))

fig = go.Figure(data=plot_data, layout=plot_layout)

po.iplot(fig)
print(df['NumOfProducts'].value_counts())

sns.countplot(df['NumOfProducts'])
df['NumOfProducts'].value_counts()/df.shape[0]*100
sns.countplot(df['NumOfProducts'],hue=df['Exited'])
pd.crosstab(index=df['Exited'], columns=df['NumOfProducts'])
# lets draw % of Exited Customer along with NumOfProducts used

plot_by_numOfProducts = df.groupby("NumOfProducts")["Exited"].mean().reset_index()

print(plot_by_numOfProducts)



plot_data = [

    go.Bar(

    x = plot_by_numOfProducts["NumOfProducts"],

    y = plot_by_numOfProducts["Exited"])

]

plot_layout = go.Layout(dict(title = "% Exited customers with NumOfProducts used"))

fig = go.Figure(data=plot_data,

               layout=plot_layout)

po.iplot(fig)
print(df['HasCrCard'].value_counts())

sns.countplot(df['HasCrCard'], hue=df['Exited'])
# lets plot % of Exited customer with HasCrCard

plot_with_hasCrCard = df.groupby("HasCrCard")["Exited"].mean().reset_index()

print(plot_with_hasCrCard)



plot_data = [

    go.Bar(

    x=plot_with_hasCrCard['HasCrCard'],

    y= plot_with_hasCrCard['Exited'])

]

plot_layout = go.Layout(dict(title="% Exited customer with HasCrCard"))

fig = go.Figure(data=plot_data, layout=plot_layout)

po.iplot(fig)
# Balance

for i in [0,1]:

    sns.distplot(df[df['Exited']==i]['Balance'])

    plt.show()
df.corr()
plt.figure(figsize=(9,9))

sns.heatmap(df.corr(), annot=True)
df.drop('Surname', axis=1, inplace=True)

df.head()
df = pd.get_dummies(data=df, columns=['Geography','Gender','HasCrCard','IsActiveMember'])

df.head()
x = df.drop('Exited', axis=1)

y = df['Exited']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x[['CreditScore','Age','Tenure','Balance','EstimatedSalary','NumOfProducts']] = scaler.fit_transform(x[['CreditScore','Age','Tenure','Balance','EstimatedSalary','NumOfProducts']])
x.head()
from sklearn.model_selection import train_test_split

from collections import Counter



x_train, x_test,y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train.shape, x_test.shape, Counter(y_train), Counter(y_test)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
def generated_report(y_actual, y_pred):

    print("Accuracy : ", accuracy_score(y_actual, y_pred))

    print(classification_report(y_actual, y_pred))

    

def generated_roc_auc_curve(model, x_test):

    y_pred_proba = model.predict_proba(x_test)[:, 1]

    fpr, tpr, thresh = roc_curve(y_test, y_pred_proba)

    auc = roc_auc_score(y_test,  y_pred_proba)

    plt.plot(fpr, tpr, label='AUC: '+str(auc))

    plt.legend()

    plt.show()

    

def Log_Reg_Model():

    lr = LogisticRegression()

    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)

    generated_report(y_test, y_pred)

    generated_roc_auc_curve(lr, x_test)
Log_Reg_Model()
from sklearn.linear_model import LogisticRegression

from imblearn.under_sampling import RandomUnderSampler

from collections import Counter
sampler = RandomUnderSampler(sampling_strategy=1, replacement=False)

x_new, y_new = sampler.fit_resample(x, y)



Counter(y_new), x_new.shape
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.2, stratify=y_new)

Counter(y_train), Counter(y_test)
Log_Reg_Model()
from imblearn.over_sampling import RandomOverSampler



x_new, y_new = RandomOverSampler(sampling_strategy=0.8).fit_resample(x, y)

Counter(y_new)
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.2, stratify=y_new)
Log_Reg_Model()
from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline



smote = SMOTE(sampling_strategy=0.87)

x_new, y_new = smote.fit_resample(x, y)



print(Counter(y))

print(Counter(y_new))



x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.2, stratify=y_new)

Counter(y_train), Counter(y_test)
Log_Reg_Model()
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from imblearn.pipeline import Pipeline

from collections import Counter

from sklearn.linear_model import LogisticRegression





cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

pipeline = Pipeline(steps=[

    ("over",SMOTE(sampling_strategy=0.6)),

    ("under", RandomUnderSampler(sampling_strategy=0.9))

])
print("Before: ", Counter(y))



x_new, y_new = pipeline.fit_resample(x, y)

print("After: ",Counter(y_new))
x_new.shape, y_new.shape
# again

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

pipeline = Pipeline(steps=[

    ("over",SMOTE(sampling_strategy=0.6)),

    ("under", RandomUnderSampler(sampling_strategy=0.9)), 

    ("model", LogisticRegression())

])



#evaluate pipeline

scores = cross_val_score(pipeline, x, y, scoring='roc_auc', cv=cv)

print(scores)

print("mean roc auc: ", np.mean(scores))
pipeline = Pipeline(steps=[

    ("over",SMOTE(sampling_strategy=0.6)),

    ("under", RandomUnderSampler(sampling_strategy=0.9))

])



x_new, y_new = pipeline.fit_resample(x, y)

x_new.shape, y_new.shape
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.2, stratify=y_new)

Counter(y_train), Counter(y_test)
Log_Reg_Model()