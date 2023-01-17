# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
customers = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
customers.head(10)
customers.any().isnull()
#no null values
customers.info()
#total charges is an object

#customers['TotalCharges'] = pd.to_numeric(customers['TotalCharges'], errors='raise')   
#converting total charges to float, found non integer elements ' '
nonintegers = customers[customers['TotalCharges'] == ' '] 
to_drop = nonintegers.index.tolist()
to_drop
customers = customers.drop(to_drop, axis='index')
customers['TotalCharges'] = pd.to_numeric(customers['TotalCharges'], errors='raise') 
customers.any().isnull()
#no NaN's in TotalCharges
#i want to convert all the yes/no's to 1's and 0's in churn
customers['Churn'] = customers['Churn'].map(dict({'Yes':1,'No':0}))

import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

demographics = ['gender', 'Partner', 'SeniorCitizen', 'Dependents']


for i in demographics:
    trace1 = go.Bar(
        x=customers.groupby(i)['Churn'].sum().reset_index()[i],
        y=customers.groupby(i)['Churn'].sum().reset_index()['Churn'],
        name= i
    )

    data = [trace1]
    layout = go.Layout(
        title= i,
        yaxis=dict(
        title='Churn'
        ),
        barmode='group',
        autosize=True,
    width=600,
    height=600,
    margin=go.Margin(
        l=70,
        r=70,
        b=100,
        t=100,
        pad=8
    )
    )
    fig = go.Figure(data=data, layout=layout)
    py.offline.iplot(fig)

seniors = customers.loc[customers['SeniorCitizen'] == 1]

nonseniors = customers.loc[customers['SeniorCitizen'] == 0]

hist_data = [seniors.groupby('tenure')['Churn'].sum(),nonseniors.groupby('tenure')['Churn'].sum()]
group_labels = ['Seniors', 'Non-Seniors']

import plotly.figure_factory as ff
fig = ff.create_distplot(hist_data, group_labels,bin_size=[1,1], curve_type='normal', show_rug=False)
py.offline.iplot(fig)
customers['InternetService'].value_counts()
customers.groupby('InternetService')['Churn'].sum()
#Nearly half of the customers who took fiber optic left
seniors.groupby('InternetService')['Churn'].sum()
mask = ['DSL','Fiber optic']
internet = customers[customers['InternetService'].isin(mask)]
categories = ['InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract',
             'PaperlessBilling','PaymentMethod']
internet.info()
dumm = pd.get_dummies(internet,columns=categories,drop_first=True)
corr = dumm.corr(method='spearman')
corr
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr,cmap=cmap, mask=mask,vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
y = dumm.iloc[:,10].values
dumm = dumm.drop(['customerID','Churn'],axis=1)

X = dumm
y


mask = ['gender','Partner','Dependents','PhoneService','MultipleLines']
X = pd.get_dummies(dumm,columns=mask,drop_first=True)

X.head()
#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 4)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred1)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred1,y_test)
accuracy

print (cm)
print (accuracy)
import lightgbm as lgb

d_train = lgb.Dataset(X_train, label=y_train)

params = {}
params['learning_rate'] = 0.2
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['num_leaves'] = 5
params['max_depth'] = 15


clf = lgb.train(params, d_train, 100)

#Prediction
y_pred2=clf.predict(X_test)

#convert into binary values
for i in range(0,552):
    if y_pred2[i]>=.52:       
       y_pred2[i]=1
    else:  
       y_pred2[i]=0
#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)
#Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred2,y_test)
cm
accuracy
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred3 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred3)
#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred3)
#Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred3,y_test)
cm
accuracy