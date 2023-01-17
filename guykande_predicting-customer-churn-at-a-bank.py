# Any results you write to the current directory are saved as output.
# import necessary libraries.note that i will be importing necessary libraries if need all along
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns#visualization
sns.set(style="ticks", color_codes=True)
import matplotlib.ticker as mtick # For specifying the axes tick format 
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization
#Import dataset into and display head of your dataset

churn = pd.read_csv('../input/Churn_Modelling.csv')
churn.head()
#drop unnecessary attributes and display new dataset
churn = churn.drop(["RowNumber","CustomerId","Surname"], axis =1)

#Describe how big is out dataset helps to understand how big will be our analysis and requirements.
print("Rows : ",churn.shape[0])
print("Columns  : ",churn.shape[1])
#I check if there is any NaN values that can bring biased scenario, all column attributes should return false to verify this 
churn.isnull().any()
#count our unique values without duplication of same figure

print ("\nUnique values :  \n",churn.nunique())
#what are our data types
churn.dtypes
#Mean=> the are a lot of average calculations in statistics so i used mean the check the average possibility of attributtes to impact the situation
churn.groupby(['Exited']).mean()
#Let's convert all the categorical variables into dummy variables
df = pd.get_dummies(churn)
df.head()
plt.figure(figsize=(10,4))
df.corr()['Exited'].sort_values(ascending = False).plot(kind='bar')
plt.figure(figsize = (20,10))
sns.heatmap((df.loc[:, ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary','Exited','Geography_France','Geography_Germany','Geography_Spain','Gender_Female','Gender_Male']]).corr(),
            annot=True,linewidths=.5);
# Passing labels and values
lab = churn["Exited"].value_counts().keys().tolist()
val = churn["Exited"].value_counts().values.tolist()

trace = go.Pie(labels = lab ,
               values = val ,
               marker = dict(colors =  [ 'royalblue' ,'lime'],
                             line = dict(color = "white",
                                         width =  0.3)
                            ),
               rotation = 90,
               hoverinfo = "label+value+text",
               hole = .2
              )
layout = go.Layout(dict(title = "Customer churn",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                       )
                  )

data = [trace]
fig = go.Figure(data = data,layout = layout)
py.iplot(fig)
#Categorical attirbutes churn rate
fig, axs = plt.subplots(2, 2, figsize=(15, 8))
sns.countplot(x= churn.Geography, hue = 'Exited' ,data=churn, ax =axs[0][0])
sns.countplot(x=churn.Gender, hue = 'Exited' ,data=churn, ax=axs[1][0])
sns.countplot(x=churn.HasCrCard, hue = 'Exited' ,data=churn, ax=axs[0][1])
sns.countplot(x=churn.IsActiveMember, hue = 'Exited' ,data=churn, ax=axs[1][1])
plt.ylabel('count')

fig, axarr = plt.subplots(3, 2, figsize=(15, 8))
sns.boxplot(y='CreditScore',x = 'Exited', hue = 'Exited',data = churn, ax=axarr[0][0])
sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = churn , ax=axarr[0][1])
sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = churn, ax=axarr[1][0])
sns.boxplot(y='Balance',x = 'Exited', hue = 'Exited',data = churn, ax=axarr[1][1])
sns.boxplot(y='NumOfProducts',x = 'Exited', hue = 'Exited',data = churn, ax=axarr[2][0])
sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = churn, ax=axarr[2][1])
df['BalanceEstimatedSalaryRatio'] = df.Balance/(df.EstimatedSalary)
df['TenureOverAge'] = df.Tenure/(df.Age)
df['CreditScoreOverAge'] = df.CreditScore/(df.Age)
df.head()

con_v=['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary','BalanceEstimatedSalaryRatio','TenureOverAge','CreditScoreOverAge']
minVec = df[con_v].min().copy()
maxVec = df[con_v].max().copy()
df[con_v] = (df[con_v]-minVec)/(maxVec-minVec)
df.head()
# Create Train & Test Data
from sklearn.model_selection import train_test_split
y = df['Exited'].values
x = df.drop(columns = ['Exited'])
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=0)
# Scaling all the variables to a range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
features = x.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(x)
x = pd.DataFrame(scaler.transform(x))
x.columns = features
# Running logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model = LogisticRegression()
result = model.fit(x_train, y_train)
prediction_test = model.predict(x_test)
print (metrics.accuracy_score(y_test, prediction_test))# Print the prediction accuracy
# getting the weights of all the variables on regression model
weights = pd.Series(model.coef_[0],
                 index=x.columns.values)
weights.sort_values()[-13:].plot(kind = 'barh')
weights.sort_values(ascending = False)

from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(x_train, y_train)
# Make predictions
prediction_test = model_rf.predict(x_test)
probs = model_rf.predict_proba(x_test)
print (metrics.accuracy_score(y_test, prediction_test))# Print the prediction accuracy
importances = model_rf.feature_importances_
weights = pd.Series(importances,
                 index=x.columns.values)
weights.sort_values()[-13:].plot(kind = 'barh')
weights.sort_values(ascending = False)
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
model.svm = SVC(kernel='linear') 
model.svm.fit(x_train,y_train)
preds = model.svm.predict(x_test)
metrics.accuracy_score(y_test, preds)# Print the prediction accuracy
from sklearn.neighbors import KNeighborsClassifier
classifiers = [
    KNeighborsClassifier(5),    
]
# iterate over classifiers
for item in classifiers:
    classifier_name = ((str(item)[:(str(item).find("("))]))
    print (classifier_name)
    # Create classifier, train it and test it.
    clf = item
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    score = clf.score(x_test, y_test)
    print (round(score,3),"\n", "- - - - - ", "\n") # Print the prediction accuracy
    

# apply on Random forest body and we will directly display five first customers as per our model
x_test["prob_true"] = prediction_test
df_risky = x_test[x_test["prob_true"] > 0.9]
display(df_risky.head()[["prob_true"]])

df_risky.shape
df_risky.head()

