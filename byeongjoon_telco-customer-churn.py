import pandas as pd

import numpy as np



#시각화를 위한 라이브러리

import matplotlib as plt

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("/kaggle/input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
df.dtypes
c=0

for i in df['TotalCharges']:

    c=c+1

    if i==" ":

        print(c,i)
df['TotalCharges']=df['TotalCharges'].replace(" ",0)

df['TotalCharges']=df['TotalCharges'].astype('float64')

df['SeniorCitizen']=df['SeniorCitizen'].astype('object')
df.loc[df['gender'] =='Male', 'gender'] = "0"

df.loc[df['gender'] =='Female', 'gender'] = "1"



df.loc[df['Partner']=='Yes','Partner']="1"

df.loc[df['Partner']=='No','Partner']="0"



df.loc[df['Dependents']=='Yes','Dependents']="1"

df.loc[df['Dependents']=='No','Dependents']="0"



df.loc[df['PhoneService']=='Yes','PhoneService']="1"

df.loc[df['PhoneService']=='No','PhoneService']="0"



df.loc[df['MultipleLines']=='Yes','MultipleLines']="1"

df.loc[df['MultipleLines']=='No','MultipleLines']="0"

df.loc[df['MultipleLines']=='No phone service','MultipleLines']="2"



df.loc[df['InternetService']=='DSL','InternetService']="1"

df.loc[df['InternetService']=='Fiber optic','InternetService']="2"

df.loc[df['InternetService']=='No','InternetService']="0"



df.loc[df['OnlineBackup']=='Yes','OnlineBackup']="1"

df.loc[df['OnlineBackup']=='No','OnlineBackup']="0"

df.loc[df['OnlineBackup']=='No internet service','OnlineBackup']="2"



df.loc[df['OnlineSecurity']=='Yes','OnlineSecurity']="1"

df.loc[df['OnlineSecurity']=='No','OnlineSecurity']="0"

df.loc[df['OnlineSecurity']=='No internet service','OnlineSecurity']="2"



df.loc[df['DeviceProtection']=='Yes','DeviceProtection']="1"

df.loc[df['DeviceProtection']=='No','DeviceProtection']="0"

df.loc[df['DeviceProtection']=='No internet service','DeviceProtection']="2"



df.loc[df['TechSupport']=='Yes','TechSupport']="1"

df.loc[df['TechSupport']=='No','TechSupport']="0"

df.loc[df['TechSupport']=='No internet service','TechSupport']="2"



df.loc[df['StreamingTV']=='Yes','StreamingTV']="1"

df.loc[df['StreamingTV']=='No','StreamingTV']="0"

df.loc[df['StreamingTV']=='No internet service','StreamingTV']="2"



df.loc[df['StreamingMovies']=='Yes','StreamingMovies']="1"

df.loc[df['StreamingMovies']=='No','StreamingMovies']="0"

df.loc[df['StreamingMovies']=='No internet service','StreamingMovies']="2"



df.loc[df['Contract']=='Month-to-month','Contract']="0"

df.loc[df['Contract']=='One year','Contract']="1"

df.loc[df['Contract']=='Two year','Contract']="2"



df.loc[df['PaperlessBilling'] =='Yes', 'PaperlessBilling'] = "0"

df.loc[df['PaperlessBilling'] =='No', 'PaperlessBilling'] = "1"



df.loc[df['PaymentMethod'] =='Electronic check', 'PaymentMethod'] = "0"

df.loc[df['PaymentMethod'] =='Mailed check', 'PaymentMethod'] = "1"

df.loc[df['PaymentMethod'] =='Bank transfer (automatic)', 'PaymentMethod'] = "2"

df.loc[df['PaymentMethod'] =='Credit card (automatic)', 'PaymentMethod'] = "3"



df.loc[df['Churn']=='Yes','Churn']="1"

df.loc[df['Churn']=='No','Churn']="0"
df.isnull().sum()
sns.boxplot(df['tenure']);
sns.boxplot(df['MonthlyCharges']);
sns.boxplot(df['TotalCharges']);
def barplot_percentages(feature, orient='v', axis_name="percentage of customers"):

    ratios = pd.DataFrame()

    g = df.groupby(feature)["Churn"].value_counts().to_frame()

    g = g.rename({"Churn": axis_name}, axis=1).reset_index()

    g[axis_name] = g[axis_name]/len(df)

    if orient == 'v':

        ax = sns.barplot(x=feature, y= axis_name, hue='Churn', data=g, orient=orient)

        ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])

    else:

        ax = sns.barplot(x= axis_name, y=feature, hue='Churn', data=g, orient=orient)

        ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])

    ax.plot()

barplot_percentages("SeniorCitizen")
df.head()

df.dtypes
g = sns.FacetGrid(df, row='SeniorCitizen', col="gender", hue="Churn")

g.map(plt.scatter, "tenure", "MonthlyCharges", alpha=0.6)

g.add_legend();
fig, axis = plt.subplots(1, 2, figsize=(12,4))

axis[0].set_title("Has partner")

axis[1].set_title("Has dependents")

axis_y = "percentage of customers"

# Plot Partner column

gp_partner = df.groupby('Partner')["Churn"].value_counts()/len(df)

gp_partner = gp_partner.to_frame().rename({"Churn": axis_y}, axis=1).reset_index()

ax = sns.barplot(x='Partner', y= axis_y, hue='Churn', data=gp_partner, ax=axis[0])

# Plot Dependents column

gp_dep = df.groupby('Dependents')["Churn"].value_counts()/len(df)

gp_dep = gp_dep.to_frame().rename({"Churn": axis_y}, axis=1).reset_index()

ax = sns.barplot(x='Dependents', y= axis_y, hue='Churn', data=gp_dep, ax=axis[1])
plt.figure(figsize=(9, 4.5))

barplot_percentages("MultipleLines", orient='h')
cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

df1 = pd.melt(df[df["InternetService"] != "No"][cols]).rename({'value': 'Has service'}, axis=1)

plt.figure(figsize=(10, 4.5))

ax = sns.countplot(data=df1, x='variable', hue='Has service')

ax.set(xlabel='Additional service', ylabel='Num of customers')

plt.show()
def kdeplot(feature):

    plt.figure(figsize=(9, 4))

    plt.title("KDE for {}".format(feature))

    ax0 = sns.kdeplot(df[df['Churn'] == '0'][feature].dropna(), color= 'navy', label= 'Churn: No')

    ax1 = sns.kdeplot(df[df['Churn'] == '1'][feature].dropna(), color= 'orange', label= 'Churn: Yes')

    

kdeplot('tenure')

kdeplot('MonthlyCharges')

kdeplot('TotalCharges')
relation=df.corr()

relation
train=df.head(5001)

test=df.tail(2032)



# train.to_csv("train.csv",index=False)

# test.to_csv("test.csv",index=False)



test2=test[["tenure",'InternetService']]
# predictors = ['SeniorCitizen', 'Partner', 'Dependents','tenure','InternetService','OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling','PaymentMethod']

predictors=["tenure",'InternetService']

 

x = train[predictors]

y = train["Churn"]
from sklearn.tree import DecisionTreeClassifier



decision = DecisionTreeClassifier().fit(x,y)

print(decision)
from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(n_jobs=-1).fit(x,y)

print(forest)
from sklearn.model_selection import cross_val_score

# from sklearn.model_selection import KFold



decision_score = cross_val_score(decision, x, y, cv=5).mean()

forest_score = cross_val_score(forest, x, y, cv=5).mean()





print("DecisionTree = {0:.6f}".format(decision_score))

print("RandomForest = {0:.6f}".format(forest_score))
prediction=decision.predict(test2)

submission = pd.DataFrame({

        "customerID": test["customerID"],

        "Churn_test": test["Churn"],

        "Churn_predict": prediction

    })

submission.loc[submission['Churn_predict']==submission['Churn_test'],'Predict']=1

submission.loc[submission['Churn_predict']!=submission['Churn_test'],'Predict']=0



# submission.to_csv('submission.csv', index=False)



print("predict percent is %.2f %%"%(sum(submission['Predict'])/len(submission)*100))
from sklearn.linear_model import LogisticRegression 

from sklearn.cross_validation import train_test_split 



X=df[["MonthlyCharges","TotalCharges",'tenure']]

Y=df['Churn'] 



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)



log_clf = LogisticRegression()

log_clf.fit(X_train,Y_train)

log_clf.score(X_test, Y_test)
test3=test[["MonthlyCharges","TotalCharges","tenure"]]

# test3=test["MonthlyCharges"]



p=log_clf.predict(test3)

submission_log = pd.DataFrame({

        "customerID": test["customerID"],

        "Churn_test": test["Churn"],

        "Churn_predict": prediction

    })

submission_log.loc[submission['Churn_predict']==submission['Churn_test'],'Predict']=1

submission_log.loc[submission['Churn_predict']!=submission['Churn_test'],'Predict']=0



# submission_log.to_csv('submission_log.csv', index=False)



print("predict percent is %.2f %%"%(sum(submission_log['Predict'])/len(submission)*100))