#Importing required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode,plot,iplot
cf.go_offline()
init_notebook_mode(connected=True)
%matplotlib inline
#Reading the csv file into a dataframe

df=pd.read_csv('../input/ks-projects-201801.csv')
df.info()
df.head()
#Dropping categories that might not be needed 
df.drop(['ID','name'],axis=1,inplace=True)
df.describe()
#Generating a count plot using seaborn to idenfity the state of all the kickstarter projects
plt.figure(figsize=(10,7))
sns.countplot(df['state'])
plt.xlabel('State')
plt.ylabel('Count')
plt.title('Count plot of the state of kickstarter projects')
# Calculating the 'success' and 'failure' rate of the projects

Percent_fail=df[df['state']=='failed']['state'].value_counts()/len(df['state'])*100
Percent_success=df[df['state']=='successful']['state'].value_counts()/len(df['state'])*100

print('Fail rate is',Percent_fail[0])
print('Success rate is',Percent_success[0])
df['main_category'].value_counts().sort_values(ascending=False)
dfsuccess=df[df['state']=='successful']
(dfsuccess['main_category'].value_counts()/df['main_category'].value_counts() *100).sort_values(ascending=False)
# Plotting a stacked bar chart (in percentage) of all the projects colorbed by their state
df1=df.groupby(('main_category','state')).size().unstack().fillna(0) #Creating the count dataframe
df1=df1.divide(df1.iloc[:,:].sum(axis=1),axis=0)*100 #Calculating percentage values of the state of each category
#df1=df1.reindex(index=list(dfdel.sum(axis=1).sort_values(ascending=False).index)) #sorting it based on total projects for each category. looks good while plotting. not necessary if plotting in percentages
df1.iplot(kind='bar',barmode='stack') #creating a stacked bar chart
# Printing the categories based on their failure rate
print(df1.sort_values(by='failed',ascending=False))

# Printing the categories based on their failure rate
print(df1.sort_values(by='successful',ascending=False))
for index,row in df1.sort_values(by='failed',ascending=False).iterrows():
    if(row['successful']>row['failed']):
        print(index,row['successful'],row['failed'])
    
df.groupby(('main_category'))['backers'].sum().sort_values(ascending=False)
df.groupby('main_category')['usd_pledged_real'].sum().sort_values(ascending=False).to_frame().reset_index().iplot(kind='pie',labels='main_category',values='usd_pledged_real')
df[(df['state']=='successful') & (df['usd_pledged_real']-df['usd_goal_real']<0)] 
df[(df['state']=='failed') & (df['usd_pledged_real']-df['usd_goal_real']>=0)] 
df[df['usd_pledged_real']-df['usd_goal_real'] >=0]['state'].value_counts()/df[df['usd_pledged_real']-df['usd_goal_real'] >=0]['state'].value_counts().sum() *100
dfcountry=df.groupby('country')['main_category'].count().to_frame().reset_index()
dfcountry.sort_values(by='main_category',ascending=False).head()
#Creating a 3-digit list of countries for using with choropleth map
countries=pd.Series(list(('AUT','AUS','BEL','CAN','CHE','DEU','DNK','ESP','FRA','GBR','HKG','IRL','ITA','JPN','LUX','MEX','IND','NLD','NOR','NZL','SWE','SGP','USA')))

### Note -- Projects originating from country listed as "N,0" are considered to be originating from India just for the sake of illustration
dfcountry['countries']=countries.values
dfcountry.drop('country',axis=1)
data = dict(
        type = 'choropleth',
        colorscale = 'Rainbow',
        locations = dfcountry['countries'],
        z = dfcountry['main_category'],
        text = dfcountry['countries'],
        colorbar = {'title':'Project Origination Countries'},
      ) 
      
layout = dict(
    title = 'Global view of countries of project origination',
    geo = dict(
        showframe = False,
        projection = {'type':'natural earth'}
    )
)

choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)
#(df2.groupby(['category','state']).size()/df2.groupby(['category','state']).size().groupby('category').sum()*100).unstack('state').fillna(0).iplot(kind='bar',barmode='stack',title='Stacked bar chart of the state of Film&Video Projects',xTitle='Category',yTitle='Percentage of Projects')

df.head()
dfm1=df.drop(['currency','deadline','goal','launched','pledged','usd pledged','usd_pledged_real','usd_goal_real','category'],axis=1)
#Converting countries and categories to dummy variables
countries_dummies=pd.get_dummies(dfm1['country'],drop_first=True)
categories=pd.get_dummies(dfm1['main_category'],drop_first=True)
dfnew=pd.concat([dfm1,countries_dummies,categories],axis=1)
dfnew1=dfnew.drop(['country','main_category'],axis=1)
#Considering only the failed or successeful projects
dfnew2=dfnew1[(dfnew1['state']=='failed') | (dfnew1['state']=='successful')]
x=dfnew2.drop('state',axis=1)
y=dfnew2['state']
from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
reg=DecisionTreeClassifier()
reg.fit(xtrain,ytrain)
ypred=reg.predict(xtest)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))
#Label Binarizing for ROC
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ytestbin=le.fit_transform(ytest)
ypredbin=le.fit_transform(ypred)
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
fpr_dt = dict()
tpr_dt = dict()
roc_auc_dt = dict()
for i in range(2):
    fpr_dt[i], tpr_dt[i], _ = roc_curve(ytestbin, ypredbin)
    roc_auc_dt[i] = auc(fpr_dt[i], tpr_dt[i])


from sklearn.model_selection import cross_val_score
acc=cross_val_score(estimator=reg,X=xtrain,y=ytrain,cv=10)
print(acc)
print('Mean efficiency is ',acc.mean())
print('Standard deviation is ',acc.std())
from sklearn.ensemble import RandomForestClassifier
reg=RandomForestClassifier()
reg.fit(xtrain,ytrain)
ypred=reg.predict(xtest)
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))
#Label Binarizing for ROC
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ytestbin=le.fit_transform(ytest)
ypredbin=le.fit_transform(ypred)
fpr_rf = dict()#Label Binarizing for ROC
tpr_rf = dict()
roc_auc_rf = dict()
for i in range(2):
    fpr_rf[i], tpr_rf[i], _ = roc_curve(ytestbin, ypredbin)
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])
acc=cross_val_score(estimator=reg,X=xtrain,y=ytrain,cv=10)
print(acc)
print('Mean efficiency is ',acc.mean())
print('Standard deviation is ',acc.std())
from sklearn.grid_search import GridSearchCV
param_grid = {'n_estimators': [10, 50, 100, 200]} 
grid = GridSearchCV(reg,param_grid,refit=True,verbose=3)
grid.fit(xtrain,ytrain)
print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_)
yprednew=grid.predict(xtest)
print(confusion_matrix(ytest,yprednew))
print(classification_report(ytest,yprednew))
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ytestbin=le.fit_transform(ytest)
ypredbin=le.fit_transform(yprednew)
fpr_rf_gs= dict()
tpr_rf_gs= dict()#Label Binarizing for ROC
roc_auc_rf_gs= dict()
for i in range(2):
    fpr_rf_gs[i], tpr_rf_gs[i], _ = roc_curve(ytestbin, ypredbin)
    roc_auc_rf_gs[i] = auc(fpr_rf_gs[i], tpr_rf_gs[i])
#print(roc_auc_score(ytestbin, ypredbin))
plt.figure(figsize=(10,7))
plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
plt.plot(fpr_dt[1], tpr_dt[1], label='Decision Tree(area = %0.4f)' % roc_auc_dt[0])
plt.plot(fpr_rf[1], tpr_rf[1], label='Random Forest (area = %0.4f)' % roc_auc_rf[0])
plt.plot(fpr_rf_gs[1], tpr_rf_gs[1], label='Random Forest Optimized (area = %0.4f)' % roc_auc_rf_gs[0])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
