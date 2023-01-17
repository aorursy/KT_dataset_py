import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

sns.set_style('whitegrid')
warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()
def getting_ranges(df):
    
    bins_ssc = (39,50,60,70,80,90)
    group_names_ssc = ['40-50','50-60','60-70','70-80','80-90']
    
    bins_hsc = (29,40,50,60,70,80,90,100)
    group_names_hsc = ['30-40','40-50','50-60','60-70','70-80','80-90','90-100'] 
    
    bins_degree = (49,60,70,80,90,100)
    group_names_degree = ['50-60','60-70','70-80','80-90','90-100']
    
    bins_etest = (49,60,70,80,90,100)
    group_names_etest = ['50-60','60-70','70-80','80-90','90-100'] 
    
    bins_mba = (49,60,70,80)
    group_names_mba = ['50-60','60-70','70-80']
    
    bins_salary = (199999,300000,400000,500000,600000,700000,800000,900000,1000000)
    group_names_salary = ['2 Lakh-3 Lakh','3 Lakh-4 Lakh','4 Lakh-5 Lakh','5 Lakh-6 Lakh','6 Lakh-7 Lakh','7 Lakh-8 Lakh','8 Lakh-9 Lakh','9 Lakh-10 Lakh']    
  
    df['ssc_perc_range'] = pd.cut(df.ssc_p,bins_ssc,labels=group_names_ssc)
    df['hsc_perc_range'] = pd.cut(df.hsc_p,bins_hsc,labels=group_names_hsc)
    df['degree_perc_range'] = pd.cut(df.degree_p,bins_degree,labels=group_names_degree)
    df['etest_perc_range'] = pd.cut(df.etest_p,bins_etest,labels=group_names_etest)
    df['mba_perc_range'] = pd.cut(df.mba_p,bins_mba,labels=group_names_mba)
    df['salary_range'] = pd.cut(df.salary,bins_salary,labels=group_names_salary)
getting_ranges(df)
pd.set_option('max_columns',50)
df.head()
plt.figure(figsize=(9,5))
sns.countplot(data=df,x='ssc_perc_range',hue='status',hue_order=['Not Placed','Placed'],palette='Set1')
plt.figure(figsize=(9,5))
sns.countplot(data=df,x='hsc_perc_range',hue='status',hue_order=['Not Placed','Placed'],palette='Set1')
plt.figure(figsize=(9,5))
sns.countplot(data=df,x='degree_perc_range',hue='status',hue_order=['Not Placed','Placed'],palette='Set1')
plt.figure(figsize=(9,5))
sns.countplot(data=df,x='mba_perc_range',hue='status',hue_order=['Not Placed','Placed'],palette='Set1')
plt.figure(figsize=(9,5))
sns.countplot(data=df,x='etest_perc_range',hue='status',hue_order=['Not Placed','Placed'],palette='Set1')
plt.figure(figsize=(9,5))
sns.countplot(data=df,x='workex',hue='status',hue_order=['Not Placed','Placed'],palette='Set1')
df.salary_range.value_counts().plot.bar(
figsize=(16,5),
color = 'green',
fontsize =14 
)
data_ml = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data_ml.drop(['sl_no','gender','ssc_b','hsc_b'],inplace=True,axis=1)
data_ml.head()
hsc_stream = pd.get_dummies(data_ml.hsc_s,prefix='hsc_stream',drop_first=True)
hsc_stream.head()
degree_trade = pd.get_dummies(data_ml.degree_t,prefix='degree_trade',drop_first=True)
degree_trade.head()
workex = pd.get_dummies(data_ml.workex,prefix='workex',drop_first=True)
workex.head()
specialisation = pd.get_dummies(data_ml.specialisation,prefix='specialisation',drop_first=True)
specialisation.head()
data_ml = pd.concat([data_ml,hsc_stream,degree_trade,workex,specialisation],axis=1)
data_ml.drop(['hsc_s','degree_t','workex','specialisation'],axis=1,inplace=True)
data_ml.head()
X = data_ml.drop(['status','salary'],axis=1)
X.head()
y = data_ml['status']
y.head()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import GridSearchCV,cross_val_score,train_test_split
Log_model = LogisticRegression()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20)

Log_model.fit(X_train,y_train)
y_pred = Log_model.predict(X_test)
print('Accuracy -> ',accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix : \n',confusion_matrix(y_test,y_pred))
print('Accuracy -> ',cross_val_score(Log_model,X,y,cv=10).mean()*100,'%')
parameters = [{'penalty' : ['l2','l1'],
               'C' : np.logspace(0, 4, 10),
               'class_weight' : ['balanced',None],
               'multi_class' : ['ovr','auto'],
               'max_iter' : np.arange(50,130,10)},
              {'penalty' : ['l2'],
               'C' : np.logspace(0, 4, 10),
               'class_weight' : ['balanced',None],
               'max_iter' : np.arange(50,130,10),
               'solver' : ['newton-cg','saga','sag','liblinear'],
               'multi_class' : ['ovr','auto']}]
Log_grid = GridSearchCV(estimator=Log_model,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)
Log_grid.fit(X,y)
Log_grid.best_params_
Log_model_grid = LogisticRegression(C= 10000.0,
 class_weight= 'balanced',
 max_iter= 90,
 multi_class= 'ovr',
 penalty= 'l2')
print('Accuracy after doing HYPERPARAMETER Tuning -> ',cross_val_score(Log_model_grid,X,y,cv=10).mean()*100,'%')
Log_model_grid.fit(X,y)
model_prediction = Log_model_grid.predict([[80,75,65,64,63,0,1,0,1,1,0]])
print('According to our MODEL, this CANDIDATE will get -> ',model_prediction)