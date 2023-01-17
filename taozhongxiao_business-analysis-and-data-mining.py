import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/UCI_Credit_Card.csv')

df.sample(3)
df = df.rename(columns={'default.payment.next.month': 'def_pay', 

                        'PAY_0': 'PAY_1'})
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, make_scorer,confusion_matrix,f1_score,roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
y = df['def_pay'].copy()

y.sample(5)
features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',

       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',

       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',

       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

X = df[features].copy()

X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,stratify=y, random_state=42)
param_grid = {'n_estimators': [250,300,350],

              'criterion': ['entropy', 'gini'], 'n_jobs' : [-1]}

grid_forest = GridSearchCV(RandomForestClassifier(), param_grid, scoring = 'roc_auc', cv=5)

%time grid_forest = grid_forest.fit(X_train, y_train)

print(grid_forest.best_estimator_)

print(grid_forest.best_score_)

forest_downsampled = grid_forest.best_estimator_
RF1=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',

                       max_depth=None, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=350,

                       n_jobs=-1, oob_score=False, random_state=None, verbose=0,

                       warm_start=False)

RF1.fit(X_train, y_train)

predictions = RF1.predict_proba(X_test)[:,1]

auc1=roc_auc_score(y_test, predictions)

print(auc1)
def get_feature_importance(classify, ftrs):

    imp = classify.feature_importances_.tolist()

    feat = ftrs

    result = pd.DataFrame({'feat':feat,'score':imp})

    result = result.sort_values(by=['score'],ascending=False)

    return result
RF1.fit(X_train, y_train)

predictions = RF1.predict(X_test)

f1_score(y_true = y_test, y_pred = predictions)

print("-------------")

print("f1 score: {}".format(round(f1_score(y_true = y_test, y_pred = predictions),3)))

print("Accuracy: {}".format(round(accuracy_score(y_true = y_test, y_pred = predictions),3)))

print("-------------")

print(get_feature_importance(RF1, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

pred = len(predictions)



print('True Positives: {}'.format(TP))

print('False Positive: {}'.format(FP))

print('True Negative: {}'.format(TN))

print('False Negative: {}'.format(FN))

print('Precision: {}'.format(round(TP/(TP+FP),2)))

print('Recall: {}'.format(round(TP/(TP+FN),2)))

print('Problematic ratio: {}'.format(round(FN/(FN+TP),2)))
param_grid = {'max_depth': np.arange(5, 10),

             'criterion' : ['gini','entropy'],

             'max_leaf_nodes': [10,20,50,100],

             'min_samples_split': [2,5,10,20,50],

             'class_weight' : ['balanced']}

# create the grid

grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 5, scoring= 'roc_auc')

# the cv option will be clear in a few cells



#training

grid_tree.fit(X_train, y_train)

#let's see the best estimator

print(grid_tree.best_estimator_)

#with its score

print(grid_tree.best_score_)
tree1=DecisionTreeClassifier(class_weight='balanced', criterion='entropy',

                       max_depth=9, max_features=None, max_leaf_nodes=20,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, presort=False,

                       random_state=None, splitter='best')

tree1.fit(X_train, y_train)

predictions = tree1.predict_proba(X_test)[:,1]

auc2=roc_auc_score(y_test, predictions)

print(auc2)
predictions = tree1.predict(X_test)

f1_score(y_true = y_test, y_pred = predictions)

print("-------------")

print("f1 score: {}".format(round(f1_score(y_true = y_test, y_pred = predictions),3)))

print("Accuracy: {}".format(round(accuracy_score(y_true = y_test, y_pred = predictions),3)))

print("-------------")

print(get_feature_importance(tree1, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

pred = len(predictions)



print('True Positives: {}'.format(TP))

print('False Positive: {}'.format(FP))

print('True Negative: {}'.format(TN))

print('False Negative: {}'.format(FN))

print('Precision: {}'.format(round(TP/(TP+FP),2)))

print('Recall: {}'.format(round(TP/(TP+FN),2)))

print('Problematic ratio: {}'.format(round(FN/(FN+TP),2)))
predictions_tree = tree1.predict_proba(X_test)[:,1]

fpr1, tpr1, _ = metrics.roc_curve(y_test,  predictions_tree)



predictions_rf = RF1.predict_proba(X_test)[:,1]

fpr2, tpr2, _ = metrics.roc_curve(y_test,  predictions_rf)





sns.set(style="white",font="Arial",font_scale=1.5)





plt.figure(figsize=(10,10))

plt.plot([0, 1], [0, 1], 'k--')



plt.plot(fpr1,tpr1,label="Decision Tree, auc="+str(round(auc2,2)),color="#E21932")

plt.plot(fpr2,tpr2,label="Random forest, auc="+str(round(auc1,2)),color="black")

plt.legend(loc=4, facecolor='white',fontsize=16)

sns.despine(left=True)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title("ROC")
from sklearn import tree

import graphviz

dot_data = tree.export_graphviz(tree1, out_file=None)  

graph = graphviz.Source(dot_data)  

graph
df.info()
# Categorical variables description

df[['SEX', 'EDUCATION', 'MARRIAGE']].describe()
# Payment delay description

df[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].describe()
fil = (df.PAY_1 == -2) | (df.PAY_1 == -1) | (df.PAY_1 == 0)

df.loc[fil, 'PAY_1'] = 0

fil = (df.PAY_2 == -2) | (df.PAY_2 == -1) | (df.PAY_2 == 0)

df.loc[fil, 'PAY_2'] = 0

fil = (df.PAY_3 == -2) | (df.PAY_3 == -1) | (df.PAY_3 == 0)

df.loc[fil, 'PAY_3'] = 0

fil = (df.PAY_4 == -2) | (df.PAY_4 == -1) | (df.PAY_4 == 0)

df.loc[fil, 'PAY_4'] = 0

fil = (df.PAY_5 == -2) | (df.PAY_5 == -1) | (df.PAY_5 == 0)

df.loc[fil, 'PAY_5'] = 0

fil = (df.PAY_6 == -2) | (df.PAY_6 == -1) | (df.PAY_6 == 0)

df.loc[fil, 'PAY_6'] = 0
fil = (df.EDUCATION == 5) | (df.EDUCATION == 6) | (df.EDUCATION == 0)

df.loc[fil, 'EDUCATION'] = 4

df.EDUCATION.value_counts()
# Bill Statement description description

df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].describe()
#Previous Payment Description description description

df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].describe()
print("Default probability in October:",df.def_pay.sum() / len(df.def_pay))
dataop=df[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]

dataop1=pd.concat([dataop['PAY_1'],dataop['PAY_2'],dataop['PAY_3'],dataop['PAY_4'],dataop['PAY_5'],dataop['PAY_6']])

dataop2=dataop1.value_counts().to_frame()

dataop2.columns=['AMOUNT']

dataop2
sns.set(style="white",font="Arial",font_scale=4.8)

plt.rcParams['figure.figsize'] = [48, 30]

g=sns.barplot(x=dataop2.index,y=dataop2['AMOUNT'],data=dataop2

,color="#E21932")  

sns.despine(left=True,bottom=False)

g.set( ylabel=" ",xlabel=" ")

plt.yticks([])

g.set_xticklabels(['Paid on time','M1','M2','M3','M4','M5','M6','M7','M8'])

#plt.savefig('.png', bbox_inches = 'tight',dpi=300)

plt.title('Average Delinquency term in 6 months  ',size = 30)

plt.show()
dfdp=df[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]

data={'Month':['4','5','6','7','8','9'],

      'Default rate':[((30000-dfdp['PAY_6'].value_counts()[0])/300),

((30000-dfdp['PAY_5'].value_counts()[0])/300),((30000-dfdp['PAY_4'].value_counts()[0])/300),

                      ((30000-dfdp['PAY_3'].value_counts()[0])/300),

                      ((30000-dfdp['PAY_2'].value_counts()[0])/300),

                      ((30000-dfdp['PAY_1'].value_counts()[0])/300)]

      }

dfdpm=pd.DataFrame(data)

dfdpm
dfdpm.loc[6]={'Month':'10','Default rate':22.12}

#dfdpm['Month'].apply(str)
plt.rcParams['figure.figsize'] = [48, 20]

sns.set(style="whitegrid",font_scale=1)

g1 = sns.catplot(x="Month",y='Default rate',kind="bar",aspect=1.2,data=dfdpm,order=['4','5','6','7','8','9','10'],color="#E21932")

g1.set( ylabel="",xlabel="")

sns.despine(left=True)

plt.title('Default rate (%)',size = 20)

plt.show()
datadpr=df

overdue_balance_4=datadpr[datadpr['PAY_6']>0]['BILL_AMT6'].sum()-datadpr[datadpr['PAY_6']>0]['PAY_AMT5'].sum()

loan_balance_4=datadpr['BILL_AMT6'].sum()-datadpr['PAY_AMT5'].sum()



overdue_balance_5=datadpr[datadpr['PAY_5']>0]['BILL_AMT5'].sum()-datadpr[datadpr['PAY_5']>0]['PAY_AMT4'].sum()

loan_balance_5=datadpr['BILL_AMT5'].sum()-datadpr['PAY_AMT4'].sum()



overdue_balance_6=datadpr[datadpr['PAY_4']>0]['BILL_AMT4'].sum()-datadpr[datadpr['PAY_4']>0]['PAY_AMT3'].sum()

loan_balance_6=datadpr['BILL_AMT4'].sum()-datadpr['PAY_AMT3'].sum()



overdue_balance_7=datadpr[datadpr['PAY_3']>0]['BILL_AMT3'].sum()-datadpr[datadpr['PAY_3']>0]['PAY_AMT2'].sum()

loan_balance_7=datadpr['BILL_AMT3'].sum()-datadpr['PAY_AMT2'].sum()



overdue_balance_8=datadpr[datadpr['PAY_2']>0]['BILL_AMT2'].sum()-datadpr[datadpr['PAY_2']>0]['PAY_AMT1'].sum()

loan_balance_8=datadpr['BILL_AMT2'].sum()-datadpr['PAY_AMT1'].sum()

                                                                                             

data={'overdue_balance':[overdue_balance_4,overdue_balance_5,overdue_balance_6,overdue_balance_7,overdue_balance_8],

      'loan_balance':[loan_balance_4,loan_balance_5,loan_balance_6,loan_balance_7,loan_balance_8],

      'Month':['4','5','6','7','8']

}

dataoverr=pd.DataFrame(data)                                                                                            

dataoverr['Overdue rate']=dataoverr['overdue_balance']/dataoverr['loan_balance']                                                                                       

dataoverr                                                                                           
df.LIMIT_BAL.describe()
df['LIMIT_BAL'].describe()
plt.rcParams['figure.figsize'] = 27.1,20

sns.set(style="white",font="Arial",font_scale=3.3)

g = sns.distplot(df.LIMIT_BAL,color="#E21932" ,bins=25,

kde_kws={"color": "#E21932", "lw": 3, },

hist_kws={ "linewidth": 3,"alpha": .8}) 

sns.despine(bottom=False,right=True,top=True,left=True)

g.set_yticklabels([])

g.set( ylabel="",xlabel="")

plt.title("Distribution of Credit line",size=30)

plt.show()
plt.rcParams['figure.figsize'] = 47.1,35.27

datacl=df[(df.EDUCATION<4)&(df.MARRIAGE!=3)]

datacl.loc[datacl.EDUCATION==1,'EDUCATION']='Graduate school'

datacl.loc[datacl.EDUCATION==2,'EDUCATION']='University'

datacl.loc[datacl.EDUCATION==3,'EDUCATION']='High school'

datacl.loc[datacl.SEX==1,'SEX']='Male'

datacl.loc[datacl.SEX==2,'SEX']='Female'

datacl.loc[datacl.MARRIAGE==1,'MARRIAGE']='Married'

datacl.loc[datacl.MARRIAGE==2,'MARRIAGE']='Single'

sns.set(style="whitegrid",font="Arial",font_scale=4)

g = sns.catplot(x="EDUCATION",y="LIMIT_BAL",hue="SEX",row="MARRIAGE",data=datacl, kind="box", height = 10 ,

aspect=3,palette = sns.color_palette(["#E21932","white"]))

g.set( ylabel="",xlabel="")

plt.show()
dataclr=df

dataclr.loc[((dataclr['LIMIT_BAL'] > 0) & (dataclr['LIMIT_BAL'] <= 50000)) , 'limit'] = '[1,5]'

dataclr.loc[((dataclr['LIMIT_BAL'] > 50000) & (dataclr['LIMIT_BAL'] <= 100000)) , 'limit'] = '(5,10]'

dataclr.loc[((dataclr['LIMIT_BAL'] > 100000) & (dataclr['LIMIT_BAL'] <= 150000)) , 'limit'] = '(10,15]'

dataclr.loc[((dataclr['LIMIT_BAL'] > 150000) & (dataclr['LIMIT_BAL'] <= 200000)) , 'limit'] = '(15,20]'

dataclr.loc[((dataclr['LIMIT_BAL'] > 200000) & (dataclr['LIMIT_BAL'] <= 300000)) , 'limit'] = '(20,30]'

dataclr.loc[((dataclr['LIMIT_BAL'] > 300000) & (dataclr['LIMIT_BAL'] <= 400000)) , 'limit'] = '(30,40]'

dataclr.loc[((dataclr['LIMIT_BAL'] > 400000) & (dataclr['LIMIT_BAL'] <= 500000)) , 'limit'] = '(40,50]'

dataclr.loc[((dataclr['LIMIT_BAL'] > 500000) & (dataclr['LIMIT_BAL'] <= 600000)) , 'limit'] = '(50,60]'

dataclr.loc[((dataclr['LIMIT_BAL'] > 600000) & (dataclr['LIMIT_BAL'] <= 800000)) , 'limit'] = '(60,80]'

dataclr.loc[((dataclr['LIMIT_BAL'] > 800000) & (dataclr['LIMIT_BAL'] <= 1000000)) , 'limit'] = '(80,100]'

data_clr=dataclr.groupby(['limit'])[['def_pay']].sum()

data_clr['derate']=dataclr.groupby(['limit'])[['def_pay']].sum()/dataclr.groupby(['limit'])[['def_pay']].count()

data_clr.reindex(['[1,5]','(5,10]','(10,15]','(15,20]', '(20,30]', '(30,40]', '(40,50]',  '(50,60]',

       '(60,80]', '(80,100]'])
sns.set(style="whitegrid",font="Arial",font_scale=3)

plt.rcParams['figure.figsize'] = [48, 15]

g=sns.barplot(x=data_clr.index,y=data_clr['derate']*100,data=data_clr

,color="#E21932",order=['[1,5]','(5,10]','(10,15]','(15,20]', '(20,30]', '(30,40]', '(40,50]',  '(50,60]',

       '(60,80]', '(80,100]']) 

sns.despine(left=True)

g.set( ylabel="",xlabel="")

plt.title('Default rate in each credit line',size = 30)

plt.show()
print("Default probability:",df.def_pay.sum() / len(df.def_pay))
dfre=dataclr 

dfrea=dfre[dfre['BILL_AMT6']!=0]

dfre1=dfrea.groupby(['limit'])['BILL_AMT6'].count()

dfre1=dfre1.to_frame()

    

dfre1.columns=(['count1'])

    

dfre1['rate1']=dfre1['count1']/25980

dfre1['on1']=dfre1.index

dfreb=dfre[dfre['BILL_AMT2']!=0]

dfre2=dfreb.groupby(['limit'])['BILL_AMT2'].count()

dfre2=dfre2.to_frame()

dfre2.columns=(['count2'])

dfre2['rate2']=dfre2['count2']/27494

dfre2['on1']=dfre2.index

dfre2

dfre3=pd.merge(dfre1,dfre2,on='on1')

dfre3=dfre3.reindex([9,5,0,1,2,3,4,6,7,8])
sns.set(style="whitegrid",font="Arial",font_scale=6)

sns.pointplot('on1','rate1',data=dfre3,color="black",label='May',marker='.',markersize=25,linewidth=5)

sns.pointplot('on1','rate2',data=dfre3,color="#C93245",label='Sep',marker='.',markersize=25,linewidth=5)

#plt.plot(dfre3.on1,dfre3.rate2,color='blue',label='August')

#,color='#E21932'

plt.title('')

plt.xlabel('')

#plt.xticklabels=(['[1,5]','(5,10]','(10,15]','(15,20]', '(20,30]', '(30,40]', '(40,50]',  '(50,60]', '(60,80]', '(80,100]' ])

plt.ylabel('')

sns.despine(left=True)

plt.legend()



plt.show()

#dfre3[['rate1','rate2']].plot()
dfre=dataclr 

dfrea=dfre[dfre['BILL_AMT6']!=0]

dfre1=dfrea.groupby(['EDUCATION'])['BILL_AMT6'].count()

dfre1=dfre1.to_frame()

    

dfre1.columns=(['count1'])

    

dfre1['rate1']=dfre1['count1']/25980

dfre1['on1']=dfre1.index

dfreb=dfre[dfre['BILL_AMT2']!=0]

dfre2=dfreb.groupby(['EDUCATION'])['BILL_AMT2'].count()

dfre2=dfre2.to_frame()

dfre2.columns=(['count2'])

dfre2['rate2']=dfre2['count2']/27494

dfre2['on1']=dfre2.index

dfre2

dfre4=pd.merge(dfre1,dfre2,on='on1')

dfre4
sns.set(style="whitegrid",font="Arial",font_scale=6)

sns.lineplot('on1','rate1',data=dfre4,color="black",label='May',marker='.',markers=True,markersize=45,linewidth=5)

sns.lineplot('on1','rate2',data=dfre4,color="#C93245",label='Sep', marker='.',markers=True,markersize=45,linewidth=5)

plt.title('')

plt.xlabel('')

plt.xticks(np.arange(1,5),['Master','University','High school','Others'])

plt.ylabel('')

sns.despine(left=True)

plt.legend()

plt.show()
dfre=dataclr

dfre['AGEBIN'] = pd.cut(dfre['AGE'], bins = np.linspace(20, 80, num = 13))

dfrea=dfre[dfre['BILL_AMT6']!=0]

dfre1=dfrea.groupby(['AGEBIN'])['BILL_AMT6'].count()

dfre1=dfre1.to_frame()   

dfre1.columns=(['count1'])

dfre1['rate1']=dfre1['count1']/25980

dfre1['on2']=dfre1.index

dfreb=dfre[dfre['BILL_AMT2']!=0]

dfre2=dfreb.groupby(['AGEBIN'])['BILL_AMT2'].count()

dfre2=dfre2.to_frame()

dfre2.columns=(['count2'])

dfre2['rate2']=dfre2['count2']/27494

dfre2['on2']=dfre2.index

dfre2

dfre6=pd.merge(dfre1,dfre2,on='on2')
sns.set(style="whitegrid",font="Arial",font_scale=3.5)

sns.pointplot('on2','rate1',data=dfre6,color="black",label='April',marker='.',markersize=45,linewidth=5)

sns.pointplot('on2','rate2',data=dfre6,color="#C93245",label='August',marker='.',markersize=45,linewidth=5)

plt.title('')

plt.xlabel('')

plt.ylabel('')

sns.despine(left=True)

plt.legend()

plt.show()
dfre=dataclr

dfre=dfre[dfre['BILL_AMT6']!=0]

dfre1=dfre.groupby(['limit'])['BILL_AMT6'].count()

dfre.groupby(['limit'])['BILL_AMT6'].count().sum()

dfre1=dfre1.to_frame()

dfre1.columns=(['count1'])

dfre1['rate']=dfre1['count1']/25980

dfre1

dfre=dfre[dfre['BILL_AMT2']!=0]

dfre1=dfre.groupby(['limit'])['BILL_AMT2'].count()

dfre.groupby(['limit'])['BILL_AMT2'].count().sum()

dfre1=dfre1.to_frame()

dfre1.columns=(['count1'])

dfre1['rate']=dfre1['count1']/27494

dfre1
df['AGE'].describe()
plt.figure(figsize = (20, 10))

sns.set(style="white",font_scale=2.4)

sns.kdeplot(df.loc[df['def_pay'] == 1, 'AGE'] , label = 'Default',color="#E21932",linewidth = 3.5)

sns.despine(left=True,bottom=False,right=True,top=True)

sns.kdeplot(df.loc[df['def_pay'] == 0, 'AGE'] , label = 'Paid on time',color="black",linewidth = 3.5)

plt.title("Dendity of Age")

plt.show()
df['AgeBin'] = 0 

df.loc[((df['AGE'] > 20) & (df['AGE'] < 30)) , 'AgeBin'] = '(20,30)'

df.loc[((df['AGE'] >= 30) & (df['AGE'] < 40)) , 'AgeBin'] = '[30,40)'

df.loc[((df['AGE'] >= 40) & (df['AGE'] < 50)) , 'AgeBin'] = '[40,50)'

df.loc[((df['AGE'] >= 50) & (df['AGE'] < 60)) , 'AgeBin'] = '[50,60)'

df.loc[((df['AGE'] >= 60) & (df['AGE'] < 70)) , 'AgeBin'] = '[60,70)'

df.loc[((df['AGE'] >= 70) & (df['AGE'] < 81)) , 'AgeBin'] = '[70,81)'
dataage=df[['AGE','def_pay']]

dataage['AGEBIN'] = pd.cut(df['AGE'], bins = np.linspace(20, 80, num = 13))

dataage1=dataage.groupby('AGEBIN').mean()

dataage1.head()
sns.set(style="whitegrid",font="Arial",font_scale=3)

plt.rcParams['figure.figsize'] = [48, 15]

g=sns.barplot(x=dataage1.index,y=dataage1['def_pay']*100,data=dataage1

,color="#E21932") 

sns.despine(left=True)

g.set( ylabel="",xlabel="")

plt.title('Default rate in each agebox',size = 40)

plt.show()
dataclient=df

grouped=dataclient.groupby(['EDUCATION'])[['def_pay']]

dacli=grouped.sum()

dacli['count']=grouped.count()

dacli['derate']=dacli['def_pay']/dacli['count']

dacli['allrate']=dacli['count']/30000

dacli['EDUCATION_']=dacli.index

dacli.reset_index()

dacli.sort_values('count',ascending=False)
sns.set(style="white",font="Calibri",font_scale=4)

plt.rcParams['figure.figsize'] = [20, 8]

g = sns.barplot(x="derate",y="EDUCATION_",color='#E21932'

,orient="h",data=dacli)

sns.despine(left=True)

g.set_xticklabels(['0.00','0.05','0.10','0.15','0.20','0.25'],ha='left')

g.set_yticklabels(['Graduate school','University','High school','Others'])

g.set( ylabel="",xlabel="")

plt.title("Default rate of clients")

plt.show()
sns.set(style="white",font="Calibri",font_scale=4)

plt.rcParams['figure.figsize'] = [20, 8]

g = sns.barplot(x="allrate",y="EDUCATION_",color='#E21932',

orient="h",data=dacli)

sns.despine(left=True)

g.set_yticklabels(['Graduate school','University','High school','Others'])

g.set_xticklabels(['0.0','0.1','0.2','0.3','0.4'],ha='left')

g.set( ylabel="",xlabel="")

plt.title("Percentage of clients")

plt.show()
dfh=dataclr

dfh=dfh[dfh['EDUCATION']!=4]

dfh['AGEBIN'] = pd.cut(dfh['AGE'], bins = np.linspace(20, 80, num = 13))

grouped=dfh.groupby(['EDUCATION','SEX'

            ,'MARRIAGE','AGEBIN'])[['def_pay']]

dfall=grouped.sum()

dfall['count']=grouped.count()

dfall['derate']=dfall['def_pay']/dfall['count']

dfall['rate']=dfall['count']/30000

dfall=dfall[dfall['count']>300]

dfall.sort_values(by='count',ascending=False).head()
dataac=df

dataac['AGEBIN'] = pd.cut(dataac['AGE'], bins = np.linspace(20, 80, num = 13))

dataac.loc[dataac['PAY_AMT6']>0,'PAY_AMT6']=1

dataac.loc[dataac['PAY_AMT5']>0,'PAY_AMT5']=1

dataac.loc[dataac['PAY_AMT4']>0,'PAY_AMT4']=1

dataac.loc[dataac['PAY_AMT3']>0,'PAY_AMT3']=1

dataac.loc[dataac['PAY_AMT2']>0,'PAY_AMT2']=1

dataac.loc[dataac['PAY_AMT1']>0,'PAY_AMT1']=1

data_ac=dataac.groupby(['EDUCATION','SEX','MARRIAGE','AGEBIN'])['PAY_AMT6','PAY_AMT5','PAY_AMT4','PAY_AMT3','PAY_AMT2','PAY_AMT1'].sum()

data_ac['Col_sum'] = data_ac.apply(lambda x: x.sum(), axis=1)

data_ac['rate']=data_ac['Col_sum']/data_ac['Col_sum'].sum()

data_ac.sort_values('Col_sum',ascending=False).head()
dataad=dataclr

dataad=dataad[(dataad['BILL_AMT1']!=0)&(dataad['BILL_AMT2']!=0)&(dataad['BILL_AMT3']!=0)

              &(dataad['BILL_AMT4']!=0)&(dataad['BILL_AMT5']!=0)&(dataad['BILL_AMT6']!=0)]

dataad=dataad[['EDUCATION','SEX','MARRIAGE','AGEBIN','LIMIT_BAL','limit','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']]

dataad['Col_sum'] = dataad[['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']].apply(lambda x: x.sum(), axis=1)

dataad['rate']=dataad['Col_sum'] /(6*dataad['LIMIT_BAL'])



data_ad=dataad.groupby(['EDUCATION','SEX','MARRIAGE','AGEBIN'])[['rate']].mean()

data_ad['count']=dataad.groupby(['EDUCATION','SEX','MARRIAGE','AGEBIN'])[['rate']].count()

data_ad=data_ad[data_ad['count']>200]

data_ad.sort_values('rate',ascending=False)
def corr_2_cols(Col1, Col2):

    res = df.groupby([Col1, Col2]).size().unstack()

    res['overdue rate'] = (res[res.columns[1]]/(res[res.columns[0]] + res[res.columns[1]]))

    return res
corr_2_cols('MARRIAGE', 'def_pay')
df.loc[df.MARRIAGE == 0, 'MARRIAGE'] = 3

df.MARRIAGE.value_counts()
marry = df.groupby(['MARRIAGE', 'def_pay']).size().unstack()

marry.columns=['Paid on time','Default']

marry
datatest=df

datatest.loc[datatest.PAY_1<4,'PAY_1']=0

datatest.loc[datatest.PAY_2<4,'PAY_2']=0

datatest.loc[datatest.PAY_3<4,'PAY_3']=0

datatest.loc[datatest.PAY_4<4,'PAY_4']=0

datatest.loc[datatest.PAY_5<4,'PAY_5']=0

datatest.loc[datatest.PAY_6<4,'PAY_6']=0

datatest.loc[datatest.PAY_1>3,'PAY_1']=1

datatest.loc[datatest.PAY_2>3,'PAY_2']=1

datatest.loc[datatest.PAY_3>3,'PAY_3']=1

datatest.loc[datatest.PAY_4>3,'PAY_4']=1

datatest.loc[datatest.PAY_5>3,'PAY_5']=1

datatest.loc[datatest.PAY_6>3,'PAY_6']=1

dataset=datatest.groupby(['MARRIAGE'])[['PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']].sum()

dataset.T.apply(sum)
df = pd.read_csv('../input/UCI_Credit_Card.csv')
df = df.rename(columns={'default.payment.next.month': 'def_pay', 

                        'PAY_0': 'PAY_1'})
bins = [20, 29, 39, 49, 59, 69, 81]

bins_names = [1, 2, 3, 4, 5, 6]

df['AgeBin'] = pd.cut(df['AGE'], bins, labels=bins_names)
df['AgeBin'] = pd.cut(df['AGE'], 6, labels = [1,2,3,4,5,6])

#because 1 2 3 ecc are "categories" so far and we need numbers

df['AgeBin'] = pd.to_numeric(df['AgeBin'])

df.loc[(df['AgeBin'] == 6) , 'AgeBin'] = 5
df['SE_MA'] = 0

df.loc[((df.SEX == 1) & (df.MARRIAGE == 1)) , 'SE_MA'] = 1 #married man

df.loc[((df.SEX == 1) & (df.MARRIAGE == 2)) , 'SE_MA'] = 2 #single man

df.loc[((df.SEX == 1) & (df.MARRIAGE == 3)) , 'SE_MA'] = 3 #divorced man

df.loc[((df.SEX == 2) & (df.MARRIAGE == 1)) , 'SE_MA'] = 4 #married woman

df.loc[((df.SEX == 2) & (df.MARRIAGE == 2)) , 'SE_MA'] = 5 #single woman

df.loc[((df.SEX == 2) & (df.MARRIAGE == 3)) , 'SE_MA'] = 6 #divorced woman

corr_2_cols('SE_MA', 'def_pay')
df['SE_AG'] = 0

df.loc[((df.SEX == 1) & (df.AgeBin == 1)) , 'SE_AG'] = 1 #man in 20's

df.loc[((df.SEX == 1) & (df.AgeBin == 2)) , 'SE_AG'] = 2 #man in 30's

df.loc[((df.SEX == 1) & (df.AgeBin == 3)) , 'SE_AG'] = 3 #man in 40's

df.loc[((df.SEX == 1) & (df.AgeBin == 4)) , 'SE_AG'] = 4 #man in 50's

df.loc[((df.SEX == 1) & (df.AgeBin == 5)) , 'SE_AG'] = 5 #man in 60's and above

df.loc[((df.SEX == 2) & (df.AgeBin == 1)) , 'SE_AG'] = 6 #woman in 20's

df.loc[((df.SEX == 2) & (df.AgeBin == 2)) , 'SE_AG'] = 7 #woman in 30's

df.loc[((df.SEX == 2) & (df.AgeBin == 3)) , 'SE_AG'] = 8 #woman in 40's

df.loc[((df.SEX == 2) & (df.AgeBin == 4)) , 'SE_AG'] = 9 #woman in 50's

df.loc[((df.SEX == 2) & (df.AgeBin == 5)) , 'SE_AG'] = 10 #woman in 60's and above

corr_2_cols('SE_AG', 'def_pay')
df['Client_6'] = 1

df['Client_5'] = 1

df['Client_4'] = 1

df['Client_3'] = 1

df['Client_2'] = 1

df['Client_1'] = 1

df.loc[((df.PAY_6 == 0) & (df.BILL_AMT6 == 0) & (df.PAY_AMT6 == 0)) , 'Client_6'] = 0

df.loc[((df.PAY_5 == 0) & (df.BILL_AMT5 == 0) & (df.PAY_AMT5 == 0)) , 'Client_5'] = 0

df.loc[((df.PAY_4 == 0) & (df.BILL_AMT4 == 0) & (df.PAY_AMT4 == 0)) , 'Client_4'] = 0

df.loc[((df.PAY_3 == 0) & (df.BILL_AMT3 == 0) & (df.PAY_AMT3 == 0)) , 'Client_3'] = 0

df.loc[((df.PAY_2 == 0) & (df.BILL_AMT2 == 0) & (df.PAY_AMT2 == 0)) , 'Client_2'] = 0

df.loc[((df.PAY_1 == 0) & (df.BILL_AMT1 == 0) & (df.PAY_AMT1 == 0)) , 'Client_1'] = 0
df['Avg_exp_5'] = ((df['BILL_AMT5'] - (df['BILL_AMT6'] - df['PAY_AMT5']))) / df['LIMIT_BAL']

df['Avg_exp_4'] = (((df['BILL_AMT5'] - (df['BILL_AMT6'] - df['PAY_AMT5'])) +

                 (df['BILL_AMT4'] - (df['BILL_AMT5'] - df['PAY_AMT4']))) / 2) / df['LIMIT_BAL']

df['Avg_exp_3'] = (((df['BILL_AMT5'] - (df['BILL_AMT6'] - df['PAY_AMT5'])) +

                 (df['BILL_AMT4'] - (df['BILL_AMT5'] - df['PAY_AMT4'])) +

                 (df['BILL_AMT3'] - (df['BILL_AMT4'] - df['PAY_AMT3']))) / 3) / df['LIMIT_BAL']

df['Avg_exp_2'] = (((df['BILL_AMT5'] - (df['BILL_AMT6'] - df['PAY_AMT5'])) +

                 (df['BILL_AMT4'] - (df['BILL_AMT5'] - df['PAY_AMT4'])) +

                 (df['BILL_AMT3'] - (df['BILL_AMT4'] - df['PAY_AMT3'])) +

                 (df['BILL_AMT2'] - (df['BILL_AMT3'] - df['PAY_AMT2']))) / 4) / df['LIMIT_BAL']

df['Avg_exp_1'] = (((df['BILL_AMT5'] - (df['BILL_AMT6'] - df['PAY_AMT5'])) +

                 (df['BILL_AMT4'] - (df['BILL_AMT5'] - df['PAY_AMT4'])) +

                 (df['BILL_AMT3'] - (df['BILL_AMT4'] - df['PAY_AMT3'])) +

                 (df['BILL_AMT2'] - (df['BILL_AMT3'] - df['PAY_AMT2'])) +

                 (df['BILL_AMT1'] - (df['BILL_AMT2'] - df['PAY_AMT1']))) / 5) / df['LIMIT_BAL']
df['Avg_5'] = ((df['BILL_AMT5'] - (df['BILL_AMT6'] - df['PAY_AMT5']))) / df['LIMIT_BAL']

df['Avg_4'] = ((df['BILL_AMT4'] - (df['BILL_AMT5'] - df['PAY_AMT4']))) / df['LIMIT_BAL']

df['Avg_3'] = ((df['BILL_AMT3'] - (df['BILL_AMT4'] - df['PAY_AMT3']))) / df['LIMIT_BAL']

df['Avg_2'] = ((df['BILL_AMT2'] - (df['BILL_AMT3'] - df['PAY_AMT2']))) / df['LIMIT_BAL']

df['Avg_1'] = ((df['BILL_AMT1'] - (df['BILL_AMT2'] - df['PAY_AMT1']))) / df['LIMIT_BAL']
df['Closeness_6'] = (df.LIMIT_BAL - df.BILL_AMT6) / df.LIMIT_BAL

df['Closeness_5'] = (df.LIMIT_BAL - df.BILL_AMT5) / df.LIMIT_BAL

df['Closeness_4'] = (df.LIMIT_BAL - df.BILL_AMT4) / df.LIMIT_BAL

df['Closeness_3'] = (df.LIMIT_BAL - df.BILL_AMT3) / df.LIMIT_BAL

df['Closeness_2'] = (df.LIMIT_BAL - df.BILL_AMT2) / df.LIMIT_BAL

df['Closeness_1'] = (df.LIMIT_BAL - df.BILL_AMT1) / df.LIMIT_BAL
df.loc[((df['LIMIT_BAL'] > 0) & (df['LIMIT_BAL'] <= 50000)) , 'limit'] = 10

df.loc[((df['LIMIT_BAL'] > 50000) & (df['LIMIT_BAL'] <= 100000)) , 'limit'] = 9

df.loc[((df['LIMIT_BAL'] > 100000) & (df['LIMIT_BAL'] <= 150000)) , 'limit'] =8

df.loc[((df['LIMIT_BAL'] > 150000) & (df['LIMIT_BAL'] <= 200000)) , 'limit'] = 7

df.loc[((df['LIMIT_BAL'] > 200000) & (df['LIMIT_BAL'] <= 300000)) , 'limit'] = 6

df.loc[((df['LIMIT_BAL'] > 300000) & (df['LIMIT_BAL'] <= 400000)) , 'limit'] = 5

df.loc[((df['LIMIT_BAL'] > 400000) & (df['LIMIT_BAL'] <= 500000)) , 'limit'] = 4

df.loc[((df['LIMIT_BAL'] > 500000) & (df['LIMIT_BAL'] <= 600000)) , 'limit'] = 3

df.loc[((df['LIMIT_BAL'] > 600000) & (df['LIMIT_BAL'] <= 800000)) , 'limit'] = 2

df.loc[((df['LIMIT_BAL'] > 800000) & (df['LIMIT_BAL'] <= 1000000)) , 'limit'] =1
df['age_lim']=df['AgeBin']*df['limit']
df['all1'].describe()
df['all3']=(df['PAY_1']*df['PAY_1']*df['PAY_1'])+(df['PAY_2']*df['PAY_2']*df['PAY_2'])+(df['PAY_3']*df['PAY_3']*df['PAY_3'])+(df['PAY_4']*df['PAY_4']*df['PAY_4'])+(df['PAY_5']*df['PAY_5']*df['PAY_5'])+(df['PAY_6']*df['PAY_6']*df['PAY_6'])
df['all1']=df['PAY_1']+df['PAY_2']+df['PAY_3']+df['PAY_4']+df['PAY_5']+df['PAY_6']
df['all2']=(df['PAY_1']*df['PAY_1'])+(df['PAY_2']*df['PAY_2'])+(df['PAY_3']*df['PAY_3'])+(df['PAY_4']*df['PAY_4'])+(df['PAY_5']*df['PAY_5'])+(df['PAY_6']*df['PAY_6'])
y = df['def_pay'].copy()
df['AGE'].describe()
features =  ['LIMIT_BAL', 'EDUCATION', 'MARRIAGE', 'PAY_1','PAY_2', 'PAY_3', 

            'PAY_4', 'PAY_5', 'PAY_6','BILL_AMT1', 'BILL_AMT2',

            'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',

            'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'AGE',

            'SE_MA', 'AgeBin', 'SE_AG', 'Avg_exp_5', 'Avg_exp_4','all1','Avg_5','Avg_4','Avg_3','Avg_2','Avg_1',

            'Avg_exp_3', 'Avg_exp_2', 'Avg_exp_1', 'Closeness_5','all3','all2',

            'Closeness_4', 'Closeness_3', 'Closeness_2','Closeness_1']

X = df[features].copy()

X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
def get_feature_importance(clsf, ftrs):

    imp = clsf.feature_importances_.tolist()

    feat = ftrs

    result = pd.DataFrame({'feat':feat,'score':imp})

    result = result.sort_values(by=['score'],ascending=False)

    plt.figure(figsize=(10,9))

    sns.set(style="whitegrid",font="Arial",font_scale=1.2)

    data1=result.iloc[:10,:]

    g=sns.catplot(x='score',y='feat',data=data1,kind="bar",color="#E21932",aspect=2.5)

    plt.show()

    return result

param_grid = {'max_depth': np.arange(5, 10),

             'criterion' : ['gini','entropy'],

             'max_leaf_nodes': [10,20,50,100],

             'min_samples_split': [2,5,10,20,50],

             'class_weight' : ['balanced']}



# create the grid

grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 5, scoring= 'roc_auc')

# the cv option will be clear in a few cells



#training

grid_tree.fit(X_train, y_train)

#let's see the best estimator

print(grid_tree.best_estimator_)

#with its score

print(grid_tree.best_score_)
tree2=DecisionTreeClassifier(class_weight='balanced', criterion='entropy',

                       max_depth=9, max_features=None, max_leaf_nodes=50,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=50,

                       min_weight_fraction_leaf=0.0, presort=False,

                       random_state=None, splitter='best')

tree2.fit(X_train, y_train)

predictions = tree2.predict_proba(X_test)[:,1]

auc4=roc_auc_score(y_test, predictions)

print(auc4)
predictions = tree2.predict(X_test)

f1_score(y_true = y_test, y_pred = predictions)

print("-------------")

print("f1 score: {}".format(round(f1_score(y_true = y_test, y_pred = predictions),3)))

print("Accuracy: {}".format(round(accuracy_score(y_true = y_test, y_pred = predictions),3)))

print("-------------")

print(get_feature_importance(tree2, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

pred = len(predictions)



print('True Positives: {}'.format(TP))

print('False Positive: {}'.format(FP))

print('True Negative: {}'.format(TN))

print('False Negative: {}'.format(FN))

print('Precision: {}'.format(round(TP/(TP+FP),2)))

print('Recall: {}'.format(round(TP/(TP+FN),2)))

print('Problematic ratio: {}'.format(round(FN/(FN+TP),2)))
from sklearn import tree

import graphviz

dot_data = tree.export_graphviz(tree2, out_file=None)  

graph = graphviz.Source(dot_data)  

graph
def get_feature_importance(clsf, ftrs):

    imp = clsf.feature_importances_.tolist()

    feat = ftrs

    result = pd.DataFrame({'feat':feat,'score':imp})

    result = result.sort_values(by=['score'],ascending=False)

    sns.set(style="whitegrid",font="Arial",font_scale=1.2)

    data1=result.iloc[:10,:]

    g=sns.catplot(x='score',y='feat',data=data1,kind="bar",color="#E21932",aspect=2)

    plt.show()

    return result



param_grid = {'n_estimators': [100,200,300,400],

              'criterion': ['entropy', 'gini'], 'n_jobs' : [-1]}

grid_forest = GridSearchCV(RandomForestClassifier(), param_grid, scoring = 'roc_auc', cv=5)

%time grid_forest = grid_forest.fit(X_train, y_train)

print(grid_forest.best_estimator_)

print(grid_forest.best_score_)

forest_downsampled = grid_forest.best_estimator_
RF2 =RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',

                       max_depth=None, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=300,

                       n_jobs=-1, oob_score=False, random_state=None, verbose=0,

                       warm_start=False)

RF2.fit(X_train, y_train)

predictions = RF2.predict_proba(X_test)[:,1]

auc3=roc_auc_score(y_test, predictions)
print(auc3)
RF2.fit(X_train, y_train)

predictions = RF2.predict(X_test)

print("-------------")

print("f1 score: {}".format(round(f1_score(y_true = y_test, y_pred = predictions),3)))

print("Accuracy: {}".format(round(accuracy_score(y_true = y_test, y_pred = predictions),3)))

print("-------------")

print(get_feature_importance(RF2, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

pred = len(predictions)



print('True Positives: {}'.format(TP))

print('False Positive: {}'.format(FP))

print('True Negative: {}'.format(TN))

print('False Negative: {}'.format(FN))

print('Precision: {}'.format(round(TP/(TP+FP),2)))

print('Recall: {}'.format(round(TP/(TP+FN),2)))

print('Problematic ratio: {}'.format(round(FN/(FN+TP),2)))
predictions_tree = tree2.predict_proba(X_test)[:,1]

fpr1, tpr1, _ = metrics.roc_curve(y_test,  predictions_tree)



predictions_rf = RF2.predict_proba(X_test)[:,1]

fpr2, tpr2, _ = metrics.roc_curve(y_test,  predictions_rf)





sns.set(style="white",font="Arial",font_scale=1.5)





plt.figure(figsize=(10,10))

plt.plot([0, 1], [0, 1], 'k--')



plt.plot(fpr1,tpr1,label="Decision Tree, auc="+str(round(auc4,2)),color="#E21932")

plt.plot(fpr2,tpr2,label="Random forest, auc="+str(round(auc3,2)),color="black")

plt.legend(loc=4, facecolor='white',fontsize=16)

sns.despine(left=True)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title("ROC")