import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns





import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots

data = pd.read_csv('../input/memory-test-on-drugged-islanders-data/Islander_data.csv')
data.head()
data.info()
data.describe()
cleaned_data = data.copy()
plt.figure(figsize=(16,6))

sns.barplot(x='Drug',y='Mem_Score_Before',data=cleaned_data, order=cleaned_data.Drug.unique().tolist())

plt.title('Distribution of Drugs')
fig = px.bar(cleaned_data, x="age", y="Mem_Score_Before", title="Mem_Score_Before over Age",

             color_discrete_sequence=['#F42272'])

fig.show()



fig = px.bar(cleaned_data, x="age", y="Mem_Score_After", title="Mem_Score_After over Age", 

              log_y=True, color_discrete_sequence=['#F42272'])

fig.show()
fig = px.sunburst(cleaned_data.sort_values(by='age', ascending=False).reset_index(drop=True), 

                 path=["first_name"], values="Mem_Score_Before", height=700,

                 title='Sunburst for Mem_Score_Before ',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value'

fig.show()
fig = px.bar(cleaned_data.sort_values('age', ascending=False)[:10][::-1], 

             x='Mem_Score_Before', y='first_name',

             title='Patient with Higest Mem_Score_Before', text='Mem_Score_Before', orientation='h')

fig.show()
fig = px.bar(cleaned_data.sort_values('age', ascending=False)[:10][::-1], 

             x='Mem_Score_After', y='first_name',

             title='Patient with Higest Mem_Score_After', text='Mem_Score_After', orientation='h')

fig.show()
bins = [0, 2, 18, 35, 65, np.inf]

names = ['<2', '2-18', '18-35', '35-65', '65+']



cleaned_data['AgeRange'] = pd.cut(cleaned_data['age'], bins, labels=names)

labels = ['A', 'S','T']

sizes = []

sizes.append(list(cleaned_data['Drug'].value_counts())[0])

sizes.append(list(cleaned_data['Drug'].value_counts())[1])

sizes.append(list(cleaned_data['Drug'].value_counts())[2])





explode = (0, 0.1, 0)

colors = ['#ffcc99','#66b3ff','#ff9999']



plt.figure(figsize= (15,10))

plt.title('Distribution of Drug',fontsize = 20)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',shadow=True, startangle=90)

plt.axis('equal')

plt.tight_layout()
sns.pairplot(cleaned_data)
from pandas.plotting import scatter_matrix



fig, ax = plt.subplots(figsize=(12,12))

scatter_matrix(cleaned_data, alpha=1, ax=ax)
df_plot = cleaned_data[(cleaned_data['Diff']>0)]

sns.boxplot('AgeRange', 'Diff', data=df_plot)
plt.figure(figsize=(10,6))

ax = sns.boxplot(x="Drug", y="Mem_Score_After", hue="Drug",

                 data=cleaned_data, palette="Set3")
grgs = cleaned_data.groupby(["Drug","Happy_Sad_group"])[["Mem_Score_After"]].mean().reset_index()

fig = px.bar(grgs[['Drug', 'Mem_Score_After','Happy_Sad_group']].sort_values('Mem_Score_After', ascending=False), 

             y="Mem_Score_After", x="Drug", color='Happy_Sad_group', 

             log_y=True, template='plotly_dark')

fig.show()
fig = px.scatter(cleaned_data, x="Mem_Score_Before", y="Mem_Score_After", color="Drug", facet_col="Drug",

           color_continuous_scale=px.colors.sequential.Viridis, render_mode="webgl")

fig.show()
fig = px.scatter(cleaned_data, x="Mem_Score_Before", y="Mem_Score_After", color="Happy_Sad_group", facet_col="Happy_Sad_group",

           color_continuous_scale=px.colors.sequential.Viridis, render_mode="webgl")

fig.show()
df = cleaned_data

fig = px.density_contour(df, x="Mem_Score_Before", y="Mem_Score_After", color="AgeRange", marginal_x="rug", marginal_y="histogram")

fig.show()
fig = px.violin(cleaned_data, y="Diff", x="Happy_Sad_group", color="Happy_Sad_group", box=True, points="all")

fig.show()
fig = px.parallel_categories(cleaned_data, color="age", color_continuous_scale=px.colors.sequential.Inferno)

fig.show()
fig = px.scatter(cleaned_data, x="Mem_Score_Before", y="Diff", size="Mem_Score_Before", color="Drug",

           hover_name="Drug", log_x=True, size_max=60)

fig.show()
fig = px.strip(cleaned_data, x="AgeRange", y="Diff", orientation="h", color="Drug")

fig.show()
ms = cleaned_data.sort_values(by=['age'],ascending=False)

ms = ms.head(30)

fig = px.funnel(ms, x='Mem_Score_Before', y='Happy_Sad_group')

fig.show()
cleaned_data.head()
cleaned_data.info()
fig,ax=plt.subplots(figsize=(15,5))

sns.heatmap(cleaned_data.isnull(), annot=True)
fig=plt.figure(figsize=(18,18))

sns.heatmap(cleaned_data.corr(), annot= True, cmap='Blues')

preprocessed_data = cleaned_data.drop(['first_name','last_name','Happy_Sad_group'],axis=1)
cleaned_data.Happy_Sad_group = cleaned_data.Happy_Sad_group.apply(lambda x: 1 if x == 'H' else 0)

data_label=cleaned_data['Happy_Sad_group']

del data['Happy_Sad_group']

data_label=pd.DataFrame(data_label)
preprocessed_data = pd.get_dummies(preprocessed_data,columns=['Drug','AgeRange'],drop_first=True)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data_scaled=MinMaxScaler().fit_transform(preprocessed_data)

data_scaled=pd.DataFrame(data=data_scaled, columns=preprocessed_data.columns)
from sklearn.model_selection import train_test_split

Xtrain,Xtest,Ytrain,Ytest = train_test_split(data_scaled, data_label, test_size=0.10,

                                             stratify=data_label,random_state=121)
from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

#from sklearn.ensemble import StackingClassifier Need to update sklearn to use inbuilt stacking classifier

from sklearn.ensemble import VotingClassifier



from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.model_selection import cross_val_score

from sklearn.metrics import f1_score
def CrossVal(dataX,dataY,mode,cv=3):

    score=cross_val_score(mode,dataX , dataY, cv=cv, scoring='accuracy')

    return(np.mean(score))
modelk=KNeighborsClassifier(algorithm='auto',n_neighbors= 5)

score_k=CrossVal(Xtrain,Ytrain,modelk)

print("Accuracy is : ",score_k)

modelk.fit(Xtrain,Ytrain)

cr = classification_report(Ytest, modelk.predict(Xtest))

print(cr)



# confusion matrix 



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,modelk.predict(Xtest)), annot= True, cmap='Reds')

k_f1=f1_score(Ytest,modelk.predict(Xtest))

plt.title('F1 Score = {}'.format(k_f1))
modellog=LogisticRegression()

score_k=CrossVal(Xtrain,Ytrain,modellog)

print("Accuracy is : ",score_k)

modellog.fit(Xtrain,Ytrain)

cr = classification_report(Ytest, modellog.predict(Xtest))

print(cr)



# confusion matrix 



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,modellog.predict(Xtest)), annot= True, cmap='Reds')

log_f1=f1_score(Ytest,modellog.predict(Xtest))

plt.title('F1 Score = {}'.format(log_f1))
modeldt=DecisionTreeClassifier(max_depth=6)

score_k=CrossVal(Xtrain,Ytrain,modeldt)

print("Accuracy is : ",score_k)

modeldt.fit(Xtrain,Ytrain)

cr = classification_report(Ytest, modeldt.predict(Xtest))

print(cr)



# confusion matrix 



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,modeldt.predict(Xtest)), annot= True, cmap='Reds')

dt_f1=f1_score(Ytest,modeldt.predict(Xtest))

plt.title('F1 Score = {}'.format(dt_f1))
modelsvc=SVC(C=0.2,probability=True,kernel='rbf',gamma=0.1)

score_k=CrossVal(Xtrain,Ytrain,modelsvc)

print("Accuracy is : ",score_k)

modelsvc.fit(Xtrain,Ytrain)

cr = classification_report(Ytest, modelsvc.predict(Xtest))

print(cr)



# confusion matrix 



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,modelsvc.predict(Xtest)), annot= True, cmap='Reds')

svc_f1=f1_score(Ytest,modelsvc.predict(Xtest))

plt.title('F1 Score = {}'.format(svc_f1))
modelrt=RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=97)

score_k=CrossVal(Xtrain,Ytrain,modelrt)

print("Accuracy is : ",score_k)

modelrt.fit(Xtrain,Ytrain)

cr = classification_report(Ytest, modelrt.predict(Xtest))

print(cr)



# confusion matrix 



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,modelrt.predict(Xtest)), annot= True, cmap='Reds')

rt_f1=f1_score(Ytest,modelrt.predict(Xtest))

plt.title('F1 Score = {}'.format(rt_f1))
modelext=ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=2)

score_k=CrossVal(Xtrain,Ytrain,modelext)

print("Accuracy is : ",score_k)

modelext.fit(Xtrain,Ytrain)

cr = classification_report(Ytest, modelext.predict(Xtest))

print(cr)



# confusion matrix 



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,modelext.predict(Xtest)), annot= True, cmap='Reds')

ext_f1=f1_score(Ytest,modelext.predict(Xtest))

plt.title('F1 Score = {}'.format(ext_f1))
modelada=AdaBoostClassifier(modellog,n_estimators=100, random_state=343, learning_rate=0.012)



score_k=CrossVal(Xtrain,Ytrain,modelada)

print("Accuracy is : ",score_k)

modelada.fit(Xtrain,Ytrain)

cr = classification_report(Ytest, modelada.predict(Xtest))

print(cr)



# confusion matrix 



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,modelada.predict(Xtest)), annot= True, cmap='Reds')

ada_f1=f1_score(Ytest,modelada.predict(Xtest))

plt.title('F1 Score = {}'.format(ada_f1))
modelgbc=GradientBoostingClassifier(n_estimators=100, random_state=43, learning_rate = 0.01)



score_k=CrossVal(Xtrain,Ytrain,modelgbc)

print("Accuracy is : ",score_k)

modelgbc.fit(Xtrain,Ytrain)

cr = classification_report(Ytest, modelgbc.predict(Xtest))

print(cr)



# confusion matrix 



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,modelgbc.predict(Xtest)), annot= True, cmap='Reds')

gbc_f1=f1_score(Ytest,modelgbc.predict(Xtest))

plt.title('F1 Score = {}'.format(gbc_f1))
fig= plt.figure(figsize=(10,10))

important=pd.Series(modelrt.feature_importances_, index=Xtrain.columns)

sns.set_style('whitegrid')

important.sort_values().plot.bar()

plt.title('Feature Importance for Random Forest Classifier')
model_f1_score = pd.Series(data=[k_f1, log_f1, dt_f1, svc_f1, rt_f1, ext_f1, ada_f1, 

                           gbc_f1], 

                           index=['KNN','logistic Regression','decision tree', 'SVM', 'Random Forest',

                                'Extra Tree', 'Ada Boost' , 'Gradient Boost'])

fig= plt.figure(figsize=(8,8))

model_f1_score.sort_values().plot.bar()

plt.title('Model F1 Score Comparison')
modelvc=VotingClassifier(estimators=[('knn',modelk),('SGD',modelsvc),('lr',modellog)],

                    voting='soft')



score_k=CrossVal(Xtrain,Ytrain,modelada)

print("Accuracy is : ",score_k)

modelvc.fit(Xtrain,Ytrain)

cr = classification_report(Ytest, modelvc.predict(Xtest))

print(cr)



# confusion matrix 



fig=plt.figure()

sns.heatmap(confusion_matrix(Ytest,modelvc.predict(Xtest)), annot= True, cmap='Reds')

model_vc_f1=f1_score(Ytest,modelvc.predict(Xtest))

plt.title('F1 Score = {}'.format(model_vc_f1))