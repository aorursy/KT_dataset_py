import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
#original data
heart=pd.read_csv("../input/datasets-216167-477177-heartcsv/datasets_216167_477177_heart.csv")
heart.shape
heart[100:105]
heart[['age','trestbps', 'thalach', 'chol' ]].describe().T.style.set_table_styles([{'selector' : '', 
                            'props' : [('border', 
                                        '5px solid tomato')]}])
heart.info()
heart.isna().sum()
g = sns.PairGrid(heart[['age', 'trestbps', 'chol', 'cp', 'thalach']])
fig = plt.gcf()

fig.set_size_inches(12,8)

g.map_upper(sns.scatterplot,color='#9574B3')
g.map_lower(sns.scatterplot, color='#ADA057')
g.map_diag(plt.hist, color='#e34a33')
sns.set()
plt.show()
g = sns.PairGrid(heart[heart['target']==0][['age', 'trestbps', 'chol', 'cp', 'thalach']])
fig = plt.gcf()

fig.set_size_inches(12,8)

g.map_upper(sns.scatterplot,color='#9574B3')
g.map_lower(sns.scatterplot, color='#ADA057')
g.map_diag(plt.hist, color='#e34a33')
sns.set()
plt.show()
g = sns.PairGrid(heart[heart['target']==1][['age', 'trestbps', 'chol', 'cp', 'thalach']])
fig = plt.gcf()

fig.set_size_inches(12,8)

g.map_upper(sns.scatterplot,color='#9574B3')
g.map_lower(sns.scatterplot, color='#ADA057')
g.map_diag(plt.hist, color='#e34a33')
sns.set()
plt.show()
cp_cat = [0, 1, 2, 3]
labels=['typical', 'asymptotic', 'nonanginal', 'nontypical']
labeldict=dict(zip(cp_cat, labels))

plt.figure(figsize=(16,5))
sns.countplot(x='cp', data=heart).set(title='Chest pain based on all sample data', xlabel=labeldict)
plt.show()
plt.figure(figsize=(16,5))
sns.countplot(x='target', hue='cp', data=heart).set(
            title=f'Chest pain per Heart Condition  :{labeldict}', xlabel='Heart disease(No=0 or Yes=1')   
plt.show()
twenys =list(range(20,30))
thirys =list(range(30,40))
forys =list(range(40,50))
fiftys =list(range(50,60))
sixtys =list(range(60,70))
sevenys =list(range(70,80))
eightys = list(range(80,90))

age_grp=[]
for age in heart['age']:
    if age in twenys:
        age_grp.append('twenys')
    if age in thirys:
        age_grp.append('thirys')
    if age in forys:
        age_grp.append('forys')
    if age in fiftys:
        age_grp.append('fiftys')        
    if age in sixtys:
        age_grp.append('sixtys')        
    if age in sevenys:
        age_grp.append('sevenys')
    if age in eightys:
        age_grp.append('eightys')
#create a new column in heart df
heart['age_grp'] = age_grp
heart = heart[['age','age_grp', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']] 

heart[heart['age_grp']=='twenys']
print(heart.groupby('age_grp').agg({'age':'count', 'chol':'mean', 'trestbps':'mean'}))
heart[heart['target']==1].groupby('age_grp').agg({'age':'count', 'chol':'mean', 'trestbps':'mean'})
heart_attack_age_grp_pct=heart[heart['target']==1].groupby('age_grp').agg({'age':'count'})/heart.groupby('age_grp').agg({'age':'count'})
heart_attack_age_grp_pct.rename(columns={'age':'heart_attack_pct'}, inplace=True)
heart_attack_age_grp_pct.reset_index(inplace=True) 
heart_attack_age_grp_pct.sort_values('heart_attack_pct', ascending=False, inplace=True)
plt.figure(figsize=(16,5))
sequential_colors = sns.color_palette("RdPu", 10)

sns.barplot(x=heart_attack_age_grp_pct['age_grp'],
            y=heart_attack_age_grp_pct['heart_attack_pct'] ).set(
                title= "Heart disease percentage per Age group", ylabel='%')  

 
plt.show()
age_grp_list=list(set(heart_attack_age_grp_pct['age_grp']))
age_grp_list=['twenys','thirys','forys','fiftys','sixtys','sevenys']
age_grp_list[5]
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,12))
k=0
for i in range(2):
    for j in range(3):
        sns.distplot(
            heart[heart['target']==1].groupby('age_grp').get_group(age_grp_list[k])['trestbps'], bins=10, ax=axs[i,j], color='red')              
        axs[i,j].set_title(f"{age_grp_list[k]} age group: BP distribution", weight='bold')
        k+=1
plt.show()
sns.set()
warnings.simplefilter(action='ignore', category=FutureWarning)
%config InlineBackend.figure_format ='retina'
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,12))
k=0
for i in range(2):
    for j in range(3):
        sns.distplot(
            heart[heart['target']==1].groupby('age_grp').get_group(age_grp_list[k])['chol'], bins=10, ax=axs[i,j], color='#D47DCB')              
        axs[i,j].set_title(f"{age_grp_list[k]} age group: Cholostrol distribution", weight='bold')    
        k+=1
plt.show()
sns.set()
warnings.simplefilter(action='ignore', category=FutureWarning)
%config InlineBackend.figure_format ='retina'
heart.head()
sns.set()
plt.figure(figsize=(15,8))
cmap = sns.dark_palette("#f20534", as_cmap=True)
sns.heatmap(
    heart[['age', 'sex','cp', 'trestbps', 'chol', 'thalach', 'target']].corr(),
    annot=True, vmin=-1, cmap=cmap).set(
    title='Corr between Heart disease Vs Chest Pain(cp), BP, Cholostrol, Exercise induced Angina(thalach)')  
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics  # ********* &&&& why we need this module????
heart_pred1=heart[['age', 'sex', 'cp', 'trestbps', 'chol','target']]
heart_pred2=heart[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs','target']]
heart_pred3=heart[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs','thal','target']]
X = heart_pred1[['age', 'sex', 'cp', 'trestbps', 'chol']]
y = heart_pred1['target']
print(len(X))
print(len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
print(f'X_train length is:{len(X_train)}, and y_train:{len(y_train)}')
print(f'X_test length is:{len(X_test)}, and y_test:{len(y_test)}')
print(f'Hence, the data of size {len(X)} is split in to train data of {len(X_train)} and test data of {len(X_test)}.') 
logreg = LogisticRegression()  
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test) 
y_pred[240:] # sample prediction output
#print(classification_report(y_test,predictions)) confusion_matrix(y_test, y_predict)
accuracy_test = logreg.score(X_test, y_test)
print(f' The accuracy score is: {round((accuracy_test*100),3)}%')
conf_matrix_dict = {'PREDICTED: NO':['TN', 'FP'], 'PREDICTED: YES':['FN', 'TP']}
conf_matrix_index = ['ACTUAL No', 'ACTUAL Yes']
conf_matrix_df = pd.DataFrame(conf_matrix_dict, index=conf_matrix_index)
conf_matrix_df.style.set_table_styles([{'selector' : '', 
                            'props' : [('border', 
                                        '5px solid tomato')]}])

print(confusion_matrix(y_test, y_pred))
print(f" Accuracy based on manual calculation of confusion matrix: {round((1-(27+27)/(103+100+27+27))*100,3)}%")
print(classification_report(y_test, y_pred))
heart_pred2=heart[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs','target']]
X = heart_pred2[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs']]
y = heart_pred2['target']
X_train, X_test, y_train,y_test=train_test_split(X,y, test_size=0.25, random_state=10)
logreg=LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred[240:] # sample prediction output
accuracy_test = logreg.score(X_test, y_test)
print(f' The accuracy score is: {round((accuracy_test*100),3)}%')
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
heart_pred5= heart[['age','cp','thalach','target']]
X=heart_pred5[['age','cp','thalach']]
y=heart['target'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
logreg = LogisticRegression() 
logreg.fit(X_train, y_train) 
y_pred = logreg.predict(X_test) 
print(classification_report(y_test, y_pred))
accuracy_test = logreg.score(X_test, y_test)
print(f' The accuracy score is: {round((accuracy_test*100),3)}%')
heart_pred3=heart[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs','thal','target']]
X = heart_pred3[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thal']]
y = heart_pred3['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10) 
logreg=LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
#instantiate the preprocessing
mmscaler = MinMaxScaler()
X_train_norm = mmscaler.fit_transform(X_train)
X_test_norm = mmscaler.fit_transform(X_test) 
logreg.fit(X_train_norm, y_train)
y_pred = logreg.predict(X_test_norm)
y_pred[240:] # sample data
accuracy_test = logreg.score(X_test_norm, y_test)
print(f' The accuracy score is: {round((accuracy_test*100),3)}%')
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
from sklearn.preprocessing import StandardScaler
#instantiate the preprocessing
sscaler = MinMaxScaler()
X_train_stan = sscaler.fit_transform(X_train)
X_test_stan = sscaler.fit_transform(X_test)
#y_train_norm = mmscaler.fit_transform(y_train)
#y_test_norm = mmscaler.fit_transform(y_test)
logreg.fit(X_train_stan, y_train)
y_pred = logreg.predict(X_test_stan)
y_pred[240:] # sample data
accuracy_test = logreg.score(X_test_stan, y_test)
print(f' The accuracy score is: {round((accuracy_test*100),3)}%')
confusion_matrix(y_test, y_pred)
(30+24)/257
print(classification_report(y_test, y_pred))
heart_pred4=heart[['age', 'cp', 'trestbps', 'chol','target']]
X = heart_pred4[['age', 'cp', 'trestbps', 'chol']]
y = heart_pred4['target']
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.25, random_state=10)
logreg=LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred[240:] # sample prediction data
accuracy_test = logreg.score(X_test, y_test)
print(f' The accuracy score is: {round((accuracy_test*100),3)}%')
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
heart_pred1=heart[['age', 'sex', 'cp', 'trestbps', 'chol','target']]
X = np.array(heart_pred1[['age', 'sex', 'cp', 'trestbps', 'chol']])
y = np.array(heart_pred1['target'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state =10 )
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred=logreg.predict(X_test)
print(y_test[240:])
print(y_pred[240:]) # sample data
accuracy_test = logreg.score(X_test,y_test)
print(f' The accuracy score is: {round((accuracy_test*100),3)}%')
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))