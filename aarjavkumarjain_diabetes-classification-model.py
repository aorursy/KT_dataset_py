import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score

from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from scipy.stats import iqr

sns.set()
path='../input/diabetes-data-set/diabetes-dataset.csv'

df=pd.read_csv(path)

df.head()
df.shape
df.describe()
df.info()
df.nunique()
fig=plt.figure(figsize=(20,20))

for i,col in enumerate(df.drop(['Pregnancies','Outcome'],axis=1)):

    ax=fig.add_subplot(4,2,i+1)

    sns.distplot(df[col])
fig=plt.figure(figsize=(15,5))

for i,col in enumerate(['Pregnancies','Outcome']):

    ax=fig.add_subplot(1,2,i+1)

    sns.countplot(df[col])
fig=plt.figure(figsize=(15,15))

for i,col in enumerate(df.drop(['Pregnancies','Outcome'],axis=1)):

    ax=fig.add_subplot(3,3,i+1)

    sns.boxplot(y=df[col],x=df['Outcome'])
fig=plt.figure(figsize=(20,20))

for i,col in enumerate(df.drop(['Pregnancies','Outcome'],axis=1)):

    ax=fig.add_subplot(4,2,i+1)

    ax1=sns.distplot(df[col][df['Outcome']==1],label='Positive')

    sns.distplot(df[col][df['Outcome']==0],label='Negative',ax=ax1)

    plt.legend()
sns.barplot(x='Pregnancies',y='Outcome',data=df,ci=None)
fig=plt.figure(figsize=(15,15))

for i,col in enumerate(df.drop(['Pregnancies','Outcome','Glucose'],axis=1)):

    ax=fig.add_subplot(3,3,i+1)

    sns.scatterplot('Glucose',df[col],hue='Outcome',data=df)
fig=plt.figure(figsize=(15,15))

for i,col in enumerate(df.drop(['Pregnancies','Outcome','Glucose','BMI'],axis=1)):

    ax=fig.add_subplot(3,3,i+1)

    sns.scatterplot('BMI',df[col],hue='Outcome',data=df)
plt.figure(figsize=(8,5))

sns.countplot(x='Pregnancies',hue='Outcome',data=df)
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.shape
def iqr_outliers(df):

    out=[]

    q1 = df.quantile(0.25)

    q3 = df.quantile(0.75)

    iqr = q3-q1

    Lower_tail = q1 - 1.5 * iqr

    Upper_tail = q3 + 1.5 * iqr

    for i in df:

        if i > Upper_tail or i < Lower_tail:

            out.append(i)

    print("Outliers:",len(out))

for col in df.drop('Outcome',axis=1).columns:

    iqr_outliers(df[col])
for col in df.drop('Outcome',axis=1).columns:

    q1 = df[col].quantile(0.25)

    q3 = df[col].quantile(0.75)

    iqr = q3-q1

    Lower_tail = q1 - 1.5 * iqr

    Upper_tail = q3 + 1.5 * iqr

    

    df[col] = np.where((df[col]<Lower_tail) | (df[col]>Upper_tail), df[col].median(),df[col])
corr=df.corr()

plt.figure(figsize=(10,10))

plt.title('Correlation')

sns.heatmap(corr > 0.90, annot=True, square=True)
scaler=StandardScaler()
scaled_df=scaler.fit_transform(df.drop('Outcome',axis=1))
scaled_df=pd.DataFrame(scaled_df,columns=df.drop('Outcome',axis=1).columns)
scaled_df.head()
x=scaled_df

y=df.Outcome
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
model_list=[]

model_f1_score=[]

model_accuracy_score=[]
model_list.append('LogisticRegression')

lm=LogisticRegression()
lm.fit(x_train,y_train)
yhat_lm=lm.predict(x_test)
lm_score=f1_score(y_test,yhat_lm)

model_f1_score.append(lm_score)

lm_score
lm_accuracy=accuracy_score(y_test,yhat_lm)

model_accuracy_score.append(lm_accuracy)

lm_accuracy
print(classification_report(y_test,yhat_lm))
sns.heatmap(confusion_matrix(y_test,yhat_lm),annot=True,fmt='',cmap='YlGnBu')
model_list.append('DecisionTreeClassifier')

tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)
yhat_tree=tree.predict(x_test)
tree_score=f1_score(y_test,yhat_tree)

model_f1_score.append(tree_score)

tree_score
tree_accuracy=accuracy_score(y_test,yhat_tree)

model_accuracy_score.append(tree_accuracy)

tree_accuracy
print(classification_report(y_test,yhat_tree))
sns.heatmap(confusion_matrix(y_test,yhat_tree),annot=True,fmt='',cmap='YlGnBu')
model_list.append('RandomForestClassifier')

forest=RandomForestClassifier()
forest.fit(x_train,y_train)
yhat_forest=forest.predict(x_test)
forest_score=f1_score(y_test,yhat_forest)

model_f1_score.append(forest_score)

forest_score
forest_accuracy=accuracy_score(y_test,yhat_forest)

model_accuracy_score.append(forest_accuracy)

forest_accuracy
print(classification_report(y_test,yhat_forest))
sns.heatmap(confusion_matrix(y_test,yhat_forest),annot=True,fmt='',cmap='YlGnBu')
model_list.append('SVC')

svc=SVC()
svc.fit(x_train,y_train)
yhat_svc=svc.predict(x_test)
svc_score=f1_score(y_test,yhat_svc)

model_f1_score.append(svc_score)

svc_score
svc_accuracy=accuracy_score(y_test,yhat_svc)

model_accuracy_score.append(svc_accuracy)

svc_accuracy
print(classification_report(y_test,yhat_svc))
sns.heatmap(confusion_matrix(y_test,yhat_svc),annot=True,fmt='',cmap='YlGnBu')
model_list.append('KNeighborsClassifier')

neighbour=KNeighborsClassifier()
neighbour.fit(x_train,y_train)
yhat_neighbour=neighbour.predict(x_test)
neighbour_score=f1_score(y_test,yhat_neighbour)

model_f1_score.append(neighbour_score)

neighbour_score
neighbour_accuracy=accuracy_score(y_test,yhat_neighbour)

model_accuracy_score.append(neighbour_accuracy)

neighbour_accuracy
print(classification_report(y_test,yhat_neighbour))
sns.heatmap(confusion_matrix(y_test,yhat_neighbour),annot=True,fmt='',cmap='YlGnBu')
model_list.append('GaussianNB')

naive=GaussianNB()
naive.fit(x_train,y_train)
yhat_naive=naive.predict(x_test)
naive_score=f1_score(y_test,yhat_naive)

model_f1_score.append(naive_score)

naive_score
naive_accuracy=accuracy_score(y_test,yhat_naive)

model_accuracy_score.append(naive_accuracy)

naive_accuracy
print(classification_report(y_test,yhat_naive))
sns.heatmap(confusion_matrix(y_test,yhat_naive),annot=True,fmt='',cmap='YlGnBu')
fig,ax=plt.subplots(figsize=(10,8))

sns.barplot(model_list,model_f1_score)

ax.set_title("F1 Score of  Test Data",pad=20)

ax.set_xlabel("Models",labelpad=20)

ax.set_ylabel("F1_Score",labelpad=20)

plt.xticks(rotation=90)



for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    ax.annotate('{:.0%}'.format(height), (x+0.25, y + height + 0.01))
fig,ax=plt.subplots(figsize=(10,8))

sns.barplot(model_list,model_accuracy_score)

ax.set_title("Accuracy of Models on Test Data",pad=20)

ax.set_xlabel("Models",labelpad=20)

ax.set_ylabel("Accuracy",labelpad=20)

plt.xticks(rotation=90)



for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    ax.annotate('{:.0%}'.format(height), (x+0.25, y + height + 0.01))