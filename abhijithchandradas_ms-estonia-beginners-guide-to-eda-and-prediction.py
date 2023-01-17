import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd  #i/o operations and data analysis

import numpy as np   # linear algebra

import matplotlib.pyplot as plt  # viz

import seaborn as sns  #viz

#Machine Learning

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier



from xgboost import XGBClassifier



plt.style.use('ggplot') #set style for charts

import warnings #filter warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')

df.head()
#check for null values

df.isna().sum()
sns.countplot('Sex',data=df)

plt.show()

df.Sex.value_counts()
print('Mean age : {}'.format(round(df.Age.mean(),2)))

plt.figure(figsize=(9,4))

plt.subplot(1,2,1)

sns.boxplot(df.Age)

plt.subplot(1,2,2)

sns.kdeplot(df.Age)

plt.tight_layout()

plt.show()
print("No: Passengers : {}".format(df.Category.value_counts()['P']))

print("No: Crew Members : {}".format(df.Category.value_counts()['C']))

sns.countplot(df.Category)

plt.show()
n=df.PassengerId.count()

print('Total Passengers : {}'.format(n))

print('Total Survivors : {}'.format(df.Survived.sum()))

print(f"Survival Percentage : {round(df.Survived.mean()*100,2)}%")
plt.title("Survived Vs Succumbed")

x=[df.Survived.value_counts()[1],df.Survived.value_counts()[0]]

labels= ['Survived', 'Dead']

explode=[0.05,0.05]

plt.pie(x=x,labels=labels, autopct='%1.2f%%', explode=explode)

plt.show()
c=df.Country.value_counts().sort_values(ascending=False)

plt.figure(figsize=(8,4))

sns.barplot(y=c.index,x=c.values*100/n, orient='h')

plt.show()
df['Ctry']=df.Country.apply(lambda x: 'Estonia' if x=='Estonia' 

                            else ('Sweden' if x=='Sweden' else 'Others'))

df.head(5)
# Delete PassengerId, Country, Firstname and Lastname as they are not required for further analysis

df=df[['Sex', 'Age','Category', 'Survived', 'Ctry']]

df.head()
print(f'Mean age of Survivors : {round(df[(df.Survived==1)].Age.mean(),2)}')

print(f'Mean age of Death : {round(df[(df.Survived==0)].Age.mean(),2)}')

plt.title("Age distribution Survived/Died")

sns.kdeplot(data=df[(df.Survived==1)].Age, label='Survived')

sns.kdeplot(data=df[ (df.Survived==0)].Age, label='Not Survived')

plt.show()
sns.countplot('Survived', hue='Sex', data=df)

plt.show()

print("Survival Rate (Male) : {}%".format(round(df.Survived[df.Sex=='M'].mean()*100,2)))

print("Survival Rate (Female) : {}%".format(round(df.Survived[df.Sex=='F'].mean()*100,2)))
sns.countplot('Survived', hue='Category', data=df)

plt.show()

print("Survival Rate for C : {}%".format(round(df.Survived[df.Category=='C'].mean()*100,2)))

print("Survival Rate for P  : {}%".format(round(df.Survived[df.Category=='P'].mean()*100,2)))
plt.figure(figsize=(8,4))

plt.title('Male')

sns.kdeplot(data=df[df.Sex=='F'].Age, label='Female')

sns.kdeplot(data=df[df.Sex=='M'].Age, label='Male')

plt.show()

print("Average age of Male Passenger : {}".format(round(df[df.Sex=='M'].Age.mean(),2)))

print("Average age of Female Passenger : {}".format(round(df[df.Sex=='F'].Age.mean(),2)))
plt.figure(figsize=(8,4))

plt.title('Category Vs Age')

sns.boxplot(data=df, x='Age', y='Category')

#sns.kdeplot(data=df[df.Sex=='M'].Age, label='Male')

plt.show()

print("Average age of MS Estonia Crew : {}"

      .format(round(df[df.Category=='C'].Age.mean(),2)))

print("Average age of Passengers : {}"

      .format(round(df[df.Category=='P'].Age.mean(),2)))
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)

plt.title('P Category')

x=[df[df.Category=='P'].Sex.value_counts()['M'],

   df[df.Category=='P'].Sex.value_counts()['F']]

labels= ['Male', 'Female']

explode=[0.05,0.05]

plt.pie(x=x,labels=labels, autopct='%1.2f%%', explode=explode)



plt.subplot(1,2,2)

plt.title("C Category")

x=[df[df.Category=='C'].Sex.value_counts()['M'],

   df[df.Category=='C'].Sex.value_counts()['F']]

plt.pie(x=x,labels=labels, autopct='%1.2f%%', explode=explode)



plt.tight_layout()

plt.show()
sns.barplot(x='Ctry', y='Survived', data=df)

plt.show()

print("Survival Rate :")

print("Sweden : {}".format(round(df.Survived[df.Ctry=='Sweden'].mean()*100,2)))

print("Estonia : {}".format(round(df.Survived[df.Ctry=='Estonia'].mean()*100,2)))

print("Others : {}".format(round(df.Survived[df.Ctry=='Others'].mean()*100,2)))
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)

plt.title('Passengers')

labels=df.Ctry.unique()

x=[]

for i in labels:

    x.append(df[df.Category=='P'].Ctry.value_counts()[i])

plt.pie(x=x,labels=labels, autopct='%1.2f%%')



plt.subplot(1,2,2)

plt.title("Crew")

x=[]

for i in labels:

    x.append(df[df.Category=='C'].Ctry.value_counts()[i])

plt.pie(x=x,labels=labels, autopct='%1.2f%%')



plt.tight_layout()

plt.show()
plt.title("Age Vs Nationality")

sns.kdeplot(data=df.Age[df.Ctry=='Sweden'], label='Sweden')

sns.kdeplot(data=df.Age[df.Ctry=='Estonia'], label='Estonia')

sns.kdeplot(data=df.Age[df.Ctry=='Others'], label='Others')

plt.show()



print("Mean Age")

print("Sweden : {}".format(round(df.Age[df.Ctry=='Sweden'].mean(),2)))

print("Estonia : {}".format(round(df.Age[df.Ctry=='Estonia'].mean(),2)))

print("Others : {}".format(round(df.Age[df.Ctry=='Others'].mean(),2)))
plt.figure(figsize=(8,4))

labels=['Male', 'Female']

explode=[0.05,0.05]

plt.suptitle("GENDER VS NATIONALITY")



plt.subplot(1,3,1)

plt.title('Sweden')

x=[df[df.Ctry=='Sweden'].Sex.value_counts()['M'],

  df[df.Ctry=='Sweden'].Sex.value_counts()['F']]

plt.pie(x=x,labels=labels, explode=explode,autopct='%1.2f%%')



plt.subplot(1,3,2)

plt.title('Estonia')

x=[df[df.Ctry=='Estonia'].Sex.value_counts()['M'],

  df[df.Ctry=='Estonia'].Sex.value_counts()['F']]

plt.pie(x=x,labels=labels, explode=explode,autopct='%1.2f%%')



plt.subplot(1,3,3)

plt.title('Others')

x=[df[df.Ctry=='Others'].Sex.value_counts()['M'],

  df[df.Ctry=='Others'].Sex.value_counts()['F']]

plt.pie(x=x,labels=labels, explode=explode,autopct='%1.2f%%')



plt.tight_layout()

plt.show()
sns.boxplot(data=df, x='Age', y='Sex', hue='Survived')

plt.show()
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)

plt.title('P Category')

sns.kdeplot(data=df[(df.Category=='P') & (df.Survived==1)].Age, label='Survived')

sns.kdeplot(data=df[(df.Category=='P') & (df.Survived==0)].Age, label='Not Survived')

plt.subplot(1,2,2)

plt.title("C Category")

sns.kdeplot(data=df[(df.Category=='C') & (df.Survived==1)].Age,label='Survived')

sns.kdeplot(data=df[(df.Category=='C') & (df.Survived==0)].Age,label='Not Survived')

plt.tight_layout()

plt.show()
sns.boxplot(data=df, x='Age', y='Ctry', hue='Survived')
sns.boxplot(data=df, x='Age', y='Sex', hue='Category')
sns.boxplot(data=df, x='Age', y='Ctry', hue='Category')
df.corr()
X=df.drop('Survived', axis=1)

y=df.Survived
#Scaling Age which is the only numeric feature

scalar=MinMaxScaler()

X.Age=scalar.fit_transform(np.array(X.Age).reshape(-1,1))

X.head()
X=pd.get_dummies(X, drop_first=True)

X.head()
X_train_a, X_test, y_train_a, y_test=train_test_split(X,y,test_size=0.2, random_state=42)

X_train,X_val,y_train,y_val=train_test_split(X_train_a,y_train_a, test_size=0.2,

                                             random_state=42)

print('Train Shape :', X_train.shape)

print('Val Shape :', X_val.shape)

print('Test Shape :', X_test.shape)
model_lr=LogisticRegression()

model_lr.fit(X_train,y_train)

print('Score in Train : {}'.format(round(model_lr.score(X_train,y_train)*100,4)))

print('Score in Validation : {}'.format(round(model_lr.score(X_val,y_val)*100,4)))

print('Score in Test : {}'.format(round(model_lr.score(X_test,y_test)*100,4)))

yval_lr=model_lr.predict(X_val)

ytest_lr=model_lr.predict(X_test)
sns.barplot(X_train.columns,model_lr.coef_[0])

plt.xticks(rotation=60)

plt.show()
model_svc=SVC(random_state=42)

model_svc.fit(X_train,y_train)

print('Score in Train : {}'.format(round(model_svc.score(X_train,y_train)*100,4)))

print('Score in Validation : {}'.format(round(model_svc.score(X_val,y_val)*100,4)))

print('Score in Test : {}'.format(round(model_svc.score(X_test,y_test)*100,4)))

yval_svc=model_svc.predict(X_val)

ytest_svc=model_svc.predict(X_test)
model_dt=DecisionTreeClassifier(criterion='entropy')

model_dt.fit(X_train,y_train)

print('Score in Train : {}'.format(round(model_dt.score(X_train,y_train)*100,4)))

print('Score in Validation : {}'.format(round(model_dt.score(X_val,y_val)*100,4)))

print('Score in Test : {}'.format(round(model_dt.score(X_test,y_test)*100,4)))

yval_dt=model_dt.predict(X_val)

ytest_dt=model_dt.predict(X_test)
sns.barplot(X_train.columns,model_dt.feature_importances_)

plt.xticks(rotation=60)

plt.show()
model_rf=RandomForestClassifier(n_estimators=100)

model_rf.fit(X_train,y_train)

print('Score in Train : {}'.format(round(model_rf.score(X_train,y_train)*100,4)))

print('Score in Validation : {}'.format(round(model_rf.score(X_val,y_val)*100,4)))

print('Score in Test : {}'.format(round(model_rf.score(X_test,y_test)*100,4)))

yval_rf=model_rf.predict(X_val)

ytest_rf=model_rf.predict(X_test)
sns.barplot(X_train.columns,model_rf.feature_importances_)

plt.xticks(rotation=60)

plt.show()
model_nb=GaussianNB()

model_nb.fit(X_train,y_train)

print('Score in Train : {}'.format(round(model_nb.score(X_train,y_train)*100,4)))

print('Score in Validation : {}'.format(round(model_nb.score(X_val,y_val)*100,4)))

print('Score in Test : {}'.format(round(model_nb.score(X_test,y_test)*100,4)))

yval_nb=model_nb.predict(X_val)

ytest_nb=model_nb.predict(X_test)
model_xgb=XGBClassifier(learning_rate=0.00005, n_estimators=600,n_jobs=100, max_depth=2)

model_xgb.fit(X_train,y_train)

print('Score in Train : {}'.format(round(model_xgb.score(X_train,y_train)*100,4)))

print('Score in Validation : {}'.format(round(model_xgb.score(X_val,y_val)*100,4)))

print('Score in Test : {}'.format(round(model_xgb.score(X_test,y_test)*100,4)))

yval_xgb=model_xgb.predict(X_val)

ytest_xgb=model_xgb.predict(X_test)
sns.barplot(X_train.columns,model_xgb.feature_importances_)

plt.xticks(rotation=60)

plt.show()
models =["LR","SVM","DTC","RFC","NB", "XGB"]

scores_val =[round(model_lr.score(X_val,y_val)*100,4),

         round(model_svc.score(X_val,y_val)*100,4),

         round(model_dt.score(X_val,y_val)*100,4),

         round(model_rf.score(X_val,y_val)*100,4),

         round(model_nb.score(X_val,y_val)*100,4),

         round(model_xgb.score(X_val,y_val)*100,4)]

scores_test=[round(model_lr.score(X_test,y_test)*100,4),

         round(model_svc.score(X_test,y_test)*100,4),

         round(model_dt.score(X_test,y_test)*100,4),

         round(model_rf.score(X_test,y_test)*100,4),

         round(model_nb.score(X_test,y_test)*100,4),

         round(model_xgb.score(X_test,y_test)*100,4)]



df_scores=pd.DataFrame({'Model':models,'Score_val':scores_val, 'Score_test':scores_test})

df_scores=df_scores.sort_values(by=['Score_val','Score_test'], ascending=False)



plt.title('Model Score Comparison')

sns.barplot(data=df_scores, x='Model', y='Score_val', color='blue', 

            label='Validation', alpha=0.8)

sns.barplot(data=df_scores, x='Model', y='Score_test', color='red', 

            label='Test', alpha=0.5)

plt.legend()

plt.ylim(70,90)

plt.show()
yval_stacked=np.column_stack((yval_lr,yval_svc,yval_xgb))

ytest_stacked=np.column_stack((ytest_lr,ytest_svc,ytest_xgb))



meta_model=LogisticRegression()

meta_model.fit(yval_stacked,y_val)

print('Score in Validation : ',round(meta_model.score(yval_stacked,y_val)*100,4))

print('Score in Test : ',round(meta_model.score(ytest_stacked,y_test)*100,4))
models.append('Meta Model')

scores_test.append(round(meta_model.score(ytest_stacked,y_test)*100,4))

df_scores=pd.DataFrame({'Model':models,'Score_test':scores_test})
df_scores.sort_values(by='Score_test', ascending=False, inplace=True)

plt.title("Model Performance")

sns.barplot(x='Model',y='Score_test', data=df_scores)

plt.ylabel("Accuracy Score")

plt.xlabel("Model")

plt.ylim(70,90)

plt.show()