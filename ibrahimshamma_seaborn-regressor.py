import pandas as pd
import os
os.getcwd()
#import data
df = pd.read_csv('../input/StudentsPerformance.csv')
df.head()
df.shape
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.3,color_codes=True,context={"lines.linewidth":2.5})
plt.figure(figsize=(10,7))
plt.title('grades distribution')
plt.ylabel('step')
sns.distplot(df['math score'],label='math')
sns.distplot(df['writing score'],label='writing')
sns.distplot(df['reading score'],label='reading')
plt.xlabel('grades')
plt.legend()
#ploting grades together with Race
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='race/ethnicity',y='math score',data=df,color='black')
sns.pointplot(x='race/ethnicity',y='writing score',data=df,color='blue')
sns.pointplot(x='race/ethnicity',y='reading score',data=df,color='#3399FF')
plt.text(0.5,75.0,'math',color='black',fontsize = 17,style = 'italic')
plt.text(0.5,72.5,'writing',color='blue',fontsize = 17,style = 'italic')
plt.text(0.5,73.75,'reading',color='#3399FF',fontsize = 17,style = 'italic')
plt.xlabel('RACE',fontsize = 15,color='blue')
plt.ylabel('score',fontsize = 15,color='blue')
plt.title('score in respect of RACE',fontsize = 20,color='blue')
#ploting each grade in respect of parent degree
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='parental level of education',y='math score',data=df,color='lime')
sns.pointplot(x='parental level of education',y='writing score',data=df,color='blue')
sns.pointplot(x='parental level of education',y='reading score',data=df,color='r')
plt.text(0.9,77.5,'math',color='lime',fontsize = 17,style = 'italic')
plt.text(0.9,75.0,'writing',color='blue',fontsize = 17,style = 'italic')
plt.text(0.9,76.25,'reading',color='r',fontsize = 17,style = 'italic')
plt.xlabel('parent degree',fontsize = 15,color='blue')
plt.ylabel('score',fontsize = 15,color='blue')
plt.title('score in respect of parental degree',fontsize = 20,color='blue')
a = []
for i in range(len(df['math score'])):
    a.append(df['math score'][i]+df['writing score'][i]+df['reading score'][i])
a=pd.DataFrame(a)
df['score']=a
df['score'].head()
#try to boxplot all grades combined together with race
plt.figure(figsize=(15,8))
sns.boxplot(x="race/ethnicity", y='score', data=df)
plt.title('score in comparision with Race')
plt.xlabel('RACE',fontsize = 15,color='blue')
plt.ylabel('score',fontsize = 15,color='blue')
plt.show()
plt.figure(figsize=(15,10))
plt.title("score distribution",fontsize=15)
sns.distplot(df['score'],color='c')
plt.show()
male = []
female = []
for i in range(len(df)):
    if df['gender'][i]=='female':
        female.append(df['score'][i])
    else:
        male.append(df['score'][i])
male=pd.DataFrame(male)
female=pd.DataFrame(female)
male['male']=male
female['female']=female
plt.figure(figsize=(15,10))
plt.title("overall score distribution",fontsize=20)
sns.distplot(male['male'],color='blue',label='male')
sns.distplot(female['female'],color='fuchsia',label='Female')
plt.show()
#predictive model 
type(df['math score'][0])
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
#encoding train data
gen_enc=enc.fit_transform(df['gender'])
gen_enc = pd.DataFrame(gen_enc,columns=['gen_enc'])
#reducing dimensions of labeled data by summing all grades
df1=pd.concat([gen_enc,df],axis=1)
df1=df1.drop(['gender'],axis=1)
df1.head(3)
dum_df = pd.get_dummies(df1[['race/ethnicity','parental level of education']])
df1 = pd.concat([df1,dum_df],axis=1)
df1 = df1.drop(['race/ethnicity','parental level of education'],axis=1)
df1['score']=np.array(df['math score'])+np.array(df['reading score'])+np.array(df['writing score'])
#remove some useless info
my_df = df1.drop(['math score','reading score','writing score'],axis=1)
enc_lunch=pd.DataFrame(enc.fit_transform(my_df['lunch']),columns=['enc_lunch'])
enc_tpc = pd.DataFrame(enc.fit_transform(my_df['test preparation course']),columns=['enc_tpc'])
my_df1 = pd.concat([enc_lunch,enc_tpc,my_df],axis=1)
my_df1 = my_df1.drop(['lunch','test preparation course'],axis=1)
my_df1.columns
#setting x and y data
from sklearn.model_selection import train_test_split
x = my_df1[['enc_lunch', 'enc_tpc', 'gen_enc', 'race/ethnicity_group A','race/ethnicity_group B', 'race/ethnicity_group C','race/ethnicity_group D', 'race/ethnicity_group E',"parental level of education_associate's degree","parental level of education_bachelor's degree","parental level of education_high school","parental level of education_master's degree","parental level of education_some college","parental level of education_some high school"]]
y = my_df1['score']
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.25)
#finally build the module
from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
#building simple cost function 
from sklearn.metrics import mean_squared_error
error = np.sqrt(mean_squared_error(y_true=y_test,y_pred=y_pred))
a = 0
for i in y_test.index :
    a +=y_test[i]
print('percentage error is {}%'.format((error/a)*100))
#let's visualize predicted data in comparision with real data
df['predictions'] = pd.DataFrame(y_pred)
plt.figure(figsize=(10,8))
sns.boxplot(x="race/ethnicity", y='score', data=df)
sns.boxplot(x="race/ethnicity", y='predictions', data=df)
plt.title('score in comparision with Race',fontsize=20)
plt.xlabel('RACE',fontsize = 15,color='blue')
plt.ylabel('score',fontsize = 15,color='black')
plt.show()
#as we can see we barely can see the difference between predictions and test data
