import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sc
sns.set()
data=pd.read_csv(r"../input/insurance/insurance.csv")
data.head(10)
data.isnull().sum()#No Missing Values
data.describe().T
data['smoker'].value_counts()
plt.subplot(1,2,1)
plt.pie(data['smoker'].value_counts(),labels=data['smoker'].value_counts().index,autopct="%.1f%%")
plt.title('Smoker vs Non-Smoker Count')

plt.subplot(1,2,2)
plt.pie(data['sex'].value_counts(),labels=data['sex'].value_counts().index,autopct="%.1f%%")
plt.title('Male vs Female Count')
plt.show()
plt.figure(figsize=(4,4))
sns.heatmap(data.corr(),cmap='coolwarm',annot=True,linewidths=0.5)
plt.show()

plt.figure(figsize=(8,12))
plt.subplot(3,1,1)
sns.boxplot(data['age'])
plt.subplot(3,1,2)
sns.boxplot(data['bmi'])
plt.subplot(3,1,3)
sns.boxplot(data['charges'])
plt.show()
sc.skew(data['charges'])
#Charges data is highly skewed
gender_smoker=pd.crosstab(data['sex'],data['smoker'])
gender_smoker
gender_smoker.plot(kind="bar")
plt.title("Smoker(Yes/No) Count by Gender")
plt.show()
sns.pairplot(data)
#LET'S GRAB THE USEFUL ONES
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.scatterplot(data['bmi'],data['charges'],hue=data.smoker)
plt.subplot(1,3,2)
sns.scatterplot(data['age'],data['charges'],hue=data.smoker)
plt.subplot(1,3,3)
sns.scatterplot(data['children'],data['charges'],hue=data.smoker)
plt.show()
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.scatterplot(data[data["sex"]=="female"]['age'],data[data["sex"]=="female"]['charges'],hue=data[data["sex"]=="female"]["smoker"])
plt.title("Female CHARGES vs AGE")
plt.subplot(1,2,2)
sns.scatterplot(data[data["sex"]=="male"]['age'],data[data["sex"]=="male"]['charges'],hue=data[data["sex"]=="male"]["smoker"])
plt.title("Male CHARGES vs AGE")
plt.show()
sns.distplot(data['age'])
plt.title("Distribution of Age")
plt.show()
young_smokers=data[(data['age']<20) & (data['smoker']=='yes')]
young_smokers.head()
sns.boxplot(x=young_smokers.charges)
plt.title("Charges range of young smokers < 20 yrs of age")
plt.show()
sns.boxplot(x=young_smokers['region'],y=young_smokers['charges'])
plt.title("Regionwise Charges Distribution for smokers of age < 20yrs ")
plt.show()
young_smokers.groupby(['region','smoker']).sum()['charges']
young_smokers[young_smokers.smoker=='yes']['region'].value_counts()
sns.boxplot(x=data['region'],y=data['charges'],hue=data.smoker)
plt.title("Regionwise Charges Distribution")
plt.show()
sns.countplot(data.region,hue=data.smoker)
higher_charge=data[data.charges>16639.912515]
plt.figure(figsize=(10,12))
plt.subplot(3,1,1)
sns.countplot(higher_charge['region'],hue=higher_charge['smoker'])
plt.subplot(3,1,2)
sns.countplot(higher_charge['sex'],hue=higher_charge['smoker'])
plt.subplot(3,1,3)
sns.boxplot(higher_charge['age'],color="0.25")
plt.show()
higher_charge.head(2)
higher_charge['BMI Category']=pd.cut(higher_charge.bmi,bins=[0,18.5,25,30,53],labels=['Underweight','Healthy weight','Overweight','Obese']).values
higher_charge['Age Category']=pd.cut(higher_charge.age,bins=[18,30,42,54,64],labels=['Young Adult','Middle Adult','Senior Adult','Elder']).values
higher_charge
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.scatterplot(higher_charge['age'],higher_charge['charges'],hue=higher_charge['BMI Category'])
plt.subplot(1,2,2)
sns.scatterplot(higher_charge['age'],higher_charge['charges'],hue=higher_charge['smoker'])
plt.show()
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.countplot(higher_charge['BMI Category'])
plt.subplot(1,2,2)
sns.countplot(higher_charge['Age Category'])
plt.show()
sns.countplot(higher_charge.sex,hue=higher_charge['BMI Category'])
plt.show()
plt.figure(figsize=(20,5))
sns.countplot(data.age,hue=data.smoker)
plt.title("AGE COUNT")
plt.show()
Ho="Gender has no effect on smoking habits"
Ha="Gender has effect on smoking habits"
chi,p_value,dof,expected=sc.chi2_contingency(gender_smoker)
if (p_value>0.05):
    print(Ho)
else:
    print(Ha)
chi
#tabular_chi=3.84,thus reject Ho
gender_smoker#clearly we have more male smokers
Ho="Region has no effect on charges"
Ha="Region has effect on charges"
sample=data.sample(100)
charges=pd.cut(sample['charges'].sort_values(ascending=False),bins=15)
chi2,p,ddof,ex=sc.chi2_contingency(pd.crosstab(sample['region'],charges))
if (chi2<58.124):
    print(Ho)
else:
    print(Ha)
sns.distplot(data['charges'])

#Forming sample dataset randomly
import random
n=1000
ks=100
sample_dataset=[]
for i in range(0,n):
    sample_dataset.append(random.choices(data["charges"],k=ks))
#Calculating mean from means of sample sets: SAMPLE MEAN/s_mean
sample_means=[]
for i in sample_dataset:
    sample_means.append(np.mean(i))
s_mean=np.mean(sample_means)
s_mean
#ACTUAL DATA MEAN
data['charges'].mean()
s_var=np.var(sample_means)
s_std=np.std(sample_means)
s_var
data['charges'].var()/(100)#Actual variance
s_std
data['charges'].std()/np.sqrt(100)#Actual standard_deviation
data[data['charges']>16639.912515]
#this comprises of about 25% data
ub_bmi=34.595000+1.5*(34.595-26.315)#upperboundary_bmi
data['bmi']=np.where(data['bmi']>ub_bmi,ub_bmi,data['bmi'])#replacing outliers in bmi with ub_bmi
data['bmi'].plot(kind="box")
plt.show()#No outliers
data=pd.get_dummies(data)#encoding CATEGORICAL DATA
data.head()
sns.heatmap(data.corr(),cmap="cool")#High corr with smoker
#Splitting data into train & test
from sklearn.model_selection import train_test_split
x=data.drop(['charges'],axis=1)
y=data['charges']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#Training the model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
#Predicting the values
y_pred=lr.predict(x_test)
y_pred_train=lr.predict(x_train)
from sklearn.metrics import mean_squared_error,r2_score#TO TEST THE MODEL
mean_squared_error(y_test,y_pred)
np.sqrt(35746516.8773678)
r2_score(y_test,y_pred)
r2_score(y_train,y_pred_train)