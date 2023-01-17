import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

performance=pd.read_csv("../input/student-performance/datasets_74977_169835_StudentsPerformance.csv")
performance
performance.head()  # Seeing the head
performance.tail()
performance.info()
performance.describe().T
performance.isna().sum()
cat_col=performance[performance.dtypes[performance.dtypes=='object'].index]
cat_col
performance['gender'].value_counts()
performance['race/ethnicity'].value_counts()
performance['parental level of education'].value_counts()
performance['lunch'].value_counts()
performance['test preparation course'].value_counts()
for i in cat_col.columns:
    plt.xticks(rotation=90)
    sns.countplot(cat_col[i])
    plt.show()
performance
performance[(performance['math score']>90) & (performance['reading score']>90) & (performance['writing score']>90)].sort_values(by=['math score','reading score','writing score'],ascending=False)

plt.figure(figsize=(8,8))
sns.boxplot(x='gender',y='math score',data=performance)
plt.show()
plt.figure(figsize=(8,8))
sns.boxplot(x='race/ethnicity',y='math score',data=performance)
plt.show()
plt.figure(figsize=(8,8))
plt.xticks(rotation=90)
sns.boxplot(x='parental level of education',y='math score',data=performance)
plt.show()
plt.figure(figsize=(8,8))
plt.xticks(rotation=90)
sns.boxplot(x='test preparation course',y='math score',data=performance)
plt.show()
num_col=performance[performance.dtypes[performance.dtypes!='object'].index]
num_col
num_col.skew()
import scipy.stats as stats
for i in num_col.columns:
    performance[str(i)+'_Boxcox'],lamb=stats.boxcox(performance[i]+1)

performance
performance['math score_Boxcox'].skew()
performance['square_math']=performance['math score']**(1.25)
performance['square_math'].skew()
num_col_boxcox=performance.columns[-4:-1]
num_col_boxcox
sns.boxplot(performance['math score_Boxcox'])
plt.show()
performance
num_col_boxcox=performance[performance.dtypes[performance.dtypes!='object'].index[-4:-1]]
num_col_boxcox
num_col_boxcox.skew()
for i in num_col_boxcox:
    sns.boxplot(num_col_boxcox[i])
    plt.show()
q3=num_col_boxcox['math score_Boxcox'].quantile(0.75)
q1=num_col_boxcox['math score_Boxcox'].quantile(0.25)
iqr=q3-q1
ub=q3+1.5*iqr
lb=q1-1.5*iqr
ub,lb
_quantile=num_col_boxcox['math score_Boxcox'].quantile(0.01)
_quantile
after_remove=num_col_boxcox['math score_Boxcox'].replace(num_col_boxcox[num_col_boxcox['math score_Boxcox']<lb]['math score_Boxcox'],num_col_boxcox['math score_Boxcox'].quantile(0.01))
sns.boxplot(after_remove)
plt.show()
for i in num_col_boxcox.columns:
    q3=num_col_boxcox[i].quantile(0.75)
    q1=num_col_boxcox[i].quantile(0.25)
    iqr=q3-q1
    ub=q3+1.5*iqr
    lb=q1-1.5*iqr
    num_col_boxcox.loc[i]=num_col_boxcox[i].replace(num_col_boxcox[num_col_boxcox[i]<lb][i],num_col_boxcox[i].quantile(0.01))
    
    
for i in num_col_boxcox:
    sns.boxplot(num_col_boxcox[i])
    plt.show()
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
num_col_sc=sc.fit_transform(num_col_boxcox)
num_col_sc=pd.DataFrame(num_col_sc,columns=num_col_boxcox.columns)
num_col_sc
num_col_sc.describe()
num_col_sc['math score_Boxcox'].plot(kind='kde')
num_col['math score'].plot(kind='kde')
cat_col=performance.select_dtypes(include='object')
cat_col
cat_col['parental level of education'].value_counts()
cat_col['parental level of education']=cat_col['parental level of education'].replace({"some high school":"high school"})
label_encoding_col=['parental level of education']
cat_col[label_encoding_col]=cat_col[label_encoding_col].replace({"master's degree":6,"bachelor's degree":5,"associate's degree":4,"some college":4,"high school":5})
cat_col['parental level of education'].value_counts()
cat_col['parental level of education']
cat_col_dum=pd.get_dummies(cat_col,drop_first=True)
cat_col_dum
final_data=pd.concat([cat_col_dum,num_col_sc],axis=1)
final_data
final_data.columns
abs(final_data.corr().loc['math score_Boxcox']).sort_values(ascending=False).head(5)
from sklearn.model_selection import train_test_split
inp=final_data['math score_Boxcox']
out=final_data.drop('math score_Boxcox',axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(inp,out,test_size=0.3,random_state=0)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

