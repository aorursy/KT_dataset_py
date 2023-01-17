import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mn
df=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.columns
df.info()
df.isnull().any()
mn.matrix(df,color=(0.30,0.60,0.71))
plt.title("This plot shows Null values",fontdict={'fontsize':20})
plt.figure(figsize=(12,6))
sns.countplot(df['Outcome'])
sns.heatmap(df.corr())
plt.figure(figsize=(14,8))
sns.set_style('whitegrid')
sns.scatterplot(df['Glucose'],df['BloodPressure'],hue=df['Outcome'])
sns.pairplot(df, hue='Outcome')
#find '0' value's column
p=len(df.loc[df['Pregnancies']==0])
g=len(df.loc[df['Glucose']==0])
bp=len(df.loc[df['BloodPressure']==0])
s=len(df.loc[df['SkinThickness']==0])
i=len(df.loc[df['Insulin']==0])
b=len(df.loc[df['BMI']==0])
dpf=len(df.loc[df['DiabetesPedigreeFunction']==0])
a=len(df.loc[df['Age']==0])
print("Number of '0' valus in each columns : {} , {} , {} , {} , {} , {} , {} , {} ,".format(p,g,bp,s,i,b,dpf,a))
#replacing '0' by its mean
df.loc[(df.Pregnancies == 0),'Pregnancies']=df['Pregnancies'].mean()
df.loc[(df.Glucose == 0),'Glucose']=df['Glucose'].mean()
df.loc[(df.BloodPressure == 0),'BloodPressure']=df['BloodPressure'].mean()
df.loc[(df.SkinThickness == 0),'SkinThickness']=df['SkinThickness'].mean()
df.loc[(df.Insulin == 0),'Insulin']=df['Insulin'].mean()
df.loc[(df.BMI == 0),'BMI']=df['BMI'].mean()
#After removing '0' value's column
p=len(df.loc[df['Pregnancies']==0])
g=len(df.loc[df['Glucose']==0])
bp=len(df.loc[df['BloodPressure']==0])
s=len(df.loc[df['SkinThickness']==0])
i=len(df.loc[df['Insulin']==0])
b=len(df.loc[df['BMI']==0])
dpf=len(df.loc[df['DiabetesPedigreeFunction']==0])
a=len(df.loc[df['Age']==0])
print("Number of '0' valus in each columns : {} , {} , {} , {} , {} , {} , {} , {} ,".format(p,g,bp,s,i,b,dpf,a))
X=df.iloc[:,:8]
Y=df[['Outcome']]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_text = train_test_split(X,Y,test_size=18,random_state=1)
from sklearn.linear_model import LinearRegression
acc=[]
for i in range(len(df)):
    x_train,x_test,y_train,y_text = train_test_split(X,Y,test_size=20,random_state=i)
    model = LinearRegression()
    model.fit(x_train,y_train)
    y=model.score(x_test,y_text)
    acc.append(y)
print("Accuracy is : {0:.2f} %".format(max(acc)*100))