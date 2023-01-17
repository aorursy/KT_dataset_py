import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('darkgrid')
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

df.head()
df.Outcome = df.Outcome.replace({0:'Non-Diab',1:'Diab'})

df.DiabetesPedigreeFunction = df.rename({'DiabetesPedigreeFunction':'DPF'},inplace = True,axis =1)

df.head()
df.dtypes
df.shape
df.info()
df.describe().T
plt.figure(dpi=120)

sns.pairplot(df)

plt.show()
plt.figure(dpi = 120)

sns.pairplot(df,hue = 'Outcome',palette = 'plasma')

plt.legend(['Non Diabetic','Diabetic'])

plt.show()
plt.figure(dpi = 120,figsize= (5,4))

mask = np.triu(np.ones_like(df.corr(),dtype = bool))

sns.heatmap(df.corr(),mask = mask, fmt = ".2f",annot=True,lw=1,cmap = 'plasma')

plt.yticks(rotation = 0)

plt.xticks(rotation = 90)

plt.title('Correlation Heatmap')

plt.show()
plt.figure(dpi = 100, figsize = (5,4))

print("Joint plot of Glucose with Other Variables ==> \n")

for i in  df.columns:

    if i != 'Glucose' and i != 'Outcome':

        print(f"Correlation between Glucose and {i} ==> ",df.corr().loc['Glucose'][i])

        sns.jointplot(x='Glucose',y=i,data=df,kind = 'regression',color = 'purple')

        plt.show()
col = list(df.columns)

idx = col.index('BloodPressure')



plt.figure(dpi = 100, figsize = (5,4))

print("Joint plot of BloodPressure with Other Variables ==> \n")

for i in  range(idx+1,len(col)-1):

    print(f"Correlation between BloodPressure and {col[i]} ==> ",df.corr().loc['BloodPressure'][col[i]])

    sns.jointplot(x='BloodPressure',y=col[i],data=df,kind = 'regression',color = 'green')

    plt.show()
col = list(df.columns)

idx = col.index('SkinThickness')



plt.figure(dpi = 100, figsize = (5,4))

print("Joint plot of SkinThickness with Other Variables ==> \n")

for i in  range(idx+1,len(col)-1):

    print(f"Correlation between SkinThickness and {col[i]} ==> ",df.corr().loc['SkinThickness'][col[i]])

    sns.jointplot(x='SkinThickness',y=col[i],data=df,kind = 'regression',color = 'blue')

    plt.show()
col = list(df.columns)

idx = col.index('Insulin')



plt.figure(dpi = 100, figsize = (5,4))

print("Joint plot of Insulin with Other Variables ==> \n")

for i in  range(idx+1,len(col)-1):

    print(f"Correlation between Insulin and {col[i]} ==> ",df.corr().loc['Insulin'][col[i]])

    sns.jointplot(x='Insulin',y=col[i],data=df,kind = 'regression',color = 'green')

    plt.show()
col = list(df.columns)

idx = col.index('BMI')



plt.figure(dpi = 100, figsize = (5,4))

print("Joint plot of BMI with Other Variables ==> \n")

for i in  range(idx+1,len(col)-1):

    print(f"Correlation between BMI and {col[i]} ==> ",df.corr().loc['BMI'][col[i]])

    sns.jointplot(x='BMI',y=col[i],data=df,kind = 'regression',color = 'green')

    plt.show()
col = list(df.columns)

idx = col.index('DPF')



plt.figure(dpi = 100, figsize = (5,4))

print("Joint plot of DPF with Other Variables ==> \n")

for i in  range(idx+1,len(col)-1):

    print(f"Correlation between DPF and {col[i]} ==> ",df.corr().loc['DPF'][col[i]])

    sns.jointplot(x='DPF',y=col[i],data=df,kind = 'regression',color = 'red')

    plt.show()
x= df.iloc[:,:-1].values

y= df.iloc[:,-1].values



from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(x)



x_new = pca.transform(x)



xs = x[:,0]

ys = x[:,1]



plt.figure(dpi=100)

sns.scatterplot(x=xs,y=ys,hue=y).set_title('Dependency of Data with Outcome')

plt.xlabel('PCA Feature 1')

plt.ylabel('PCA Feature 2')

plt.show()