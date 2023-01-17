from pandas import Series,DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")
import statistics
from scipy import stats
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures 
from statsmodels.formula.api import ols
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import statsmodels.api as sm
df=pd.read_csv("Islander_data.csv")
df.head(5)
lm=ols(" Diff ~ Drug",data=df).fit()
table=sm.stats.anova_lm(lm)
print(table)
stats.ttest_ind(h1,h2,equal_var=False)
h1=[];h2=[];h3=[];
for i in np.arange(len(df["Drug"])):
    if df["Drug"][i]=="A":
        h1=np.append(h1,df["Diff"][i])
    elif df["Drug"][i]=="S":
        h2=np.append(h2,df["Diff"][i])
    elif df["Drug"][i]=="T":
        h3=np.append(h3,df["Diff"][i])
stats.ttest_ind(h1,h2,equal_var=False)
stats.ttest_ind(h1,h3,equal_var=False)
stats.ttest_ind(h2,h3,equal_var=False)
stats.ttest_ind(df["Mem_Score_Before"].values,df["Mem_Score_After"].values,equal_var=False)
df["Dosage"].unique()
d=[sum((pd.get_dummies(df["Dosage"]))[1]),sum((pd.get_dummies(df["Dosage"]))[2]),sum((pd.get_dummies(df["Dosage"]))[3])]
plt.pie(d,labels=["dose1","dose2","dose3"],radius=1.6,shadow=True,explode=[.1,.1,.1],autopct='%1.1f%%')
h=0
s=0
for i in np.arange(len(df["Happy_Sad_group"])):
    if df["Happy_Sad_group"][i]=="H":
        h=h+1
    else:
        s=s+1
m=[h,s]
ms=Series(m,["happy","sad"])
ms
d1=[sum((pd.get_dummies(df["Drug"]))["A"]),sum((pd.get_dummies(df["Drug"]))["S"]),sum((pd.get_dummies(df["Drug"]))["T"])]
plt.pie(d1,labels=["Drug_A","Drug_S","Drug_T"],radius=1.6,shadow=True,autopct='%1.1f%%')
df["Mem_Score_Before"].plot(kind="kde")
df["Diff"].plot(kind="kde")
df["age"].plot(kind="kde")
(np.corrcoef(df["age"].values,df["Diff"].values))[0,1]
(np.corrcoef(df["Mem_Score_After"].values,df["Mem_Score_Before"].values))[0,1]
sns.lmplot("Mem_Score_After","Mem_Score_Before",df,hue="Happy_Sad_group")
sns.kdeplot(df["Diff"],df["age"],shade=True,color="green")
