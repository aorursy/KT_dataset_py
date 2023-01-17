import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
import seaborn as sns
df = pd.read_csv("../input/Z-Alizadeh sani dataset (2).csv")
df=df.drop(columns=["Exertional CP"])
df=df.dropna() 
np.any(df.isnull()) == True 
df1=df[['Age','Weight','Length','BMI','BP','PR','FBS','CR','TG','LDL','HDL','BUN','ESR','HB','K','Na','WBC','Lymph','Neut','PLT','EF-TTE','Cath']]
df2=df.drop(columns=['Age','Weight','Length','BMI','BP','PR','FBS','CR','TG','LDL','HDL','BUN','ESR','HB','K','Na','WBC','Lymph','Neut','PLT','EF-TTE'])
for i in range(0,21):
    sns.boxplot(x='Cath',y=df1.iloc[:, i],hue='Cath',data=df1,palette='Set2')
    plt.show()
for i in range (32):
    k1=pd.crosstab(df2["Cath"],df2.iloc[:,i], normalize='columns')
    k1.plot.bar(stacked=True)
from scipy import stats
for i in range (32):
    t=pd.crosstab(index=df2.iloc[:,i], columns=df2["Cath"])
    chi2_stat, p_val, dof, ex = stats.chi2_contingency(t)
    print(p_val)
df.info()
df.replace(('Y', 'N'), (1, 0), inplace=True)
for i in range(0,303):
    if df.at[i,'Cath']=='Cad':
         df.at[i,'Cath'] = 0
    else:
        df.at[i,'Cath'] = 1
df['Cath']=df['Cath'].astype(str).astype(int)
for i in range(0,303):
    if df.at[i,'Sex']=='Male':
         df.at[i,'Sex'] = 0
    else:
        df.at[i,'Sex'] = 1
df['Sex']=df['Sex'].astype(str).astype(int)
for i in range(0,303):
    if df.at[i,'VHD']=='mild':
         df.at[i,'VHD'] = 1
    elif df.at[i,'VHD']=='Moderate':
        df.at[i,'VHD'] = 2
    elif df.at[i,'VHD']=='Severe':
        df.at[i,'VHD'] = 3
df['VHD']=df['VHD'].astype(str).astype(int)
df.info()
df[df.columns[:]].corr()['Cath'][:-1]
#correlation top 10 related to Cath
abs(df[df.columns[:]].corr()['Cath'][:-1])>0.2
#df10=df[["Age","DM","HTN","BP","Typical Chest Pain","Atypical","Nonanginal","Tinversion","EF-TTE","Region RWMA","Cath"]]
df10=df[["Age","DM","HTN","BP","Typical Chest Pain","FBS","Nonanginal","Tinversion","EF-TTE","Region RWMA","Cath"]]
sns.heatmap(df10.corr(), annot=True, fmt=".2f")
plt.show()
df20=df[["Age","DM","HTN","BP","Typical Chest Pain","FBS","Nonanginal","Tinversion","EF-TTE","Region RWMA","VHD","K","ESR","PR","Q Wave","Diastolic Murmur","St Depression","TG","St Elevation", "Cath"]]
Var_Corr = df20.corr()
# plot the heatmap and annotation on it
sns.heatmap(abs(Var_Corr), xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns)
#df=df10
#df=df20
y=df['Cath']
X=df.drop(columns=['Cath'])
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)
print(X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
k_range = list(range(1, 26))
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
accuracy1= max(scores)
print(accuracy1)

# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred2 = logreg.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy2=accuracy_score(y_test, y_pred2)

from sklearn.metrics import confusion_matrix 
print(confusion_matrix(y_test, y_pred2))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred2))
from sklearn.metrics import roc_curve  
fpr, tpr, thresholds = roc_curve(y_test, y_pred2) 
from sklearn.metrics import auc  
print(auc(fpr, tpr))
from sklearn.metrics import log_loss
print(log_loss(y_test, y_pred2))
from sklearn.tree import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred3 = clf.predict(X_test)
accuracy3=accuracy_score(y_test, y_pred3)

print(confusion_matrix(y_test, y_pred3))
print(classification_report(y_test, y_pred3))
fpr, tpr, thresholds = roc_curve(y_test, y_pred3) 
print(auc(fpr, tpr))  
print(log_loss(y_test, y_pred3))
from collections import OrderedDict
c=OrderedDict([('knn',[accuracy1]),('logestic',[accuracy2]),('desiciontree',[accuracy3])])
print(pd.DataFrame(c))


