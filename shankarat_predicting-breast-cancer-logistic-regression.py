#Import necessory libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix,classification_report,precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import plotly 
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Get the CSV data here and print head
df = pd.read_csv("/kaggle/input/breast-cancer/breast cancer.csv",index_col=0)
df.head()
#print summary
print ('Shape        ------>', df.shape)
print ('Each Column and data type and its count','\n')
print ( df.info())
#DROP ALERT 1 : Unnamed:32 column has all nulls. Safe to remove the column.
df = df.drop(['Unnamed: 32'],axis=1)
# Dataframe statistics
df.describe()
# Validate each class to understand if the dataset is imbalanced.

print ('Total Belign Tumor (B)    :  {} and its percentage is {} %'.format(df.diagnosis.value_counts()[0], round(df.diagnosis.value_counts()[0]/df.diagnosis.value_counts().sum()*100,2)) )
print ('Total Malignant Tumor (M) :  {} and its percentage is {} %'.format(df.diagnosis.value_counts()[1], round(df.diagnosis.value_counts()[1]/df.diagnosis.value_counts().sum()*100,2)) )
#Plot each class freequency
plt.figure(figsize=(6,4))
sns.countplot(x='diagnosis',data=df,palette='rocket')
df_corr = df.corr().abs()
mask = np.zeros_like(df_corr)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(25, 20))
sns.heatmap(df_corr, mask=mask, vmin=-1, vmax=1, annot=True,
            square=True, center=0, linewidths=.5)
plt.tight_layout()
#Shape of df_corr (just to cross verify)
print ('Correlation matrix Shape =  ', df_corr.shape)
scaler = StandardScaler()
X = df.drop('diagnosis', axis=1)
X_scaled = scaler.fit_transform(X)
pca = PCA()
pca.fit_transform(X_scaled)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Principal Components')
plt.ylabel('Variance Covered')
plt.title('PCA')
plt.show()
pca = PCA(n_components=13)
new_pcomp = pca.fit_transform(X_scaled)
princi_comp = pd.DataFrame(new_pcomp,columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC7','PC9','PC10','PC11','PC12','PC13'])
princi_comp

df['diagnosis'] = df['diagnosis'].replace({'M':1,'B':0})
y = df['diagnosis']
x_train,x_test,y_train,y_test = train_test_split(princi_comp,y,test_size = 0.25, random_state= 355)

clf = LogisticRegression()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print (classification_report(y_test,y_pred,digits=2))
precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.step(recall, precision, color='r', alpha=0.8,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs Recall')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show()
cfm = confusion_matrix(y_test,y_pred)
cfm
tp = cfm[0,0]
fp = cfm[0,1]
fn = cfm[1,0]
tn = cfm[1,1]
print ('True Positive  >', tp)
print ('False Positive >', fp)
print ('False Negetive >', fn)
print ('True Negetive  >', tn)
ax=plt.subplot()
sns.heatmap(cfm,annot=True,ax=ax,linewidths=5,linecolor='b',center=0)
ax.set_xlabel('Predicted Labels');ax.set_ylabel('True Labels')
ax.set_title('Confusion matrix')
ax.xaxis.set_ticklabels(['Belign','Malignant'])
ax.yaxis.set_ticklabels(['Belign','Malignant'])
plt.show()
print ('Final prediction >', round((tp+tn)/(len(y_pred))*100,2))
