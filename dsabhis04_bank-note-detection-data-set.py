#Just an EXPLORATORY analysis. 
#Visulaizing it and running through some Algorithms.
#WILL UP LOAD MORE ALGORITHMS IN SHORT TIME.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

d = pd.read_csv('../input/bank_notes.csv',encoding='latin1')


df = pd.DataFrame(d)
df.head()
sns.countplot(x='Target',data=df)
df.info()
df.describe()
sns.barplot(x='Target',y='variance',data=df,hue='Target')
sns.boxplot(x='Target',y='variance',data=df,hue='Target')
sns.jointplot(x='Target',y='variance',data=df)#,hue='Target')
sns.stripplot(x='Target',y='variance',data=df,hue='Target')
sns.boxplot(x='Target',y='skewness',data=df,hue='Target')
sns.boxplot(x='Target',y='curtosis',data=df,hue='Target')
sns.boxplot(x='Target',y='entropy',data=df,hue='Target')
heatmp = df.corr()
sns.heatmap(heatmp,annot=True)
#sns.clustermap(df)
sns.distplot(df.variance)
sns.distplot(df.curtosis)
sns.distplot(df.entropy)
sns.pairplot(df,hue='Target')
df['Target'].value_counts()
df.head()
y=df['Target']
y.head()
X=df.drop('Target',axis=1)
feat = pd.DataFrame(X)
feat.head()
feat.info()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feat, y, test_size=0.30,stratify=y )
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(criterion='gini')
dtree.fit(X_train,y_train)
prediction = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print (confusion_matrix(y_test,prediction))
print('\n')
print (classification_report(y_test,prediction))
print('\n')
print (accuracy_score(y_test,prediction))
print('\n')
print ('Accuracy on training set:{:.3f}'.format(dtree.score(X_train,y_train)))
print ('Accuracy on training set:{:.3f}'.format(dtree.score(X_test,y_test)))
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(X)
x_pca= pca.transform(X)
x_pca.shape
plt.figure(figsize=(10,6))
plt.scatter(x_pca[:,0],x_pca[:,0],c=y,cmap='plasma')
plt.xlabel('First PC')
plt.ylabel('second PC')
df_comp = pd.DataFrame(pca.components_,columns=list(X))
plt.figure(figsize=(10,8))
sns.heatmap(df_comp,cmap='plasma')
pca = PCA().fit(X)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Bank Note Variance Explained')
plt.show()
# Gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
gradt=GradientBoostingClassifier(random_state=0,max_depth=3)
gradt.fit(X_train,y_train)
grade_pred = gradt.predict(X_test)
print(confusion_matrix(y_test,grade_pred))
print('\n')
print(classification_report(y_test,grade_pred))
print('\n')
print(accuracy_score(y_test,grade_pred))
print ('Accuracy on training set:{:.3f}'.format(gradt.score(X_train,y_train)))
print ('Accuracy on testing set:{:.3f}'.format(gradt.score(X_test,y_test)))

# scaling
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)
scaled_data
d_scale = pd.DataFrame(scaled_data,columns=list(X))
# decision tree for scaled data
X_train, X_test, y_train, y_test = train_test_split(d_scale, df['Target'], test_size=0.33,stratify=df['Target'])
dtree.fit(X_train,y_train)
scale_pred = dtree.predict(X_test)
print(confusion_matrix(y_test,scale_pred))
print('\n')
print(classification_report(y_test,scale_pred))
print('\n')
print(accuracy_score(y_test,scale_pred))
print ('Accuracy on training set:{:.3f}'.format(dtree.score(X_train,y_train)))
print ('Accuracy on testing set:{:.3f}'.format(dtree.score(X_test,y_test)))


# Random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=12)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print (confusion_matrix(y_test,rfc_pred))
print('\n')
print (classification_report(y_test,rfc_pred))
print('\n')
print (accuracy_score(y_test,rfc_pred))
