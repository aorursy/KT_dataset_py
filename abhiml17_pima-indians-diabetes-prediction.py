import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline
df_pi=pd.read_csv('../input/PimaIndians.csv')
df_pi.head(3)
df_pi.describe()
df_pi.info()
#converting test column which is categorical to numerical negatif=0 ,and positif=1
df_pi['test'].replace(['negatif','positif'],[0,1],inplace=True)
df_pi.head(3)
plt.figure(figsize=(12,6))
sb.heatmap(df_pi[df_pi.columns[0:]].corr(),annot=True)
#As, we can see Glucose level and age are two main factors that are cooreralted with diabetes in this case 
sb.jointplot(x='test',y='age',data=df_pi)
sb.barplot(x='test',y='glucose',data=df_pi)
#mean glucose level of people not having diabetes is around 110 and  people having diabetes is around 140
sb.barplot(x='test',y='age',data=df_pi)
#mean age level of people not having diabetes is around 28 and  people having diabetes is around 35
df_pi.columns
#Let's do Feature scaling on the entire data set
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X=df_pi[['pregnant', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi','diabetes', 'age']]
y=df_pi['test']           
std_scaler.fit(X)
scaled_X=std_scaler.transform(X)
final_X=pd.DataFrame(scaled_X,columns=['pregnant', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi','diabetes', 'age'])
final_X.head(3)
#NON PCA Approach
# Let's divide the data set into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_X,y,test_size=0.30)
#try SVM
from sklearn.svm import SVC
model = SVC()
model.kernel='rbf'
model.C=0.25
model.fit(X_train,y_train)
#predict
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))

# Try Logistic Regression
from sklearn.linear_model import LogisticRegression
lgrm=LogisticRegression()
lgrm.fit(X_train,y_train)
preds=lgrm.predict(X_test)
print(classification_report(y_test,preds))

# Try KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(classification_report(y_test,pred))

#Try using PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_X)
X_pca = pca.transform(scaled_X)
X_pca.shape
scaled_X.shape
final_X_pca=pd.DataFrame(X_pca,columns=['FC 1', 'FC 2'])
X_train, X_test, y_train, y_test = train_test_split(final_X_pca,y,test_size=0.30)
#SVM +PCA
model = SVC()
model = SVC()
model.kernel='rbf'
model.C=0.25
model.fit(X_train,y_train)
#predict
predictions = model.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
