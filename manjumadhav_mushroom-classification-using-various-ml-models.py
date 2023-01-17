import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df=pd.read_csv('../input/mushrooms.csv')
df.head()
df.info()
df.describe()
sns.countplot(x='class',data=df)
lb_class = LabelEncoder()
df["class_code"] = lb_class.fit_transform(df["class"])
df[["class", "class_code"]].head(5)
sns.set(style="darkgrid")
fig,axs=plt.subplots(nrows=8,ncols=3,figsize=(30, 75))

i=0
j=0
k=0

for col in df.columns:
    
    i=int(k/3)
    j=k%3
    
    axe=sns.countplot(x=col, hue="class", data=df,ax=axs[i][j]) # for Seaborn version 0.7 and more
    
    bars = axe.patches
    half = int(len(bars)/2)
    left_bars = bars[:half]
    right_bars = bars[half:]

    for left, right in zip(left_bars, right_bars):
        height_l = np.nan_to_num(left.get_height())
        height_r = np.nan_to_num(right.get_height())
        total = height_l + height_r

        axe.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")
        axe.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")
    
    k=k+1
df.columns
df=pd.get_dummies(data=df,columns=[ 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat'],drop_first=False)
print(len(df.columns))
df.head()
X=df.drop(['class','class_code'],axis=1)
y=df['class_code']
n_components=[1,10,20,30,40,50,75,100]

for comp in n_components:
    pca_comp=PCA(n_components=comp)
    pca_comp.fit_transform(X)
    print(comp,sum(pca_comp.explained_variance_ratio_)*100)
pca = PCA(n_components=40)
pca_x=pd.DataFrame(pca.fit_transform(X))

sum(pca.explained_variance_ratio_)
pca_x.head()
X_train, X_test, y_train, y_test = train_test_split(pca_x,y,test_size=0.3,random_state=15)
LR_model= LogisticRegression()

LR_model.fit(X_train,y_train)

LR_y_pred = LR_model.predict(X_test)  

accuracy=accuracy_score(y_test, LR_y_pred)*100
print("Accuracy Score: ","{0:.2f}".format(accuracy))
sns.heatmap(pd.DataFrame(confusion_matrix(y_test, LR_y_pred)),annot=True,fmt="g", cmap='viridis')
GB_model= GaussianNB()

GB_model.fit(X_train,y_train)

GB_y_pred = GB_model.predict(X_test) 

accuracy=accuracy_score(y_test, GB_y_pred)*100
print("Accuracy Score: ","{0:.2f}".format(accuracy))
sns.heatmap(pd.DataFrame(confusion_matrix(y_test, GB_y_pred)),annot=True,fmt="g", cmap='viridis')
RF_model=RandomForestClassifier(n_estimators=10)

RF_model.fit(X_train,y_train)
RF_y_pred = RF_model.predict(X_test) 

accuracy=accuracy_score(y_test, RF_y_pred)*100
print("Accuracy Score: ","{0:.2f}".format(accuracy))
sns.heatmap(pd.DataFrame(confusion_matrix(y_test, RF_y_pred)),annot=True,fmt="g", cmap='viridis')
SVM_model=svm.LinearSVC()

SVM_model.fit(X_train,y_train)
SVM_y_pred = SVM_model.predict(X_test)   

accuracy=accuracy_score(y_test, SVM_y_pred)*100

print("Accuracy Score: ","{0:.2f}".format(accuracy))
sns.heatmap(pd.DataFrame(confusion_matrix(y_test, SVM_y_pred)),annot=True,fmt="g", cmap='viridis')
knn_model=KNeighborsClassifier()

knn_model.fit(X_train,y_train)

knn_y_pred = knn_model.predict(X_test)  

accuracy=accuracy_score(y_test, knn_y_pred)*100
print("Accuracy Score: ","{0:.2f}".format(accuracy))
sns.heatmap(pd.DataFrame(confusion_matrix(y_test, knn_y_pred)),annot=True,fmt="g", cmap='viridis')