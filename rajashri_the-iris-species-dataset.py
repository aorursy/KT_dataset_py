
# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
%matplotlib inline  

#scikit-learn.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler,LabelEncoder 
train=pd.read_csv(r'../input/Iris.csv')
df=train.copy()
df.head(10)
df.shape
df.columns # names of all coumns.
df.drop(['Id'],axis=1,inplace=True)
df.index # indices of rows.
df.isnull().any()
msno.matrix(df) # just one final time to visualize.
for col in df.columns:
    print("Number of values in column " ,col," : ",df[col].count())
df.describe()
def plot(feature):
    fig,axes=plt.subplots(1,2)
    sns.boxplot(data=df,x=feature,ax=axes[0])
    sns.distplot(a=df[feature],ax=axes[1],color='#ff4125')
    fig.set_size_inches(15,5)
plot('SepalLengthCm')
plot('SepalWidthCm')
plot('PetalLengthCm')
plot('PetalWidthCm')
sns.factorplot(data=df,x='Species',kind='count')
g = sns.PairGrid(df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',
       'Species']], hue = "Species")
g = g.map(plt.scatter).add_legend()
le=LabelEncoder()
df['Species']=le.fit_transform(df['Species'])
scaler=StandardScaler()
scaled_df=scaler.fit_transform(df.drop('Species',axis=1))
X=scaled_df
Y=df['Species'].as_matrix()
df.head(10)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42)
clf_lr=LogisticRegression(C=10)
clf_lr.fit(x_train,y_train)
pred=clf_lr.predict(x_test)
print(accuracy_score(pred,y_test))
clf_knn=KNeighborsClassifier()
clf_knn.fit(x_train,y_train)
pred=clf_knn.predict(x_test)
print(accuracy_score(pred,y_test))
clf_svm_lin=LinearSVC()
clf_svm_lin.fit(x_train,y_train)
pred=clf_svm_lin.predict(x_test)
print(accuracy_score(pred,y_test))
clf_svm=SVC()
clf_svm.fit(x_train,y_train)
pred=clf_svm.predict(x_test)
print(accuracy_score(pred,y_test))
models=[LogisticRegression(),LinearSVC(),SVC(),KNeighborsClassifier()]
model_names=['LogisticRegression','LinearSVM','rbfSVM','KNearestNeighbors']

acc=[]
d={}

for model in range(len(models)):
    clf=models[model]
    clf.fit(x_train,y_train)
    pred=clf.predict(x_test)
    acc.append(accuracy_score(pred,y_test))
     
d={'Modelling Algo':model_names,'Accuracy':acc}
acc_frame=pd.DataFrame(d)
acc_frame
sns.factorplot(data=acc_frame,y='Modelling Algo',x='Accuracy',kind='bar',size=5,aspect=1.5)
