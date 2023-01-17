import numpy as np # Numerical Python
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline
from plotly.offline import iplot
import plotly.offline as py
py.init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore') 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
sns.set()
dataset=pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
dataset.head()
dataset.info()
dataset.describe().T
plt.figure(figsize=(15,9))
sns.distplot(dataset['fixed acidity'])
plt.show()
plt.figure(figsize=(15,9))
sns.distplot(dataset['volatile acidity'])
plt.show()
plt.figure(figsize=(15,9))
sns.distplot(dataset['citric acid'])
plt.show()
plt.figure(figsize=(15,9))
sns.distplot(dataset['residual sugar'])
plt.show()
plt.figure(figsize=(15,9))
sns.distplot(dataset['chlorides'])
plt.show()
plt.figure(figsize=(15,9))
sns.distplot(dataset['free sulfur dioxide'])
plt.show()
plt.figure(figsize=(15,9))
sns.distplot(dataset['total sulfur dioxide'])
plt.show()
plt.figure(figsize=(15,9))
sns.distplot(dataset['density'])
plt.show()
plt.figure(figsize=(15,9))
sns.distplot(dataset['pH'])
plt.show()
plt.figure(figsize=(15,9))
sns.distplot(dataset['sulphates'])
plt.show()
plt.figure(figsize=(15,9))
sns.distplot(dataset['alcohol'])
plt.show()
plt.figure(figsize=(15,9))
sns.countplot(dataset['quality'])
plt.show()
#Correlation Heatmap
corelation_matrix=dataset.corr()
fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corelation_matrix, annot=True, linewidths=0.5,linecolor="red", fmt= '.2f',ax=ax,cmap='inferno')
plt.show()
plt.figure(figsize=(15,9))
sns.pairplot(dataset,hue="quality",palette=sns.color_palette("RdBu_r", 7))
plt.legend()
plt.show()
dataset.info()
plt.figure(figsize=(15,9))
sns.catplot(x="quality", y="fixed acidity", data=dataset,kind='violin')
plt.show()
plt.figure(figsize=(15,9))
sns.catplot(x="quality", y="volatile acidity", data=dataset,kind='violin')
plt.show()
plt.figure(figsize=(15,9))
sns.catplot(x="quality", y="citric acid", data=dataset,kind='violin')
plt.show()
plt.figure(figsize=(10,10))
sns.catplot(x="quality", y="residual sugar", data=dataset,kind='violin')
plt.show()
plt.figure(figsize=(10,10))
sns.catplot(x="quality", y="chlorides", data=dataset,kind='violin')
plt.show()
plt.figure(figsize=(10,10))
sns.catplot(x="quality", y="free sulfur dioxide", data=dataset,kind='violin')
plt.show()
plt.figure(figsize=(10,10))
sns.catplot(x="quality", y="total sulfur dioxide", data=dataset,kind='violin')
plt.show()
plt.figure(figsize=(10,10))
sns.catplot(x="quality", y="density", data=dataset,kind='violin')
plt.show()
plt.figure(figsize=(10,10))
sns.catplot(x="quality", y="pH", data=dataset,kind='violin')
plt.show()
plt.figure(figsize=(10,10))
sns.catplot(x="quality", y="sulphates", data=dataset,kind='violin')
plt.show()
plt.figure(figsize=(10,10))
sns.catplot(x="quality", y="alcohol", data=dataset,kind='violin')
plt.show()
#Normalizing the data
normalized_data=dataset.copy()
for column in normalized_data.columns:
    normalized_data[column]=normalized_data[column]/normalized_data[column].max()
    
normalized_data=normalized_data.round(3)
normalized_data.head()
fig,ax1 = plt.subplots(figsize =(15,9))
sns.pointplot(x=normalized_data['volatile acidity'],y=normalized_data['quality'],data=normalized_data,color='sandybrown',alpha=0.7)
sns.pointplot(x=normalized_data['citric acid'],y=normalized_data['quality'],data=normalized_data,color='seagreen',alpha=0.6)
sns.pointplot(x=normalized_data['alcohol'],y=normalized_data['quality'],data=normalized_data,color='red',alpha=0.6)
plt.xticks(rotation=90)
plt.text(5.5,1,'Volatile Acidity-Quality',color='sandybrown',fontsize = 18,style = 'italic')
plt.text(5.4,0.96,'Citric Acid-Quality',color='seagreen',fontsize = 18,style = 'italic')
plt.text(5.3,0.92,'Alcohol-Quality',color='red',fontsize = 18,style = 'italic')
plt.xlabel('X - Axis',fontsize = 15,color='black')
plt.ylabel('Y - Axis',fontsize = 15,color='black')
plt.title('Volatile Acidity-Quality vs Citric Acid-Quality vs Alcohol-Quality',fontsize = 20,color='blue')
plt.grid()
#In this part i changed dependent variables as 1,2 and 3 to get better results
a=0
for i in dataset['quality'].values:
    if i==8 or i==7:
        dataset['quality'][a]=3
    elif i==6 or i==5:
        dataset['quality'][a]=2
    elif i==4 or i==3:
        dataset['quality'][a]=1
    a=a+1
dataset['quality'].value_counts()
import plotly.express as px
fig = px.scatter_3d(dataset, x='alcohol',
                    y='volatile acidity', 
                    z='sulphates', 
                   color='quality', 
       color_continuous_scale='solar'
       )
iplot(fig)
X = dataset.iloc[:,0:-1].copy()
Y = dataset.iloc[:,-1].copy()
Y=Y.values
import statsmodels.api as sm
x=sm.add_constant(X)
y=Y.copy()
results=sm.OLS(Y,x).fit()
print(results.summary())
X.drop(['density','fixed acidity'],axis=1,inplace=True) #These features have big noise. So i drop them.
x=sm.add_constant(X) 
y=Y.copy()
results=sm.OLS(Y,x).fit()
print(results.summary())
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)
from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA,KernelPCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from xgboost import XGBClassifier
models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('K-NN', KNeighborsClassifier()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Decision Tree (CART)',DecisionTreeClassifier())) 
models.append(('AdaBoostClassifier', AdaBoostClassifier()))
models.append(('SVC', SVC(kernel = 'rbf', random_state = 42)))
models.append(('BaggingClassifier', BaggingClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier())) 
models.append(('XGBoost', XGBClassifier(n_estimators=200)))
from sklearn.metrics import classification_report
np.random.seed(123) #To get the same results

for name, model in models:
    model = model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    from sklearn import metrics
    print(name," --> Accuracy: ",(metrics.accuracy_score(Y_test, Y_pred)*100),"%")
    report = classification_report(Y_test, Y_pred)
    print(report)
from sklearn.model_selection import cross_val_score
Bagging_Classifier=BaggingClassifier()
Bagging_Accuracies=cross_val_score(estimator=Bagging_Classifier,X=X,y=Y,cv=10,n_jobs=-1)

RandomForest_Classifier=RandomForestClassifier()
RandomForest_Accuricies=cross_val_score(estimator=RandomForest_Classifier,X=X,y=Y,cv=10,n_jobs=-1)

final_results=pd.DataFrame(index=['Results'],columns=['Bagging Classifier','Random Forest Classifier'],data=[[Bagging_Accuracies.mean(),RandomForest_Accuricies.mean()]])
final_results