import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(color_codes=True)
import plotly.express as px
import plotly.io as pio
pio.renderers.default='notebook'
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/heart-disease-uci/heart.csv')
df.head()
df.columns.values
df.target.value_counts()
df.shape
df.info()
df.describe().T
corr_matrix = df.corr()
corr_matrix['target'].sort_values(ascending=False)
sns.countplot(data=df,x='target');
plt.pie(x=df.target.value_counts(),labels=['Disease','No Disease'],autopct='%1.1f%%',shadow=True);
plt.legend(loc='best');
sns.countplot(data=df,x='target',hue='sex');
plt.title('Heart Disease Frequency for Sex');
plt.xlabel('Sex (0 = Female, 1 = Male)');
plt.xticks(rotation=0);
plt.legend(["Haven't Disease", "Have Disease"]);
plt.ylabel('Frequency');
df.isnull().sum()
sns.distplot(df.age,kde=False);
df['age'].value_counts()
pd.crosstab(df.age,df.target).plot(kind='bar',figsize=(20,6));
plt.title('Heart Disease Frequency for Ages');
plt.xlabel('Age');
plt.ylabel('Frequency');
plt.legend(["Haven't Disease", "Have Disease"]);
plt.figure(figsize=(20,6))
sns.barplot(x='cp',y='age',data=df,hue='target');
plt.legend(loc='upper right');
px.scatter(data_frame=df,x='age',y='thalach',title='Distribution of Max Heart Rate over Age',color='target')
pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['#11A5AA','#AA1190' ])
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease or Not')
plt.show()
data = df[['trestbps','chol','thalach']]
px.box(data_frame=data)
px.scatter(data_frame=df,x='age',y='chol')
plt.figure(figsize=(10,7))
sns.boxplot(x='target',y='age',data=df)
sns.swarmplot(x='target',y='age',data=df,palette="Pastel1")
sns.pairplot(data=df)
from sklearn.ensemble import ExtraTreesRegressor
X = df.drop('target',axis=1)
Y = df['target']
model = ExtraTreesRegressor()
model.fit(X,Y)
print(model.feature_importances_)
# plot graph of feature importance for better visualization
feat_import = pd.Series(model.feature_importances_,index = X.columns)
feat_import.nlargest(8).plot(kind='barh')
plt.show()
X = df.drop('target',axis=1).values
Y = df['target'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state=42)
score = []
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
lr = LogisticRegression()
lr.fit(X_train,y_train)
s1 = np.mean(cross_val_score(lr,X_train,y_train,scoring='accuracy',cv=10))
score.append(s1*100)
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
s2 = np.mean(cross_val_score(dt,X_train,y_train,scoring='accuracy',cv=10))
score.append(s2*100)
rf = RandomForestClassifier(n_estimators=300)
rf.fit(X_train,y_train)
s3 = np.mean(cross_val_score(rf,X_train,y_train,scoring='accuracy',cv=10))
score.append(s3*100)
svc = SVC()
svc.fit(X_train,y_train)
s4 = np.mean(cross_val_score(svc,X_train,y_train,scoring='accuracy',cv=10))
score.append(s4*100)
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
s5 = np.mean(cross_val_score(knn,X_train,y_train,scoring='accuracy',cv=10))
score.append(s5*100)
nb = GaussianNB()
nb.fit(X_train,y_train)
s6 = np.mean(cross_val_score(nb,X_train,y_train,scoring='accuracy',cv=10))
score.append(s6*100)
models = ['LogisticRegression','DecisionTreeClassifier','RandomForestClassifier','SVC','KNeighborsClassifier','GaussianNB']
for i in range(len(models)):
    print('The Accuracy Score for',models[i],'is',score[i])
from sklearn.model_selection import RandomizedSearchCV
parameters = {
    'n_estimators':range(10,500,10),
    'criterion': ('gini','entropy'),
    'max_features':('auto','sqrt','log2'),
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4],
    'bootstrap': [True,False]
}
grid = RandomizedSearchCV(rf,param_distributions=parameters,scoring='accuracy',cv=5,verbose=0,n_iter=50,random_state=42,n_jobs=1)
grid.fit(X_train,y_train)
grid.best_params_
grid.best_score_
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
model = Sequential()
model.add(Dense(units=32,kernel_initializer='uniform',activation='relu',input_dim=13))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=300,batch_size=10)
metrics = pd.DataFrame(model.history.history)
model.evaluate(X_test,y_test)
metrics[['loss','val_loss']].plot()
metrics[['accuracy','val_accuracy']].plot()
pred = model.predict_classes(X_test)
pred
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm = confusion_matrix(y_test,pred)
cm
accuracy_score(y_test,pred)
print(classification_report(y_test,pred))
