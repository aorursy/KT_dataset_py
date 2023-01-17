#Loading The Libraries

#For uploading and accessing the data
import pandas as pd
import numpy as np

#For visualizations
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.head()
df.shape
df.rename(columns ={'age':'Age','sex':'Sex','cp':'Chest_pain','trestbps':'Resting_blood_pressure','chol':'Cholesterol','fbs':'Fasting_blood_sugar',
                    'restecg':'ECG_results','thalach':'Maximum_heart_rate','exang':'Exercise_induced_angina','oldpeak':'ST_depression','ca':'Major_vessels',
                   'thal':'Thalassemia_types','target':'Heart_attack','slope':'ST_slope'}, inplace = True)
df2 = df.copy()
df1 = df.copy()
df.head()
#1 = Male and 0 = Female in 'Sex' column.
df.isnull().sum()
df1['Sex'].replace({1:'Male',0:'Female'},inplace = True)
df1['Heart_attack'].replace({1:'Heart_attack - Yes',0:'Heart_attack - No'},inplace = True)
s= df1.groupby(['Sex','Age'])['Heart_attack'].count().reset_index().sort_values(by='Heart_attack',ascending=False)
s.head(20).style.background_gradient(cmap='Purples')
s= df1.groupby(['Sex','Chest_pain'])['Heart_attack'].count().reset_index().sort_values(by='Heart_attack',ascending=False)
s.head(10).style.background_gradient(cmap='Blues')
s= df1.groupby(['Sex','Resting_blood_pressure'])['Heart_attack'].count().reset_index().sort_values(by='Heart_attack',ascending=False)
s.head(10).style.background_gradient(cmap='Greens')
s= df1.groupby(['Sex','Cholesterol'])['Heart_attack'].count().reset_index().sort_values(by='Heart_attack',ascending=False)
s.head(10).style.background_gradient(cmap='Reds')
s= df1.groupby(['Sex','Fasting_blood_sugar'])['Heart_attack'].count().reset_index().sort_values(by='Heart_attack',ascending=False)
s.head(10).style.background_gradient(cmap='Reds')
s= df1.groupby(['Sex','ECG_results'])['Heart_attack'].count().reset_index().sort_values(by='Heart_attack',ascending=False)
s.head(10).style.background_gradient(cmap='Greys')
s= df1.groupby(['Sex','Maximum_heart_rate'])['Heart_attack'].count().reset_index().sort_values(by='Heart_attack',ascending=False)
s.head(10).style.background_gradient(cmap='Oranges')
s= df1.groupby(['Sex','Exercise_induced_angina'])['Heart_attack'].count().reset_index().sort_values(by='Heart_attack',ascending=False)
s.head(10).style.background_gradient(cmap='Purples')
s= df1.groupby(['Sex','ST_depression'])['Heart_attack'].count().reset_index().sort_values(by='Heart_attack',ascending=False)
s.head(10).style.background_gradient(cmap='Blues')
s= df1.groupby(['Sex','ST_slope'])['Heart_attack'].count().reset_index().sort_values(by='Heart_attack',ascending=False)
s.head(10).style.background_gradient(cmap='Greens')
s= df1.groupby(['Sex','Major_vessels'])['Heart_attack'].count().reset_index().sort_values(by='Heart_attack',ascending=False)
s.head(10).style.background_gradient(cmap='Reds')
s= df1.groupby(['Sex','Thalassemia_types'])['Heart_attack'].count().reset_index().sort_values(by='Heart_attack',ascending=False)
s.head(10).style.background_gradient(cmap='Greys')
df2.drop(['Sex','Fasting_blood_sugar','Heart_attack','Chest_pain','ECG_results','Exercise_induced_angina','ST_slope','ST_depression','Major_vessels','Thalassemia_types'],axis = 'columns',inplace = True)
plt.figure(figsize=(10,5))
sns.heatmap(df2.corr(), annot=True, linewidth=0.5, cmap='coolwarm')
sns.set(rc={'figure.figsize':(20,5)})
df['Age'].plot.hist(bins = 15, color = 'skyblue')
df['Resting_blood_pressure'].plot.hist(bins = 15, color = 'green')
df['Cholesterol'].plot.hist(bins = 10, color = 'lightgrey')
df['Maximum_heart_rate'].plot.hist(bins = 20, color = 'lightcoral')
sns.countplot(x = 'Age',hue = 'Chest_pain', data = df, color = 'green')
sns.countplot(x = 'Age',hue = 'Fasting_blood_sugar', data = df,palette="Set3")
sns.countplot(x = 'Age',hue = 'ECG_results', data = df,palette="Set2")
sns.countplot(x = 'Age',hue = 'Exercise_induced_angina', data = df,palette="Set1")
sns.countplot(x = 'Age',hue = 'ST_slope', data = df, color = "black")
sns.countplot(x = 'Age',hue = 'Major_vessels', data = df, palette='Set1')
sns.countplot(x = 'Age',hue = 'Thalassemia_types', data = df)
sns.countplot(x = 'Age',hue = 'Heart_attack', data = df1, palette = 'Set2')
sns.relplot(x ='Age', y ='Chest_pain', col = 'Sex', data = df1, color = 'red', height = 5)
sns.relplot(x ='Age', y ='Resting_blood_pressure', col = 'Sex', data = df1, color = 'green')
sns.relplot(x ='Age', y ='Cholesterol', col = 'Sex', data = df1, color = 'black')
sns.relplot(x ='Age', y ='Fasting_blood_sugar', col = 'Sex', data = df1, color = 'grey')
sns.relplot(x ='Age', y ='ECG_results', col = 'Sex', data = df1, color = 'lightcoral')
sns.relplot(x ='Age', y ='Maximum_heart_rate', col = 'Sex', data = df1, color = 'turquoise')
sns.jointplot(x =df['Age'], y =df1['Exercise_induced_angina'], data = df1, color = 'green')
sns.relplot(x ='Age', y ='ST_depression', col = 'Sex', data = df1, color = 'crimson')
sns.relplot(x ='Age', y ='Major_vessels', col = 'Sex', data = df1, color = 'orange')
sns.relplot(x ='Age', y ='Thalassemia_types', col = 'Sex', data = df1, color = 'teal')
sns.catplot(x ='Age', y ='Heart_attack', col = 'Sex', data = df1, color = 'crimson', kind = 'box')
sns.relplot(x = 'Age', y = 'Resting_blood_pressure', kind = 'line', data=df,aspect = 1,height = 7, color = 'green')
sns.relplot(x = 'Age', y = 'Cholesterol', kind = 'line', data=df,aspect = 1,height = 7, color = 'orange')
sns.relplot(x = 'Age', y = 'Maximum_heart_rate', kind = 'line', data=df,aspect = 1,height = 7, color = 'teal')
sns.set(style="ticks", color_codes=True)
sns.pairplot(df)
#Libraries for model selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

#Libraries for various model parameter selection.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from sklearn.metrics import accuracy_score,confusion_matrix
import scikitplot as skplt
from sklearn import metrics
dummy1 = pd.get_dummies(df.Chest_pain)
dummy2 = pd.get_dummies(df.Thalassemia_types)
dummy3 = pd.get_dummies(df.ECG_results)
dummy4 = pd.get_dummies(df.ST_slope)
dummy5 = pd.get_dummies(df.Major_vessels)
merge = pd.concat([df,dummy1,dummy2,dummy3,dummy4,dummy5],axis = 'columns')
final = merge.drop(['Chest_pain','Thalassemia_types','ECG_results','ST_slope','Major_vessels'],axis = 1)
final.head()
x = final.drop(['Heart_attack'], axis = 1)
y = final['Heart_attack']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20, random_state = 5)
feature_scaler = MinMaxScaler()
x_train = feature_scaler.fit_transform(x_train)
x_test = feature_scaler.transform(x_test)
accuracy = []
C = [0.01,0.1, 1, 5, 10]

Log = LogisticRegression()

parameters = {'C': [.1 ,2, 5, 10, 15, 20]}

log_regressor = GridSearchCV(Log, parameters, scoring='neg_mean_squared_error' ,cv =5)
log_regressor.fit(x_train, y_train)
log_regressor.best_params_
model1 = LogisticRegression(C=0.1)
model1.fit(x_train,y_train)
accuracy1 = model1.score(x_test,y_test)
accuracy.append(accuracy1)
print('Logistic Regression Accuracy -->',((accuracy1)*100))
pred1 = model1.predict(x_test)
matrix1 = (y_test,pred1)
skplt.metrics.plot_confusion_matrix(y_test, pred1,figsize=(10,5))
criterion = ['gini','entropy']
splitter = ['best','random']

Tree = DecisionTreeClassifier()

parameters = {'criterion': ['gini','entropy']}

tree_classifier = GridSearchCV(Tree, parameters, scoring='neg_mean_squared_error' ,cv =5)
tree_classifier.fit(x_train, y_train)
tree_classifier.best_params_
model2 = DecisionTreeClassifier(criterion = 'gini')
model2.fit(x_train,y_train)
accuracy2 = model2.score(x_test,y_test)
accuracy.append(accuracy2)
print('Decision Tree Accuracy -->',((accuracy2)*100))
pred2 = model2.predict(x_test)
matrix2 = (y_test,pred2)
skplt.metrics.plot_confusion_matrix(y_test ,pred2 ,figsize=(10,5))
penalty = ['l1','l2']
C = [0.01,0.1,1,5,10,15,20]
loss = ['hinge','squared_hinge']

SVM = LinearSVC()

parameters = {'penalty':['l1','l2'],'C': [.01,.1,1,5,10,15,20],'loss':['hinge','squared_hinge']}

SVM_classifier = GridSearchCV(SVM, parameters, scoring='neg_mean_squared_error' ,cv =5)
SVM_classifier.fit(x_train, y_train)
SVM_classifier.best_params_
model3 = LinearSVC(C = 15,loss = 'hinge',penalty = 'l2')
model3.fit(x_train,y_train)
accuracy3 = model3.score(x_test,y_test)
accuracy.append(accuracy3)
print('SVM Classifier Accuracy -->',((accuracy3)*100))
pred3 = model3.predict(x_test)
matrix3 = (y_test,pred3)
skplt.metrics.plot_confusion_matrix(y_test ,pred3 ,figsize=(10,5))
model4 = GaussianNB()

model4.fit(x_train, y_train)
accuracy4 = model4.score(x_test,y_test)
accuracy.append(accuracy4)
print('Gaussian NB Accuracy -->',((accuracy4)*100))
pred4 = model4.predict(x_test)
matrix4 = (y_test,pred4)
skplt.metrics.plot_confusion_matrix(y_test ,pred4 ,figsize=(10,5))
model5 = MultinomialNB()

model5.fit(x_train, y_train)
accuracy5 = model5.score(x_test,y_test)
accuracy.append(accuracy5)
print('Multinomial NB Accuracy -->',((accuracy5)*100))
pred5 = model5.predict(x_test)
matrix5 = (y_test,pred5)
skplt.metrics.plot_confusion_matrix(y_test ,pred5 ,figsize=(10,5))
n_estimators = [250,500,750,1000]
criterion = ['gini','entropy']
max_features = ['auto','sqrt','log2']
random_state = [5]

RF = RandomForestClassifier()

parameters = {'n_estimators': [250,500,750,1000],'criterion': ['gini','entropy'],'max_features':['auto','sqrt','log2']}

RFClassifier = GridSearchCV(RF, parameters, scoring='neg_mean_squared_error' ,cv =5)
RFClassifier.fit(x_train, y_train)
RFClassifier.best_params_
model6 = RandomForestClassifier(criterion = 'entropy',max_features = 'log2',n_estimators = 250, random_state = 5)
model6.fit(x_train,y_train)
accuracy6 = model6.score(x_test,y_test)
accuracy.append(accuracy6)
print('Random Forest Classifier Accuracy -->',((accuracy6)*100))
pred6 = model6.predict(x_test)
matrix6 = (y_test,pred6)
skplt.metrics.plot_confusion_matrix(y_test ,pred6 ,figsize=(10,5))
n_estimators = [250,500,750,1000]
loss = ['deviance','exponential']
max_features = ['auto','sqrt','log2']

GB = GradientBoostingClassifier()

parameters = {'n_estimators': [250,500,750,1000],'loss': ['deviance','exponential'],'max_features':['auto','sqrt','log2']}

GBClassifier = GridSearchCV(GB, parameters, scoring='neg_mean_squared_error' ,cv =5)
GBClassifier.fit(x_train, y_train)
GBClassifier.best_params_
model7 = GradientBoostingClassifier(loss = 'deviance',max_features = 'log2',n_estimators = 500, random_state = 5)
model7.fit(x_train,y_train)
accuracy7 = model7.score(x_test,y_test)
accuracy.append(accuracy7)
print('Gradient Boosting Classifier Accuracy -->',((accuracy7)*100))
pred7 = model7.predict(x_test)
matrix7 = (y_test,pred7)
skplt.metrics.plot_confusion_matrix(y_test ,pred7 ,figsize=(10,5))
Krange = range(1,20)
scores = {}
scores_list = []
for k in Krange:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    scores[k] = metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))
    
plt.plot(Krange,scores_list)
plt.xlabel("Value of K")
plt.ylabel("Accuracy")
model8 = KNeighborsClassifier(n_neighbors = 5)
model8.fit(x_train,y_train)
accuracy8 = model8.score(x_test,y_test)
accuracy.append(accuracy8)
print('Gradient Boosting Classifier Accuracy -->',((accuracy8)*100))
pred8 = model8.predict(x_test)
matrix8 = (y_test,pred8)
skplt.metrics.plot_confusion_matrix(y_test ,pred8 ,figsize=(10,5))
Models = ['Logistic Regression','Decision Tree','SVM Classifier','Gaussian NB','Multinomial NB','Random Forest Classifier','Gradient Boost Classifier','K-Nearest Neighbors']
total = list(zip(Models,accuracy))
output = pd.DataFrame(total, columns = ['Models','Accuracy'])
s = output.groupby(['Models'])['Accuracy'].mean().reset_index().sort_values(by='Accuracy',ascending=False)
s.head(10).style.background_gradient(cmap='Reds')
sns.lineplot(x = 'Models',y = 'Accuracy',data = output, color = 'Green' )
