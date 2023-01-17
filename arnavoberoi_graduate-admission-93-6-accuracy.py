import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import statsmodels.formula.api as sm

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score,recall_score,precision_score,mean_absolute_error,mean_squared_error,r2_score

from sklearn.preprocessing import StandardScaler,LabelEncoder, MinMaxScaler

warnings.filterwarnings('ignore')

sns.set_style('darkgrid')

%matplotlib inline



#Reading the CSV into a dataframe

df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
#Renaming the LOR and Chance of Admit column. 

df = df.rename({'LOR ':'LOR','Chance of Admit ':'Chance of Admit'}, axis=1)



#Removing Serial No.

df.drop('Serial No.',axis = 1,inplace = True)



#Checking the head of the dataset

df.head()
#Checking the info of the data

df.info()
#Minimum GRE Score

print('The minimum GRE Score is:',round(df['GRE Score'].min(),2))

#Maximum GRE Score

print('The maximum GRE Score is:',round(df['GRE Score'].max(),2))

#Average GRE Score

print('The average GRE Score is:',round(df['GRE Score'].mean(),2),'\n')



#Minimum TOEFL Score

print('The minimum TOEFL Score is:',round(df['TOEFL Score'].min(),2))

#Maximum TOEFL Score

print('The maximum TOEFL Score is:',round(df['TOEFL Score'].max(),2))

#Average TOEFL Score

print('The average TOEFL Score is:',round(df['TOEFL Score'].mean(),2),'\n')



#Minimum SOP

print('The minimum SOP is:',round(df['SOP'].min(),2))

#Maximum SOP

print('The maximum SOP is:',round(df['SOP'].max(),2))

#Average SOP

print('The average SOP is:',round(df['SOP'].mean(),2),'\n')



#Minimum LOR

print('The minimum LOR is:',round(df['LOR'].min(),2))

#Maximum LOR

print('The maximum LOR is:',round(df['LOR'].max(),2))

#Average LOR

print('The average LOR is:',round(df['LOR'].mean(),2),'\n')



#Minimum CGPA

print('The minimum CGPA is:',round(df['CGPA'].min(),2))

#Maximum CGPA

print('The maximum CGPA is:',round(df['CGPA'].max(),2))

#Average CGPA

print('The average CGPA is:',round(df['CGPA'].mean(),2),'\n')



#Minimum University Rating

print('The minimum University Rating is:',df['University Rating'].min())

#Maximum University Rating

print('The maximum University Rating is:',df['University Rating'].max())

#Average University Rating

print('The average University Rating is:',round(df['University Rating'].mean(),2),'\n')



#Minimum Chance of Admit

print('The minimum University Rating is:',df['Chance of Admit'].min())

#Maximum Chance of Admit

print('The maximum University Rating is:',df['Chance of Admit'].max())

#Average Chance of Admit

print('The average University Rating is:',round(df['Chance of Admit'].mean(),2))
fig,ax = plt.subplots(4,2,figsize=(18,20))



sns.distplot(df['GRE Score'], color = 'red',kde_kws={'color': 'white', 'lw': 3, 'label': 'KDE'}, ax = ax[0,0])

ax[0,0].set_title('Distribution of GRE Score', fontsize = 15)

ax[0,0].set_xlabel('GRE Score')

ax[0,0].set_ylabel('count')



sns.distplot(df['TOEFL Score'], color = 'purple',kde_kws={'color': 'orange', 'lw': 3, 'label': 'KDE'},ax = ax[0,1])

ax[0,1].set_title('Distribution of TOEFL Score', fontsize = 15)

ax[0,1].set_xlabel('TOEFL Score')

ax[0,1].set_ylabel('count')



sns.distplot(df['CGPA'], color = 'blue',kde_kws={'color': 'pink', 'lw': 3, 'label': 'KDE'},ax = ax[1,0])

ax[1,0].set_title('Distribution of CGPA', fontsize = 15)

ax[1,0].set_xlabel('CGPA')

ax[1,0].set_ylabel('count')



sns.countplot(df['University Rating'],palette = 'Set3',ax = ax[1,1])

ax[1,1].set_title('Distribution of University Rating', fontsize = 15)

ax[1,1].set_xlabel('University Rating')

ax[1,1].set_ylabel('count')



sns.countplot(df['SOP'],palette = 'Greys',ax = ax[2,0])

ax[2,0].set_title('Distribution of SOP', fontsize = 15)

ax[2,0].set_xlabel('SOP')

ax[2,0].set_ylabel('count')



sns.countplot(df['LOR'],palette = 'Blues',ax = ax[2,1])

ax[2,1].set_title('Distribution of LOR', fontsize = 15)

ax[2,1].set_xlabel('LOR')

ax[2,1].set_ylabel('count')



fig1 = sns.countplot(df['Research'],palette = 'rainbow',ax = ax[3,0])

fig1.set(xticklabels=["Research Not Done","Research Done"])

ax[3,0].set_title('Research done or not', fontsize = 15)

ax[3,0].set_xlabel('Research')

ax[3,0].set_ylabel('count')



sns.distplot(df['Chance of Admit'], color = 'orange',kde_kws={'color': 'black', 'lw': 3, 'label': 'KDE'}, ax = ax[3,1])

ax[3,1].set_title('Distribution of Chance of Admit', fontsize = 15)

ax[3,1].set_xlabel('Chance of Admit')

ax[3,1].set_ylabel('count')



plt.tight_layout();
plt.figure(figsize = (8,8))

sns.heatmap(df.corr(),annot = True,square = True)

plt.title('Correlation Heatmap');
fig,ax = plt.subplots(2,3,figsize=(15,10))



sns.scatterplot(x="GRE Score", y="Chance of Admit", data=df,color = 'red',ax = ax[0,0])

ax[0,0].set_title('Importance of GRE Score for admission', fontsize = 12)



sns.scatterplot(x="TOEFL Score", y="Chance of Admit",data=df,ax = ax[0,1])

ax[0,1].set_title('Importance of TOEFL Score for admission', fontsize = 12)



sns.scatterplot(x="SOP", y="Chance of Admit", data=df,ax = ax[0,2],color = 'pink')

ax[0,2].set_title('Importance of SOP for admission', fontsize = 12)



sns.scatterplot(x="LOR", y="Chance of Admit", data=df,color = 'purple',ax = ax[1,0])

ax[1,0].set_title('Importance of LOR for admission', fontsize = 12)



sns.scatterplot(x="CGPA", y="Chance of Admit",color = 'green' ,data=df,ax = ax[1,1])

ax[1,1].set_title('Importance of CGPA for admission', fontsize = 12)



sns.scatterplot(x="Research", y="Chance of Admit",color = 'orange',data=df,ax = ax[1,2])

ax[1,2].set_title('Importance of Research for admission', fontsize = 12)



plt.tight_layout();
X = df.drop('Chance of Admit',axis = 1)

y = df['Chance of Admit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
linereg = LinearRegression()

linereg.fit(X_train, y_train)
predictions = linereg.predict(X_test)
print('Mean Absolute Error:', mean_absolute_error(y_test, predictions))

print('Mean Squared Error:', mean_squared_error(y_test, predictions))

print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, predictions)))
print('R2 score for test data:',r2_score(y_test,predictions))
bins = (0.0, 0.75, 1.0)

group_names = ['bad', 'good']

df['Chance of Admit'] = pd.cut(df['Chance of Admit'], bins = bins, labels = group_names)

label_quality = LabelEncoder()

df['Chance of Admit'] = label_quality.fit_transform(df['Chance of Admit'])
x = df.drop('Chance of Admit',axis=1) 

y = df['Chance of Admit']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state = 1)
model = LogisticRegression()

model.fit(x_train,y_train)

preds = model.predict(x_test)
print(confusion_matrix(y_test, preds))
model.score(x_train,y_train)
print(accuracy_score(y_test,preds))
print(precision_score(y_test, preds))
print(f1_score(y_test, preds))
print(recall_score(y_test, preds))
mn = MinMaxScaler()

x_train = mn.fit_transform(x_train)

x_test = mn.fit_transform(x_test)

model = LogisticRegression()

model.fit(x_train,y_train)

preds = model.predict(x_test)
model.score(x_train,y_train)
print(confusion_matrix(y_test, preds))
print(accuracy_score(y_test,preds))
print(precision_score(y_test, preds))
print(f1_score(y_test, preds))
print(recall_score(y_test, preds))
params_dict={'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],'penalty':['l1','l2']}

cv = GridSearchCV(LogisticRegression(),param_grid=params_dict,scoring='accuracy',cv=10)
cv.fit(x_train,y_train)

preds=cv.predict(x_test)

cv.score(x_train,y_train)
print(confusion_matrix(y_test, preds))
accuracy_score(preds,y_test)
print(precision_score(y_test, preds))
print(f1_score(y_test, preds))
print(recall_score(y_test, preds))
cv.best_estimator_
cv.best_params_