#importing  python  libraries

import pandas as pd

import numpy as np

import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt

%matplotlib inline

import os



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

#Import data set

df = pd.read_csv('../input/heart.csv')

df.head(5)# this command shows first 5 rows of data frame
df.info()  
df['target'].value_counts()
#checking for Missing Data 



missing_data=df.isnull()

for column in missing_data.columns.values.tolist():

    print(column)

    print (missing_data[column].value_counts())

    print("")    
#Correct data format



df.dtypes

#Conclusion:All the dtypes are correct format
%%capture

! pip install seaborn
#correlation of independent variable and dependent variable 

df.corr()
df[['cp','target']].corr()
df.describe() # this will shows the descriptive statistics of data frame 
#Heat map

plt.figure(figsize=(15,7))

corr = df.corr()

sns.heatmap(corr, annot=True )
sns.distplot(df['age'],color='Red',hist_kws={'alpha':1,"linewidth": 2}, kde_kws={"color": "k", "lw": 3, "label": "KDE"})

#Most people age is from 40 to 60
fig,ax=plt.subplots(figsize=(16,6))

sns.pointplot(x='age',y='cp',data=df,color='Lime',hue='target',linestyles=["-", "--"])

plt.title('Age vs Cp')

#People with heart disease tend to have higher 'cphest pain' at all ages only exceptions at age 45 and 49
sns.countplot(x='ca',data=df,hue='target',palette='YlOrRd',linewidth=3)

# People with 'ca' as 0 have highest chance of heart disease


pearson_coef, p_value = stats.pearsonr(df['age'], df['target'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(df['sex'], df['target'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(df['cp'], df['target'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(df['trestbps'], df['target'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(df['chol'], df['target'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(df['fbs'], df['target'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(df['restecg'], df['target'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(df['thalach'], df['target'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(df['exang'], df['target'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(df['oldpeak'], df['target'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(df['slope'], df['target'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(df['ca'], df['target'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(df['thal'], df['target'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


x=df.drop('target',axis=1)

x.head()

y=df['target']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state =0)
#First we import the library of Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score





lr=LogisticRegression() #create an object



# now we train the model using fit

lr.fit(x_train,y_train) 



# in this line model makes predictions

predictions=lr.predict(x_test)



#we find the accuracy of model and saved in accuracy varaible

accuracy=accuracy_score(y_test,predictions)





print(predictions)

print(f"LogisticRegression  Accuracy Score is  {accuracy}")
#First we import the library of Support vector machine



from sklearn.svm import SVC



svc=SVC()



# now we train the model using fit

svc.fit(x_train,y_train) 



# in this line model is tested

sv_predictions=svc.predict(x_test)



#we find the accuracy of model and saved in sv_accuracy varaible

sv_accuracy=accuracy_score(y_test,sv_predictions)





print(sv_predictions)

print(f"Support vectoe Machine   Accuracy Score is  {sv_accuracy}")

from sklearn.naive_bayes import GaussianNB



nb=GaussianNB()



# now we train the model using fit

nb.fit(x_train,y_train) 



# in this line model is tested

nb_predictions=nb.predict(x_test)



#we find the accuracy of model and saved in nb_accuracy varaible

nb_accuracy=accuracy_score(y_test,nb_predictions)





print(nb_predictions)

print(f" Gaussian Naive Bayes Algorithm Accuracy Score is  {nb_accuracy}")

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=50)



# now we train the model using fit

rf.fit(x_train,y_train) 



# in this line model is tested

rf_predictions=rf.predict(x_test)



#we find the accuracy of model and saved in rf_accuracy varaible

rf_accuracy=accuracy_score(y_test,rf_predictions)





print(rf_predictions)

print(f" RAndom Forest Algorithm Accuracy Score is  {rf_accuracy}")

from sklearn.neighbors import KNeighborsClassifier



knn=KNeighborsClassifier(3)



# now we train the model using fit

knn.fit(x_train,y_train) 



# in thisn line model is tested

knn_predictions=knn.predict(x_test)



#we find the accuracy of model and saved in sv_accuracy varaible

knn_accuracy=accuracy_score(y_test,knn_predictions)





print(knn_predictions)

print(f" K Nearest  neighbour Cklassification Accuracy Score is  {knn_accuracy}")

from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(), x, y)
cross_val_score(SVC(), x, y)
cross_val_score(GaussianNB(),x,y)
cross_val_score(KNeighborsClassifier(),x,y)
cross_val_score(RandomForestClassifier(n_estimators=30),x,y)
scores1 = cross_val_score(RandomForestClassifier(n_estimators=5),x, y, cv=10) 

np.average(scores1)
scores2 = cross_val_score(RandomForestClassifier(n_estimators=20),x, y, cv=10)

np.average(scores2)
scores3 = cross_val_score(RandomForestClassifier(n_estimators=28),x, y, cv=10)

np.average(scores3)
scores4 = cross_val_score(RandomForestClassifier(n_estimators=50),x, y, cv=10)

np.average(scores4)