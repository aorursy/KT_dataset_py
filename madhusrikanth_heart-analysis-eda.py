import numpy as np 

import pandas as pd 
import matplotlib.pyplot as plt #importing data visualization libraries

import seaborn as sns
df_heart = pd.read_csv("../input/heart_dataset.csv") #to find any other trends in heart data to predict certain cardiovascular events or find any clear indications of heart health

df_heart #dataset is imported,Data acquisition step
import cufflinks as cf

cf.go_offline()       #It is an interactive plot and cufflinks is used to link to pandas

#information about attributes



# 1.age

# 2. sex

# 3. chest pain type (4 values)

# 4. resting blood pressure

# 5. cholestoral in mg/dl

# 6. fasting blood sugar > 120 mg/dl

# 7. resting electrocardiographic results (values 0,1,2)

# 8. maximum heart rate achieved

# 9. exercise induced angina

# 10. oldpeak = ST depression induced by exercise relative to rest

# 11. the slope of the peak exercise ST segment

# 12. number of major vessels (0-3) colored by flourosopy

# 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
df_heart.info() #to see columns and total value present, total value = 303, total columns = 14
df_heart.describe() #to get mean,standard  deviation etc...
#to check whether dataset is empty or not

sns.heatmap(df_heart.isnull(),yticklabels=False,cbar=False,cmap='viridis') #this daatset is not empty
df_heart['target'].iplot(kind='hist',color='purple')

#presence of heart disease is greater than absence 

#This is an interactive plot using plotly
df_heart['target'].plot.hist()

#frequency of presence is greater than absence
#to check count of gender (where 1=male ,0=female is given) using seaborn count plot library

sns.countplot(df_heart['sex']) #as we can see count of females are greater than count of males

#male count approx from visualization will be 98

#femlae count approm from visualization will be 200
#to check count of presence or absence of heart disease which is (0=absensce,1=presence is given)

sns.countplot(x='target',data=df_heart)

#as we can predict from plot count of presence of disease is more than absence 
sns.set_style('whitegrid') #styling using sns

sns.countplot(x='target',hue='sex',data=df_heart,palette='RdBu_r') #palette is also styling parameter

#Insights from the graph are:

#females are highly suffering from heart disease as compared to males



#to compare at what age heart disease as occured the most

sns.barplot(x='target',y='age',data=df_heart)#bar plot is used for visualization

#heart disease predicted to occur for females between age 55-60

#heart disease predicted to occur for males is between 45-50
#to analyze the age in the given dataset

sns.jointplot(x='sex',y='age',data=df_heart,kind='hex') #insights obtained from visualization

#female age is between 30-80 approx

#male age is between 30-75 approx
#to check for highest type of chect pest(as values given are 1,2,3,4) between the two genders

sns.boxplot(x="sex", y="chol", hue="cp",data=df_heart, palette="coolwarm") #insights obtained from visualization

#High cholesterol can cause a dangerous accumulation of cholesterol and other deposits on the walls of your arteries (atherosclerosis). These deposits (plaques) can reduce blood flow through your arteries, which can cause complications, such as: Chest pain.

#For females = chest pain type 0(with highest count of chol),chest pain type 3(with least count of chol)



#For males   = chest pain type 0(with highest count of chol),chest pain type 3(with least count of hol)
sns.barplot(x=df_heart['cp'],y=df_heart['exang']) #insights obtained from visualization

#chest pain type 0 gender will suffer from much pain as due to chest pain angina pain is caused,type 1 has least angina pain
sns.jointplot(x='age',y='thalach',data=df_heart) #insights obtained from data visualization

#maximum heart rate (thalach) decreases with age.
sns.boxplot(x='sex',y='age',hue='thal',data=df_heart)#insights obtained from data visulization graphs

#thal type 2 = females (age>70)

#thal type 2 = males(age>70)

#males do not suffer from thal type 0 and 1

#males suffer from thal 1
#Data Cleaning(heart disease due to cholestrol)

df_heart.head()
Sex = pd.get_dummies(df_heart['sex'],drop_first=True) #first only for male we will predict whether heart disease will occur or not using confusion matrix

Sex
cp1 = pd.get_dummies(df_heart['cp'],drop_first=True)

df_heart.drop(['sex','trestbps','fbs','restecg','thalach','oldpeak','slope','ca','cp','thal'],axis=1,inplace=True)#unnecessay columns are dropped

df_heart1 = pd.concat([df_heart,Sex,cp1],axis=1)
df_heart1
#Logistical regression to predict target (heart disease due to chol as of which chest pain is observed which leads to angina pain)

#Create a feature data and label data

X = df_heart1.drop('target',axis=1) #except target include all attributes which is feature,axis= 1 is should be specified as the attribute is in the column

y = df_heart1['target'] #label to be predicted
#Split data into train and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)#pass the feature and label as parameters

#Train the model and predict



from sklearn.linear_model import LogisticRegression #this is model used to predict the label
lm = LogisticRegression() #create an instance
#Fit the data

lm.fit(X_train,y_train) #ignore the warning
#Prediction

predict = lm.predict(X_test)
#To get report of prediction

from sklearn.metrics import classification_report
print(classification_report(y_test,predict))#use orginal label and predicted label to compare
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predict))

#TN - 33 which means predicted not to have occured and have not occured in real world also

#FP - 11 which means predicted to have occured but have not occured

#FN - 9 which mean predicted to have not have not occured but have occured (It is good tht value is small)

#TN - 38 which means predicted to have occured and have occured
#to predict decrease in heart rate with age and increase in heart rate with cholestrol leads to heart attack

#Data Cleaning

df_heart = pd.read_csv("../input/heart_dataset.csv") 

df_heart
df_heart.columns
df_heart.drop(['sex','trestbps','fbs','restecg','thal','oldpeak','slope','ca','cp','target','exang'],axis=1,inplace=True)

df_heart1 = pd.concat([df_heart],axis=1)
df_heart1
sns.lmplot(x='thalach',y='age',data=df_heart1) #with increase in age heart rate decreases
sns.lmplot(x='thalach',y='chol',data=df_heart1) #with increase in cholestrol heart rate increases
#Machine learning alogrithm

#Feature and label data

X = df_heart1.drop('thalach',axis=1) #age of person and cholestrol which is feature

y = df_heart1['thalach'] #label: Heart rate

X
#Train and test data split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
#Test using linear regression model

from sklearn.linear_model import LinearRegression
lm = LinearRegression() #create an instance
lm.fit(X_train,y_train) #fit the model using trained data
# print the intercept

print(lm.intercept_)#intercept 
print(lm.coef_)
#Prediction

predictions = lm.predict( X_test) #to predict data for feature test data
plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y') 
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient']) #negative coefficient means indirectly proportional

coeff_df                                                            #positive coefficient means directly proprotional(indexes value and column value)
#Insights

#With one unit increase in age ,there will be  1 unit decrease in heart rate

#With one unit increase in cholestrol,there will be 1 unit increase in heart rate 

#This bad cholestrol could lead to heart attack or stroke
df_heart = pd.read_csv("../input/heart_dataset.csv") 

df_heart
df_heart['hatl'] = df_heart.apply(lambda _: '', axis=1) #Empty column is added (dummy column is added)
df_heart #halt is added which is dummy column
value = df_heart['fbs']

value #value of fasting blood sugar value 
#by using this one can determine whether patient might have heart attack or not

#To determine sugar levels

def _fbs_(value):

    if value == 1 :

        return 'Safe level(No heart attack)'

    else:

        return 'Not Safe level(Heart Attack)'

    

#If value of fasting blood sugar(fbs) >120 which is 1 then person might not be predicted suffer from any heart related problems but if it is 0 then person might be predicted to suffer from heart related problem
df_heart['hatl'].apply(_fbs_).head() #we r adding to it as function to determine level of heart attack

#Insights obtained from the analysis and prediction : larger number of people are suffering from heart disease related issues, so methods have to be improvised accodingly