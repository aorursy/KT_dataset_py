import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

#Showing full path of datasets
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

#Disable warnings
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv")
df.head()
#Number of rows and columns in our dataset
df.shape
#The 24 columns 
df.columns
#As mentioned in the dataset description , 
#we should exclude the variable Risk-MM when training a binary classification model.
#Not excluding it will leak the answers to your model and reduce its predictability.

df.drop(['RISK_MM'],axis=1,inplace=True)
#Basic Information of dataset

df.info()
#Before looking at the description of the data
#We can see that there are few columns with very less data
#Evaporation,Sunshine,Cloud9am,Cloud3pm
#It is better to remove these four columns as it will affect our prediction even if we
#fill the na values...

#Date and Location is also not required
#As we are predicting rain in australia and not when and where in australia


drop_cols = ['Evaporation','Sunshine','Cloud9am','Cloud3pm','Date','Location']

df.drop(columns=drop_cols,inplace=True,axis=1)
df.info()
#Basic description of our data
#Numerical features first
df.describe()
#Including Categorical features with include object
df.describe(include='object')
#Now including all the features
df.describe(include='all')
#Our dataset consists of 142193 rows and the count for many features is less than 142193.
#This shows presence of Null values.
#Let's look at the null values..

df.isna().sum()
df.skew()
#Filling missing values

#We can see that there are outliers in our data
#So the best way to fill the na values in our numerical features is with median
#Because median deals the best with outliers

#Let's separate numerical and categorical
#data type of numerical features is equal to float64
#With the help of following list comprehension we separate the numerical features...

num = [col for col in df.columns if df[col].dtype=="float64"]

for col in num:
    df[col].fillna(df[col].median(),inplace=True)
    
cat = [col for col in df.columns if df[col].dtype=="O"]
for col in cat:
    df[col].fillna(df[col].mode()[0],inplace=True)
#Check missing values
df.isna().sum()
df.corr().style.background_gradient(cmap="Reds")
#With the use of heatmap
corr = df.corr()

fig = plt.figure(figsize=(12,12))
sns.heatmap(corr,annot=True,fmt=".1f",linewidths="0.1")
print("Numerical features :: {}\n".format(num))
print("No of Numerical features :: {}".format(len(num)))
plt.figure(figsize=(15,15))
plt.subplots_adjust(hspace=0.5)

i=1
colors = ['Red','Blue','Green','Cyan',
         'Red','Blue','Green','Cyan',
         'Red','Blue','Green','Cyan']
j=0
for col in num:
    plt.subplot(3,4,i)
    a1 = sns.distplot(df[col],color=colors[j])
    i+=1
    j+=1
plt.figure(figsize=(15,30))
plt.subplots_adjust(hspace=0.5)

i=1
for col in num:
    plt.subplot(6,2,i)
    a1 = sns.boxplot(data=df,x="RainTomorrow",y=col)
    i+=1
#Create a loop that finds the outliers in train and test  and removes it
features_to_examine = ['Rainfall','WindGustSpeed','WindSpeed9am','WindSpeed3pm']

for col in features_to_examine:
    IQR = df[col].quantile(0.75) - df[col].quantile(0.25) 
    Lower_Bound = df[col].quantile(0.25) - (IQR*3)
    Upper_Bound = df[col].quantile(0.75) + (IQR*3)
    
    print("The outliers in {} feature are values <<< {} and >>> {}".format(col,Lower_Bound,Upper_Bound))
    
    minimum = df[col].min()
    maximum = df[col].max()
    print("The minimum value in {} is {} and maximum value is {}".format(col,minimum,maximum))
    
    if maximum>Upper_Bound:
          print("The outliers for {} are value greater than {}\n".format(col,Upper_Bound))
    elif minimum<Lower_Bound:
          print("The outliers for {} are value smaller than {}\n".format(col,Lower_Bound))
plt.figure(figsize=(15,30))
plt.subplots_adjust(hspace=0.5)

i=1
for col in num:
    plt.subplot(6,2,i)
    a1 = sns.barplot(data=df,x="RainTomorrow",y=col)
    i+=1
plt.figure(figsize=(15,5))
plt.subplots_adjust(hspace=0.5)

i=1
features_list = ["MaxTemp","Temp9am","Temp3pm"]
for feature in features_list:
    plt.subplot(1,3,i)
    sns.scatterplot(data=df,x="MinTemp",y=feature,hue="RainTomorrow")
    i+=1
plt.figure(figsize=(15,8))
plt.subplots_adjust(hspace=0.5)

plt.subplot(3,2,1)
sns.scatterplot(data=df,x="WindSpeed9am",y="WindGustSpeed",hue="RainTomorrow")

plt.subplot(3,2,2)
sns.scatterplot(data=df,x="WindSpeed3pm",y="WindGustSpeed",hue="RainTomorrow")

plt.subplot(3,2,3)
sns.scatterplot(data=df,x="Humidity9am",y="Humidity3pm",hue="RainTomorrow")

plt.subplot(3,2,4)
sns.scatterplot(data=df,x="Temp9am",y="Temp3pm",hue="RainTomorrow")

plt.subplot(3,2,5)
sns.scatterplot(data=df,x="MaxTemp",y="Temp9am",hue="RainTomorrow")

plt.subplot(3,2,6)
sns.scatterplot(data=df,x="Humidity3pm",y="Temp3pm",hue="RainTomorrow")
cat
df['WindGustDir'].value_counts()
fig = plt.figure(figsize=(15,5))
sns.countplot(data=df,x="WindGustDir",hue="RainTomorrow");
df['WindDir9am'].value_counts()
fig = plt.figure(figsize=(15,5))
sns.countplot(data=df,x="WindDir9am",hue="RainTomorrow");
df['WindDir3pm'].value_counts()
fig = plt.figure(figsize=(15,5))
sns.countplot(data=df,x="WindDir3pm",hue="RainTomorrow");
df['RainTomorrow'].value_counts()
sns.countplot(data=df,x="RainTomorrow")
from sklearn.model_selection import train_test_split as tts
y=df[['RainTomorrow']]
X=df.drop(['RainTomorrow'],axis=1)

X_train,X_test,y_train,y_test = tts(X,y,test_size=0.3,random_state=0)
X_train
X_test
#We'll plot these four as subplots 

plt.figure(figsize=(15,30))
plt.subplots_adjust(hspace=0.5)

features_to_examine = ['Rainfall','WindGustSpeed','WindSpeed9am','WindSpeed3pm']
i=1
for col in features_to_examine:
    plt.subplot(6,2,i)
    fig = df[col].hist(bins=10)
    fig.set_xlabel(col)
    fig.set_ylabel('RainTomorrow')
    i+=1
def remove_outliers(df,col,Lower_Bound,Upper_Bound):    
    minimum = df[col].min()
    maximum = df[col].max()
    
    if maximum>Upper_Bound:
        return np.where(df[col]>Upper_Bound,Upper_Bound,df[col])
          
    elif minimum<Lower_Bound:
        return np.where(df[col]<Lower_Bound,Lower_Bound,df[col])
for df1 in [X_train,X_test]:
    df1['Rainfall'] = remove_outliers(df1,'Rainfall',-1.799,2.4)
    df1['WindGustSpeed'] = remove_outliers(df1,'WindGustSpeed',-14.0,91.0)
    df1['WindSpeed9am'] = remove_outliers(df1,'WindSpeed9am',-29.0,55.0)
    df1['WindSpeed3pm'] = remove_outliers(df1,'WindSpeed3pm',-20.0,57.0)
#If we look at their boxplots we can see that the outliers are now capped...
plt.figure(figsize=(15,30))
plt.subplots_adjust(hspace=0.5)

features_to_examine = ['Rainfall','WindGustSpeed','WindSpeed9am','WindSpeed3pm']
i=1
for col in features_to_examine:
    plt.subplot(6,2,i)
    fig = sns.boxplot(data=X_train,y=col)
    fig.set_xlabel(col)
    fig.set_ylabel('RainTomorrow')
    i+=1
#Describe helps us understand more about the mean and max values

X_train[features_to_examine].describe()
X_test[features_to_examine].describe()
#Our next step is to encode all the categorical variables.
#first we will convert our target variable

for df2 in [y_train,y_test]:
    df2['RainTomorrow'] = df2['RainTomorrow'].replace({"Yes":1,
                                                    "No":0})


import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
#Now we will make our training dataset

X_train = pd.concat([X_train[num],X_train[['RainToday_0','RainToday_1']],
                    pd.get_dummies(X_train['WindGustDir']),
                    pd.get_dummies(X_train['WindDir9am']),
                    pd.get_dummies(X_train['WindDir3pm'])],axis=1)

X_train.head()
#Same for testing set

X_test = pd.concat([X_test[num],X_test[['RainToday_0','RainToday_1']],
                    pd.get_dummies(X_test['WindGustDir']),
                    pd.get_dummies(X_test['WindDir9am']),
                    pd.get_dummies(X_test['WindDir3pm'])],axis=1)
X_test.head()
#our training and testing set is ready for our model
#But ,before that we need to bring all the features to same scale with feature scaling
#For this we will use MinMaxScaler
#As there our negative values in our dataset and MinMaxScaler scales our data in range -1 to 1.

cols = X_train.columns

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train,columns=cols)
X_test = pd.DataFrame(X_test,columns=cols)
from sklearn.linear_model import LogisticRegression

# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)
#Prediction on Xtest

y_pred_test = logreg.predict(X_test)

y_pred_test
#using predict_proba gives the probability value for the target feature

logreg.predict_proba(X_test)
#probability of getting no rain (0)

logreg.predict_proba(X_test)[:,0]
#probability of getting rain (1)

logreg.predict_proba(X_test)[:,1]
#Check accuracy with accuracy_score

from sklearn.metrics import accuracy_score

predict_test = accuracy_score(y_test,y_pred_test)

print("Accuracy of model on test set :: {}".format(predict_test))
#Creating confusion matrix

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred_test)
print(confusion_matrix)
#Classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))
#Comparing train and test accuracy

y_pred_train = logreg.predict(X_train)
y_pred_train
#Check accuracy of our model with train set

predict_train = accuracy_score(y_train,y_pred_train)
print("Accuracy of our model on train set :: {}".format(predict_train))
#Overall Accuracy

print("Accuracy of our model :: {}".format(logreg.score(X_test,y_test)))
#Let's try to improve the accuracy of our model

#Let's try different C values

#Now what is C
#C=100

# instantiate the model
logreg100 = LogisticRegression(solver='liblinear',C=100, random_state=0)


# fit the model
logreg100.fit(X_train, y_train)

#Prediction on Xtest

y_pred_test = logreg100.predict(X_test)

y_pred_test
predict_test = accuracy_score(y_test,y_pred_test)

print("Accuracy of model on test set :: {}".format(predict_test))
#Overall Accuracy

print("Accuracy of our model :: {}".format(logreg100.score(X_test,y_test)))
#Confusion matrix
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred_test)
print(confusion_matrix)
#Classification report
print(classification_report(y_test, y_pred_test))
#Let's increase the regularization strength

#C=0.01

# instantiate the model
logreg001 = LogisticRegression(solver='liblinear',C=0.01, random_state=0)


# fit the model
logreg001.fit(X_train, y_train)

#Prediction on Xtest

y_pred_test = logreg001.predict(X_test)

y_pred_test
predict_test = accuracy_score(y_test,y_pred_test)

print("Accuracy of model on test set :: {}".format(predict_test))
#Overall Accuracy

print("Accuracy of our model :: {}".format(logreg001.score(X_test,y_test)))
# store the predicted probabilities for class 1 - Probability of rain

y_pred1 = logreg100.predict_proba(X_test)[:, 1]
y_pred0 = logreg100.predict_proba(X_test)[:, 0]
# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)
plt.hist(y_pred0, bins = 10)

# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities')


# set the x-axis limit
plt.xlim(0,1)

#Set legend
plt.legend('upper left' , labels = ['Rain','No Rain'])

# set the title
plt.xlabel('Predicted probabilities')
plt.ylabel('Frequency')