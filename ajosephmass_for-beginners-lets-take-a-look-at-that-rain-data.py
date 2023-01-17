# Import the required modules for you analysis

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns



from sklearn import preprocessing # To get MinMax Scaler function

from sklearn.model_selection import train_test_split # required to split the data

from sklearn.feature_selection import SelectKBest # to select best feature

from sklearn.feature_selection import chi2 # for feature Selection

from sklearn.metrics import classification_report # to extract classsification report

from sklearn.metrics import confusion_matrix # to extract confusion matrix

from sklearn.metrics import accuracy_score # to get the scores

from sklearn.linear_model import LogisticRegression # for logistic regression

from sklearn.ensemble import RandomForestClassifier # for random forrest 

from sklearn.naive_bayes import GaussianNB # for Naive Bayes

from sklearn.feature_selection import RFE # Recursive Feature Selection

from sklearn.feature_selection import RFECV # Recursive Feature Selection with Cross Validation

from sklearn.decomposition import PCA # To apply PCA



# To plot inline

%matplotlib inline



# Load the csv to a dataframe

df = pd.read_csv('../input/weatherAUS.csv')



# Let us see whats inside..

df.head()
# Drop RISK_MM

df = df.drop(columns=['RISK_MM'],axis=1)

df.head()
# Droping columns

df = df.drop(columns=['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'],axis=1)

df.head()
# Check for any null

df.isnull().sum()
df = df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am'],axis=1)

df.head()
# substituting 1 and 0 inplace of yes and no

df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)

df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

df.head()
# lets take a look at the values from each column

df.describe()
# Since only less than 1% of following columns are missing values, lets just replace these with the mean.

# (You can also use the imput pre-processing class to do this)

df.MinTemp.fillna(df.MinTemp.mean(),inplace=True)

df.MaxTemp.fillna(df.MaxTemp.mean(),inplace=True)

df.Rainfall.fillna(df.Rainfall.mean(),inplace=True)

df.WindSpeed9am.fillna(df.WindSpeed9am.mean(),inplace=True)

df.WindSpeed3pm.fillna(df.WindSpeed3pm.mean(),inplace=True)

df.Humidity9am.fillna(df.Humidity9am.mean(),inplace=True)

df.Temp9am.fillna(df.Temp9am.mean(),inplace=True)

df.Temp3pm.fillna(df.Temp3pm.mean(),inplace=True)

df.RainToday.fillna(df.RainToday.mean(),inplace=True)

df.Humidity3pm.fillna(df.Humidity3pm.mean(),inplace=True)



# And for columns with close to 5% missing values lets randomly fill them with values close to the mean value but 

# within one standard deviation.

WindGustSpeed_avg = df['WindGustSpeed'].mean()

WindGustSpeed_std = df['WindGustSpeed'].std()

WindGustSpeed_null_count = df['WindGustSpeed'].isnull().sum()

WindGustSpeed_null_random_list = np.random.randint(WindGustSpeed_avg - WindGustSpeed_std, WindGustSpeed_avg + WindGustSpeed_std, size=WindGustSpeed_null_count)

df['WindGustSpeed'][np.isnan(df['WindGustSpeed'])] = WindGustSpeed_null_random_list



Pressure9am_avg = df['Pressure9am'].mean()

Pressure9am_std = df['Pressure9am'].std()

Pressure9am_null_count = df['Pressure9am'].isnull().sum()

Pressure9am_null_random_list = np.random.randint(Pressure9am_avg - Pressure9am_std, Pressure9am_avg + Pressure9am_std, size=Pressure9am_null_count)

df['Pressure9am'][np.isnan(df['Pressure9am'])] = Pressure9am_null_random_list



Pressure3pm_avg = df['Pressure3pm'].mean()

Pressure3pm_std = df['Pressure3pm'].std()

Pressure3pm_null_count = df['Pressure3pm'].isnull().sum()

Pressure3pm_null_random_list = np.random.randint(Pressure3pm_avg - Pressure3pm_std, Pressure3pm_avg + Pressure3pm_std, size=Pressure3pm_null_count)

df['Pressure3pm'][np.isnan(df['Pressure3pm'])] = Pressure3pm_null_random_list





#replace negative values with 0

df.clip(lower=0)

df[df < 0] = 0

# From selectkbest find the best score and plot it

from sklearn.feature_selection import SelectKBest, chi2

X = df.loc[:,df.columns!='RainTomorrow']

y = df[['RainTomorrow']]

selector = SelectKBest(chi2, k=3)

selector.fit(X, y)

X_new = selector.transform(X)

scores = selector.scores_

print(X.columns[selector.get_support(indices=True)]) #top 3 columns



# Plot the scores. Lets see if we can visualise which are the best?

plt.bar(range(len(X.columns)), scores)

plt.xticks(range(len(X.columns)), X.columns, rotation='vertical')

plt.show()
# Plotting the corealation heat map 

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
# Select the required columns for testing and training

X = df.loc[:, ['Rainfall', 'Humidity3pm', 'Humidity9am', 'WindGustSpeed', 'Temp3pm', 'MaxTemp']].shift(-1).iloc[:-1].values

y = df.iloc[:-1, -1:].values.astype('int')



# Logistic Regression 

# Split the data to appropriate testing sample size

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

model_lr = LogisticRegression(random_state=0)

model_lr.fit(X_train,y_train)

prediction_lr = model_lr.predict(X_test)

score = accuracy_score(y_test,prediction_lr)

print('Accuracy - Logistic Regression:',score)



# Making the Confusion Matrix

cm = confusion_matrix(y_test, prediction_lr)



# Ploting the Confusion Matrix

f, ax = plt.subplots(figsize = (3,3))

sns.heatmap(cm,annot=True,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Predictions")

plt.ylabel("Test Values")

plt.show()
# Splitting the data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)



# Random Forrest

model_rf = RandomForestClassifier(n_estimators= 25, max_depth= None,max_features = 0.4,random_state= 11 )

# Fitting the data

model_rf.fit(X_train, y_train)

# Making a prediction

prediction_rf =model_rf.predict(X_test)

score = accuracy_score(y_test,prediction_rf)

print('Accuracy - Random Forrest:',score)

print(classification_report(y_test, prediction_rf))



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, prediction_rf)



# Plotting the data

f, ax = plt.subplots(figsize = (3,3))

sns.heatmap(cm,annot=True,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Predictions")

plt.ylabel("Test Values")

plt.show()
# Splitting the data    

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)



# Gaussian Navie Bayes

model_gNB = GaussianNB()

model_gNB.fit(X_train,y_train)

prediction_gNB = model_gNB.predict(X_test)

score = accuracy_score(y_test,prediction_gNB)

print('Accuracy - GaussianNB:',score)



# Making the Confusion Matrix

cm = confusion_matrix(y_test, prediction_gNB)



# Plotting the matrix

f, ax = plt.subplots(figsize = (3,3))

sns.heatmap(cm,annot=True,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Predictions")

plt.ylabel("Test Values")

plt.show()
