import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing the libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#Loading the dataset

employee_df = pd.read_csv("/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
# Viewing the first lines



employee_df.head()
employee_df.info()
# Analyzing statistical information about numerical variables

employee_df.describe()
# Transforming some categorical variables with YES / NO content to numeric 0/1
employee_df['Attrition'].value_counts()
employee_df['OverTime'].value_counts()
employee_df['Over18'].value_counts()
employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

employee_df['OverTime'] = employee_df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)

employee_df['Over18'] = employee_df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)

employee_df.head()
# Plotting a histogram to visualize how each feature is distributed into dataset



employee_df.hist(bins = 30, figsize = (20,20), color = 'b');
# It makes sense to drop 'EmployeeCount' , 'Standardhours' and 'Over18' since they do not change from one employee to the other

# Let's drop 'EmployeeNumber' as well

employee_df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)
employee_df.head()

# Now we have 31 columns
# Let's see how many employees left the company! 

left_df = employee_df[employee_df['Attrition'] == 1]

stayed_df = employee_df[employee_df['Attrition'] == 0]



# Count the number of employees who stayed and left

# It seems that we are dealing with an imbalanced dataset 



print("Total =", len(employee_df))



print("Number of employees who left the company:", len(left_df))

print(f"Percentage of employees who left the company: {1.*len(left_df)/len(employee_df)*100.0:.2f}%") 

print("Number of employees who did not leave the company (stayed) =", len(stayed_df))

print(f"Percentage of employees who did not leave the company (stayed): {1.*len(stayed_df)/len(employee_df)*100.0:.2f}%") 

# Lets have a look in the statistics of the employees who stayed and left to make some comparisions



left_df.describe()
stayed_df.describe()
# Lets have a look in the different correlations between the features



correlations = employee_df.corr()

f, ax = plt.subplots(figsize = (20, 20))

sns.heatmap(correlations, annot = True);
# Lets investigate if there is any correlation between people who left the company with some specific variables such as 'Age', 'JobRole', 'MaritalStatus', 'JobInvolvement' and 'JobLevel'



plt.figure(figsize=[25, 12])

sns.countplot(x = 'Age', hue = 'Attrition', data = employee_df);
plt.figure(figsize=[20,20])



plt.subplot(411)



sns.countplot(x = 'JobRole', hue = 'Attrition', data = employee_df)

plt.title("In which position the Attrition is higher / lower?");
# Let's see the Monthly Income vs. Job Role



plt.figure(figsize=(10, 10))

sns.boxplot(x = 'MonthlyIncome', y = 'JobRole', data = employee_df);

plt.title("How is the distribution of wages among the different positions?");
sns.countplot(x = 'MaritalStatus', hue = 'Attrition', data = employee_df);

plt.title("Marital Status Vs Attrition");
sns.countplot(x = 'JobInvolvement', hue = 'Attrition', data = employee_df);

plt.title("How does the level of involvement at work affect the Attrition?");
sns.countplot(x = 'JobLevel', hue = 'Attrition', data = employee_df)

plt.title("Job level Vs Attrition");
# Let's use KDE (Kernel Density Estimate) to visualize the probability density of a continuous variable.



# Investigating DistanceFromHome



plt.figure(figsize=(12,7))

sns.kdeplot(left_df['DistanceFromHome'], label = 'Employees who left', shade = True, color = 'r')

sns.kdeplot(stayed_df['DistanceFromHome'], label = 'Employees who Stayed', shade = True, color = 'b')

plt.xlabel('Distance From Home');

plt.ylabel('Attrition');

plt.title("Does the distance from home to work impact Attrition?");
# Investigating YearsWithCurrManager



plt.figure(figsize=(12,7))

sns.kdeplot(left_df['YearsWithCurrManager'], label = 'Employees who left', shade = True, color = 'r')

sns.kdeplot(stayed_df['YearsWithCurrManager'], label = 'Employees who Stayed', shade = True, color = 'b')

plt.xlabel('Years With Current Manager');

plt.title("Does the length of stay as a Current Manager influence the departure of employees?");
# Investigating TotalWorkingYears



plt.figure(figsize=(12,7))

sns.kdeplot(left_df['TotalWorkingYears'], shade = True, label = 'Employees who left', color = 'r')

sns.kdeplot(stayed_df['TotalWorkingYears'], shade = True, label = 'Employees who Stayed', color = 'b')

plt.xlabel('Total Working Years');

plt.ylabel('Attrition');

plt.title("Is there a relationship between total working time in the company and Attrition?");
# Checking the types of each feature

employee_df.dtypes
# Separating categorical data from the rest of the dataframe to then convert it to numeric

X_cat = employee_df[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]

X_cat
# Converting the categorical features into numbers using OneHotEncoder

from sklearn.preprocessing import OneHotEncoder



onehotencoder = OneHotEncoder()

X_cat = onehotencoder.fit_transform(X_cat).toarray()

X_cat.shape
# Converting into dataframe

X_cat = pd.DataFrame(X_cat)

X_cat 
# Separating the numerical data

X_numerical = employee_df[['Age', 'DailyRate', 'DistanceFromHome','Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome','MonthlyRate','NumCompaniesWorked',	'OverTime',	'PercentSalaryHike', 'PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears'	,'TrainingTimesLastYear', 'WorkLifeBalance','YearsAtCompany','YearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager']]

X_numerical
# Concatenating the categorical dataset X_cat and the numerical dataset X_numerical into a unique dataset



X_all = pd.concat([X_cat, X_numerical], axis = 1)

X_all
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X = scaler.fit_transform(X_all)

X
# Separating the feature that we want to predict



y = employee_df['Attrition']

y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

X_train.shape
X_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



model = LogisticRegression()
# Training the data



model.fit(X_train, y_train)
# Making predictions and visualizing the accuracy



LRC_pred = model.predict(X_test)





print("Accuracy: {}%".format( 100 * accuracy_score(LRC_pred, y_test)))
# Comparing the results using Confusion Matrix



from sklearn.metrics import confusion_matrix, classification_report
# Testing Set Performance



cm = confusion_matrix(LRC_pred, y_test)

sns.heatmap(cm, annot=True);
# Analyzing the KPI (Key Performance Indicator)



print(classification_report(y_test, LRC_pred))
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier()
# Training the data



model.fit(X_train, y_train)
# Making predictions and visualizing the accuracy



RFC_pred = model.predict(X_test)

print("Accuracy: {}%".format( 100 * accuracy_score(RFC_pred, y_test)))
# Testing Set Performance



cm = confusion_matrix(RFC_pred, y_test)

sns.heatmap(cm, annot=True)
# Analyzing the KPI (Key Performance Indicator)



print(classification_report(y_test, RFC_pred))
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)

KNNC_pred = model.predict(X_test)

print("Accuracy: {}%".format( 100 * accuracy_score(KNNC_pred, y_test)))
# Testing Set Performance



cm = confusion_matrix(KNNC_pred, y_test)

sns.heatmap(cm, annot=True)
# Analyzing the KPI (Key Performance Indicator)



print(classification_report(y_test, KNNC_pred))
import tensorflow as tf
# Creating the layers

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units=500, activation='relu', input_shape=(50, )))

model.add(tf.keras.layers.Dense(units=500, activation='relu'))

model.add(tf.keras.layers.Dense(units=500, activation='relu'))

model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
model.summary()
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])
# Training the model



epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50)
ANNC_pred = model.predict(X_test)

ANNC_pred = (ANNC_pred > 0.5)

print("Accuracy: {}%".format( 100 * accuracy_score(ANNC_pred, y_test)))
epochs_hist.history.keys()
plt.plot(epochs_hist.history['loss'])

plt.title('Model Loss Progress During Training')

plt.xlabel('Epoch')

plt.ylabel('Training Loss')

plt.legend(['Training Loss']);
plt.plot(epochs_hist.history['accuracy'])

plt.title('Model Accuracy Progress During Training')

plt.xlabel('Epoch')

plt.ylabel('Training Accuracy')

plt.legend(['Training Accuracy']);
# Testing Set Performance

cm = confusion_matrix(y_test, ANNC_pred)

sns.heatmap(cm, annot=True);
print(classification_report(y_test, ANNC_pred))
# Showing the results



print("Logistic Regression Classifier: {:.2f}% Accuracy".format( 100 * accuracy_score(LRC_pred, y_test)))

print("Random Forest Classifier: {:.2f}% Accuracy".format( 100 * accuracy_score(RFC_pred, y_test)))

print("K-Nearest Neighbors Classifier: {:.2f}% Accuracy".format( 100 * accuracy_score(KNNC_pred, y_test)))

print("Artificial Neural Network Classifier: {:.2f}% Accuracy".format( 100 * accuracy_score(ANNC_pred, y_test)))