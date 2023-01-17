#importing relevant libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#uploading the Titanic dataset
df1 = pd.read_csv('../input/gender_submission.csv')
df2 = pd.read_csv('../input/test.csv')
df3 = pd.read_csv('../input/train.csv')
#previewing our data i.e the gender_submission data
df1.head()
#previewing our test data i.e test data
df2.head()
#previewing our train data
df3.head()
#copy of train data
clean_data = df3.copy()
#copy of test data
clean_data2 = df2.copy()
#y_train
y=clean_data[['Survived']].copy()
#Dropping unnecessary columns
df2 = df2.drop(['PassengerId','Name','Ticket','Cabin','Embarked','Fare'],axis=1)
#Replacing null values with mean
df2 = df2.fillna(df2.mean())
#are there any null values?
df3.isnull().any()
#Replacing null values with mean
df3 = df3.fillna(df3.mean())
df3.columns
#Dropping unnecessary columns
df3 = df3.drop(['PassengerId','Name','Ticket','Cabin','Embarked','Fare','Survived'],axis=1)
#creating a copy of gender in both train and test datasets before encoding
gender=df3[['Sex']].copy()

gender2=df2[['Sex']].copy()
values = array(gender)
#Integer encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
values2 = array(gender2)
#Integer encoding(df2)
label_encoder2 = LabelEncoder()
integer_encoded2 = label_encoder2.fit_transform(values2)
#Binary encoding
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#Binary encoding(df2)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded2 = integer_encoded.reshape(len(integer_encoded),1)
onehot_encoded2 = onehot_encoder.fit_transform(integer_encoded2)
#Inverting back NOT RUN YET
#inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0:])])
#Replacing the coded column in the dataframe for train data
sex = pd.DataFrame(integer_encoded)
df3['Sex'] = sex
#Replacing the coded column in the dataframe for test data
sex = pd.DataFrame(integer_encoded2)
df2['Sex'] = sex
#Assessing descriptives on Sex variable in train dataset
df3['Sex'].describe()
#basic information about our train data
df3.info()
#type of each column values
df3.dtypes
df3['Age'].head()
df3.index
df3.describe()
#number of times the unique Survived appear
clean_data['Survived'].value_counts()
df3['Sex'].value_counts()
df2['Sex'].value_counts()
#Checking the variables in train data
df3.columns
plt.style.use("classic")
#plt.style.use("fivethirtyeight")
plt.style.use("ggplot")
#plt.style.use("seaborn-whitegrid")
#plt.style.use("seaborn-pastel")
#plt.style.use(["dark_background", "fivethirtyeight"])
#histogram for Age

count, bin_edges = np.histogram(df3,15)

df3['Age'].plot(kind = 'hist',
               figsize = (10,6),
               bins = 15,
               xticks = bin_edges
               )

plt.title(' Histogram of Age in Titanic (train) dataset')
plt.xlabel('Age')
plt.show()
df3['Sex'].value_counts()
import seaborn as sns
plt.figure(figsize=(10,6))
sns.set(style="darkgrid")
ax = sns.countplot(x="Sex", data=clean_data,saturation=.80)
import seaborn as sns
plt.figure(figsize=(10,6))
sns.set(style="darkgrid")
#tit = sns.load_dataset("clean_data")
ax = sns.countplot(x="Pclass",hue="Sex", data=clean_data)
import seaborn as sns
plt.figure(figsize=(10,6))
sns.set(style="darkgrid")
ax = sns.countplot(x='SibSp', data=clean_data, saturation=.80)
import seaborn as sns
plt.figure(figsize=(10,6))

sns.set(style="darkgrid")
ax = sns.countplot(x='Parch', data=clean_data,saturation=.80)
clean_data['Survived'].corr(clean_data['Fare'])
clean_data['Survived'].corr(df3['Age'])
clean_data['Survived'].corr(df3['Sex'])
clean_data['Survived'].corr(df3['Pclass'])
clean_data['Survived'].corr(df3['SibSp'])
clean_data['Survived'].corr(df3['Parch'])
#scatterplot of Age and survival in seaborn
plt.figure(figsize=(10,6))

sns.set(font_scale=1.5)
x = df3['Age']
y = clean_data['Survived']
sns.regplot(x,y, fit_reg=True, color = 'green')
#scatterplot of Sex and survival in seaborn
plt.figure(figsize=(10,6))

x = df3['Sex']
y = clean_data['Survived']
sns.regplot(x,y, fit_reg=True)
#scatterplot of Pclass and survival in seaborn
plt.figure(figsize=(10,6))

x = df3['Pclass']
y = clean_data['Survived']
sns.regplot(x,y, fit_reg=True)
#scatterplot of Parch and survival in seaborn
plt.figure(figsize=(10,6))
x = df3['Parch']
y = clean_data['Survived']
sns.regplot(x,y, fit_reg=True)
#scatterplot of SibSp and survival in seaborn
plt.figure(figsize=(10,6))
x = df3['SibSp']
y = clean_data['Survived']
sns.regplot(x,y, fit_reg=True)
plt.figure(figsize=(10,6))
sns.violinplot(y = 'Survived', x = 'Parch', data = clean_data, inner = 'quartile')
plt.figure(figsize=(10,6))
sns.violinplot(y = 'Survived', x = 'SibSp', data = clean_data, inner = 'quartile')
plt.figure(figsize=(10,6))
sns.violinplot(y = 'Survived', x = 'Sex', data = clean_data, inner = 'quartile')
plt.figure(figsize=(10,6))
sns.violinplot(y = 'Survived', x = 'Pclass', data = clean_data, inner = 'quartile')
survival_feautures = ['Sex','Pclass','Parch','SibSp', 'Age']
X = df3[survival_feautures].copy()
X.columns
y.head()
#Fitting a DecisionTree

survival_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=389)
survival_classifier.fit(X, y)
type(survival_classifier)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#Fitting a RandomForest

rf_model = RandomForestClassifier(n_estimators=1000,max_features=3,oob_score=True,random_state = 389)
rf_model.fit(X,y)
df2.columns
df2.head()
df2.info()
#predicting on method 1
predictions = survival_classifier.predict(df2)
#predicting on method 2
predictions2 = rf_model.predict(df2)
predictions[:10]
predictions2[:10]
type(predictions2)
#data frame on prediction
pred_survivors = pd.DataFrame(predictions)
pred_survivors.columns = ['survived']

frames = [clean_data2,pred_survivors]

result = pd.concat(frames, axis=1)
#data frame on prediction 2
pred_survivors2 = pd.DataFrame(predictions2)
pred_survivors2.columns = ['survived2']

frames2 = [clean_data2, pred_survivors2]

result2 = pd.concat(frames2, axis=1)
#Dropping unnecessary columns from the output
output = result.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)

output2 = result2.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
#Print out the head of the expected output
output.head()
#Print out the head of the expected output
output2.head()
pred_survivors2.head()
#dropping passengerId to assess accuracy on survived column
df1 = df1.drop(['PassengerId'],axis=1)

accuracy_score(y_true=df1,y_pred=predictions)
print("OOB_accuracy:")
print(rf_model.oob_score_)
for feature, imp in zip(X,rf_model.feature_importances_):
    print(feature,imp)
#printing output (method 1) in csv

output.to_csv('survived_submission.csv', header=True, index=True, sep=',')
#Creating Submission dataframe for Kaggle
mysubmission = pd.DataFrame({"PassengerId":clean_data2["PassengerId"],
                            "Survived":pred_survivors2["survived2"]})

#Saving to CSV
mysubmission.to_csv("survived_mysubmission.csv", index=False)