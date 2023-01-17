# Importing the Required Libraries for data analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
# Reading the dataset as a dataframe
file_name = '../input/data.csv'
data_df = pd.read_csv(file_name)
# A sneak peek into the dataframe
data_df.head()
data_df.describe()
# Finding the count of Null Values in each column
pd.DataFrame(data_df.isna().sum())
# Dropping the Column that have all Nan/Null values only
data_df=data_df.set_index('id')
data_df.drop(columns=['Unnamed: 32'],axis=1,inplace=True)
print(data_df.shape)
data_df.head()
sns.countplot(data_df.diagnosis,label='count')
B, M = data_df.diagnosis.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)
print('Percentage of Benign: ',B/(B+M)*100)
print('Number of Malignant : ',M/(B+M)*100)
encoder = LabelEncoder()
data_df.diagnosis = encoder.fit_transform(data_df.diagnosis)
data_df.head()
# Using Seaborn pair plot to take a look at the data graph
sns.pairplot(data_df,dropna=True)
plt
X = data_df.drop(columns=['diagnosis'],axis=1).values
y = data_df['diagnosis']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,
                                                 random_state=0)
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
