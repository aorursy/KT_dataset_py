# Libraries loading for the work
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Sklearn Library requirements for decision tree model building
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
# print the data files path 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
# Check to see if any missing values in the df
pd.isnull(df).sum()
df.info()
df.head()
df.describe()
labelencoder=LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])

df.head()
# Column viel-type column is not having any thing other than 0 hence this does not contribute for classification, hence removed
df = df.drop("veil-type", axis = 1)
# Splitting the data into train and test sets
y= df["class"]
x= df.drop("class",axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state= 42)
# Let us explore how after splitting the files look like
y.head()
# How many rows. 
len(y)
# y is Complete categorical column 
x.info()
x_train.head()
# Complete columns (All predictors) are in x_train 
y_train
x.info()
dt = DecisionTreeClassifier(random_state = 43)
dt.fit(x_train, y_train)
plt.figure(figsize=(30,20))
plt.title("Decision Tree")
plot_tree(dt, feature_names=x_train.columns,  filled=True, rounded = True,fontsize= 16)
y_pred = dt.predict(x_test)
y_actual = pd.DataFrame(y_test.value_counts())
y_actual = y_actual.reset_index()
y_actual.columns = ['Class', 'AcutalCnt']

y_predicted = pd.DataFrame(y_pred, columns=["Predicted"])["Predicted"]
y_predicted = pd.DataFrame(y_predicted.value_counts())
y_predicted = y_predicted.reset_index()
y_predicted.columns = ["Class","PredictCnt"]
y_predicted

confusion_df = pd.merge(y_actual, y_predicted, on='Class', how='outer')
confusion_df['Error'] = abs(confusion_df['AcutalCnt']-confusion_df['PredictCnt'])
confusion_df
accuracy = (confusion_df.AcutalCnt.sum()-confusion_df.Error.sum())/confusion_df.AcutalCnt.sum()*100
print(confusion_df)
print("Model Accuracy is", accuracy)

