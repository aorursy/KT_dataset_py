
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Libraries for decision tree modelling, viewing the decision tree etc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# Read the csv file into dataframe
df = pd.read_csv("/kaggle/input/iris/Iris.csv")
#Check the dataframe information
df.info()
# There are 4 - Numerical Features and one categorical column.
# There are totally 150 rows or observations are in data
# Check to see if any missing values in the df
pd.isnull(df).sum()
# Observed No missing values in the file
df = df.drop('Id', axis = 1)
# Splitting the data into train and test sets
y= df["Species"]
x= df.drop("Species",axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state= 42)

# Let us explore how after splitting the files look like
y.head()
# How many rows. 
len(y)
# Complete categorical column is y
x.info()
x_train.head()
# Complete non categorical columns (All predictors) are in x_train
y_train
x.info()
dt = DecisionTreeClassifier(random_state = 43)
dt.fit(x_train, y_train)
plt.figure(figsize=(30,20))
plt.title("Decision Tree")
plot_tree(dt, feature_names=x_train.columns, class_names= y, filled=True, rounded = True,fontsize= 16)
y_pred = dt.predict(x_test)
y_actual = pd.DataFrame(y_test.value_counts())
y_actual = y_actual.reset_index()
y_actual.columns = ['Condition', 'AcutalCnt']

y_predicted = pd.DataFrame(y_pred, columns=["Predicted"])["Predicted"]
y_predicted = pd.DataFrame(y_predicted.value_counts())
y_predicted = y_predicted.reset_index()
y_predicted.columns = ["Condition","PredictCnt"]
y_predicted

confusion_df = pd.merge(y_actual, y_predicted, on='Condition', how='outer')
confusion_df['Error'] = abs(confusion_df['AcutalCnt']-confusion_df['PredictCnt'])
confusion_df
accuracy = (confusion_df.AcutalCnt.sum()-confusion_df.Error.sum())/confusion_df.AcutalCnt.sum()*100
print(confusion_df)
print("Model Accuracy is", accuracy)
