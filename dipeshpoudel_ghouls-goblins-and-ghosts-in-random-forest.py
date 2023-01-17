# Importing the Required Libaries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
plt.style.use('seaborn')
# Reading the Train and Test files
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/sample_submission.csv")
submission["type"] = "Unknown"
print("Train Data Null Values Count \n",train_df.isna().sum())
print("Test Data Null Values Count \n",test_df.isna().sum())
for col in train_df.drop(columns=['id','color','type'],axis=1).columns:
    sns.distplot(train_df[col])
    plt.show()
for col in test_df.drop(columns=['id','color'],axis=1).columns:
    sns.distplot(test_df[col])
    plt.show()
train_df.head()
test_df.head()
train_df.color.value_counts().plot(kind='bar')
test_df.color.value_counts().plot(kind='bar')
train_data = train_df.drop(columns=['id'],axis=1)
train_data.head()
test_data = test_df.drop(columns=['id'])
test_data.head()
train_data.type.value_counts()
le = LabelEncoder()
train_data['type'] = le.fit_transform(train_data['type'])
print(train_data.type.value_counts())
train_data_x = train_data.drop(columns=['type'],axis=1)
train_data_y = train_data['type'].values
train_data_x = pd.get_dummies(train_data_x,columns=['color'],drop_first=True).values
y_data = pd.get_dummies(test_data, columns=['color'], drop_first=True).values
rfclf = RandomForestClassifier(n_estimators=1000)
rfclf.fit(train_data_x,train_data_y)
y_pred = rfclf.predict(y_data)
submission['type'] = y_pred
submission['type'] = submission.type.map({0:"Ghost", 1:"Ghoul", 2:"Goblin"})
submission.to_csv('../working/submission.csv', index=False)
