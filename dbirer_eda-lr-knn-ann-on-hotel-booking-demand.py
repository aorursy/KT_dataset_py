# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #data visualization
import matplotlib.pyplot as plt #data visualization

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")
df.head()
print("Number of instance :",df.shape[0])
print("Number of column :",df.shape[1])
def missing_values(df):
    return {i:df[i].isna().sum() for i in df.columns}
print(missing_values(df))
df=df.drop(["company","agent"],axis=1)
df.dtypes
df=df.dropna()
df.corr()["is_canceled"].sort_values(ascending=False)
plt.figure(figsize=(16,12))
sns.heatmap(df.corr(),annot=True)
plt.ylim(16,0);
plt.figure(figsize=(10,6))
sns.countplot(x="hotel",hue="is_canceled",data=df);
plt.figure(figsize=(10,6))
sns.distplot(df["lead_time"],bins=25,
             kde=False);
df["arrival_date_year"].value_counts()
sns.countplot(x="arrival_date_year",hue="is_canceled",data=df);
df["arrival_date_month"].value_counts()
plt.figure(figsize=(12,6))
sns.countplot(x="arrival_date_month",hue="is_canceled",data=df,
              order=["January","February","March","April","May","June","July","August","September","October","November","December"]);
df=df.drop("arrival_date_week_number",axis=1)
plt.figure(figsize=(12,6))
sns.countplot(x="arrival_date_day_of_month",data=df);
plt.figure(figsize=(10,6))
sns.distplot((df["stays_in_weekend_nights"] + df["stays_in_week_nights"]),kde=False,bins=25);
df["adults"].value_counts()
df["children"].value_counts()
df["babies"].value_counts()
sns.countplot(x="meal",hue="is_canceled",data=df);
countries=df["country"].value_counts().index
comes=list(df["country"].value_counts().values)
plt.figure(figsize=(15,10))
plt.pie(comes,labels=countries);
plt.figure(figsize=(10,6))
sns.countplot(df["market_segment"],hue="is_canceled",data=df);
numeric_f= ["lead_time","arrival_date_day_of_month","stays_in_weekend_nights","stays_in_week_nights",
            "adults","children","babies","is_repeated_guest", "previous_cancellations","previous_bookings_not_canceled",
            "required_car_parking_spaces", "total_of_special_requests", "adr"]

categorical_f= ["hotel","arrival_date_month","meal","market_segment",
                "distribution_channel","reserved_room_type","deposit_type","customer_type"]

all_cols = numeric_f + categorical_f +["is_canceled"]
df_new=df[all_cols]
df=pd.get_dummies(df_new,columns=categorical_f,drop_first=True)
X=df.drop("is_canceled",axis=1).values
y=df["is_canceled"].values
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
scaler=MinMaxScaler()

X_train_sc=scaler.fit_transform(X_train)
X_test_sc=scaler.transform(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score , classification_report 
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression(solver="lbfgs",max_iter=1000)
log_reg.fit(X_train_sc,y_train)
log_predict=log_reg.predict(X_test_sc)
print("Accuracy score of Logistic Regression :",accuracy_score(y_test,log_predict))
print("*"*100)
print("\nConfusion Matrix of Logistic Regression :\n",confusion_matrix(y_test,log_predict))
print("*"*100)
print("\nClassification report of Logistic Regression : \n\n",classification_report(y_test,log_predict))
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_sc,y_train)
knn_predict=knn.predict(X_test_sc)
print("Accuracy score of KNN :",accuracy_score(y_test,knn_predict))
print("*"*100)
print("\nConfusion Matrix of KNN :\n",confusion_matrix(y_test,knn_predict))
print("*"*100)
print("\nClassification report of KNN : \n\n",classification_report(y_test,knn_predict))
import tensorflow as tf
X_train.shape
model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units=60,activation="relu"))
model.add(tf.keras.layers.Dropout(rate=0.2))

model.add(tf.keras.layers.Dense(units=30,activation="relu"))
model.add(tf.keras.layers.Dropout(rate=0.2))

model.add(tf.keras.layers.Dense(units=15,activation="relu"))
model.add(tf.keras.layers.Dropout(rate=0.2))

model.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

model.compile(optimizer="adam",loss="binary_crossentropy")
model.fit(X_train_sc,y_train,validation_data=(X_test_sc,y_test),epochs=250)
losses=pd.DataFrame(model.history.history)
ann_predict=model.predict_classes(X_test_sc)
losses.plot();
print("Accuracy score of 4Layer Neural Network :",accuracy_score(y_test,ann_predict))
print("*"*100)
print("\nConfusion Matrix of 4Layer Neural Network :\n",confusion_matrix(y_test,ann_predict))
print("*"*100)
print("\nClassification report of 4Layer Neural Network : \n\n",classification_report(y_test,ann_predict))
