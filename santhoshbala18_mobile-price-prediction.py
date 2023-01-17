# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df.head()
train_df.describe()
train_df.info()
#Confirms that there is no null entry
train_df['price_range'].unique()
#There are 4 price ranges 0,1,2,3 --> Multilabel Classification problem
import matplotlib.pyplot as plt
import seaborn as sns
#For Data visualization
plt.hist(train_df['battery_power'])
plt.show()
#Low power batteries are slightly more in count
sns.countplot(train_df['price_range'])
plt.show()
##Data is split equally across all ranges
sns.boxplot(train_df['price_range'],train_df['clock_speed'])
#Variance of clock speed is slightly more for mobiles in Category '0'
sns.countplot(train_df['dual_sim'])
plt.show()
##Slightly more number of phones have dual sim
sns.boxplot(train_df['dual_sim'],train_df['price_range'])
#Price Range of dual sim phones are considerably higher. This Denotes that Dual sim plays an important role in classification
plt.hist(train_df['fc'])
plt.show()
sns.boxplot(train_df['fc'],train_df['price_range'])
plt.show()
#This Shows price range and fc have less correlation
sns.boxplot(train_df['four_g'],train_df['price_range'])
plt.show()
#Price Range of 4G phones are considerably higher. This Denotes that 4G plays an important role in classification
plt.hist(train_df['int_memory'])
plt.show()
plt.scatter(train_df['price_range'],train_df['int_memory'])
plt.show()
plt.hist(train_df['mobile_wt'])
plt.show()
##Almost evenly spread across data set
train_df['n_cores'].unique()
sns.boxplot(train_df['n_cores'],train_df['price_range'])
plt.show()
train_df.loc[(train_df['price_range']==0) & (train_df['n_cores']==8)]['n_cores'].count()
#67 mobiles in Price range of 0 is having 8 Cores
len(train_df['ram'].unique())
##Too many different values for RAM
sns.boxplot(train_df['sc_h'],train_df['price_range'])
plt.show()
#Some screen sizes are in high price range
plt.hist(train_df['sc_w'])
plt.show()
#Width ranges mostly in 0-7
sns.boxplot(train_df['talk_time'],train_df['price_range'])
plt.show()
sns.boxplot(train_df['three_g'],train_df['price_range'])
plt.show()
#Price Range of 3G phones are considerably higher. This Denotes that 3G plays an important role in classification
sns.boxplot(train_df['touch_screen'],train_df['price_range'])
plt.show()
#Price Range of touch screen phone is low.. quite strange considering all the 4G,3G and Wifi phones are in higher price range
sns.boxplot(train_df['wifi'],train_df['price_range'])
plt.show()
#Price Range of wifi phones are considerably higher. This Denotes that wifi plays an important role in classification
corr = train_df.corr()
sns.heatmap(corr,cmap='YlGnBu',vmin=-1,vmax=1)
test_df.columns
#Since all the features are in different range, preprocessing scalar is applied
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X = train_df.drop('price_range',axis=1)
y = train_df['price_range']

scaler.fit(X)
X_transformed = scaler.transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_transformed,y,test_size=0.3)


#Simple Logistic regression model with scaled features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train,y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
#Simple Logistic Regression model with all the feature produces accuracy of 83%


scaler = StandardScaler()
#Selecting only the important columns based on above visualizations
X = train_df[['battery_power','blue','dual_sim','four_g','px_height','px_width','ram','touch_screen','wifi','fc']]
y = train_df['price_range']

scaler.fit(X)
X_transformed = scaler.transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_transformed,y,test_size=0.3)
from sklearn.metrics import confusion_matrix
model = LogisticRegression()
model.fit(X_train,y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))
#Simple Logistic Regression model with only the important features got more accuracy


from sklearn.svm import SVC
#SVC classifier
model = SVC()
model.fit(X_train,y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))

#SVC model with only the important features got more accuracy


from sklearn.tree import DecisionTreeClassifier
#Decision Tree Classifier classifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))

#As it is Seen,Decision tree overfits the data, resulting in pretty low test accuracy


from sklearn.ensemble import GradientBoostingClassifier
#Gradient Boosting Classifier classifier
model = GradientBoostingClassifier()
model.fit(X_train,y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))

#Boosting Slightly reduces overfitting
