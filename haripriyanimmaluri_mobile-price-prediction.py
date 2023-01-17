# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020/train_data.csv")
test_df = pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020/test_data.csv")
sample_submission=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020/sample_submission.csv")
train_df.head()
train_df.describe()
train_df.info()
train_df['price_range'].unique()

import matplotlib.pyplot as plt
import seaborn as sns
plt.hist(train_df['battery_power'])
plt.show()
sns.countplot(train_df['price_range'])
plt.show()
sns.boxplot(train_df['price_range'],train_df['clock_speed'])
sns.countplot(train_df['dual_sim'])
plt.show()
sns.boxplot(train_df['dual_sim'],train_df['price_range'])
plt.hist(train_df['fc'])
plt.show()
sns.boxplot(train_df['fc'],train_df['price_range'])
plt.show()
sns.boxplot(train_df['four_g'],train_df['price_range'])
plt.show()
plt.hist(train_df['int_memory'])
plt.show()
plt.scatter(train_df['price_range'],train_df['int_memory'])
plt.show()
plt.hist(train_df['mobile_wt'])
plt.show()
train_df['n_cores'].unique()
sns.boxplot(train_df['n_cores'],train_df['price_range'])
plt.show()
train_df.loc[(train_df['price_range']==0) & (train_df['n_cores']==8)]['n_cores'].count()
len(train_df['ram'].unique())
sns.boxplot(train_df['sc_h'],train_df['price_range'])
plt.show()
plt.hist(train_df['sc_w'])
plt.show()
sns.boxplot(train_df['talk_time'],train_df['price_range'])
plt.show()
sns.boxplot(train_df['three_g'],train_df['price_range'])
plt.show()
sns.boxplot(train_df['touch_screen'],train_df['price_range'])
plt.show()
sns.boxplot(train_df['wifi'],train_df['price_range'])
plt.show()
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
scaler = StandardScaler()
#Selecting only the important columns based on above visualizations
X = train_df[['battery_power','bluetooth','dual_sim','four_g','px_height','px_width','ram','touch_screen','wifi','fc']]
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
from sklearn.svm import SVC
#SVC classifier
model = SVC()
model.fit(X_train,y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))

from sklearn.tree import DecisionTreeClassifier
#Decision Tree Classifier classifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))
from sklearn.ensemble import GradientBoostingClassifier
#Gradient Boosting Classifier classifier
model = GradientBoostingClassifier()
model.fit(X_train,y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))
data={'id':sample_submission['id'],'price_range':sample_submission['price_range']}
result=pd.DataFrame(data)
result.to_csv('/kaggle/working/result_svc.csv',index=False)
output=pd.read_csv('/kaggle/working/result_svc.csv')
print(output)