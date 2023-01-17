import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# reading the dataset
data = pd.read_csv("../input/diamonds.csv")
#เอาค่าจาก Data มาเช็ค Type
data.dtypes 
#ตรวจสอบรูปเเบบของข้อมูล
data.info()
#แสดงตารางออกมา 10 อันดับ
data.head(10)
#อธิบาย Data
data.describe()
#กำหนดตัวแปร
data = data.drop(data.loc[data.x <= 0].index)
data = data.drop(data.loc[data.y <= 0].index)
data = data.drop(data.loc[data.z <= 0].index)
data["ratio"] = data.x / data.y
premium = ["D","E","F","G","H"]

# I,J ให้เป็น normal

# ฟังก์ชั่นแบ่งกลุ่มเพชร
def data_split(status):
    if status in premium:
        return 'premium'
    else:
        return 'normal'

# ฟังก์ชั่นแบ่งกลุ่มของลูกค้าเป็นตัวเลข
def data_split_num(status):
    if status in premium:
        return 1
    else:
        return 0
# สร้าง column และเรียกใช้ฟังก์ชั่น
data['data_split'] = data['color'].apply(data_split)
data['data_split_num'] = data['color'].apply(data_split_num)
data.head()
#correlation matrix for 15 variables with largest correlation
corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
k = 8 #number of variables for heatmap
cols = corrmat.nlargest(k, 'price')['price'].index
cm = np.corrcoef(data[cols].values.T)

# Generate a mask for the upper triangle
mask = np.zeros_like(cm, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


hm = sns.heatmap(cm, vmax=1, mask=mask, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
print(" Diamond Carat = " + str(np.mean(data.carat)))
plt.subplots(figsize=(10,7))
sns.distplot(data.carat)
plt.show()
sns.countplot(y = data.cut)
plt.show()
print(" Diamond Depth Value = " + str(np.mean(data.depth)))
plt.subplots(figsize=(10,7))
sns.distplot(data.depth)
plt.show()
plt.subplots(figsize=(10,7))
sns.countplot(data.color)
plt.show()
from collections import Counter
plt.pie(list(dict(Counter(data.color)).values()),
        labels = list(dict(Counter(data.color)).keys()),
        shadow = True,
        startangle = 0,
        explode = (0.1,0.1,0.1,0.1,0.1,0.1, 0.1));
plt.legend(list(dict(Counter(data.color)).keys()),loc = 2, bbox_to_anchor=(1.1, 1))
plt.show()
sns.countplot(data.clarity)
plt.show()
plt.pie(list(dict(Counter(data.clarity)).values()),
        labels = list(dict(Counter(data.clarity)).keys()),
        shadow = True,
        startangle = 0);
plt.legend(list(dict(Counter(data.clarity)).keys()),loc = 2, bbox_to_anchor=(1.1, 1))
plt.show()
print("Mean Diamond Table Value = " + str(np.mean(data.table)))
plt.subplots(figsize=(10,7))
sns.distplot(data.table)
plt.show()
#กราฟราคา
plt.subplots(figsize=(15,7))
sns.distplot(data.price)
plt.show()
# คัดเลือก feature ที่จำเป็นต่อการทำโมเดล
feature_selected = ['carat', 'cut', 'color', 'clarity', 'price','data_split','data_split_num']
data = data[feature_selected]
data.head()
# Check correlations
sns.heatmap(data.corr(), annot=True)
# check missing value อีกรอบเพื่อความชัว
data.isnull().any()
# แปลง categories ในเป็นตัวเลขโดยการทำ one hot encoding 
categories = ['carat', 'cut', 'color', 'clarity', 'price']
raw_model_data = pd.get_dummies(data.copy(), columns=categories,drop_first=True)
# กำหนด feature ของ X และ y
X = raw_model_data.drop(columns=['data_split', 'data_split_num'],axis=1)
y = raw_model_data['data_split']

# แบ่ง X_train, X_test, y_train, y_test และกำหนดค่า randon_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)
# กำหนดค่าของ DecisionTree 
decision_tree = DecisionTreeClassifier()
# ทำการ Train Model 
decision_tree.fit(X_train, y_train)
# ทำการ predict 
y_pred = decision_tree.predict(X_test)
# ดูค่า accuracy ของ model
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_decision_tree
acc_decision_tree = round(accuracy_score(y_test, y_pred) * 100, 2)
acc_decision_tree
# กำหนดค่าของ KNN 
knn = KNeighborsClassifier(n_neighbors = 3)
# ทำการ train model
knn.fit(X_train, y_train)
# ทำการ predict 
y_pred = knn.predict(X_test)
# ดูค่า accuracy
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
acc_knn
acc_knn = round(accuracy_score(y_test, y_pred) * 100, 2)
acc_knn
# กำหนดค่าของ Gaussian Naive Bayes
gaussian = GaussianNB()
# ทำการ train model
gaussian.fit(X_train, y_train)
# ทำการ predict 
y_pred = gaussian.predict(X_test)
# ดูค่า accuracy
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
acc_gaussian
acc_gaussian = round(accuracy_score(y_test, y_pred) * 100, 2)
acc_gaussian
perceptron = Perceptron()
# ทำการ train model
perceptron.fit(X_train, y_train)
# ทำการ predict
Y_pred = perceptron.predict(X_test)
# ดูค่า accuracy
acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
acc_perceptron
acc_perceptron = round(accuracy_score(y_test, y_pred) * 100, 2)
acc_perceptron
# กำหนดค่าของ Stochastic Gradient Descent
sgd = SGDClassifier()
# ทำการ train model
sgd.fit(X_train, y_train)
# ทำการ predict
y_pred = sgd.predict(X_test)
# ดูค่า accuracy
acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
acc_sgd
acc_sgd = round(accuracy_score(y_test, y_pred) * 100, 2)
acc_sgd
# สร้าง dataframe เพื่อเก็บข้อมูล Score
models = pd.DataFrame({
    'Model': ['KNN', 'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Decision Tree'],
    'Score': [acc_knn, acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd,  acc_decision_tree]})

# แสดงผลออกมาโดยเรียงจากมากไปน้อย
models.sort_values(by='Score', ascending=False)
print("X_value = %s, Predicted=%s" % (X_test, y_pred[1]))