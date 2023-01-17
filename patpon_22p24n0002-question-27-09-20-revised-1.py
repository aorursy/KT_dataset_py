import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
import math
import math
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/train.csv", index_col='PassengerId')
data
head = data[:10]

print("len(data) = ", len(data)) # ดูจำนวนแถวของข้อมูล (ในที่นี้คือมีผู้โดยสารกี่คน)
print("data.shape = ", data.shape) # ดูจำนวนแถวและคอลัมน์ของข้อมูล
# ดูข้อมูลของผู้โดยสารที่ PassengerId == 4
print(data.loc[4])
# ดูข้อมูลคอลัมน์ Age
ages = data["Age"] # หรือจะใช้ data.Age ก็ได้
print(ages[:10])
type(data["Age"])
# เลือกดูข้อมูลจากทั้งแถวและคอลัมน์
data.loc[5:10, ("Fare", "Pclass")] # หรือจะใช้ data[["Fare","Pclass"]].loc[5:10] ก็ได้
data.describe()
data.iloc[5]
data['Age'] = data['Age'].fillna(value=data['Age'].mean())
data['Fare'] = data['Fare'].fillna(value=data['Fare'].mean())
data.iloc[5] # ข้อมูลที่ทำการแทนที่ค่า NaN แล้ว

print(data.loc[13, 'Survived'])
print(data.loc[666, 'Survived'])
# [เฉลย]
data['Survived'].mean() # หรือ data.describe().loc['mean', 'Survived']
# หรือจะใช้ len(data[data['Survived']==1])/len(data) ก็ได้
#ค้นหาโดยใช้ชื่อ
names = ["Margaret Brown", # มอลลีผู้ไม่มีวันจม
         "Thomas Andrews", # วิศวกรอาวุโส ผู้ออกแบบและควบคุมการต่อเรือไททานิค
         "Madeleine Force", # ภรรยาของ John Jacob Astor ซึ่งเป็นผู้ที่มีฐานะร่ำรวยที่สุดในเรือ
         "Cosmo Duff-Gordon" # ท่านบารอนเน็ตที่โดนโจมตีว่าแย่งผู้หญิงและเด็กขึ้นเรือชูชีพก่อน
        ]

import re
characters = pd.DataFrame()

for name in names:
    condition = True
    for w in re.split('\W+', name):
        condition &= data['Name'].str.contains(w)        
    characters = characters.append(data[condition])

characters
# หาคนที่จ่ายค่าตั๋วสูงที่สุด
print("Max ticket price: ", np.max(data["Fare"]))
print("\nThe guy who paid the most:\n", data.loc[np.argmax(data["Fare"])])
# [เฉลย]
print("Max passenger age: ", np.max(data["Age"]))
print("\nThe oldest guy on the ship:\n", data.loc[np.argmax(data["Age"])])
# [เฉลย]

mean_fare_men = np.mean(data[data['Sex']=='male']['Fare']) # หรือ data[data['Sex']=='male']['Fare'].mean()
mean_fare_women = np.mean(data[data['Sex']=='female']['Fare']) # หรือ data[data['Sex']=='female']['Fare'].mean()

print(mean_fare_men, mean_fare_women)
# [แบบฝึกหัด] ระหว่าง เด็ก (อายุน้อยกว่า 18 ปี) กับ ผู้ใหญ่ ใครมีโอกาสรอดชีวิตมากกว่ากัน



# [เฉลย]

child_survival_rate = np.mean(data[data['Age']<18]['Survived']) # หรือ data[data['Age']<18]['Survived'].mean()
adult_survival_rate = np.mean(data[data['Age']>=18]['Survived']) # หรือ data[data['Age']>=18]['Survived'].mean()

print(child_survival_rate, adult_survival_rate)
plt.hist(data['Age'])
plt.show()

plt.hist(data['Fare'], bins=50)
plt.show()

plt.hist2d(data['Age'], data['Fare'])
plt.show()
# [แบบฝึกหัด] ทำ scatter plot แสดงอายุและค่าโดยสาร ของผู้โดยสารแต่ละคน



# [โบนัส] ขอคารวะหากท่านแยกสีจุดของผู้โดยสารหญิงและชายด้วย



# [เฉลย]
plt.scatter(data['Age'], data['Fare'])
plt.show()

plt.scatter(data[data['Sex']=='male']['Age'], data[data['Sex']=='male']['Fare'], c='skyblue')
plt.scatter(data[data['Sex']=='female']['Age'], data[data['Sex']=='female']['Fare'], c='pink')
plt.show()
# ทำการเรียนรู้โดยใช้ random forest และกันผู้โดยสารร้อยคนสุดท้ายไว้ใช้สำหรับทดสอบ

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

features = data[["Fare", "SibSp"]].copy()
answers = data["Survived"]

model = RandomForestClassifier(n_estimators=100)
model.fit(features[:-100], answers[:-100])

test_predictions = model.predict(features[-100:])
print("Test accuracy:", accuracy_score(answers[-100:], test_predictions))
# [แบบฝึกหัด] ลองเพิ่ม feature แล้วทำให้ test accuracy ได้อย่างน้อย 0.8
# (บาง feature เช่น Sex ต้องมีการแปลงเป็นตัวเลขก่อน เช่น 1 เป็นผู้ชาย 0 เป็นผู้หญิง)
# (เมื่อทำการเรียนรู้แล้ว สามารถใช้ model.feature_importances_ เพื่อดูว่า feature แต่ละตัวมีความสำคัญแค่ไหนบ้าง)





from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

features = data[["Fare", "SibSp", "Parch", "Pclass", "Age", "Sex"]].copy().replace({'male': 1, 'female': 0})
answers = data["Survived"]

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(features[:-100], answers[:-100])

test_predictions = rf_model.predict(features[-100:])
print("Test accuracy:", accuracy_score(answers[-100:], test_predictions))

rf_model.feature_importances_
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz

dt_model = DecisionTreeClassifier(max_leaf_nodes=5)
dt_model.fit(features[:-100], answers[:-100])

test_predictions = dt_model.predict(features[-100:])
print("Test accuracy:", accuracy_score(answers[-100:], test_predictions))
dot_data = export_graphviz(dt_model, out_file=None, 
                           feature_names=["Fare", "SibSp", "Parch", "Pclass", "Age", "Sex"], 
                           class_names=["Perished", "Survived"], 
                           filled=True, rounded=True, special_characters=True)  
graphviz.Source(dot_data)
from sklearn.metrics import accuracy_score
accuracy_score(answers[-100:], test_predictions)
#answer[-100:] = y_test 
#test_predictions = answer
from sklearn.metrics import precision_score
precision_score(answers[-100:], test_predictions)
from sklearn.metrics import recall_score
recall_score(answers[-100:], test_predictions)
from sklearn.metrics import f1_score
f1_score(answers[-100:], test_predictions)
from sklearn.metrics import classification_report
print(classification_report(answers[-100:], test_predictions))
