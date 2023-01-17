a = 5
print(a * 2)
# รันอันนี้ก่อน

import math
# [แบบฝึกหัด]

# ลองพิมพ์คำว่า math.a

# แล้วหาฟังก์ชันที่คำนวณ arctangent จาก parameter สองตัว (จะมีเลข 2 อยู่ข้างท้ายฟังก์ชัน)

# จากนั้นก็แสดง docstring ของฟังก์ชันนี้ขึ้นมาดู





import pandas as pd

data = pd.read_csv("../input/train.csv", index_col='PassengerId') # บรรทัดนี้ทำการอ่านข้อมูล และให้ output ออกมาเป็น pandas.DataFrame
# ทำการเลือกแถว 

head = data[:10]



head # ถ้าพิมพ์ expression ที่ข้างท้ายของเซลล์ Jupyter Notebook จะแสดงค่าของ expression นั้นออกมา
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
# [แบบฝึกหัด] เลือกผู้โดยสารหมายเลข 13 และ 666 แล้วดูว่ารอดชีวิตหรือไม่





# [เฉลย]

print(data.loc[13, 'Survived'])

print(data.loc[666, 'Survived'])
# [แบบฝึกหัด] คำนวณอัตราการรอดชีวิตของผู้โดยสาร





# [เฉลย]

data['Survived'].mean() # หรือ data.describe().loc['mean', 'Survived']

# หรือจะใช้ len(data[data['Survived']==1])/len(data) ก็ได้
# ค้นหาโดยใช้ชื่อ

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
import numpy as np



# ทดลองสร้าง numpy.ndarray ขึ้นมา

a = np.array([1, 2, 3, 4, 5])

b = np.array([5, 4, 3, 2, 1])

print("a = ", a)

print("b = ", b)



# สามารถใช้ operation ทางคณิตศาสตร์กับตรรกศาสตร์กับ array ได้ โดยถ้าเป็น array กับ scalar ก็จะทำ operation นั้น กับทุก element ใน array

print("a + 1 =", a + 1)

print("a * 2 =", a * 2)

print("a == 2", a == 2)

# ... หรือจะเป็น array กับ array

print("a + b =", a + b)

print("a * b =", a * b)
# [แบบฝึกหัด] คำนวณครึ่งนึงของผลคูณระหว่างแต่ละ element ของ a และ b





# [เฉลย]

a*b/2
# [แบบฝึกหัด] คำนวณแต่ละ element ของ a ยกกำลังสอง แล้วหารด้วย (b บวก 1)





# [เฉลย]

a**2/(b+1)
%%time 



# Option I: ใช้ Python เพียวๆ

arr_1 = range(1000000)

arr_2 = range(99,1000099)



a_sum = []

a_prod = []

sqrt_a1 = []

for i in range(len(arr_1)):

    a_sum.append(arr_1[i]+arr_2[i])

    a_prod.append(arr_1[i]*arr_2[i])

    a_sum.append(arr_1[i]**0.5)

    

arr_1_sum = sum(arr_1)

%%time



# Option II: เริ่มจาก Python แล้วแปลงเป็น Numpy

arr_1 = range(1000000)

arr_2 = range(99,1000099)



arr_1, arr_2 = np.array(arr_1), np.array(arr_2)



a_sum = arr_1 + arr_2

a_prod = arr_1 * arr_2

sqrt_a1 = arr_1 ** .5

arr_1_sum = arr_1.sum()

%%time



# Option III: ใช้ Numpy เพียวๆ

arr_1 = np.arange(1000000)

arr_2 = np.arange(99,1000099)



a_sum = arr_1 + arr_2

a_prod = arr_1 * arr_2

sqrt_a1 = arr_1 ** .5

arr_1_sum = arr_1.sum()

a = np.array([1, 2, 3, 4, 5])

b = np.array([5, 4, 3, 2, 1])

c = ['male', 'male', 'female', 'female', 'male']



print("numpy.sum(a) = ", np.sum(a)) # คำนวณผลรวมทั้งหมดใน array

print("numpy.mean(a) = ", np.mean(a)) # คำนวณค่าเฉลี่ยใน array

print("numpy.min(a) = ",  np.min(a)) # หาค่าน้อยที่สุด

print("numpy.argmin(b) = ", np.argmin(b)) # หา index ของค่าน้อยที่สุด

print("numpy.dot(a, b) = ", np.dot(a, b)) # คำนวณ dot product

print("numpy.unique() = ", np.unique(c)) # ตัดค่าที่ซ้ำออก



# ดูเพิ่มได้ที่ http://bit.ly/2u5q430
# หาคนที่จ่ายค่าตั๋วสูงที่สุด

print("Max ticket price: ", np.max(data["Fare"]))

print("\nThe guy who paid the most:\n", data.loc[np.argmax(data["Fare"])])
# [แบบฝึกหัด] จงหาผู้โดยสารที่อายุเยอะที่สุดในเรือ





# [เฉลย]

print("Max passenger age: ", np.max(data["Age"]))

print("\nThe oldest guy on the ship:\n", data.loc[np.argmax(data["Age"])])
print("Boolean operations")



print('a = ', a)

print('b = ', b)

print("a > 2", a > 2)

print("numpy.logical_not(a>2) = ", np.logical_not(a>2))

print("numpy.logical_and(a>2,b>2) = ", np.logical_and(a > 2,b > 2))

print("numpy.logical_or(a>4,b<3) = ", np.logical_or(a > 2, b < 3))



print("\n shortcuts")

print("~(a > 2) = ", ~(a > 2))                    # logical_not(a > 2)

print("(a > 2) & (b > 2) = ", (a > 2) & (b > 2))  # logical_and

print("(a > 2) | (b < 3) = ", (a > 2) | (b < 3))  # logical_or
a = np.array([0, 1, 4, 9, 16, 25])

print("a = ", a)



ix = np.array([1, 2, 3])

print("Select by element index") # เลือกโดยใช้ array ของ index

print("a[[1, 2, 3]] = ", a[ix])



print("\nSelect by boolean mask") # เลือกโดยใช้ boolean mask

print("a[a > 5] = ", a[a > 5]) # เลือกเฉพาะตัวที่มากกว่า 5

print("(a % 2 == 0) =", a % 2 == 0) # ทำให้ได้ค่า True สำหรับเลขคู่ False สำหรับเลขคี่

print("a[a % 2 == 0] =", a[a % 2 == 0]) # เลือกมาเฉพาะเลขคู่



# สามารถใช้ boolean mask กับ Pandas ได้เช่นกัน

print("data[(data['Age'] < 18) & (data['Sex'] == 'male')] = (below)") # เลือกเฉพาะเด็กผู้ชายที่อายุต่ำกว่า 18 ปี

data[(data['Age'] < 18) & (data['Sex'] == 'male')]
# [แบบฝึกหัด] โดยเฉลี่ยแล้วผู้หญิงหรือผู้ชายจ่ายค่าโดยสารมากกว่ากัน





# [เฉลย]



mean_fare_men = np.mean(data[data['Sex']=='male']['Fare']) # หรือ data[data['Sex']=='male']['Fare'].mean()

mean_fare_women = np.mean(data[data['Sex']=='female']['Fare']) # หรือ data[data['Sex']=='female']['Fare'].mean()



print(mean_fare_men, mean_fare_women)
# [แบบฝึกหัด] ระหว่าง เด็ก (อายุน้อยกว่า 18 ปี) กับ ผู้ใหญ่ ใครมีโอกาสรอดชีวิตมากกว่ากัน





# [เฉลย]



child_survival_rate = np.mean(data[data['Age']<18]['Survived']) # หรือ data[data['Age']<18]['Survived'].mean()

adult_survival_rate = np.mean(data[data['Age']>=18]['Survived']) # หรือ data[data['Age']>=18]['Survived'].mean()



print(child_survival_rate, adult_survival_rate)
import matplotlib.pyplot as plt

%matplotlib inline



# line plot

plt.plot([0,1,2,3,4,5], [0,1,4,9,16,25]) # array แรก คือพิกัดของจุดในแกน X และ array หลัง คือพิกัดของจุดในแกน Y
# scatter plot

plt.scatter([0,1,2,3,4,5], [0,1,4,9,16,25])



plt.show() # โชว์พล็อต
# สามารถใส่สีและรูป marker ได้

plt.scatter([1,1,2,3,4,4.5], [3,2,2,5,15,24],

            c = ["red","blue","orange","green","cyan","gray"], marker = "x")



# ถ้าไม่สั่ง show() เสียก่อน ก็จะพล็อตอย่างอื่นลงรูปเดียวกันได้

plt.plot([0,1,2,3,4,5], [0,1,4,9,16,25], c = "black")



# ใส่ข้อความเข้าไปเสียหน่อย

plt.title("Conspiracy theory proven!!!")

plt.xlabel("Per capita alcohol consumption")

plt.ylabel("# Layers in the state of the art image classifier")
# จากข้อมูลดิบ สามารถนำมาทำ histogram ได้

plt.hist([0,1,1,1,2,2,3,3,3,3,3,4,4,5,5,5,6,7,7,8,9,10])

plt.show()



plt.hist([0,1,1,1,2,2,3,3,3,3,3,4,4,5,5,5,6,7,7,8,9,10], bins = 5)
# [แบบฝึกหัด] พล็อต histogram ของอายุผู้โดยสาร และ histogram ของค่าโดยสาร







# [โบนัส] ลองพล็อต histogram สองมิติของอายุและค่าโดยสาร (สามารถใช้ Tab เพื่อหาฟังก์ชันที่ต้องการได้)





# [เฉลย]



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





# [เฉลย]



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