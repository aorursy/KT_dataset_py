# นำเข้า libary ที่จะใช้งาน
import pandas as pd
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import svm
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# โหลดขอ้มูล
train_data = pd.read_csv('../input/titanic/train.csv')
# แสดงตัวอย่างข้อมูล
train_data.head()
# เขียน function เพื่อแปลงข้อความระบุเพศเป็นตัวเลข (male เป็น 1 และ female เป็น 0)
def Sex_to_Bi(data):
    sex = []
    for i in data:
        if i == 'male':
            sex.append(1)
        elif i == 'female':
            sex.append(0)
        else:
            sex.append(NaN)
    return sex

sex_train = Sex_to_Bi(train_data['Sex'])

print(sex_train)
# สร้างคอลัมน์ใหม่ที่ใช้ตัวเลขระบุเพศแทนข้อความ
train_data['Sexes'] = sex_train

train_data.head()
# ตัดคอลัมน์ที่ไม่จำเป็นออก 
train_data = train_data.drop(columns=['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'])

train_data.head()
print(train_data.shape)

# ตัดแถวที่มีข้อมูลไม่ครบถ้วนออก
train_data = train_data.dropna(axis=0)
print(train_data.shape)
# ตรวจเช็ครายคอลัมน์อีกครั้ง
print(train_data[train_data['Survived'].isnull()])
# แบ่งส่วนที่จะเป็นข้อมูล และlabel แยกออกจากกัน (โดยจะให้ Survived เป็น label)
train_data_X = train_data.drop(columns=['Survived'])
train_data_Y = train_data.drop(columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sexes'])

train_data_Y.head()
# ทำการสร้างโมเดล Decision Tree
dt_model = DecisionTreeClassifier(max_leaf_nodes=5)
dt_model.fit(train_data_X, train_data_Y)
# แสดงกราฟ Decision Tree
dot_data = export_graphviz(dt_model, out_file=None, 
                           feature_names=["Pclass", "Age", "SibSp", "Parch", "Fare", "Sexes"], 
                           class_names=["Perished", "Survived"], 
                           filled=True, rounded=True, special_characters=True)

graphviz.Source(dot_data)
# แบ่งข้อมูลออกเป็นชุดเรียนรู้และชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(train_data_X, train_data_Y, test_size = 0.2)
# สร้างโมเดล Naive Bays
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
# ทดสอบโมเดล Naive Bays ที่ได้
nb_model.score(X_test, y_test)
# แปลงข้อมูลท้ังหมดให้อยู่ในรูปของ array
x_train = np.array(X_train)
x_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
# ทำการ normalize ข้อมูล
def normalize(dataset):
    mean = np.mean(dataset, axis = 0)
    stddev = np.std(dataset, axis = 0)
    return (dataset - mean)/stddev

x_train = normalize(x_train)
x_test = normalize(x_test)
# สร้างโมเดล LinearRegression
class LinearRegression(tf.keras.Model):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.dense = tf.keras.layers.Dense(1)
        
    def call(self, x):
        return self.dense(x)
# สร้าง Neural Network
ne_model = LinearRegression()
ne_model.compile(loss = 'mse', optimizer = 'adam')
ne_model.fit(x_train, y_train, epochs=1000, verbose = 0)
# ประเมินประสิทธิภาพโดยใช้ข้อมูลชุดเรียนรู้มาทดสอบ
ne_model.evaluate(x_train, y_train)
# ประเมินประสิทธิภาพโดยใช้ข้อมูลชุดทดสอบมาทดสอบ
ne_model.evaluate(x_test, y_test)
# แบ่งุดข้อมูลเป็น 5-fold cross validation
kf = KFold(5)
data = []

for train in kf.split(x_train):
    data.append(train)

print(len(data))
# สร้างโมเดล SVM แบบ linear และสั่งเทรนโมเดล
svm_li = svm.SVC(kernel = 'linear').fit(x_train, y_train)
svm_predicted = svm_li.predict(x_test)
confusion = confusion_matrix(y_test, svm_predicted)
df = pd.DataFrame(confusion)
# แสดง Confusion Matrix ของโมเดล SVM แบบ linear
plt.figure(figsize = (5.5, 4))
sns.heatmap(df, annot = True)
plt.title('SVM Linear Kernel \nAccuracy: {0:.3f}'
         .format(accuracy_score(y_test, svm_predicted)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
'''
แสดงค่า 
recall แต่ละ class
precision ของแต่ละ class
f-measure (ในที่นี้คือ f1-score) ของแต่ละ class
average f-measure ของทั้งชุดข้อมูล
'''
print(classification_report(y_test, svm_predicted))
# สร้างโมเดล SVM แบบ rbf และสั่งเทรนโมเดล
svm_rbf = svm.SVC(kernel = 'rbf').fit(x_train, y_train)
svm_predicted = svm_rbf.predict(x_test)
confusion = confusion_matrix(y_test, svm_predicted)
df = pd.DataFrame(confusion)
# แสดง Confusion Matrix ของโมเดล SVM แบบ rbf
plt.figure(figsize = (5.5,4))
sns.heatmap(df, annot =True)
plt.title('SVM RBF Kernel \nAccuracy: {0:.3f}'
         .format(accuracy_score(y_test, svm_predicted)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
'''
แสดงค่า 
recall แต่ละ class
precision ของแต่ละ class
f-measure (ในที่นี้คือ f1-score) ของแต่ละ class
average f-measure ของทั้งชุดข้อมูล
'''
print(classification_report(y_test, svm_predicted))