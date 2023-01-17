# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import Libraries

import sys

!{sys.executable} -m pip install pydotplus

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from IPython.display import Image 

import pydotplus



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



%matplotlib inline
# ดึงข้อมูลจากไฟล์มาเก็บไว้ในตัวแปร loan

loan = pd.read_csv("../input/loan-dataset/loan.csv")



# ก็อปปี้ข้อมูลเผื่อไว้

original_loan = loan.copy()

loan.head()
# ตรวจสอบรูปแบบของข้อมูล

loan.info()
# ดูข้อมูล column 'issue_d'

loan['issue_d'].head()
# แปลงค่า column 'issue_d' เป็นปี

dt_series = pd.to_datetime(loan['issue_d'])

loan['year'] = dt_series.dt.year

loan['year'].head()
# ลบสถานะ Current ออกจาก loan_staus และตรวจสอบว่ามีสถานะใดบ้าง

loan = loan[loan['loan_status']!='Current']

loan['loan_status'].value_counts()
# กำหนดสถานะผู้กู้ที่ไม่ดี

bad_loan = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off", "In Grace Period", 

            "Late (16-30 days)", "Late (31-120 days)"]

# ฟังก์ชั่นแบ่งกลุ่มของผู้กู้เป็น catagory

def loan_condition(status):

    if status in bad_loan:

        return 'Bad'

    else:

        return 'Good'

# ฟังก์ชั่นแบ่งกลุ่มของผู้กู้เป็นตัวเลข  

def loan_condition_num(status):

    if status in bad_loan:

        return 1

    else:

        return 0

# สร้าง column และเรียกใช้ฟังก์ชั่น

loan['loan_condition'] = loan['loan_status'].apply(loan_condition)

loan['loan_condition_num'] = loan['loan_status'].apply(loan_condition_num)

loan.head()
# plot ดูจำนวนข้อมูลให้แต่ละปี

sns.countplot(x="year", data=loan);
# เลือกข้อมูลที่จะใช้ในการวิเคราะห์และทำโมเดล

loan = loan[loan['year']<=2012]

loan['year'].value_counts()
# plot ดูค่าเฉลี่ยของเงินที่ผู้กู้ยืมไปในแต่ละปี

plt.figure(figsize=(12,8))

sns.barplot('year', 'loan_amnt', data=loan, palette='tab10', ci=None)

plt.title('Issuance of Loans', fontsize=16)

plt.xlabel('Year', fontsize=14)

plt.ylabel('Average loan amount issued', fontsize=14)
# plot เพื่อดูสัดส่วนสถานะของผู้กู้ยืมทั้งหมด



colors = ["#3791D7", "#D72626"]

labels ="Good Loans", "Bad Loans"



plt.suptitle('Information on Loan Conditions', fontsize=20)



loan["loan_condition"].value_counts().plot.pie(autopct='%1.2f%%',  shadow=True, colors=colors, 

                                             labels=labels, fontsize=12, startangle=70)



# plot เพื่อดูจำนวนผู้กู้ที่ไม่ดีในแต่ละปีคิดเป็น%

graph_dims = (24,10)

fig, ax = plt.subplots(figsize=graph_dims)

palette = ["#3791D7"]

sns.barplot(x="year", y="loan_condition_num", data=loan, palette=palette, estimator=lambda x: sum(x)/len(x), ci = None)

ax.set(ylabel="(% of Bad Loans)")
# plot เพื่อดูว่าในเงินที่ให้กู้ในแต่ละประเภทมีผู้กู้ที่ไม่ดีกี่ %

graph_dims = (24,10)

fig, ax = plt.subplots(figsize=graph_dims)



palette = ["#3791D7"]

sns.barplot(x="purpose", y="loan_condition_num", data=loan, palette=palette, estimator=lambda x: sum(x)/len(x), ci = None)

ax.set(ylabel="(% of Bad Loans)")
loan.info()
# ตรวจสอบว่ามี columns มีค่าว่างหรือไม่ ถ้ามีจะเป็น True ถ้าไม่จะเป็น False

loan.replace([np.inf, -np.inf], np.nan)

loan.isnull().any()
# plot เพื่อดูระยะเวลาการจ้างงานของผู้กู้ที่ไม่ดีในแต่ละปี

graph_dims = (24,10)

fig, ax = plt.subplots(figsize=graph_dims)



palette = ["#3791D7"]

sns.barplot(x="emp_length", y="loan_condition_num", data=loan, palette=palette, estimator=lambda x: sum(x)/len(x), ci = None)

ax.set(ylabel="(% of Bad Loans)")
# plot เพื่อดูระยะเวลาการจ้างงานของผู้กู้ทั้งหมดในแต่ละปี

plt.figure(figsize=(24,10))

palette = ["#3791D7"]

sns.countplot(loan['emp_length'], palette=palette);
# หาจำนวนของ catagory (emp_length = ระยะเวลาการจ้างงาน)

loan['emp_length'].value_counts()
# ฟังก์ชั่นใส่ค่าว่างให้กับ columns

def impute_emp_length(x):

    if pd.isnull(x):

        return '7 years'

    else:

        return x

    

loan['emp_length'] = loan['emp_length'].apply(impute_emp_length)
# plot เพื่อดู distribution รายได้ในแต่ละปีของผู้กู้

plt.figure(figsize=(24,10))

sns.distplot(loan['annual_inc'])
# หาค่ามัธยฐาน

loan['annual_inc'].median()
# ฟังก์ชั่นใส่ค่าว่างให้กับ columns

def impute_annual_inc(x):

    if pd.isnull(x):

        return 60000.0

    else:

        return x

    

loan['annual_inc'] = loan['annual_inc'].apply(impute_annual_inc)
# plot เพื่อดู distribution ของจำนวนการสอบถามข้อมูลใน 6 เดือนที่ผ่านมา (ไม่รวมการสอบถามเกี่ยวกับรถยนต์และการจำนอง)

plt.figure(figsize=(24,10))

sns.distplot(loan['inq_last_6mths'], fit=norm, kde=False)
# หาค่ามัธยฐาน

loan['inq_last_6mths'].median()
# ฟังก์ชั่นใส่ค่าว่างให้กับ columns

def impute_inq_last_6mths(x):

    if pd.isnull(x):

        return 1.0

    else:

        return x

    

loan['inq_last_6mths'] = loan['inq_last_6mths'].apply(impute_inq_last_6mths)
# plot เพื่อดู distribution จำนวนของวงเงินเครดิตคงค้างของผู้ยืม

plt.figure(figsize=(24,10))

sns.distplot(loan['open_acc'], fit=norm, kde=False)
# หาค่ามัธยฐาน

loan['open_acc'].median()
# ฟังก์ชั่นใส่ค่าว่างให้กับ columns

def impute_open_acc(x):

    if pd.isnull(x):

        return 9.0

    else:

        return x

    

loan['open_acc'] = loan['open_acc'].apply(impute_open_acc)
# plot เพื่อดู distribution จำนวนวงเงินทั้งหมดในปัจจุบันของผู้กู้

plt.figure(figsize=(24,10))

sns.distplot(loan['total_acc'], fit=norm, kde=False)
# หาค่ามัธยฐาน

loan['total_acc'].median()
# ฟังก์ชั่นใส่ค่าว่างให้กับ columns

def impute_total_acc(x):

    if pd.isnull(x):

        return 21.0

    else:

        return x

    

loan['total_acc'] = loan['total_acc'].apply(impute_total_acc)
# plot เพื่อดู distribution จำนวนเร็กคอร์ดที่เสียหาย

plt.figure(figsize=(24,10))

sns.distplot(loan['pub_rec'], fit=norm, kde=False)
# หาค่ามัธยฐาน

loan['pub_rec'].median()
# ฟังก์ชั่นใส่ค่าว่างให้กับ columns

def impute_pub_rec(x):

    if pd.isnull(x):

        return 1.0

    else:

        return x

    

loan['pub_rec'] = loan['pub_rec'].apply(impute_pub_rec)
# plot เพื่อดู distribution อัตราการใช้ประโยชน์จากหมุนเงินหรือจำนวนเครดิตที่ผู้กู้ใช้เมื่อเทียบกับเครดิตหมุนเวียนทั้งหมดที่มีอยู่

plt.figure(figsize=(24,10))

sns.distplot(loan['revol_util'], fit=norm, kde=False)
# หาค่ามัธยฐาน

loan['revol_util'].median()
# ฟังก์ชั่นใส่ค่าว่างให้กับ columns

def impute_revol_util(x):

    if pd.isnull(x):

        return 56.8

    else:

        return x

    

loan['revol_util'] = loan['revol_util'].apply(impute_revol_util)
# plot เพื่อดู distribution dti(อัตราส่วนที่คำนวณโดยใช้การชำระหนี้รายเดือนทั้งหมดของผู้กู้ในภาระหนี้ทั้งหมดไม่รวมการจำนองที่ขอ หารด้วยรายได้ต่อเดือนที่ผู้กู้รายงาน)

plt.figure(figsize=(24,10))

sns.distplot(loan['dti'], fit=norm, kde=False)
# หาค่ามัธยฐาน

loan['dti'].median()
# ฟังก์ชั่นใส่ค่าว่างให้กับ columns

def impute_dti(x):

    if pd.isnull(x):

        return 15.09

    else:

        return x

    

loan['dti'] = loan['dti'].apply(impute_revol_util)
# ตรวจสอบอีกครั้งว่ามี columns มีค่าว่างหรือไม่

loan.isnull().any()
# คัดเลือก feature ที่จำเป็นต่อการทำโมเดล

feature_selected = ['purpose', 'verification_status', 'loan_amnt', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'annual_inc', 'dti', 'inq_last_6mths', 'open_acc', 'total_acc', 'pub_rec', 'revol_util', 'revol_bal', 'addr_state', 'term', 'loan_condition', 'loan_condition_num']

loan = loan[feature_selected]

loan.head()
# Check correlations

sns.heatmap(loan.corr(), annot=True)
# Drop installment

loan = loan.drop(columns=['loan_amnt'])

# Check correlations อีกรอบ

sns.heatmap(loan.corr(), annot=True)
# ดูข้อมูลโดยรวม

loan.head()
# check missing value อีกรอบเพื่อความชัว

loan.isnull().any()
# แปลง categories ในเป็นตัวเลขโดยการทำ one hot encoding 

categories = ['purpose', 'verification_status', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'addr_state', 'term']

raw_model_data = pd.get_dummies(loan.copy(), columns=categories,drop_first=True)
# กำหนด feature ของ X และ y

X = raw_model_data.drop(columns=['loan_condition', 'loan_condition_num'],axis=1)

y = raw_model_data['loan_condition']



# แบ่ง X_train, X_test, y_train, y_test และกำหนดค่า randon_state

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)
# กำหนดค่าของ Forest (n_estimators คือจำนวนของต้นไม้)

random_forest = RandomForestClassifier(n_estimators=600, class_weight="balanced")

# ทำการ Train Model

random_forest.fit(X_train, y_train)

# ทำการ predict 

y_pred = random_forest.predict(X_test)

# แสดง report ของการ prediction

print(classification_report(y_test,y_pred))
# เก็บค่า accuracy ของ prediction

acc_random_forest = round(accuracy_score(y_test, y_pred) * 100, 2)
# กำหนดค่าของ DecisionTree 

decision_tree = DecisionTreeClassifier(splitter="best",min_samples_split=10000, class_weight="balanced")

# ทำการ Train Model 

decision_tree.fit(X_train, y_train)

# ทำการ predict 

y_pred = decision_tree.predict(X_test)

# แสดง report ของการ prediction

print(classification_report(y_test,y_pred))
# เก็บค่า accuracy ของ prediction

acc_decision_tree = round(accuracy_score(y_test, y_pred) * 100, 2)
# กำหนดค่าของ KNN 

knn = KNeighborsClassifier(n_neighbors = 2, algorithm="auto", leaf_size = 50)

# ทำการ train model

knn.fit(X_train, y_train)

# ทำการ predict 

y_pred = knn.predict(X_test)

# ดูค่า accuracy ของ prediction

print(classification_report(y_test,y_pred))
# เก็บค่า accuracy ของ prediction

acc_knn = round(accuracy_score(y_test, y_pred) * 100, 2)
# กำหนดค่าของ Gaussian Naive Bayes

gaussian = GaussianNB()

# ทำการ train model

gaussian.fit(X_train, y_train)

# ทำการ predict 

y_pred = gaussian.predict(X_test)

# ดูค่า accuracy ของ prediction

print(classification_report(y_test,y_pred))
# เก็บค่า accuracy ของ prediction

acc_gaussian = round(accuracy_score(y_test, y_pred) * 100, 2)
# กำหนดค่าของ Perceptron

perceptron = Perceptron(max_iter=2000, class_weight="balanced")

# ทำการ train model

perceptron.fit(X_train, y_train)

# ทำการ predict

Y_pred = perceptron.predict(X_test)

# ดูค่า accuracy ของ prediction

print(classification_report(y_test,y_pred))
# เก็บค่า accuracy ของ prediction

acc_perceptron = round(accuracy_score(y_test, y_pred) * 100, 2)
# กำหนดค่าของ Stochastic Gradient Descent

sgd = SGDClassifier(l1_ratio=0.16,class_weight="balanced")

# ทำการ train model

sgd.fit(X_train, y_train)

# ทำการ predict

y_pred = sgd.predict(X_test)

# ดูค่า accuracy ของ prediction

print(classification_report(y_test,y_pred))
# เก็บค่า accuracy ของ prediction

acc_sgd = round(accuracy_score(y_test, y_pred) * 100, 2)
# สร้าง dataframe เพื่อเก็บข้อมูล Score

models = pd.DataFrame({

    'Model': ['KNN', 'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Decision Tree'],

    'Score': [acc_knn, acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd,  acc_decision_tree]})



# แสดงผลออกมาโดยเรียงจากมากไปน้อย

models.sort_values(by='Score', ascending=False)
# Export Decision tree 

dot_data = StringIO()

export_graphviz(decision_tree, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True ,class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

# Export เป็นไฟล์ png

graph.write_png('dtree_pipe.png')

# Export เป็นไฟล์ pdf

graph.write_pdf('dtree_pipe.pdf')

# แสดง Decision tree pipe 

Image(graph.create_png())
# ตัวอย่างการ Prediction

print("X_value = %s, Predicted=%s" % (X_test, y_pred[3000]))