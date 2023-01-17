#เตรียมข้อมูลและไลบรารี่

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn import metrics

from sklearn.cross_validation import train_test_split #training and testing data

import random

import matplotlib.pyplot as plt

%matplotlib inline

import os

from matplotlib import rcParams

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
#ดึงข้อมูลจากไฟล์มาเก็บไว้ในตัวแปร df

df = pd.read_csv('../input/googleplaystore.csv')

df.head()
#ตรวจสอบข้อมูล

df.info() 
#Clean data โดยการตรวจสอบข้อมูลว่ามีค่า null ที่ต้องแก้ไขหรือไม่ เนื่องจากวัตถุประสงค์หลักของเราคือการ predicting the ratings ของแอพ จึงต้องยึดคอลัมน์ Rating เป็นหลัก ด้วยการลบค่า NaN ทั้งหมด เพื่อให้ง่ายต่อการ predict



df.dropna(inplace = True)

df.info()
#ตรวจสอบข้อมูลในการที่จะ clean data

df.head()
#Cean data ของคอลัมน์ Category โดยจะใช้ one-hot encoding เพื่อให้มี dummy variable ที่จะเป็นตัวแปรเชิงคุณภาพที่จะใช้ในการทำ regression ด้วยการแปลงข้อมูล Category ที่มี type เป็น String ไปเป็น integer



categoryVal = df['Category'].unique() #เก็บค่าข้อมูลที่ไม่ซ้ำกัน

categoryValCount = len(categoryVal) #เก็บค่าจำนวนข้อมูลของ Category

category_dict = {}

for i in range(0,categoryValCount): #วนลูปเพื่อเก็บค่าข้อมูลที่ไม่ซ้ำลงใน Array เพื่อเตรียมนำไปแปลงข้อมูล

    category_dict[categoryVal[i]] = i

df['Category_c'] = df['Category'].map(category_dict).astype(int) #แปลงข้อมูลไปเป็น integer

df['Category_c'].head()
#Clean data ของคอลัมน์ Rating โดยการ Convert จาก rating section ไปเป็น integers เนื่องจากในกรณีนี้ความเข้มข้นของข้อมูลมีความสัมพันธ์กันมาก เราจึงไม่ใช้ one-hot encoding แต่ใช้เป็น rating classification แทน ด้วยแปลงค่า Rating ที่มี Type จาก String ไปเป็น integer



RatingL = df['Content Rating'].unique() #เก็บค่า rating ที่ไม่ซ้ำกัน

RatingDict = {}

for i in range(len(RatingL)): #วนลูปเพื่อเก็บค่าข้อมูลที่ไม่ซ้ำลงใน Array เพื่อเตรียมนำไปแปลงข้อมูล

    RatingDict[RatingL[i]] = i

df['Content Rating'] = df['Content Rating'].map(RatingDict).astype(int) #แปลงข้อมูลไปเป็น integer

df['Content Rating'].head()
#Clean data ของคอลัมน์ Review โดยการ Convert จาก text ไปเป็น numberic ด้วยการแปลงเป็น integer



df['Reviews'] = df['Reviews'].astype(int) #แปลงข้อมูลไปเป็น integer

df['Reviews'].head()
#Clean data ของคอลัมน์ Sizes โดยการปรับเปลี่ยนขนาดที่กำกับไว้จาก type ที่เป็น String ไปเป็น float ด้วยการตรวจสอบหน่วยเพื่อแปลงค่าให้เป็น numberic และเติมข้อมูลที่เป็นค่า missing values โดยการใช้ ffill



def change_size(size): #สร้าง function เปลี่ยนขนาด

    if 'M' in size: #เช็คเงื่อนไขว่าเป็น M หน่วย Mega หรือไม่

        x = size[:-1] #เก็บค่าเฉพาะตัวเลข โดยการ [:-1] คือลบตัวสุดท้ายที่เป็นหน่วยออก เก็บไว้ในตัวแปร x

        x = float(x)*1000000 #คูณตามจำนวนหน่วยนั้นๆ 

        return(x)

    elif 'k' == size[-1:]: #เช็คเงื่อนไขว่าเป็น K หน่วย Kilos หรือไม่ และทำเหมือนกับเงื่อนไข if 

        x = size[:-1]

        x = float(x)*1000

        return(x)

    else: #มีเฉพาะตัวเลข จึงไม่ต้องแก้ไขอะไร

        return None

    

df['Size'] = df['Size'].map(change_size)

df.Size.fillna(method = 'ffill', inplace = True) #เติมข้อมูลที่เป็นค่า missing value ให้เป็น NA

df['Size'].head()
#Clean data ของคอลัมน์ Installs โดยการแปลงข้อมูลจาก text ให้เป็น numberic ด้วยการลบตัวสุดท้ายที่เป็นเครื่องหมาย + และสัญลักษณ์ , ออกไปก็จะเหลือเพียงตัวเลข ทำให้กลายเป็น numberic



df['Installs'] = [int(i[:-1].replace(',','')) for i in df['Installs']] # ลบตัวสุดท้ายออก โดยใช้ [:-1] และใช้ .replace() medthod ในการลบสัญลักษณ์ ,

df['Installs'].head()
#Clean data ของคอลัมน์ Type โดยการ Convert จาก classification ที่มี 2 ค่าคือ free/paid ไปเป็นค่า binary ด้วยการตรวจสอบว่าเป็นคำไหนแล้วส่งออกค่าเป็น 0/1 หรือค่า binary 



def type_cat(types): #สร้าง function เพื่อ Convert ข้อมูลจาก classification

    if types == 'Free': #เช็คคำว่าเป็น Free หรือไม่

        return 0

    else: 

        return 1



df['Type'] = df['Type'].map(type_cat)

df['Type'].head()
#Clean data ของคอลัมน์ Price โดยแปลงข้อมูลจาก text ให้เป็น numberic ด้วยการตรวจสอบค่าว่าถ้าไม่เป็นค่า 0 จะลบตัวหน้าทิ้งซึ่งก็คือ ' $ ' ที่ไม่ใช่ค่า numberic และทำให้ข้อมูลเป็นชนิด float  



def price_clean(price): #สร้าง fucntion ตรวจสอบ

    if price == '0': #ตรวจสอบว่าเป็น 0 หรือไม่

        return 0

    else:

        price = price[1:] # [1:] คือการลบตัวหน้าสุดออกไป เหลือไว้เพียงตัวเลข

        price = float(price) 

        return price



df['Price'] = df['Price'].map(price_clean).astype(float) #แปลงข้อมูลเป็น float

df['Price'].head()
#Clean data ของคอลัมน์ Genres ซึ่งเป็นสับเซตของคอลัมน์ Category ในกรณีนี้เราจะใช้ one-hot encoding เพื่อให้มี dummy variable ที่จะเป็นตัวแปรอิสระเพิ่มขึ้น เนื่องจากจะพิจารณาค่าจากตัวเลขของ genres ด้วยการแปลงข้อมูลที่มี type เป็น String ไปเป็น integer ด้วยการเก็บข้อมูลที่ไม่ซ้ำกันไว้ในตัวแปร



GenresL = df.Genres.unique() #เก็บค่าข้อมูลที่ไม่ซ้ำกัน

GenresDict = {}

for i in range(len(GenresL)): #วนลูปเพื่อเก็บค่าข้อมูลที่ไม่ซ้ำกัน ไว้เตรียมแปลงข้อมูล

    GenresDict[GenresL[i]] = i

df['Genres_c'] = df['Genres'].map(GenresDict).astype(int) #แปลงข้อมูลป็น integer

df['Genres_c'].head()
#Clean data โดยการลบคอลัมน์ที่ไม่ได้ใช้หรือไม่มีเป็นประโยชน์ออก เนื่องจากเรามุ่งเน้นไปที่การ prdicting the rating ทำให้คอลัมน์ Last Updated, Current Ver , Android Ver และ App ไม่จำเป็นต่อ Model



df.drop(labels = ['Last Updated','Current Ver','Android Ver','App'], axis = 1, inplace = True)

df.info()
#ผลลัพธ์ของการ Clean data จนได้ dataframe ที่มีการ integer encoding ของตัวแปร category 



df.head()
#สร้าง dataframe อีกชุดสำหรับสร้าง dummy variable เพื่อ encoding ของแต่ละ Category



df2 = pd.get_dummies(df, columns=['Category'])

df2.head()
# Rating Distribution

rcParams['figure.figsize'] = 15,8

ratings_plot = sns.countplot(x="Rating",data=df, palette = "inferno")

ratings_plot.set_xticklabels(ratings_plot.get_xticklabels(), rotation=90, ha="right")

ratings_plot 

plt.title('Rating Distribution',size = 20)
# Reviews distibution 

rcParams['figure.figsize'] = 15,8

g = sns.kdeplot(df.Reviews, color="Green", shade = True)

g.set_xlabel("Reviews")

g.set_ylabel("Frequency")

plt.title('Reveiw Distribution',size = 20)
#ประเมินค่า error term โดยการรับเงื่อนไขข้อผิดพลาดจาก metrics method เพื่อนำมาเปรียบเทียบประสิทธิภาพของ Model ด้วยการใช้ผลลัพธ์ที่คาดการณ์ (y_predict) กับผลลัพธ์ที่เกิดขึ้นจริง (y_true) มาประเมิน



def Evaluationmatrix(y_true, y_predict): #Function แสดงประเมินค่า error term #สร้าง Function การประเมินค่า error

    print ('Mean Squared Error: '+ str(metrics.mean_squared_error(y_true,y_predict))) #แสดงผลลัพธ์จาก Metrics ต่างๆสที่วัดประสิทธิภาพของ Model

    print ('Mean absolute Error: '+ str(metrics.mean_absolute_error(y_true,y_predict))) 

    print ('Mean squared Log Error: '+ str(metrics.mean_squared_log_error(y_true,y_predict)))



def Evaluationmatrix_dict(y_true, y_predict, name = 'Linear - Integer'): #Function หาค่า errorterm ของ metric แบบต่างๆ โดยใช้ค่า ค่าผลลัพธ์ที่คาดการณ์ (y_predict) กับผลลัพธ์ที่เกิดขึ้นจริง (y_true)ประเมิน

    dict_matrix = {}

    dict_matrix['Series Name'] = name

    dict_matrix['Mean Squared Error'] = metrics.mean_squared_error(y_true,y_predict) #เก็บค่าการประเมิน error ที่ได้จาก Metrics ต่างๆ สำหรับการวัดประสิทธิภาพของ Model

    dict_matrix['Mean Absolute Error'] = metrics.mean_absolute_error(y_true,y_predict)

    dict_matrix['Mean Squared Log Error'] = metrics.mean_squared_log_error(y_true,y_predict)

    return dict_matrix
#excluding Genre_c column

from sklearn.linear_model import LinearRegression 



#Integer encoding

X = df.drop(labels = ['Category','Rating','Genres','Genres_c'],axis = 1) #ใช้ df dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y = df.Rating

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) #กำหนดขนาด train และ test data

model1a = LinearRegression() #สร้าง linear regression model

model1a.fit(X_train,y_train)

Results1a = model1a.predict(X_test)



#สร้างผลลัพธ์ที่เป็น dataframe และ addition of first entry

resultsdf = pd.DataFrame()

resultsdf = resultsdf.from_dict(Evaluationmatrix_dict(y_test,Results1a),orient = 'index')

resultsdf = resultsdf.transpose()



#One-hot encoding

X_d = df2.drop(labels = ['Rating','Genres','Category_c','Genres_c'],axis = 1)  #ใช้ df2 dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_d = df2.Rating

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30) #กำหนดขนาด train และ test data

model1a_d = LinearRegression() #สร้าง linear regression model

model1a_d.fit(X_train_d,y_train_d)

Results1a_d = model1a_d.predict(X_test_d)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_d,Results1a_d, name = 'Linear - One-hot'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe



#Integer encoding without Reviews

X_r = df.drop(labels = ['Category','Rating','Genres','Genres_c','Reviews'],axis = 1) #ใช้ df dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_r = df.Rating

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.30) #กำหนดขนาด train และ test data

model1a_r = LinearRegression() #สร้าง linear regression model

model1a_r.fit(X_train_r,y_train_r)

Results1a_r = model1a_r.predict(X_test_r)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_r,Results1a_r, name = 'Linear - Integer without Reviews'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe



#One-hot encoding without Reviews

X_rd = df2.drop(labels = ['Rating','Genres','Category_c','Genres_c','Reviews'],axis = 1)  #ใช้ df2 dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_rd = df2.Rating

X_train_rd, X_test_rd, y_train_rd, y_test_rd = train_test_split(X_rd, y_rd, test_size=0.30) #กำหนดขนาด train และ test data

model1a_rd = LinearRegression() #สร้าง linear regression model

model1a_rd.fit(X_train_rd,y_train_rd)

Results1a_rd = model1a_rd.predict(X_test_rd)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_rd,Results1a_rd, name = 'Linear - One-hot without Reviews'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe

#พล็อตกราฟที่ได้จาก Model

plt.figure(figsize=(12,7))

sns.regplot(Results1a,y_test,color='brown', label = 'Integer')

sns.regplot(Results1a_d,y_test_d,color='red',label = 'One-hot')

sns.regplot(Results1a_r,y_test_r,color='orange',label = 'Integer without Reviews')

sns.regplot(Results1a_rd,y_test_rd,color='yellow',label = 'One-hot without Reviews')

plt.legend()

plt.title('Linear Model - Excluding Genres')

plt.xlabel('Predicted Ratings')

plt.ylabel('Actual Ratings')

plt.show()
#แสดงค่าเฉลี่ยจริงที่ได้ , ค่าเฉลี่ยและค่าเบี่ยงเบนมาตรฐานของทั้ง Integer และ Dummy encoding ที่ได้จากการทำ model

print ('Actual mean of population\t\t  ' + str(y.mean()))

print ('Integer encoding(mean)\t\t\t  ' + str(Results1a.mean()))

print ('Integer encoding(mean)  without Reviews\t  '+ str(Results1a_r.mean()))

print ('One-hot encoding(mean)\t\t\t  '+ str(Results1a_d.mean()))

print ('One-hot encoding(mean)  without Reviews\t  '+ str(Results1a_rd.mean()))

print ('Integer encoding(std)\t\t\t  '  + str(Results1a.std()))

print ('Integer encoding(std)   without Reviews\t  '+ str(Results1a_r.std()))

print ('One-hot encoding(std)\t\t\t  '+ str(Results1a_d.std()))

print ('One-hot encoding(std)   without Reviews\t  '+ str(Results1a_rd.std()))
#Including Genre_c column



#Integer encoding

X = df.drop(labels = ['Category','Rating','Genres','Genres_c'],axis = 1) #ใช้ df dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y = df.Rating

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) #กำหนดขนาด train และ test data

model1b = LinearRegression() #สร้าง linear regression model

model1b.fit(X_train,y_train)

Results1b = model1b.predict(X_test)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results1b, name = 'Linear(inc Genres) - Integer'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe



#One-hot encoding

X_d = df2.drop(labels = ['Rating','Genres','Category_c','Genres_c'],axis = 1)  #ใช้ df2 dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_d = df2.Rating

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30)  #กำหนดขนาด train และ test data

model1b_d = LinearRegression() #สร้าง linear regression model

model1b_d.fit(X_train_d,y_train_d)

Results1b_d = model1b_d.predict(X_test_d)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_d,Results1b_d, name = 'Linear(inc Genres) - One-hot'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe



#Integer encoding without Reviews

X_r = df.drop(labels = ['Category','Rating','Genres','Genres_c','Reviews'],axis = 1) #ใช้ df dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_r = df.Rating

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.30) #กำหนดขนาด train และ test data

model1b_r = LinearRegression() #สร้าง linear regression model

model1b_r.fit(X_train_r,y_train_r)

Results1b_r = model1b_r.predict(X_test_r)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_r,Results1b_r, name = 'Linear(inc Genres) - Integer without Reviews'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe



#One-hot encoding without Reviews

X_rd = df2.drop(labels = ['Rating','Genres','Category_c','Genres_c','Reviews'],axis = 1)  #ใช้ df2 dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_rd = df2.Rating

X_train_rd, X_test_rd, y_train_rd, y_test_rd = train_test_split(X_rd, y_rd, test_size=0.30) #กำหนดขนาด train และ test data

model1b_rd = LinearRegression() #สร้าง linear regression model

model1b_rd.fit(X_train_rd,y_train_rd)

Results1b_rd = model1b_rd.predict(X_test_rd)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_rd,Results1b_rd, name = 'Linear(inc Genres) - One-hot without Reviews'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe

#พล็อตกราฟที่ได้จาก Model

plt.figure(figsize=(12,7))

sns.regplot(Results1b,y_test,color='brown', label = 'Integer')

sns.regplot(Results1b_d,y_test_d,color='red',label = 'One-hot')

sns.regplot(Results1b_r,y_test_r,color='orange',label ='Integer without Reviews')

sns.regplot(Results1b_rd,y_test_rd,color='yellow',label = 'One-hot without Reviews')

plt.legend()

plt.title('Linear Model - Including Genres')

plt.xlabel('Predicted Ratings')

plt.ylabel('Actual Ratings')

plt.show()
#แสดงค่าเฉลี่ยจริงที่ได้ , ค่าเฉลี่ยและค่าเบี่ยงเบนมาตรฐานของทั้ง Integer และ Dummy encoding ที่ได้จากการทำ model

print ('Actual mean of population\t\t  ' + str(y.mean()))

print ('Integer encoding(mean)\t\t\t  ' + str(Results1b.mean()))

print ('Integer encoding(mean)  without Reviews\t  '+ str(Results1b_r.mean()))

print ('One-hot encoding(mean)\t\t\t  '+ str(Results1b_d.mean()))

print ('One-hot encoding(mean)  without Reviews\t  '+ str(Results1b_rd.mean()))

print ('Integer encoding(std)\t\t\t  ' + str(Results1b.std()))

print ('Integer encoding(std)   without Reviews\t  '+ str(Results1b_r.std()))

print ('One-hot encoding(std)\t\t\t  '+ str(Results1b_d.std()))

print ('One-hot encoding(std)   without Reviews\t  '+ str(Results1b_rd.std()))
#Excluding Genre_c column

from sklearn import svm



#Integer encoding

X = df.drop(labels = ['Category','Rating','Genres','Genres_c'],axis = 1) #ใช้ df dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y = df.Rating

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) #กำหนดขนาด train และ test data

model2a = svm.SVR() #สร้าง SVR model

model2a.fit(X_train,y_train)

Results2a = model2a.predict(X_test)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results2a, name = 'SVR - Integer'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe



#One-hot encoding

X_d = df2.drop(labels = ['Rating','Genres','Category_c','Genres_c'],axis = 1)  #ใช้ df2 dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_d = df2.Rating

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30) #กำหนดขนาด train และ test data

model2a = svm.SVR() #สร้าง SVR model

model2a.fit(X_train_d,y_train_d)

Results2a_d = model2a.predict(X_test_d)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_d,Results2a_d, name = 'SVR - One-hot'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe



#Integer encoding without Reviews

X_r = df.drop(labels = ['Category','Rating','Genres','Genres_c','Reviews'],axis = 1) #ใช้ df dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_r = df.Rating

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.30) #กำหนดขนาด train และ test data

model2a_r = svm.SVR() #สร้าง SVR model

model2a_r.fit(X_train_r,y_train_r)

Results2a_r = model2a_r.predict(X_test_r)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_r,Results2a_r, name = 'SVR - Integer without Reviews'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe



#One-hot encoding without Reviews

X_rd = df2.drop(labels = ['Rating','Genres','Category_c','Genres_c','Reviews'],axis = 1)  #ใช้ df2 dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_rd = df2.Rating

X_train_rd, X_test_rd, y_train_rd, y_test_rd = train_test_split(X_rd, y_rd, test_size=0.30) #กำหนดขนาด train และ test data

model2a_rd = svm.SVR() #สร้าง SVR model

model2a_rd.fit(X_train_rd,y_train_rd)

Results2a_rd = model2a_rd.predict(X_test_rd)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_rd,Results2a_rd, name = 'SVR - One-hot without Reviews'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe

#พล็อตกราฟที่ได้จาก Model

plt.figure(figsize=(12,7))

sns.regplot(Results2a,y_test,color='brown', label = 'Integer')

sns.regplot(Results2a_d,y_test_d,color='red',label = 'One-hot')

sns.regplot(Results2a_r,y_test_r,color='orange',label = 'Integer without Reviews')

sns.regplot(Results2a_rd,y_test_rd,color='yellow',label = 'One-hot without Reviews')

plt.legend()

plt.title('SVR Model - Excluding Genres')

plt.xlabel('Predicted Ratings')

plt.ylabel('Actual Ratings')

plt.show()
#แสดงค่าเฉลี่ยจริงที่ได้ , ค่าเฉลี่ยและค่าเบี่ยงเบนมาตรฐานของทั้ง Integer และ Dummy encoding ที่ได้จากการทำ model

print ('Actual mean of population\t\t  ' + str(y.mean()))

print ('Integer encoding(mean)\t\t\t  ' + str(Results2a.mean()))

print ('Integer encoding(mean)  without Reviews\t  '+ str(Results2a_r.mean()))

print ('One-hot encoding(mean)\t\t\t  '+ str(Results2a_d.mean()))

print ('One-hot encoding(mean)  without Reviews\t  '+ str(Results2a_rd.mean()))

print ('Integer encoding(std)\t\t\t  ' + str(Results2a.std()))

print ('Integer encoding(std)   without Reviews\t  '+ str(Results2a_r.std()))

print ('One-hot encoding(std)\t\t\t  '+ str(Results2a_d.std()))

print ('One-hot encoding(std)   without Reviews\t  '+ str(Results2a_rd.std()))
#Including Genre_c column



#Integer encoding

X = df.drop(labels = ['Category','Rating','Genres','Genres_c'],axis = 1) #ใช้ df dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y = df.Rating

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) #กำหนดขนาด train และ test data

model2b = svm.SVR()  #สร้าง SVR model

model2b.fit(X_train,y_train)

Results2b = model2b.predict(X_test)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results2b, name = 'SVR(inc Genres) - Integer'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe



#One-hot encoding

X_d = df2.drop(labels = ['Rating','Genres','Category_c','Genres_c'],axis = 1)  #ใช้ df2 dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_d = df2.Rating

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30) #กำหนดขนาด train และ test data

model2b = svm.SVR()  #สร้าง SVR model

model2b.fit(X_train_d,y_train_d)

Results2b_d = model2b.predict(X_test_d)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_d,Results2b_d, name = 'SVR(inc Genres) - One-hot'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe



#Integer encoding without Reviews

X_r = df.drop(labels = ['Category','Rating','Genres','Genres_c','Reviews'],axis = 1) #ใช้ df dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_r = df.Rating

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.30) #กำหนดขนาด train และ test data

model2b_r = svm.SVR() #สร้าง linear regression model

model2b_r.fit(X_train_r,y_train_r)

Results2b_r = model2b_r.predict(X_test_r)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_r,Results2b_r, name = 'SVR(inc Genres) - Integer without Reviews'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe



#One-hot encoding without Reviews

X_rd = df2.drop(labels = ['Rating','Genres','Category_c','Genres_c','Reviews'],axis = 1)  #ใช้ df2 dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_rd = df2.Rating

X_train_rd, X_test_rd, y_train_rd, y_test_rd = train_test_split(X_rd, y_rd, test_size=0.30) #กำหนดขนาด train และ test data

model2b_rd = svm.SVR() #สร้าง linear regression model

model2b_rd.fit(X_train_rd,y_train_rd)

Results2b_rd = model2b_rd.predict(X_test_rd)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_rd,Results2b_rd, name = 'SVR(inc Genres) - One-hot without Reviews'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe
#พล็อตกราฟที่ได้จาก Model

plt.figure(figsize=(12,7))

sns.regplot(Results2b,y_test,color='brown', label = 'Integer')

sns.regplot(Results2b_d,y_test_d,color='red',label = 'One-hot')

sns.regplot(Results2b_r,y_test_r,color='orange',label = 'Integer without Reviews')

sns.regplot(Results2b_rd,y_test_rd,color='yellow',label = 'One-hot without Reviews')

plt.legend()

plt.title('SVR Model - Including Genres')

plt.xlabel('Predicted Ratings')

plt.ylabel('Actual Ratings')

plt.show()
#แสดงค่าเฉลี่ยจริงที่ได้ , ค่าเฉลี่ยและค่าเบี่ยงเบนมาตรฐานของทั้ง Integer และ Dummy encoding ที่ได้จากการทำ model

print ('Actual mean of population\t\t  ' + str(y.mean()))

print ('Integer encoding(mean)\t\t\t  ' + str(Results2a.mean()))

print ('Integer encoding(mean)  without Reviews\t  '+ str(Results2a_r.mean()))

print ('One-hot encoding(mean)\t\t\t  '+ str(Results2a_d.mean()))

print ('One-hot encoding(mean)  without Reviews\t  '+ str(Results2a_rd.mean()))

print ('Integer encoding(std)\t\t\t  ' + str(Results2a.std()))

print ('Integer encoding(std)   without Reviews\t  '+ str(Results2a_r.std()))

print ('One-hot encoding(std)\t\t\t  '+ str(Results2a_d.std()))

print ('One-hot encoding(std)   without Reviews\t  '+ str(Results2a_rd.std()))
#Excluding Genre_c column

from sklearn.ensemble import RandomForestRegressor



#Integer encoding

X = df.drop(labels = ['Category','Rating','Genres','Genres_c'],axis = 1) #ใช้ df dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y = df.Rating

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) #กำหนดขนาด train และ test data

model3a = RandomForestRegressor() #สร้าง RFR Model

model3a.fit(X_train,y_train)

Results3a = model3a.predict(X_test)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results3a, name = 'RFR - Integer'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe



#One-hot encoding

X_d = df2.drop(labels = ['Rating','Genres','Category_c','Genres_c'],axis = 1)  #ใช้ df2 dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_d = df2.Rating

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30) #กำหนดขนาด train และ test data

model3a_d = RandomForestRegressor()  #สร้าง RFR Model

model3a_d.fit(X_train_d,y_train_d)

Results3a_d = model3a_d.predict(X_test_d)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results3a_d, name = 'RFR - One-hot'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe



#Integer encoding without Reviews

X_r = df.drop(labels = ['Category','Rating','Genres','Genres_c','Reviews'],axis = 1) #ใช้ df dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_r = df.Rating

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.30) #กำหนดขนาด train และ test data

model3a_r = RandomForestRegressor() #สร้าง linear regression model

model3a_r.fit(X_train_r,y_train_r)

Results3a_r = model3a_r.predict(X_test_r)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_r,Results3a_r, name = 'RFR - Integer without Reviews'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe



#One-hot encoding without Reviews

X_rd = df2.drop(labels = ['Rating','Genres','Category_c','Genres_c','Reviews'],axis = 1)  #ใช้ df2 dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_rd = df2.Rating

X_train_rd, X_test_rd, y_train_rd, y_test_rd = train_test_split(X_rd, y_rd, test_size=0.30) #กำหนดขนาด train และ test data

model3a_rd = RandomForestRegressor() #สร้าง linear regression model

model3a_rd.fit(X_train_rd,y_train_rd)

Results3a_rd = model3a_rd.predict(X_test_rd)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_rd,Results3a_rd, name = 'RFR - One-hot without Reviews'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe
#พล็อตกราฟที่ได้จาก Model

plt.figure(figsize=(12,7))

sns.regplot(Results3a,y_test,color='brown', label = 'Integer')

sns.regplot(Results3a_d,y_test_d,color='red',label = 'One-hot')

sns.regplot(Results3a_r,y_test_r,color='orange',label = 'Integer without Reviews')

sns.regplot(Results3a_rd,y_test_rd,color='yellow',label = 'One-hot without Reviews')

plt.legend()

plt.title('RFR Model - Excluding Genres')

plt.xlabel('Predicted Ratings')

plt.ylabel('Actual Ratings')

plt.show()
#แสดงค่าเฉลี่ยจริงที่ได้ , ค่าเฉลี่ยและค่าเบี่ยงเบนมาตรฐานของทั้ง Integer และ Dummy encoding ที่ได้จากการทำ model

print ('Actual mean of population\t\t  ' + str(y.mean()))

print ('Integer encoding(mean)\t\t\t  ' + str(Results3a.mean()))

print ('Integer encoding(mean)  without Reviews\t  '+ str(Results3a_r.mean()))

print ('One-hot encoding(mean)\t\t\t  '+ str(Results3a_d.mean()))

print ('One-hot encoding(mean)  without Reviews\t  '+ str(Results3a_rd.mean()))

print ('Integer encoding(std)\t\t\t  ' + str(Results3a.std()))

print ('Integer encoding(std)   without Reviews\t  '+ str(Results3a_r.std()))

print ('One-hot encoding(std)\t\t\t  '+ str(Results3a_d.std()))

print ('One-hot encoding(std)   without Reviews\t  '+ str(Results3a_rd.std()))
#Including Genre_c column



#Integer encoding

X = df.drop(labels = ['Category','Rating','Genres','Genres_c'],axis = 1) #ใช้ df dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y = df.Rating

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) #กำหนดขนาด train และ test data

model3b = RandomForestRegressor() #สร้าง RFR Model

model3b.fit(X_train,y_train)

Results3b = model3b.predict(X_test)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results3b, name = 'RFR(inc Genres) - Integer'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe



#One-hot encoding

X_d = df2.drop(labels = ['Rating','Genres','Category_c','Genres_c'],axis = 1)  #ใช้ df2 dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_d = df2.Rating

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30) #กำหนดขนาด train และ test data

model3b_d = RandomForestRegressor() #สร้าง RFR Model

model3b_d.fit(X_train_d,y_train_d)

Results3b_d = model3b_d.predict(X_test_d)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results3b_d, name = 'RFR(inc Genres) - One-hot'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe



#Integer encoding without Reviews

X_r = df.drop(labels = ['Category','Rating','Genres','Genres_c','Reviews'],axis = 1) #ใช้ df dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_r = df.Rating

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.30) #กำหนดขนาด train และ test data

model3b_r = RandomForestRegressor() #สร้าง linear regression model

model3b_r.fit(X_train_r,y_train_r)

Results3b_r = model3b_r.predict(X_test_r)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_r,Results3b_r, name = 'RFR(inc Genres) - Integer without Reviews'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe



#One-hot encoding without Reviews

X_rd = df2.drop(labels = ['Rating','Genres','Category_c','Genres_c','Reviews'],axis = 1)  #ใช้ df2 dataframe และลบคอลัมน์ที่ไม่ใช้ออกไป

y_rd = df2.Rating

X_train_rd, X_test_rd, y_train_rd, y_test_rd = train_test_split(X_rd, y_rd, test_size=0.30) #กำหนดขนาด train และ test data

model3b_rd = RandomForestRegressor() #สร้าง linear regression model

model3b_rd.fit(X_train_rd,y_train_rd)

Results3b_rd = model3b_rd.predict(X_test_rd)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_rd,Results3b_rd, name = 'RFR(inc Genres) - One-hot without Reviews'),ignore_index = True) #ใส่ผลลัพธ์ที่ได้ลงไปใน results dataframe
#พล็อตกราฟที่ได้จาก Model

plt.figure(figsize=(12,7))

sns.regplot(Results3b,y_test,color='brown', label = 'Integer')

sns.regplot(Results3b_d,y_test_d,color='red',label = 'One-hot')

sns.regplot(Results3b_r,y_test_r,color='orange',label = 'Integer without Reviews')

sns.regplot(Results3b_rd,y_test_rd,color='yellow',label = 'One-hot without Reviews')

plt.legend()

plt.title('RFR Model - Including Genres')

plt.xlabel('Predicted Ratings')

plt.ylabel('Actual Ratings')

plt.show()
#แสดงค่าเฉลี่ยจริงที่ได้ , ค่าเฉลี่ยและค่าเบี่ยงเบนมาตรฐานของทั้ง Integer และ Dummy encoding ที่ได้จากการทำ model

print ('Actual mean of population\t\t  ' + str(y.mean()))

print ('Integer encoding(mean)\t\t\t  ' + str(Results3b.mean()))

print ('Integer encoding(mean)  without Reviews\t  '+ str(Results2a_r.mean()))

print ('One-hot encoding(mean)\t\t\t  '+ str(Results3b_d.mean()))

print ('One-hot encoding(mean)  without Reviews\t  '+ str(Results2a_rd.mean()))

print ('Integer encoding(std)\t\t\t  ' + str(Results3b.std()))

print ('Integer encoding(std)   without Reviews\t  '+ str(Results2a_r.std()))

print ('One-hot encoding(std)\t\t\t  '+ str(Results3b_d.std()))

print ('One-hot encoding(std)   without Reviews\t  '+ str(Results3b_rd.std()))
#กราฟแสดงค่าตัวแปรต่างๆจาก dataframe สำหรับ integer encoded 

Feat_impt = {}

for col,feat in zip(X.columns,model3a.feature_importances_):

    Feat_impt[col] = feat



Feat_impt_df = pd.DataFrame.from_dict(Feat_impt,orient = 'index')

Feat_impt_df.sort_values(by = 0, inplace = True)

Feat_impt_df.rename(index = str, columns = {0:'Pct'},inplace = True)



plt.figure(figsize= (14,10))

Feat_impt_df.plot(kind = 'barh',figsize= (20,12),legend = False)

plt.show()
#กราฟแสดงค่าตัวแปรต่างๆจาก dataframe สำหรับ dummy encoded 

Feat_impt_d = {}

for col,feat in zip(X_d.columns,model3a_d.feature_importances_):

    Feat_impt_d[col] = feat



Feat_impt_df_d = pd.DataFrame.from_dict(Feat_impt_d,orient = 'index')

Feat_impt_df_d.sort_values(by = 0, inplace = True)

Feat_impt_df_d.rename(index = str, columns = {0:'Pct'},inplace = True)



plt.figure(figsize= (14,14))

Feat_impt_df_d.plot(kind = 'barh',figsize= (16,8),legend = False)

plt.show()
plt.figure(figsize = (10,10))

sns.regplot(x="Reviews", y="Rating", color = 'darkorange',data=df[df['Reviews']<1000000]);

plt.title('Rating VS Reveiws',size = 20)
#Rating VS Size

plt.figure(figsize = (10,10))

g = sns.jointplot(x="Size", y="Rating",color = 'orangered', data=df, size = 10);
plt.figure(figsize = (10,10))

sns.regplot(x="Installs", y="Rating", color = 'teal',data=df);

plt.title('Rating VS Installs',size = 20)
#กราฟที่แสดงการเปรียบเทียบผลลัพธ์ของการประเมินเพื่อวัดประสิทธิภาพของแต่ละ Model

resultsdf.set_index('Series Name', inplace = True)



#พล็อตกราฟค่า MSE , MAE และ MSLE

plt.figure(figsize = (18,25))

plt.subplot(3,1,1)

resultsdf['Mean Squared Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.3, 0.4, 0.6, 1), title = 'Mean Squared Error')

plt.subplot(3,1,2)

resultsdf['Mean Absolute Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.5, 0.4, 0.6, 1), title = 'Mean Absolute Error')

plt.subplot(3,1,3)

resultsdf['Mean Squared Log Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.7, 0.4, 0.6, 1), title = 'Mean Squared Log Error')

plt.show()
