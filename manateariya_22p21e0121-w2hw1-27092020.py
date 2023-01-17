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
data = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv')

import csv

with open('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv') as csvfile:

    reader = csv.reader(csvfile)

    data1=[]

    data2=[]

    data3=[]

    data4=[]

    data5=[]

    data6=[]

    data7=[]

    data8=[]

    data9=[]

    data10=[]

    count=0

    for i in reader:

        if(count==0):

            data1.append(i)

            data2.append(i)

            data3.append(i)

            data4.append(i)

            data5.append(i)

            data6.append(i)

            data7.append(i)

            data8.append(i)

            data9.append(i)

            data10.append(i)

            count=count+1

        if(count==1):

            data1.append(i)

            count=count+1

        elif(count==2):

            data2.append(i)

            count=count+1

        elif(count==3):

            data3.append(i)

            count=count+1

        elif(count==4):

            data4.append(i)

            count=count+1

        elif(count==5):

            data5.append(i)

            count=count+1

        elif(count==6):

            data6.append(i)

            count=count+1

        elif(count==7):

            data7.append(i)

            count=count+1

        elif(count==8):

            data8.append(i)

            count=count+1

        elif(count==9):

            data9.append(i)

            count=count+1

        elif(count==10):

            data10.append(i)

            count=1



data1.pop(1)

data1[0]



print(data1[-1])

data1[-1][0][:2] 

data1[-1][0][3:5]

data1[-1][0][11:16]

            

    
with open('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv') as csvfile:

    reader = csv.reader(csvfile)

    wether1=[]

    for t in reader:

        wether1.append(t)



data1all=[]

data2all=[]

data3all=[]

data4all=[]

data5all=[]

data6all=[]

data7all=[]

data8all=[]

data9all=[]

data10all=[]

for i in range (1,len(data1)):

    for y in range (1,len(wether1)):

        if data1[i][0][:2]== wether1[y][0][8:10] and data1[i][0][3:5] == wether1[y][0][5:7] and data1[i][0][11:16] and wether1[y][0][11:16]:

            data1all.append([data1[i][0],data1[i][1],data1[i][2],data1[i][3],data1[i][4],data1[i][5],data1[i][6],wether1[y][3],wether1[y][4],wether1[y][5]])

    

    

    

    



for i in range (1,len(data2)):

    for y in range (1,len(wether1)):

        if data2[i][0][:2]== wether1[y][0][8:10] and data2[i][0][3:5] == wether1[y][0][5:7] and data2[i][0][11:16] and wether1[y][0][11:16]:

            data2all.append([data2[i][0],data2[i][1],data2[i][2],data2[i][3],data2[i][4],data2[i][5],data2[i][6],wether1[y][3],wether1[y][4],wether1[y][5]])  

            

for i in range (1,len(data3)):

    for y in range (1,len(wether1)):

        if data3[i][0][:2]== wether1[y][0][8:10] and data3[i][0][3:5] == wether1[y][0][5:7] and data3[i][0][11:16] and wether1[y][0][11:16]:

            data3all.append([data3[i][0],data3[i][1],data3[i][2],data3[i][3],data3[i][4],data3[i][5],data3[i][6],wether1[y][3],wether1[y][4],wether1[y][5]])

            

for i in range (1,len(data4)):

    for y in range (1,len(wether1)):

        if data4[i][0][:2]== wether1[y][0][8:10] and data4[i][0][3:5] == wether1[y][0][5:7] and data4[i][0][11:16] and wether1[y][0][11:16]:

            data4all.append([data4[i][0],data4[i][1],data4[i][2],data4[i][3],data4[i][4],data4[i][5],data4[i][6],wether1[y][3],wether1[y][4],wether1[y][5]])



for i in range (1,len(data5)):

    for y in range (1,len(wether1)):

        if data5[i][0][:2]== wether1[y][0][8:10] and data5[i][0][3:5] == wether1[y][0][5:7] and data5[i][0][11:16] and wether1[y][0][11:16]:

            data5all.append([data5[i][0],data5[i][1],data5[i][2],data5[i][3],data5[i][4],data5[i][5],data5[i][6],wether1[y][3],wether1[y][4],wether1[y][5]])

for i in range (1,len(data6)):

    for y in range (1,len(wether1)):

        if data6[i][0][:2]== wether1[y][0][8:10] and data6[i][0][3:5] == wether1[y][0][5:7] and data6[i][0][11:16] and wether1[y][0][11:16]:

            data6all.append([data6[i][0],data6[i][1],data6[i][2],data6[i][3],data6[i][4],data6[i][5],data6[i][6],wether1[y][3],wether1[y][4],wether1[y][5]])

for i in range (1,len(data7)):

    for y in range (1,len(wether1)):

        if data7[i][0][:2]== wether1[y][0][8:10] and data7[i][0][3:5] == wether1[y][0][5:7] and data7[i][0][11:16] and wether1[y][0][11:16]:

            data7all.append([data7[i][0],data7[i][1],data7[i][2],data7[i][3],data7[i][4],data7[i][5],data7[i][6],wether1[y][3],wether1[y][4],wether1[y][5]])

for i in range (1,len(data8)):

    for y in range (1,len(wether1)):

        if data8[i][0][:2]== wether1[y][0][8:10] and data8[i][0][3:5] == wether1[y][0][5:7] and data8[i][0][11:16] and wether1[y][0][11:16]:

            data8all.append([data8[i][0],data8[i][1],data8[i][2],data8[i][3],data8[i][4],data8[i][5],data8[i][6],wether1[y][3],wether1[y][4],wether1[y][5]])

for i in range (1,len(data9)):

    for y in range (1,len(wether1)):

        if data9[i][0][:2]== wether1[y][0][8:10] and data9[i][0][3:5] == wether1[y][0][5:7] and data9[i][0][11:16] and wether1[y][0][11:16]:

            data9all.append([data9[i][0],data9[i][1],data9[i][2],data9[i][3],data9[i][4],data9[i][5],data9[i][6],wether1[y][3],wether1[y][4],wether1[y][5]])

for i in range (1,len(data10)):

    for y in range (1,len(wether1)):

        if data10[i][0][:2]== wether1[y][0][8:10] and data10[i][0][3:5] == wether1[y][0][5:7] and data10[i][0][11:16] and wether1[y][0][11:16]:

            data10all.append([data10[i][0],data10[i][1],data10[i][2],data10[i][3],data10[i][4],data10[i][5],data10[i][6],wether1[y][3],wether1[y][4],wether1[y][5]])



            

            

            

            

            

data2all
data1plant_mean_mul21=[]

data2plant_mean_mul21=[]

data3plant_mean_mul21=[]

data4plant_mean_mul21=[]

data5plant_mean_mul21=[]

data6plant_mean_mul21=[]

data7plant_mean_mul21=[]

data8plant_mean_mul21=[]

data9plant_mean_mul21=[]

data10plant_mean_mul21=[]



dataplant_mean_mul21=[data1plant_mean_mul21,

                        data2plant_mean_mul21,

                        data3plant_mean_mul21,

                        data4plant_mean_mul21,

                        data5plant_mean_mul21,

                        data6plant_mean_mul21,

                        data7plant_mean_mul21,

                        data8plant_mean_mul21,

                        data9plant_mean_mul21,

                        data10plant_mean_mul21]

dataall=[data1all,

        data2all,

        data3all,

        data4all,

        data5all,

        data6all,

        data7all,

        data8all,

        data9all,

        data10all]

data1plant=[]

data2plant=[]

data3plant=[]

data4plant=[]

data5plant=[]

data6plant=[]

data7plant=[]

data8plant=[]

data9plant=[]

data10plant=[]



data_plant=[data1plant,

            data2plant,

            data3plant,

            data4plant,

            data5plant,

            data6plant,

            data7plant,

            data8plant,

            data9plant,

            data10plant]

    

dot=0

for t in dataall:

    olddate=''

    date=''

    d=[]

    count=0   

    for i in t:

        number=0

        date = i[0]

        if count==0:

            count=1

        elif date == olddate:

            count=count+1

            d.append(i)

        elif date != olddate:

            #print(d)

            data_plant[dot].append(d)

            d=[]

            d.append(i)

            olddate=date

            count=1

    dot=dot+1



#data1plant_mean_mul21=[] #หาค่าเฉลี่ยในเวลาเดียวกันคูณ21ไปเลยเป็นการชดเชยข้อมูลที่หายไป

dot=0

for e in data_plant:

    for d in e:

        s=0

        v=0

        w=0

        o=0

        c=0

        e=0

        z=0

        r=0

        for t in d:

            c=c+1

            s=s+float(t[3])

            v=v+float(t[4])

            w=w+float(t[5])

            o=o+float(t[6])

            e=float(t[7])

            z=float(t[8])

            r=float(t[9])

        if(c==0):

            c=1

        s=(s/c)*21

        v=(v/c)*21

        w=(w/c)*21

        o=(o/c)*21

        dataplant_mean_mul21[dot].append([e,z,r,s,v,w,o])

    dot=dot+1





data2plant_mean_mul21



    
num=int((len(data1plant_mean_mul21))*(90/100))

train_data1=data1plant_mean_mul21[:num]

test_data1=data1plant_mean_mul21[num:]

test_data1_feature=[]

test_data1_target=[]

train_data1_feature=[]

train_data1_target=[]

for q in train_data1:

    train_data1_feature.append(q[:6])

    train_data1_target.append(q[6])

for u in test_data1:

    test_data1_feature.append(q[:6])

    test_data1_target.append(q[6])

train_data1_feature 

#train_data1_target



print(len(train_data1),len(test_data1),len(data1plant_mean_mul21))

num=int((len(data2plant_mean_mul21))*(90/100))

train_data2=data2plant_mean_mul21[:num]

test_data2=data2plant_mean_mul21[num:]

train_data2_feature=[]

train_data2_target=[]

for q in train_data2:

    train_data2_feature.append(q[:6])

    train_data2_target.append(q[6])

test_data2_feature=[]

test_data2_target=[]

for u in test_data2:

    test_data2_feature.append(q[:6])

    test_data2_target.append(q[6])



num=int((len(data3plant_mean_mul21))*(90/100))

train_data3=data3plant_mean_mul21[:num]

test_data3=data3plant_mean_mul21[num:]

train_data3_feature=[]

train_data3_target=[]

for q in train_data3:

    train_data3_feature.append(q[:6])

    train_data3_target.append(q[6])

test_data3_feature=[]

test_data3_target=[]

for u in test_data3:

    test_data3_feature.append(q[:6])

    test_data3_target.append(q[6])



num=int((len(data4plant_mean_mul21))*(90/100))

train_data4=data4plant_mean_mul21[:num]

test_data4=data4plant_mean_mul21[num:]

train_data4_feature=[]

train_data4_target=[]

for q in train_data4:

    train_data4_feature.append(q[:6])

    train_data4_target.append(q[6])

test_data4_feature=[]

test_data4_target=[]

for u in test_data4:

    test_data4_feature.append(q[:6])

    test_data4_target.append(q[6])



num=int((len(data5plant_mean_mul21))*(90/100))

train_data5=data5plant_mean_mul21[:num]

test_data5=data5plant_mean_mul21[num:]

train_data5_feature=[]

train_data5_target=[]

for q in train_data5:

    train_data5_feature.append(q[:6])

    train_data5_target.append(q[6])

test_data5_feature=[]

test_data5_target=[]

for u in test_data5:

    test_data5_feature.append(q[:6])

    test_data5_target.append(q[6])



num=int((len(data6plant_mean_mul21))*(90/100))

train_data6=data6plant_mean_mul21[:num]

test_data6=data6plant_mean_mul21[num:]

train_data6_feature=[]

train_data6_target=[]

for q in train_data6:

    train_data6_feature.append(q[:6])

    train_data6_target.append(q[6])

test_data6_feature=[]

test_data6_target=[]

for u in test_data6:

    test_data6_feature.append(q[:6])

    test_data6_target.append(q[6])



num=int((len(data7plant_mean_mul21))*(90/100))

train_data7=data7plant_mean_mul21[:num]

test_data7=data7plant_mean_mul21[num:]

train_data7_feature=[]

train_data7_target=[]

for q in train_data7:

    train_data7_feature.append(q[:6])

    train_data7_target.append(q[6])

test_data7_feature=[]

test_data7_target=[]

for u in test_data7:

    test_data7_feature.append(q[:6])

    test_data7_target.append(q[6])



num=int((len(data8plant_mean_mul21))*(90/100))

train_data8=data8plant_mean_mul21[:num]

test_data8=data8plant_mean_mul21[num:]

train_data8_feature=[]

train_data8_target=[]

for q in train_data1:

    train_data8_feature.append(q[:6])

    train_data8_target.append(q[6])

test_data8_feature=[]

test_data8_target=[]

for u in test_data8:

    test_data8_feature.append(q[:6])

    test_data8_target.append(q[6])



num=int((len(data9plant_mean_mul21))*(90/100))

train_data9=data9plant_mean_mul21[:num]

test_data9=data9plant_mean_mul21[num:]

train_data9_feature=[]

train_data9_target=[]

for q in train_data9:

    train_data9_feature.append(q[:6])

    train_data9_target.append(q[6])

test_data9_feature=[]

test_data9_target=[]

for u in test_data9:

    test_data9_feature.append(q[:6])

    test_data9_target.append(q[6])

    

num=int((len(data10plant_mean_mul21))*(90/100))

train_data10_feature

train_data10=data10plant_mean_mul21[:num]

test_data10=data10plant_mean_mul21[num:]

train_data10_feature=[]

train_data10_target=[]

for q in train_data10:

    train_data10_feature.append(q[:6])

    train_data10_target.append(q[6])

test_data10_feature=[]

test_data10_target=[]

for u in test_data10:

    test_data10_feature.append(q[:6])

    test_data10_target.append(q[6])



print(train_data9_feature[100],train_data9_target[100])
from sklearn.linear_model import LinearRegression

from sklearn import metrics

from catboost import CatBoostRegressor

import matplotlib.pyplot as plt  



model1 =LinearRegression().fit(train_data1_feature , train_data1_target)

m1r=model1.predict(test_data1_feature)

mae = metrics.mean_absolute_error(m1r, test_data1_target)

print("MAE:",mae)

mse = metrics.mean_squared_error(m1r, test_data1_target)

print("MSE:",mse)

rmse = np.sqrt(metrics.mean_squared_error(m1r, test_data1_target))

print("RMSE:",rmse)

print("Percent Error in random point:",((m1r[100]-test_data1_target[100])/test_data1_target[100])*100,"%")

y=[]

for i in range(0,len(m1r)):

    y.append(i)

    

plt.plot(y,m1r,label = "model") 

plt.plot(y,test_data10_target, label = "test value")

plt.xlabel('time') 

plt.ylabel('total yeild')  

plt.title('predict vs test data') 

  

plt.legend() 

  

# function to show the plot 

plt.show()



model2 =LinearRegression().fit(train_data2_feature , train_data2_target)

m2r=model2.predict(test_data2_feature)

mae = metrics.mean_absolute_error(m2r, test_data2_target)

print("MAE:",mae)

mse = metrics.mean_squared_error(m2r, test_data2_target)

print("MSE:",mse)

rmse = np.sqrt(metrics.mean_squared_error(m2r, test_data2_target))

print("RMSE:",rmse)

print("Percent Error in random point:",((m2r[100]-test_data2_target[100])/test_data2_target[100])*100,"%")



y=[]

for i in range(0,len(m2r)):

    y.append(i)

    

plt.plot(y,m2r,label = "model") 

plt.plot(y,test_data2_target, label = "test value")

plt.xlabel('time') 

plt.ylabel('total yeild')  

plt.title('predict vs test data') 

  

plt.legend() 

  

# function to show the plot 

plt.show()







model3 =LinearRegression().fit(train_data3_feature , train_data3_target)

m3r=model3.predict(test_data3_feature)

mae = metrics.mean_absolute_error(m3r, test_data3_target)

print("MAE:",mae)

mse = metrics.mean_squared_error(m3r, test_data3_target)

print("MSE:",mse)

rmse = np.sqrt(metrics.mean_squared_error(m3r, test_data3_target))

print("RMSE:",rmse)

print("Percent Error in random point:",((m3r[100]-test_data3_target[100])/test_data3_target[100])*100,"%")



y=[]

for i in range(0,len(m3r)):

    y.append(i)

    

plt.plot(y,m3r,label = "model") 

plt.plot(y,test_data3_target, label = "test value")

plt.xlabel('time') 

plt.ylabel('total yeild')  

plt.title('predict vs test data') 

  

plt.legend() 

  

# function to show the plot 

plt.show()







model4 =LinearRegression().fit(train_data4_feature , train_data4_target)

m4r=model4.predict(test_data4_feature)

mae = metrics.mean_absolute_error(m4r, test_data4_target)

print("MAE:",mae)

mse = metrics.mean_squared_error(m4r, test_data4_target)

print("MSE:",mse)

rmse = np.sqrt(metrics.mean_squared_error(m4r, test_data4_target))

print("RMSE:",rmse)

print("Percent Error in random point:",((m4r[100]-test_data4_target[100])/test_data4_target[100])*100,"%")



y=[]

for i in range(0,len(m4r)):

    y.append(i)

    

plt.plot(y,m4r,label = "model") 

plt.plot(y,test_data4_target, label = "test value")

plt.xlabel('time') 

plt.ylabel('total yeild')  

plt.title('predict vs test data') 

  

plt.legend() 

  

# function to show the plot 

plt.show()







model5 =LinearRegression().fit(train_data5_feature , train_data5_target)

m5r=model5.predict(test_data5_feature)

mae = metrics.mean_absolute_error(m5r, test_data5_target)

print("MAE:",mae)

mse = metrics.mean_squared_error(m5r, test_data5_target)

print("MSE:",mse)

rmse = np.sqrt(metrics.mean_squared_error(m5r, test_data5_target))

print("RMSE:",rmse)

print("Percent Error in random point:",((m5r[100]-test_data5_target[100])/test_data5_target[100])*100,"%")

y=[]

for i in range(0,len(m5r)):

    y.append(i)

    

plt.plot(y,m5r,label = "model") 

plt.plot(y,test_data5_target, label = "test value")

plt.xlabel('time') 

plt.ylabel('total yeild')  

plt.title('predict vs test data') 

  

plt.legend() 

  

# function to show the plot 

plt.show()





model6 =LinearRegression().fit(train_data6_feature , train_data6_target)

m6r=model6.predict(test_data6_feature)

mae = metrics.mean_absolute_error(m6r, test_data6_target)

print("MAE:",mae)

mse = metrics.mean_squared_error(m6r, test_data6_target)

print("MSE:",mse)

rmse = np.sqrt(metrics.mean_squared_error(m6r, test_data6_target))

print("RMSE:",rmse)

print("Percent Error in random point:",((m6r[100]-test_data6_target[100])/test_data6_target[100])*100,"%")

y=[]

for i in range(0,len(m6r)):

    y.append(i)

    

plt.plot(y,m6r,label = "model") 

plt.plot(y,test_data6_target, label = "test value")

plt.xlabel('time') 

plt.ylabel('total yeild')  

plt.title('predict vs test data') 

  

plt.legend() 

  

# function to show the plot 

plt.show()





model7 =LinearRegression().fit(train_data7_feature , train_data7_target)

m7r=model7.predict(test_data7_feature)

mae = metrics.mean_absolute_error(m7r, test_data7_target)

print("MAE:",mae)

mse = metrics.mean_squared_error(m7r, test_data7_target)

print("MSE:",mse)

rmse = np.sqrt(metrics.mean_squared_error(m7r, test_data7_target))

print("RMSE:",rmse)

print("Percent Error in random point:",((m7r[100]-test_data7_target[100])/test_data7_target[100])*100,"%")



y=[]

for i in range(0,len(m7r)):

    y.append(i)

    

plt.plot(y,m7r,label = "model") 

plt.plot(y,test_data7_target, label = "test value")

plt.xlabel('time') 

plt.ylabel('total yeild')  

plt.title('predict vs test data') 

  

plt.legend() 

  

# function to show the plot 

plt.show()





model8 =LinearRegression().fit(train_data8_feature , train_data8_target)

m8r=model8.predict(test_data8_feature)

mae = metrics.mean_absolute_error(m8r, test_data8_target)

print("MAE:",mae)

mse = metrics.mean_squared_error(m8r, test_data8_target)

print("MSE:",mse)

rmse = np.sqrt(metrics.mean_squared_error(m8r, test_data8_target))

print("RMSE:",rmse)

print("Percent Error in random point:",((m8r[100]-test_data8_target[100])/test_data8_target[100])*100,"%")

y=[]

for i in range(0,len(m8r)):

    y.append(i)

    

plt.plot(y,m8r,label = "model") 

plt.plot(y,test_data8_target, label = "test value")

plt.xlabel('time') 

plt.ylabel('total yeild')  

plt.title('predict vs test data') 

  

plt.legend() 

  

# function to show the plot 

plt.show()





model9 =LinearRegression().fit(train_data9_feature , train_data9_target)

m9r=model9.predict(test_data9_feature)

mae = metrics.mean_absolute_error(m9r, test_data9_target)

print("MAE:",mae)

mse = metrics.mean_squared_error(m9r, test_data9_target)

print("MSE:",mse)

rmse = np.sqrt(metrics.mean_squared_error(m9r, test_data9_target))

print("RMSE:",rmse)

print("Percent Error in random point:",((m9r[100]-test_data9_target[100])/test_data9_target[100])*100,"%")



y=[]

for i in range(0,len(m9r)):

    y.append(i)

    

plt.plot(y,m9r,label = "model") 

plt.plot(y,test_data9_target, label = "test value")

plt.xlabel('time') 

plt.ylabel('total yeild')  

plt.title('predict vs test data') 

  

plt.legend() 

  

# function to show the plot 

plt.show()





model10 =LinearRegression().fit(train_data10_feature , train_data10_target)

m10r=model10.predict(test_data10_feature)

mae = metrics.mean_absolute_error(m10r, test_data10_target)

print("MAE:",mae)

mse = metrics.mean_squared_error(m10r, test_data10_target)

print("MSE:",mse)

rmse = np.sqrt(metrics.mean_squared_error(m10r, test_data10_target))

print("RMSE:",rmse)

print("Percent Error in random point:",((m10r[100]-test_data10_target[100])/test_data10_target[100])*100,"%")

y=[]

for i in range(0,len(m10r)):

    y.append(i)

    

plt.plot(y,m10r,label = "model") 

plt.plot(y,test_data10_target, label = "test value")

plt.xlabel('time') 

plt.ylabel('total yeild')  

plt.title('predict vs test data') 

  

plt.legend() 

  

# function to show the plot 

plt.show()







day=3

num=int((day*24*60)/15)

data3day=data3plant_mean_mul21[:num]

print(len(data3day))

print(data3day[1])



data3day_pre=[]

data3day_test=[]



for y in data3day:

    data3day_pre.append(y[:6])

    data3day_test.append(y[6])

print(data3day_pre[1])

    
m10_3dr=model7.predict(data3day_pre)

#m10_3dr



mae = metrics.mean_absolute_error(m10_3dr, data3day_test)

print("MAE:",mae)

mse = metrics.mean_squared_error(m10_3dr, data3day_test)

print("MSE:",mse)

rmse = np.sqrt(metrics.mean_squared_error(m10_3dr, data3day_test))

print("RMSE:",rmse)

print("Percent Error in random point:",((m10_3dr[100]-data3day_test[100])/data3day_test[100])*100,"%")

y=[]

for i in range(0,len(m10_3dr)):

    y.append(i)

    

plt.plot(y,m10_3dr,label = "model") 

plt.plot(y,data3day_test, label = "test value")

plt.xlabel('time') 

plt.ylabel('total yeild')  

plt.title('predict vs test data every 15 minute') 

  

plt.legend() 

  

# function to show the plot 

plt.show()





print("Final result----> test data total yield in 72 hour:",data3plant_mean_mul21[num][6],"-->Predict data total yield:",m10_3dr[-1])

day=7

num=int((day*24*60)/15)

data7day=data2plant_mean_mul21[:num]

print(len(data7day))

print(data7day[1])



data7day_pre=[]

data7day_test=[]



for y in data7day:

    data7day_pre.append(y[:6])

    data7day_test.append(y[6])

print(data7day_pre[1])
m10_7dr=model7.predict(data7day_pre)

#m10_3dr



mae = metrics.mean_absolute_error(m10_7dr, data7day_test)

print("MAE:",mae)

mse = metrics.mean_squared_error(m10_7dr, data7day_test)

print("MSE:",mse)

rmse = np.sqrt(metrics.mean_squared_error(m10_7dr, data7day_test))

print("RMSE:",rmse)

print("Percent Error in random point:",((m10_7dr[100]-data3day_test[100])/data7day_test[100])*100,"%")

y=[]

for i in range(0,len(m10_7dr)):

    y.append(i)

    

plt.plot(y,m10_7dr,label = "model") 

plt.plot(y,data7day_test, label = "test value")

plt.xlabel('time') 

plt.ylabel('total yeild')  

plt.title('predict vs test data every 15 minute') 

  

plt.legend() 

  

# function to show the plot 

plt.show()



print("Final result----> test data total yield in 168 hour:",data3plant_mean_mul21[num][6],"-->Predict data total yield:",m10_7dr[-1])
