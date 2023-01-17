#decimal to binary conversion
def convert_num(num):
    str2=""
    if(num==0):
        return '000000000'
    i=9
    while(i):
        str2=str(int(num%2))+str2
        num=int(num/2)
        i-=1
    return str2

def convert_num2(num):
    str2=""
    if(num==0):
        return '000000000'
    i=9
    while(i):
        str2=str(int(num%2))+str2
        num=int(num/2)
        i-=1
    return str2
#Generate the data set of 1000 record with randome no, 
#col 1=first no in binary, 
#col 2 = second no in binary
#col 3 = sum of two no in binary
import numpy as np
import random
x=[]
y=[]
z=[]
sample_length=1000
for i in range(sample_length):
    x.append(random.randint(0,255))
    y.append(random.randint(0,255))
    z.append(x[i]+y[i])
for i in range(sample_length):
    x[i]=convert_num(x[i])
    y[i]=convert_num(y[i])
    z[i]=convert_num2(z[i])
    
import pandas as pd
df=pd.DataFrame([x,y,z]).T
df.head()
#convert dataset into single digit sum carry full adder 
#each 1 row of dataset will be converted into 8 row and 4 column where :
#first col*8 row = carry
#second col*8 row = first no.
#third col* 8 row = second no.
#fourth col* 8 row = sum
cc=[]
xx=[]
yy=[]
zz=[]
for i in df.index:
    t1=list(df.loc[i][0])
    t2=list(df.loc[i][1])
    t1=t1[::-1]
    t2=t2[::-1]
    temp=0
    for mp in range(len(t1)-1):
        xx.append(int(t1[mp]))
        yy.append(int(t2[mp]))
        cc.append(temp)
        if((t1[mp]=='1' and t2[mp]=='1') or (temp==1 and (t1[mp]=='1' or t2[mp]=='1')) ):
            #add carry as 1 when both digit are 1
            temp=1
        else:
            #add carry as 0 
            temp=0 
    xx.append(0)
    yy.append(0)
    cc.append(temp)
      
    for j in list(df.loc[i][2])[::-1]:
        zz.append(int(j))

#split x_train and y_train

y_t=pd.DataFrame(zz)#y_train
x_train=[]
for i in range(len(xx)):
    temp=np.asarray([cc[i],xx[i],yy[i]])
    temp=temp.reshape(1,3)# reshape to make it compatibal with LSTM layer(2 Dim)
    x_train.append(temp)      
x_train=np.asarray(x_train)
x_train.shape# 3 dim input for lstm 
#One hot encoding for outout label (1,0)
y0,y1=[],[]
for i in y_t[0]:
    if(i==1):
        y0.append(0)
        y1.append(1)
    else:
        y0.append(1)
        y1.append(0)
y_train=pd.DataFrame([y0,y1]).T
#create a model
from keras.layers import LSTM,Dense
from keras.models import Sequential
model=Sequential()
model.add(LSTM(64,batch_input_shape=(9,1,3)))
model.add(Dense(64,activation='relu'))
model.add(Dense(2,activation='softmax')) 
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics = ["accuracy"])
model.summary()
#fit the model
history=model.fit(x=x_train, y=y_train, batch_size=9, epochs=10,  validation_split=0.3)
#Prediction
cc=[]
xx=[]
yy=[]
zz=[]
print('Enter First no decimal :no should be in b/w 0-255 (8 bit no) : ')

first_i=int(input())
print('Enter second in decimal :no should be in b/w 0-255 (8 bit no) : ')
second_i=int(input())
#convert into binary
first_i_b,second_i_b=convert_num(first_i),convert_num(second_i)
df=pd.DataFrame([convert_num(first_i),convert_num(second_i)]).T
#convert query format which is compatible to model
for i in df.index:
    t1=list(df.loc[i][0])
    t2=list(df.loc[i][1])
    t1=t1[::-1]
    t2=t2[::-1]
    temp=0
    for mp in range(len(t1)-1):
        xx.append(int(t1[mp]))
        yy.append(int(t2[mp]))
        cc.append(temp)
        if((t1[mp]=='1' and t2[mp]=='1') or (temp==1 and (t1[mp]=='1' or t2[mp]=='1')) ):
            #add carry as 1 when both digit are 1
            temp=1
        else:
            #add carry as 0 
            temp=0 
    xx.append(0)
    yy.append(0)
    cc.append(temp)

x_test=[]
for i in range(len(xx)):
    temp=np.asarray([cc[i],xx[i],yy[i]])
    temp=temp.reshape(1,3)
    #x_train.append(temp)     
    x_test.append(temp)

x_test=np.asarray(x_test)
y_pred=model.predict(x_test)

#function to convert binary to decimal
def con_b_to_d(x):
    lt=len(x)
    y=0
    for i in x:
        y=y+(2**(lt-1))*(int(i))
        lt=lt-1
    return y

output=[]
for i in y_pred:
    if(i[0]>i[1]):
        output.append(0)
    elif(i[0]<=i[1]):
        output.append(1)
output=[str(i) for i in output[::-1]]
output=''.join(output)

#print output
print('First no in binary  :',first_i_b , '    First no. in Decimal  :', first_i)
print('Second no in binary :',second_i_b, '    Second no. in Decimal :',second_i)
print('Sum in binary       :',output,     '    Sum in Decimal        :',con_b_to_d(output))
