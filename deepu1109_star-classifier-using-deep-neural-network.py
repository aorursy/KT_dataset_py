import tensorflow as tf 

from tensorflow import keras

from sklearn import preprocessing, model_selection

import numpy as np 

import pandas as pd 
df = pd.read_csv('../input/star-dataset/6 class csv.csv')

df.head()
df.tail()
df.isnull().values.any()
df.corr()
import matplotlib.pyplot as plt



def plot_corr(df):

    corr = df.corr()

    fig,ax = plt.subplots(figsize = (6,6))

    ax.matshow(corr)

    plt.xticks(range(len(corr.columns)),corr.columns)

    plt.yticks(range(len(corr.columns)),corr.columns)

    

    

plot_corr(df) 
# star class:  Star Type

#     0     :  Brown Dwarf

#     1     :  Red Dwarf

#     2     :  White Dwarf

#     3     :  Main-Sequence

#     4     :  Supergiant

#     5     :  Hypergiant
brown_dwarf = len(df.loc[df['Star type'] == 0])

red_dwarf = len(df.loc[df['Star type'] == 1])

white_dwarf = len(df.loc[df['Star type'] == 2])

main_sequence = len(df.loc[df['Star type'] == 3])

supergiant = len(df.loc[df['Star type'] == 4])

hypergiant = len(df.loc[df['Star type'] == 5])



print("Brown dwarf = {} ".format(brown_dwarf))

print("Red dwarf  = {} ".format(red_dwarf))

print("White Dwarf = {} ".format(white_dwarf))

print("Main Sequence = {} ".format(main_sequence))

print("Supergiant= {} ".format(supergiant))

print("Hypergiant = {} ".format(hypergiant)) 

print("Total stars in the dataset = {} ".format(len(df)))
x = np.array(df.drop(['Star type', 'Star color','Spectral Class'],1))   # Excludes Star type, Star color and Spectral Class

y = np.array(df['Star type'], dtype ='float')                           # Only Star type column

y.shape = (len(y),1)                                                    # Shaping the star type column into a column vector
x_train ,x_test , y_train, y_test = model_selection.train_test_split(x,y, test_size = 0.3)    #Splits data into 70:30 ratio
x_f_train = preprocessing.scale(x_train)

x_f_test = preprocessing.scale(x_test)

y_f_train = y_train

y_f_test = y_test
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(200,activation = tf.nn.relu))

model.add(tf.keras.layers.Dense(300,activation = tf.nn.relu))

model.add(tf.keras.layers.Dense(6,activation = tf.nn.softmax))





model.compile(optimizer = tf.train.AdamOptimizer(),

       loss = 'sparse_categorical_crossentropy',

       metrics=['accuracy'])
model.fit(x_f_train,y_f_train, epochs = 100)
val_loss,val_acc = model.evaluate(x_f_test,y_f_test)

print("Loss % = {} , Accuracy % = {} ".format(val_loss*100,val_acc*100))
# [1,0,0,0,0,0] = Brown Dwarf

# [0,1,0,0,0,0] = Red Dwarf

# [0,0,1,0,0,0] = White Dwarf

# [0,0,0,1,0,0] = Main Sequence

# [0,0,0,0,1,0] = Supergiant

# [0,0,0,0,0,1] = Hypergiant



arr = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
z = np.round(model.predict(x_f_test))



print('_____________________________________________________')

print("Prediction : P-Class : O-Class ")



c1=c2=c3=c4=c5=c6=c7=c8=c9=c10=c11=c12=0  #counter variables



for i in range(0,len(z)):

    if np.array_equal(z[i],arr[0]):

        print("{}  :  {} :  {}".format(z[i],0,y_test[i]))

        c1+=1

    elif np.array_equal(z[i],arr[1]):

        print("{}  :  {} :  {}".format(z[i],1,y_test[i]))

        c2+=1

    elif np.array_equal(z[i],arr[2]):

        print("{}  :  {} :  {}".format(z[i],2,y_test[i]))

        c3+=1  

    elif np.array_equal(z[i],arr[3]):

        print("{}  :  {} :  {}".format(z[i],3,y_test[i]))

        c4+=1 

    elif np.array_equal(z[i],arr[4]):

        print("{}  :  {} :  {}".format(z[i],4,y_test[i]))

        c5+=1

    elif np.array_equal(z[i],arr[5]):

        print("{}  :  {} :  {}".format(z[i],5,y_test[i]))

        c6+=1    



print('_____________________________________________________')

print("Predicted NO. of Brown Dwarfs = {}".format(c1))

print("Predicted NO. of Red Dwarfs = {}".format(c2))

print("Predicted NO. of White Dwarfs = {}".format(c3))

print("Predicted NO. of Main Sequence stars = {}".format(c4))

print("Predicted NO. of Supergiants = {}".format(c5))

print("Predicted NO. of Hypergiants = {}".format(c6))

print("Total tested stars = {}".format(len(z)))



m = y_test



print('_____________________________________________________')



for i in range(0,len(m)):

    if m[i] == 0:

        c7+=1

    elif m[i] == 1 :

        c8+=1 

    elif m[i] == 2 :

        c9+=1 

    elif m[i] == 3 :

        c10+=1 

    elif m[i] == 4 :

        c11+=1 

    elif m[i] == 5 :

        c12+=1     





print("Original NO. of Brown Dwarfs = {}".format(c7))

print("Original NO. of Red Dwarfs = {}".format(c8))

print("Original NO. of White Dwarfs = {}".format(c9))

print("Original NO. of Main Sequence stars = {}".format(c10))

print("Original NO. of Supergiants = {}".format(c11))

print("Original NO. of Hypergiants = {}".format(c12))

print("Total tested stars = {}".format(len(x_test)))



print('_____________________________________________________')

print('Accuracy = {}%'.format((val_acc*100)))
df1 = pd.read_csv('../input/star-dataset/6 class csv.csv')







x1 = np.array(df1.drop(['Star color','Spectral Class'],1))

y1 = np.array(df1['Star type'], dtype ='float')

y1.shape = (len(y1),1)

c1 =0



for i in range(0,len(x1)):

    if x1[i][4] == 0:

        a = plt.scatter(x1[i][0],x1[i][3], s = 30 , c = 'green', marker = '.')

    elif x1[i][4]== 1:

        b = plt.scatter(x1[i][0],x1[i][3],s = 50 , c = 'red',marker = '.')

    elif x1[i][4]== 2:

        c = plt.scatter(x1[i][0],x1[i][3],s = 75 , c = 'gray',marker = '.')

    elif x1[i][4]== 3:

        d = plt.scatter(x1[i][0],x1[i][3],s = 90 , c = 'brown',marker = '.')     

    elif x1[i][4]== 4:

        e = plt.scatter(x1[i][0],x1[i][3],s = 100 , c = 'orange',marker = 'o') 

    elif x1[i][4]== 5:

        f = plt.scatter(x1[i][0],x1[i][3],s = 150 , c = 'blue',marker = 'o')

        

        

    c1+=1





print("Total Counted Stars = {}".format(c1)) 

plt.xlabel("Temperature(K)")

plt.ylabel("Absolute Magnitude(Mv)")

plt.title("H-R Diagram of Total Stars ")

plt.legend((a,b,c,d,e,f),('Brown Dwarf','Red Dwarf','White Dwarf','Main Sequence','Supergiant','Hypergiant'))

m = np.linspace(0,40000,100)

n = (0*m) -7.5

plt.plot(m,n,'--m')

f = np.linspace(0,6000,100)

g = (0*f)+15.4

plt.plot(f,g,'--c')

plt.scatter(5778,4.83,s = 95, c= 'yellow',marker = 'o' )

plt.gca().invert_xaxis()

plt.gca().invert_yaxis()

plt.show()
##  Yellow coloured point in the Main Sequence star band denotes our "SUN"
c2=0



for i in range(0,len(y_train)):

    if y_train[i] == 0:

        a = plt.scatter(x_train[i][0],x_train[i][3], s = 30 , c = 'green', marker = '.')

    elif y_train[i]== 1:

        b = plt.scatter(x_train[i][0],x_train[i][3],s = 50 , c = 'red',marker = '.')

    elif y_train[i]== 2:

        c = plt.scatter(x_train[i][0],x_train[i][3],s = 75 , c = 'gray',marker = '.')

    elif y_train[i]== 3:

        d = plt.scatter(x_train[i][0],x_train[i][3],s = 90 , c = 'brown',marker = '.')      

    elif y_train[i]== 4:

        e = plt.scatter(x_train[i][0],x_train[i][3],s = 100 , c = 'orange',marker = 'o') 

    elif y_train[i]== 5:

        f = plt.scatter(x_train[i][0],x_train[i][3],s = 150 , c = 'blue',marker = 'o')    

    c2+=1





print("Total Trained Stars = {}".format(c2)) 

plt.xlabel("Temperature(K)")

plt.ylabel("Absolute Magnitude(Mv)")

plt.title("H-R Diagram of Trained Stars ")

plt.legend((a,b,c,d,e,f),('Brown Dwarf','Red Dwarf','White Dwarf','Main Sequence','Supergiant','Hypergiant'))

m = np.linspace(0,40000,100)

n = (0*m) -7.5

plt.plot(m,n,'--m')

f = np.linspace(0,6000,100)

g = (0*f)+15.4

plt.plot(f,g,'--c')

plt.gca().invert_xaxis()

plt.gca().invert_yaxis()

plt.show()
c3=0



for i in range(0,len(y_test)):

    if y_test[i] == 0:

        a = plt.scatter(x_test[i][0],x_test[i][3], s = 30 , c = 'green', marker = '.')

    elif y_test[i]== 1:

        b = plt.scatter(x_test[i][0],x_test[i][3],s = 50 , c = 'red',marker = '.')

    elif y_test[i]== 2:

        c = plt.scatter(x_test[i][0],x_test[i][3],s = 75 , c = 'gray',marker = '.')

    elif y_test[i]== 3:

        d = plt.scatter(x_test[i][0],x_test[i][3],s = 90 , c = 'brown',marker = '.')   

    elif y_test[i]== 4:

        e = plt.scatter(x_test[i][0],x_test[i][3],s = 100 , c = 'orange',marker = 'o')

    elif y_test[i]== 5:

        f = plt.scatter(x_test[i][0],x_test[i][3],s = 150 , c = 'blue',marker = 'o')     

    c3+=1





print("Total Tested Stars = {}".format(c3)) 

plt.xlabel("Temperature(K)")

plt.ylabel("Absolute Magnitude(Mv)")

plt.title("H-R Diagram of Tested Stars ")

plt.legend((a,b,c,d,e,f),('Brown Dwarf','Red Dwarf','White Dwarf','Main Sequence','Supergiant','Hypergiant'))

m = np.linspace(0,40000,100)

n = (0*m) -7.5

plt.plot(m,n,'--m')

f = np.linspace(0,6000,100)

g = (0*f)+15.4

plt.plot(f,g,'--c')

plt.gca().invert_xaxis()

plt.gca().invert_yaxis()

plt.show()
c4 = 0



for i in range(0,len(z)):

    if np.array_equal(z[i],arr[0]):

        a = plt.scatter(x_test[i][0],x_test[i][3], s = 30 , c = 'green', marker = '.')

    elif np.array_equal(z[i],arr[1]):

        b = plt.scatter(x_test[i][0],x_test[i][3],s = 50 , c = 'red',marker = '.')

    elif np.array_equal(z[i],arr[2]):

        c = plt.scatter(x_test[i][0],x_test[i][3],s = 75 , c = 'gray',marker = '.')

    elif np.array_equal(z[i],arr[3]):

        d = plt.scatter(x_test[i][0],x_test[i][3],s = 90 , c = 'brown',marker = '.')    

    elif np.array_equal(z[i],arr[4]):

        e = plt.scatter(x_test[i][0],x_test[i][3],s = 100 , c = 'orange',marker = 'o')

    elif np.array_equal(z[i],arr[5]):

        f = plt.scatter(x_test[i][0],x_test[i][3],s = 150 , c = 'blue',marker = 'o')     

    c4+=1



print("Total Predicted Stars = {}".format(c4)) 

plt.xlabel("Temperature(K)")

plt.ylabel("Absolute Magnitude(Mv)")

plt.title("H-R Diagram of Stars ")

plt.legend((a,b,c,d,e,f),('Brown Dwarf','Red Dwarf','White Dwarf','Main Sequence','Supergiant','Hypergiant'))

m = np.linspace(0,40000,100)

n = (0*m) -7.5

plt.plot(m,n,'--m')

f = np.linspace(0,6000,100)

g = (0*f)+15.4

plt.plot(f,g,'--c')

plt.gca().invert_xaxis()

plt.gca().invert_yaxis()

plt.show()

print("Accuracy = {} %".format(val_acc*100))
## Clearly H-R Diagram of Tested Stars matched with the H-R Diagram of the Predicted Stars

## Thus,our Model learned effectively and hence can be applied to predict future data of stars

## The best part is that it can predict star type of thousands of stars at once in just a few seconds  

## This makes our model extremely valuable



## N.B. : More the data(accurate data) provided , better it will predict.
import pickle



data = np.hstack((x_f_train,y_f_train))    # Merging the label column (y_test) with the X_test  i.e the total training set



with open("6_class_model.pickle","wb") as f:

    pickle.dump( data , f)
######   End Of This Notebook . Thanks For Reading . Hope that u learned a bit of Astronomy and application of ML   ########