import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn import svm

import numpy as np

from sklearn import linear_model

from sklearn.metrics import accuracy_score

from tabulate import tabulate

np.set_printoptions(precision=20)



import math



fi = open("../input/diabetes.csv")

fi.readline()

data = np.loadtxt(fi, delimiter = ",")

data.shape

split = int(0.75 * data.shape[0])#split training/test data to 0.75/0.25

train = data[:split]; test = data[split:]



trainip = train[:,:-1]; trainop = train[:,-1]

testip = test[:,:-1]; testop = test[:,-1]
import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn import svm

import numpy as np

from sklearn import linear_model

from sklearn.metrics import accuracy_score

from tabulate import tabulate

np.set_printoptions(precision=20)



import math



fi = open("../input/diabetes.csv")

fi.readline()

data = np.loadtxt(fi, delimiter = ",")

data.shape

split = int(0.75 * data.shape[0])#split training/test data to 0.75/0.25

train = data[:split]; test = data[split:]



trainip = train[:,:-1]; trainop = train[:,-1]

testip = test[:,:-1]; testop = test[:,-1]

########################## Pregnant 0 ########################

tmp = np.unique(trainip[:,0])

prg1 = np.zeros((len(tmp),4), dtype = np.float16)

prg1[:,0] = tmp

for i in range(0, len(trainip)):

    tp1 = np.where(prg1[:,0] == trainip[i,0])

    tp2 = int(tp1[0])

    prg1[tp2,1] +=1

    if (trainop[i] == 1):

        prg1[tp2,2] +=1

#get the diabetes fractions

prg1[:,3] = prg1[:,2] / prg1[:,1]

#print("Num Pregnancy, count in dataset, diabetes diagnosis count, fraction diagnosis")

#print(prg1)

#set the regression model

regr = linear_model.LinearRegression()

#select model minus outliers or best fit model

#print("Discount pregnancy for values = 0, 13, 14, 15, 17")

a = prg1[:,np.newaxis,0]; b = prg1[:,np.newaxis,3]

c = prg1[1:13,np.newaxis,0]; d = prg1[1:13,np.newaxis,3]

#perform linear regression

regr.fit(c, d)

#display coeff and intercept for calculating model

#print('Coefficient: ', regr.coef_," and intercept: ", regr.intercept_)



ps = plt.GridSpec(5,3, wspace=0.4, hspace=0.4)

fig = plt.figure(1, figsize=(5,8))

fig.add_subplot(ps[:2,0:])

plt.grid(True)

plt.axis([0, 20, 0, 1])

plt.ylabel('% Diabetes diagnosis')

plt.xlabel('Number of pregnancies')

plt.title('Pregnancy %')

plt.scatter(a,b, color='red')



fig.add_subplot(ps[3:,0:])

plt.grid(True)

plt.axis([0, 20, 0, 1])

plt.ylabel('% Diabetes diagnosis')

plt.xlabel('Number of pregnancies')

plt.title('Pregnancy % without outliers')

plt.scatter(c,d, color='black')

plt.plot(c, regr.predict(c), color='blue',linewidth=3)

#1st Eqn

x11 = float(regr.coef_) ; x12 = float(regr.intercept_)    #Preg coef regr

x11 = round(x11,8); x12 = round(x12,8)



header1 = ["Pregnancy", "Count", "Diagnosed count", "Fraction"]

print(tabulate(prg1, header1, tablefmt="fancy_grid"))

plt.show()
######################  Plasma  1 ###########################################

#lower, upper band limits, count, diagnosis count, % diagnosis

plas = np.zeros((20,5), dtype = np.float16)

tmp = np.arange(0, 200, 10); plas[:,0] = tmp

tmp = np.arange(9, 209, 10); plas[:,1] = tmp



#print("Lower, upper bands, band count, diabetes count, fraction count")



for i in range(0, len(trainip)):

    tp1 = np.where((plas[:,0] <= trainip[i,1]) & (plas[:,1] >= trainip[i,1]))

    tp2 = int(tp1[0])

    plas[tp2,2] +=1

    if (trainop[i] == 1):

        plas[tp2,3] +=1



#Avoid zero numerator division

plas[7:20,4] = plas[7:20,3] / plas[7:20,2]        

regr = linear_model.LinearRegression()

a = plas[7:20,np.newaxis,0]; b = plas[7:20,np.newaxis,4]

regr.fit(a, b)

ps = plt.GridSpec(5,3, wspace=0.4, hspace=0.4)

fig = plt.figure(2, figsize=(5,8))

fig.add_subplot(ps[:2,0:])

plt.grid(True)

plt.axis([0, 210, 0, 1])

plt.ylabel('% Diabetes diagnosis')

plt.xlabel('Plasma Readings')

plt.title('Plasma %, Bin width = 10')

plt.scatter(a,b, color='red')

plt.plot(a, regr.predict(a), color='red',linewidth=3)

#######################

#Bands are too high despite good regression line

#Using smaller bin size

#######################



#Plasma = 0 appears to be invalid and so we can start with the 1st count

#Plasma = 44

    

plas1 = np.zeros((32,7), dtype = np.float16)

tmp = np.arange(41, 200, 5); plas1[:,0] = tmp

tmp = np.arange(45, 205, 5); plas1[:,2] = tmp



#print("Lower, mid, upper bands, band count, diabetes count, fraction count, eqn %")

for i in range(0, len(trainip)):

    tp1 = np.where((plas1[:,0] <= trainip[i,1]) & (plas1[:,2] >= trainip[i,1]))

    #print(trainip[i,1], tp1, len(tp1))

    if (trainip[i,1]!= 0):

        tp2 = int(tp1[0])

        plas1[tp2,3] +=1

        if (trainop[i] == 1):

            plas1[tp2,4] +=1

# midpoint calculation

plas1[:,1] = plas1[:,0] + ((plas1[:,2] - plas1[:,0])/2)



#calculate the fraction for diabetes diagnosis

for i in range(0,len(plas1)):

    if (plas1[i,4] !=0):

        plas1[i,5] = plas1[i,4] / plas1[i,3]

    

for i in range(0,len(plas1)):

    plas1[i,6] = math.exp(-0.00035*(math.pow((plas1[i,1]-175),2) + (0.95*plas1[i,1]) + 250))





x21 =  -0.00035; x22 = -175; x23 = 0.95; x24 = 250  



c = plas1[:,np.newaxis,1]; d = plas1[:,np.newaxis,5]



fig.add_subplot(ps[3:,0:])

plt.grid(True)

plt.axis([0, 210, 0, 1])

plt.ylabel('% Diabetes diagnosis')

plt.xlabel('Plasma Readings')

plt.title('Plasma %, Bin width = 5')

plt.scatter(c,d, color='black')

plt.scatter(plas1[:,1], plas1[:,6], color='blue')

plt.plot(plas1[:,1], plas1[:,6], color='blue',linewidth=3)

print("Lc = Lower Count, UC = Upper Count, mid = Average of upper and lower")

print("BC = Band Count is the number within the band above")

print("DC = Diabetes diagnosed within the bands")

print("Frac = Diagnosed count/Band Count")

print("Func = Diagnosed count calculated from function obtained from the graph")

header2 = ["LC", "Mid", "UC", "BC", "DC", "Frac", "Func"]

print(tabulate(plas1, header2, tablefmt="fancy_grid"))

plt.show()
###########################  BP 2 ##########################################        





tmp = np.unique(trainip[:,2])

bp = np.zeros((len(tmp),4), dtype = np.float16)

bp[:,0] = tmp

for i in range(0, len(trainip)):

    tp1 = np.where(bp[:,0] == trainip[i,2])

    tp2 = int(tp1[0])

    bp[tp2,1] +=1

    if (trainop[i] == 1):

        bp[tp2,2] +=1



for i in range(0,len(tmp)):

    if (bp[i,2] !=0):

        bp[i,3] = bp[i,2] / bp[i,1]



#print(bp)

a = bp[:,np.newaxis,0]; b = bp[:,np.newaxis,3]



ps = plt.GridSpec(5,3, wspace=0.4, hspace=0.4)

fig = plt.figure(3, figsize=(5,8))

fig.add_subplot(ps[:2,0:])

plt.axis([0, 125, 0, 1])

plt.grid(True)

plt.scatter(a,b, color='red')

plt.ylabel('% Diabetes diagnosis')

plt.xlabel('BP Readings')

plt.title('BP % Scatter with Outliers')



#remove outliers and points that reduces a good lin regr

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 32, 34, 35, 37, 38, 39, 42]



x = np.array(x)

bp1 = np.delete(bp, x, axis=0)



regr = linear_model.LinearRegression()

c = bp1[:,np.newaxis,0]; d = bp1[:,np.newaxis,3]

#perform linear regression

regr.fit(c, d)

#print('Coefficient: ', regr.coef_," and intercept: ", regr.intercept_)



fig.add_subplot(ps[3:,0:])

plt.grid(True)

plt.axis([0, 125, 0, 1])

plt.ylabel('% Diabetes diagnosis')

plt.xlabel('BP Readings')

plt.title('BP %, lin regr w/o Outliers')



plt.scatter(c,d, color='black')

plt.plot(c, regr.predict(c), color='blue',linewidth=3)



#1st Eqn

x31 = float(regr.coef_) ; x32 = float(regr.intercept_)    #BP coef regr

x31 = round(x31,8); x32 = round(x32,8)



header3 = ["BP", "Count", "Diagnosed count", "Fraction"]

print(tabulate(bp, header3, tablefmt="fancy_grid"))

plt.show()
################################## bmi 5 ########################################



tmp = np.unique(trainip[:,5])

bmi = np.zeros((len(tmp),4), dtype = np.float64)

bmi[:,0] = tmp

for i in range(0, len(trainip)):

    tp1 = np.where(bmi[:,0] == trainip[i,5])

    tp2 = int(tp1[0])

    bmi[tp2,1] +=1

    if (trainop[i] == 1):

        bmi[tp2,2] +=1

#get the diabetes fractions

for i in range(0,len(tmp)):

    if (bmi[i,2] !=0):

        bmi[i,3] = bmi[i,2] / bmi[i,1]



a = bmi[:,np.newaxis,0]; b = bmi[:,np.newaxis,3]

#print(bmi[0:30])

#print("The 1st 30 values shows that bmi should start at 18 and not 0 ")



# 5 grid space tall, 3 wide

ps = plt.GridSpec(5,3, wspace=0.4, hspace=0.4)

fig = plt.figure(4, figsize=(5,8))

fig.add_subplot(ps[:2,0:])

plt.axis([0, 75, 0, 1])

plt.grid(True)

plt.scatter(a,b, color='red')

plt.ylabel('% Diabetes diagnosis')

plt.xlabel('BMI Readings')

plt.title('BMI % Scatter ')





plt.scatter(a,b, color='red')

#We need to band limit the values for clarity

#bmi values = 0 are discounted



############################



#lower, middle, upper band limits, count, diagnosis count, % diagnosis

bmi1 = np.zeros((17,6), dtype = np.float64)

tmp = np.arange(18, 68, 3); bmi1[:,0] = tmp

tmp = np.arange(20.9, 70.9, 3); bmi1[:,2] = tmp



#print(bmi1)

for i in range(0, len(trainip)):

    tp1 = np.where((bmi1[:,0] <= trainip[i,5]) & (bmi1[:,2] >= trainip[i,5]))

    #print(trainip[i,5], tp1, len(tp1))

    if (trainip[i,5]!= 0):

        tp2 = int(tp1[0])

        bmi1[tp2,3] +=1

        if (trainop[i] == 1):

            bmi1[tp2,4] +=1



# midpoint calculation

bmi1[:,1] = bmi1[:,0] + ((bmi1[:,2] - bmi1[:,0])/2)



#calculate the fraction for diabetes diagnosis

for i in range(0,len(bmi1)):

    if (bmi1[i,4] !=0):

        bmi1[i,5] = bmi1[i,4] / bmi1[i,3]



a = bmi1[:,np.newaxis,1]; b = bmi1[:,np.newaxis,5]

#print(a, b)



regr = linear_model.LinearRegression()

#perform linear regression

#print(bmi1)

c = bmi1[0:12,np.newaxis,1]; d = bmi1[0:12,np.newaxis,5]

#print(c,d)

regr.fit(c, d)



fig.add_subplot(ps[3:,0:])

plt.grid(True)

plt.axis([0, 70, 0, 1])

plt.ylabel('% Diabetes diagnosis')

plt.xlabel('BMI Readings')

plt.title('BMI %, Scatter, Bin=2.9, \n lin Regr in blue over red outliers ')



plt.scatter(a,b, color='red')



plt.scatter(c,d, color='black')

plt.plot(c, regr.predict(c), color='blue',linewidth=3)





x41 = float(regr.coef_) ; x42 = float(regr.intercept_)    #bmi coef regr

x41 = round(x41,8); x42 = round(x42,8)



print("LC = Lower Count, UC = Upper Count, Mid = Average of upper and lower")

print("BC = Band Count is the number within the band above")

print("DC = Diabetes diagnosed within the bands")

print("Frac = Diagnosed count/Band Count")

header4 = ["LC", "Mid", "UC", "BC", "DC", "Frac"]

print(tabulate(bmi1, header4, tablefmt="fancy_grid"))

plt.show()
########################  dpf 6 ########################################





dpf = np.zeros((13,6), dtype = np.float16)

tmp = np.arange(0.001, 2.601, 0.2); dpf[:,0] = tmp

tmp = np.arange(0.2, 2.8, 0.2); dpf[:,2] = tmp





for i in range(0, len(trainip)):

    tp1 = np.where((dpf[:,0] <= trainip[i,6]) & (dpf[:,2] >= trainip[i,6]))

    #print(trainip[i,6], tp1, len(tp1))

    if (trainip[i,6]!= 0):

        tp2 = int(tp1[0])

        dpf[tp2,3] +=1

        if (trainop[i] == 1):

            dpf[tp2,4] +=1



# midpoint calculation

dpf[:,1] = dpf[:,0] + ((dpf[:,2] - dpf[:,0])/2)



#calculate the fraction for diabetes diagnosis

for i in range(0,len(dpf)):

    if (dpf[i,4] !=0):

        dpf[i,5] = dpf[i,4] / dpf[i,3]



a = dpf[:,np.newaxis,1]; b = dpf[:,np.newaxis,5]

#print(dpf)



ps = plt.GridSpec(5,3, wspace=0.4, hspace=0.4)

fig = plt.figure(5, figsize=(5,8))

fig.add_subplot(ps[:2,0:])

plt.axis([0, 3, 0, 1])

plt.grid(True)

plt.scatter(a,b, color='red')

plt.ylabel('% Diabetes diagnosis')

plt.xlabel('DPF Readings, scatter')

plt.title('DPF % Scatter, bins 0.199')



#remove outliers

x = [7, 8, 11, 12]

x = np.array(x)



dpf1 = np.delete(dpf,(x), axis=0)

c = dpf1[:,np.newaxis,1]; d = dpf1[:,np.newaxis,5]



regr = linear_model.LinearRegression()

regr.fit(c, d)



fig.add_subplot(ps[3:,0:])

plt.grid(True)

plt.axis([0, 3, 0, 1])

plt.ylabel('% Diabetes diagnosis')

plt.xlabel('DPF Readings')

plt.title('DPF %, Scatter, Bin=0.199, \n lin Regr w/o outliers ')

plt.scatter(c,d, color='black')

plt.plot(c, regr.predict(c), color='blue',linewidth=3)



#print('DPF Coefficient: ', regr.coef_," and intercept: ", regr.intercept_)

x51 = float(regr.coef_ ); x52 = float(regr.intercept_)    #bmi coef regr

x51 = round(x51,8); x52 = round(x52,8)

print("LC = Lower Count, UC = Upper Count, Mid = Average of upper and lower")

print("BC = Band Count is the number within the band above")

print("DC = Diabetes diagnosed within the bands")

print("Frac = Diagnosed count/Band Count")

header5 = ["LC", "Mid", "UC", "BC", "DC", "Frac"]

print(tabulate(dpf, header5, tablefmt="fancy_grid"))

plt.show()
################################### Age ###################################



age = np.zeros((21,7), dtype = np.float16)

tmp = np.arange(21, 83, 3); age[:,0] = tmp

tmp = np.arange(23, 85, 3); age[:,2] = tmp



for i in range(0, len(trainip)):

    tp1 = np.where((age[:,0] <= trainip[i,7]) & (age[:,2] >= trainip[i,7]))

    #print(trainip[i,1], tp1, len(tp1))

    if (trainip[i,7]!= 0):

        tp2 = int(tp1[0])

        age[tp2,3] +=1

        if (trainop[i] == 1):

            age[tp2,4] +=1



# midpoint calculation

age[:,1] = age[:,0] + ((age[:,2] - age[:,0])/2)



#calculate the fraction for diabetes diagnosis

for i in range(0,len(age)):

    if (age[i,4] !=0):

        age[i,5] = age[i,4] / age[i,3]





#Due to aqrt of -ve numbers, the graph will stop before zero

for i in range(0,16):

    age[i,6] = math.pow((0.2704 - ((0.2704 / (24 * 24)) * math.pow((age[i,1] - 45),2))),0.5)



e1 = age[:,np.newaxis,1]; e2 = age[:,np.newaxis,5]; e3 = age[:,np.newaxis,6]



ps = plt.GridSpec(5,3, wspace=0.4, hspace=0.4)

fig = plt.figure(8, figsize=(8,5))

fig.add_subplot(ps[:,:])

plt.grid(True)

plt.axis([0, 83, 0, 1])

plt.ylabel('% Diabetes diagnosis')

plt.xlabel('Age Readings')

plt.title('Age %, Bin width = 2')

plt.scatter(e1,e2, color='red')

plt.plot(e1, e2, color='red',linewidth=3)

plt.scatter(e1,e3, color='blue')

plt.plot(e1, e3, color='blue',linewidth=3)



print("BC = Band Count is the number within the band above")

print("DC = Diabetes diagnosed within the bands")

print("Frac = Diagnosed count/Band Count")

print("Func = Diagnosed count calculated from function obtained from the graph")

header2 = ["LC", "Mid", "UC", "BC", "DC", "Frac", "Func"]

print(tabulate(age, header2, tablefmt="fancy_grid"))

plt.show()
##############################################################################



print("\nThe equations for the line graphs are as follows ")

print("1st Pregnancy Eqn, y = ",x11,"x + ",x12)

#print("2nd Plasma Eqn, y = 1 / (",x21,"x^2 + ",x22,"x + ",x23,")")

print("2nd Plasma Eqn, y = exp^(-0.00035(x-175)^2 + 0.95x + 250 )")

print("3rd BP Eqn, y = ",x31,"x + ",x32)

print("4th BMI Eqn, y = ",x41,"x + ",x42)

print("5th DPF Eqn, y = ",x51,"x + ",x52)

print("8th Age Eqn, y = math.pow(0.2704 - (0.00090278 * math.pow((x - 45),2)),0.5)   ")



trainip1 = np.zeros((len(trainip),6), dtype = np.float64)

trainop1 = trainop

testip1 = np.zeros((len(testip),6), dtype = np.float64)

testop1 = testop



    

trainip1[:,0] = (trainip[:,0] * x11) + x12

for i in range(0,len(trainip1)):

    trainip1[i,1] = math.exp(-0.00035*(math.pow((trainip[i,1]-175),2) + (0.95*trainip[i,1]) + 250))

trainip1[:,2] = (trainip[:,2] * x31) + x32

trainip1[:,3] = (trainip[:,5] * x41) + x42

trainip1[:,4] = (trainip[:,6] * x51) + x52



for i in range(0,len(trainip1)):

    if (trainip[i,7] < 70):

        trainip1[i,5] = math.pow((0.2704 - ((0.2704 / (24 * 24)) * math.pow((trainip[i,7] - 45),2))),0.5)



#Eliminate -ve numbers by setting floor limits   

for i in range(0, len(trainip)):

    if (trainip[i,1] <= 75):

        trainip1[i,1] = 0

        

for i in range(0, len(trainip)):

    if (trainip[i,2] <= 40):

        trainip1[i,2] = 0

for i in range(0, len(trainip)):

    if (trainip[i,5] <= 19):

        trainip1[i,3] = 0



#processing the p.d.f. for the test set

testip1[:,0] = (testip[:,0] * x11) + x12

for i in range(0,len(testip1)):

    testip1[i,1] = math.exp(-0.00035*(math.pow((testip[i,1]-175),2) + (0.95*testip[i,1]) + 250))

testip1[:,2] = (testip[:,2] * x31) + x32

testip1[:,3] = (testip[:,5] * x41) + x42

testip1[:,4] = (testip[:,6] * x51) + x52



for i in range(0,len(testip1)):

    if(testip[i,7] < 70):

        testip1[i,5] = math.pow((0.2704 - ((0.2704 / (24 * 24)) * math.pow((testip[i,7] - 45),2))),0.5)





#Eliminate -ve numbers by setting floor limits for the test set

for i in range(0, len(testip)):

    if (testip[i,1] <= 75):

        testip1[i,1] = 0

        

for i in range(0, len(testip)):

    if (testip[i,2] <= 40):

        testip1[i,2] = 0

for i in range(0, len(testip)):

    if (testip[i,5] <= 19):

        testip1[i,3] = 0



fi = open ("test.tst", "w")

for i1 in range(0, len(testip)):

    fi.write ('\n')

    fi.write ('%s ' %(testip1[i1]))

fi.close()    

    

fo = open ("train.tst", "w")

for i1 in range(0, len(trainip1)):

    fo.write ('\n')    

    fo.write ('%s ' %(trainip1[i1]))

fo.close()



#clf = svm.SVC(gamma=1, C=10)#RBF Settings

#clf = svm.SVC(gamma=1, C= 5)#RBF Settings

clf = svm.SVC(kernel="linear", gamma=1, C= 0.5)#RBF Settings

clf.fit(trainip1,trainop1)

print("The training and test samples sizes are ")

print(trainip1.shape, trainop1.shape, testip1.shape, testop1.shape)



target = clf.predict(testip1)

print('\nAccuracy is ',accuracy_score(testop1,target ))
############################# Tricep 3 #####################################



tmp = np.unique(trainip[:,3])

tri = np.zeros((len(tmp),6), dtype = np.float64)

tri[:,0] = tmp

for i in range(0, len(trainip)):

    tp1 = np.where(tri[:,0] == trainip[i,3])

    tp2 = int(tp1[0])

    tri[tp2,1] +=1

    if (trainop[i] == 1):

        tri[tp2,2] +=1

#get the diabetes fractions

for i in range(0,len(tmp)):

    if (tri[i,2] !=0):

        tri[i,3] = tri[i,2] / tri[i,1]



a = tri[:,np.newaxis,0]; b = tri[:,np.newaxis,3]



ps = plt.GridSpec(5,3, wspace=0.4, hspace=0.4)

fig = plt.figure(6, figsize=(5,8))

fig.add_subplot(ps[:2,0:])

plt.axis([0, 70, 0, 1])



c = tri[5:41,np.newaxis,0]; d = tri[5:41,np.newaxis,3]

#print(c,d)



regr = linear_model.LinearRegression()

regr.fit(c, d)



plt.grid(True)

plt.scatter(a,b, color='red')

plt.scatter(c,d, color='black')

plt.plot(c, regr.predict(c), color='blue',linewidth=3)

plt.ylabel('% Diabetes diagnosis')

plt.xlabel('Tricep Readings')

plt.title('Tricep % Scatter \n lin Regr (w & w/o Outliers) ')



x611 = float(regr.coef_) ; x612 = float(regr.intercept_)    #tri coef regr

x611 = round(x611,8); x612 = round(x612,8)



##################



tri1 = np.zeros((12,6), dtype = np.float32)

tmp = np.arange(6, 66, 5); tri1[:,0] = tmp

tmp = np.arange(10, 70, 5); tri1[:,2] = tmp





for i in range(0, len(trainip)):

    tp1 = np.where((tri1[:,0] <= trainip[i,3]) & (tri1[:,2] >= trainip[i,3]))

    #print(trainip[i,3], tp1, len(tp1))

    if (trainip[i,3]!= 0):

        tp2 = int(tp1[0])

        tri1[tp2,3] +=1

        if (trainop[i] == 1):

            tri1[tp2,4] +=1



# midpoint calculation

tri1[:,1] = tri1[:,0] + ((tri1[:,2] - tri1[:,0])/2)



#calculate the fraction for diabetes diagnosis

for i in range(0,len(tri1)):

    if (tri1[i,4] !=0):

        tri1[i,5] = tri1[i,4] / tri1[i,3]



a = tri1[:,np.newaxis,1]; b = tri1[:,np.newaxis,5]

#print(tri1)



a = tri1[:,np.newaxis,1]; b = tri1[:,np.newaxis,5]

c = tri1[0:9,np.newaxis,1]; d = tri1[0:9,np.newaxis,5]



fig.add_subplot(ps[3:,0:])

plt.axis([0, 70, 0, 1])

plt.grid(True)



regr = linear_model.LinearRegression()

regr.fit(c, d)

#print(c,d)

#print(regr.coef_,regr.intercept_) 



plt.scatter(a,b, color='red')

plt.scatter(c,d, color='black')

plt.plot(c, regr.predict(c), color='blue',linewidth=3)

plt.ylabel('% Diabetes diagnosis')

plt.xlabel('Tricep Readings')

plt.title('Tricep % Scatter, bin = 4, \n lin Regr (w & w/o Outliers) ')



x621 = float(regr.coef_) ; x622 = float(regr.intercept_)    #tri coef regr

x621 = round(x621,8); x622 = round(x622,8)



print("LC = Lower Count, UC = Upper Count, Mid = Average of upper and lower")

print("BC = Band Count is the number within the band above")

print("DC = Diabetes diagnosed within the bands")

print("Frac = Diagnosed count/Band Count")

header6 = ["LC", "Mid", "UC", "BC", "DC", "Frac"]

print(tabulate(tri1, header6, tablefmt="fancy_grid"))

plt.show()



##############################################################################################



trainip2 = np.zeros((len(trainip),7), dtype = np.float64)

trainop2 = trainop

testip2 = np.zeros((len(testip),7), dtype = np.float64)

testop2 = testop



trainip2[0:len(trainip),0:6] = trainip1.copy(); trainop2 = trainop.copy()

testip2[0:len(testip),0:6] = testip1.copy(); testop2 = testop.copy()



print("\nThe equations for the line graphs are as follows ")

print("1st Pregnancy Eqn, y = ",x11,"x + ",x12)

#print("2nd Plasma Eqn, y = 1 / (",x21,"x^2 + ",x22,"x + ",x23,")")

print("2nd Plasma Eqn, y = exp^(-0.00035(x-175)^2 + 0.95x + 250 )")

print("3rd BP Eqn, y = ",x31,"x + ",x32)

print("4th BMI Eqn, y = ",x41,"x + ",x42)

print("5th DPF Eqn, y = ",x51,"x + ",x52)

print("8th Age Eqn, y = math.pow(0.2704 - (0.00090278 * math.pow((x - 45),2)),0.5)   ")



print("6th Tricep skin thickness, y = ",x621,"x + ",x622)







trainip2[:,6] = (trainip[:,3] * x621) + x622

testip2[:,6] = (testip[:,3] * x621) + x622



tp3 = np.where(trainip[:,3] == 0)

trmp3 = trainip.copy()

trmp3 = np.delete(trmp3, tp3, axis=0)

trainip2 = np.delete(trainip2, tp3, axis=0)

trainop2 = np.delete(trainop2, tp3, axis=0)



tp4 = np.where(testip[:,3] == 0)

tsp3 = testip.copy()

tsp3 = np.delete(tsp3, tp4, axis=0)

testip2 = np.delete(testip2, tp4, axis=0)

testop2 = np.delete(testop2, tp4, axis=0)





clf = svm.SVC(gamma=1, C= 12)#RBF Settings

#clf = svm.SVC(kernel="linear", gamma=1, C= 5)#linear Settings

clf.fit(trainip2,trainop2)

print("The training and test sample sets has shrunk to ")

print(trainip2.shape, trainop2.shape, testip2.shape, testop2.shape)



target1 = clf.predict(testip2)

print('\n The new and improved Accuracy is now',accuracy_score(testop2,target1 ))
########################### Insulin Serum 4 ################################





tmp = np.unique(trainip[:,4])

ser = np.zeros((len(tmp),4), dtype = np.float64)

ser[:,0] = tmp

for i in range(0, len(trainip)):

    tp1 = np.where(ser[:,0] == trainip[i,4])

    tp2 = int(tp1[0])

    ser[tp2,1] +=1

    if (trainop[i] == 1):

        ser[tp2,2] +=1

#get the diabetes fractions

for i in range(0,len(tmp)):

    if (ser[i,2] !=0):

        ser[i,3] = ser[i,2] / ser[i,1]



a = ser[:,np.newaxis,0]; b = ser[:,np.newaxis,3]



ps = plt.GridSpec(5,3, wspace=0.4, hspace=0.4)

fig = plt.figure(7, figsize=(5,8))

fig.add_subplot(ps[:2,0:])

plt.axis([0,850, 0, 1])



c = ser[5:41,np.newaxis,0]; d = ser[5:41,np.newaxis,3]



plt.grid(True)

plt.scatter(a,b, color='red')

plt.ylabel('% Diabetes diagnosis')

plt.xlabel('Insulin Serum')

plt.title('Insulin Serum % Scatter ')



##################



ser1 = np.zeros((9,6), dtype = np.float32)

tmp = np.arange(1, 901, 100); ser1[:,0] = tmp

tmp = np.arange(100, 1000, 100); ser1[:,2] = tmp





for i in range(0, len(trainip)):

    tp1 = np.where((ser1[:,0] <= trainip[i,4]) & (ser1[:,2] >= trainip[i,4]))

    #print(trainip[i,3], tp1, len(tp1))

    if (trainip[i,4]!= 0):

        tp2 = int(tp1[0])

        ser1[tp2,3] +=1

        if (trainop[i] == 1):

            ser1[tp2,4] +=1



# midpoint calculation

ser1[:,1] = ser1[:,0] + ((ser1[:,2] - ser1[:,0])/2)



#calculate the fraction for diabetes diagnosis

for i in range(0,len(ser1)):

    if (ser1[i,4] !=0):

        ser1[i,5] = ser1[i,4] / ser1[i,3]



ser1 = np.around(ser1,9)

a = ser1[:,np.newaxis,1]; b = ser1[:,np.newaxis,5]

#print(ser1)





a = ser1[:,np.newaxis,1]; b = ser1[:,np.newaxis,5]

#print(a,b)



#remove outliers and points that reduces a good lin regr

x = [6,7]

x = np.array(x)

ser2 = np.delete(ser1, x, axis=0)



c = ser2[:,np.newaxis,1]; d = ser2[:,np.newaxis,5]

#print(c,d)





fig.add_subplot(ps[3:,0:])

plt.axis([0, 850, 0, 1])

plt.grid(True)



regr = linear_model.LinearRegression()

regr.fit(c, d)



plt.scatter(a,b, color='red')

plt.scatter(c,d, color='black')

plt.plot(c, regr.predict(c), color='blue',linewidth=3)

plt.ylabel('% Diabetes diagnosis')

plt.xlabel('Insulin Serum Readings')

plt.title('Insulin Serum fractions Scatter, bin = 49, \n lin Regr (w & w/o Outliers) ')



#print(regr.coef_,regr.intercept_)

x71 = float(regr.coef_) ; x72 = float(regr.intercept_)    #ser coef regr

x71 = round(x71,8); x72 = round(x72,8)



print("LC = Lower Count, UC = Upper Count, Mid = Average of upper and lower")

print("BC = Band Count is the number within the band above")

print("DC = Diabetes diagnosed within the bands")

print("Frac = Diagnosed count/Band Count")

header7 = ["LC", "Mid", "UC", "BC", "DC", "Frac"]

print(tabulate(ser1, header7, tablefmt="fancy_grid"))

plt.show()
#######################################################################################               



trainip3 = np.zeros((len(trainip),8), dtype = np.float64)

trainop3 = trainop

testip3 = np.zeros((len(testip),8), dtype = np.float64)

testop3 = testop



trainip3[0:len(trainip),0:6] = trainip1.copy(); trainop3 = trainop.copy()

testip3[0:len(testip),0:6] = testip1.copy(); testop3 = testop.copy()



print("\nThe equations for the line graphs are as follows ")

print("1st Pregnancy Eqn, y = ",x11,"x + ",x12)

#print("2nd Plasma Eqn, y = 1 / (",x21,"x^2 + ",x22,"x + ",x23,")")

print("2nd Plasma Eqn, y = exp^(-0.00035(x-175)^2 + 0.95x + 250 )")

print("3rd BP Eqn, y = ",x31,"x + ",x32)

print("4th BMI Eqn, y = ",x41,"x + ",x42)

print("5th DPF Eqn, y = ",x51,"x + ",x52)

print("8th Age Eqn, y = math.pow(0.2704 - (0.00090278 * math.pow((x - 45),2)),0.5)   ")



print("6th Tricep skin thickness, y = ",x621,"x + ",x622)

print("7th Insulin Serum, y = ",x71,"x + ",x72)





trainip3[:,6] = (trainip[:,3] * x621) + x622

testip3[:,6] = (testip[:,3] * x621) + x622



trainip3[:,7] = (trainip[:,4] * x71) + x72

testip3[:,7] = (testip[:,4] * x71) + x72



tp3 = np.where(trainip[:,3] == 0)

trmp3 = trainip.copy()

trmp3 = np.delete(trmp3, tp3, axis=0)

trainip3 = np.delete(trainip3, tp3, axis=0)

trainop3 = np.delete(trainop3, tp3, axis=0)



tp4 = np.where(testip[:,3] == 0)

tsp3 = testip.copy()

tsp3 = np.delete(tsp3, tp4, axis=0)

testip3 = np.delete(testip3, tp4, axis=0)

testop3 = np.delete(testop3, tp4, axis=0)



tp3 = np.where(trmp3[:,4] == 0)

trmp3 = np.delete(trmp3, tp3, axis=0)

trainip3 = np.delete(trainip3, tp3, axis=0)

trainop3 = np.delete(trainop3, tp3, axis=0)





tp4 = np.where(tsp3[:,4] == 0)

tsp3 = np.delete(tsp3, tp4, axis=0)

testip3 = np.delete(testip3, tp4, axis=0)

testop3 = np.delete(testop3, tp4, axis=0)





clf = svm.SVC(gamma=1, C= 20)#RBF Settings

#clf = svm.SVC(kernel="linear", gamma=1, C= 5)#linear Settings

clf.fit(trainip3,trainop3)

print("The training and test sample sets has shrunk to ")

print(trainip3.shape, trainop3.shape, testip3.shape, testop3.shape)



target2 = clf.predict(testip3)

print('\n The new and improved Accuracy is now',accuracy_score(testop3,target2 ))