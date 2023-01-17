import numpy as np

import pandas as pd

import random

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline

import cv2 as cv

import os

from sklearn.model_selection import train_test_split
times = os.listdir("/kaggle/input/fonts-timesarial/TimesNewRoman")

arial = os.listdir("/kaggle/input/fonts-timesarial/Arial")

times.sort()

arial.sort()
X_train = []

X_test = []



y_train = []

y_test = []



lastChar = "A"

tempx = []

for i in times:

    if i[0]==lastChar:

        tempx.append(cv.imread("/kaggle/input/fonts-timesarial/TimesNewRoman/"+i,0).flatten())

    else:

        tempy = [0]*len(tempx)

        lastChar = i[0]

        a,b,c,d = train_test_split(tempx,tempy)

        X_train+=a

        X_test+=b

        y_train+=c

        y_test+=d

        tempx=[]

        tempx.append(cv.imread("/kaggle/input/fonts-timesarial/TimesNewRoman/"+i,0).flatten())

tempy = [0]*len(tempx)

a,b,c,d = train_test_split(tempx,tempy)

X_train+=a

X_test+=b

y_train+=c

y_test+=d





# Another Font

lastChar = "A"

tempx = []

for i in arial:

    if i[0]==lastChar:

        tempx.append(cv.imread("/kaggle/input/fonts-timesarial/Arial/"+i,0).flatten())

    else:

        tempy = [1]*len(tempx)

        lastChar = i[0]

        a,b,c,d = train_test_split(tempx,tempy)

        X_train+=a

        X_test+=b

        y_train+=c

        y_test+=d

        tempx=[]

        tempx.append(cv.imread("/kaggle/input/fonts-timesarial/Arial/"+i,0).flatten())

tempy = [1]*len(tempx)

a,b,c,d = train_test_split(tempx,tempy)

X_train+=a

X_test+=b

y_train+=c

y_test+=d
z = list(zip(X_train, y_train))

random.shuffle(z)

X_train, y_train = zip(*z)



z = list(zip(X_test, y_test))

random.shuffle(z)

X_test, y_test = zip(*z)
X_train = np.asarray(X_train,dtype = np.float32)

X_test = np.asarray(X_test,dtype = np.float32)

y_train = np.asarray(y_train)

y_test = np.asarray(y_test)



print('Training data and target sizes: \n{}, {}'.format(X_train.shape,y_train.shape))

print('Test data and target sizes: \n{}, {}'.format(X_test.shape,y_test.shape))
model = cv.ml.RTrees_create()

model.setMaxDepth(50)

model.setActiveVarCount(0)

term_type, n_trees, epsilon = cv.TERM_CRITERIA_MAX_ITER, 16, 1

model.setTermCriteria((term_type, n_trees, epsilon))

model.train(X_train/255, cv.ml.ROW_SAMPLE, y_train)
s = model.predict(X_test/255)

count = 0

for i in range(len(s[1])):

    if s[1][i][0] == y_test[i]:

        count+=1

print(count*100/len(s[1]))

print(count)
test = os.listdir("/kaggle/input/fonts-timesarial/Mixed")

print(len(test))
images = []

for i in test:

    images.append(cv.imread("/kaggle/input/fonts-timesarial/Mixed/"+i,0).flatten())

images = np.asarray(images,dtype = np.float32)

s = model.predict(images/255)

ans = []

for i in range(0,1471):

    ans.append(s[1][i])


a =[19,20,21,24,25,26,29,30,31,59,60,61,64,65,66,89,90,91,99,100,101,114,115,116,119,120,121,124,125,126,154,155,156,184,185,186,189,190,191,199,200,201,219,220,221,229,230,231,239,240,241,289,290,291,294,295,296,304,305,306,309,310,311,359,360,361,364,365,366,384,385,386,399,400,401,404,405,406,409,410,411,414,415,416,424,425,426,429,430,431,439,440,441,464,465,466,469,470,471,474,475,476,479,480,481,484,485,486,494,495,496,499,500,501,504,505,506,509,510,511,514,515,516,519,520,521,529,530,531,549,550,551,554,555,556,559,560,561,564,565,566,579,580,581,594,595,596,599,600,601,604,605,606,614,615,616,624,625,626,634,635,636,644,645,646,649,650,651,664,665,666,669,670,671,684,685,686,689,690,691,694,695,696,714,715,716,719,720,721,724,725,726,729,730,731,734,735,736,744,745,746,749,750,751,764,765,766,784,785,786,799,800,801,804,805,806,809,810,811,819,820,821,824,825,826,829,830,831,839,840,841,844,845,846,849,850,851,859,860,861,864,865,866,874,875,876,919,920,921,924,925,926,929,930,931,934,935,936,939,940,941,944,945,946,954,955,956,959,960,961,979,980,981,994,995,996,999,1000,1001,1004,1005,1006,1014,1015,1016,1019,1020,1021,1024,1025,1026,1039,1040,1041,1044,1045,1046,1069,1070,1071,1074,1075,1076,1079,1080,1081,1084,1085,1086,1094,1095,1096,1099,1100,1101,1104,1105,1106,1114,1115,1116,1119,1120,1121,1139,1140,1141,1149,1150,1151,1164,1165,1166,1169,1170,1171,1174,1175,1176,1179,1180,1181,1189,1190,1191,1204,1205,1206,1224,1225,1226,1234,1235,1236,1239,1240,1241,1244,1245,1246,1249,1250,1251,1269,1270,1271,1274,1275,1276,1284,1285,1286,1289,1290,1291,1299,1300,1301,1309,1310,1311,1314,1315,1316,1319,1320,1321,1324,1325,1326,1329,1330,1331,1339,1340,1341,1349,1350,1351,1384,1385,1386,1399,1400,1401,1414,1415,1416,1424,1425,1426,1429,1430,1431,1439,1440,1441,1449,1450,1451,1454,1455,1456,1464,1465,1466,1469,1470,1471]
count = 0

for i in a:

    if ans[i-1] == 1:

        count+=1

print(count/len(ans))

# No of Blacks counted Right