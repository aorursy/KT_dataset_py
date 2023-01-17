# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statistics as st

import matplotlib.pyplot as plt 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
files=[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file1 = os.path.join(dirname, filename)

        files.append(file1)

files
files_csv = [f for f in files if f[-3:] == 'csv']

print('There are total ', len(files_csv), 'files')
files_csv.sort()

files_csv

len(files_csv)

files_csv=files_csv[:79]

len(files_csv)
df = pd.DataFrame()

shp = np.zeros(80)

i=0



for f in files_csv:

    

    data = pd.read_csv(f, names=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13"])

    shp[i] = len(data)

    df = df.append(data, sort=False)

    i = i+1

    

 

    

print(df.shape)

df.to_csv(r'/kaggle/working/file1.csv', index = False)

# df.drop(['f1'])

# print(df.loc[810].values)

df

import matplotlib.pyplot as plt #for data visualization 

import seaborn as sns

plt.figure(1 , figsize = (17 , 8))



cor = sns.heatmap(df.corr(), annot = True)
# Extra

# Extracting each column

ti = df.loc[:, 'f1'].values #Time Index

ar = df.loc[:, 'f2'].values #Altitude Roll

ap = df.loc[:, 'f3'].values #Altitude pitch

ay = df.loc[:, 'f4'].values #Altitude Yaw

rrx = df.loc[:, 'f5'].values #Rotation rate x

rry = df.loc[:, 'f6'].values #Rotation rate y

rrz = df.loc[:, 'f7'].values #Rotation rate z

gx = df.loc[:, 'f8'].values #Gravity x

gy = df.loc[:, 'f9'].values #Gravity y

gz = df.loc[:, 'f10'].values #Gravity z

uax = df.loc[:, 'f11'].values #User Acceleration x

uay = df.loc[:, 'f12'].values #User Acceleration y

uaz = df.loc[:, 'f13'].values #User Acceleration z



print(ti.shape)

print(ar.shape)

print(rrx.shape)

print(gx.shape)

print(uax.shape)

# print(uax[40091])
mydata = df.loc[:,:].values

print(mydata.shape)

print(mydata[0,1])
#Extra

shp
files_csv[78]
# jan_one = st.mean(mydata[0:int(shp[0]),1])

means = np.zeros((len(files_csv),mydata.shape[1]))

# Finding mean of each sample. For this, rows will be number of samples or number of files

a=0

for rows in range (0,len(files_csv)):

    for cols in range (0, mydata.shape[1]):

        means[rows, cols] = st.mean(mydata[a:(int(shp[rows])+a), cols])

    a=a+int(shp[rows])

#         means[rows, cols] = st.mean(mydata[(int(shp[rows-1])+1):(int(shp[rows])+int(shp[rows-1])+1),cols])



# print(rows, cols)

# print((int(shp[rows-1])+1),(int(shp[rows])+int(shp[rows-1])+1))

means[78]

means

noise = np.zeros((len(files_csv), mydata.shape[1]))

for m in range (0, len(files_csv)-1):

    noise[m] = means[m+1,cols] - means[m,cols]

#     for n in range (0, mydata.shape[1]):

        

noise.shape



#Extra

means.shape
noise_pd = pd.DataFrame(noise)

# print(noise_pd[2])

noise_pd[0].plot()
print(type(mydata[0]))

print(type(noise_pd[0]))
# Scatter plot of Noise

plt.scatter(noise[0:80,0], noise[0:80,1], label= "stars", color= "green", marker= "*", s=30)

# It noise is very low as most of the plot is at zero
print(mydata.shape)

# Plotting altitude roll against the time (1 to 16)

plt.plot(mydata[0:int(shp[0]),0], mydata[0:int(shp[0]),1]) # Plotting first sample out of 80

plt.plot(mydata[812:int(shp[1]+811),0], mydata[812:int(shp[0]+811),1]) # Plotting 2nd sample out of 80



# Both the lines shows that they start from min, goes to the max, and comes back to the minimum

# It means that glass is picked up, reach mid point, reached end point (back at rest) in 16 seconds
# Scatter plot of the same above data (only for 1st sample out of 80)

plt.scatter(mydata[0:int(shp[0]),0], mydata[0:int(shp[0]),1], label= "stars", color= "green", marker= "*", s=30)



# It clearly shows that it increases from 0 to 0.8 in 1st second, and then comes back to 0 in 16th second.

# Between these, data is in a specific range which shows that gyro is in a range, which is the mid point, whcih is 'drinking water position'.
# As we have seen the results of 1 sample, now trying to assess all the 80 samples of altitude roll

plt.scatter(means[0:80,0], means[0:80,1], label= "stars", color= "green", marker= "*", s=30)



# For this, mean is plotted.

# It also shows that major data is pressent between 3 and 10 seconds, which shows that it is mid point.
# Let us see what happens if we plot all the altitudes

plt.scatter(means[0:80,0], means[0:80,1], label= "stars", s=10)

plt.scatter(means[0:80,0], means[0:80,2], label= "stars", s=10)

plt.scatter(means[0:80,0], means[0:80,3], label= "stars", s=10)



# Again, majority lies at the mid point (data scattered in between 16 seconds)
# Trying to plot each second of each sample of atltitude roll

a=0

for rows in range (0,len(files_csv)):

    plt.plot(mydata[a:(int(shp[rows])+a), 0], mydata[a:(int(shp[rows])+a), 1])

    a=a+int(shp[rows])

    

# It can be shown that data is very much mixed up. It is because, all the samples are giving almost same values.
# Plotting all the data of altitude roll in one plot file.

plt.plot(mydata[:,0], mydata[:,1])



# It is also mixed up, so we will stick to above scatter plots of 1 sample. And for all samples, mean data is to be plotted.
# Plotting means of all the altitudes as lines

plt.plot(means[:,1])

plt.plot(means[:,2])

plt.plot(means[:,3])
# Finding the variance to know how much the data is varying

var=[]

for i in range (1, means.shape[1]):

    var1 = st.variance(means[:,i])

    var.append(var1)

plt.plot(var)

# len(var)
var=[]

for i in range (1, mydata.shape[1]):

    var1 = st.variance(mydata[(0*50):(6*50),i])

    var.append(var1)

plt.plot(var)

# len(var)
var=[]

for i in range (1, mydata.shape[1]):

    var1 = st.variance(mydata[(6*50):(13*50),i])

    var.append(var1)

plt.plot(var)

# len(var)
var=[]

for i in range (1, mydata.shape[1]):

    var1 = st.variance(mydata[(14*50):(16*50+11),i])

    var.append(var1)

plt.plot(var)

# len(var)
import json

f = open('/kaggle/input/drink-from-a-glass/all three_events_export.json')

jdata = json.load(f) 

len(jdata)

f1 = open('/kaggle/input/drink-from-a-glass/only one_events_export.json')

j1data = json.load(f1) 

len(j1data)

j1data

f11 = pd.DataFrame(jdata)

f12 = pd.DataFrame(j1data)

f11.shape

# f11[3,2]
url = '/kaggle/input/drink-from-a-glass/only one_events_export.json'

mn1 = pd.read_json(url)

mn1.shape

type(mn1)

mn11 = mn1.loc[:,:].values



mn11.shape

mn11[:,2].shape

l = mn11[:,2]

l.shape

# len(l[1])

# l1 = l[1]

# len(l1[1])

# l2 = l1[1]

# len(l2)

# type(l2)

# l3 = pd.DataFrame.from_dict(l2)

# l4 = l3.loc[:,:].values

# l4[1,0]
url = '/kaggle/input/drink-from-a-glass/only one_events_export.json'

mn1 = pd.read_json(url)

mn1.shape

type(mn1)

mn11 = mn1.loc[:,:].values



mn11.shape



l=[]

l0=mn11[:,0]

l1=mn11[:,1]

l2=mn11[:,2]

l.append(l0)

l.append(l1)

l.append(l2)

len(l[0])



ll00 = l0[0]

ll01 = l0[1]

ll02 = l0[2]

ll03 = l0[3]

ll04 = l0[4]

type(ll04)



ll10 = l1[0]

ll11 = l1[1]

ll12 = l1[2]

ll13 = l1[3]

ll14 = l1[4]

type(ll14)



ll20 = l2[0]

ll21 = l2[1]

ll22 = l2[2]

ll23 = l2[3]

ll24 = l2[4]

type(ll24)

len(ll24)

len(ll24[1])



lll200 = ll20[0]

lll201 = ll20[1]

lll202 = ll20[2]

lll203 = ll20[3]

lll204 = ll20[4]

lll205 = ll20[5]

type(lll205)

r00 = pd.DataFrame.from_dict(lll200)

fr00 = r00.loc[:,:].values

r00



lll210 = ll21[0]

lll211 = ll21[1]

lll212 = ll21[2]

lll213 = ll21[3]

lll214 = ll21[4]

lll215 = ll21[5]

type(lll215)

r01 = pd.DataFrame.from_dict(lll210)

fr01 = r01.loc[:,:].values

r01



lll220 = ll22[0]

lll221 = ll22[1]

lll222 = ll22[2]

lll223 = ll22[3]

lll224 = ll22[4]

lll225 = ll22[5]

type(lll225)

r02 = pd.DataFrame.from_dict(lll220)

fr02 = r02.loc[:,:].values

r02



# p=[]

# p

# p.append(lll220)

# p.append(lll221)

# p.append(lll222)

# p.append(lll223)

# p.append(lll224)

# p.append(lll225)

# len(p)



lll230 = ll23[0]

lll231 = ll23[1]

lll232 = ll23[2]

lll233 = ll23[3]

lll234 = ll23[4]

lll235 = ll23[5]

type(lll235)

r03 = pd.DataFrame.from_dict(lll230)

fr03 = r03.loc[:,:].values

r03



lll240 = ll24[0]

lll241 = ll24[1]

lll242 = ll24[2]

lll243 = ll24[3]

lll244 = ll24[4]

lll245 = ll24[5]

type(lll245)

r04 = pd.DataFrame.from_dict(lll240)

fr04 = r04.loc[:,:].values

r04

import pandas as pd

from itertools import dropwhile

import csv

with open(url) as f:

    f = dropwhile(lambda x: x.startswith("#!!"), f)

    r = csv.reader(f)

    df1 = pd.DataFrame().from_records(r)

df1.shape

d1 = df1.loc[:,:].values

d1.shape

d1[42000,1]
f11.to_csv(r'/kaggle/working/all3_json_out.csv', index = False)

f12.to_csv(r'/kaggle/working/only1_json_out.csv', index = False)
d1 = pd.DataFrame({

    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],

    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]

})



from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=3)

kmeans.fit(d1)
mydata[0:811,:]
# d2 = pd.DataFrame(mydata[0:811,:])

d2 = pd.DataFrame(mydata)

d2.shape


kmeans = KMeans(n_clusters=3, max_iter=300)

kmeans.fit(d2)

labels = kmeans.predict(d2)

centroids = kmeans.cluster_centers_

labels
lb = pd.DataFrame(labels)

cn = pd.DataFrame(centroids)
lb.to_csv(r'/kaggle/working/labels.csv', index = False)

cn.to_csv(r'/kaggle/working/centroids.csv', index = False)
# v1 = np.zeros((1,13))

# v1.shape

v1=st.variance(mydata[0:204,5])

print(v1)

v2=st.variance(mydata[204:458,5])

print(v2)

v3=st.variance(mydata[458:811,5])

print(v3)

v4=st.variance(mydata[0:811,5])

print(v4)
v1=st.variance(mydata[0+811:204+811,5])

print(v1)

v2=st.variance(mydata[204+811:458+811,5])

print(v2)

v3=st.variance(mydata[458+811:811+811,5])

print(v3)

v4=st.variance(mydata[0+811:811+811,5])

print(v4)
kk = np.concatenate((mydata, lb), axis=1)

kk[:,13]

train_data = pd.DataFrame(kk)

train_data.to_csv(r'/kaggle/working/train_data.csv', index = False)
X_train = mydata[:,1:]

y_train = labels

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(X_train, y_train)
inp = pd.read_csv('/kaggle/input/drink-from-a-glass/04_01_2017_07_25_12_13_00.csv')

X_t = inp.loc[:,:].values

X_t.shape

X_test = X_t[:,1:]

X_test

# X_t[:,1:12]
X_test.shape
y_pred = classifier.predict(X_test)

y_pred

# y_train
len(labels)

labels[len(labels)-1]
labels

if labels[1] == 0:

    print('0 means glass is in lifting position')

    pl=labels[1]

elif (lables[1]==1):

    print('1 means glass is in lifting position')

    pl=labels[1]

elif (lables[1]==2):

    print('2 means glass is in lifting position')

    pl=labels[1]

else:

    print('Try Again')



if (labels[len(labels)-1] == 0):

    print('0 means glass is in retrieving position')

    rt=labels[len(labels)-1]

elif (labels[len(labels)-1]==1):

    print('1 means glass is in retrieving position')

    rt=labels[len(labels)-1]

elif (labels[len(labels)-1]==2):

    print('2 means glass is in retrieving position')

    rt=labels[len(labels)-1]

else:

    print('Try Again')

if pl+rt==1:

    iop=2

elif pl+rt==2:

    iop=1

elif pl+rt==3:

    iop=0

else:

    print('Try again')

iop
pred=np.array(len(y_pred))

y_pred.shape

type(y_pred)

y_pred[43]
pred=[]

for c in range (0,len(y_pred)):

    if y_pred[c]==pl:

        a='Picking'

        pred.append(a)

#         print('It is in picking position')

    elif y_pred[c]==rt:

        a='Retrieving'

        pred.append(a)

#         print('It is in retrieving position')

    elif y_pred[c]==iop:

        a='Operation'

        pred.append(a)

#         print('It is in OPERATING position')

pred
X_test.shape

len(pred)

predd=np.array(pred)

predd.shape

pn = pd.DataFrame(predd)

pn
# finale = np.concatenate((X_test, predd), axis=1)

Xt1 = pd.DataFrame(X_test)

predd = pd.DataFrame(pred)

predd

Xt1

finale = pd.concat([inp, predd], axis=1, sort=False)

finale

# train_data = pd.DataFrame(kk)

finale.to_csv(r'/kaggle/working/Finale.csv', index = False)