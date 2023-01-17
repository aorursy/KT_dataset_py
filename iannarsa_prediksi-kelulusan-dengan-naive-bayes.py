import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
trans_df = pd.read_csv("../input/nilaimhs/transkrip.csv")

trans_df.head(10)
a=len(trans_df[trans_df['nilai'] == 'A'])

b=len(trans_df[trans_df['nilai'] == 'B'])

c=len(trans_df[trans_df['nilai'] == 'C'])

d=len(trans_df[trans_df['nilai'] == 'D'])

e=len(trans_df[trans_df['nilai'] == 'E'])



jn = a+b+c+d+e

ap = a/jn*100

bp = b/jn*100

cp = c/jn*100

dp = d/jn*100

ep = e/jn*100
lulus = a+b+c

tlulus = d+e

jumlah = lulus+tlulus



plulus=lulus/jumlah*100

ptlulus=tlulus/jumlah*100



objects = ('Lulus', 'Tidak Lulus')

y_pos = np.arange(len(objects))

performance = [lulus, tlulus]

 

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Jumlah Matakuliah')

plt.title('Kelulusan Matakuliah')

 

plt.show()

print("Persentase")

print("Matakuliah Yang Lulus : "+str(plulus))

print("Matakuliah Yang Tidak Lulus : "+str(ptlulus))
objects = ('A', 'B', 'C', 'D', 'E')

y_pos = np.arange(len(objects))

performance = [a,b,c,d,e]

 

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Jumlah Nilai')

plt.title('Nilai')



plt.show()

print("A : "+str(ap)+" %")

print("B : "+str(bp)+" %")

print("C : "+str(cp)+" %")

print("D : "+str(dp)+" %")

print("E : "+str(ep)+" %")
n=trans_df['kode']

n=len(n)

ka=trans_df.get_value(0,'kode')



praktik = 0

keterampilan = 0

teori = 0

for i in range (n):

    ka=trans_df.get_value(i,'kode')

    if ka[-1] == "P":

        praktik+=1

    elif ka[-1] == "K":

        keterampilan+=1

    elif ka[-1] == "T":

        teori+=1



pr=praktik/n*100

ke=keterampilan/n*100

te=teori/n*100
 # Data to plot

print('Persentase Jenis Matakuliah Yang Diambil')

labels = 'Teori', 'Praktik', 'Keterampilan'

sizes = [teori, praktik, keterampilan]

colors = ['pink', 'tan', 'skyblue']

explode = (0.01, 0.01, 0.01)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

 

plt.axis('equal')

plt.show()
import matplotlib.pyplot as plt



# The slices will be ordered and plotted counter-clockwise.



print('Persentase Jenis Matakuliah Yang Diambil')

labels = 'Teori', 'Praktik', 'Keterampilan'

sizes = [teori, praktik, keterampilan]

colors = ['yellowgreen', 'gold', 'lightskyblue']

explode = (0.01, 0.01, 0.01)  # explode 1st slice





plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True)

        

#draw a circle at the center of pie to make it look like a donut

centre_circle = plt.Circle((0,0),0.9,color='gray', fc='white',linewidth=0)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)





# Set aspect ratio to be equal so that pie is drawn as a circle.

plt.axis('equal')

plt.show()  
presensi_df = pd.read_csv("../input/presensi/presensi.csv")

presensi_df.head()
hdr=presensi_df.hadir.sum()

aph=presensi_df.alpha.sum()

jl = hdr+aph
# Data to plot

print('Presensi')

labels = 'Hadir', 'Alpha'

sizes = [hdr, aph]

colors = ['orchid', 'lightpink']

explode = (0.2, 0.01)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.3f%%', shadow=True, startangle=140)

 

plt.axis('equal')

plt.show()
# Import LabelEncoder

import pandas as pd
data_df = pd.read_csv("../input/nilaimhs/data_training.csv")

data_df
def convert(data_df):

    for i in range (len(data_df['IP_SMT1'])):

        if data_df.loc[i,'IP_SMT1'] == 4:

            data_df.loc[i,'IP_SMT1'] = 'A'

        elif data_df.loc[i, 'IP_SMT1'] >=3 and data_df.loc[i, 'IP_SMT1']<4:

            data_df.loc[i,'IP_SMT1'] = 'B'

        elif data_df.loc[i, 'IP_SMT1'] >=2 and data_df.loc[i, 'IP_SMT1']<3:

            data_df.loc[i,'IP_SMT1'] = 'C'

        elif data_df.loc[i, 'IP_SMT1'] >=1 and data_df.loc[i, 'IP_SMT1']<2:

            data_df.loc[i,'IP_SMT1'] = 'D'

        else:

            data_df.loc[i,'IP_SMT1'] = 'E'



    for i in range (len(data_df['IP_SMT2'])):

        if data_df.loc[i,'IP_SMT2'] == 4:

            data_df.loc[i,'IP_SMT2'] = 'A'

        elif data_df.loc[i, 'IP_SMT2'] >=3 and data_df.loc[i, 'IP_SMT2']<4:

            data_df.loc[i,'IP_SMT2'] = 'B'

        elif data_df.loc[i, 'IP_SMT2'] >=2 and data_df.loc[i, 'IP_SMT2']<3:

            data_df.loc[i,'IP_SMT2'] = 'C'

        elif data_df.loc[i, 'IP_SMT2'] >=1 and data_df.loc[i, 'IP_SMT2']<2:

            data_df.loc[i,'IP_SMT2'] = 'D'

        else:

            data_df.loc[i,'IP_SMT2'] = 'E'



    for i in range (len(data_df['IP_SMT3'])):

        if data_df.loc[i,'IP_SMT3'] == 4:

            data_df.loc[i,'IP_SMT3'] = 'A'

        elif data_df.loc[i, 'IP_SMT3'] >=3 and data_df.loc[i, 'IP_SMT3']<4:

            data_df.loc[i,'IP_SMT3'] = 'B'

        elif data_df.loc[i, 'IP_SMT3'] >=2 and data_df.loc[i, 'IP_SMT3']<3:

            data_df.loc[i,'IP_SMT3'] = 'C'

        elif data_df.loc[i, 'IP_SMT3'] >=1 and data_df.loc[i, 'IP_SMT3']<2:

            data_df.loc[i,'IP_SMT3'] = 'D'

        else:

            data_df.loc[i,'IP_SMT3'] = 'E'



    for i in range (len(data_df['IP_SMT4'])):

        if data_df.loc[i,'IP_SMT4'] == 4:

            data_df.loc[i,'IP_SMT4'] = 'A'

        elif data_df.loc[i, 'IP_SMT4'] >=3 and data_df.loc[i, 'IP_SMT4']<4:

            data_df.loc[i,'IP_SMT4'] = 'B'

        elif data_df.loc[i, 'IP_SMT4'] >=2 and data_df.loc[i, 'IP_SMT4']<3:

            data_df.loc[i,'IP_SMT4'] = 'C'

        elif data_df.loc[i, 'IP_SMT4'] >=1 and data_df.loc[i, 'IP_SMT4']<2:

            data_df.loc[i,'IP_SMT4'] = 'D'

        else:

            data_df.loc[i,'IP_SMT4'] = 'E'



    for i in range (len(data_df['sks1'])):

        if data_df.loc[i,'sks1'] >= 23:

            data_df.loc[i,'sks1'] = 'A'

        elif data_df.loc[i, 'sks1'] >=20 and data_df.loc[i, 'sks1']<23:

            data_df.loc[i,'sks1'] = 'B'

        elif data_df.loc[i, 'sks1'] >=18 and data_df.loc[i, 'sks1']<20:

            data_df.loc[i,'sks1'] = 'C'

        elif data_df.loc[i, 'sks1'] >=15 and data_df.loc[i, 'sks1']<18:

            data_df.loc[i,'sks1'] = 'D'

        else:

            data_df.loc[i,'sks1'] = 'E'



    for i in range (len(data_df['sks2'])):

        if data_df.loc[i,'sks2'] >= 23:

            data_df.loc[i,'sks2'] = 'A'

        elif data_df.loc[i, 'sks2'] >=20 and data_df.loc[i, 'sks2']<23:

            data_df.loc[i,'sks2'] = 'B'

        elif data_df.loc[i, 'sks2'] >=18 and data_df.loc[i, 'sks2']<20:

            data_df.loc[i,'sks2'] = 'C'

        elif data_df.loc[i, 'sks2'] >=15 and data_df.loc[i, 'sks2']<18:

            data_df.loc[i,'sks2'] = 'D'

        else:

            data_df.loc[i,'sks2'] = 'E'



    for i in range (len(data_df['sks3'])):

        if data_df.loc[i,'sks3'] >= 23:

            data_df.loc[i,'sks3'] = 'A'

        elif data_df.loc[i, 'sks3'] >=20 and data_df.loc[i, 'sks3']<23:

            data_df.loc[i,'sks3'] = 'B'

        elif data_df.loc[i, 'sks3'] >=18 and data_df.loc[i, 'sks3']<20:

            data_df.loc[i,'sks2'] = 'C'

        elif data_df.loc[i, 'sks3'] >=15 and data_df.loc[i, 'sks3']<18:

            data_df.loc[i,'sks3'] = 'D'

        else:

            data_df.loc[i,'sks3'] = 'E'



    for i in range (len(data_df['sks4'])):

        if data_df.loc[i,'sks4'] >= 23:

            data_df.loc[i,'sks4'] = 'A'

        elif data_df.loc[i, 'sks4'] >=20 and data_df.loc[i, 'sks4']<23:

            data_df.loc[i,'sks4'] = 'B'

        elif data_df.loc[i, 'sks4'] >=18 and data_df.loc[i, 'sks4']<20:

            data_df.loc[i,'sks4'] = 'C'

        elif data_df.loc[i, 'sks4'] >=15 and data_df.loc[i, 'sks4']<18:

            data_df.loc[i,'sks4'] = 'D'

        else:

            data_df.loc[i,'sks4'] = 'E'



    for i in range (len(data_df['Pendapatan_Wali'])):

        if data_df.loc[i,'Pendapatan_Wali'] >= 4000000:

            data_df.loc[i,'Pendapatan_Wali'] = 'A'

        elif data_df.loc[i, 'Pendapatan_Wali'] >=3000000 and data_df.loc[i, 'Pendapatan_Wali']<4000000:

            data_df.loc[i,'Pendapatan_Wali'] = 'B'

        else:

            data_df.loc[i,'Pendapatan_Wali'] = 'C'

    return data_df
from sklearn import preprocessing

# Converting string labels into numbers.

le = preprocessing.LabelEncoder()

def convertNumber(data_df):

    le = preprocessing.LabelEncoder()

    data_df.Jenis_Kelamin = le.fit_transform(data_df.Jenis_Kelamin)

    data_df.Jenis_Seleksi = le.fit_transform(data_df.Jenis_Seleksi)

    data_df.Pendapatan_Wali = le.fit_transform(data_df.Pendapatan_Wali)

    data_df.Pendidikan_Ibu = le.fit_transform(data_df.Pendidikan_Ibu)

    data_df.IP_SMT1 = le.fit_transform(data_df.IP_SMT1)

    data_df.IP_SMT2 = le.fit_transform(data_df.IP_SMT2)

    data_df.IP_SMT3 = le.fit_transform(data_df.IP_SMT3)

    data_df.IP_SMT4 = le.fit_transform(data_df.IP_SMT4)

    data_df.sks1 = le.fit_transform(data_df.sks1)

    data_df.sks2 = le.fit_transform(data_df.sks2)

    data_df.sks3 = le.fit_transform(data_df.sks3)

    data_df.sks4 = le.fit_transform(data_df.sks4)

    

    return data_df

convertNumber(data_df)

data_df.KET = le.fit_transform(data_df.KET)

data_df
#Combining

features=zip(data_df.Jenis_Kelamin, data_df.Jenis_Seleksi, data_df.Pendapatan_Wali, data_df.Pendidikan_Ibu,

             data_df.IP_SMT1, data_df.IP_SMT2, data_df.IP_SMT3, data_df.IP_SMT4, data_df.sks1, data_df.sks2,

             data_df.sks3, data_df.sks4)

features = list(features)



label = zip(data_df.KET)

label = list(label)

print (features)
from sklearn import preprocessing

#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB



#Create a Gaussian Classifier

model = GaussianNB()



# Train the model using the training sets

model.fit(features,label)
data = pd.read_csv("../input/nilaimhs/data_test.csv") 



# creating a dict file 

convert(data)

convertNumber(data)



#Predict Output

predicted= model.predict([[

    data.Jenis_Kelamin[0],data.Jenis_Seleksi[0],data.Pendapatan_Wali[0],data.Pendidikan_Ibu[0],data.IP_SMT1[0],

data.IP_SMT2[0],data.IP_SMT3[0],data.IP_SMT4[0],data.sks1[0],data.sks2[0],data.sks3[0],data.sks4[0]

]])

print ("Predicted Value:", predicted)
if predicted == 0:

    print("Hasil Prediksi : Lulus Tepat Waktu")

else:

    print("Hasil Prediksi : Tidak Lulus Tepat Waktu")