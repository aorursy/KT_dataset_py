# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from  openpyxl import *

from xlrd import open_workbook





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# Any results you write to the current directory are saved as output.


mind_exel = open_workbook("/kaggle/input/py_mind.xlsx")

sense_exel = open_workbook("/kaggle/input/py_sense.xlsx")

science_exel = open_workbook("/kaggle/input/py_science.xlsx")

opinion_exel = open_workbook("/kaggle/input/py_opinion.xlsx")



#the students of classes

mind_students = mind_exel.sheet_names()

sense_students = sense_exel.sheet_names()

science_students = science_exel.sheet_names()

opinion_students = opinion_exel.sheet_names()



mind_len = len(mind_students)-1

sense_len = len(sense_students)-1

science_len = len(science_students)-1

opinion_len = len(opinion_students)-1



mind_sheets = 0

sense_sheets = 0

science_sheets = 0

opinion_sheets = 0

#Bos bir DF olusturduk.

all_class = pd.DataFrame(columns=("Class", "Name", "True", "False", "Empty", "DYB"), index=None)

while mind_sheets < mind_len:

    mind_sheets += 1 # her tabloda dolasabilmek icin sayac.

    mind_ex = pd.read_excel("/kaggle/input/py_mind.xlsx", sheet_name=mind_sheets)

    mind_results = mind_ex.tail(3) #en alttaki dogru yanlis boslari hucrelerini alir

    DYB = list(mind_ex.iloc[0:20,1]) #OGRENCI SIKLARINI ALIR

    class_name = "py_mind" # sinif ismi

    name = mind_results.columns[0] # 0 indexli satir ve sutunu alir bu bizim ogrenci ismimiz olacak

    true = mind_results.iloc[0,1] # Dogru sayisini

    false = mind_results.iloc[1,1] # yanlis sayisini

    empty = mind_results.iloc[2,1] # bosu alir

    student = {'Class': class_name, 'Name': name, 'True': true, 'False': false, "Empty": empty, "DYB":DYB}    

    all_class = all_class.append(student, ignore_index=True) # duzenli olan dic'i,  olusturdugumuz bos DF ye ekledik. her bir ogrenci ve sinif icin tum bunlar yapilacak



while sense_sheets < sense_len:

    sense_sheets += 1

    sense_ex = pd.read_excel("/kaggle/input/py_sense.xlsx", sheet_name=sense_sheets)

    sense_results = sense_ex.tail(3)

    DYB = list(sense_ex.iloc[0:20,1])

    class_name = "py_sense"

    name = sense_results.columns[0]

    name = str(name)

    true = sense_results.iloc[0,1]

    false = sense_results.iloc[1,1]

    empty = sense_results.iloc[2,1]

    student = {'Class': class_name, 'Name': name, 'True': true, 'False': false, "Empty": empty, "DYB":DYB}    

    all_class = all_class.append(student, ignore_index=True)



while science_sheets < science_len:

    science_sheets += 1

    science_ex = pd.read_excel("/kaggle/input/py_science.xlsx", sheet_name=science_sheets)

    science_results = science_ex.tail(3)

    DYB = list(science_ex.iloc[0:20,1])

    class_name = "py_science"

    name = science_results.columns[0]

    name = str(name)

    true = science_results.iloc[0,1]

    false = science_results.iloc[1,1]

    empty = science_results.iloc[2,1]

    student = {'Class': class_name, 'Name': name, 'True': true, 'False': false, "Empty": empty, "DYB":DYB}    

    all_class = all_class.append(student, ignore_index=True)



while opinion_sheets < opinion_len:

    opinion_sheets += 1

    opinion_ex = pd.read_excel("/kaggle/input/py_opinion.xlsx", sheet_name=opinion_sheets)

    opinion_results = opinion_ex.tail(3)

    DYB = list(opinion_ex.iloc[0:20,1])

    class_name = "py_opinion"

    name = opinion_results.columns[0]

    name = str(name)

    true = opinion_results.iloc[0,1]

    false = opinion_results.iloc[1,1]

    empty = opinion_results.iloc[2,1]

    student = {'Class': class_name, 'Name': name, 'True': true, 'False': false, "Empty": empty, "DYB":DYB}    

    all_class = all_class.append(student, ignore_index=True)
print(all_class)
#Sinava giren ogrenci sayisi

total = (mind_len + opinion_len + science_len + sense_len) #siniflarin toplam sayisi

print("Sinava giren ogrenci sayisi =",total)
#2.soru = tum sinifin dogru,yanlis,bos sayisi

all_true = all_class["True"]

all_false = all_class["False"]

all_empty = all_class["Empty"]

print("dogru sayilari =", sum(all_true), "Yanlis sayilari =", sum(all_false), "Bos sayilari = ", sum(all_empty))
#soru = 3 tum siniflarin Dogru yanlis bos ortalamalari

print("Tum siniflarin;\nDogru ortalamasi =",all_true.mean(),"\nYanlis ortalamasi = ", all_false.mean(), "\nBos ortalamasi =", all_empty.mean())
#siniflarin ayri ayri df yapilmasi ve ortalama alinmasi

c_mind = all_class[all_class["Class"]=="py_mind"]

c_mind_ort = c_mind["True"]

c_sense = all_class[all_class["Class"]=="py_sense"]

c_sense_ort = c_sense["True"]

c_science = all_class[all_class["Class"]=="py_science"]

c_science_ort = c_science["True"]

c_opinion = all_class[all_class["Class"]=="py_opinion"]

c_opinion_ort = c_opinion["True"]

print("sinif ortalamasi","\npy_mind ortalamasi =", c_mind_ort.mean(), "\npy_Sense ortalamasi =", c_sense_ort.mean(),"\npy_science ortalamasi =",c_science_ort.mean(), "\npy_opinion ortalamasi =", c_opinion_ort.mean())

#sinif sinif en basarili kisiler

mind_1st= c_mind.sort_values(by='True', ascending=False).head(1)

sense_1st= c_sense.sort_values(by='True', ascending=False).head(1)

science_1st= c_science.sort_values(by='True', ascending=False).head(1)

opinion_1st= c_opinion.sort_values(by='True', ascending=False).head(1)

print("py Mind en basarili ogrenci =", mind_1st.Name, "\npy Sense en basarili ogrenci =", sense_1st.Name, "\npy Science en basarili ogrenci =", science_1st.Name,"\npy Opinion en basarili ogrenci =", opinion_1st.Name)

#siniflarin standart sapmalari

mind_std = print(c_mind["True"].std())

sense_std = print(c_sense["True"].std())

science_std = print(c_science["True"].std())

opinion_std = print(c_opinion["True"].std())
# Grafiklestirme modulu

import matplotlib.pyplot as plt

#sinif ortalamalarini

y = np.array([c_mind_ort.mean(),c_sense_ort.mean(),c_science_ort.mean(),c_opinion_ort.mean()])

x = np.array([1,2,3,4])



#bar plot

ind = np.arange(5)

plt.bar(x,y)

plt.title("bar plot")

plt.xlabel("x cubugu")

plt.ylabel("y cubugu")

plt.xticks(ind, (" ", 'mind', 'sense', 'science', 'opinion'))

plt.show()
#standart sapma bar plot

print("*** Siniflari standart sapmalari***")

y = np.array([c_mind["True"].std(),c_sense["True"].std(),c_science["True"].std(),c_opinion["True"].std()])

print(y)

x = np.array([1,2,3,4])

ind = np.arange(5)

plt.bar(x,y)

plt.title("Standart sapma bar plot")

plt.xlabel("x cubugu")

plt.ylabel("y cubugu")

plt.xticks(ind, (" ", 'mind', 'sense', 'science', 'opinion'))

plt.show()
#siniflarin dogrularinin plot cizimi



plt.plot(c_mind["True"], color="red", label="Mind")

plt.plot(c_sense["True"], color="blue", label="sense")

plt.plot(c_science["True"], color="black", label="science")

plt.plot(c_opinion["True"], color="green", label="opinion")

plt.legend()

#legend x ve y yi olusturur

plt.xlabel("Classes")

#x label'i

plt.ylabel("True")

#show demezseniz gostermez.

plt.show()
#siniflarin dogrularinin scatter cizimi



plt.scatter(c_mind["True"].index ,c_mind["True"], color="red", label="Mind")

plt.scatter(c_sense["True"].index,c_sense["True"], color="blue", label="sense")

plt.scatter(c_science["True"].index,c_science["True"], color="black", label="science")

plt.scatter(c_opinion["True"].index, c_opinion["True"], color="green", label="opinion")

plt.legend()

#legend x ve y yi olusturur

plt.xlabel("Classes")

#x label'i

plt.ylabel("True")

#show demezseniz gostermez.

plt.show()