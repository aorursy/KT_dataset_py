# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
tips=pd.read_csv("../input/tips.csv")

tips

cinsiyet=[]

for i in tips.sex:

    if i=="Female":

        cinsiyet.append("Kadın")

    else :

        cinsiyet.append("Erkek")

            

sigara=[]

for i in tips.smoker:

    if i=="No":

        sigara.append("Hayır")

    else :

        sigara.append("Evet")           



gunler=[]

for i in tips.day:

    if i=="Sun":

        gunler.append("Pazar")

    elif i=="Sat":    

        gunler.append("Cumartesi")

    elif i=="Thur":    

        gunler.append("Persembe")

    else :    

        gunler.append("Cuma")



yemek=[]

for i in tips.time:

    if i=="Dinner":

        yemek.append("Aksam")

    else :

        yemek.append("Ogle")



bahsis={"Toplam_fatura":tips["total_bill"],

        "Bahsis":tips["tip"],

        "Cinsiyet":cinsiyet,

        "Sigara":sigara,

        "Gunler":gunler,

        "Zaman":yemek,

        "Porsiyon":tips["size"]}



Bahsis=pd.DataFrame(bahsis)
Bahsis
import seaborn as sns

import matplotlib.pyplot as plt

import warnings



warnings.filterwarnings('ignore')
#MATERYAL TASARIMI GRAFIKLERI



#1.POINT PLOT



sns.set_context("poster")

sns.set_style("dark")

sns.catplot(x="Sigara",

            y="Toplam_fatura",

            data=Bahsis,

            kind="point",

            hue="Cinsiyet",

            col="Gunler",

            col_wrap=2)



plt.show()
#2. POINT PLOT



sns.set_context("poster")

sns.set_style("dark")

sns.catplot(x="Cinsiyet",

            y="Toplam_fatura",

            data=Bahsis,

            kind="point",

            hue="Sigara",

            col="Zaman")



plt.show()
#3. POINT PLOT



sns.set_context("poster")

sns.set_style("dark")

sns.catplot(x="Cinsiyet",

            y="Toplam_fatura",

            data=Bahsis,

            kind="point")



plt.show()
#SACTTER PLOT

#1.SCATTER PLOT



hue_colors={"Erkek":"black","Kadın":"red"}

plt.figure(figsize=(15,10))

sns.scatterplot(x="Toplam_fatura",

                y="Bahsis",

                data=Bahsis,

                hue="Cinsiyet",

                palette=hue_colors)

plt.show()
#2.SCATTER PLOT



sns.relplot(x="Toplam_fatura",

            y="Bahsis",

            data=Bahsis,

            kind="scatter",

            hue="Sigara",

            col="Zaman",

            row="Cinsiyet")

plt.show()
#3. SCATTER PLOT



sns.relplot(x="Toplam_fatura",

            y="Bahsis",

            data=Bahsis,

            kind="scatter",

            hue="Cinsiyet",

            col="Gunler",

            col_wrap=2,

            style="Sigara")

plt.show()
#4. BOX PLOT



g=sns.catplot(x="Zaman",

              y="Toplam_fatura",

              data=Bahsis,

              col="Gunler",

              col_wrap=2,

              kind="box")

plt.show()



g=sns.catplot(x="Cinsiyet",

              y="Toplam_fatura",

              data=Bahsis,

              col="Gunler",

              col_wrap=2,

              kind="box")

plt.show()



g=sns.catplot(x="Cinsiyet",

              y="Toplam_fatura",

              data=Bahsis,

              kind="box")

plt.show()
#5.BAR PLOT



sns.catplot(x="Gunler",

            y="Toplam_fatura",

            data=Bahsis,

            kind="bar",

            order=["Persembe","Cuma","Cumartesi","Pazar"],

            ci=None)

plt.xticks(rotation=90)

plt.show()



sns.catplot(x="Cinsiyet",

            y="Toplam_fatura",

            data=Bahsis,

            kind="bar",

            hue="Sigara",

            col="Zaman",

            ci=None)

plt.show()



sns.catplot(x="Cinsiyet",

            y="Bahsis",

            data=Bahsis,

            kind="bar",

            hue="Zaman",

            col="Gunler",

            col_wrap=2,

            col_order=["Persembe","Cuma","Cumartesi","Pazar"],

            ci=None)

plt.show()
#6.LINE PLOT



gunler_no=[]

for i in tips.day:

    if i=="Sun":

        gunler_no.append(4)

    elif i=="Sat":    

        gunler_no.append(3)

    elif i=="Thur":    

        gunler_no.append(1)

    else :    

        gunler_no.append(2)



Bahsis["Gun"]=gunler_no





sns.relplot(x="Gun",

            y="Bahsis",

            data=Bahsis,

            kind="line",

            ci=None,

            hue="Cinsiyet",

            style="Cinsiyet",

            markers=True)

plt.show()





sns.relplot(x="Gun",

            y="Toplam_fatura",

            data=Bahsis,

            kind="line",

            ci=None,

            hue="Cinsiyet",

            style="Cinsiyet",

            markers=True)

plt.show()



sns.relplot(x="Gun",

            y="Toplam_fatura",

            data=Bahsis,

            kind="line",

            ci=None,

            hue="Cinsiyet",

            style="Cinsiyet",

            markers=True,

            col="Zaman")

plt.show()
#7.COUNT PLOT



g=sns.catplot(x="Cinsiyet",

            data=Bahsis,

            kind="count")

g.set(ylabel="Kisi Sayısı")

plt.show()



g=sns.catplot(x="Sigara",

            data=Bahsis,

            kind="count")



g.set(ylabel="Kisi Sayısı")

plt.show()





g=sns.catplot(x="Sigara",

            data=Bahsis,

            kind="count",

            hue="Cinsiyet")



g.set(ylabel="Kisi Sayısı")

plt.show()