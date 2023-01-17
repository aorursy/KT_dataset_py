# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid") #burda matplotlibin stilini seçiyoruz(arkası karelere bölünmüş). diğer stilleri görmek için plt.style.avaliable yaz çalıştır çıkar. 

import seaborn as sns

from collections import Counter



import warnings 

warnings.filterwarnings("ignore") #Pythondan kaynaklı hataları gösterme demek.



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df=pd.read_csv("/kaggle/input/titanic/train.csv")

test_df=pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId=test_df["PassengerId"]
train_df.columns
train_df.head()
train_df.describe()
train_df.info() #sütunlarla ilgili özellikler.
def bar_plot(variable):

    """

    input: variable ex: "Sex"

    output: barplot & value count

    

    """

    #get feature 

    var=train_df[variable]  

    #count number of categorical variable (value)

    varValue=var.value_counts() #hangi başlıktan kaç tane olduğunu sayıyor.(value_counts)

    

    #visualize

    plt.figure(figsize=(9,3)) #tablolarımızın boyutu

    plt.bar(varValue.index,varValue) #sütun grafiği gibi tür sayılarını görmemizi sağlıyor.(x eksenine indexleri yaz y eksenine adetlerini)

    plt.xticks(varValue.index,varValue.index.values) #x eksenindeki noktalar sadece bizim çeşitlerimiz kadar olsun demek.

    plt.ylabel("Frequency") #y label adı ( adet)

    plt.title(variable) #tablo başlığı gelen sütunumuzun adı olsun.

    plt.show() #her zamanki gibi.

    print("{}: \n {}".format(variable,varValue)) #ekrana gelen sütun adını ve hangi türden kaçar tane olduğunu yazdık.

    
category1=["Survived","Sex","Pclass","Embarked","SibSp","Parch"] #kategorik sütunlarımızı bir listeye attık.

for i in category1:

    bar_plot(i) #kategorik sütunlarımızın hepsini tek tek üstteki fonksiyonumuza yolladık.
category2=["Cabin","Name","Ticket"] #bunlar çok çeşitli sütunlar olduğu için herhangi bir analiz yapınca anlamlı olmayan şeylerler karşılaşmak olası.

for i in category2:

    print("{} \n {}".format(i,train_df[i].value_counts()))
def hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(train_df[variable],bins=50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} Histogram".format(variable))

    plt.show()
numerical=["Age","Fare","PassengerId"]

for i in numerical:

    hist(i)
##Pclass vs Survived

train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index = False).mean().sort_values(by="Survived",ascending=False)

#tablodaki Pclass ve Survived sütunlarını al bunları Pclass a göre sınıflandır. Ve ortalamasını al. Büyükten küçüğe sırala.

#Bu tablo ile hangi sınıf insanın sağ olma olasılığını gözlemleyebiliriz.(1 sağ 0 ölüydü.)

train_df[["Sex","Survived"]].groupby(["Sex"],as_index = False).mean().sort_values(by="Survived",ascending=False)

#bu tablo ile kadının hayatta kalma olasılığının daha çok olduğunu görebiliriz.
train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index = False).mean().sort_values(by="Survived",ascending=False) #(mean dan sonraki ölüm oranına göre büyüktn küçüğe sıralamak için.)

#yanında bir kardeşi veya eşi olanların hayatta kalma olasılığı en yüksekmiş.
train_df[["Parch","Survived"]].groupby(["Parch"],as_index = False).mean().sort_values(by="Survived",ascending=False)

#Yanımızda 3 kişilik bir ailemiz varsa kurtulma olasılığımız daha yüksekmiş.
#hangi sınıftan kaç tane var?

train_df[["Pclass","Sex"]].groupby(["Pclass"],as_index=False).count().sort_values(by="Sex",ascending=False)
train_df["expensiveness"]=["cheap" if i<50.0 else "expensive" for i in train_df.Fare]

dataexpensive=train_df[["expensiveness","Survived"]].groupby(["expensiveness"],as_index=False).mean().sort_values(by="Survived",ascending=False)

plt.figure(figsize=(9,3))

plt.bar(dataexpensive.expensiveness,dataexpensive.Survived,color="gray")

plt.title("The effect of wealth on survival")

plt.show()

print(dataexpensive.expensiveness[0],":",dataexpensive.Survived[0])

print(dataexpensive.expensiveness[1],":",dataexpensive.Survived[1])
#ayrık veriyi bulmak için olan formülü kullandık.

def detect(df,features):

    outlier_indices=[]

    for i in features:

        #1st quartile

        q1=np.nanpercentile(df[i],25) #1.medyanı buluyor.

        #3.quartile

        q3=np.nanpercentile(df[i],75) #3.medyanı buluyor.

        #IQR

        ıqr=q3-q1  #formülden 

        #outlier step

        outlier_step=ıqr*1.5  #formülden.

        #detect outlier and their indeces

        outlier_list_col=df[(df[i]<q1-outlier_step) | (df[i]>q3+outlier_step)].index #ayrık verinin indexini bulduk.

        #store indeces

        outlier_indices.extend(outlier_list_col) #outlier_indices dizimize ekledik.

    

    outlier_indices=Counter(outlier_indices) #liste içindeki elemanların kaçar tane olduğunu bulur.

    multiple_outliers=list(i for i,v in outlier_indices.items() if v>2) #eğer aynı veriden 2 tane varsa bul diziye at.

    return multiple_outliers
train_df.loc[detect(train_df,["Age","SibSp","Parch","Fare"])]
#drop outliers  #düzeni bozan verileri sil.

train_df=train_df.drop(detect(train_df,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)
train_df_len=len(train_df)

train_df=pd.concat([train_df,test_df],axis=0).reset_index(drop=True)

train_df
train_df.columns[train_df.isnull().any()] #içinde null veri olan sütunları getir.
train_df.isnull().sum() #hangi sütunda ne kadar boş karakter var onları bul.
train_df[train_df["Embarked"].isnull()] #embarked sütunun boş olan satırı bul getir.

#hangi limandan bindiklerini bilmediğimiz kişiler 1.sınıf yolcu ve 80 dolara bilet almış.
train_df.boxplot(column="Fare",by="Embarked")

plt.show()

#aşağıdaki grafikte anlaşılacağı gibi 80 ve daha çok veren kişiler en çok C den katılmış.C nin medyanı 80 e en yakın liman.Q olma ihtiamli çok düşük çünkü en ucuz biletler orda satılmış.
train_df.Embarked=train_df["Embarked"].fillna("C") #burda da embarkedi boş olanları C yi attık.

train_df[train_df["Embarked"].isnull()]
train_df[train_df["Fare"].isnull()]
train_df.Fare=train_df["Fare"].fillna(train_df[train_df["Pclass"]==3]["Fare"].mean())

#sınıfı 3 olan kişiler ortalama ne kadar  ödemiş onu bulduk.Ödemeyen kişi de ortalama o kadar ödemiştir diyip onu kabul ettik.

train_df[train_df["Fare"].isnull()]  