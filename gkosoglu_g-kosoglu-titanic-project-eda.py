# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")#uyarıları kapatmak için





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId = test_df["PassengerId"]
train_df.columns #12 features
train_df.head()
train_df.describe()
#plt.style.available # buradaki tarzlara bak
train_df.info()
def bar_plot(variable):

    """

    

    input : variable ex : "Sex"

    output: bar plot & value count

    

    """

    # get feature

    var = train_df[variable]

    #count number of categorical variables(value / sample)

    varValue = var.value_counts() #value_counts :o cinsiyetten kac tane olduğunu sayar.

    

    #visualize

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index,varValue)

    plt.xticks (varValue.index,varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}:\n {}".format(variable,varValue))
category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for c in category1:

    bar_plot(c)
category2 = ["Cabin","Name","Ticket"]

for c in category2:

    print("{} \n".format(train_df[c].value_counts()))
def plot_hist(variable):

    plt.figure(figsize = (9,3))

    plt.hist(train_df[variable], bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()

    
numericVar = ["Fare","Age","PassengerId"]

for n in numericVar:

    plot_hist(n)
# Pclass v.s. Survived 



train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index = False).mean().sort_values(by="Survived",ascending = False)



# That means First Class passengers %62 percent survived. Money survives you!
# Sex v.s. Survived 



train_df[["Sex","Survived"]].groupby(["Sex"],as_index = False).mean().sort_values(by="Survived",ascending = False)



# Women survived in % 74 percent, men died.
# SibSp v.s. Survived 



train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index = False).mean().sort_values(by="Survived",ascending = False)



# If you have a sister/brother or family member you have %50 of survive.
# Parch v.s. Survived 



train_df[["Parch","Survived"]].groupby(["Parch"],as_index = False).mean().sort_values(by="Survived",ascending = False)



# If you have a parent of child, you will  % 60 percent survive.
#It help to visualize the data healthy. Removes the outlier data.
def detect_outliers(df,features):

    outlier_indices = []

    

    for c in features:

        

        #first quartile

        

        Q1 = np.percentile(df[c],25)

        

        #third quartile

        

        Q3 = np.percentile(df[c],75)

        

        #IQR:

        

        IQR = Q3-Q1

        

        #outlier step

        

        outlier_step = IQR * 1.5

        

        #detect outlier and their indices

        

        outlier_list_col = df[(df[c]< Q1- outlier_step) | (df[c] > Q3 + outlier_step)].index

        

        # store indices

        outlier_indices.extend(outlier_list_col)

        

        

        # tum sayısal değerler için kullanılır

        

    outlier_indices = Counter(outlier_indices) # hangi indisler kacar outlier içerir onu gösterir.

    multiple_outliers = list(i for i, v in outlier_indices.items() if v> 2) #o indisten 2den cok varsa cıkart yoksa kalsın demektir.



    

    return multiple_outliers
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]
#drop outliers:

train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)
train_df_len = len(train_df)

train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop =True)

# indexi resetlemesinin sebebi tekrar bastan indexleme yapsın diye.
train_df.head()
# Find Missing Values
train_df.columns[train_df.isnull().any()] # train data frame içerisindeki missing values içeren featureları(columns) cıkartır
#kac tane olduklarıın öğrenmek için:

train_df.isnull().sum()



# yani 418 yolcunun yasayıp yasamadıgını bilmiyoruz, 1007 yolcunun kabin numarasını, ve 256 yolcunun yaşını bilmiyoruz gibi



# Burada Survived kısmını Machine learning algoritması ile dolduracağım.

#Kabin bilgisi zaten önemli değil

# Age bilgisini de doldurabilirim.Ancak su an yeterince bilgiye sahip değilim.

train_df[train_df["Embarked"].isnull()] #nerden bindikleri belli değil 

#neye göre doldurmalıyız? Cabin numarsına belki Pclassa göre arsılaştırabiliriz.

#Fare göre(kac para odediğine göre bakabiliriz)
train_df.boxplot(column = "Fare", by ="Embarked")

plt.show()



# Q  dusuk değerler ödemişler. S ise muhtemelen ikinci sınıfta binmişler.

# 80 değerine en yakın olan C limanı gibi duruyor. Box yuze yakın bir değere kadar cıkmış.

# S ve Q da outlierları saymazsan zaten 80 para birimine ulaşamamış bile.
train_df["Embarked"] = train_df ["Embarked"].fillna("C")

#artık doldurduk
train_df[train_df["Fare"].isnull()] 

# tek bir yolcunun kaç para ödediği yok sadece nereden bindiği belli (S) limanı.

# nereden bindiğine ve sınıfına bakılabilinir.

#Pclassa bakalım.
# find the average of Fare for Class 3

#np.mean(train_df[train_df["Pclass"]==3]["Fare"])



train_df["Fare"]= train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))



# burada yaptığı su şekilde:

 #1. Pclass 3 olan yolculara bakıyor ve onların odediklerinin ortalamasını alıyor. 

# 2. Sonra bu değeri kullanarak Fare içindeki boş değerleri bununla dolduruyor.

train_df[train_df["Fare"].isnull()] 