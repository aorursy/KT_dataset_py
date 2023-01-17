# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import the dataset

train_df=pd.read_csv("/kaggle/input/titanic/train.csv")

test_df=pd.read_csv("/kaggle/input/titanic/test.csv")

test_passengerId=test_df["PassengerId"]

train_df.columns



"""

Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',

       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],

      dtype='object')



"""

train_df.head()
train_df.describe()
train_df.info()

def bar_plot(variable):

    """

        input: variable example : "Sex"

        output : bar plot & value count

    """

    #get feature

    var= train_df[variable] 

    #count number of categorical variable(value/sample)

    varValue=var.value_counts() #kaç kategori oldğunu ve kategorilerde kaç tane  veri olduğunu buluruz

    

    #visualize

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index, varValue) # x ve y e ekseninden oluşan bir bar plottur  

    plt.xticks(varValue.index , varValue.index.values)

    plt.ylabel("Frekans")

    plt.title(variable)

    plt.show()

    print(format(variable) + "\n"+  format(varValue))

    

    
category1= [ "Survived" , "Sex" , "Pclass" , "Embarked" , "SibSp" , "Parch"]

for c in category1:

    bar_plot(c)

    
category2=["Cabin","Name","Ticket"]

for c in category2:

    print(format(train_df[c].value_counts()))
def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(train_df[variable],bins=50)

    plt.xlabel(variable)

    plt.ylabel("Frekansı")

    plt.title(format(variable))

    plt.show()
numericVar=["Fare","Age","PassengerId"]



for n in numericVar:

    plot_hist(n)

    
# Pclass - Survived





"""

	Pclass	Survived        1.Classtaki %62 olasılıkla bir yolcu hayatta kalmış

0	1	0.629630            2.Classtaki %47 olasılıkla bir yolcu hayatta kalmış

1	2	0.472826            3.Classtaki %24 olasılıkla bir yolcu hayatta kalmış

2	3	0.242363

"""



train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived",ascending=False) #ortalamasını gösteriyoruz 

# Pclass ve surviived featurelerimizi aldık Pclassa göre gruplandırdık . Bunların ortalamasını aldık ve sıralı bir şekilde yazdırdık
#sex-survived





"""

	Sex	Survived

0	female	0.742038    kadınlar %74 oranında hayatta kalmış

1	male	0.188908    ekrkler %18 oranında hayatta kalmış



"""



train_df[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived",ascending=False)
#Sibsp-survived





"""



SibSp	Survived

1	1	0.535885       yanında 1 kişi olanların hayatta kalma oranı %53

2	2	0.464286       yanında 2 kişi olanların hayatta kalma oranı %46

0	0	0.345395       yanında 0 kişi olanların hayatta kalma oranı %34

3	3	0.250000       yanında 3 kişi olanların hayatta kalma oranı %25

4	4	0.166667       yanında 4 kişi olanların hayatta kalma oranı %16

5	5	0.000000       yanında 5 kişi olanların hayatta kalma oranı %0

6	8	0.000000       yanında 8 kişi olanların hayatta kalma oranı %0



"""





train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived",ascending=False)
#Parch - Survived 



"""



Parch	Survived

3	3	0.600000    yanımızda çccuk-yada aileüyemiz(parent ) varsa ve bu sayı 3 ise %60 oranında hayattayız demek

1	1	0.550847    yanımızda çccuk-yada aileüyemiz(parent ) varsa ve bu sayı 1 ise %55 oranında hayattayız demek

2	2	0.500000    yanımızda çccuk-yada aileüyemiz(parent ) varsa ve bu sayı 2 ise %50 oranında hayattayız demek

0	0	0.343658    yanımızda çccuk-yada aileüyemiz(parent ) yok  ise %34 oranında hayattayız demek

5	5	0.200000    yanımızda çccuk-yada aileüyemiz(parent ) varsa ve bu sayı 5 ise %20 oranında hayattayız demek

4	4	0.000000    yanımızda çccuk-yada aileüyemiz(parent ) varsa ve bu sayı 4 ise %0 oranında hayattayız demek

6	6	0.00000     yanımızda çccuk-yada aileüyemiz(parent ) varsa ve bu sayı 6 ise %0 oranında hayattayız demek

"""



train_df[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived",ascending=False)
def detec_outlier(df,features):

    outlier_indices=[]

    for c in features:

        #1st quartile

        Q1=np.percentile(df[c],25)

        #3rd quartile

        Q3=np.percentile(df[c],75)

        #IQR 

        IQR=Q3-Q1

        #Outlie step

        outlier_step=IQR * 1.5

        #detec outlier and their indeces

        outlier_list_column=df[(df[c]< Q1 - outlier_step) | (df[c]>Q3 + outlier_step)].index

        #store indeces

        outlier_indices.extend(outlier_list_column)

        

    outlier_indices=Counter(outlier_indices)

    multiple_outliers=list ( i for i , v in outlier_indices.items() if v > 2)

    return multiple_outliers

        
train_df.loc[detec_outlier(train_df,["Age","SibSp","Parch","Fare"])]
#drop outlier 

train_df = train_df.drop(detec_outlier(train_df,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)

train_df
#data frammleri birleştirmemiz gerekiyor 

train_df_len=len (train_df)

train_df=pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
train_df.head()
train_df.columns[train_df.isnull().any()] #hangi column larda null değeri var tespit ediyoruz
train_df.isnull().sum() 

"""

PassengerId       0

Survived        418

Pclass            0

Name              0

Sex               0

Age             256

SibSp             0

Parch             0

Ticket            0

Fare              1

Cabin          1007

Embarked          2

dtype: int64



"""
#embarked in veride nerede ?

train_df[train_df["Embarked"].isnull()]
#fare kullanarak embarked hakkında fikir sahibi olabiliriz

train_df.boxplot(column="Fare",by="Embarked")

plt.show()





""" 

Burada çıkan sonuçları değerlendirdiğimizde 

Q limanında binenler çok az para ödemişler . büyük ihtimal bunlar 3.sınıf insanlar 

S limanında binenler  biraz daha para vermişler bunlarda büyük ihrimal 2. insanlar

C limannda ise genelde yüksek fiyat ödeyenler var medyan değerkeri 100 ün biraz altında  



Bizim üst taraftaki embarked değerlerini kıyasladığımızda orada fare kısmında 80.0 değeri vardı buda bizim bu aşşağıdaki tablomuzda

gösteriyor ki bun insanlarlar muhtemelen C limanından bindiler gemiye

bunun için burada kayıp değer olarak verilen embarked değerini biz burada C limanından binmiş olarak verebiliriz 





"""
train_df["Embarked"]=train_df["Embarked"].fillna("C")

# Embarked değeri boş olan 2 tane tek girdi vardı ve bunların fare leri 80 idi biz bunları yolcuların ödedikleri paraya göre 

#istatistiğini çıkarıp medyanlarını  karşılaştığımızda  gördük ki faar yani ödedikleri para miktarı yüksek olanlar(80'de bu sınıfta) C limannda binmişler 

#bizde bu boş olan miisssing data değerlerine faar leri yani ödedikleri para 80 olduğu için bunları C limanı olarak yazdık

train_df[train_df["Embarked"].isnull()]

#kontrol ediyoruz 
train_df[train_df["Fare"].isnull()] # burada fare değeri boş olan kim var buna bakıyoruz

#1033 nolu yolcunun fare değeri yokmuş diğer feature değerlreinden bazılarıda bilinmiyor 

#burada biziö bakaileceğimiz  husus var biri Pclass ı yani sınıfları diğeri ise hangi limandan bindiği

#Pclass ın 3 olması 


np.mean(train_df[train_df["Pclass"]==3]["Fare"]) # pclass değeri 3 olan yolcuların ortalama ödedileri ücrete bakıyoruz =12.7 

train_df["Fare"]=train_df["Fare"].fillna(np.mean( train_df [train_df ["Pclass"]==3 ] ["Fare"])) #NAN olan fare değerine yukarıda bulduğumuz ortalamayı yazıyoruz 

train_df[train_df["Pclass"].isnull()] # kontrol ediyoruz
list1=["SibSp","Parch","Age" ,"Fare", "Survived"]

sns.heatmap(train_df[list1].corr(),annot=True , fmt=".2f")
g = sns.factorplot(x="SibSp",y="Survived",data=train_df,kind="bar",size=6)

g.set_ylabels("Hayatta Kalma Olasılığı")

plt.show()
a= sns.factorplot(x="Parch",y="Survived",kind="bar",data=train_df,size=6)

a.set_ylabels("Hayatta Kalma Olasılığı")

plt.show()
c = sns.factorplot(x="Pclass",y="Survived",data=train_df,kind="bar",size=6)

c.set_ylabels("Hayatta Kalma Olasılığı")

plt.show()



d = sns.FacetGrid(train_df,col="Survived")

d.map(sns.distplot,"Age",bins=25)

plt.show()



g = sns.FacetGrid(train_df,col="Survived",row="Pclass",size=3)

g.map(plt.hist,"Age",bins=25)

g.add_legend()

plt.show
g= sns.FacetGrid(train_df,row="Embarked", size= 3)

g.map(sns.pointplot,"Pclass","Survived","Sex")

g.add_legend()

plt.show()
g= sns.FacetGrid(train_df,row="Embarked",col="Survived", size= 4)

g.map(sns.barplot,"Sex","Fare")

g.add_legend()

plt.show()
train_df[train_df["Age"].isnull()]
sns.factorplot(x="Sex",y="Age",data=train_df,kind="box")

plt.show()
sns.factorplot(x="Sex",y="Age",hue="Pclass",data=train_df,kind="box")

plt.show()
sns.factorplot(x="Parch",y="Age",data=train_df, kind="box")

sns.factorplot(x="SibSp",y="Age",data=train_df, kind="box")

plt.show()
train_df["Sex"]=[1 if i == "male" else 0 for i in train_df["Sex"]]
sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(),annot=True)

plt.show()
#Age kısmında boşlukları doldurmak için kod yazmamız lazım bunun için ilk önce kaç tane null değer var görüntülemeyliyiz

index_nan_age=list(train_df["Age"][train_df["Age"].isnull()].index)

#index_nan_age

for i in index_nan_age:

    age_iliski=train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) & (train_df["Parch"] == train_df.iloc[i]["Parch"])& (train_df["Pclass"] == train_df.iloc[i]["Pclass"])) ].median()

    #bazı değerlerdede prediction yapamadığımız oluyor bunun sebebi diğer değerlerdeki nan valuelr

    #bunun için eski kullandığımız en basit yöntem olan medyanlarıyla dolduralım

    age_median =train_df["Age"].median()

    if not np.isnan(age_iliski):

        train_df["Age"].iloc[i]=age_iliski

    else :

        train_df["Age"].iloc[i]=age_median
#yukarıda null değerlerini yok ettiğimiz age sütununa tekrak bakıyoruz null değer varmı diye

train_df[train_df["Age"].isnull()]
train_df["Name"].head(10)

name = train_df["Name"]

train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]

train_df["Title"].head(10)
sns.countplot(x="Title", data = train_df)

plt.xticks(rotation = 60)

plt.show()
#Kategorikal Değerlere Dönüştürme



train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")

train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]

train_df["Title"].head(20)
sns.countplot(x="Title", data = train_df)

plt.xticks(rotation = 60)

plt.show()
g = sns.factorplot(x = "Title", y = "Survived", data = train_df, kind = "bar")

g.set_xticklabels(["Master","Mrs","Mr","Other"])

g.set_ylabels("Survival Probability")

plt.show()
train_df.drop(labels = ["Name"], axis = 1, inplace = True)
train_df.head()

train_df = pd.get_dummies(train_df,columns=["Title"])

train_df.head()
train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1
train_df.head()
g = sns.factorplot(x = "Fsize", y = "Survived", data = train_df, kind = "bar")

g.set_ylabels("Survival")

plt.show()
train_df["family_size"] = [1 if i < 5 else 0 for i in train_df["Fsize"]]
train_df.head(10)
sns.countplot(x = "family_size", data = train_df)

plt.show()
g = sns.factorplot(x = "family_size", y = "Survived", data = train_df, kind = "bar")

g.set_ylabels("Survival")

plt.show()
train_df = pd.get_dummies(train_df, columns= ["family_size"])

train_df.head()
train_df["Embarked"].head()

sns.countplot(x = "Embarked", data = train_df)

plt.show()
train_df = pd.get_dummies(train_df, columns=["Embarked"])

train_df.head()
train_df["Ticket"].head(20)

a = "A/5. 2151"

a.replace(".","").replace("/","").strip().split(" ")[0]

tickets = []

for i in list(train_df.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        tickets.append("x")

train_df["Ticket"] = tickets
train_df["Ticket"].head(20)

train_df.head()

train_df = pd.get_dummies(train_df, columns= ["Ticket"], prefix = "T")

train_df.head(10)
sns.countplot(x = "Pclass", data = train_df)

plt.show()

train_df["Pclass"] = train_df["Pclass"].astype("category")

train_df = pd.get_dummies(train_df, columns= ["Pclass"])

train_df.head()
train_df["Sex"] = train_df["Sex"].astype("category")

train_df = pd.get_dummies(train_df, columns=["Sex"])

train_df.head()
train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)
from sklearn.model_selection import train_test_split, StratifiedKFold , GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier , VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_df_len
test = train_df[train_df_len:]

test.drop(labels=["Survived"],axis=1,inplace = True)
test.head()
train = train_df[:train_df_len]

X_train = train.drop(labels = "Survived", axis = 1)

y_train = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)

print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

acc_log_train = round(logreg.score(X_train, y_train)*100,2) 

acc_log_test = round(logreg.score(X_test,y_test)*100,2)

print("Training Accuracy: % {}".format(acc_log_train))

print("Testing Accuracy: % {}".format(acc_log_test))
random_state = 42

classifier = [DecisionTreeClassifier(random_state = random_state),

             SVC(random_state = random_state),

             RandomForestClassifier(random_state = random_state),

             LogisticRegression(random_state = random_state),

             KNeighborsClassifier()]



dt_param_grid = {"min_samples_split" : range(10,500,20),

                "max_depth": range(1,20,2)}



svc_param_grid = {"kernel" : ["rbf"],

                 "gamma": [0.001, 0.01, 0.1, 1],

                 "C": [1,10,50,100,200,300,1000]}



rf_param_grid = {"max_features": [1,3,10],

                "min_samples_split":[2,3,10],

                "min_samples_leaf":[1,3,10],

                "bootstrap":[False],

                "n_estimators":[100,300],

                "criterion":["gini"]}



logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty": ["l1","l2"]}



knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}

classifier_param = [dt_param_grid,

                   svc_param_grid,

                   rf_param_grid,

                   logreg_param_grid,

                   knn_param_grid]
cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)

    clf.fit(X_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",

             "LogisticRegression",

             "KNeighborsClassifier"]})



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),

                                        ("rfc",best_estimators[2]),

                                        ("lr",best_estimators[3])],

                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(X_train, y_train)

print(accuracy_score(votingC.predict(X_test),y_test))