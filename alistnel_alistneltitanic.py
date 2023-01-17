# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')





import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")  #py'dan dolayı kod çalısıyor ancak uyarıları gösterme dioruz, hata varsa ztn gözükecek



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



train_df = pd.read_csv("/kaggle/input/titanic/train.csv")    #veri seti okuma

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")    



#verisetini incelemek adına bir baktık passangerId tcno gibi. İlerde değişiklik yaparken  unutmayalım diye simdiden kaydedelim

test_PassengerId = test_df["PassengerId"]
train_df.head()
train_df.info()             #train datasının başlıklarının veri tipleri
#kategorik özellikleri incelemeye başlayalım, verileri görselleştirip inceleyelim



def bar_plot(variable):

    '''

    input= variable örn: "Cabin"

    output= bar plot & value count

    '''

    var = train_df[variable]

    varValue = var.value_counts()  # x.value_counts() metodu değişken içinde kaç tane sınıf ve sınıflarda kaç örnek olduğunu döner

    

    

    #görselleştirme

    plt.figure(figsize= (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Sayılar")

    plt.title(variable)

    plt.show()

    print("{}:\n {}".format(variable,varValue) )
#şimdi yazdığımız methodu cağıralım, 

#cat diye bir listem olsun ondan cekelim ve bakalım



category1 = ["Survived", "Sex", "Pclass", "Embarked","SibSp","Parch"]

for c in category1:

    bar_plot(c)
#yukarıdakileri görselleştirmek kolay

#ama name gibi sınıflar başa bela. bunları grafikten zyade düz yazalım.



category2 = ["Cabin", "Name", "Ticket"]

for c in category2:

    print("{}\n".format(train_df[c].value_counts()))
#sayısal değişkenleri incelerkende histogram plot kullanalım



def plot_hist(variable):

    plt.figure(figsize = (9,3))

    plt.hist(train_df[variable], bins= 80) #bins histogram örnekleme

    plt.xlabel(variable) 

    plt.ylabel("Sayılar")

    plt.title("{} histogram tablosu".format(variable))

    plt.show()
numCategory = ["Fare", "Age", "PassengerId"]

for m in numCategory:

    plot_hist(m)
# Pclass vs Survived



# pclass ile surv çekip, pclass'a göre gruplama yapıp, ortalamasını alalım

 

train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean()  #sonuna .sort_values(by='Survived', ascending = False) ile sıralanmış şekliyle yazabilirim
# Sex vs Survived   



 

train_df[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by='Survived', ascending = False)
#SibSp - Survived 

train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by='Survived', ascending = False)
#Parch - Survived 

train_df[["Parch", "Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
def detect_outliers(df,features):

    outlier_indices = []

    

    for c in features:

        # Q1

        Q1 = np.percentile(df[c],25)

        # Q3

        Q3 = np.percentile(df[c],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier step

        outlier_step = IQR * 1.5

        # outlier tespit

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        # çıktıları toplamak

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    

    return multiple_outliers
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]
#outlier'ları çıkarmak



train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)
sayac = 0
#boş değerler tüm verisetinin içerisinde, hepsini tespit edeceğiz ve bundan dolayı train ve test df'lerini bir araya getiriyoruz

train_df_len = len(train_df)



if sayac == 0:

        train_df = pd.concat([train_df, test_df], axis = 0).reset_index(drop= True) 

        sayac+=1



#şimdi bu secsion'ı birden fazla run edersen her defasında yeni train ile testi devamlı bir şekilde birleştirecek HATA!
train_df_len
#simdi train df içinde missing value olanlara bakalım

train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()   #boş olanların kacar tane olduğuna bakalım
#doldurma/düzeltme yöntemleri: boşları çıkarabilirim -veri kaybı-

# içlerini doldururum



train_df[train_df["Embarked"].isnull()]  #hangi değerler boş bakalım

#şimdi az olan bu şeylerin doldurulmasını yorumlayalım

#sayısal olarak Fare-Cabin-PClass arasında bir bağlantı kurup boş değerleri doldurbilirz

train_df.boxplot(column = "Fare", by= "Embarked")  #

plt.show()

#C Q S limanlarından yeşil cizgiler -medyanları- bizim 80'lik Fare'ımıza C daha yakın

# o zaman embarkedlarımızı C limanına göre yaparız
train_df["Embarked"] = train_df["Embarked"].fillna("C") 

#şaun embarkedlar C limanına göre dolduruldu
train_df[train_df["Embarked"].isnull()]  #hangi değerler boş bakalım

#bakın artık boş  değerler yok edildi
#aynısını Fare  için uygulayalım

train_df[train_df["Fare"].isnull()] #hangi adamın Fare'ı Nan



# 1033 noluadam para bilgisi yok

#adam 3.ncü sınf yolcu simdi bundan yola cıkarak deneyelim birşeyler
train_df[train_df["Pclass"]==3]["Fare"]   #pclass 3 olanların Fare'ları
np.mean(train_df[train_df["Pclass"]==3]["Fare"])  #ortalaması
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))
train_df[train_df["Fare"].isnull()]  #artık fare'larda boş data yok
train_df[train_df["PassengerId"]== 1044]  

#kontrol ediyorum harbiden ortalamyı bizim adamın Fare'ına koymuş HELAL!
#korelasyon matrisi için Seaborn kullanacağız

list1 = ["SibSp", "Parch", "Age", "Fare","Survived"]

sns.heatmap(train_df[list1].corr(), annot=True, fmt= ".2f")   

#annot corelasyon üzerindeki değerler

plt.show()
g = sns.factorplot(x="SibSp", y="Survived", data= train_df, kind= "bar", size= 5)

plt.show()
g = sns.factorplot(x="Parch", y="Survived", data=train_df, kind="bar", size=5)

plt.show()
g= sns.factorplot(x="Pclass", y="Survived", data= train_df, size=5, kind="bar")

plt.show()
g = sns.FacetGrid(train_df, col="Survived")

g.map(sns.distplot, "Age", bins=25)

plt.show()
g = sns.FacetGrid(train_df, col="Survived", row="Pclass", size=3)

g.map(plt.hist, "Age", bins= 25)

g.add_legend()

plt.show()
g= sns.FacetGrid(train_df, row="Embarked", size=3)

g.map(sns.pointplot, "Pclass", "Survived", "Sex")

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df, row="Embarked", col="Survived", size=2.5)

g.map(sns.barplot, "Sex", "Fare")

g.add_legend()

plt.show()
train_df[train_df["Age"].isnull()]  #yaş bilgiisi bulunmayan yolcuları listele
sns.factorplot(x="Sex", y="Age", data=train_df, kind="box")

plt.show()
sns.factorplot(x="Sex", y="Age", hue="Pclass", data=train_df, kind="box")

plt.show()
sns.factorplot(x="Parch", y="Age", data=train_df, kind="box")

sns.factorplot(x="SibSp", y="Age", data=train_df, kind="box")

plt.show()
# korelasyonu sayılar arasında arıyoruz Sex feature'ı male female gibi str ifadeler

# Sex feature'ını sayısallaştıralım

train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]] 
sns.heatmap(train_df[["Age","Sex", "SibSp", "Parch", "Pclass"]].corr(), annot=True)

plt.show()
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)

#age'lerin isnull olanlarını atar



#simdi bu indexleri tek tek dolasalım ve oradaki sibsp, parch, pclass'lara 

#göre tahmin ettirelim

for i in index_nan_age:

    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) & (train_df["Parch"] == train_df.iloc[i]["Parch"]) & (train_df["Pclass"] == train_df.iloc[i]["Pclass"])) ].median()

    #bazılarında NAN kalmış, sebebi gelen yerlerde de NaN olabilir, onlar için eski usul

    age_med= train_df["Age"].median()

    if not np.isnan(age_pred):

        train_df["Age"].iloc[i] = age_pred 

    else:

        train_df["Age"].iloc[i] = age_med 
train_df[train_df["Age"].isnull()]   #age içerisindeki tüm NaN değerleri temizlendi
train_df["Name"].head(10)
name= train_df["Name"]

train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name] #boşluk var
#mr mrs dısındakiler var, acaba bunların hayatta kalma durumları nasıl



sns.countplot(x="Title", data=train_df)

plt.xticks(rotation= 60)

plt.show()
train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")

train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]

train_df["Title"].head(20)

#BURADA önemli bir ayrıntı var, bu seccsion birden fazla defa çalışırsa hepsini 3 yapar

# karşılaştırma yapısında bulamayacağı için hepsi 3 olur

#böyle sorunlar için run all yapabiliriz
sns.countplot(x="Title", data=train_df)

plt.xticks(rotation= 60)

plt.show()
g = sns.factorplot(x="Title", y="Survived", data= train_df, kind = "bar")

plt.show()
train_df.drop(labels="Name", axis=1, inplace= True)   #name'i Title feature'ı ile mükemmeleştirdik sil gitsn
train_df.head(10)
train_df = pd.get_dummies(train_df,columns=["Title"])  #title'ı bile bölüyoruz

train_df.head()
#burası parch ve sibsp birleşimi olacak olan famly grubunun oluşturulması

# kardeş ana baba sayılarını toplayalım ki aile üye sayısı çıksın

#+1 yapalım ki kimsesizler 1 kişilik bir aile olsunlar



train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1
g = sns.factorplot(x="Fsize", y="Survived", data= train_df, kind="bar")

plt.show()
train_df["family_size"] = [1 if i< 5 else 0 for i in train_df["Fsize"]]
train_df.head(15)
sns.countplot(x="family_size", data= train_df)   #iyi güzelce bölmüş, kaçar tane?

plt.show()
g = sns.factorplot(x="family_size", y="Survived", data= train_df, kind="bar")

plt.show()
train_df = pd.get_dummies(train_df, columns=["family_size"])

train_df.head()
train_df["Embarked"].head()
sns.countplot(x="Embarked", data=train_df)

plt.show()
train_df = pd.get_dummies(train_df, columns=["Embarked"])

train_df.head()
train_df["Ticket"].head(14)  #baştaki bazı yazılar ilginç vs değiştirelim
tickets = []

for i in list(train_df.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        tickets.append("x")

train_df["Ticket"] = tickets
train_df["Ticket"].head(20)
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
train_df.columns
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_df_len
test = train_df[train_df_len:]

test.drop(labels = ["Survived"],axis = 1, inplace = True)
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
test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)

results = pd.concat([test_PassengerId, test_survived],axis = 1)

results.to_csv("titanic.csv", index = False)
test_survived