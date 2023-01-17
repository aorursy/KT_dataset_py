#First of all we define the libraries (Python) maybe we can add rest of the libs 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

plt.style.use("seaborn-whitegrid") #This command provides that the graphics has a grid by using seaborn-whitegrid style

#if we want to learn the other styles run this command >>>>>> plt.style.available 

import seaborn as sns 

from collections import Counter #By this command we can see the options after using dat(.)

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Then we need read the csv files 

train_df=pd.read_csv("/kaggle/input/titanic/train.csv")

test_df=pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId = test_df["PassengerId"]
#Let's see the columns 

train_df.columns
train_df.head()
train_df.describe()
train_df.info()

def bar_plot(variable):

    """

        input: variable ex: "Sex"

        output: bar plot & value count

    """

    # get feature

    var= train_df[variable]

    #count number of categorical variable(value/sample)

    varValue = var.value_counts()

    

    #visualize

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index,varValue)

    plt.xticks(varValue.index,varValue.index.values) #With this command we can limit the ticks  

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))
category1=["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for c in category1:

    bar_plot(c)
category2=["Cabin","Name","Ticket"]

for c in category2:

    print("{} \n".format(train_df[c].value_counts()))
def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(train_df[variable],bins=100)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distrubution with hist".format(variable))

    plt.show()
numericVar = ["Fare","Age","PassengerId"]

for n in numericVar:

    plot_hist(n)

    
# Pclass vs Survived 

train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending = False)
# Sex vs Survived

# Pclass vs Survived 

train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived",ascending = False)
train_df["Sex"].value_counts()
train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index= False).mean().sort_values(by="Survived",ascending = False)
#Parch vs Survived 

train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending = False)
def detect_outliers(df,features):

    outlier_indices = []

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(df[c],25)

        # 3rd quartile

        Q3 = np.percentile(df[c],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier step

        outlier_step = IQR * 1.5

        # detect outlier and their indeces

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        # store indeces

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    

    return multiple_outliers
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]
train_df= train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)
#Burada tüm dataframeleri birleştirmeliyiz çünkü makine öğrenmesinde herhangi boş değer görürse hata verecektir 

train_df_len = len(train_df)

train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
train_df.columns[train_df.isnull().any()] #Herhangi null değeri barındıran feature var mı diye baktık ki var
train_df.isnull().sum()
#İster doldururum ister boşaltırım ama bizim için doldurmak daha optimum bir çözümdür 

train_df[train_df.Embarked.isnull()]

#Burada embarked ları doldurmak için bilet fiyatına ve Pclass a bakmak mantıklı olacaktır 
train_df.boxplot(column="Fare",by="Embarked")

plt.show()

#Görüldüğü üzere medyanı en yüksek olan C yani 1st class yolcular muhtemelen C dir
train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df.Embarked.isnull()] #Görüldüğü üzere boş yer kalmadı 
train_df[train_df.Fare.isnull()]
#3.sınıf bir yolcu ve S limanından binmiş 

train_df[train_df.Pclass == 3]["Fare"].mean() #Burada 3.sınıf yolcuların ödediği bilet fiyatı ortalamasını aldık ve bununla dolduralım.
train_df["Fare"] = train_df["Fare"].fillna(train_df[train_df.Pclass == 3]["Fare"].mean())
train_df[train_df.Fare.isnull()] #Görüldüğü üzere boş bir yer kalmadı 
#Burada Seaborn Kütüphanemizde yer alan heatmap i kullanarak korelasyon analizi yapacagız 

list1= ["SibSp","Parch","Age","Fare","Survived"]

sns.heatmap(train_df[list1].corr(), annot=True, fmt =".2f") #annot True kutularda değer gözükmesini, ftm .2f ise virgülden sonra 2 değer göstermeye yarar

plt.show()

g=sns.factorplot(x="SibSp",y="Survived",data=train_df, kind="bar",size=7)

g.set_ylabels("Surived Probability")

plt.show()
g=sns.factorplot(x="Parch",y="Survived", kind="bar",data=train_df,size=7)

g.set_ylabels("Survived Probability")

plt.show()
g=sns.factorplot(x="Pclass",y="Survived",data=train_df, kind="bar", size=7)

g.set_ylabels("Survived Probability")

plt.show()
g=sns.FacetGrid(train_df, col="Survived")

g.map(sns.distplot, "Age",bins=25)

plt.show()
g=sns.FacetGrid(train_df, col="Survived",row="Pclass",size=2)

g.map(plt.hist, "Age", bins=25)

g.add_legend()

plt.show()
g=sns.FacetGrid(train_df, row="Embarked", size=3)

g.map(sns.pointplot, "Pclass","Survived","Sex")

g.add_legend()

plt.show()
g=sns.FacetGrid(train_df, row="Embarked", col="Survived", size =3)

g.map(sns.barplot, "Sex","Fare")

g.add_legend()

plt.show()
train_df.isnull().sum()
train_df[train_df["Age"].isnull()] #Age Feature içindeki nulları bul ve train df içinde göster
sns.factorplot(x="Sex", y="Age", data = train_df, kind = "box")

plt.show()
sns.factorplot(x="Sex",y="Age",hue="Pclass", data = train_df, kind="box")

plt.show()
sns.factorplot(x="Parch",y="Age", data = train_df, kind="box")

sns.factorplot(x="SibSp",y="Age", data = train_df, kind="box")

plt.show()
train_df["Sex"] = [1 if i =="male" else 0 for i in train_df["Sex"]]
#Bir de korelasyona bakalım

sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(), annot=True)

plt.show()

index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index) #Burada df deki Age değişkenlerindeki boşları bul indexlerini bul ve listele

for i in index_nan_age: 

    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"])&

                                (train_df["Parch"] == train_df.iloc[i]["Parch"])& 

                                (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median() #Burada indexlerini çektik ve medyan aldık

    age_med = train_df["Age"].median() 

    if not np.isnan(age_pred):

        train_df["Age"].iloc[i] = age_pred

    else: 

        train_df["Age"].iloc[i] = age_med
train_df[train_df["Age"].isnull()] 
train_df["Name"].head(10) # Görüldüğü üzere insanların isimleri var ve title yani ünvanları var
s = "Allen, Mr. William Henry" #elemanımız var biz burda Mr ı almaya calısıyoruz 

#s.split(".")[0].split(",")[-1] #Dediğimde Allen, Mr alırız ama biz sadece Mr istediğimizden , e göre de ayırıcaz 

#Görüldüğü gibi şimdi bunu tüm datada uygulayalım

#Ancak burada ' Mr' şeklinde yani bir bosluk oluyor o yüzden biz boşluk kaldıralım 

s.split(".")[0].split(",")[-1].strip() #Strip ile de boşlugu kaldırdık
name = train_df["Name"]

train_df["Title"] = [i.split(".")[0].split(",")[1].strip() for i in name]
train_df["Title"].head(10) #Görüldüğü üzere Mr ve Mrs olarak ayırdık
sns.countplot(x="Title",data=train_df)

plt.xticks(rotation = 45)

plt.show()
#Converting to Categorical 

train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other") 

#Burada yazılı olanları other adı altında birleştirdik 
sns.countplot(x="Title",data=train_df)

plt.xticks(rotation = 45)

plt.show()
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i== "Ms" or i=="Mle" or i=="Mrs" else 2 if i=="Mr" else 3 for i in train_df["Title"]]
sns.countplot(x="Title",data=train_df)

plt.show()
train_df["Title"].head()
g = sns.factorplot(x="Title",y="Survived",data= train_df, kind="bar")

g.set_xticklabels(["Master","Mrs","Mr","Other"])

g.set_ylabels("Survival Probability")

train_df.drop(["Name"],axis=1, inplace= True) #Tüm sutunu sildik

train_df.head(10)
train_df = pd.get_dummies(train_df , columns = ["Title"]) #Burada Title Feature ı 4 e böldük 

train_df.head()
train_df.rename(columns = {"Title_0":"Master","Title_1":"Mrs","Title_2":"Mr","Title_3":"Other"},inplace = True)

#We changed the column names
train_df.head()
train_df["Famsize"] = train_df["SibSp"] + train_df["Parch"] +1 #Yolcunun kendisini de ekledik

train_df.head()
g = sns.factorplot(x="Famsize",y="Survived",data = train_df, kind ="bar")

g.set_ylabels("Survive Prob")

plt.show()

train_df["Fsize"] = [1 if i<5 else 0 for i in train_df["Famsize"]]

train_df.head()
g = sns.countplot(x="Fsize",data = train_df)

plt.show()

train_df["Fsize"].value_counts()
g = sns.factorplot(x="Fsize",y="Survived",data = train_df, kind ="bar")

g.set_ylabels("Survive Prob")

plt.show()
train_df["Fsize"].value_counts()
train_df = pd.get_dummies(train_df, columns =["Fsize"])
#Şimdi isimlerini değiştirelim 

train_df.rename(columns = {"Fsize_0":"famsize >5","Fsize_1":"famsize <5"},inplace = True)

train_df.head()
train_df["Embarked"].head()
sns.countplot(x="Embarked",data=train_df)

plt.show()

train_df["Embarked"].value_counts()
train_df = pd.get_dummies(train_df, columns = ["Embarked"])

train_df.head()
train_df["Ticket"].head(20)
a = "A/5. 2151"

a.replace(".","").replace("/","").strip().split(" ")[0] 
ticket_list = []

for i in list(train_df.Ticket):

    if not i.isdigit():

        ticket_list.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else: 

        ticket_list.append("x")

train_df["Ticket"] = ticket_list
train_df["Ticket"].head()
train_df = pd.get_dummies(train_df, columns= ["Ticket"], prefix= ("T"))

train_df.head(20)
sns.countplot(x = "Pclass", data=train_df)

plt.show()

train_df["Pclass"].value_counts()
train_df["Pclass"] = train_df["Pclass"].astype("category")

train_df = pd.get_dummies(train_df, columns = ["Pclass"])

train_df.head()
train_df["Sex"] = train_df["Sex"].astype("category")

train_df = pd.get_dummies(train_df, columns =["Sex"])

train_df.head()
train_df.drop(labels = ["PassengerId","Cabin"],axis=1, inplace=True)

train_df.columns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score 
train_df_len #Bu bizim dataframe boyumuzu ver
test = train_df[train_df_len:] #881 ile 1299 arasındaki kısım bu validation olacak

test.drop(labels = ["Survived"],axis =1,inplace =True)

test
train = train_df[:train_df_len] #ilk 881 lik kısım

X_train= train.drop(labels =["Survived"],axis =1)

y_train = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.33,random_state=42)

print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_test",len(y_train))

print("y_test",len(y_test))

print("Test",len(test)) #Bu validation test 
logreg = LogisticRegression()

logreg.fit(X_train,y_train) #y train de labeller var(survived 1 ve 0) x train de ise feature var

acc_log_train = round(logreg.score(X_train,y_train)*100,2)

acc_log_test = round(logreg.score(X_test,y_test)*100,2)

print("Training Accuracy % {}".format(acc_log_train))

print("Test Accuracy % {}".format(acc_log_test))
#Hypertuning En ideal parametrelerin secimidir ve aşağıda gördüklerimiz ml algoritmalarının parametreleridir

#Ve bunları hepsini belirtiyor ve deniyoruz 

#aşağıda vermiş oldugumuz değerleri geniş almak çok daha iyidir ama daha uzun zaman alacaktır.



#Grid Search ise bu parametrelerin birbirleri ile olan kombinasyonlarını inceler ve en iyi değerleri arar 

#mesela 1 değeri ile euclidean distance ve 2 ile euclidean dist karsılastırır hangisi iyise onu alır



random_state = 42 

classifier = [DecisionTreeClassifier(random_state = random_state),

             SVC(random_state = random_state),

             RandomForestClassifier(random_state= random_state),

             LogisticRegression(random_state = random_state),

             KNeighborsClassifier()]



dt_param_grid = {"min_samples_split": range(10,500,20),

                "max_depth": range(1,20,2)}



svc_param_grid = {"kernel": ["rbf"],

                 "gamma": [0.001,0.01,0.1, 1],

                 "C" : [1, 10, 50, 100, 200, 300, 1000]}



rf_param_grid = {"max_features":[1,3,10],

                "min_samples_split" : [2,3,10],

                "min_samples_leaf":[1,3,10],

                "bootstrap" :[False],

                "n_estimators" : [100,300],

                "criterion":["gini"]}



logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty" : ["l1","l2"]}



knn_param_grid ={"n_neighbors":np.linspace(1,19,10,dtype = int).tolist(),

                "weights": ["uniform","distance"],

                "metric":["euclidean","manhattan"]}



classifier_param = [dt_param_grid,

                   svc_param_grid,

                   rf_param_grid,

                   logreg_param_grid,

                   knn_param_grid ]

                     
cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)

    #n_jobs = -1 hızlı işler ve verbose = 1 tüm adımları göstermeye yarar

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

#Amaç survived edip etmedğini anlamaktı 

#örneğin alex diye bir yolcu var ve RandomForest da alex 1 yani yaşıyor

                                    #DecisionTree de 0 ve LogisticRegression da 0 yani ölü

    #Eğer voting = Hard dersek 2 tane 0 var ve 1 tane 1 var yani 2>1 den Alex ölü deriz 

    #Ama voting = soft dersek olasılıkları değerlendirip daha doğal bir bir sonuc verir yine ölü cıkabilir ancak daha mantıklı olacaktır 



votingC = votingC.fit(X_train,y_train)

print(accuracy_score(votingC.predict(X_test),y_test)) #accuracy_score bir methoddur 

test_survived = pd.Series(votingC.predict(test),name = "Survived").astype(int)

results = pd.concat([test_PassengerId,test_survived],axis = 1)

results.to_csv("titanic.csv",index = False)

results 