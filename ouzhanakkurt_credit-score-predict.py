import pandas as pd

import numpy as np

import seaborn as sns



Credit = pd.read_csv("../input/german-credit-data-with-risk/german_credit_data.csv")

df = Credit.copy()

df.drop(["Unnamed: 0"],axis=1,inplace=True)

df.head(3)
df.info()
df.shape
df.corr()
# Looking Corr table



from matplotlib import pyplot as plt



df1 = df.copy()

df1 = df1.corr()

plt.figure(figsize=(8.5,5.5))

corr = sns.heatmap(df1,xticklabels=df1.columns,yticklabels=df1.columns,annot=True)
df.describe().T
df.head(2)
from IPython.display import Image

import os

!ls ../input/



Image("../input/files-png/Screenshot_1.png")
# EN

# Creat a new variable beacuse scaling is good for forest methods





df.insert(1,"Cat Age",df["Age"])

for i in df["Cat Age"]:

    if i<25:

        df["Cat Age"]=df["Cat Age"].replace(i,"0-25")

    elif (i>=25) and (i<40):

        df["Cat Age"]=df["Cat Age"].replace(i,"25-30")

    elif (i>=40) and (i<45):

        df["Cat Age"]=df["Cat Age"].replace(i,"30-35")

    elif (i>=45) and (i<40):

        df["Cat Age"]=df["Cat Age"].replace(i,"35-40")

    elif (i>=40) and (i<50):

        df["Cat Age"]=df["Cat Age"].replace(i,"40-50")

    elif (i>=50) and (i<76):

        df["Cat Age"]=df["Cat Age"].replace(i,"50-75")
df.head(2)
Image("../input/files-png/Screenshot_2.png")
Image("../input/files-png/Screenshot_3.png")
from scipy import stats



df1 = df.copy()



df1 = df1[["Credit amount","Purpose"]]



group = pd.unique(df1.Purpose.values)



d_v1 = {grp:df1["Credit amount"][df1.Purpose == grp] for grp in group}



stats.levene(d_v1['radio/TV'],d_v1['furniture/equipment'],d_v1['car'],d_v1['business'],d_v1['domestic appliances'],d_v1['repairs'],

                     d_v1['vacation/others'],d_v1['education'])



# EN

# Is there a difference between the amount of credit the Purpose variable demands?





# TR

# Varyanslar homojen degil bu durumda KRUSKALL testi tercih edilebilir ama biz normal parametrik olani kullanacagiz
f, p = stats.f_oneway(d_v1['radio/TV'],d_v1['furniture/equipment'],d_v1['car'],d_v1['business'],d_v1['domestic appliances'],d_v1['repairs'],

                     d_v1['vacation/others'],d_v1['education'])



("F istatistik: "+str(f)+" | P value degeri: "+str(p))



# EN

# Apply Oneway anova



# TR

# Sonuca bakinca siniflar arasinda Cekilen kredi miktarinda anlamli bir farklilik gozukmekte
(df.groupby(by=["Purpose"])[["Credit amount"]].agg("sum") / df["Credit amount"].sum())*100



# EN

# In the result, there is different between groups.

# In this query we can see difference





# TR

# Bu esitsizliginde hangi gruplar tarafindan olusturuldugunu gorebilmekteyiz

# Car & furniture/equipment & radio/TV en fazla kredi verilen siniflar olmus

# Onceki islemlere donup baktigimiz zaman vacation/others sinifi BAD degeri en yuksekti ve burada da verilen kredi dusuk gozukuyor
table = pd.crosstab(df["Sex"],df["Risk"])



table



# EN

# if there are two categorical variable, use to Chi2 test.



# TR

# Bazi Iki kategorik grubu ve bu siniflardan birinde olma durumuna gore bad yada good olma arasinda bir iliski var mi karsilastiralim
observed_values = table.values

print("Observe Values: -\n",observed_values)
from scipy.stats import chi2_contingency



chi2, p, dof, ex = chi2_contingency(table)

print("Kikare degeri {} ve P value {}".format(chi2,p))



# EN

# P value < 0.05 H0 rejected

# There is a depend between Risk and Sex 





# TR

# H0: Iki degisken arasinda bir iliski yoktur

# H1: Iki degisken arasinda iliski vardır



# Sonuca bakinca H0 red edilmistir. Yani iki degisken arasinda bir iliski vardir.

# Kadin yada erkek olma durumuna gore Bad olma ve Good olma durumu degiskenlik gostermektedir
# Bir diger merak ettigim durum ise Ev durumuna gore acaba risk gruplari arasinda anlamli bir farklilik var midir?



housing_risk = pd.crosstab(df["Housing"],df["Risk"])



housing_risk
observed_values1 = housing_risk.values

print("Observe Values: -\n",observed_values1)
chi2, p, dof, ex = chi2_contingency(housing_risk)

print("Kikare degeri {} ve P value {}".format(chi2,p))



# EN

# P value < 0.05 H0 rejected

# There is a depend between Risk and Housing 





# H0: Iki degisken arasinda bir iliski yoktur

# H1: Iki degisken arasinda iliski vardır



# Gozle gorulebilir bir farklilik oldugu belliydi.

# Lakin bu formel degildi ve test sonucu bunun formelligi de kesinlesmis bulunmakta.

# Ev sahibi durumuna gore krediyi odeyememe(bad) veya krediyi odeme(good) durumu arasinda anlamli bir iliski vardir
# EN 

# I am creating new variable

# This variable is Monthly pay



# TR

# Simdi var olan data setinden bir degisken turetecegim

# Bu degisken ise Cekilen kredi miktarinin odenecek aya bolumu

# Yani kisinin aylik odemesi



df["monthly pay"] = (df["Credit amount"] / df["Duration"]) 
df.head(3)
df[df["Sex"] == "female"]["monthly pay"].agg(["min","max"])



# EN

# Min and Max difference is huge

# If we will use the MEAN in this case do not use

# instead we should use MEDIAN 





# TR

# Aylik odeme miktarinda median kullanmak daha saglikli
Image("../input/another/Screenshot_5.png")
# Coralation between variables



# Simdi korelasyon degerlerine bakacagim 



df.corr()
# EN 

# I am creating new variable again but this variable job is only help raise prediction





# TR

# Ardindan toplam cekilen kredi miktarinin karesini olarak modele ekleyecegim

# Bunu yapmamdaki sebep ise modelde ekstra bir degisken daha tutarak her hangi bir aciklayicilik artacak mi?

# Ayni zamanda modele predict(tahmin) acisindan bir katkisi olacak mi gormek istiyorum



df["Credit amount**2"] = df["Credit amount"]**2

df.head(2)
df.corr()
df.head()
# EN

# Scaling Duration 





# TR

# Degisken donusumlerini hepsini gerceklestirip hangi durumda modeller nasil sonuclar veriyor onlari gormek icin son olarak,

# Duration kismini da olceklendiriyorum sebebi ise agaclandirma yapilarinda kategorik olanlar daha iyi sonuc verebilmekte

# Ayni zamanda degisken Vadeyi belirtiyor buda ayni zaman da kesikli oldugunu ve kategorik bir yapi sergiledigini hissetiriyor



df.insert(9,"Cat Duration",df["Duration"])

for i in df["Cat Duration"]:

    if i<12:

        df["Cat Duration"]=df["Cat Duration"].replace(i,"0-12")

    elif (i>=12) and (i<24):

        df["Cat Duration"]=df["Cat Duration"].replace(i,"12-24")

    elif (i>=24) and (i<36):

        df["Cat Duration"]=df["Cat Duration"].replace(i,"24-36")

    elif (i>=36) and (i<48):

        df["Cat Duration"]=df["Cat Duration"].replace(i,"36-48")

    elif (i>=48) and (i<60):

        df["Cat Duration"]=df["Cat Duration"].replace(i,"48-60")

    elif (i>=60) and (i<=72):

        df["Cat Duration"]=df["Cat Duration"].replace(i,"60-72")
df.head(2)
# EN

# I scaled Age and Duration now drop that variables it.





# TR

# Age(Yas) degiskenini ve Duration degiskenini cikariyorum veri setinden 

# Ayni zamanda kategorik olanlari Dummy hale getirecegim



df.drop(labels=["Age","Duration"],inplace=True,axis=1)

df.head(2)
df.nunique()
# EN

# Preparation



# TR

# 2 den fazla olanlara Get dummy yaptim, cunku diger turlu bunlarin arasinda bir siralama olacagini dusunerek sayisi yuksek olana agirlik verecekti



df = pd.get_dummies(df,columns= ["Cat Age"], prefix= ["Cat Age"]) 

df = pd.get_dummies(df,columns= ["Job"], prefix= ["Job"])

df = pd.get_dummies(df,columns= ["Housing"], prefix= ["Housing"])

df = pd.get_dummies(df,columns= ["Saving accounts"], prefix= ["Saving accounts"])

df = pd.get_dummies(df,columns= ["Cat Duration"], prefix= ["Cat Duration"])

df = pd.get_dummies(df,columns= ["Checking account"], prefix= ["Checking account"])

df = pd.get_dummies(df,columns= ["Purpose"], prefix= ["Purpose"])

df = pd.get_dummies(df,columns= ["Sex"], prefix= ["Sex"])
# Geriye kalan Sex ve Risk degiskenine LabelEncoder yapacagim 2 sinif olmasindan sebep



from sklearn.preprocessing import LabelEncoder



lbe = LabelEncoder()





df["Risk"] = lbe.fit_transform(df["Risk"])



df.head(2)
# Already we have Sex column, drop other



df.drop(labels=["Sex_female"],axis=1,inplace=True)









# Simdi DEGISKENLERDEKI NaN degerleri dusurmustuk. Bu sekilde model kuracagim ve daha sonra impute islemi gerceklestirip sonuclari degerlendirecegim.



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler  

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split



y = df["Risk"]

X = df.drop(['Risk'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                   test_size=0.30, 

                                                   random_state=42)
int_float_train = list(X_train.select_dtypes(include=["float64","int64"]))

int_float_train
int_float_test = list(X_test.select_dtypes(include=["float64","int64"]))

int_float_test
scaler = StandardScaler()



X_train_scaled = X_train.copy()

X_test_scaled = X_test.copy()



X_train_scaled.loc[:,int_float_train] = scaler.fit_transform(X_train_scaled.loc[:,int_float_train])

X_test_scaled.loc[:,int_float_test] = scaler.fit_transform(X_test_scaled.loc[:,int_float_test])
X_train_scaled.head()
tree_model=DecisionTreeClassifier().fit(X_train,y_train)

randomforest_model=RandomForestClassifier().fit(X_train,y_train)

ysa_model= MLPClassifier().fit(X_train_scaled, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



print("Karar agacinin Train seti Accuracy Scoru : ",accuracy_score(y_train,tree_model.predict(X_train)))

print("Karar agacinin Test seti Accuracy Scoru : ",accuracy_score(y_test,tree_model.predict(X_test)))

print("Test setine ait Recall, F1 Score gibi detaylar: \n",classification_report(y_test,tree_model.predict(X_test)))
print("YSA  Train seti Accuracy Scoru : ",accuracy_score(y_train,ysa_model.predict(X_train_scaled)))

print("YSA Test seti Accuracy Scoru : ",accuracy_score(y_test,ysa_model.predict(X_test_scaled)))

print("Test setine ait Recall, F1 Score gibi detaylar: \n",classification_report(y_test,ysa_model.predict(X_test_scaled)))
print("Random Forest Train seti Accuracy Scoru : ",accuracy_score(y_train,randomforest_model.predict(X_train)))

print("Random Forest Test seti Accuracy Scoru : ",accuracy_score(y_test,randomforest_model.predict(X_test)))

print("Test setine ait Recall, F1 Score gibi detaylar: \n",classification_report(y_test,randomforest_model.predict(X_test)))
# EN

# We can see Recall & F1 Score of predict 0 is low

# the reason is Unbalanced data





# TR

# Dikkati ceken durum 0 tahmin durumunun dusuk olmasidir

# Bunun sebebi ise dengesiz(unbalanced) bir veri seti olmasidir



df["Risk"].value_counts()




from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



def YsaClassifier(X_train,y_train,X_test,y_test):

        scaler = StandardScaler()



        X_train_scale = scaler.fit_transform(X_train)



        X_test_scale =  scaler.fit_transform(X_test)

  

  

        mlp_regres = MLPClassifier().fit(X_train_scale,y_train) 

        y_pred = mlp_regres.predict(X_test_scale)

        Accuaracy = accuracy_score(y_test,y_pred)

        matrix = classification_report(y_test,y_pred)

  



        params = {"alpha":[0.1,0.01,0.02,0.005], # alpha icin aranacak degerler

              "hidden_layer_sizes":[(20,20),(100,50,150),(300,200,150)], # gizli katmanin dereceleri ve sayilari icin aranacak parametreler

              "activation":["relu","logistig"],

              'solver': ['adam', 'lbfgs']}# Son olarak birde iki tane fonksiyon var onlari denesin denedik 



        mlp_c = MLPClassifier()

  

        mlp_c = GridSearchCV(mlp_c,params,

                       cv = 10,

                       n_jobs = -1,

                       verbose = 2)



        mlp_c_tune = mlp_c.fit(X_train_scale,y_train)



        bos = []

        for i in mlp_c_tune.best_params_:

            bos.append(mlp_c_tune.best_params_[i])



        mlp_tuned = MLPClassifier(activation=bos[0],

                         alpha=bos[1],hidden_layer_sizes=bos[2],

                         solver=bos[3]).fit(X_train_scale,y_train)



        y_pred1 = mlp_tuned.predict(X_test_scale)



        Accuaracy1 = accuracy_score(y_test,y_pred1)

        matrix1 = classification_report(y_test,y_pred1)





        print("Tune Edilmemis Tahmin sonuclari Accuracy degeri: ",Accuaracy)

        print("-------------------------------")

        print("Tune Edilmemis Recall,F1 etc  sonuc: \n",matrix)

        print("****************************************************")

        print("Tune sonrasi Tahmin sonuclari Accuracy degeri:",Accuaracy1)

        print("-------------------------------")

        print("Tune sonrasi Recall,F1 etc sonuc: \n",matrix1)





YsaClassifier(X_train,y_train,X_test,y_test)
def DecisionClassifier(X_train,y_train,X_test,y_test):

        cart = DecisionTreeClassifier()

        cart_model = cart.fit(X_train, y_train)

        y_pred = cart_model.predict(X_test)

        Accuaracy = accuracy_score(y_test, y_pred)

        matrix = classification_report(y_test,y_pred)



        cart_grid = {"max_depth": range(1,10),

            "min_samples_split" : list(range(2,50)),

             "criterion":["gini","entropy"],

             "min_samples_leaf": range(50,100)}



        cart = DecisionTreeClassifier()

        cart_cv = GridSearchCV(cart, cart_grid, cv = 10, n_jobs = -1, verbose = 2)

        cart_cv_model = cart_cv.fit(X_train, y_train)



        bos = []



        for i in cart_cv_model.best_params_:

              bos.append(cart_cv_model.best_params_[i])



        cart = DecisionTreeClassifier(criterion = bos[0],max_depth = bos[1],min_samples_leaf=bos[2],min_samples_split = bos[3])

        cart_tuned = cart.fit(X_train, y_train)



        y_pred = cart_tuned.predict(X_test)

        Accuaracy1 = accuracy_score(y_test, y_pred)

        matrix1 = classification_report(y_test,y_pred)



        print("En iyi parametre degerleri: ",cart_cv_model.best_params_)

        print("Tune edilmemis modelin Accuaracy ve Detaylar: ")

        print(Accuaracy)

        print(matrix)

        print("**********************************************")

        print("Tune sonrasi Accuaracy ve Detaylar: ")

        print(Accuaracy1)

        print(matrix1) 





DecisionClassifier(X_train,y_train,X_test,y_test)
def RandomForestsClass(X_train,y_train,X_test,y_test):

        rf_model = RandomForestClassifier().fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)

        Accuracy = accuracy_score(y_test, y_pred)  

        Matrix = classification_report(y_test,y_pred)



        params = {"max_depth": [2,5,8,10],

            "max_features": [2,5,8],

            "n_estimators": [10,500,1000],

            "min_samples_split": [7,15,20]}





        rf_model = RandomForestClassifier()



        rf_cv_model = GridSearchCV(rf_model,

                           params,

                           cv = 10,

                           n_jobs = -1,

                           verbose = 2)



        rf_cv_model.fit(X_train,y_train)  



        bos = []



        for i in rf_cv_model.best_params_:



              bos.append(rf_cv_model.best_params_[i])



        final_tune = RandomForestClassifier(max_depth=bos[0],max_features=bos[1],min_samples_split=bos[2],n_estimators=bos[3])



        final_tune = final_tune.fit(X_train,y_train)



        y_pred = final_tune.predict(X_test)



        Accuracy1 = accuracy_score(y_test,y_pred)

        Matrix1 = classification_report(y_test,y_pred)





        Importance = pd.DataFrame({"Importance": final_tune.feature_importances_*100},

                         index = X_train.columns)

        importance = Importance.sort_values(by = "Importance", 

                       axis = 0, 

                       ascending = True).plot(kind ="barh", color = "yellow"),plt.xlabel("Değişken Önem Düzeyleri")



        print("En iyi parametre degerleri: ",rf_cv_model.best_params_)

        print("                                               ")

        print("Tune oncesi Accuracy ve Recall,F1 degerleri: ",Accuracy,"\n",Matrix)

        print("Tune sonrasi Accuracy ve Recall,F1  degerleri: ",Accuracy1,"\n",Matrix1)

        print("                                                ")

        print(importance)





RandomForestsClass(X_train,y_train,X_test,y_test)
X_train.columns



# EN

# After at the Random forest, the best importance factors come to light

# Their names are

# 1-'monthly pay'

# 2-'Credit amount**2'

# 3-'Credit amount'

# 4-'Checking account_little'

# 5-'Cat Duration_48-60'

# 6-'Checking account_moderate'





# TR

# Degisken onem duzeyine baktigimiz da cikan sonucta oncelikli olarak;

# 1-'monthly pay'

# 2-'Credit amount**2'

# 3-'Credit amount'

# 4-'Checking account_little'

# 5-'Cat Duration_48-60'

# 6-'Checking account_moderate'



# durumlarina gore tahmin sonuclarinda en etkili olanlar olarak gorulmekte

# Burada 2 adet degiskeni kendim olusturmustum. "Monthly pay" ve "Credit amount**2" ikincisini tamamiyle tahmin sonucunu artirmak adina kullandim.

# Simdi bu degiskenlerden basit bir karar agaci gorseli gorelim
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier



degisken = ["monthly pay","Credit amount","Checking account_little","Cat Duration_48-60","Checking account_moderate"]



data = df.loc[:,degisken]



data.head(2)





X = data

y = df["Risk"]



forest = RandomForestClassifier(max_depth = 3, n_estimators=5)

forest.fit(X,y)



estimator = forest.estimators_[4]

target_names = ["Result 0","Result 1"]





from sklearn.tree import export_graphviz



export_graphviz(estimator,out_file="tree_limited.dot",feature_names=X.columns,

                class_names=target_names,rounded = True, proportion = False, precision = 2, filled = True)



forest_1 = RandomForestClassifier(max_depth = None, n_estimators=5)

forest_1 = forest_1.fit(X,y)

estimator_non = forest_1.estimators_[4]



export_graphviz(estimator_non, out_file='tree_nonlimited.dot', feature_names = X.columns,

                class_names = target_names,

                rounded = True, proportion = False, precision = 2, filled = True)
!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=600



from IPython.display import Image

Image(filename = 'tree_limited.png')