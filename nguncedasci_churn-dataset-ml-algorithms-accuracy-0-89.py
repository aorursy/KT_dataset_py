# Gerekli kütüphaneleri import ettim.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# veriyi import ettim
churn = pd.read_csv("../input/churn-predictions-personal/Churn_Predictions.csv")
df=churn.copy()
df.head()
df.info()
df.describe().T
# Row number değişkenini attım, indexi yeniden düzenledim.
df=df.drop("RowNumber", axis=1)
df=df.reset_index(drop=True)

# Gender ve Geography kategorik değişkenlerine one hot encoding uyguladım.
df=pd.get_dummies(df,columns=["Geography","Gender"], drop_first=True)
df.head()

df1=df.copy()

# Age değişkeninde 18-30=1 , 30-40=2 ,40-50=3, 50-60=4, 60-92=5 olarak atadım
df1.loc[(df1["Age"]>=18) & (df1["Age"]<=30), "Age"]=1
df1.loc[(df1["Age"]>30) & (df1["Age"]<=40), "Age"]=2
df1.loc[(df1["Age"]>40) & (df1["Age"]<=50), "Age"]=3
df1.loc[(df1["Age"]>50) & (df1["Age"]<=60), "Age"]=4
df1.loc[(df1["Age"]>60) & (df1["Age"]<=92), "Age"]=5



#kredi_skor_tablosu
def kredi_skor_tablosu(row):
    
    kredi_skoru = row.CreditScore
    if kredi_skoru >= 300 and kredi_skoru < 500:
        return (2)
    elif kredi_skoru >= 500 and kredi_skoru < 601:
        return (3)
    elif kredi_skoru >= 601 and kredi_skoru < 661:
        return (4)
    elif kredi_skoru >= 661 and kredi_skoru < 781:
        return (5)
    elif kredi_skoru >= 851:
        return (7)
    elif kredi_skoru >= 781 and kredi_skoru < 851:
        return (6)
    elif kredi_skoru < 300:
        return (1)
    
df1 = df1.assign(credit_score_table=df1.apply(lambda x: kredi_skor_tablosu(x), axis=1))



# emeklilik ile ilgili yeni bir değişken oluşturdum.( Alm, İsp =65 , İtalya=66)
# retired
df1["retired"]=df["Age"]

df1.loc[(df1["retired"]>=65) & (df1["Geography_Germany"]==1), "retired"]=1
df1.loc[(df1["retired"]>=65) & (df1["Geography_Spain"]==1), "retired"]=1
df1.loc[(df1["retired"]>=66) & (df["Geography_Spain"]==0) & (df["Geography_Germany"]==0), "retired"]=1


df1.loc[(df1["retired"]<65) & (df1["Geography_Germany"]==1), "retired"]=0
df1.loc[(df1["retired"]<65) & (df1["Geography_Spain"]==1), "retired"]=0
df1.loc[(df1["retired"]<66) & (df["Geography_Spain"]==0) & (df["Geography_Germany"]==0), "retired"]=0


# Tenure/NumOfProducts
df1["Tenure/NumOfProducts"]=df1["Tenure"]/df1["NumOfProducts"]


# 405 değerinin altındakilerin hepsi churn olmuş(20 değer), outlier gibi kenarda kalmışlar atmadım yeni değişken oluşturdum
#smallerthan405
df1["smallerthan405"]=df["CreditScore"]

df1.loc[(df1["smallerthan405"]<405), "smallerthan405"]=1
df1.loc[(df1["smallerthan405"]>405), "smallerthan405"]=0


# NOP* isminde değişken oluşturdum. Bu değişkeni, number of products'ın her bir ürün bazındaki exit durumuna göre sıraladım.
# Mevcut number of products'ı incelediğimde: NOP=1,mean=0.27    NOP=2,mean=0.07    NOP=3,mean=0.82    NOP=4,mean=1
df1["NOP*"]=df["NumOfProducts"]
df1.loc[(df1["NOP*"]==2), "NOP*"]=1
df1.loc[(df1["NOP*"]==1), "NOP*"]=2
df1.loc[(df1["NOP*"]>2), "NOP*"]=3

#Balance'ı 0 olanların hiç exit olmadığını gözlemledim. Bu nedenle yeni değişken ile Balance'ı 0 ve 0 olmayanlar şeklinde ayırdım.
#Balance0
df1["Balance0"]=df1["Balance"]
df1.loc[(df1["Balance0"]==0), "Balance0"]=0
df1.loc[(df1["Balance0"]!=0), "Balance0"]=1


# Tahmin edilen maaşın yaşla oranı olabilir diye düşündüm.18 yaşına kadar para kazanmamışlardır dedim.
# Fakat veri setinde en küçük yaş 18 olduğu için Age-17'ye böldüm.
# Estimated Salary/Age
df1["ES/Age"]=df1["EstimatedSalary"]/(df["Age"]-17)


# Tenure/Age
df1["Tenure/Age"]=df1["Tenure"] / (df["Age"]-17)

# Balance/ ES
df1["Balance/ES"]=df1["Balance"] / df1["EstimatedSalary"]

#Tahmin edilen maaşı aylığa dönüştürdüm. Amacım vergileri de çıkarıp aylık yalın maaş bulmaktı ancak maaş vergileri için 
#net rakamlar yerine aralıklar bulduğum için uygulamaya geçiremedim.
#Estimated Salary (monthly)
df1["EstimatedSalary"]=df1["EstimatedSalary"]/12

# Tenure'de 0 olan değerler vardı, bu nedenle inf gelmemesi için 1 eklenmiş haline böldüm.
# ES/Tenure 
df1["ES/Tenure"]=df1["EstimatedSalary"]/(df1["Tenure"]+1)

# ES/Score
df1["ES/Score"]=df1["EstimatedSalary"]/df1["credit_score_table"]     

# DROP FEATURE 
#Kredi skor sıralamasını anlatan bir değişken oluşturduğum için asıl değişkeni veri setinden çıkardım.
df1=df1.drop(["CreditScore"], axis=1)
df1=df1.drop(["Tenure"], axis=1)
df1=df1.drop(["Balance"], axis=1)
df1.head(3)
#Robust Scaler uygulayacağım değişkenleri seçtim.
df1_num=df1[["Age","NumOfProducts","EstimatedSalary", 
             "credit_score_table","Tenure/NumOfProducts","NOP*","ES/Age",
             "Tenure/Age","Balance/ES","ES/Tenure","ES/Score"]]

# Scaling işlemini uyguladığım veri setine x_transformed adını verdim.
col=df1_num.columns
x_transformed=pd.DataFrame(RobustScaler().fit(df1_num).transform(df1_num), columns=col)
x_transformed.head()
# Scale işlemini yaptığım ve yapmadığım değişkenlerle -churn veri setindeki değişken sırasını da dikkate alarak-
# yeni bir dataframe oluşturdum. Exited isimli y değişkenini ise koymadım. Böylelikle bağımsız değişkenleri bir
# dataframede toplamış oldum. X ismini koydum.
X=pd.concat([x_transformed.loc[:,"Age":"NumOfProducts"],df1.loc[:,"HasCrCard":"IsActiveMember"],
             x_transformed.loc[:,"EstimatedSalary"], df1.loc[:,"Geography_Germany":"Gender_Male"],
             x_transformed.loc[:, "credit_score_table"], df1.loc[:,"retired"],
             x_transformed.loc[:,"Tenure/NumOfProducts"],df1.loc[:,"smallerthan405"],
             x_transformed.loc[:,"NOP*"],df1.loc[:,"Balance0"],
             x_transformed.loc[:, "ES/Age":"ES/Score"]], axis=1)
X.head(2)
# X' daha önceden tanımlamıştım, şimdi ise y'yi tanımladım. 
y=df1["Exited"]

#split işlemi
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=12345)
# rf model kurulumu
rf_model = RandomForestClassifier().fit(X_train,y_train)
y_pred = rf_model.predict(X_test)

# validasyon hatası, accuracy skoru, confusion matrix
cv_results = cross_val_score(rf_model, X_train, y_train, cv = 10, scoring= "accuracy")

print(cv_results.mean())
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Feature importance'a baktım. Retired önemsiz görünüyor, çıkararak tekrar hataları ölçtüm, 
#değişen bir şey olmadı. Diğer modellerde önemli olabilir diye bıraktım.

importance=rf_model.feature_importances_
plt.figure(figsize=(8,8))
plt.barh(X.columns,importance)
plt.show()
# gbm model kurulumu
gbm_model=GradientBoostingClassifier().fit(X_train,y_train)
y_pred = gbm_model.predict(X_test)

# validasyon hatası, accuracy skoru, confusion matrix
cv_results = cross_val_score(gbm_model, X_train, y_train, cv = 10, scoring= "accuracy")

print(cv_results.mean())
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# lgbm model kurulumu
lgbm_model=LGBMClassifier().fit(X_train,y_train)
y_pred = lgbm_model.predict(X_test)

# validasyon hatası, accuracy skoru, confusion matrix
cv_results = cross_val_score(lgbm_model, X_train, y_train, cv = 10, scoring= "accuracy")

print(cv_results.mean())
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
smt = SMOTE(random_state=12345)
X_res, y_res = smt.fit_sample(X, y)

print('Resampled dataset shape {}'.format(Counter(y_res)))
#split işlemi
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, 
                                                    test_size=0.20, 
                                                    random_state=12345)

# lgbm model kurulumu
lgbm_model=LGBMClassifier(random_state=12345).fit(X_train,y_train)
y_pred = lgbm_model.predict(X_test)

# validasyon hatası, accuracy skoru, confusion matrix
cv_results = cross_val_score(lgbm_model, X_train,y_train, cv = 10, scoring= "accuracy")

print("cross_val_score(train):", cv_results.mean())

cv_results = cross_val_score(lgbm_model, X_test,y_test, cv = 10, scoring= "accuracy")
print("cross_val_score(test):", cv_results.mean())


y_train_pred = lgbm_model.predict(X_train)
print("accuracy_score(train):",accuracy_score(y_train, y_train_pred))
print("accuracy_score(test):",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues');