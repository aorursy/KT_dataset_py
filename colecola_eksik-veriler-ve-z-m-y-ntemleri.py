#import pandas as pd



import pandas as pd

# Veri setini Notebook'a yüklüyoruz

melb_data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



#Price verileri içerisinde eksik olan sütünları dikkate almıyoruz.

melb_target = melb_data.Price

melb_predictors = melb_data.drop(['Price'], axis = 1)



#Örnek verileri korumak uğruna, sadece numerik verileri kullanıyoruz.

melb_numeric_predictors = melb_predictors.select_dtypes(exclude = ['object'])

#Sci-Kit Kütüphanesinde var olan RandomForestRedressor kütüphanesini dahil ediyoruz.

from sklearn.ensemble import RandomForestRegressor

#Sci-Kit Kütüphanesinde mevcut olan metrics kütüphanesindeki MEA fonksiyorunu dahil ediyoruz

from sklearn.metrics import mean_absolute_error

#Scikit-Leanr Kütüphanesinden kullanacağımız modeli dahil ediyoruz.(train_test_split) 

from sklearn.model_selection import train_test_split



#Daha önce tanımlamış olduğumuz, train ve test verileri için X, y değerlerini tanımlıyoruz

X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors,

                                                       melb_target,

                                                       train_size = 0.7,

                                                       test_size = 0.3,

                                                       random_state = 0)

#Verisetin

def score_dataset(X_train, X_test, y_train, y_test):

        model = RandomForestRegressor()

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        return mean_absolute_error(y_test, preds)

cols_with_missing = [col for col in X_train.columns

                                if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(cols_with_missing, axis = 1)

reduced_X_test  = X_test.drop(cols_with_missing, axis = 1)

print("Eksik verilerin olduğu kolonların hesapdışı tutulması sonucu oluşan Ortalama Mutlak Hata model puanı")

print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))
#Skikit-Learn kütüphanesinde bulunan Imputer(atfetme) fonksiyonunu çağırıyoruz

from sklearn.preprocessing import Imputer

my_imputer = Imputer()

imputed_X_train = my_imputer.fit_transform(X_train)

imputed_X_test = my_imputer.transform(X_test)

print("Ortalama Mutlak Hata puanı (Imputation) Ihtam( Boş verilere veriler atanması sonucu):")

print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))
#X_train ve X_test değerleri kopyalanıyor ve yeni bir X_train set oluşturuluyor

imputed_X_train_plus = X_train.copy()

imputed_X_test_plus  = X_test.copy()



cols_with_missing = (col for col in X_train.columns

                                 if X_train[col].isnull().any())

for col in cols_with_missing:

    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()

    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()



    #imputation

my_imputer = Imputer()

imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)

imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)



print("İlave kolonlardan(boş verilerin yerine eklenen) sonra oluşan Ortalama Mutlak Hata puanı(MHO-MAE):")

print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))
