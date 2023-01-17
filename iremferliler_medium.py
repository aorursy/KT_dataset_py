#Python 3 ortamı, yüklü birçok faydalı analitik kütüphanesiyle birlikte gelir.

import time

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from keras import models

from keras.layers import Dense, Dropout

from sklearn.preprocessing import MinMaxScaler

#Test verisetini pandas kütüphanesi ile tanımlıyoruz.

df_test = pd.read_csv('../input/test.csv')

#Eğitim verisetini pandas kütüphanesi ile tanımlıyoruz.

df = pd.read_csv('../input/train.csv')
#Sütunları kontrol ediyoruz.

df.columns
#İki veri setindeki toplam yolcu sayısı

df_test.shape[0] + df.shape[0]

#Hayatta kalma oranı

df['Survived'].mean()
#Eğitim setindeki ilk 3 satır

df.head(3)

#eğitim setinde eksik değerleri olan sütunları listele

df.columns[df.isna().any()]

#test setinde eksik değerleri olan sütunları listele

df_test.columns[df_test.isna().any()]
def get_title(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'Unknown'

 

def replace_titles(x):

    title = x['Title']

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title
def prepare_data(df):

    

    # Eksik değerleri en çok kullanılan ile doldur.

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Eksik yaş değerlerini veri setinin yaş ortalaması ile

    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Eksik ücret değerlerini veri setinin ücret ortalaması ile

    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    

    # Age, SibSp, Parch, Fare verilerini 0 ile 1 arasında ölçeklendir.

    scaler = MinMaxScaler()

    df[['Age','SibSp','Parch','Fare']] = scaler.fit_transform(df[['Age','SibSp','Parch','Fare']])

    

    # Cinsiyetleri 0 ve 1 olarak değiştir.

    df['Sex'] = df['Sex'].map({'female':0,'male':1}).astype(int)

    

    # Class sütununa One hot encoding uygula 

    df_class = pd.get_dummies(df['Pclass'],prefix='Class')

    df[df_class.columns] = df_class

    

    # Biniş limanına One hot encoding uygula.

    df_emb = pd.get_dummies(df['Embarked'],prefix='Emb')

    df[df_emb.columns] = df_emb

    

    # Ad sütunundan başlıkları çıkar ve yeni sütunu doldur.

    df['Title'] = df['Name'].map(lambda x: get_title(x))

    # Replace titles with Mr, Mrs or Miss

    df['Title'] = df.apply(replace_titles, axis=1)

    # Dönüştüren başlıklar için One hot encoding uygula.

    df_title = pd.get_dummies(df['Title'],prefix='Title')

    df[df_title.columns] = df_title

    

    return
#Eğitim veri setini çalıştırın ve sütun adlarını kontrol edin:

prepare_data(df)

 

df.columns
#Modelde kullanılabilir sütunlar

columns = ['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Class_1', 'Class_2',

       'Class_3', 'Emb_C', 'Emb_Q', 'Emb_S', 'Title_Master',

       'Title_Miss', 'Title_Mr', 'Title_Mrs']
# Seçilen sütunlar ile oluşturduğum giriş

X = np.array(df[columns])

# Hayatta kalan sütun

y = np.array(df['Survived'])
network = models.Sequential()

network.add(Dense(32, activation='relu', ))

network.add(Dropout(rate=0.2))

network.add(Dense(16, activation='relu'))

network.add(Dropout(rate=0.2))

network.add(Dense(5, activation='relu'))

network.add(Dropout(rate=0.1))

network.add(Dense(1, activation='sigmoid'))
# Modeli derle

network.compile(optimizer='adam',

                loss='binary_crossentropy',

                metrics=['accuracy'])

 

# Geçmiş ölçümleri eğit ve kaydet.

history = network.fit(X, y, epochs=50, batch_size=10, verbose=0, validation_split=0.33)

 

# Test doğruluğu ve periyot grafiğini çiz

plt.plot(history.history['acc'], label = 'eğitim')

plt.plot(history.history['val_acc'], label = 'test')

plt.title('Model Doğruluğu')

plt.ylabel('Doğruluk')

plt.xlabel('Epoch')

plt.legend(loc='lower right')

plt.show()
prepare_data(df_test)

 

X_pred = np.array(df_test[columns])

y_pred = network.predict(X_pred)

 

y_pred = y_pred.reshape(418)

 

# Yolcu kimliğini ve hayatta kalma tahminini birleştir

df_subm = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':y_pred})

 

# İlk 5 satırı listele

df_subm.head()
#Binary(0 veya 1) dönüştür.

def binary(x):

#Eğer 0,5'in tahmin üzerinde ise 1 olarak kaydet.Değilse 0 olarak kaydet.

    if x > 0.5:

        return 1

    else:

        return 0

    

# Survived sütunundaki tüm değerlere aynısını uygula.

df_subm['Survived'] = df_subm['Survived'].apply(binary)

df_subm.head(5)
time_string = time.strftime("%Y%m%d-%H%M%S")

 

# Dosya adını belirle

filename = 'titanic_submission_irem' '.csv'

 

# Csv olarak kaydet

df_subm.to_csv(filename,index=False)

 

print('Saved file: ' + filename)