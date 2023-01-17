# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv(os.path.join(dirname, filename))

data.head()
print(data.shape[0]) #row size

print(data.shape[1]) #column size
print(data.dtypes)

#6 adet categorical sütunumuz olduğunu görebiliriz
#Veri setimizi dataframe'e dönüştürelim

data_df = pd.DataFrame(data)
#gdp'li sütunlarda isim yüzünden hata aldığım için $ sembolünü kaldırıp dollars ile değiştiriyorum

data_df.columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population',

       'suicides/100kpop', 'country-year', 'HDI_for_year',

       'gdp_for_year_dollars', 'gdp_per_capita_dollars', 'generation']

data_df.columns.values
#gdp_for_year_dollars sütunu virgül kullanılarak string olarak kaydedilmiş, bunu numerically olacak şekilde çeviriyorum

data_df['gdp_for_year_dollars'] = data_df['gdp_for_year_dollars'].str.replace(',','').astype(int)
print(data_df.dtypes)
#numerical sütunlar-> year, suicides_no, population, suicides/100k pop, HDI for year, gdp_for_year_dollars, gdp_per_capita_dollars 

#categorical-object sütunlar-> country, sex, age, generation (kümeleme/sınıflandırma)



#country-year zaten varolan iki sütunun birleşimi olduğu için gereksiz, bu yüzden siliyorum.

del data_df['country-year']
data_df.isnull().sum().sort_values(ascending=False)
#İlk önce Correlation kullanarak sütunlar arasındaki ilişkiye bakalım

correlation = data_df.corr()

correlation
import seaborn as sb

sb.heatmap(correlation, 

            xticklabels=correlation.columns,

            yticklabels=correlation.columns,

            cmap='RdBu_r',

            annot=True,

            linewidth=0.9)
#Yukarıdaki tabloyı özet olarak inceleyelim:

#"1" değerleri verinin kendisi olduğu için %100 ilişkili görünüyor, amacımız 1'e yakın değerleri analiz etmek

#HDI_for_year ~ gdp_per_capita_dollars ve population ~ gdp_fpr_year_dollars arasında çok güçlü bir ilişki var.

#Aynı şekilde Population ~ suicides_no arasında da güçlü bir ilişki var; bu demektir ki intihar oranlarının artması popülasyon ile doğru orantılı.

#HDI_for_year çok fazla NaN değere sahip olduğu için sütunu silelim:

del data_df['HDI_for_year']



#HDI_for_year özellikle linear regresyonda bize yardımcı olabilir bu yüzden sütunu silmek yerine NaN değerleri ortalama değer ile dolduralım:

#data_df['HDI_for_year'] = data_df['HDI_for_year'].fillna(data_df['HDI_for_year'].median())

#data_df['HDI_for_year'].isnull().any()
data_df.dtypes.value_counts()
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(16,7))

bar_age = sns.barplot(x = 'sex', y = 'suicides_no', hue = 'age',data = data_df)

#Görselden de anlaşılacağı üzere, number of suicides kadın ve erkeklerde en fazla 35-54 yaş aralığında görülüyor.
#Şimdi de suicides sayılarının generation ile bağlantısına bakalım;

#G.I Generation: 1920 ve öncesi

#Slient: 1921 - 1945

#Boomers: 1946 - 1964

#Generation X: 1965 - 1976

#Millenials: 1977 - 1995

#Generation Z: 1996 - Günümüz



data_df["generation"].value_counts().plot.bar()

plt.title("Generation", size = 20)

plt.ylabel("Frequency");

#Tablodan görüldüğü üzere; en fazla vaka sayısı genellikle X jenerasyonunda gerçekleşiyor. Bir önceki görsel sonucunu doğrular nitelikte.

#Generation X (1965 - 1976) = 35-54 yaş aralığı
#Yıllara göre intihar sayılarına bakalım

import matplotlib.pyplot as plt

import seaborn as sns

plt.hist(data_df['year'], color = 'purple', edgecolor = 'black',bins = 60)

plt.xlabel('year')

plt.ylabel('suicides_no')

plt.show()
#Bu grafikte daha net yıllara göre dağılımı görebiliriz

sns.distplot(data_df['year'])
sns.distplot(data_df['suicides/100kpop'])
#En fazla suicides oranına sahip ilk 10 ülkeye bakalım

#Fakat burada number of suicides yerine suicides/100kpop sütununu kullanmamız ülke çapında popülasyona göre sonuçlar elde etmemizi sağlayacaktır

most_countries = data_df.groupby('country').mean().sort_values(by='suicides/100kpop',ascending=False)['suicides/100kpop']

most_countries.head(10)
data_df.info()
data_df.describe()
#categorical sütunları numerically olarak çevirmeliyiz.

#categorical -> country, sex, age, generation

from sklearn.preprocessing import LabelBinarizer



lb = LabelBinarizer()



for col in ['country', 'sex', 'age', 'generation']:

    data_df[col] = lb.fit_transform(data_df[col])
#for warning

#skaled data for knn

import warnings

warnings.filterwarnings('ignore', module='sklearn')

from sklearn.preprocessing import MinMaxScaler



msc = MinMaxScaler()

data_df = pd.DataFrame(msc.fit_transform(data_df), 

                    columns=data_df.columns)
data_df.head(5)
#seperate columns except age

x_cols = [x for x in data.columns if x != 'age']
# Split the data into two dataframes

X_data = data_df[x_cols]

y_data = data_df['age']
#fit knn model with k = 3 then predict



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3, weights = "uniform")

knn = knn.fit(X_data, y_data)



y_pred = knn.predict(X_data)
# Function to calculate the % of values that were correctly predicted

def accuracy(real, predict):

    return sum(y_data == y_pred) / float(real.shape[0])
print(accuracy(y_data, y_pred))
#weight=uniform, p=1

knn = KNeighborsClassifier(n_neighbors=3,weights='uniform',p=1)

knn = knn.fit(X_data, y_data)

y_pred = knn.predict(X_data)

print(accuracy(y_data, y_pred))
# k should be odd numbers for deciding the solution so;

k_range = list(range(1,20))

odd_k_range = [];

def justOddNumbers(k_range) :

    for i in k_range :

        if i % 2 != 0 :

            odd_k_range.append(i)

    return odd_k_range



justOddNumbers(k_range)

print(odd_k_range)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

import numpy as np

cv_scores = []

k_and_accuracy = {}



for k in odd_k_range:

    knn = KNeighborsClassifier(n_neighbors = k, weights = "uniform", p=1)



    knn = knn.fit(X_data, y_data)

    y_pred = knn.predict(X_data)

    def accuracy(real, predict):

        return sum(y_data == y_pred) / float(real.shape[0])

    accuracy = accuracy(y_data, y_pred)

    print('k = {}, accuracy = {}' .format(k, accuracy))

    

#Görüldüğü gibi k=1 =~ 1 değerine yakın bir sonuç alıyoruz. Bunun nedeni KNN modeli bizim datamız aslında.
#Cross Validation'dan sonra her bir k değerimiz için accuracy sonuçları:



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

import numpy as np

cv_scores = []

k_and_accuracy = {}



for k in odd_k_range:

    knn = KNeighborsClassifier(n_neighbors = k, weights = "uniform", p=1)



    cv_scores = cross_val_score(knn, X_data, y_data, cv=10, scoring='accuracy')

    cv_mean_scores = round(cv_scores.mean(),3)

 

    print('k = {}, accuracy = {}' .format(k, cv_mean_scores))

#En iyi sonuç;

#k=19 -> accuracy 0.832