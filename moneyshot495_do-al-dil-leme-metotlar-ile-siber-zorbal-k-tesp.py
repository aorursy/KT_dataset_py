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
df = pd.read_csv("/kaggle/input/siber-zorbalk/tweetset.csv",encoding="windows-1254")
df.head()
# Veri setinde kayıp verilerin olup olmadığına bakıyoruz ve düzeltilemeyecek kadar olan feature'leri siliyoruz
print("Kayıp Veriler :{}".format(df.isnull().sum()))

df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4","Unnamed: 5"],axis=1,inplace=True)
#Label encoder işlemi yaparak veri seti içerisinde bulunan "Negatif" değerli 0 "Pozitif" değerleri ise 1 yapıyoruz.
df["sınıf"] = [0 if (i=="Negatif") else 1 for i in df["Tip"]]
df.head()
import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import PorterStemmer
import re
from nltk.stem import WordNetLemmatizer
from snowballstemmer import TurkishStemmer
import nltk
punctation = string.punctuation
#punctuation ='''!()-[]{};':'"\,<>./?@#$%^&*_~'''
#Özel karakterleri temizleme
def ozelkarakter_temizleme (metin):
    return metin.translate(str.maketrans("","",punctation))

#--------------------
#Stopword temizleme

#stopword = set(stopwords.words("turkish"))
stopword = "acaba, ama, ancak, artık, asla, aslında, az,bana, bazen, bazı, bazıları, bazısı, belki, ben, beni, benim, beş, bile, bir, birçoğu, birçok, birçokları, biri, birisi, birkaç, birkaçı, birşey, birşeyi, biz, bize, bizi, bizim, böyle, böylece, bu, buna, bunda, bundan, bunu, bunun, burada, bütün,çoğu, çoğuna, çoğunu, çok, çünkü,da, daha, de, değil, demek, diğer, diğeri, diğerleri, diye, dolayı,elbette, en,fakat, falan, felan, filan, gene, gibi,hangi, hangisi, hani, hatta, hem, henüz, hep, hepsi, hepsine, hepsini, her, her biri, herkes, herkese, herkesi, hiç, hiç kimse, hiçbiri, hiçbirine, hiçbirini,için, içinde, ile, ise, işte,kaç, kadar, kendi, kendine, kendini, ki, kim, kime, kimi, kimin, kimisi,madem, mı, mi, mu, mü,nasıl, ne, ne kadar, ne zaman, neden, nedir, nerde, nerede, nereden, nereye, nesi, neyse, niçin, niye,ona, ondan, onlar, onlara, onlardan, onların, onu, onun, orada, oysa, oysaki,öbürü, ön, önce, ötürü, öyle, sana, sen, senden, seni, senin, siz, sizden, size, sizi, sizin, son, sonra, seobilog,şayet, şey, şimdi, şöyle, şu, şuna, şunda, şundan, şunlar, şunu, şunun,tabi, tamam, tüm, tümü, üzere,var, ve, veya, veyahut,ya, ya da, yani, yerine, yine, yoksa,zaten, zira"

def stopwords_temizleme (metin):
    return " ".join([kelime for kelime in str(metin).split() if kelime not in stopword])

#-------------------
count = Counter()

#sık kullanılan kelimeleri temizleme
for metin in df["Paylaşım"].values:
    for kelime in metin.split():
        count[kelime] += 1
count.most_common(10) # en sık tekrar eden 10 kelimeyi gösterir
frekans = set([i for (i,j) in count.most_common(15)])
nadir = 15
nadir_kelime = set([i for (i,j) in count.most_common()[:-nadir-1:-1]])
def frekans_sil(metin):
    return " ".join([kelime for kelime in str(metin).split() if kelime not in frekans])


#----------------Kelime Kökünü Alma
#lemma = WordNetLemmatizer("turkish")
#Lemmatizer

#def kelime_kök_alma (metin):
#    return " ".join([lemma.lemmatize(kelime) for kelime in metin.split()])
 
    
snowBallStememr = TurkishStemmer()
def kelime_kök_alma(metin):
    wordlist = nltk.word_tokenize(metin)
    stemWords = [snowBallStememr.stemWord(kelime) for kelime in wordlist]
    return " ".join(stemWords)
    
    
    
#---------
#Emojileri Silme

def emoji_silme (metin):
    emoji = re.compile("["
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"                                 
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002500-\U00002BEF"                                 
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji.sub(r"",metin)

df["Paylaşım"] = df["Paylaşım"].str.lower()
df["ozel_karaktersiz"] = df["Paylaşım"].apply(lambda metin : ozelkarakter_temizleme(metin))
df["stop_word"] = df["ozel_karaktersiz"].apply(lambda metin : stopwords_temizleme(metin) )
df["sık_kullanılan"] = df["stop_word"].apply(lambda metin : frekans_sil(metin) )
df["kelime_kok"] = df["sık_kullanılan"].apply(lambda kelime : kelime_kök_alma(kelime))
df["emojisiz"] = df["kelime_kok"].apply(lambda metin : emoji_silme(metin))
df.head(50)
# Veri seti üzerinde gerçekleşen değişimleri gözlemleyebilmek için her bir işlemi farklı feature'ler oluşturarak yapmıştık.
# Şimdi ise işimize yaramayacak olan feature'leri siliyoruz 
df.drop(["Paylaşım","ozel_karaktersiz","stop_word","sık_kullanılan","kelime_kok"],axis=1,inplace=True)
df.head()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (9,9))
sorted_counts = df['sınıf'].value_counts()
plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90, counterclock = False, wedgeprops = {'width' : 0.6},
       autopct='%1.1f%%', pctdistance = 0.7, textprops = {'color': 'black', 'fontsize' : 15}, shadow = True,
        colors = sns.color_palette("Paired")[7:])
plt.text(x = -0.35, y = 0, s = 'Toplam Paylaşım: {}'.format(df.shape[0]))
plt.title('Veri Setindeki Paylaşımların Dağılımları', fontsize = 16);


sns.barplot(x=[1,0],y = df["sınıf"].value_counts())

df["karakter_len"]= df["emojisiz"].apply(len)
plt.figure("0-1 histogram grafiği")

sns.distplot(df[df["sınıf"]==0]["karakter_len"].values,bins=20 , label = " Negatif değerlerinin histogram")

sns.distplot(df[df["sınıf"]==1]["karakter_len"].values,bins = 20 ,label="Pozitif değerlerin Histogram")

plt.xlabel("Karakter Uzunluğu")
plt.ylabel("Frekans(Yoğunluk)")
plt.legend(loc="best")
plt.show()
df["kelime"] = df["emojisiz"].apply(lambda x : len(x.split()))

plt.figure("kelimelerin 0 ve 1 değerlerine göre kda sı")

sns.distplot(df[df["sınıf"]==0]["kelime"].values,bins=20,label=" 0 değeri için hist")
sns.distplot(df[df["sınıf"]==1]["kelime"].values,bins=20,label="1 değeri için hist")

plt.xlabel("Kelimee Uzunlukları")
plt.ylabel("Frekans (yoğunluk)")
plt.legend(loc="best")
plt.show()

from wordcloud import WordCloud
metin = df.emojisiz.tolist()
metin_kombin = " ".join(metin)
plt.figure(figsize=(14,14))
plt.imshow(WordCloud().generate(metin_kombin))
plt.axis("off")
# pozitif sınıflandırıcının görselleştirilmesi

pozitif = df.emojisiz[df.sınıf == 1]

pozitif_metin = pozitif.tolist()
pozitif_metin_ekleme = " ".join(pozitif_metin)
plt.figure(figsize=(14,14))
plt.imshow(WordCloud().generate(pozitif_metin_ekleme))
plt.axis("off")
plt.title("Pozitif Metin")
# Negatif Metin içindeki en çok kullanılan kelimeler

negatif = df.emojisiz[df.sınıf==0]
negatif_metin= negatif.tolist()
negatif_metin_ekle = " ".join(negatif_metin)
plt.figure(figsize=(14,14))
plt.imshow(WordCloud().generate(negatif_metin_ekle))
plt.axis("off")
plt.title("Negatif Metnin Dağılımı")

train_size = int(len(df)*0.8)
test_size = int(len(df)-train_size)

print("Eğitim Boyuyu=" ,train_size)
print("Test Boyuyu=" ,test_size)
def df_split (df,train_size):
    train = df[:train_size]
    test= df[train_size:]
    return train,test
train_y,test_y = df_split(df["sınıf"],train_size)
train_x,test_x = df_split(df["emojisiz"],train_size)
from sklearn.svm import SVC
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier , VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

def k_fold (vectorizer, model, data, name):
   
    pipeline = Pipeline([('vect', vectorizer),
                         ('chi', SelectKBest(chi2, k="all")),
                         ('clf', model)])
    kf = KFold(n_splits=11, shuffle=True)
    scores = []
    
    for train_index, test_index in kf.split(df):
        train_text = df.iloc[train_index]['emojisiz'].values.astype('U')
        train_y = df.iloc[train_index]['sınıf'].values.astype('U')

        test_text = df.iloc[test_index]['emojisiz'].values.astype('U')
        test_y = df.iloc[test_index]['sınıf'].values.astype('U')

        model = pipeline.fit(train_text, train_y)
        predictions = model.predict(test_text)
        with warnings.catch_warnings():
            warnings.filterwarnings(action='once')
            score = accuracy_score(test_y, predictions)
            print(score)
            scores.append(score)
            skor=str(sum(scores)/len(scores))
            
    
        return skor
    
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
cv_result=[]
stop_words = set(stopwords.words('turkish'))
vectorizer = TfidfVectorizer(min_df=10, max_df=0.95, sublinear_tf=True, norm='l2',ngram_range=(1, 3), encoding='windows-1254', stop_words=stop_words, analyzer='word')
models = [('LogisticRegression', LogisticRegression(solver='newton-cg', multi_class='multinomial')),
          ('SVC', SVC(kernel = "rbf")),
          ('SGDClassifier', SGDClassifier(tol=1e-3, penalty='l2')),
          ("MultinomialNB",MultinomialNB()),
          ("KNeighborsClassifier",KNeighborsClassifier()),
          ("RandomForestClassifier",RandomForestClassifier (random_state=16)),
          ("DecisionTreeClassifier",DecisionTreeClassifier(random_state=16)),
          ("AdaBoostClassifier",AdaBoostClassifier()),
          ("Bagging Classifier",BaggingClassifier())
         ]
sonuc = {}
for name, model in models:
        
        sonuc.update({name : k_fold(vectorizer, model, df, name)}) 
        print("***************************** \n",sonuc)
        
        
# Denediğimi algoritmaları bir sonuc adlı Dictionary değişkenine kaydetmiştik. 
# Burada ise bu sözlüğü bir data frame dönüşütürüyoruz
sozluk = list(sonuc.items())
array_dic = np.array(sozluk)
print("Dizi Şekli*** \n",array_dic,"\n")
visualDF = pd.DataFrame ( data = array_dic ,columns= ["Model Adı","Başarım Oranı"])
visualDF.head(9)

g = sns.barplot("Başarım Oranı","Model Adı",data=visualDF)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")
def basarım (sonuc):
    max_basarım = max(sonuc,key = sonuc.get)
    return (max_basarım)
print("Yapmış olduğumuz 9 farklı makine öğrenmesi algoritması arasında en başarılı sonucu veren algoritma = {} ' dir".format(basarım(sonuc)))    