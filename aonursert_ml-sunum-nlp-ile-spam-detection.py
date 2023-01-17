# Natural Language Processing işlemini sağlayan kütüphaneyi ekliyoruz.
import nltk
# Bu çalışmada kullanılacak 'stopwords' paketini indirmemizi sağlayan kod.
nltk.download_shell()
# Veri temizleme için gerekli veri bilimi kütüphanelerini ekliyoruz.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Mesajları kullanmak için bir değişkene aldık.
messages = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv", encoding="ISO-8859-1")
messages.head()
# Gereksiz kolonları sildik ve gerekli kolonlara daha açıklayıcı isimler verdik.
messages = messages[["v1", "v2"]]
messages = messages.rename(columns={"v1": "label", "v2": "message"})
messages.head()
# Burada spam tespiti için yararlı olabilecek mesajın uzunluğunu her mesaj için bir satır olarak ekledik.
messages["length"] = messages["message"].apply(len)
messages.head()
# Görüldüğü üzere mesajın uzunluğu, mesajın spam olup olmadığını belirlemek için iyi bir yöntem.
# Çünkü bu grafiklerle spam mesajların daha uzun olduğu sonucuna vardık.
messages.hist(column="length", by="label", bins=60, figsize=(12,4))
# Natural language process için datasetimizi hazırlıyoruz.
# Öncelikle noktalama işaretlerini kaldırmak için gerekli kütüphaneyi ekliyoruz.
import string
string.punctuation
# Bunlar İngilizce'de stopword'ler ve bir mesajın spam ya da olup olmadığına dair çok bilgi vermiyor.
# Yani sıklıkla kullanılan bu kelimeleri de kaldırıyoruz.
from nltk.corpus import stopwords
stopwords.words("english")
def text_process(mess):
    """
    1. Noktalama işaretlerini kaldırıyoruz.
    2. Sıklıkla kullanılan kelimeleri (stopwords) kaldırıyoruz.
    3. Geriye temiz spam tespitinde kullanılabilecek cümleyi döndürüyoruz.
    """
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = "".join(nopunc)
    nostopwords = [word for word in nopunc.split() if word.lower() not in stopwords.words("english")]
    nostopwords = " ".join(nostopwords)
    return nostopwords
messages.head()
# Sonucu görmek için sadece ilk 5 cümleyi temizliyoruz.
messages["message"].head(5).apply(text_process)
# Daha sonra temizleme işlemini bütün sütuna uyguluyoruz.
messages["message"] = messages["message"].apply(text_process)
messages.head()
# Bundan sonra metin halindeki verimiz hazır, tahminleme yapabilmek için bunları tam sayı değerleriyle ifade etmeliyiz.
from sklearn.feature_extraction.text import CountVectorizer
# bow (Bag Of Words): Kelimelerin, tam sayı değerleriyle ifade edilmesi.
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages["message"])
# Örneğin veri setindeki 4. cümleyi tam sayılarla ifade ediyoruz.
mess4 = messages["message"][3]
print(mess4)
bow4 = bow_transformer.transform([mess4])
print(bow4)
# Şimdi bütün mesajların tam sayı değerleriyle ifade edildiği veriyi elde ediyoruz.
messages_bow = bow_transformer.transform(messages["message"])
# Sayılarla ifade edilen kelimelerin ağırlıklarını yani ne sıklıkla görüldüğünü hesaplamak için TfidfTransformer kullanıyoruz.
# Bu bize spam mesajlarda ağırlıklı olarak hangi kelimelerin kullanıldığı ve
# Normal mesajlarda ağırlıklı olarak hangi kelimelerin kullanıldığına dair bilgi veriyor.
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(messages_bow)
# Öncelikle veri setindeki 4. cümlenin kelimelerinin tam sayı değerlerinin ağırlıklarını buluyoruz.
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)
# Şimdi bütün mesajların tam sayı değerlerinin ağırlıklarının ifade edildiği veriyi elde ediyoruz.
messages_tfidf = tfidf_transformer.transform(messages_bow)
# Tahminlemeye geçebiliriz.
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages["label"])
spam_detect_model.predict(tfidf4)[0]
messages["label"][3]
# MultinomialNB kullanarak veri setimizi eğittik ve kelime 4 için doğru sonuç elde ettik.
# Ancak tam olarak düzgün tahminleme yapmak için veri setimizi train ve test halinde ikiye bölüyoruz.
from sklearn.model_selection import train_test_split
msg_train, msg_test, lbl_train, lbl_test = train_test_split(messages["message"], messages["label"], test_size=0.3)
# Ardından üstteki işlemleri Pipeline kullanarak tekrar tahminleme işlemine kadar yapıyoruz.
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ("bow", CountVectorizer(analyzer=text_process)),
    ("tfidf", TfidfTransformer()),
    ("classifier", MultinomialNB())
])
# Pipeline ile otomatikleştirilmiş işlemler sayesinde verimizi eğitiyoruz.
pipeline.fit(msg_train, lbl_train)
# Daha sonra verimizi tahminliyoruz.
predictions = pipeline.predict(msg_test)
# Tahminlerimizin ne kadar doğru olduğunu kontrol ediyoruz.
from sklearn.metrics import classification_report
print(classification_report(lbl_test, predictions))
# Oldukça güzel bir sonuç çıktı.
pipeline.predict([msg_test.iloc[10]])
messages.iloc[10]
# Tek bir 10. cümleyi tahminleme için de sonuç doğru çıktı.