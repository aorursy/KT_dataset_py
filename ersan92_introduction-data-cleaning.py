text1 = "Ethics are built right into the ideals and objectives of the United Kingdoms."
len(text1)
text2 = text1.split(' ')
text2
len(text2)
[w for w in text2 if len(w) >3]
[w for w in text2 if w.istitle()]
[w for w in text2 if w.endswith('s')]
text3 = "To be or not to be"
text4 = text3.split(' ')
len(text4)
len(set(text4))
set(text4)
# İçinde 'appointment' kelimesi geçenleri getir.
df['text'].str.contains('appointment')
text5 = "ouagadougou"
text6 = text5.split('ou')
# split() aranan değeri bulup çıkarır. Fakat başta ve sonda ise boşluk bırakır.
text6
# 'ou' yu text6 ile birleştir
'ou'.join(text6)
# text5 teki tüm karakterleri yazdır
list(text5)
# text5 teki tüm karakterleri yazdır
[w for w in text5]
text8 = "  A quick brown fox jumped over the lazy dog.   "
text8.split(' ')
# Baştaki ve sondaki Boşlukları kaldır.
text9 = text8.strip()
text9
text9
text9.find('o')
# küçük 'o' büyük 'O' yap.
text9.replace('o','O')
te = "istanbul bir yeditepe şehridir."
te.replace('i','İ')
# Örnek. Aşağıdaki metin içerisinden '@' ile başlayanları çek al bir değişkene at.
text7 = '@UN @UN_Women "Ethics are built right into the ideals and objectives of the United Nations" \
#UNSG @ NY Society for Ethical Culture bit.ly/2guVelr'
text8 = text7.split(' ')
import re

[w for w in text8 if re.search('@[A-Za-z0-9_]+',w)]
text = '"Ethics are built right into the ideals and objectives of the United Nations." #UNSG @ NY Scoenty for Ethical Culturebit.ly/2gruvdsf @UN @UN_Women'
# ilk önce boşluklara bölelim.
text11 = text.split(' ')
text11
#TEST aşağıdaki metinden hastag leri bul getir.
tweet = "@nltk Text analysis is awesome! #regex #pandas #python"
[w for w in tweet.split() if w.startswith('#')]
[w for w in text11 if w.startswith('#')]
[w for w in text11 if w.startswith('@')]
import re
[w for w in text11 if re.search('@[A-Za-z0-9_]',w)]
[w for w in text11 if re.search('@[A-Za-z0-9_]',w)]
[w for w in text11 if re.search('@\w+',w)]
[w for w in text11 if re.search('@',w)]
text12 = "ouagadougou"
# Bu karakterleri bul getir.
re.findall(r'[aeiou]', text12)
# Bu karakterler haricinde kalanları getir.
re.findall(r'[^aeiou]',text12)
# Bu da search ile oluyor mu diye kontrol edişim
[w for w in text12 if re.search(r'[aeiou]',w)]
import pandas as pd

time_sentences = ["Monday: The doctor's appointment is at 2:45pm.", 
                  "Tuesday: The dentist's appointment is at 11:30 am.",
                  "Wednesday: At 7:00pm, there is a basketball game!",
                  "Thursday: Be back home by 11:15 pm at the latest.",
                  "Friday: Take the train at 08:10 am, arrive at 09:00am."]

df = pd.DataFrame(time_sentences, columns=['text'])
df
df['text'].str.len()
# her satırın karakter sayısnı bulun.
df['text'].str.len()
# find the number of tokens for each string in df['text']
df['text'].str.split().str.len()
# find which entries contain the word 'appointment'
df['text'].str.contains('appointment')
# find how many times a digit occurs in each string
df['text'].str.count(r'\d')
# find all occurances of the digits
df['text'].str.findall(r'\d')
# group and find the hours and minutes
df['text'].str.findall(r'(\d?\d):(\d\d)')
# Günleri '???' ile değiştirin.
df['text'].str.replace(r'\w+day\b','???')
# replace weekdays with 3 letter abbrevations
df['text'].str.replace(r'(\w+day\b)', lambda x: x.groups()[0][:3])
# çıkarılan grupların ilk eşleşmesinden yeni sütunlar oluşturun
df['text'].str.extract(r'(\d?\d):(\d\d)')
# Tüm zamanı dönemi ve saatleri çıkarın.
df['text'].str.extractall(r'((\d?\d):(\d\d) ?([ap]m))')
# grup isimleriyle tüm zamanı, saatleri, dakikaları ve dönemi çıkarın
df['text'].str.extractall(r'(?P<time>(?P<hour>\d?\d):(?P<minute>\d\d) ?(?P<period>[ap]m))')
import pandas as pd
time_sentences = ["Monday: The doctor's appointment is at 2:45pm.", 
                  "Tuesday: The dentist's appointment is at 11:30 am.",
                  "Wednesday: At 7:00pm, there is a basketball game!",
                  "Thursday: Be back home by 11:15 pm at the latest.",
                  "Friday: Take the train at 08:10 am, arrive at 09:00am."]

df = pd.DataFrame(time_sentences, columns=['Text'])
df
df['Text'].str.len()
# Her satırdakı boşluk sayısına bakalım
df['Text'].str.split().str.len()
# Bir dizede bir desen içerip içermediğine bakmamız için
df['Text'].str.contains('appointment')
# her satırda kaç karakter olduğuna bakmak için
df['Text'].str.count(r'\d')
# Findall ile satırlarda bulunan sayıları getir.
df['Text'].str.findall(r'\d')
df['Text'].str.findall(r'(\d?\d):(\d\d)')
df['Text'].str.replace(r'\w+day\b','???')
df['Text'].str.replace(r'(\w+day\b)', lambda x : x.groups()[0][:3])
# Gruplar hazırlamaya bakalım.
df['Text'].str.extract(r'(\d?\d):(\d\d)')
df['Text'].str.extractall(r'((\d?\d):(\d\d) ?([ap]m))')
df['Text'].str.extractall(r'(?P<Time>(?P<hour>\d?\d):(?P<Dakika>\d\d) ?(?P<period>[ap]m))')
text = ['This is dirty TEXT: A phone number +001234561234, moNey 3.333, some date like 09.08.2016 and weird Čárákterš.']
[w for w in text]
print(text)
