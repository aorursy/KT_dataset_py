#kütüphaneler
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
#veri setini okuyalım:
#veri setinde 1 tane abğımsız değişken,1 tane bağımlı değişken var.
#'Sentence' adlı veri setinde yapılan yorumlar mevcut
#'Class' adlı veri setinde yapılan bu yorumların negatif -->0 ya da pozitif -->1 olması durumu mevcut.
data=pd.read_csv('../input/yorumlar/comment')
data.head()
#veri setindeki 'class' adlı değişkenin ismini 'Sentiment' olarak değiştirdik.
data=data.rename(columns={'Class':'Sentiment'})
data.head()
#veri setimizde 1386 tane pozitif yorum,1362 tane negatif yorum var.
data['Sentiment'].value_counts()
sns.countplot(data.Sentiment);
#yapılan işlemlerde sınıf dönüştürme işlemi yapıldı:0 --> negatif,1 --> pozitif oldu.
data["Sentiment"].replace(0,value="negatif",inplace=True)
data["Sentiment"].replace(1,value="pozitif",inplace=True)
data
#'df' adlı yeni bir dataframe oluşturuduk
#'text' adlı değişekene,'Sentence' adlı değişkendeki gözlemleri atadık.
#'label' adlı değişkene,'Sentiment' adlı değişkendeki gözlemleri atadık.
df=pd.DataFrame()
df["text"]=data["Sentence"]
df["label"]=data["Sentiment"]
df
#buyuk-kucuk donusumu
df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#noktalama işaretleri
df['text'] = df['text'].str.replace('[^\w\s]','')
#sayılar
df['text'] = df['text'].str.replace('\d','')
#stopwords
import nltk
from nltk.corpus import stopwords
sw = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
#seyreklerin silinmesi
sil = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))
#lemmi
from textblob import Word
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.
split()]))
#metin ön işleme uygulandıktan sonra veri setinin son hali:
df
#train-test ayrımı yapıldı:
train_x,test_x,train_y,test_y=model_selection.train_test_split(df["text"],
df["label"])
train_x.head()
train_y.head()
#bağımlı değişkene,0-1 dönüşümü yapıldı:
encoder=preprocessing.LabelEncoder()
train_y=encoder.fit_transform(train_y)
test_y=encoder.fit_transform(test_y)
train_y[0:5]
test_y[0:5]
#değişkenler oluşturuldu.
vectorizer=CountVectorizer()
vectorizer.fit(train_x)
x_train_count=vectorizer.transform(train_x)
x_test_count=vectorizer.transform(test_x)
#değişkenlerime bakalım:
vectorizer.get_feature_names()[0:5]
#1810 tane değişken oluşturulmuş
len(vectorizer.get_feature_names())
#lojisyik regresyon ile model kuruldu.
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_count, train_y)
accuracy = model_selection.cross_val_score(loj_model,
                                           x_test_count,
                                           test_y,
                                           cv = 10).mean()
print("Count Vectors Doğruluk Oranı:", accuracy)
#bir cümle oluşturuldu
yeni_yorum=pd.Series('this film is very nice and good i like it')
v=CountVectorizer()
v.fit(train_x)
yeni_yorum=v.transform(yeni_yorum)
#kurulan model cümlenin negatif mi pozitif mi olduğunu tahmin etti:
#0 --> negatif, 1-->pozitif
#modelintahmin ettiği sonuç doğru
loj_model.predict(yeni_yorum)
#yeni bir cümler oluşturuldu
#cümle negatif bir cümle
yeni_yorum2=pd.Series("no not good look at that shit very bad")
v2=CountVectorizer()
v2.fit(train_x)
yeni_yorum2=v2.transform(yeni_yorum2)
#modelin tahmin ettiği tahmin doğru
#0 -->negatif, 1 -->pozitif
loj_model.predict(yeni_yorum2)
#hiperparametre aralıkları belirlendi:
xgb_params={'n_estimators':[5,10],
'subsample':[0.6,0.8],
'max_depth':[3,4],
'learning_rate':[0.1,0.01],
'min_samples_split':[2,5]}
#gridsearch cv yöntemi ile optimum hiperparametreler bulundu:
xgb=XGBClassifier()
xgb_cv_model=GridSearchCV(xgb,xgb_params,cv=10,n_jobs=1).fit(x_train_count,train_y)
#optimum hiperparametre değerleri:
xgb_cv_model.best_params_
#bulunan optimum hiperparametreler ile final modeli kuruldu:
xgb_model=XGBClassifier(n_estimators=10,subsample=0.6,max_depth=4,learning_rate=0.1,min_samples_split=2)
xgb_cv_model=xgb_model.fit(x_train_count,train_y)
y_pred=xgb_cv_model.predict(x_test_count)
accuracy_score(test_y,y_pred)
#şimdi yeni_yorum ve yeni_yorum2 adlı değişkenlerdedi cümlelerin negataif mi pozitif mi olduğunu kurmuş olduğumuz modelin tahmin etme işlemine bakalım:
#yeni_yorum:'this film is very nice and good i like it' (poztif bir cümle)
#doğru tahmin etti
#0 -->negatif ,1 -->pozitif
xgb_cv_model.predict(yeni_yorum)
#yeni_yorum2:'no not good look at that shit very bad'  (negatif bir cümle)
#yanlış tahmin etti
#0 -->negatif  ,1 -->pozitif
xgb_cv_model.predict(yeni_yorum2)
