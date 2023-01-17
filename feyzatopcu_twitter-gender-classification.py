import pandas as pd

import re

import nltk # natural language tool kit

nltk.download("stopwords")      # corpus diye bir klasore indiriliyor

from nltk.corpus import stopwords  # sonra ben corpus klasorunden import ediyorum



    

    

import nltk as nlp

lemma = nlp.WordNetLemmatizer()



data = pd.read_csv(r"../input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv",encoding = "latin1")



data = pd.concat([data.gender,data.description],axis=1)



data.dropna(axis = 0,inplace = True)

data.gender = [1 if each == "female" else 0 for each in data.gender]



print(data.head())

print(data.info())
first_description = data.description[4]

description = re.sub("[^a-zA-Z]"," ",first_description)  # a dan z ye ve A dan Z ye kadar olan harfleri bulma geri kalanları " " (space) ile degistir

description = description.lower()   # buyuk harftan kucuk harfe cevirme

#%% clening data 

description_list = []

for description in data.description:

    # regular expression RE mesela "[^a-zA-Z]"

    description = re.sub("[^a-zA-Z]"," ",description)

    description = description.lower()   # buyuk harftan kucuk harfe cevirme

    # description = description.split()

    # split yerine tokenizer kullanabiliriz

    # split kullanırsak "shouldn't " gibi kelimeler "should" ve "not" diye ikiye ayrılmaz ama word_tokenize() kullanirsak ayrilir

    description = nltk.word_tokenize(description)

    # stopwords (irrelavent words) gereksiz kelimeler

    description = [ word for word in description if not word in set(stopwords.words("english"))]

    # lemmatazation loved => love   gitmeyecegim = > git

    lemma = nlp.WordNetLemmatizer()

    description = [ lemma.lemmatize(word) for word in description] #[ ps.stem(word) for word in description]

    description = " ".join(description)  #vektor için list değil string hali gerekli

    description_list.append(description)

description_list[0:5]
from sklearn.feature_extraction.text import CountVectorizer # bag of words yaratmak icin kullandigim metot

max_features = 5000



count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")



sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()  # x



print("en sik kullanilan {} kelimeler: {}".format(max_features,count_vectorizer.get_feature_names()))

y = data.iloc[:,0].values   # male or female classes

x = sparce_matrix

# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)

# %% naive bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

#%% prediction

y_pred = nb.predict(x_test)



print("accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))