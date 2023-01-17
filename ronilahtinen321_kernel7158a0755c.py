import pandas as pd
from sklearn.neural_network import MLPClassifier

import spacy
data = pd.read_csv("../input/spam_emails_part.csv")
data.head(10)

nlp = spacy.load("en_core_web_sm")
data["text"] = data["text"].apply(lambda row: " ".join([w.lemma_ for w in nlp(row)]))

data.head(10)

import sklearn
from sklearn.model_selection import train_test_split
ennustavat_muuttujat_opetus, ennustavat_muuttujat_testaus, ennustettava_muuttuja_opetus, ennustettava_muuttuja_testaus = sklearn.model_selection.train_test_split(data['text'], data['label'], test_size=0.1, random_state=1)
from sklearn.feature_extraction.text import CountVectorizer
vektorisoija = CountVectorizer() 
ennustavat_muuttujat_opetus = vektorisoija.fit_transform(ennustavat_muuttujat_opetus)



model = MLPClassifier(solver='lbfgs',
                    hidden_layer_sizes=(400, 90))
model.fit(ennustavat_muuttujat_opetus, ennustettava_muuttuja_opetus)
#
from sklearn.metrics import confusion_matrix

ennustavat_muuttujat_testaus_vektorina = vektorisoija.transform(ennustavat_muuttujat_testaus)
## Koodi - Suomi :
# Haluaisin vektorisoida nyt myös testaukseen käytettävän "ennustavat_muuttujat_testaus"-datan 
#... jotta voin käyttää sitä tekoälytyökalun kanssa ennustamiseen. 


ennuste = svm.predict(ennustavat_muuttujat_testaus_vektorina)
## Koodi - Suomi :
# Haluaisin käyttää predict-työkalua ennustamaan testi-datalla.

totuustaulu = confusion_matrix(ennustettava_muuttuja_testaus, ennuste)
# Koodi - Suomi :
# Haluaisin käyttää"confusion_matrix"-työkalua joka vertaa ennustetta oikeisiin tuloksiin
# Antaisin sille parametreina testamiseen tarkoitetun ennustettavat muuttujat sisältävän kirjekuoren(oikeat vastaukset)
#... ja sitten itse ennusteen
# voisitko sujauttaa vertailun lopputuloksen kirjekuoreen nimeltään "totuustaulu"

print(totuustaulu)
# Koodi - Suomi :
#voisitko tulostaa ruudulle "totuustaulu"-kirjekuoren sisäl