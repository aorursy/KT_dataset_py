#Meine Imports

import re

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

import string

import nltk

import emoji

import regex

from sklearn.model_selection import train_test_split



%matplotlib inline

df = pd.read_csv('../input/-chat1-german/_chat1_german.txt', sep='\n', header=None)





print(df.head(2))





#Rohdaten als Source umbennenen und eine Kopie des Orginal data frames erstellen

df["source"] = df[df.columns[0]]

df = df.drop(columns=df.columns[0])

#df.rename(columns={df.columns[0]: "source"})

dfraw = df.copy() #backup



df.head(2)

#Dataframe Split

df = dfraw.copy()



#Angehängte Dateien enthalten einen LEFT-TO-RIGHT Marker der Entfernt werden muss  \u200e (unsichtbares Zeichen)

df["source"] = df["source"].str.replace(u"\u200e", "")





#

df['datum'] = df["source"].apply(lambda x: re.findall('^\[(.{18})\] .*:.*',x))

df['username'] = df["source"].apply(lambda x: re.findall('^\[.{18}\] (.*):.*',x))

df['nachricht'] = df["source"].apply(lambda x: re.findall('^\[.{18}\] .*:(.*)',x))

df['username_kurz'] = df["source"].apply(lambda x: re.findall('^\[.{18}\] ([A-z]{4}).*:.*',x))











#unlist

df['username'] = [','.join(map(str, l)) for l in df['username']]

df['nachricht'] = [','.join(map(str, l)) for l in df['nachricht']]

df['datum'] = [','.join(map(str, l)) for l in df['datum']]

df['username_kurz'] = [','.join(map(str, l)) for l in df['username_kurz']]



#leere String-Einträge entfernen

df['username_kurz'] = df['username_kurz'][df['username_kurz'] != ""]



#drop source

df = df.drop(columns="source")





df.head(10)
# Informationen über unseren Dataframe





print("Spaltenanzahl=", len(df.columns))

print("Anzahl an Reihen", len(df.index))

print("Nullwerte?")

print(df.isnull().sum())

print("-.-----------------------")



#allgemeine infos von pandas

print(df.info())

print(df.describe().T)



#ersten und letzten zeilen

print(df.head(2))

print(df.tail(2))

#Für die kommende Auswertung werde ich nur mit den beiden Spalten username und nachricht weiterarbeiten

train = df[['username_kurz','username', 'nachricht']].copy()



train.tail(4)
#Berechnung neuer Spalten und einfügen in den "train"-Dataframe



#Neue Spalte für die gezählten Wörter

train['wort_anzahl'] = train['nachricht'].apply(lambda x: len(str(x).split(" ")))



#Wie viele Zeichen hat jede einzelne Chatnachricht?

train['zeichen_anzahl'] = train['nachricht'].str.len()



def durchschnittl_wortlaenge(text):

    woerter = text.split()

    try:

        return (sum(len(wort) for wort in woerter)/len(woerter))

    except:

        return 0





train['durchschnittl_wortlaenge'] = train['nachricht'].apply(lambda x: durchschnittl_wortlaenge(x))



train.head()
#TODO: Deutsche Stopwörter importieren

import nltk

#nltk.download('stopwords') #bereits erledigt, bitte kontrollieren ob es vll. erneut heruntergeladen werden muss!



from nltk.corpus import stopwords



#korrekt importiert?

#stopwords.words() #Anmerkung: Auskommentiert, da es in kaggle einen sehr sehr langen output string ergibt
stopwoerter = stopwords.words('german')



train['stopwoerter'] = train['nachricht'].apply(lambda x: len([x for x in x.split() if x in stopwoerter]))



train.head()
#Zählen der verwendeten Zahlen innerhalb der Chatnachrichten

train['anzahl_zahlen'] = train['nachricht'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))



#Zählen von "ALL-CAPS"

train['capslock_vergessen'] = train['nachricht'].apply(lambda x: len([x for x in x.split() if x.isupper()]))



train.head() #Anmerkung: eine Zahl gefolgt von einem Punkt wird nicht als Zahl gezählt z.B. "3."
%timeit

#emoji counter von https://stackoverflow.com/questions/19149186/how-to-find-and-count-emoticons-in-a-string-using-python

def split_count(text):

    emoji_counter = 0

    data = regex.findall(r'\X', text)

    for word in data:

        if any(char in emoji.UNICODE_EMOJI for char in word):

            emoji_counter += 1

            # Remove from the given text the emojis

            text = text.replace(word, '') 



    words_counter = len(text.split())



    return emoji_counter



train['anzahl_emojis'] = train['nachricht'].apply(lambda x: split_count(x))



train[['anzahl_emojis','nachricht']].head(7)
train.head()
#Violinplot der wort_anzahl



sns.violinplot(y=train["wort_anzahl"], x=train["username_kurz"])
#Violinplotverwendete Emojis

sns.violinplot(y=train["anzahl_emojis"], x=train["username_kurz"])
#Violinplot Zeichen Anzahl

sns.violinplot(y=train["zeichen_anzahl"], x=train["username_kurz"])
train.head()
#In Training und Test Datensatz aufspalten

combi = train.copy() 



laenge = ((len(combi['nachricht']))*0.8)

laenge = round(laenge)

print(laenge)





train = combi[0:laenge].copy()



test = combi[laenge:].copy()



train.head()
combi.describe()
#Funktion um gezielt bestimmte Muster in Texten zu entfernen

def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)

        

    return input_txt   
#Backup von Dataframe combi

combi_backup = combi.copy()



combi = combi_backup.copy()

combi.head()
#Removing twitter handles (@user)

combi['nachricht_bereinigt'] = np.vectorize(remove_pattern)(combi['nachricht'], "@[\w]*")



#Rmoving special characters, numbers, punctuations

combi['nachricht_bereinigt'] = combi['nachricht_bereinigt'].str.replace("[^a-zA-Z#]", " ")



#Removing Short Words

combi['nachricht_bereinigt'] = combi['nachricht_bereinigt'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))



combi.head()
#Tokenization

tokenized_chat = combi['nachricht_bereinigt'].apply(lambda x: x.split())

tokenized_chat.head()
for i in range(len(tokenized_chat)):

    tokenized_chat[i] = ' '.join(tokenized_chat[i])



combi['nachricht_bereinigt'] = tokenized_chat



combi['nachricht_bereinigt'].head()
combi.head()
#Nutzernamen den Variablen zuordnen (TODO: per loop für alle User mach "for x in s.index")

s = train["username_kurz"].value_counts()

print(s)



user1 = s.index[0]

user2 = s.index[1]



print(user1)

print(user2)
#Visualisierung

#Alle Wörter

all_words = ' '.join([text for text in combi['nachricht_bereinigt']])

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
#Visualisierung

#für einzelne User (1)

print("Ergebnis für=", user1)

normal_words =' '.join([text for text in combi['nachricht_bereinigt'][combi['username_kurz'] == user1]])



wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
#Visualisierung

#für einzelne User (2)

print("Ergebnis für=", user2)

normal_words =' '.join([text for text in combi['nachricht_bereinigt'][combi['username_kurz'] == user2]])



wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
def word_extract(x):

    words = []

    for i in x:

        ht = re.findall(r"([a-zA-Z]\w+)", i)

        words.append(ht)



    return words



words_user1 = word_extract(combi['nachricht_bereinigt'][combi['username_kurz'] == user1])

words_user2 = word_extract(combi['nachricht_bereinigt'][combi['username_kurz'] == user2])





#unlist

words_user1 = sum(words_user1,[])

words_user2 = sum(words_user2,[])
#Wörter überprüfen

#print(words_user1) #Anmerkung: Auskommentiert, da es in Kaggle einen sehr sehr langen output-String ergibt.
#Balkendiagramm User1

print("Ergebnis für=", user1)

a = nltk.FreqDist(words_user1)

d = pd.DataFrame({'Wort': list(a.keys()),

                  'Anzahl': list(a.values())})

# 15 häufigsten Worte     

d = d.nlargest(columns="Anzahl", n = 15) 

plt.figure(figsize=(16,5))

ax = sns.barplot(data=d, x= "Wort", y = "Anzahl")

ax.set(ylabel = 'Anzahl')

plt.show()
#Balkendiagramm User2

print("Ergebnis für=", user2)

a = nltk.FreqDist(words_user2)

d = pd.DataFrame({'Wort': list(a.keys()),

                  'Anzahl': list(a.values())})

# 15 häufigsten Worte     

d = d.nlargest(columns="Anzahl", n = 15) 

plt.figure(figsize=(16,5))

ax = sns.barplot(data=d, x= "Wort", y = "Anzahl")

ax.set(ylabel = 'Anzahl')

plt.show()
data = combi[combi.username_kurz.notnull()]
from sklearn.feature_extraction.text import CountVectorizer

bow = CountVectorizer(max_df=0.90, min_df=2, max_features=1000)

# bag-of-words feature matrix

Explaining_Features = bow.fit_transform(data['nachricht_bereinigt'])
#Extracting the target variable

y = data['username_kurz']
#Splitting into train and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Explaining_Features, y, test_size=0.2)
#Importing accuracy metrics and classifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

from sklearn.model_selection import ShuffleSplit #For cross validation (however, the sample is already quite small)

from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import LinearSVC, SVC

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import time
models = [

    MultinomialNB(),

    BernoulliNB(),

    KNeighborsClassifier(),

    LogisticRegression(),

    SGDClassifier(),

    LinearSVC(),

    SVC(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    MLPClassifier()

]
%timeit

#Print out fitting accuracy

for model in models:

    print("===============================================")

    classifier_name = str(type(model).__name__)

    print("Testing " + classifier_name)

    now = time.time()

    list_of_labels = sorted(list(set(y_train)))

    fit = model.fit(X_train, y_train)

    print("Learing time {0}s".format(time.time() - now))

    now = time.time()

    predictions = fit.predict(X_test)

    print("Predicting time {0}s".format(time.time() - now))

    print("===============================================")

    precision = precision_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)

    recall = recall_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)

    accuracy = accuracy_score(y_test, predictions)

    cms = confusion_matrix(y_test,predictions)

    print("=================== Results ===================")

    print("Precision   " + str(precision))

    print("Recall      " + str(recall))

    print("Accuracy: " + str(accuracy))

    print("============== Confusion Matrix ===============")

    print(cms)

    print("===============================================\n\n")