# Hier laden we onze gereedschapskist in



import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn import preprocessing



np.random.seed(10)

from sklearn.metrics import accuracy_score

from sklearn import svm

from matplotlib import cm

import graphviz

from sklearn.tree import export_graphviz

from numpy import array



import matplotlib.pyplot as plt

import seaborn as sbn

from matplotlib import cm

from IPython.display import display

from IPython.display import display, Markdown



from pylab import rcParams  # Plaatjes worden wat groter



rcParams["figure.figsize"] = 10, 10



pd.set_option("display.max_colwidth", -1)  # We willen zo veel mogelijk text zien
def get_message_length(text):

    return len(text)



def contains_www(text):

    return int("www." in text)



def frequency_free(text):

    return len([word for word in text.split(" ") if str.lower(word) == "free"])



def number_of_dots(text):

    return text.count(".")



def contains_personal_word(text):

    personal_words = ['i', 'we', 'you', 'he', 'she'] # <---------- HIER AANVULLEN, lijst van persoonlijke woorden

    words = text.split(' ') # splits het bericht op in woorden

    words = [str.lower(word) for word in words] # zet alle woorden om in kleine letters

    personal_words_in_text = filter(lambda word: word in personal_words, words) # haal alle niet persoonlijke woorden uit bericht

    return len(list(personal_words_in_text)) # tel aantal overgebleven persoonlijke woorden



def number_of_spam_words(text):

    spam_words = ["won", "win", "free", "claim", "mobile", "nokia", "1st", "£", "$"]

    words = text.split(" ")                                                             # splits het bericht op in woorden

    words = [str.lower(word) for word in words]                                         # zet alle woorden om in kleine letters

    spam_words_in_text = filter(lambda word: word in spam_words, words)                 # haal alle niet persoonlijke woorden uit bericht

    return len(list(spam_words_in_text))                                                # tel aantal overgebleven persoonlijke woorden



def plot_svc(x, y, z, model):

    xx, yy = np.meshgrid(np.arange(0, 7, 0.02), np.arange(0, 7, 0.02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    plt.axis("off")

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    plt.scatter(x, y, c=z, cmap=cm.coolwarm, s=100)



def show_features(df):

    display(Markdown("Features: \n - " + " \n - ".join(df.columns)))

df = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv",encoding='latin-1') # laad bestand en maak tabel

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)                   # verwijder nutteloze kolommen

df = df.rename(columns = {'v1': "label", 'v2': 'text'})                                   # hernoem de kolommen naar iets dat wij kunnen begrijpen

df.head()                                                                                 # laat het bovenste stukje van de tabel zien
df.groupby('label').count()/df.shape[0]*100
df.groupby('label').count().plot.pie(y='text', figsize=(5, 5))
print("HAM\n--------------------")

print(df[df['label'] == 'ham']['text'].sample(5).values)
print("SPAM\n-----------------")

print(df[df['label'] == 'spam']['text'].sample(5).values)
def count_number_of_digits(text):

    return sum(c.isdigit() for c in text)



df['number_of_digits'] = df['text'].apply(count_number_of_digits)

df.head()
def count_number_of_upper_cases(text):

    return sum(c.isupper() for c in text)



df['number_of_upper_cases'] = df['text'].apply(count_number_of_upper_cases)

df.head()
df['message_length']           = df['text'].apply(get_message_length)

df['contains_www']             = df['text'].apply(contains_www)

df['frequency_free']           = df['text'].apply(frequency_free)

df['number_of_personal_words'] = df['text'].apply(contains_personal_word)

df['number_of_dots']           = df['text'].apply(number_of_dots)

df['number_of_spam_words']     = df['text'].apply(number_of_spam_words)

df.head()
df_features = df.loc[:, [column for column in df.columns if column is not "text"]]

df_features.head()
le = preprocessing.LabelEncoder()

labels = le.fit_transform(df['label'])

df_features['label'] = labels

df_features.head()
data = array([[1,  2,  1],

              [2,  5,  1],

              [3,  3,  1],

              [5,  6,  0],

              [6,  1,  0]])



plt.scatter(x=data[:,0], y=data[:,1], c=data[:,2], cmap=cm.coolwarm, s=100)
data = array([[1,  2,  1],

              [2,  5,  1],

              [3,  3,  1],

              [5,  6,  0],

              [6,  1,  0]])



features =  data[:,0:-1]

labels   =  data[:,-1]



model = svm.SVC(gamma='scale',kernel='linear')

model.fit(features, labels)  



plot_svc(features[:,0], features[:,1],labels,model)
data = array([[1,  2,  1],

              [2,  5,  1],

              [2.5,4,  0],

              [3,  3,  1],

              [5,  6,  0],

              [6,  1,  0]])



features =  data[:,0:-1]

labels   =  data[:,-1]



model = svm.SVC(gamma='scale',kernel='linear')

model.fit(features, labels)  



plot_svc(features[:,0], features[:,1],labels,model)

data = array([[1,  2,  1],

              [2,  5,  1],

              [2.5,4,  0],

              [3,  3,  1],

              [5,  6,  0],

              [6,  1,  0]])



features =  data[:,0:-1]

labels   =  data[:,-1]



model = svm.SVC(gamma='scale',kernel='poly', degree=5)

model.fit(features, labels)  



plot_svc(features[:,0], features[:,1],labels,model)
data = array([[1, 2, 1],

       [2,   5,   1],

       [2.5, 4,   0],

       [4,   3,   1],

       [4,   1,   1],

       [3.3, 6,   0],

       [5,   3,   0],

       [6,   1,   0],

       [3.5, 4,   1],

       [5,   4.5, 0],

       [6,   6,   1]])



features =  data[:,0:-1]

labels   =  data[:,-1]



model = svm.SVC(gamma='scale',kernel='poly', degree=7) # verander hier de graad van de polynoom

model.fit(features, labels)  



plot_svc(features[:,0], features[:,1],labels,model)
data = array([[1, 2, 1],

       [2, 5, 1],

       [2.5, 4, 0],

       [4, 3, 1],

       [4, 1, 1],

       [3.3, 6, 0],

       [5, 3, 0],

       [6, 1, 0],

       [3.5, 4, 1],

       [5, 4.5, 0],

       [6, 6, 1]])



features =  data[:,0:-1]

labels   =  data[:,-1]



model = svm.SVC(gamma='scale',kernel='linear')

model.fit(features, labels)  



plot_svc(features[:,0], features[:,1],labels,model)
show_features(df_features)
import plotly.express as px

fig = px.scatter_3d(df_features, x='message_length', y='number_of_spam_words', z='number_of_dots',color='label')

fig.show()
feature_names = ['number_of_spam_words', 'message_length', 'number_of_dots']

X = df_features[feature_names]

y = df_features['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
model = svm.SVC(gamma='scale',kernel='linear')

model.fit(X_train, y_train) ;
y_pred = model.predict(X_test)

accuracy_score(y_test, y_pred)
# Maak de features

feature_names = ['number_of_spam_words', 'message_length', 'number_of_dots']

X = df_features[feature_names]

y = df_features['label']



# Split in train- en test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)



# Maak een beslisboom

clf = DecisionTreeClassifier(max_depth=2) # <---- verander hier de diepte van de boom!

clf = clf.fit(X_train,y_train)



# Voorspel de output

y_pred = clf.predict(X_test)



# Bereken de accuracy

accuracy_score(y_test,y_pred)
data = export_graphviz(clf,out_file=None,feature_names=feature_names,class_names=['ham', 'spam'],   

                         filled=True, rounded=True,  

                         special_characters=True)

graphviz.Source(data)
## Your Code Here...


