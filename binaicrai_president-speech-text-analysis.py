import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import nltk

import random

from sklearn.metrics import confusion_matrix
# President 1



with open("../input/apj01.txt", "r", encoding = 'latin-1') as f:

    apj01_docs = f.read()

    apj01_docs = apj01_docs.split('\n')



with open("../input/apj02.txt", "r", encoding = 'latin-1') as f:

    apj02_docs = f.read()

    apj02_docs = apj02_docs.split('\n')

    

with open("../input/apj03.txt", "r", encoding = 'latin-1') as f:

    apj03_docs = f.read()

    apj03_docs = apj03_docs.split('\n')



with open("../input/apj04.txt", "r", encoding = 'latin-1') as f:

    apj04_docs = f.read()

    apj04_docs = apj04_docs.split('\n')

    

with open("../input/apj05.txt", "r", encoding = 'latin-1') as f:

    apj05_docs = f.read()

    apj05_docs = apj05_docs.split('\n')

    

with open("../input/apj6.txt", "r", encoding = 'latin-1') as f:

    apj06_docs = f.read()

    apj06_docs = apj06_docs.split('\n')

    

with open("../input/apj07.txt", "r", encoding = 'latin-1') as f:

    apj07_docs = f.read()

    apj07_docs = apj07_docs.split('\n')

    

with open("../input/apj08.txt", "r", encoding = 'latin-1') as f:

    apj08_docs = f.read()

    apj08_docs = apj08_docs.split('\n')

    

with open("../input/apj09.txt", "r", encoding = 'latin-1') as f:

    apj09_docs = f.read()

    apj09_docs = apj09_docs.split('\n')

    

with open("../input/apj10.txt", "r", encoding = 'latin-1') as f:

    apj10_docs = f.read()

    apj10_docs = apj10_docs.split('\n')

    

with open("../input/apj11.txt", "r", encoding = 'latin-1') as f:

    apj11_docs = f.read()

    apj11_docs = apj11_docs.split('\n')

    

with open("../input/apj12.txt", "r", encoding = 'latin-1') as f:

    apj12_docs = f.read()

    apj12_docs = apj12_docs.split('\n')
combined_apj = apj01_docs+apj02_docs+apj03_docs+apj04_docs+apj05_docs+apj06_docs+apj07_docs+apj08_docs+apj09_docs+apj10_docs+apj11_docs+apj12_docs
combined_apj
# President 2



with open("../input/niranjan01.txt", "r", encoding = 'latin-1') as f:

    niranjan01_docs = f.read()

    niranjan01_docs = niranjan01_docs.split('\n')

        

with open("../input/niranjan02.txt", "r", encoding = 'latin-1') as f:

    niranjan02_docs = f.read()

    niranjan02_docs = niranjan02_docs.split('\n')

    

with open("../input/niranjan03.txt", "r", encoding = 'latin-1') as f:

    niranjan03_docs = f.read()

    niranjan03_docs = niranjan03_docs.split('\n')

    

with open("../input/niranjan04.txt", "r", encoding = 'latin-1') as f:

    niranjan04_docs = f.read()

    niranjan04_docs = niranjan04_docs.split('\n')

    

with open("../input/niranjan05.txt", "r", encoding = 'latin-1') as f:

    niranjan05_docs = f.read()

    niranjan05_docs = niranjan05_docs.split('\n')
combined_niranjan = niranjan01_docs+niranjan02_docs+niranjan03_docs+niranjan04_docs+niranjan05_docs
combined_niranjan
# President 3



with open("../input/obama01.txt", "r", encoding = 'latin-1') as f:

    obama01_docs = f.read()

    obama01_docs = obama01_docs.split('\n')

    

with open("../input/obama02.txt", "r", encoding = 'latin-1') as f:

    obama02_docs = f.read()

    obama02_docs = obama02_docs.split('\n')

    

with open("../input/obama03.txt", "r", encoding = 'latin-1') as f:

    obama03_docs = f.read()

    obama03_docs = obama03_docs.split('\n')

    

with open("../input/obama04.txt", "r", encoding = 'latin-1') as f:

    obama04_docs = f.read()

    obama04_docs = obama04_docs.split('\n')

    

with open("../input/obama05.txt", "r", encoding = 'latin-1') as f:

    obama05_docs = f.read()

    obama05_docs = obama05_docs.split('\n')

    

with open("../input/obama06.txt", "r", encoding = 'latin-1') as f:

    obama06_docs = f.read()

    obama06_docs = obama06_docs.split('\n')

    

with open("../input/obama07.txt", "r", encoding = 'latin-1') as f:

    obama07_docs = f.read()

    obama07_docs = obama07_docs.split('\n')

    

with open("../input/obama08.txt", "r", encoding = 'latin-1') as f:

    obama08_docs = f.read()

    obama08_docs = obama08_docs.split('\n')

    

with open("../input/obama09.txt", "r", encoding = 'latin-1') as f:

    obama09_docs = f.read()

    obama09_docs = obama09_docs.split('\n')

    

with open("../input/obama10.txt", "r", encoding = 'latin-1') as f:

    obama10_docs = f.read()

    obama10_docs = obama10_docs.split('\n')

    

with open("../input/obama11.txt", "r", encoding = 'latin-1') as f:

    obama11_docs = f.read()

    obama11_docs = obama11_docs.split('\n')

    

with open("../input/obama12.txt", "r", encoding = 'latin-1') as f:

    obama12_docs = f.read()

    obama12_docs = obama12_docs.split('\n')
combined_obama = obama01_docs+obama02_docs+obama03_docs+obama04_docs+obama05_docs+obama06_docs+obama07_docs+obama08_docs+obama09_docs+obama10_docs+obama11_docs+obama12_docs

combined_obama
# President 4



with open("../input/pra01.txt", "r", encoding = 'latin-1') as f:

    pra01_docs = f.read()

    pra01_docs = pra01_docs.split('\n')

    

with open("../input/pra02.txt", "r", encoding = 'latin-1') as f:

    pra02_docs = f.read()

    pra02_docs = pra02_docs.split('\n')

    

with open("../input/pra03.txt", "r", encoding = 'latin-1') as f:

    pra03_docs = f.read()

    pra03_docs = pra03_docs.split('\n')

    

with open("../input/pra04.txt", "r", encoding = 'latin-1') as f:

    pra04_docs = f.read()

    pra04_docs = pra04_docs.split('\n')

    

with open("../input/pra05.txt", "r", encoding = 'latin-1') as f:

    pra05_docs = f.read()

    pra05_docs = pra05_docs.split('\n')

    

with open("../input/pra06.txt", "r", encoding = 'latin-1') as f:

    pra06_docs = f.read()

    pra06_docs = pra06_docs.split('\n')

    

with open("../input/pra07.txt", "r", encoding = 'latin-1') as f:

    pra07_docs = f.read()

    pra07_docs = pra07_docs.split('\n')

    

with open("../input/pra08.txt", "r", encoding = 'latin-1') as f:

    pra08_docs = f.read()

    pra08_docs = pra08_docs.split('\n')

    

with open("../input/pra09.txt", "r", encoding = 'latin-1') as f:

    pra09_docs = f.read()

    pra09_docs = pra09_docs.split('\n')

    

with open("../input/pra10.txt", "r", encoding = 'latin-1') as f:

    pra10_docs = f.read()

    pra10_docs = pra10_docs.split('\n')

    

with open("../input/pra11.txt", "r", encoding = 'latin-1') as f:

    pra11_docs = f.read()

    pra11_docs = pra11_docs.split('\n')

    

with open("../input/pra12.txt", "r", encoding = 'latin-1') as f:

    pra12_docs = f.read()

    pra12_docs = pra12_docs.split('\n')
combined_pra = pra01_docs+pra02_docs+pra03_docs+pra04_docs+pra05_docs+pra06_docs+pra07_docs+pra08_docs+pra09_docs+pra10_docs+pra11_docs+pra12_docs
combined_pra
# President 5



with open("../input/romney01.txt", "r", encoding = 'latin-1') as f:

    romney01_docs = f.read()

    romney01_docs = romney01_docs.split('\n')

    

with open("../input/romney02.txt", "r", encoding = 'latin-1') as f:

    romney02_docs = f.read()

    romney02_docs = romney02_docs.split('\n')

    

with open("../input/romney03.txt", "r", encoding = 'latin-1') as f:

    romney03_docs = f.read()

    romney03_docs = romney03_docs.split('\n')

    

with open("../input/romney04.txt", "r", encoding = 'latin-1') as f:

    romney04_docs = f.read()

    romney04_docs = romney04_docs.split('\n')

    

with open("../input/romney05.txt", "r", encoding = 'latin-1') as f:

    romney05_docs = f.read()

    romney05_docs = romney05_docs.split('\n')

    

with open("../input/romney06.txt", "r", encoding = 'latin-1') as f:

    romney06_docs = f.read()

    romney06_docs = romney06_docs.split('\n')

    

with open("../input/romney07.txt", "r", encoding = 'latin-1') as f:

    romney07_docs = f.read()

    romney07_docs = romney07_docs.split('\n')

    

with open("../input/romney08.txt", "r", encoding = 'latin-1') as f:

    romney08_docs = f.read()

    romney08_docs = romney08_docs.split('\n')

    

with open("../input/romney09.txt", "r", encoding = 'latin-1') as f:

    romney09_docs = f.read()

    romney09_docs = romney09_docs.split('\n')
combined_romney = romney01_docs+romney02_docs+romney03_docs+romney04_docs+romney05_docs+romney06_docs+romney07_docs+romney08_docs+romney09_docs
combined_romney
print(len(combined_apj))

print(len(combined_niranjan))

print(len(combined_obama))

print(len(combined_pra))

print(len(combined_romney))
final_combined_doc = combined_apj + combined_niranjan + combined_obama + combined_pra + combined_romney

labels = ['APJ']*len(combined_apj) + ['NIR']*len(combined_niranjan) + ['OBA']*len(combined_obama) + ['PRA']*len(combined_pra) + ['ROM']*len(combined_romney)



final_combined_df = pd.DataFrame({"Review": final_combined_doc, "Sentiment": labels})

#combined_df = combined_df.sample(frac=1)

final_combined_df = final_combined_df.sample(frac=1)





final_combined_df.head(50)
final_combined_df.isna().sum(axis = 0)
final_combined_df2 = final_combined_df.copy()
final_combined_df2.reset_index(drop = True, inplace = True)
final_combined_df2.head(20)
final_combined = pd.DataFrame(final_combined_df2, columns=['Review', 'Sentiment']).to_csv('Final_Combined.csv')
df = final_combined_df2.sort_values(['Review'])

df.reset_index(drop = True, inplace = True)

df.head()
df = df[df.Review != '']

df.head()
df = df[df.Review != ' ']

df.head()
df.reset_index(drop = True, inplace = True)

df
combined_df = df.copy()
from sklearn.model_selection import train_test_split



y = combined_df['Sentiment'].tolist()

X = combined_df.loc[:,'Review'].tolist()



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=1224)
from sklearn.feature_extraction.text import TfidfVectorizer

tokenizer = TfidfVectorizer(ngram_range = (1,3), stop_words=None,min_df=10) # min_df and stop_words act as dim reduction

X_train_tf = tokenizer.fit_transform(X_train).toarray()

X_test_tf = tokenizer.transform(X_test).toarray()



print(X_train_tf.shape)

print(X_test_tf.shape)
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=1200, n_iter=10, random_state=42)

X_train_tf = svd.fit_transform(X_train_tf)

print(svd.explained_variance_ratio_.sum())

X_test_tf = svd.transform(X_test_tf)
X_test_tf.shape
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, classification_report



NBclassifier = GaussianNB()

NBclassifier.fit(X_train_tf, y_train)



## Predictions

train_nb_preds = NBclassifier.predict(X_train_tf)

test_nb_preds = NBclassifier.predict(X_test_tf)



print("Train Accuracy",NBclassifier.score(X_train_tf,y_train))

print(confusion_matrix(y_train,train_nb_preds))

print(classification_report(y_train,train_nb_preds))



print("Test Accuracy",NBclassifier.score(X_test_tf,y_test))

print(confusion_matrix(y_test,test_nb_preds))

print(classification_report(y_test,test_nb_preds))
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()

logit.fit(X_train_tf,y_train)



## Predictions

train_logit_preds = logit.predict(X_train_tf)

test_logit_preds = logit.predict(X_test_tf)



print("Train Accuracy",logit.score(X_train_tf,y_train))

print(confusion_matrix(y_train,train_logit_preds))

print(classification_report(y_train,train_logit_preds))



print("Test Accuracy",logit.score(X_test_tf,y_test))

print(confusion_matrix(y_test,test_logit_preds))

print(classification_report(y_test,test_logit_preds))


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)

classifier.fit(X_train_tf, y_train)
rfc_train_pred = classifier.predict(X_train_tf)

rfc_test_pred = classifier.predict(X_test_tf)
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, make_scorer, accuracy_score

print("Train")

print(accuracy_score(y_train,rfc_train_pred))

print(confusion_matrix(y_train,rfc_train_pred))

print(classification_report(y_train,rfc_train_pred))



print("Test")

print(accuracy_score(y_test,rfc_test_pred))

print(confusion_matrix(y_test,rfc_test_pred))

print(classification_report(y_test,rfc_test_pred))