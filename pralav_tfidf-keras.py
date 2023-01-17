import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df1 = pd.read_csv("../input/pdb_data_no_dups.csv")
df1.shape
df1.head()
df1.structureId.nunique()
df1['publicationYear'].min()
py= df1.groupby("publicationYear").structureId.count()
py.drop(labels = 201.0, inplace = True)
plt.figure(figsize=(12,6))

sns.scatterplot(x = py.keys(), y = py )

plt.ylabel("Number of structure determined")

print ("There are {} different biomolecules present in the dataset".format(df1.classification.nunique()))
clas = (df1.classification.value_counts())[0:20]

clas
plt.figure(figsize=(12,6))

clas.plot(kind = "bar")

plt.ylabel("Counts")

plt.xlabel("Different biomolecule")

plt.title("Top 20 Different molecules in the database")
df1.macromoleculeType.nunique()
df1.macromoleculeType.value_counts()
explode_list = [0.1, 0, 0, 0.1, 0.1]

plt.figure(figsize=(15,6))

df1.macromoleculeType.value_counts()[0:5].plot(kind = 'pie', autopct='%1.1f%%',labels=None,

                                         pctdistance=1.12, explode = explode_list, startangle=45,

                                              shadow = True)

plt.legend(df1.macromoleculeType.value_counts()[0:5].index, loc='upper left') 

plt.axis('equal') 

plt.title("Percent of macromolecules")

plt.show()

df1.experimentalTechnique.value_counts()[0:5]
df2 = pd.read_csv("../input/pdb_data_seq.csv")

df2.head()
df2.head()
df2.structureId.nunique()
df2.shape
df2.macromoleculeType.value_counts()
df = df1.merge(df2, how ="inner", on ="structureId")
df.head()
df.columns
df.drop(['residueCount_y', 'macromoleculeType_y'], axis =1, inplace = True)
df = df[df['macromoleculeType_x'] == "Protein"]
df.shape
df['macromoleculeType_x'].nunique()
df.isnull().sum()
df = df[pd.notnull(df['sequence'])]
df = df[pd.notnull(df['classification'])]
df.isnull().sum()
df.loc[df['sequence'].str.contains('(^XXXX)+')]
index_to_drop = df.loc[df['sequence'].str.contains('(^XXXX)+')].index
df.drop(index = index_to_drop, inplace = True)
df.shape
df_prot = df[["structureId", "classification", "sequence", "chainId"]]
df_prot["seq_length"] = df_prot["sequence"].apply(len)
df_prot = df_prot[df_prot['seq_length'] > 20]
plt.figure(figsize=(6,6))

sns.distplot(df_prot["seq_length"], bins = 80, kde = False, hist_kws={'edgecolor':'k'})

plt.title("Histogram of seq. length")

plt.ylabel("Frequency")
clas_prot = df_prot.groupby("classification").structureId.count().sort_values(ascending = False)[0:20]
ax = clas_prot.plot(kind = "barh", figsize=(10,10), color='steelblue')

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)



for i, v in enumerate(clas_prot):

    ax.text(v + 3, i + .045, str(v), color='black')

    

plt.show()
clas_prot.keys()
leng = df_prot.groupby("classification").seq_length.mean().get(clas_prot.keys())
leng.plot(kind = "bar", figsize=(8,8))

plt.title("Average sequence length of protein")

plt.ylabel("Length of sequence")
df_prot_fin = df_prot[df_prot['classification'].isin(clas_prot.keys())]

df_prot_fin.shape
## df_prot_fin.to_csv("twenty_five.csv")
df_prot_fin["classification"].nunique()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
X_train, X_test,y_train,y_test = train_test_split(df_prot_fin['sequence'], df_prot_fin['classification'], test_size = 0.3, 

                                                  random_state = 101,)
tfidf_transformer = TfidfVectorizer(analyzer = "char_wb", ngram_range= (4,4), 

                                    sublinear_tf= True )

tfidf_transformer.fit(X_train)
vector = tfidf_transformer.transform(X_train)
vector.shape
vector1 = tfidf_transformer.transform(X_test)
vector1.shape
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
model1 = MultinomialNB()
model1.fit(vector, y_train)
pred1 = model1.predict(vector1)
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, pred1))
MR_accuracy = (accuracy_score(y_test, pred1))
MR_accuracy
from sklearn.model_selection import cross_val_score
model1 = MultinomialNB()
scores = cross_val_score(model1, vector, y_train, cv=5, scoring = "accuracy")
print("Scores:", scores)

print("\n")

print("Mean:", scores.mean())

print("\n")

print("Standard Deviation:", scores.std())
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(vector, y_train)
yh = lg.predict(vector1)
print(classification_report(y_test, yh))
LR_accuracy = (accuracy_score(y_test, yh))
LR_accuracy
scores = cross_val_score(lg, vector, y_train, cv=5, scoring = "accuracy")
print("Scores:", scores)

print("\n")

print("Mean:", scores.mean())

print("\n")

print("Standard Deviation:", scores.std())
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(vector, y_train)
yh2 = clf.predict(vector1)
print(classification_report(y_test, yh2))
RF_accuracy = (accuracy_score(y_test, yh2))

RF_accuracy
scores = cross_val_score(clf, vector, y_train, cv=5, scoring = "accuracy")
print("Scores:", scores)

print("\n")

print("Mean:", scores.mean())

print("\n")

print("Standard Deviation:", scores.std())
from sklearn.svm import LinearSVC

lsv = LinearSVC()

lsv.fit(vector, y_train)

yh3 = lsv.predict(vector1)

LSV_accuracy = round(accuracy_score(y_test, yh3),2)

from sklearn.svm import LinearSVC
lsv = LinearSVC()
lsv.fit(vector, y_train)
yh3 = lsv.predict(vector1)
LSV_accuracy = round(accuracy_score(y_test, yh3),2)
LSV_accuracy
scores = cross_val_score(lsv, vector, y_train, cv=5, scoring = "accuracy")
print("Scores:", scores)

print("\n")

print("Mean:", scores.mean())

print("\n")

print("Standard Deviation:", scores.std())
Accuracy = pd.DataFrame(index = ["MB", "LG", "SVC", "RF"], data = [MR_accuracy, LR_accuracy, LSV_accuracy, RF_accuracy], columns = ["Accuracy"])
Accuracy
import seaborn as sns
Accuracy.plot(kind = "bar")
from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils
encoder = LabelEncoder()



encoder.fit(y_train)

encoded_y = encoder.transform(y_train)



dummy_ytrain = np_utils.to_categorical(encoded_y)
encoder1 = LabelEncoder()



encoder1.fit(y_test)

encoded_y1 = encoder1.transform(y_test)



dummy_ytest = np_utils.to_categorical(encoded_y1)
dummy_ytrain[0:4]
from keras.models import Sequential

from keras.layers import Dense
input_dim = (vector.shape[1])
model = Sequential()

model.add(Dense(units = 128, activation = "relu", input_dim=input_dim)) # the numbers of features as the input_dim



model.add(Dense(units= 20, activation = "softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
history = model.fit(vector, dummy_ytrain,

epochs=5,

verbose=1,

validation_split=0.33,

batch_size=128)
scores = model.evaluate(vector1, dummy_ytest, verbose=1)
scores
Accuracy = pd.DataFrame(index = ["MB", "LG", "SVC", "RF", "Keras"], data = [MR_accuracy, LR_accuracy, LSV_accuracy, RF_accuracy, scores[1]], columns = ["Accuracy"])
Accuracy.plot(kind = "bar")