from warnings import filterwarnings

filterwarnings(action='ignore')



import numpy as np

import pandas as pd

pd.set_option("display.max_columns", None)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier



from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



import pickle
df = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
df.head(4)
print(f"Dataset have {df.shape[0]} rows and {df.shape[1]} columns")
df.isna().sum()
plt.figure(figsize=(10, 6))

ax = sns.countplot(x="species", data=df)



plt.xlabel("Species", fontsize=14)

plt.ylabel("Count", fontsize=14)

plt.title("Species vs count", fontsize=16)

plt.show()
def get_target(val):

    res = ''

    if val == 'Iris-setosa':

        res = 1

    elif val == 'Iris-versicolor':

        res = 2

    elif val == 'Iris-virginica':

        res = 3

    return res





df['target'] = df['species'].apply(get_target)
df.head(2)
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']



for col in features:

    plt.figure(figsize=(10, 6))

    sns.distplot(df[col])

    plt.title("Distribution of "+col, fontsize=16)

    plt.show()
plt.figure(figsize=(10, 10))

sns.heatmap(df.drop(columns=['species']).corr(), fmt='.2f', annot=True)



plt.show()
X, y = df.drop(columns=['species', 'target']), df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y)
scaler = StandardScaler()

scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test  = scaler.transform(X_test)
label_enc = LabelEncoder()

label_enc.fit(y_train)



y_train = label_enc.transform(y_train)

y_test = label_enc.transform(y_test)
def cvs(clf, name, c=5):

    res =  cross_val_score(clf, X, y, cv=c)

    _data = {}

    _data['algo'] = name

    _data['mean'] = res.mean()

    _data['std'] = res.std()

    return _data
scores = []



lr = LogisticRegression()

knn = KNeighborsClassifier()

gnb = GaussianNB()

bnb = BernoulliNB()

mnb = MultinomialNB()

rfc = RandomForestClassifier()

abc = AdaBoostClassifier()

etc = ExtraTreesClassifier()

xgb = XGBClassifier()

lgc = LGBMClassifier()



scores.append(cvs(lr, 'lr'))

scores.append(cvs(knn, 'knn'))

scores.append(cvs(gnb, 'gnb'))

scores.append(cvs(bnb, 'bnb'))

scores.append(cvs(mnb, 'mnb'))

scores.append(cvs(rfc, 'rfc'))

scores.append(cvs(abc, 'abc'))

scores.append(cvs(etc, 'etc'))

scores.append(cvs(xgb, 'xgb'))

scores.append(cvs(lgc, 'lgc'))
df_algo = pd.DataFrame.from_records(scores)

df_algo = df_algo.sort_values(by=['std'])



print(df_algo)
sns.set_style("white")

fig = plt.figure(figsize=(24, 12))



ax1 = sns.pointplot(x=df_algo.algo.tolist(), y=df_algo["mean"], markers=['o'], linestyles=['-'], color='red')

for i, score in enumerate(df_algo["mean"].tolist()):

    ax1.text(i, score + 0.002, '{:.2f}'.format(score), horizontalalignment='left', fontsize=24, color='black', weight='semibold')



ax2 = sns.pointplot(x=df_algo.algo.tolist(), y=df_algo["std"], markers=['o'], linestyles=['-'])

for i, score in enumerate(df_algo["std"].tolist()):

    ax2.text(i, score + 0.002, '{:.6f}'.format(score), horizontalalignment='left', fontsize=20, color='black', weight='semibold')





plt.title('Scores of Models', fontsize=30)

plt.xticks(fontsize=24)

plt.yticks(fontsize=24)

plt.show()
rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)



print(rfc.score(X_train, y_train), rfc.score(X_test, y_test))
ypred = rfc.predict(X_test)

print(classification_report(y_test, ypred))

print("\n\n")

print(confusion_matrix(y_test, ypred))
# Save the Modle to file in the current working directory

model_filename = "./model.pkl"  

label_enc_filename = "./label_encoder.obj"

scaler_filename = "./scaler.obj"



with open(model_filename, 'wb') as file_model:  

    pickle.dump(rfc, file_model)

    

with open(label_enc_filename, 'wb') as file_enc:  

    pickle.dump(label_enc, file_enc)

    

with open(scaler_filename, 'wb') as file_scaler:  

    pickle.dump(scaler, file_scaler)
# Load the Model back from file

with open(model_filename, 'rb') as file_model:  

    model = pickle.load(file_model)





# Load the Encoder back from file

with open(label_enc_filename, 'rb') as file_enc:  

    lblencoder = pickle.load(file_enc)



    

# Load the Scaler back from file

with open(scaler_filename, 'rb') as file_scaler:  

    scaler = pickle.load(file_scaler)
y_testpred = rfc.predict(X_test)



print(classification_report(y_test, y_testpred))

print(confusion_matrix(y_test, y_testpred))