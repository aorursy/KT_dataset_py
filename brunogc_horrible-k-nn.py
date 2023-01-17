import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os



# Use input to run in kaggle

# DATA_PATH = "../input/"

DATA_PATH = "../input/"







print(os.listdir(DATA_PATH))



# Any results you write to the current directory are saved as output.
# Bibliotecas de visualização

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

train_df = pd.read_csv(DATA_PATH + "train.csv")

test_df = pd.read_csv(DATA_PATH + "test.csv")

sample = pd.read_csv(DATA_PATH + "sampleSubmission.csv")
train_df.head()
test_df.head()
train_df.isna().sum()
test_df.isna().sum()
train_df.drop(columns="Surname", inplace=True)

test_df.drop(columns="Surname", inplace=True)
# Vamos primeiro substituir o gênero pelo o que aparece mais frequentemente

sns.countplot(train_df["Gender"])

plt.show()

sns.countplot(train_df["Geography"])

plt.show()
# Primeiro o pais

train_df.fillna({"Geography": "France"}, inplace=True)

test_df.fillna({"Geography": "France"}, inplace=True)



# Agora o gênero

train_df.fillna({"Gender": "Male"}, inplace=True)

test_df.fillna({"Gender": "Male"}, inplace=True)



# lembre de passar o parâmetro INPLACE para ele modificar o dataframe inicial!
# Primeiro o pais

mean = train_df["EstimatedSalary"].mean()

train_df.fillna({"EstimatedSalary": mean}, inplace=True)

test_df.fillna({"EstimatedSalary": mean}, inplace=True)

for col in train_df.columns:

    print(col, "\t\t", train_df[col].dtype)
from sklearn.preprocessing import LabelEncoder



for col in ["Gender", "Geography"]:

    # Criar o encoder

    encoder = LabelEncoder()

    encoder.fit(train_df[col])

    

    # Substituir aquela coluna no treino e teste

    train_df[col] = encoder.transform(train_df[col])

    test_df[col] = encoder.transform(test_df[col])
train_df[["Gender", "Geography"]].head()
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(train_df.drop(columns="Exited"), train_df["Exited"], test_size=0.3, random_state=42)
x_train.head()
from sklearn.neighbors import KNeighborsClassifier



# Treinar

model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)
from sklearn.metrics import roc_auc_score



# Por padrão o método "predict_proba" nos retorna a probablidade tanto da classe 0 como da 1; 

# Estamos interesado só na da 1 (a da classe 0 é o complemento disso para chegar em 1)

y_preds_val = model.predict_proba(x_val)[:, 1]

result_val = roc_auc_score(y_val, y_preds_val)



print(f"Resultado com K=5 sobre o conjunto de validação", result_val)
model = KNeighborsClassifier(n_neighbors=5)

model.fit(train_df.drop(columns=["Exited"]), train_df["Exited"])
y_preds_test = model.predict_proba(test_df)[:, 1]
test_df.shape
len(y_preds_test)
# Use this to test on Kaggle

sample["Exited"] = y_preds_test
sample.to_csv("bad_knn.csv", index=False)