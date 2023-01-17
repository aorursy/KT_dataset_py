# Imports(everything i might require)

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
# get makemytrip & test csv files as a DataFrame
makemytrip_df = pd.read_csv("C:/Users/MAHE/Desktop/datasete9dc3ed (1)/dataset/train.csv")
test_df    = pd.read_csv("C:/Users/MAHE/Desktop/datasete9dc3ed (1)/dataset/test.csv")
# Preparing the data
#F and G
#genarating a value using labelencoder
encoder = preprocessing.LabelEncoder()
makemytrip_df["F"] = encoder.fit_transform(makemytrip_df["F"].fillna('NaN'))
makemytrip_df["G"] = encoder.fit_transform(makemytrip_df["G"].fillna('NaN'))
test_df["F"] = encoder.fit_transform(test_df["F"].fillna('NaN'))
test_df["G"] = encoder.fit_transform(test_df["G"].fillna('NaN'))


#assigninf NaN values to '0' values given by encoder
import numpy
makemytrip_df["F"] = makemytrip_df["F"].replace(0 , numpy.NaN)
makemytrip_df["G"] = makemytrip_df["G"].replace(0, numpy.NaN)

test_df["F"] = test_df["F"].replace(0,numpy.NaN)
test_df["G"] = test_df["G"].replace(0,numpy.NaN)

#filling '0' values with mean of the columns

makemytrip_df["F"] = makemytrip_df["F"].fillna(makemytrip_df["F"].mean())
makemytrip_df["G"] = makemytrip_df["G"].fillna(makemytrip_df["G"].mean())

test_df["F"] = test_df["F"].fillna(test_df["F"].mean())
test_df["G"] = test_df["G"].fillna(test_df["G"].mean())

#round off
makemytrip_df["F"] = round(makemytrip_df["F"], 0)
makemytrip_df["G"] = round(makemytrip_df["G"], 0)

test_df["F"] = round(test_df["F"], 0)
test_df["G"] = round(test_df["G"], 0)

# converting float to int
makemytrip_df['F'] = makemytrip_df['F'].astype(int)
makemytrip_df['G'] = makemytrip_df['G'].astype(int)
test_df['F'] = test_df['F'].astype(int)
test_df['G'] = test_df['G'].astype(int)
# A
#fill the missing values with the most occurred value, which is "b".
makemytrip_df.columns = makemytrip_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()


makemytrip_df["A"] = makemytrip_df["A"].fillna("b")
test_df["A"] = test_df["A"].fillna("b")

# assigning int values to str
makemytrip_df['A'] = makemytrip_df['A'].map({'a': 1, 'b': 2}).astype(int)
test_df['A'] = test_df['A'].map({'a': 0, 'b': 1}).astype(int)
# B

# since there are missing "B" values
makemytrip_df["B"] = makemytrip_df["B"].fillna(makemytrip_df["B"].mean())
test_df["B"] = test_df["B"].fillna(test_df["B"].mean())

makemytrip_df["B"] = round(makemytrip_df["B"], 0)
test_df["B"] = round(test_df["B"],0)

# convert from float to int
test_df['B'] = test_df['B'].astype(int)
makemytrip_df['B'] = makemytrip_df['B'].astype(int)
#C

makemytrip_df["C"] = round(makemytrip_df["C"], 0)
test_df["C"] = round(test_df["C"],0)

# convert from float to int
test_df['C'] = test_df['C'].astype(int)
makemytrip_df['C'] = makemytrip_df['C'].astype(int)
# D
# only in makemytrip_df, fill the two missing values with the most occurred value, which is "u".
makemytrip_df["D"] = makemytrip_df["D"].fillna("u")
test_df["D"] = test_df["D"].fillna("u")
# assigning int values to str
encoder = preprocessing.LabelEncoder()
makemytrip_df["D"] = encoder.fit_transform(makemytrip_df["D"].fillna('NaN'))
test_df["D"] = encoder.fit_transform(test_df["D"].fillna('NaN'))
#H
makemytrip_df["H"] = round(makemytrip_df["H"], 0)
test_df["H"] = round(test_df["H"], 0)
test_df['H'] = test_df['H'].astype(int)
makemytrip_df['H'] = makemytrip_df['H'].astype(int)

#I and J and L and M
# assigning int values 
encoder = preprocessing.LabelEncoder()
makemytrip_df["J"] = encoder.fit_transform(makemytrip_df["J"].fillna('NaN'))
makemytrip_df["I"] = encoder.fit_transform(makemytrip_df["I"].fillna('NaN'))
makemytrip_df["L"] = encoder.fit_transform(makemytrip_df["L"].fillna('NaN'))
makemytrip_df["M"] = encoder.fit_transform(makemytrip_df["M"].fillna('NaN'))
test_df["J"] = encoder.fit_transform(test_df["J"].fillna('NaN'))
test_df["I"] = encoder.fit_transform(test_df["I"].fillna('NaN'))
test_df["L"] = encoder.fit_transform(test_df["L"].fillna('NaN'))
test_df["M"] = encoder.fit_transform(test_df["M"].fillna('NaN'))
#N
#replacing 0 values with the mean 
makemytrip_df["N"] = makemytrip_df["N"].replace(0, numpy.NaN)
test_df["N"] = test_df["N"].replace(0,numpy.NaN)
makemytrip_df["N"] = makemytrip_df["N"].fillna(makemytrip_df["N"].mean())
test_df["N"] = test_df["N"].fillna(test_df["N"].mean())

makemytrip_df["N"] = round(makemytrip_df["N"], 0)
test_df["N"] = round(test_df["N"], 0)

test_df['N'] = test_df['N'].astype(int)
makemytrip_df['N']  = makemytrip_df['N'].astype(int)
# define training and testing sets

X_train = makemytrip_df.drop(["P", "id"],axis=1)
Y_train = makemytrip_df["P"]
X_test  = test_df.drop("id", axis=1 ).copy()
# training data 
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
# extracting data.csv file
submission = pd.DataFrame({
        "id": test_df ["id"],
        "P": Y_pred
    })
submission.to_csv('submission.csv', index=False)