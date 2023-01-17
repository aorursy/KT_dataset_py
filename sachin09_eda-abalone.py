#Basic Dependencies
import numpy as np;
import pandas as pd;
from os.path import exists,basename,dirname,join;
import os;
import matplotlib.pyplot as plt;
from time import time
#Loading Dataset
csv_path = "../input/abalone.csv"
def read_csv_f(*path):
    new_path = join(*path)
    return pd.read_csv(new_path)

df = read_csv_f(csv_path)
df.head(3) # Preview of Data
df_attributes = list(df.columns)
num = df.shape[0]
print(df_attributes, num ,sep= "\n", end= "")
#Encode Categoorical data into numeric data
from sklearn.preprocessing import LabelEncoder

def encode_category(data,columns,copy=False):
    meta=dict()
    X=data
    if copy is True:
        X=data.copy()
    lb_make = LabelEncoder()
    for col in columns:
        X[col] = lb_make.fit_transform(X[col])
        meta[col]=list(lb_make.classes_)
    return X,meta

df,encode_info = encode_category(df, ["Sex"])
print(encode_info)
df.head(2)
df.info()
#The Coorelation b/w various features are indicating the degree of association b/w them
#In Abalone Dataset , there is a strong association b/w Weight, Diameter and Length
attributes = ["Length", "Diameter", "Height", "Rings", "Whole weight"]
print(df[attributes].corr())
df[["Length", "Diameter", "Height", "Rings", "Whole weight","Viscera weight", "Shucked weight", "Shell weight"]].corr()
attributes = ["Length", "Diameter", "Height", "Rings", "Whole weight"]
X = df[attributes]
y = df["Sex"]
print(X.head(2))
X.corr()
#Features Distribution for Abalone Snails
figures = [221,222,223,224,111]

for attr,fig in zip(attributes,figures):
    plt.figure( figsize=(7,7))
    plt.subplot(fig)
    plt.hist(df[attr])
    plt.legend()
    plt.title(attr)
    plt.grid()
    plt.show()
female, infant, male = df[ df["Sex"] == 0], df[ df["Sex"] == 1], df[ df["Sex"] == 2]
temp = pd.DataFrame( {"infant": infant.max(), "male": male.max(), "female": female.max() })
temp