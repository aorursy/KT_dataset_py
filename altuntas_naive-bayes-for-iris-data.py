import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import warnings

warnings.filterwarnings('ignore')



from sklearn.naive_bayes import GaussianNB  #naive bayes alghorithms

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/Iris.csv')

df = df.sample(frac=1)  #We are shuffle our data cause it's not random but it will change our predict

                        #cause when it's shuffle same Species can be still together and it will effect our traning
print(df.head())

print(df.isnull().sum())

print(df.dtypes)
features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']

features_x = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']         

df = df[features]

traning_df = df[:100]

testing_df = df[100:]
y = traning_df.Species              #y is our answers

x = traning_df[features_x]          #x is our datas without answers
model_iris = GaussianNB()

model_iris.fit(x,y)             #we are giving x and y so naive bayes alghoritms can learn 
testing_df_x = testing_df[features_x]     #here we are giving testing df's features not the asnwer

predict = model_iris.predict(testing_df_x)#we want asnwers we want to predict Species 

predict = pd.DataFrame(predict,columns=['Predict'])



print(testing_df_x[:6])                #here our features

print('\n',predict[:6])                #here our predict(btw it's id's start from 0 cause returning value is like this)

print('\n',testing_df[:6])             #and both of them together we can check
i=list(range(0,50))

testing_df = testing_df.set_index([i])          #we changing the index for compare our predict and results





for x in range(0,50):

    if testing_df.Species[x] != predict.Predict[x] :

        

        print("wrong predict\n",testing_df.iloc[x],"\n")
features_x = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']         

df = df[features]

traning_df = df[:100]

testing_df = df[100:]





def find_outlier(x):

    q1 = np.percentile(x, 25)

    q3 = np.percentile(x, 75)

    iqr = q3-q1

    floor = q1 - 1.5*iqr

    ceiling= q3 + 1.5*iqr

    outlier_indices = list(x.index[(x<floor)|(x>ceiling)])

    outlier_values = list(x[(x<floor)|(x>ceiling)])

    return outlier_indices, outlier_values



for columns in features_x:

    outlier_traning_df = np.sort(find_outlier(traning_df[columns]))

    traning_df = traning_df.drop(outlier_traning_df[0])



y = traning_df.Species                

x = traning_df[features_x] 



model_iris = GaussianNB()

model_iris.fit(x,y)



testing_df_x = testing_df[features_x]     

predict = model_iris.predict(testing_df_x)

predict = pd.DataFrame(predict,columns=['Predict'])



list(range(0,50))

testing_df = testing_df.set_index([i])





for x in range(0,50):

    if testing_df.Species[x] != predict.Predict[x] :

        

        print("wrong predict\n",testing_df.iloc[x],"\n")
