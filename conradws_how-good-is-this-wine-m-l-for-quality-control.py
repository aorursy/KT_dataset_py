import os



print(os.listdir("../input"))

#We import the library of pandas ( remember pandas is like Excel, but after having taken some sort of illegal steroids)

import pandas as pd

#I previously uploaded our data to this link

#data_url=("https://github.com/busyML/Wine-Quality-Control/blob/master/winewhite.xlsx?raw=true")



#We load our data from that link to Pandas 

data = pd.read_csv("../input/wineQualityWhites.csv")



data.drop(columns=["quality","Unnamed: 0"], inplace=True)

#We print out the first 20 rows of our data to visualize what we are working with here

data.head(21)
import numpy as np # This library allows to easily carry out simple and complex mathematical operations.

import matplotlib.pyplot as plt #Allows us to plot data, create graphs and visualize data. Perfect for your Powerpoint slides ;)

import sklearn #The one and only. This amazing library holds all the secrets. Containing powerful algorithms packed in a single line of code, this is where the magic will happen.

import sklearn.model_selection # more of sklearn. It is a big library, but trust me it is worth it.

import sklearn.preprocessing 

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, explained_variance_score,mean_absolute_error,mean_squared_error,precision_score,recall_score, accuracy_score,f1_score

from sklearn.utils import shuffle

import pandas as pd

from pandas.plotting import radviz





import random # Allows us to call random numbers, occasionally very useful.

#from google.colab import files #Allows to upload and download files directly from the browser.

import pprint#Allows us to neatly display text

from collections import OrderedDict







#Unsupervised Learning

from sklearn.neighbors import NearestNeighbors,LocalOutlierFactor

from sklearn.cluster import KMeans

#we use the "sample" command of pandas to shuffle our data, the random state means that we will always shuffle the data the same way so that when different people load this code, they will all get the same results.

data= data.sample(frac=1, random_state=85)



#we print out the first 20 rows of our data to check that it has indeed been shuffled, on the left we have the index number which we can also think of as an ID number.

data.head(21)
#From The SKlearn library, we can load this handy algorithm called "Local Outlier Factor", we'll call it lof for short from now on.

lof = LocalOutlierFactor(novelty=True)



#Using the ".fit" command, we are ordering our algorithm to learn from our data what a normal white wine should be

lof.fit(data)



print("Learning Done!")



#Chemical Data for Wine 1

wine_1= [[6.8,0.32,0.16,7,0.045,30,145,0.9949,3.18,0.47,9.6]]



#Chemical Data for Wine 2

wine_2 = [[7.6,1.58,0.0,2.1,0.136,5.0,9.0,0.99476,3.5,0.4,10.9]]



#Chemical Data for Wine 3

wine_3=[[5.2,0.37,0.2,7.6,0.046,35,110,0.9954,3.29,0.58,9.6]]





#We can use the ".predict" command to ask the algorithm to detect an anomaly, if it outputs "1", the wine is normal, if not it will output a "-1"





#We can then create a simple "if/else" condition that will give us the outcome in plain English.

print('for wine 1 :')

if lof.predict(wine_1)==1:

  print("This wine is normal, it passes quality control.")

else:

  print("Abnormal wine detected! Human checking is needed on this one!")

  

print('for wine 2 :')

if lof.predict(wine_2)==1:

  print("This wine is normal, it passes quality control.")

else:

  print("Abnormal wine detected! Human checking needed !")

  

print('for wine 3 :')

if lof.predict(wine_3)==1:

  print("This wine is normal, it passes quality control.")

else:

  print("Abnormal wine detected! Human checking is needed on this one!")

  

  

  

  



#we import the kmeans algorithm from sklearn

kmeans= KMeans(n_clusters=3)



#we use the ".fit" command to use the kmeans algorithm on our data

kmeans.fit(data)



#We create a new column in a data spreadsheet that records for each wine the category it was given

data['category']= kmeans.fit_predict(data)



#prints out the different categories we have and the number of wines that were assigned to it

data['category'].value_counts()
data.head(21)
#We initialize a list of empty lists that will later contain the wines of each category

category_0=[]

category_1=[]

category_2=[]



#this function will sort the first 100 wines of our spreadsheet based on what category they belong to.

for i in range (100):

  if (data.iloc[i]['category'])==0:

    category_0.append(data.index[i])

  if (data.iloc[i]['category'])==1:

    category_1.append(data.index[i])

  if (data.iloc[i]['category'])==2:

    category_2.append(data.index[i])



#Let´s print out the id number numbers belonging to each category.    

print(len(category_0),"wines in category 0:",category_0)



print(len(category_1),"wines in category 1:",category_1)



print(len(category_2),"wines in category 2:",category_2)

                      
#Here we use a short function to convert the categories numbers to plain English labels that we´ll be able to understand.



data['category'] = data['category'].apply(lambda x:"High Price" if x==2 else x)

data['category'] = data['category'].apply(lambda x:"Medium Price" if x==0 else x)

data['category'] = data['category'].apply(lambda x:"Low Price" if x==1 else x)



#We print the top part of our dataset to observe the changes

data.head(21)
#We create an excel file that contains the wine with their new categories

data.to_csv("wines with price categories.csv")



#We use the ".download" command to download the new excel file to our browser

#files.download("wines with price categories.xlsx")
#We can use the ".predict" command for this



#A simple condition to interpret the output in plain english

if kmeans.predict(wine_1)==2:

        print("Wine 1 should be high priced (more than $50)")

if kmeans.predict(wine_1)==0:

        print("Wine 1 should be medium priced ($30-50)")

if kmeans.predict(wine_1)==1:

        print("Wine 1 should be low priced (less than $30)")





if kmeans.predict(wine_3)==2:

        print("Wine 3 should be high priced (more than $50)")

if kmeans.predict(wine_3)==0:

        print("Wine 3 should be medium priced ($30-$50)")

if kmeans.predict(wine_3)==1:

        print("Wine 3 should be low priced (less than $30)")

  

  

#We create a program called "wine_categorizer"



def wine_categorizer():

  

  #The first prompt asks the user whether they have data correctly formatted. If not, they will have to enter it manually.

  

  prompt1=input("Do you have the wine data in the following format:[fixed acidity,volatile acidity,citric acid....]? (yes/no)")

  #if that is the case....

  if prompt1=="yes" or prompt1=="Yes" or prompt1=="y" or prompt1=="YES":

    #...we ask the user to simply copy and paste the line of data

    print("ok great! just copy and paste the data below")

    

    inputted_data=(input(":"))

    #This variable changes the user´s input from a string to a numerical list, that we can compute it

    formatted_data=[list(map(float,inputted_data.split(',')))]

  #if not we get the user to input the data manually, one variable at a time

  else:

    

    print("Ok, no problem, let´s do it manually:")

    

    entered_fixed_acidity=float(input("the wine´s fixed acidity:"))

    entered_volatile_acidity=float(input("volatile acidity:"))

    entered_citric_acid=float(input("citric acid:"))

    entered_residual_sugar=float(input("residual sugar:"))

    entered_chlorides=float(input("chloride levels:"))

    entered_free_sulfur_dioxide=float(input("free sulfur dioxide level:"))

    entered_total_sulfur_dioxide=float(input("total sulfur dioxide :"))

    entered_density=float(input("density:"))

    entered_pH=float(input("pH level :"))

    entered_sulphates=float(input("sulphates :"))

    entered_alcohol=float(input("alcohol% :"))

    #formatting the data so it can computed by our algorithms

    formatted_data=[[entered_fixed_acidity,entered_volatile_acidity,entered_citric_acid,entered_residual_sugar,entered_chlorides,entered_free_sulfur_dioxide,entered_total_sulfur_dioxide,entered_density,entered_pH,entered_sulphates,entered_alcohol]]

  #perform anomaly detection on the entered data and save it the variable "anomaly_check"

  anomaly_check=lof.predict(formatted_data)

  

  #if the anomaly check returns a 1, our data is not an anomaly

  if anomaly_check==1:

    print("This wine is normal, it passes quality control.")

    

     #if the anomaly check returns a -1, our data is an anomaly



  else:

    print("Abnormal wine detected! Human checking needed !")

  #If the wine is an anomaly, then we terminate the program early ( no need to proceed to the price categorization.)

  if anomaly_check==-1:

  #Asking the user whether they want to check a new wine. If the answer is "Yes", the program restarts  

    prompt2=input("Would you like to restart program to check another wine?(Yes/No)")

    if prompt2=="yes" or prompt2=="Yes" or prompt2=="y" or prompt2=="YES":

      wine_categorizer()

    else:

      quit()

 #If the wine is deemed normal by the wine, the program moves onto the price category algorithm 

  if anomaly_check==1:

      print("We will now proceed to the price categorization")

      

      #we use the ".predict" to what the price category the inputted wine would be

      price_category_check=kmeans.predict(formatted_data)

      if price_category_check==2:

        print("This wine should be high priced (more than $50)")

      if price_category_check==0:

        print("This wine should be medium priced ($30-50)")

      if price_category_check==1:

        print("This wine should be low priced (less than $30)")

      prompt2=input("Would you like to restart program to check another wine?(Yes/No)")

      if prompt2=="yes" or prompt2=="Yes" or prompt2=="y" or prompt2=="YES":

        wine_categorizer()

      else:

        quit()

  

wine_categorizer()