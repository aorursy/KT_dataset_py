# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
traindata = pd.read_csv("../input/train_revised.csv")

traindata.head()
traindata["travel_date"] = pd.to_datetime(traindata["travel_date"],infer_datetime_format=True)

traindata["travel_date"] = traindata["travel_date"].dt.dayofweek

traindata["travel_date"].head()
sunday_data = traindata[traindata['travel_date']==6]

sunday_data.head(10)

sunday_data["travel_from"].value_counts().head(7)
#the number of people travelling from Kijauri 

kijauri_data= traindata[traindata["travel_from"]=="Kijauri"]

kijauri_data
#getting the number of people travelling from Kijauri

kijauri_travel_730 = kijauri_data[kijauri_data["travel_time"]<"7:30"]

total_kijauri_travel=len(kijauri_travel_730.index)

print("The Total number of People travelling from Kijauri before 7.30 is :" ,total_kijauri_travel)



kijauri_shuttle = kijauri_travel_730[kijauri_travel_730["car_type"]=="shuttle"]

total_kijauri_travel_shuttle=len(kijauri_shuttle.index)

print("The Total number of People travelling from Kijauri via Shuttle before 7.30 is :" ,total_kijauri_travel_shuttle)



probability_shuttle_travel = (total_kijauri_travel_shuttle/total_kijauri_travel)*100

print("The Probability of People travelling from Kijauri before 7.30 using shuttle is :" ,probability_shuttle_travel,"Percent")
#getting the number of people travelling in kisii

kisii_terminus= traindata[traindata["travel_from"]=="Kisii"]

kisii_terminus
# calculating people who pay via mobile money in kisii  terminus

kisii_terminus_mpesa=kisii_terminus[kisii_terminus["payment_method"]=="Mpesa"]

mpesa_in_kisii=len(kisii_terminus_mpesa.index)

print("Number of people using mpesa payment: ",mpesa_in_kisii)



# calculating people who pay viacash in kisii  terminus

kisii_terminus_cash=kisii_terminus[kisii_terminus["payment_method"]=="Cash"]

cash_in_kisii=len(kisii_terminus_cash.index)

print("Number of people using cash payment: ",cash_in_kisii)



#obtaining the number of people who made payment in kisii terminus

print("Total number of people who travelled from kisii: ",cash_in_kisii+mpesa_in_kisii)



#obtaining the percentage of people who made payment in kisii terminus via mpesa

percentage_of_people_using_mpesa = (mpesa_in_kisii/(mpesa_in_kisii+cash_in_kisii))*100

print("The Percentage of People using Mpesa in Kisii: ",percentage_of_people_using_mpesa)



#obtaining the percentage of people who made payment in kisii terminus via mpesa

percentage_of_people_using_cash = (cash_in_kisii/(mpesa_in_kisii+cash_in_kisii))*100

print("The Percentage of People using Cash in Kisii: ",percentage_of_people_using_cash)
#obtaining data that only has MK in receipt payment.

mk_data = traindata[traindata['payment_receipt'].str.contains("MK")]

mk_data
#differentiating the column for receipt payment

mk_receipt_payament = mk_data["payment_receipt"]

mk_receipt_payament
#Converting the list into a string through the function convert the list



def convert(list): 

      

    # Converting integer list to string list 

    s = [str(i) for i in list] 

      

    # Join list items using join() 

    res = str("".join(s)) 

      

    return(res) 

  

# calling the function to convert the list to string

list = mk_receipt_payament

mk_receipt_payament_string =(convert(list))

mk_receipt_payament_string

#checking for the data type to check if conversion was successful

type(mk_receipt_payament_string)
#Spliting the character in string and then convering them to strings

def split(word): 

    return [char for char in word]  

      

# Driver code 

word = mk_receipt_payament_string



mk_list_receipt_payment=(split(word))

mk_list_receipt_payment

#creating triagrams from NLTK library

import nltk

from nltk import trigrams

mk_trigrams=(trigrams(mk_list_receipt_payment))



#finding the most appearing trigram from the dataset using counter from collections

from collections import Counter

data = Counter(mk_trigrams)

data.most_common()   # Returns all unique items and their counts


