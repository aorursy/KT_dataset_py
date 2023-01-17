#Disabling the warnings
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
    
            print(os.path.join(dirname, filename))
import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
import openpyxl

df = pd.read_excel('/kaggle/input/company-types/Data_Science_Internship_Assignment.xlsx' , sheetname="Data")# reading the data

#creating keywords lists
edu_list = ['school','university','academy','training','nursery','tuition','learning','education',
           'institute','institution']
gnp_list = ['not-for-profit','non-profit','non-governmental']


def find_edu (name):
    for keyword in edu_list:
        if keyword in str(name):
            return True
    return False 

def find_gnp (name):
    for keyword in gnp_list:
        if keyword in str(name):
            return True
    return False 

def select_year (date):
    date = str(date)
    date= int(date[:4])
    return date
 
# creating company's columns    
df['university']=df.NAME.str.lower().apply(find_edu)
df['govern_np']=df.NAME.str.lower().apply(find_gnp) | df["TAGLINE"].str.lower().apply(find_gnp) | df["TAGS"].str.lower().apply(find_gnp) | df.WEBSITE.str.contains(".gov")
df["year"]=df["LAUNCH DATE"].apply(select_year)
df["mature_company"]=(df["year"]<1990)& ( ~df['university'])&( ~df['govern_np'])
df["startsup"]=(df["year"]>=1990)& ( ~df['university'])&( ~df['govern_np'])
df["unclassified"]=(~df["mature_company"])&(~df["startsup"])& ( ~df['university'])&( ~df['govern_np'])

# updating "TYPE" column
df["TYPE"]=df[['university','govern_np',"mature_company","startsup","unclassified"] ].idxmax(axis=1)
df_count = df["TYPE"].value_counts()
# returning original dataframe format
df = df.drop(['university','govern_np',"mature_company","startsup","unclassified","year"],axis = 1)

# Load excel file, remove sheets to be updated, overwrite the file
book = openpyxl.load_workbook("Data_Science_Internship_Assignment.xlsx")
book.remove(book['Data'])
book.remove(book['Count'])
book.save("Data_Science_Internship_Assignment.xlsx")

with pd.ExcelWriter("Data_Science_Internship_Assignment.xlsx", engine="openpyxl", mode="a") as writer:
    df.to_excel(writer, sheet_name="Data", startrow=0, startcol=0)
    df_count.to_excel(writer, sheet_name="Count", startrow=0, startcol=0)
    writer.save()

# Plotting bar chart

print("Amount of companies according to their types:\n",df_count)
type_list = df["TYPE"].unique()
plt.title("Amount of companies according to their types")
colors = ['b','g','m','r']
plt.bar(type_list, df_count, color = colors)
plt.show()
    
