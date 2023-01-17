# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#!pip install package_name

import os

import time

import datetime

from random import randint

import pandas as pd

import numpy as np

import matplotlib.pyplot as pp

import seaborn as sns

from PIL import Image

from wordcloud import WordCloud, ImageColorGenerator

pd.set_option('display.max_columns', 300)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



multiple_choice = pd.read_csv(

    "/kaggle/input/kaggle-survey-2019//multiple_choice_responses.csv",      # relative python path to subdirectory

    sep=',',           # Tab-separated value file.

    header=[0],

    na_values=''      # Take any '.' or '??' values as NA

)

MC = multiple_choice.drop(multiple_choice.index[0])

other_text = pd.read_csv(

    "/kaggle/input/kaggle-survey-2019//other_text_responses.csv",      # relative python path to subdirectory

    sep=',',           # Tab-separated value file.

    header=[0],

    dtype={"Time from Start to Finish (seconds)": int},             # Parse the salary column as an integer 

    na_values=''      # Take any '.' or '??' values as NA

)

OT = other_text.drop(other_text.index[0])

questions_only = pd.read_csv(

    "/kaggle/input/kaggle-survey-2019//questions_only.csv",      # relative python path to subdirectory

    sep=',',           # Tab-separated value file.

    na_values=''      # Take any '.' or '??' values as NA

)



survey_schema = pd.read_csv(

    "/kaggle/input/kaggle-survey-2019//survey_schema.csv",      # relative python path to subdirectory

    sep=',',           # Tab-separated value file.

    header=[0,1],

    na_values=''      # Take any '.' or '??' values as NA

)



MC["Time from Start to Finish (seconds)"]=pd.to_numeric(MC["Time from Start to Finish (seconds)"])



#Functions

Countries = ["Argentina", "Brazil", "Chile", "Colombia", "Mexico", "Peru"]



def LatamCountries(C):

    SubSet=set(Countries)

    if C in SubSet:

        Output=1

    else:

        Output=0

    return (Output)



def Grap(data,lev=1,stacked=True, Col={"Female": "F", "Male": "M","Prefer not to say": "not to say","Prefer to self-describe": "self-describe","Respondents": "Respondents","sum": "Total"},kind='bar',sub=False,color="rb"):

    table = data

    a = table.unstack(level=lev).rename(columns=Col)

    a.plot(kind=kind,figsize=(12, 12),stacked=stacked,subplots=sub,color=color)

    

#The TransQX functions are meant to rearrange the question choices so that they appear in a coherent order in the charts.



def TransQ1(Text):

    if Text == '70+':

        return 70

    Output = str(Text).split("-")

    b = randint(int(Output[0]), int(Output[1]))

    return b





def TransQ6(Text):

    if Text=="0-49 employees":

        return 1

    if Text=="1000-9,999 employees":

        return 4

    if Text=="250-999 employees":

        return 3

    if Text=="50-249 employees":

        return 2

    if Text=="> 10,000 employees":

        return 5



    

def TransQ10(Text):

    if Text == "$0-999":

        return 1

    if Text == "1,000-1,999":

        return 2

    if Text == "10,000-14,999":

        return 8

    if Text == "100,000-124,999":

        return 19

    if Text == "15,000-19,999":

        return 9

    if Text == "2,000-2,999":

        return 3

    if Text == "20,000-24,999":

        return 10

    if Text == "25,000-29,999":

        return 11

    if Text == "3,000-3,999":

        return 4

    if Text == "30,000-39,999":

        return 12

    if Text == "4,000-4,999":

        return 5

    if Text == "40,000-49,999":

        return 13

    if Text == "5,000-7,499":

        return 6

    if Text == "50,000-59,999":

        return 14

    if Text == "60,000-69,999":

        return 15

    if Text == "7,500-9,999":

        return 7

    if Text == "70,000-79,999":

        return 16

    if Text == "80,000-89,999":

        return 17

    if Text == "90,000-99,999":

        return 18

    if Text == "125,000-149,999":

        return 20

    if Text == "150,000-199,999":

        return 21

    if Text == "200,000-249,999":

        return 22

    if Text == "250,000-299,999":

        return 23

    if Text == "300,000-500,000":

        return 24

    if Text == "> $500,000":

        return 25





def TransQ4(Text):

    if Text == "Professional degree":

        return "c"

    if Text == "Bachelor’s degree":

        return "d"

    if Text == "Master’s degree":

        return "e"

    if Text == "Doctoral degree":

        return "f"

    if Text == "I prefer not to answer":

        return "g"

    if Text == "No formal education past high school":

        return "a"

    if Text == "Some college/university study without earning a bachelor’s degree":

        return "b"





def TransQ11(Text):

    if Text == "$0 (USD)":   

        return 1

    if Text == "$1-$99":

        return 2

    if Text == "$10,000-$99,999":

        return 5

    if Text == "$100-$999":

        return 3

    if Text == "$1000-$9,999":

        return 4

    if Text == "> $100,000 ($USD)":

        return 6

    

def TransQ15(Text):    

    if Text == "1-2 years":

        return 3

    if Text == "10-20 years":

        return 6

    if Text == "20+ years":

        return 7

    if Text == "3-5 years":

        return 4

    if Text == "5-10 years":

        return 5

    if Text == "< 1 years":

        return 2

    if Text == "I have never written code":

        return 1



    

def TransQ22(Text): 

    if Text == "2-5 times":

        return 3

    if Text == "6-24 times":

        return 4

    if Text == "> 25 times":

        return 5

    if Text == "Never":

        return 1

    if Text == "Once":

        return 2

    

    

def TransQ23(Text):

    if Text == "1-2 years":

        return 2

    if Text == "10-15 years":

        return 7

    if Text == "2-3 years":

        return 3

    if Text == "20+ years":

        return 9

    if Text == "15-20 years":

        return 8  

    if Text == "3-4 years":

        return 4

    if Text == "4-5 years":

        return 5

    if Text == "5-10 years":

        return 6

    if Text == "< 1 years":

        return 1



    

def Cut(Text): 

    return Text.split("(")[0]

    



def TransQText(temp, noun):

    for t in temp:

        Final1[t] = Final1[t].str.split("(", n=1, expand=True) 

    Final1[noun] = Final1[temp].apply(lambda x:'; '.join(x.astype(str).replace("nan","")), axis=1)



    """

    This funtion change the database 

    """



def Clean(text):

    while text.count("; ; ")>0:

        text = text.replace("; ; ","; ")

    while text.find("  ")>0:

        text = text.replace("  "," ")

    if text[0] == ";":

        text=text[1:]

    try:

        if  text[-2] == ";":

            text = text[:-2]

        if  text[-1] == ";":

            text = text[:-1] 

    except:

        pass

    return text



def OutPut(Country="Argentina",Question="Q12B"):

    a = Final1[Final1["Q3"]==Country]

    b = ""

    for i in range(len(a[Question])):

        if len(a[Question].iloc[i])>3:

            b = b + " ; " + a[Question].iloc[i]

    b = b.replace("  "," ")

    b = b.split("; ")

    Output=dict()

    for i in b:

        if i.endswith(" "):

            i = i[:-1]

        if i in Output.keys():

            Output[i] += 1

        else:

            Output[i] = 1

    return Output



def PlotWord(Countries=["Argentina", "Brazil", "Chile", "Colombia", "Mexico", "Peru"],Question="Q12B"):

    i=0

    fig = pp.figure(figsize=[20,9])

    for Country in Countries:

        text = OutPut(Country,Question)

        ax = fig.add_subplot(2,3,i+1)

        flag_color = np.array(Image.open("/kaggle/input/images/B%s.jpg"%(Country[:3])))

        flag_color = flag_color[::3, ::3]

        flag_mask = flag_color.copy()

        flag_mask[flag_mask.sum(axis=2) == 0] = 255

        wc = WordCloud( mask=flag_mask,width=500,max_words=5000,relative_scaling=0)

        wc.generate_from_frequencies(text)

        image_colors = ImageColorGenerator(flag_color,default_color=[255,255,255])

        wc.recolor(color_func=image_colors)

        ax.set_title(Country)

        ax.imshow(wc, interpolation="bilinear")

        ax.axis("off")

        i+=1

SubMC=MC[["Time from Start to Finish (seconds)","Q1",

                 "Q2",#"Q2_OTHER_TEXT",

                 "Q3",

                 "Q4",

                 "Q5",#"Q5_OTHER_TEXT",

                 "Q6",

                 "Q10",

                 "Q11",

                 "Q12_Part_1","Q12_Part_2","Q12_Part_3","Q12_Part_4","Q12_Part_5","Q12_Part_6","Q12_Part_7","Q12_Part_8","Q12_Part_9","Q12_Part_10","Q12_Part_11","Q12_Part_12",#"Q12_OTHER_TEXT",              

                 "Q13_Part_1","Q13_Part_2","Q13_Part_3","Q13_Part_4","Q13_Part_5","Q13_Part_6","Q13_Part_7","Q13_Part_8","Q13_Part_9","Q13_Part_10","Q13_Part_11","Q13_Part_12",#"Q13_OTHER_TEXT",

                 "Q14","Q14_Part_1_TEXT","Q14_Part_2_TEXT","Q14_Part_3_TEXT","Q14_Part_4_TEXT","Q14_Part_5_TEXT",#"Q14_OTHER_TEXT",

                 "Q15",

                 "Q16_Part_1","Q16_Part_2","Q16_Part_3","Q16_Part_4","Q16_Part_5","Q16_Part_6","Q16_Part_7","Q16_Part_8","Q16_Part_9","Q16_Part_10","Q16_Part_11","Q16_Part_12",#"Q16_OTHER_TEXT",

                 "Q17_Part_1","Q17_Part_2","Q17_Part_3","Q17_Part_4","Q17_Part_5","Q17_Part_6","Q17_Part_7","Q17_Part_8","Q17_Part_9","Q17_Part_10","Q17_Part_11","Q17_Part_12",#"Q17_OTHER_TEXT",

                 "Q18_Part_1","Q18_Part_2","Q18_Part_3","Q18_Part_4","Q18_Part_5","Q18_Part_6","Q18_Part_7","Q18_Part_8","Q18_Part_9","Q18_Part_10","Q18_Part_11","Q18_Part_12",#"Q18_OTHER_TEXT",

                 "Q21_Part_1","Q21_Part_2","Q21_Part_3","Q21_Part_4","Q21_Part_5",#"Q21_OTHER_TEXT",

                 "Q19",

                 "Q22",

                 "Q23",

                 "Q27_Part_1","Q27_Part_2","Q27_Part_3","Q27_Part_4","Q27_Part_5","Q27_Part_6",#"Q27_OTHER_TEXT",

                 "Q31_Part_1","Q31_Part_2","Q31_Part_3","Q31_Part_4","Q31_Part_5","Q31_Part_6","Q31_Part_7","Q31_Part_8","Q31_Part_9","Q31_Part_10","Q31_Part_11","Q31_Part_12",#"Q31_OTHER_TEXT",

]]



SubOT=OT[["Q2_OTHER_TEXT",

    "Q5_OTHER_TEXT",

    "Q12_OTHER_TEXT",

    "Q13_OTHER_TEXT",

    "Q14_OTHER_TEXT","Q14_Part_1_TEXT","Q14_Part_2_TEXT","Q14_Part_3_TEXT","Q14_Part_4_TEXT","Q14_Part_5_TEXT",

    "Q16_OTHER_TEXT",

    "Q17_OTHER_TEXT",

    "Q18_OTHER_TEXT",

    "Q19_OTHER_TEXT",

    "Q21_OTHER_TEXT",

    "Q27_OTHER_TEXT",

    "Q31_OTHER_TEXT",

]]

temp=["Q2_OTHER_TEXT",

    "Q5_OTHER_TEXT",

    "Q12_OTHER_TEXT",

    "Q13_OTHER_TEXT",

    "Q14_OTHER_TEXT","Q14_Part_1_TEXT","Q14_Part_2_TEXT","Q14_Part_3_TEXT","Q14_Part_4_TEXT","Q14_Part_5_TEXT",

    "Q16_OTHER_TEXT",

    "Q17_OTHER_TEXT",

    "Q18_OTHER_TEXT",

    "Q19_OTHER_TEXT",

    "Q21_OTHER_TEXT",

    "Q27_OTHER_TEXT",

    "Q31_OTHER_TEXT"

     ]

for t in temp:

    t2=t+"_OT"

    SubMC[t2]=SubOT[t]
SubMC["Respondents"]=SubMC["Q3"].apply(LatamCountries)

Respondents=SubMC["Respondents"] == 1

Final1=SubMC[Respondents]
temp = ["Q12_Part_1","Q12_Part_2","Q12_Part_3","Q12_Part_4","Q12_Part_5","Q12_Part_6","Q12_Part_7","Q12_Part_8","Q12_Part_9","Q12_Part_10","Q12_Part_11","Q12_Part_12","Q12_OTHER_TEXT_OT"]

noun = "Q12B"

a = TransQText(temp,noun)

Final1[noun] = Final1[noun].apply(Clean)



temp = ["Q13_Part_1","Q13_Part_2","Q13_Part_3","Q13_Part_4","Q13_Part_5","Q13_Part_6","Q13_Part_7","Q13_Part_8","Q13_Part_9","Q13_Part_10","Q13_Part_11","Q13_Part_12","Q13_OTHER_TEXT_OT"]

noun = "Q13B"

a = TransQText(temp,noun)

Final1[noun] = Final1[noun].apply(Clean)





temp = ["Q14","Q14_OTHER_TEXT_OT","Q14_Part_1_TEXT_OT" ,"Q14_Part_2_TEXT_OT","Q14_Part_3_TEXT_OT","Q14_Part_4_TEXT_OT","Q14_Part_5_TEXT_OT"]

noun = "Q14B"

a = TransQText(temp,noun)

Final1[noun] = Final1[noun].apply(Clean)





temp = ["Q16_Part_1","Q16_Part_2","Q16_Part_3","Q16_Part_4","Q16_Part_5","Q16_Part_6","Q16_Part_7","Q16_Part_8","Q16_Part_9","Q16_Part_10","Q16_Part_11","Q16_Part_12","Q16_OTHER_TEXT_OT"]

noun = "Q16B"

a = TransQText(temp,noun)

Final1[noun] = Final1[noun].apply(Clean)





temp = ["Q17_Part_1","Q17_Part_2","Q17_Part_3","Q17_Part_4","Q17_Part_5","Q17_Part_6","Q17_Part_7","Q17_Part_8","Q17_Part_9","Q17_Part_10","Q17_Part_11","Q17_Part_12","Q17_OTHER_TEXT_OT"]

noun = "Q17B"

a = TransQText(temp,noun)

Final1[noun] = Final1[noun].apply(Clean)





temp = ["Q18_Part_1","Q18_Part_2","Q18_Part_3","Q18_Part_4","Q18_Part_5","Q18_Part_6","Q18_Part_7","Q18_Part_8","Q18_Part_9","Q18_Part_10","Q18_Part_11","Q18_Part_12","Q18_OTHER_TEXT_OT"]

noun = "Q18B"

a = TransQText(temp,noun)

Final1[noun] = Final1[noun].apply(Clean)





temp = ["Q21_Part_1","Q21_Part_2","Q21_Part_3","Q21_Part_4","Q21_Part_5","Q21_OTHER_TEXT_OT"]

noun = "Q21B"

a = TransQText(temp,noun)

Final1[noun] = Final1[noun].apply(Clean)





temp = ["Q19","Q19_OTHER_TEXT_OT"]

noun = "Q19B"

a = TransQText(temp,noun)

Final1[noun] = Final1[noun].apply(Clean)





temp = ["Q27_Part_1","Q27_Part_2","Q27_Part_3","Q27_Part_4","Q27_Part_5","Q27_Part_6","Q27_OTHER_TEXT_OT"]

noun = "Q27B"

a = TransQText(temp,noun)

Final1[noun] = Final1[noun].apply(Clean)





temp=["Q31_Part_1","Q31_Part_2","Q31_Part_3","Q31_Part_4","Q31_Part_5","Q31_Part_6","Q31_Part_7","Q31_Part_8","Q31_Part_9","Q31_Part_10","Q31_Part_11","Q31_Part_12","Q31_OTHER_TEXT_OT"]

noun="Q31B"

a=TransQText(temp,noun)

Final1[noun]=Final1[noun].apply(Clean)







Final1["Q1B"]  = Final1["Q1"].apply(TransQ1)

Final1["Q6B"]  = Final1["Q6"].apply(TransQ6)

Final1["Q10B"] = Final1["Q10"].apply(TransQ10)

Final1["Q11B"] = Final1["Q11"].apply(TransQ11)

Final1["Q4B"]  = Final1["Q4"].apply(TransQ4)

Final1["Q15B"] = Final1["Q15"].apply(TransQ15)

Final1["Q22B"] = Final1["Q22"].apply(TransQ22)

Final1["Q23B"] = Final1["Q23"].apply(TransQ23)
table1 = pd.pivot_table(Final1, values=["Respondents"], index=["Q1","Q2"],aggfunc=[np.sum])

Grap(table1,color= ['#FF0000',  '#0000FF','#00FF00' , '#FF00FF'])
# Count in groups of country and gender

table2 = pd.pivot_table(Final1, values=["Respondents"], index=["Q3","Q2"],aggfunc=[np.sum])

Grap(table2,color= ['#FF0000',  '#0000FF','#00FF00' , '#FF00FF'])
# Count in groups of country and academic level

Col={"c":"Professional degree","d":"Bachelor’s degree","e":"Master’s degree","f":"Doctoral degree","g":"I prefer not to answer","a":"No formal education past high school","b":"Some college/university study without earning a bachelor’s degree"}

color= ['#C0C0C0',  '#808000','#00FF00' , '#008000', "#800080","#000080", "#008080"]

table5 = pd.pivot_table(Final1, values=["Respondents"], index=["Q3","Q4B"],aggfunc=[np.sum])

Grap(table5,Col=Col,color=color)
# Count in groups of country and position

color= ['#C0C0C0',  '#808000','#00FF00' , '#008000', "#800080","#000080", "#008080"]

table6 = pd.pivot_table(Final1, values=["Respondents"], index=["Q5","Q3"],aggfunc=[np.sum])

Grap(table6,color=color)
# Count in groups of country and company size

color= ['#C0C0C0',  '#808000','#00FF00' , '#008000', "#800080","#000080", "#008080"]

order=["0-49 employees","50-249 employees","250-999 employees","1000-9,999 employees","> 10,000 employees"]

table8 = pd.pivot_table(Final1, values=["Respondents"], index=["Q6","Q3"],aggfunc=[np.sum])

Grap(table8,Col=Col,stacked=False,color=color)
# Count in groups of country and compensation

color= ['#C0C0C0',  '#808000','#00FF00' , '#008000', "#800080","#000080", "#008080"]

Col={1:"$0-999",2:"1,000-1,999",8:"10,000-14,999",19:"100,000-124,999",9:"15,000-19,999",3:"2,000-2,999",10:"20,000-24,999",

    11:"25,000-29,999",4:"3,000-3,999",12:"30,000-39,999",5:"4,000-4,999",13:"40,000-49,999",6:"5,000-7,499",

    14:"50,000-59,999",15:"60,000-69,999",7:"7,500-9,999",16:"70,000-79,999",17:"80,000-89,999",18:"90,000-99,999",

    20:"125,000-149,999", 21:"150,000-199,999",22:"200,000-249,999",23:"250,000-299,999",24:"300,000-500,000",25:"> $500,000"}

table9 = pd.pivot_table(Final1, values=["Respondents"], index=["Q10B","Q3"],aggfunc=[np.sum])

Grap(table9,Col=Col,color=color)
# Count in groups of country and expense in ML learning

color= ['#C0C0C0',  '#808000','#00FF00' , '#008000', "#800080","#000080", "#008080"]



Col={1:"$0 (USD)",2:"$1-$99",5:"$10,000-$99,999",3:"$100-$999",4:"$1000-$9,999",6:"> $100,000 ($USD)"}

table10 = pd.pivot_table(Final1, values=["Respondents"], index=["Q3","Q11B"],aggfunc=[np.sum])

Grap(table10,Col=Col,stacked=False,color=color)
Col={3:"1-2 years",6:"10-20 years",7:"20+ years",4:"3-5 years",5:"5-10 years",2:"< 1 years",1:"I have never written code"}

color= ['#C0C0C0',  '#808000','#00FF00' , '#008000', "#800080","#000080", "#008080"]

table12 = pd.pivot_table(Final1, values=["Respondents"], index=["Q3","Q15B"],aggfunc=[np.sum])

Grap(table12,Col=Col,stacked=False,color=color)
Col={2:"1-2 years",7:"10-15 years",3:"2-3 years",9:"20+ years",8:"15-20 years",4:"3-4 years",5:"4-5 years",

     6:"5-10 years",1:"< 1 years"}

color= ['#C0C0C0',  '#808000','#00FF00' , '#008000', "#800080","#000080", "#008080"]



table15 = pd.pivot_table(Final1, values=["Respondents"], index=["Q3","Q23B"],aggfunc=[np.sum])

Grap(table15,Col=Col,stacked=False,color=color)
pp.figure(figsize=(16, 6))

Col = {1:"$0-999",2:"1,000-1,999",8:"10,000-14,999",19:"100,000-124,999",9:"15,000-19,999",3:"2,000-2,999",10:"20,000-24,999",

    11:"25,000-29,999",4:"3,000-3,999",12:"30,000-39,999",5:"4,000-4,999",13:"40,000-49,999",6:"5,000-7,499",

    14:"50,000-59,999",15:"60,000-69,999",7:"7,500-9,999",16:"70,000-79,999",17:"80,000-89,999",18:"90,000-99,999",

    20:"125,000-149,999", 21:"150,000-199,999",22:"200,000-249,999",23:"250,000-299,999",24:"300,000-500,000",25:"> $500,000"}

FinalCompensation = Final1[(Final1["Q2"] =="Male") | (Final1["Q2"] == "Female")]

#(df.subtype == 2)  (df.subtype == 3)

#sns.violinplot(x="Q10", y="Q1B", hue="Q2",Col=Col, data=FinalSueldo, palette="Pastel1",split=True,scale="count")



chart =sns.violinplot(x="Q10", y="Q1B", hue="Q2",Col=Col, data=FinalCompensation, palette=[ '#0000FF','#FF0000'],split=True,order=['$0-999', '1,000-1,999', '2,000-2,999', '3,000-3,999', '4,000-4,999', '5,000-7,499', '7,500-9,999', '10,000-14,999', '15,000-19,999', '20,000-24,999', '25,000-29,999', '30,000-39,999', '40,000-49,999', '50,000-59,999', '60,000-69,999', '70,000-79,999', '80,000-89,999', '90,000-99,999', '100,000-124,999', '125,000-149,999', '150,000-199,999', '200,000-249,999', '250,000-299,999', '300,000-500,000'])

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
PlotWord(Countries=["Argentina", "Brazil", "Chile", "Colombia", "Mexico", "Peru"],Question="Q12B") 
PlotWord(Countries=["Argentina", "Brazil", "Chile", "Colombia", "Mexico", "Peru"],Question="Q13B") 
PlotWord(Countries=["Argentina", "Brazil", "Chile", "Colombia", "Mexico", "Peru"],Question="Q14B") 
PlotWord(Countries=["Argentina", "Brazil", "Chile", "Colombia", "Mexico", "Peru"],Question="Q16B") 
PlotWord(Countries=["Argentina", "Brazil", "Chile", "Colombia", "Mexico", "Peru"],Question="Q17B") 
PlotWord(Countries=["Argentina", "Brazil", "Chile", "Colombia", "Mexico", "Peru"],Question="Q18B") 
PlotWord(Countries=["Argentina", "Brazil", "Chile", "Colombia", "Mexico", "Peru"],Question="Q19B")
PlotWord(Countries=["Argentina", "Brazil", "Chile", "Colombia", "Mexico", "Peru"],Question="Q21B")
PlotWord(Countries=["Argentina", "Brazil", "Chile", "Colombia", "Mexico", "Peru"],Question="Q27B")
PlotWord(Countries=["Argentina", "Brazil", "Chile", "Colombia", "Mexico", "Peru"],Question="Q31B")