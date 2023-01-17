import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import nltk

import matplotlib.pyplot as plt

from matplotlib import pyplot

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
Data=pd.read_csv("../input/Amazon_Unlocked_Mobile.csv")

Data=Data.dropna(axis=0)
def ReviewLength(Dat):

    

    Dat.dropna(axis=0)

    Price=np.asarray(Dat['Price'])

    Review_Length=np.asarray(Dat['Reviews'].str.len())

    return(Review_Length,Price)
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



def Phone_Stat(Dataframe):



    Phones={}



    for i,j in Dataframe.iterrows():

    

          if(j['Product Name'] in Phones):

        

                  Phones[j['Product Name']].append(j['Rating'])

        

          else:

        

                  Phones[j['Product Name']]=[j['Rating']]

        

 

    Mean=[]

    Product=[]

    SD=[]

    N=[]



    for i in Phones:



        Mean.append(np.asarray(Phones[i]).mean())

        Product.append(i)

        SD.append(np.asarray(Phones[i]).std())

        N.append(len(Phones[i]))

        

    Phone_Stat={'Product':Product,

                 'Mean':Mean,

                 'SD':SD,

                 'Number':N}



    Phone_Stat=pd.DataFrame(Phone_Stat)

    Phone_Stat.sort_values(['Mean','Number'],ascending=False)

    

    return Phone_Stat
def Word_Freq(Data):

    

     Words={}

     for i in Data['Reviews']:

            Word_List=word_tokenize(i)

            for j in Word_List:

                    if(j in Words):

                             Words[j]+=1

                    else:

            

                             Words[j]=1

            

     Keys=[] 

     Values=[]

     

     Custom=[]

     stop_words=set(stopwords.words("english"))

     Custom.append("phone")

     Custom.append("The")



     for i in Words:

    

                if(i not in stop_words and i.isalpha() and i not in Custom):

        

                        Keys.append(i)

                        Values.append(Words[i])



     Word_Stat={'Word':Keys,'Count':Values}

     Word_Stat=pd.DataFrame(Word_Stat) 

     Word_Stat=Word_Stat.sort_values(['Count'],ascending=False)

        

     return(Word_Stat[1:30])

 
def Tokenize_n_Filter(Data,Word):

    

    New_Data=[]

    for i in Data['Reviews']:

        tokens=nltk.word_tokenize(i)

        for j in range(0,len(tokens)):

            

            if(tokens[j]==Word):

                

                New_Data.append(i)

                j=len(tokens)

    Data={'Reviews':New_Data}

    New_Data=pd.DataFrame(Data)

    return(New_Data)

            
Data.sample(frac=0.1).head(n=7)
Data.describe()
fig, ax =plt.subplots(1,2)

sns.kdeplot(Data['Rating'],shade=True,color="yellow",ax=ax[0])

sns.kdeplot(Data['Price'],shade=True,color="blue",ax=ax[1])

fig.show()
Data['Product Name'].value_counts().head(n=10)
Expensive=Data[Data['Price']>250]

N_Expensive=Data[Data['Price']<250]
len(Expensive)
len(N_Expensive)
(len(Expensive)/float(len(Expensive)+len(N_Expensive)))*100
sns.kdeplot(Expensive['Rating'],shade=True,color="red")
sns.kdeplot(N_Expensive['Rating'],shade=True,color="green")
sns.kdeplot(Data['Review Votes'],shade=True,color="pink")
sns.kdeplot(Data['Reviews'].str.len(),shade=True,color="pink")
sns.regplot(x='Price',y='Rating',data=Data)
sns.regplot(x='Price',y='Review Votes',data=Data)
Review_Length,Price=ReviewLength(Data)

sns.regplot(x=Price,y=Review_Length)
print(Review_Length.mean())
sns.kdeplot(Data['Review Votes'],shade=True,color="pink")
Top_B=Data['Brand Name'].value_counts().head(n=5).index.tolist()

print(Data['Brand Name'].value_counts().head(n=10))
Length=(Data['Brand Name'].value_counts().sum())

print((Data['Brand Name'].value_counts().head(n=10).sum())/(Length)*100)
new_Df=pd.DataFrame()



Phones_B=[]



for i in Data['Brand Name'].value_counts().head(n=10).index.tolist():

    

    Phones_B.append(i)

    

for j in Phones_B:

  

    new_Df=new_Df.append(Data[Data['Brand Name']==j])
new_Df.head(n=5)
fig,ax=plt.subplots(figsize=(15,10))

sns.boxplot(x="Brand Name",y="Rating",data=new_Df,ax=ax)
fig,ax=plt.subplots(figsize=(15,10))

sns.boxplot(x="Brand Name",y="Price",data=new_Df,ax=ax)
Data_RL,Data_P=ReviewLength(new_Df)

sns.boxplot(x="Brand Name",y=Data_RL,data=new_Df)
Samsung=Data[Data['Brand Name']=='Samsung']
sns.kdeplot(Data[Data['Brand Name']=='Samsung']['Rating'],shade=True,color="orange")
sns.kdeplot(Data[Data['Brand Name']=='Samsung']['Price'],shade=True,color="blue")
print(Samsung['Product Name'].value_counts().head(n=10))
print(((Samsung['Product Name'].value_counts().head(n=10).sum())/len(Samsung))*100)
Samsung_Phones=Samsung['Product Name']
S_Phone_Stat=Phone_Stat(Samsung)

four=S_Phone_Stat[S_Phone_Stat['Number']>800]

  

plt.figure(figsize=(12,10))    

for i in four.iterrows():



    plt.text(i[1]['Number'],i[1]['Mean'],i[1]['Product'],fontsize=10)

    print(i[1]['Product'])

     

plt.scatter('Number','Mean',data=four)

plt.show()
sns.barplot(x="Count",y="Word",data=Word_Freq(Samsung),color="b")
sns.barplot(x="Count",y="Word",data=Word_Freq(Samsung[Samsung['Rating']<3]),color="r")
sns.barplot(x="Count",y="Word",data=Word_Freq(Samsung[Samsung['Rating']>3]),color="g")
Samsung_RL,Samsung_P=ReviewLength(Samsung)

sns.regplot(x=Samsung_P,y=Samsung_RL)
Apple=Data[Data['Brand Name']=='Apple']
print(Apple['Product Name'].value_counts().head(n=10))
print(((Apple['Product Name'].value_counts().head(n=10).sum())/len(Apple))*100)
sns.kdeplot(Data[Data['Brand Name']=='Apple']['Rating'],shade=True,color="orange")
sns.kdeplot(Data[Data['Brand Name']=='Apple']['Price'],shade=True,color="blue")
A_Phone_Stat=Phone_Stat(Apple)

four_A=A_Phone_Stat[A_Phone_Stat['Number']>600]

  

plt.figure(figsize=(12,10))    

for i in four_A.iterrows():



    plt.text(i[1]['Number'],i[1]['Mean'],i[1]['Product'],fontsize=10)

     

plt.scatter('Number','Mean',data=four_A)

plt.show()
sns.barplot(x="Count",y="Word",data=Word_Freq(Apple),color="b")
sns.barplot(x="Count",y="Word",data=Word_Freq(Apple[Apple['Rating']<3]),color="r")
sns.barplot(x="Count",y="Word",data=Word_Freq(Apple[Apple['Rating']>3]),color="g")
Apple_RL,Apple_P=ReviewLength(Apple)

sns.regplot(x=Apple_P,y=Apple_RL)
HTC=Data[Data['Brand Name']=='HTC']
sns.kdeplot(Data[Data['Brand Name']=='HTC']['Rating'],shade=True,color="orange")
sns.kdeplot(Data[Data['Brand Name']=='HTC']['Price'],shade=True,color="blue")
print(HTC['Product Name'].value_counts().head(n=10))
H_Phone_Stat=Phone_Stat(HTC)

four_H=H_Phone_Stat[H_Phone_Stat['Number']>400]

  

plt.figure(figsize=(12,10))    

for i in four_H.iterrows():



    plt.text(i[1]['Number'],i[1]['Mean'],i[1]['Product'],fontsize=11)

     

plt.scatter('Number','Mean',data=four_H)

plt.show()
print(((HTC['Product Name'].value_counts().head(n=10).sum())/len(HTC))*100)
sns.barplot(x="Count",y="Word",data=Word_Freq(HTC),color="b")
sns.barplot(x="Count",y="Word",data=Word_Freq(HTC[HTC['Rating']<3]),color="r")
sns.barplot(x="Count",y="Word",data=Word_Freq(HTC[HTC['Rating']>3]),color="g")
HTC_RL,HTC_P=ReviewLength(HTC)

sns.regplot(x=HTC_P,y=HTC_RL)
CNPGD=Data[Data['Brand Name']=='CNPGD']
sns.kdeplot(Data[Data['Brand Name']=='CNPGD']['Rating'],shade=True,color="orange")
sns.kdeplot(Data[Data['Brand Name']=='CNPGD']['Price'],shade=True,color="blue")
print(CNPGD['Product Name'].value_counts().head(n=10))
print(((CNPGD['Product Name'].value_counts().head(n=10).sum())/len(CNPGD))*100)
sns.barplot(x="Count",y="Word",data=Word_Freq(CNPGD)[1:20],color="b")
sns.barplot(x="Count",y="Word",data=Word_Freq(CNPGD[CNPGD['Rating']>3])[1:20],color="g")
sns.barplot(x="Count",y="Word",data=Word_Freq(CNPGD[CNPGD['Rating']<3])[1:20],color="r")
CNPGD_RL,CNPGD_P=ReviewLength(HTC)

sns.regplot(x=CNPGD_P,y=CNPGD_RL)
CNP_Phone_Stat=Phone_Stat(CNPGD)

four_CNP=CNP_Phone_Stat[CNP_Phone_Stat['Number']>400]

  

plt.figure(figsize=(12,10))    

for i in four_CNP.iterrows():



    plt.text(i[1]['Number'],i[1]['Mean'],i[1]['Product'],fontsize=11)

     

plt.scatter('Number','Mean',data=four_CNP)

plt.show()
OtterBox=Data[Data['Brand Name']=='OtterBox']
sns.kdeplot(Data[Data['Brand Name']=='OtterBox']['Rating'],shade=True,color="orange")
sns.kdeplot(Data[Data['Brand Name']=='OtterBox']['Price'],shade=True,color="blue")
print(OtterBox['Product Name'].value_counts().head(n=10))
print(((OtterBox['Product Name'].value_counts().head(n=10).sum())/len(OtterBox))*100)
sns.barplot(x="Count",y="Word",data=Word_Freq(OtterBox),color="b")
sns.barplot(x="Count",y="Word",data=Word_Freq(OtterBox[OtterBox['Rating']>3]),color="g")
sns.barplot(x="Count",y="Word",data=Word_Freq(OtterBox[OtterBox['Rating']<3]),color="r")
HTC_RL,HTC_P=ReviewLength(HTC)

sns.regplot(x=HTC_P,y=HTC_RL)
OT_Phone_Stat=Phone_Stat(OtterBox)

four_OT=OT_Phone_Stat[OT_Phone_Stat['Number']>400]

  

plt.figure(figsize=(12,10))    

for i in four_OT.iterrows():



    plt.text(i[1]['Number'],i[1]['Mean'],i[1]['Product'],fontsize=11)

     

plt.scatter('Number','Mean',data=four_OT)

plt.show()