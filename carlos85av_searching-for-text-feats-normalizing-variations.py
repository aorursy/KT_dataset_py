import pandas as pd

import numpy as np

import os

import json

import nltk, re, math, collections

from nltk.corpus import stopwords

from nltk.corpus import wordnet

import matplotlib.pylab as plt

import operator

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb 

from datetime import datetime

import matplotlib.pyplot as plt 

import seaborn as sns 

from os import path

from wordcloud import WordCloud

import matplotlib.pyplot as plt

train_v = pd.read_csv('../input/training_variants')

test_v = pd.read_csv('../input/test_variants')

train_t = pd.read_csv('../input/training_text',sep='\|\|',skiprows=1,engine='python',names=["ID","Text"])

test_t = pd.read_csv('../input/test_text',sep='\|\|',skiprows=1,engine='python',names=["ID","Text"])



train = pd.merge(train_v, train_t, how='left', on='ID').fillna('')

y_labels = train['Class'].values



test = pd.merge(test_v, test_t, how='left', on='ID').fillna('')

test_id = test['ID'].values
print("there are ",len(train["Variation"]),"rows for the training set")

print("there are ",len(set(list(train["Variation"]))), " different values for variations")

print("there are ",len(set(list(train["Gene"]))), " different values for genes")
train["Variation"][:50]
def variationProc(variations, genes):

    vari2=[]

    for i in range(0, len(variations)):

        esfusion=False

        texto=variations[i].lower()

        texto = texto.replace(" ","")

        texto = texto.replace("_","#")        

        texto = texto.replace("\'","")

        texto = texto.replace("-","#")

        texto = texto.replace("Exon ","Exon") 

        

        if "truncating" in texto:

            texto="trunc"+"#null"+"#null"+"#null"+"#null"+"#null"

        elif "promotermut" in texto:

            texto="promotermut"+"#null"+"#null"+"#null"+"#null"+"#null"

        elif "promoterhyper" in texto:

            texto="promoterhyper"+"#null"+"#null"+"#null"+"#null"+"#null"

        elif "ampli" in texto:

            texto="ampli"+"#null"+"#null"+"#null"+"#null"+"#null"

        elif "overex" in texto:

            texto="overex"+"#null"+"#null"+"#null"+"#null"+"#null"

        elif "dnabinding" in texto:

            texto="dnabinding"+"#null"+"#null"+"#null"+"#null"+"#null"

        elif "wildtype" in texto:

            texto="wildtype"+"#null"+"#null"+"#null"+"#null"+"#null"

        elif "epigeneticsil" in texto:

            texto="epigeneticsil"+"#null"+"#null"+"#null"+"#null"+"#null"

        elif "copynumberloss" in texto:

            texto="copynumberloss"+"#null"+"#null"+"#null"+"#null"+"#null"

        elif "hypermethyl" in texto:

            texto="hypermethyl"+"#null"+"#null"+"#null"+"#null"+"#null"

        elif "singlenucleotidepolymo" in texto:

            texto="singlenucleotidepolymo"+"#null"+"#null"+"#null"+"#null"+"#null"  

        elif "exon" in texto:

            texto=texto+"#null"+"#null"+"#null"+"#null"+"#null"

        elif "fs" in texto:

            texto=texto.replace("fs","")

            if re.match("(\D+)(\d+)(\D+)", texto):

                if texto[1:5].isnumeric():

                    texto="fs"+"#"+texto[0:1]+"#"+texto[1:5]+"#null"+"#null"+"#null"

                elif texto[1:4].isnumeric():                       

                    texto="fs"+"#"+texto[0:1]+"#"+texto[1:4]+"#null"+"#null"+"#null"

                else:                        

                    texto="fs"+"#"+texto[0:1]+"#"+texto[1:3]+"#null"+"#null"+"#null"

            elif re.match("(\D+)(\d+)", texto):

                if " " not in texto:

                    texto="fs"+"#"+texto[0:1]+"#"+texto[1:]+"#null"+"#null"+"#null"

            else:

                texto="fs"+"#"+texto

        elif "deletion/insertion" in texto:

            texto = texto.replace("deletion/insertion","delins")

        elif "delins" in texto:

            texto=texto.replace("delins","")

            if "#" not in texto:

                if re.match("(\D+)(\d+)(\D+)", texto):

                    texto="delins"+"#"+texto[0:1]+"#"+texto[1:4]+"#"+texto[4:4]+"#null"+"#null"+"#null"

                elif re.match("(\D+)(\d+)", texto):

                    texto="delins"+"#"+texto[0:1]+"#"+texto[1:]+"#null"+"#null"+"#null"

                else:

                    texto="delins"+"#null"+"#null"+"#null"+"#null"+"#null"

            else:

                lista=texto.split("#")

                if re.match("(\D+)(\d+)", lista[0]):

                    texto="delins"+"#"+lista[0][0:1]+"#"+lista[0][1:]+"#"

                if re.match("(\D+)(\d+)", lista[1]):

                    texto=texto+lista[1][0:1]+"#"+lista[1][1:4]+"#"+lista[1][4:5]

        elif "fusion" in texto:

            esfusion=True

            texto = texto.replace("fusion","")

            lista=texto.split("#")

            if len(lista)==2:

                if genes[i].lower() in lista[0].lower():

                    texto = lista[1].lower()+"#fusion" +"#null"+"#null"+"#null"+"#null"+"#null"

                else:

                    texto = lista[0].lower()+"#fusion"+"#null"+"#null"+"#null"+"#null"+"#null"

            else:

                texto = genes[i].lower()+"#fusion"+"#null"+"#null"+"#null"+"#null"+"#null"

        elif "deletion" in texto:

            texto = texto.replace("deletion","del")

            texto = texto.replace("3del","del")

        elif "del" in texto:

            texto=texto.replace("del","")

            if "#" not in texto:

                if re.match("(\D+)(\d+)", texto):

                    texto="del"+"#"+texto[0:1]+"#"+texto[1:]

            else:

                lista=texto.split("#")

                texto="del"

                if re.match("(\D+)(\d+)", lista[0]):

                    texto=texto+"#"+lista[0][0:1]+"#"+lista[0][1:]+"#"

                if re.match("(\D+)(\d+)", lista[1]):

                    texto=texto+lista[1][0:1]+"#"+lista[1][1:]+"#"   

                if re.match("(\d+)(\d+)", lista[0]):

                    texto=texto+"#null#"+lista[0] 

                if re.match("(\d+)(\d+)", lista[1]):

                    texto=texto+"#null#"+lista[1]

        elif "insertion" in texto:

            texto = texto.replace("insertion","ins")

        elif "ins" in texto:

            texto=texto.replace("ins","")

            if "#" not in texto:

                if re.match("(\D+)(\d+)(\D+)", texto):

                    if "#" not in texto:

                        texto="ins"+"#"+texto[0:1]+"#"+texto[1:4]+"#"+texto[4:4]

                elif re.match("(\D+)(\d+)", texto):

                    texto="ins"+"#"+texto[0:1]+"#"+texto[1:]

            else:

                lista=texto.split("#")

                texto="ins"

                if re.match("(\D+)(\d+)", lista[0]):

                    texto=texto+"#"+lista[0][0:1]+"#"+lista[0][1:]+"#"

                if re.match("(\D+)(\d+)", lista[1]):

                    texto=texto+lista[1][0:1]+"#"+lista[1][1:4]+"#"+lista[1][4:5]

                if re.match("(\d+)(\d+)", lista[0]):

                    texto=texto+"#null#"+lista[0] 

                if re.match("(\d+)(\d+)", lista[1]):

                    texto=texto+"#null#"+lista[1]

        elif "dup" in texto:

            texto=texto.replace("dup","")

            if " " not in texto:

                if texto[1:].isnumeric():

                    if re.match("(\D+)(\d+)(\D+)", texto):

                        if " " not in texto:

                            texto="dup"+"#"+texto[0:1]+"#"+texto[1:4]+"#"+texto[4:4]

                    elif re.match("(\D+)(\d+)", texto):

                        texto="dup"+"#"+texto[0:1]+"#"+texto[1:]

                else:

                    texto="dup"

            else:

                lista=texto.split("#")

                if re.match("(\D+)(\d+)", lista[0]):

                    texto="dup"+"#"+lista[0][0:1]+"#"+lista[0][1:]+"#"

                if re.match("(\D+)(\d+)", lista[1]):

                    texto=texto+lista[1][0:1]+"#"+lista[1][1:4]+"#"+lista[1][4:5]

        elif "splice" in texto:

            texto=texto.replace("splice","")

            if "#" not in texto:

                if texto[1:].isnumeric():

                    if re.match("(\d+)", texto):

                        texto="splice"+"#null#"+texto+"#null"+"#null"+"#null"

                    elif re.match("(\D+)(\d+)", texto):

                        texto="splice"+"#null#"+texto[1:]+"#null"+"#null"+"#null"

                else:

                    texto="splice"+"#null"+"#null"+"#null"+"#null"+"#null"

            else:

                lista=texto.split("#")

                if re.match("(\D+)(\d+)", lista[0]):

                    texto="splice"+"#null#"+lista[0][1:]

                else:

                    texto="splice"+"#null#"+lista[0]

                    if re.match("(\D+)(\d+)", lista[1]):

                        texto=texto+"#null#"+lista[1][1:]+"#null"

                    else:

                        texto=texto+"#null#"+lista[1]

        elif re.match("(\D)(\d+)(\D+)", texto):

            if " " not in texto:

                texto="sub"+"#"+texto[0:1]+"#"+texto[1:len(texto)-1]+"#null"+"#null"+"#"+texto[len(texto)-1:]

        elif re.match("(\D)(\d+)", texto):

            if " " not in texto:

                texto="sub"+"#"+texto[0:1]+"#"+texto[1:]+"#null"+"#null"+"#null"

        else:

            texto="others"+"#null"+"#null"+"#null"+"#null"+"#null"

        

        if esfusion:

            vari2.append(genes[i].lower()+"#"+texto)

        else:

            vari2.append(genes[i].lower()+"#null#"+texto)

            

    mat=[]   

    for linea in vari2:

        linea.replace(" ","")

        linea.replace("##","#")

        lista=linea.split("#")

        lineaadd=[]

        if len(lista)>4:

            lista[4] = re.sub("\D", "", lista[4])

            if lista[4]=="":

                lista[4]=np.nan

            else:

                float(lista[4])

        if len(lista)>6:

            lista[6] = re.sub("\D", "", lista[6])

            if lista[6]=="":

                lista[6]=np.nan

            else:

                float(lista[6])

        for te in range(0,len(lista)):

            if lista[te] == "null" or lista[te] == "":

                lista[te]=None

                if te==4 or te==6:

                    lista[te]=np.nan

        if len(lista)<8:

            for j in range(len(lista),9):

                if j==4 or j==6:

                    lista.append(np.nan)

                else:

                    lista.append(None)

        

        for j in range(0,8):

            lineaadd.append(lista[j])

        mat.append(lineaadd)

        

    print("Done...")

    return(mat)

print("Processing gene and variation with VariationProc...")

vartra=variationProc(train["Variation"], train["Gene"])
vardf = pd.DataFrame(vartra, columns=["gene1","gene2","operation","letter1","number1","letter2","number2","objletter"])

vardf['Class'] = train["Class"]
vardf
plt.figure(figsize=(11,7))

sns.countplot(x="Class", data=train)

plt.ylabel('Frequency', fontsize=13)

plt.xlabel('Classes', fontsize=13)

plt.title("Frequency of Classes", fontsize=18)

plt.show()

plt.figure(figsize=(11,57))

sns.countplot(y="gene1", data=vardf)

plt.ylabel('Gene1', fontsize=13)

plt.xlabel('Frequency', fontsize=13)

plt.title("Frequency of Feature Gene1", fontsize=18)

plt.show()
plt.figure(figsize=(10,50))

sns.stripplot(x="Class", y="gene1", data=vardf, jitter=True);

plt.show()
plt.figure(figsize=(10,30))

sns.countplot(y="gene2", data=vardf)

plt.ylabel('Gene2', fontsize=13)

plt.xlabel('Frequency', fontsize=13)

plt.title("Frequency of Feature Gene2", fontsize=18)

plt.show()
plt.figure(figsize=(10,30))

sns.stripplot(x="Class", y="gene2", data=vardf, jitter=True);

plt.show()
plt.figure(figsize=(11,7))

sns.countplot(y="operation", data=vardf)

plt.ylabel('Operation', fontsize=13)

plt.xlabel('Frequency', fontsize=13)

plt.title("Frequency of Feature Operation", fontsize=18)

plt.show()
plt.figure(figsize=(10,10))

sns.stripplot(x="Class", y="operation", data=vardf, jitter=True);

plt.show()
plt.figure(figsize=(11,7))

sns.countplot(x="letter1", data=vardf)

plt.ylabel('Frequency', fontsize=13)

plt.xlabel('Letter1', fontsize=13)

plt.title("Frequency of Feature Letter1", fontsize=18)

plt.show()
plt.figure(figsize=(10,10))

sns.stripplot(x="Class", y="letter1", data=vardf, jitter=True);

plt.show()
plt.figure(figsize=(10,10))

sns.stripplot(x="Class", y="number1", data=vardf, jitter=True);

plt.show()
plt.figure(figsize=(11,7))

sns.countplot(x="letter2", data=vardf)

plt.ylabel('Frequency', fontsize=13)

plt.xlabel('Letter2', fontsize=13)

plt.title("Frequency of Feature Letter2", fontsize=18)

plt.show()
plt.figure(figsize=(10,10))

sns.stripplot(x="Class", y="letter2", data=vardf, jitter=True);

plt.show()
plt.figure(figsize=(10,10))

sns.stripplot(x="Class", y="number2", data=vardf, jitter=True);

plt.show()
plt.figure(figsize=(11,7))

sns.countplot(x="objletter", data=vardf)

plt.ylabel('Frequency', fontsize=13)

plt.xlabel('ObjLetter', fontsize=13)

plt.title("Frequency of Feature ObjLetter", fontsize=18)

plt.show()
plt.figure(figsize=(10,10))

sns.stripplot(x="Class", y="objletter", data=vardf, jitter=True);

plt.show()
c1, c2, c3, c4, c5, c6, c7, c8, c9 = "", "", "", "", "", "", "", "", ""



for i in train[train["Class"]==1]["ID"]:

    c1+=train["Text"][i]+" "





for i in train[train["Class"]==2]["ID"]:

    c2+=train["Text"][i]+" "



for i in train[train["Class"]==3]["ID"]:

    c3+=train["Text"][i]+" "

    

for i in train[train["Class"]==4]["ID"]:

    c4+=train["Text"][i]+" "

    

for i in train[train["Class"]==5]["ID"]:

    c5+=train["Text"][i]+" "

    

    

for i in train[train["Class"]==6]["ID"]:

    c6+=train["Text"][i]+" "

    

for i in train[train["Class"]==7]["ID"]:

    c7+=train["Text"][i]+" "

    

for i in train[train["Class"]==8]["ID"]:

    c8+=train["Text"][i]+" "

    

    

for i in train[train["Class"]==9]["ID"]:

    c9+=train_t["Text"][i]+" "

 

def tokenize(_str):

    stops = set(stopwords.words("english"))

    tokens = collections.defaultdict(lambda: 0.)

    wnl = nltk.WordNetLemmatizer()

    for m in re.finditer(r"(\w+)", _str, re.UNICODE):

        m = m.group(1).lower()

        if len(m) < 2: continue

        if m in stops: continue

        if m.isnumeric():continue

        m = wnl.lemmatize(m)

        tokens[m] += 1 

    return tokens
texts_for_training=[]

texts_for_test=[]

num_texts_train=len(train)

num_texts_test=len(test)



print("Tokenizing training texts")

for i in range(0,num_texts_train):

    if((i+1)%1000==0):

        print("Text %d of %d\n"%((i+1), num_texts_train))

    texts_for_training.append(tokenize(train["Text"][i]))

    



print("Tokenizing test texts")

for i in range(0,num_texts_test):

    if((i+1)%1000==0):

        print("Text %d of %d\n"%((i+1), num_texts_test))

    texts_for_test.append(tokenize(test["Text"][i]))
print("Tokenizing cluster 1")

cluster1=tokenize(c1)



print("Tokenizing cluster 2")

cluster2=tokenize(c2)



print("Tokenizing cluster 3")

cluster3=tokenize(c3)



print("Tokenizing cluster 4")

cluster4=tokenize(c4)



print("Tokenizing cluster 5")

cluster5=tokenize(c5)



print("Tokenizing cluster 6")

cluster6=tokenize(c6)



print("Tokenizing cluster 7")

cluster7=tokenize(c7)



print("Tokenizing cluster 8")

cluster8=tokenize(c8)



print("Tokenizing cluster 9")

cluster9=tokenize(c9)
def uniqsPerClass(clase, objective, exact):



    uniqs = collections.defaultdict(lambda: 0.)



    for t, v in clase.items():

        apears=0

        if t in cluster1:

            apears+=1

        if t in cluster2:

            apears+=1

        if t in cluster3:

            apears+=1

        if t in cluster4:

            apears+=1

        if t in cluster5:

            apears+=1

        if t in cluster6:

            apears+=1

        if t in cluster7:

            apears+=1  

        if t in cluster8:

            apears+=1

        if t in cluster9:

            apears+=1

    

        if exact:            

            if apears==objective:

                uniqs[t]=v

        else:

            if apears<(objective+1):

                uniqs[t]=v

    return uniqs

uniC1=uniqsPerClass(cluster1,1,False)

uniC2=uniqsPerClass(cluster2,1,False)

uniC3=uniqsPerClass(cluster3,1,False)

uniC4=uniqsPerClass(cluster4,1,False)

uniC5=uniqsPerClass(cluster5,1,False)

uniC6=uniqsPerClass(cluster6,1,False)

uniC7=uniqsPerClass(cluster7,1,False)

uniC8=uniqsPerClass(cluster8,1,False)

uniC9=uniqsPerClass(cluster9,1,False)

def termsComps(file):

    c1,c2,c3,c4,c5,c6,c7,c8,c9=0.,0.,0.,0.,0.,0.,0.,0.,0.

    for t, v in file.items():

        if t in uniC1:

            c1+=v

        if t in uniC2:

            c2+=v

        if t in uniC3:

            c3+=v

        if t in uniC4:

            c4+=v

        if t in uniC5:

            c5+=v

        if t in uniC6:

            c6+=v

        if t in uniC7:

            c7+=v

        if t in uniC8:

            c8+=v

        if t in uniC9:

            c9+=v

        suma=c1+c2+c3+c4+c5+c6+c7+c8+c9

        if suma==0:

            suma=1

            

    return [c1/suma,c2/suma,c3/suma,c4/suma,c5/suma,c6/suma,c7/suma,c8/suma,c9/suma]
uniqsTextMatr=[]

for file in texts_for_training:

    uniqsTextMatr.append(termsComps(file))
uniqText = pd.DataFrame(uniqsTextMatr, columns=['class'+str(c+1) for c in range(9)])

uniqText['RealClass'] = train["Class"]
uniqText
def precisionT(subclas, realclas, takeNullConsider):

    correct,total=0.,0.

    for i in range(0, len(realclas)):

        if not takeNullConsider:

            if not vacuo(uniqTextList[i][0:9]):

                total+=1

                if uniqTextList[i][0:9].index(max(uniqTextList[i][0:9]))==realclas[i]-1:

                    correct+=1

        else:

            total+=1

            if uniqTextList[i][0:9].index(max(uniqTextList[i][0:9]))==realclas[i]-1:

                correct+=1

    return correct/total



def precisionCoverNull(subclas, realclas,classtocover):

    correct,total=0.,0.

    for i in range(0, len(realclas)):

        if not vacuo(uniqTextList[i][0:9]):

            total+=1

            if uniqTextList[i][0:9].index(max(uniqTextList[i][0:9]))==realclas[i]-1:

                correct+=1

        else:

            total+=1

            if classtocover==realclas[i]:

                correct+=1

    return correct/total





def vacuo(row):

    if row[0]==0.0 and row[1]==0.0 and row[2]==0.0 and row[3]==0.0 and row[4]==0.0 and row[5]==0.0 and row[6]==0.0 and row[7]==0.0 and row[8]==0.0:

        return True

    else:

        return False

    

noinfo=0

for i in range(0,len(uniqText)):

    row=[]

    row.append(uniqText["class1"][i])

    row.append(uniqText["class2"][i])

    row.append(uniqText["class3"][i])

    row.append(uniqText["class4"][i])

    row.append(uniqText["class5"][i])

    row.append(uniqText["class6"][i])

    row.append(uniqText["class7"][i])

    row.append(uniqText["class8"][i])

    row.append(uniqText["class9"][i])

    if vacuo(row):

        noinfo+=1

    

        

print("There are ",len(uniqText)-noinfo, " texts of ",len(uniqText)," in training set that can be classified in their correct class only with the \"unique words per class\" information")



uniqTextList=uniqText.values.tolist()  



forcompare=[]

for i in range(0,len(uniqTextList)):

    forcompare.append(uniqTextList[i][0:9])

    
print(precisionT(forcompare,uniqText["RealClass"],False))
print(precisionT(forcompare,uniqText["RealClass"],True))
uniqText.describe()
def dictotext(dic):

    text=""

    for t,v in dic.items():

        for i in range(0,int(v)):

            text=text+t+" "

    return text            
print("there are ",len(uniC1),"unique words in class1")

text = dictotext(uniC1)

wordcloud = WordCloud(width=800, height=400, max_font_size=80,collocations = False,).generate(text)

plt.figure(figsize=(20,5))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
print("there are ",len(uniC2),"unique words in class2")

text = dictotext(uniC2)

wordcloud = WordCloud(width=800, height=400, max_font_size=80,collocations = False).generate(text)

plt.figure(figsize=(20,5))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()


print("there are ",len(uniC3),"unique words in class3")

text = dictotext(uniC3)

wordcloud = WordCloud(width=800, height=400, max_font_size=80,collocations = False).generate(text)

plt.figure(figsize=(20,5))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
print("there are ",len(uniC4),"unique words in class4")

text = dictotext(uniC4)

wordcloud = WordCloud(width=800, height=400, max_font_size=80,collocations = False).generate(text)

plt.figure(figsize=(20,5))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
print("there are ",len(uniC5),"unique words in class5")

text = dictotext(uniC5)

wordcloud = WordCloud(width=800, height=400, max_font_size=80,collocations = False).generate(text)

plt.figure(figsize=(20,5))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
print("there are ",len(uniC6),"unique words in class6")

text = dictotext(uniC6)

wordcloud = WordCloud(width=800, height=400, max_font_size=80,collocations = False).generate(text)

plt.figure(figsize=(20,5))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
print("there are ",len(uniC7),"unique words in class7")

text = dictotext(uniC7)

wordcloud = WordCloud(width=800, height=400, max_font_size=80,collocations = False).generate(text)

plt.figure(figsize=(20,5))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
print("there are ",len(uniC8),"unique words in class8")

text = dictotext(uniC8)

wordcloud = WordCloud(width=800, height=400, max_font_size=80,collocations = False).generate(text)

plt.figure(figsize=(20,5))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
print("there are ",len(uniC9),"unique words in class9")

text = dictotext(uniC9)

wordcloud = WordCloud(width=800, height=400, max_font_size=80,collocations = False).generate(text)

plt.figure(figsize=(20,5))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
norel=uniqsPerClass(cluster8,9,True)
print("there are ",len(norel),"words that appears in all classes")



text = dictotext(norel)

wordcloud = WordCloud(width=800, height=400, max_font_size=80,collocations = False).generate(text)

plt.figure(figsize=(20,5))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
uniC1=uniqsPerClass(cluster1,2,False)

uniC2=uniqsPerClass(cluster2,2,False)

uniC3=uniqsPerClass(cluster3,2,False)

uniC4=uniqsPerClass(cluster4,2,False)

uniC5=uniqsPerClass(cluster5,2,False)

uniC6=uniqsPerClass(cluster6,2,False)

uniC7=uniqsPerClass(cluster7,2,False)

uniC8=uniqsPerClass(cluster8,2,False)

uniC9=uniqsPerClass(cluster9,2,False)



uniqsTextMatr=[]

for file in texts_for_training:

    uniqsTextMatr.append(termsComps(file))

    

uniqText = pd.DataFrame(uniqsTextMatr, columns=['class'+str(c+1) for c in range(9)])

uniqText['RealClass'] = train["Class"]
uniqText

noinfo=0

for i in range(0,len(uniqText)):

    row=[]

    row.append(uniqText["class1"][i])

    row.append(uniqText["class2"][i])

    row.append(uniqText["class3"][i])

    row.append(uniqText["class4"][i])

    row.append(uniqText["class5"][i])

    row.append(uniqText["class6"][i])

    row.append(uniqText["class7"][i])

    row.append(uniqText["class8"][i])

    row.append(uniqText["class9"][i])

    if vacuo(row):

        noinfo+=1

    

        

print("There are ",len(uniqText)-noinfo, " texts of ",len(uniqText)," in training set that can be classified in their correct class only with the \"uniqsPerClass\" function information")



uniqTextList=uniqText.values.tolist()  



forcompare=[]

for i in range(0,len(uniqTextList)):

    forcompare.append(uniqTextList[i][0:9])

    
print(precisionT(forcompare,uniqText["RealClass"],False))
print(precisionT(forcompare,uniqText["RealClass"],True))
uniC1=uniqsPerClass(cluster1,3,False)

uniC2=uniqsPerClass(cluster2,3,False)

uniC3=uniqsPerClass(cluster3,3,False)

uniC4=uniqsPerClass(cluster4,3,False)

uniC5=uniqsPerClass(cluster5,3,False)

uniC6=uniqsPerClass(cluster6,3,False)

uniC7=uniqsPerClass(cluster7,3,False)

uniC8=uniqsPerClass(cluster8,3,False)

uniC9=uniqsPerClass(cluster9,3,False)



uniqsTextMatr=[]

for file in texts_for_training:

    uniqsTextMatr.append(termsComps(file))

    

uniqText = pd.DataFrame(uniqsTextMatr, columns=['class'+str(c+1) for c in range(9)])

uniqText['RealClass'] = train["Class"]

uniqText
noinfo=0

for i in range(0,len(uniqText)):

    row=[]

    row.append(uniqText["class1"][i])

    row.append(uniqText["class2"][i])

    row.append(uniqText["class3"][i])

    row.append(uniqText["class4"][i])

    row.append(uniqText["class5"][i])

    row.append(uniqText["class6"][i])

    row.append(uniqText["class7"][i])

    row.append(uniqText["class8"][i])

    row.append(uniqText["class9"][i])

    if vacuo(row):

        noinfo+=1

    

        

print("There are ",len(uniqText)-noinfo, " texts of ",len(uniqText)," in training set that can be classified in their correct class only with the \"uniqsPerClass\" function information")



uniqTextList=uniqText.values.tolist()  



forcompare=[]

for i in range(0,len(uniqTextList)):

    forcompare.append(uniqTextList[i][0:9])
print(precisionT(forcompare,uniqText["RealClass"],False))
print(precisionT(forcompare,uniqText["RealClass"],True))