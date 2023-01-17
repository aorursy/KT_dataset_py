import pandas as pd
import matplotlib.pyplot as plt 
data_train = pd.read_csv("../input/nlp-getting-started/train.csv")
byTarget = data_train.groupby(data_train["target"]).count()["id"]
label = ["Fake","Real"]
color = ["blue", "red"]
plt.pie(byTarget,labels=label,colors=color,startangle=90)
#data split real or fake
grouped_data = data_train.groupby("target")
data_train_real = grouped_data.get_group(1)
data_train_fake = grouped_data.get_group(0)

#WordClud
from wordcloud import WordCloud
tweet_text_real = data_train_real["text"]
txt_real = ""
for i in range(len(tweet_text_real)):
    txt_real = txt_real + tweet_text_real.iloc[i]
wordcloud_real = WordCloud(background_color="white",width=800,height=600).generate(txt_real)

tweet_text_fake = data_train_fake["text"]
txt_fake = ""
for i in range(len(tweet_text_fake)):
    txt_fake = txt_fake + tweet_text_fake.iloc[i]
wordcloud_fake = WordCloud(background_color="white",width=800,height=600).generate(txt_fake)

wordcloud_real.to_file("word_cloud_real.png")
wordcloud_fake.to_file("word_cloud_fake.png")

import spacy
import re

###Local Name Extraction###
nlp = spacy.load('en_core_web_sm') #model
#delet url
txt_real = re.sub(r'^https?:\/\/.*[\r\n]*', '', txt_real, flags=re.MULTILINE)
txt_fake = re.sub(r'^https?:\/\/.*[\r\n]*', '', txt_fake, flags=re.MULTILINE)
doc = nlp(txt_real) #load text
doc2 = nlp(txt_fake) #load text
count_real=0
count_fake=0

#extract place name
for d in doc.ents:
    if d.label_ == "GPE": #if proper noun is local name
        count_real+=1

print(f"real tweet:{count_real}")
        
#extract place name
for d in doc2.ents:
    if d.label_ == "GPE": #if proper noun is local name
        count_fake+=1

print(f"fake tweet:{count_fake}")
import re

###Count Word(Name of Place)###
#add columns(num_place)
data_train = data_train.assign(num_place=0)

#count
for i in range(len(data_train["text"])):
    #delet url
    data_train["text"][i] = re.sub(r'^https?:\/\/.*[\r\n]*', '', data_train["text"][i], flags=re.MULTILINE)
    doc = nlp(data_train["text"][i])
    #num of place name
    place_count=0
    #extract place name
    for d in doc.ents:
        if d.label_ == "GPE": #if proper noun is local name
            place_count+=1
        
    data_train["num_place"][i] = place_count

data_train.describe()
from matplotlib import pyplot as plt
import seaborn as sns

#data split real or fake
grouped_data = data_train.groupby("target")
data_train_real = grouped_data.get_group(1)
data_train_fake = grouped_data.get_group(0)

#boxplot
sns.set()
sns.set_style('whitegrid')
sns.set_palette('gray')
fig = plt.figure()
ax = fig.add_subplot()
ax.boxplot([data_train_real["num_place"],data_train_fake["num_place"]], labels=["Real News","Fake News"])
ax.set_title("Number of Local Name Words ")
ax.set_ylabel("number")
ax.set_ylim(-1, 5)

plt.show()
import pandas as pd

data_train["keyword"] = data_train["keyword"].fillna("blank")

#percentage of real tweet
per_byKeyword = data_train.groupby("keyword").mean()["target"].reset_index()

#translation into DataFrame
per_byKeyword = pd.DataFrame(per_byKeyword,columns=["keyword","target"])

foa_byKeyword = per_byKeyword.rename(columns={"target":"FoA"})

foa_byKeyword.head()
#convert
for i in range(len(data_train["id"])):
    
    for j in range(len(foa_byKeyword["keyword"])):
        
        if data_train["keyword"][i] == foa_byKeyword["keyword"][j]:
            
            data_train["keyword"][i] = foa_byKeyword["FoA"][j]

data_train = data_train.rename(columns={"keyword":"FoA"})

data_train.head()