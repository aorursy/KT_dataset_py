!pip install googletrans
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("../input/contradictory-my-dear-watson/train.csv")

data.head()
## check sentences with less than 5 characters



for i in range(len(data)):

    if len(data["premise"][i])<5 or len(data["hypothesis"][i])<5:

        print(data["premise"][i]+ ">>"+ data["hypothesis"][i])
## check max avg and min len of sentences

premise=[]

hypothesis=[]



for i in range(len(data)):

    premise.append(len(data["premise"][i]))

    hypothesis.append(len(data["hypothesis"][i]))

    

print("Average len of characters in premise",sum(premise)//len(premise))

print("Maximum len of characters in premise",max(premise))

print("Minimum len of characters in premise",min(premise),end="\n\n")



print("Average len of characters in hypothesis",sum(hypothesis)//len(hypothesis))

print("Maximum len of characters in hypothesis",max(hypothesis))

print("Minimum len of characters in hypothesis",min(hypothesis))



language_data=pd.DataFrame(data["language"].value_counts()).reset_index().rename(columns={"index":"language","language":"counts"})
import seaborn as sns

import matplotlib.pyplot as plt





plt.figure(figsize=(20,5))



sns.set(style="whitegrid")

ax = sns.barplot(x=language_data["language"], y=language_data["counts"])
languages4plot=[]

Percentage=[]

for ind in range(len(language_data)):

    Percentage.append(round((language_data["counts"][ind]/sum(language_data["counts"]))*100,2))

    languages4plot.append(language_data['language'][ind])
explode=np.random.uniform(0,0,len(Percentage))

plt.figure(figsize=(10,10))

plt.pie(Percentage, explode=explode, labels=languages4plot, autopct='%1.1f%%',

        shadow=True, startangle=90)

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()

plt.savefig("unbalanced_data.png")
lang_abv=set(data["lang_abv"])

lang_abv
english_data=data[data["lang_abv"]=="en"]

english_data.sample(frac=1)
from googletrans import Translator

import googletrans

translator=Translator()
googletrans.LANGUAGES
set(lang_abv)-set(googletrans.LANGUAGES.keys())
set(data["language"])
all_df=[]

english_index=set(english_data.sample(frac=1).index)

for lang in lang_abv:

    print(len(english_index))

    print(lang)

    lang_data=data[data["lang_abv"]==lang]

    if lang=="en":

        balance_num=808

    else:

        balance_num=808-len(lang_data)

        

    required_ind=list(english_index)[:balance_num]

    lang_english_data=data.iloc[required_ind]

    premise=[]

    hypothesis=[]

    lang_abv_list=[]

    

    if lang=="en":

        all_df.append(lang_english_data)

        

    else:

    

        for i in lang_english_data.index:

            if lang=="zh":

                lang="zh-cn"

    #         print(lang_english_data["premise"][i],lang)

            try:



                premise_trans=translator.translate(lang_english_data["premise"][i],dest=lang).text



                hypothesis_trans=translator.translate(lang_english_data["hypothesis"][i],dest=lang).text



                lang_abv_list.append(lang)



            except:

                premise_trans=lang_english_data["premise"][i]

                hypothesis_trans=lang_english_data["hypothesis"][i]

                lang_abv_list.append("en")





            premise.append(premise_trans)

            hypothesis.append(hypothesis_trans)





        lang_english_data["premise"]=premise

        lang_english_data["hypothesis"]=hypothesis

        lang_english_data['lang_abv']=lang_abv_list



        all_df.append(pd.concat([lang_english_data,lang_data]))



    

    english_index=english_index-set(list(english_index)[:balance_num])



balanced_df=pd.concat(all_df)
google_language=googletrans.LANGUAGES
language_list=[]



for lang in balanced_df["lang_abv"]:

    if lang=="zh" or lang=="zh-cn":

        language_list.append("chinease")

    else:

        language_list.append(google_language[lang])

    
balanced_df["language"]=language_list
balanced_df.reset_index(drop=True, inplace=True)
balanced_df.to_csv("balanced_data.csv",index=False)
language_abbr=list(balanced_df["language"].value_counts().index)

percentage=[round((i/sum(balanced_df["language"].value_counts().values))*100,2) for i in balanced_df["language"].value_counts().values]
explode=np.random.uniform(0,0,len(percentage))

plt.figure(figsize=(10,10))

plt.pie(percentage, explode=explode, labels=language_abbr, autopct='%1.1f%%',

        shadow=True, startangle=90)

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()

plt.savefig("balanced.png")
from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

tokenize_premise=[]

tokenize_hypothesis=[]

languages=[]



for ind in range(len(balanced_df)):

    tokenize_premise.append(len(list(tokenizer.tokenize(balanced_df["premise"][ind], return_tensors="tf"))))

    tokenize_hypothesis.append(len(list(tokenizer.tokenize(balanced_df["hypothesis"][ind], return_tensors="tf"))))

    languages.append(balanced_df["language"][ind])

    
print("Average len of premise",sum(tokenize_premise)//len(tokenize_premise))

print("Maximum len of premise",max(tokenize_premise))

print("Minimum len of premise",min(tokenize_premise),end="\n\n")



print("Average len of hypothesis",sum(tokenize_hypothesis)//len(tokenize_hypothesis))

print("Maximum len of hypothesis",max(tokenize_hypothesis))

print("Minimum len of hypothesis",min(tokenize_hypothesis))
plt.figure(figsize=(20,5))



sns.set(style="whitegrid")

ax = sns.scatterplot(x=languages, y=tokenize_premise)
plt.figure(figsize=(20,5))



sns.set(style="whitegrid")

ax = sns.scatterplot(x=languages, y=tokenize_hypothesis)
X_data=data[["premise", "hypothesis"]]

Y_data=data["label"]
count_df=pd.DataFrame(Y_data.value_counts()).reset_index().rename(columns={"index":"label","label":"counts"})

labels=list(count_df["label"])

sizes=[(i/len(Y_data))*100 for i in count_df["counts"]]

explode = (0, 0.1, 0)
plt.figure(figsize=(10,5))

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()