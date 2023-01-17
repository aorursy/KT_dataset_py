all_files=["Youtube01-Psy.csv","Youtube02-KatyPerry.csv","Youtube03-LMFAO.csv","Youtube04-Eminem.csv"]
list_=[]

import pandas as pd

for file_ in all_files:

    frame = pd.read_csv(("../input/images/"+file_),index_col=None, header=0)

    list_.append(frame)



# concatenate all dfs into one

df = pd.concat(list_, ignore_index=True)
df
df.drop(columns=["COMMENT_ID","AUTHOR","DATE"],inplace=True)
df["CONTENT"][700]
import html

df["CONTENT"]=df["CONTENT"].apply(html.unescape)

df["CONTENT"]=df["CONTENT"].str.replace("\ufeff","")
df["CONTENT"][700]
df["CONTENT"]=df["CONTENT"].str.replace("(<a.+>)","htmllink")
df[df["CONTENT"].str.contains("<.+>")]["CONTENT"]
df["CONTENT"]=df["CONTENT"].str.replace("<.+>","")
df["CONTENT"]=df["CONTENT"].str.replace("\'","")
df["CONTENT"]=df["CONTENT"].str.lower()
df[df["CONTENT"].str.contains("\.com|watch\?")]
df["CONTENT"][1573]
df["CONTENT"]=df["CONTENT"].str.replace(r"\S*\.com\S*|\S*watch\?\S*","htmllink")
df["CONTENT"]=df["CONTENT"].str.replace("\W"," ")
df["CONTENT"][1573]
df["CONTENT"][14]
df["CLASS"].value_counts(normalize=True)
vocab=[]

for comment in df["CONTENT"]:

    for word in comment.split():

        vocab.append(word)
vocabulary=list(set(vocab))

len(vocabulary)
# Create a column for each of the unique word in our vocabulary inorder to get the count of words 

for word in vocabulary:

    df[word]=0
df.head()
# looping through data frame and counting words 

for index,value in enumerate(df["CONTENT"]):

  for l in value.split():

    df[l][index]+=1
df.head()
#Total number of words in each class

df.groupby("CLASS").sum().sum(axis=1)
# Assign variables to all values required in calculation

p_ham=0.47604

p_spam=0.52396

n_spam=df[df["CLASS"]==1].drop(columns=["CONTENT","CLASS"]).sum().sum()

n_ham=df[df["CLASS"]==0].drop(columns=["CONTENT","CLASS"]).sum().sum()

n_vocabulary=len(vocabulary)
# Slicing dataframe for each class

df_sspam=df[df["CLASS"]==1]

df_hham=df[df["CLASS"]==0]
parameters_spam = {unique_word:0 for unique_word in vocabulary}

parameters_ham = {unique_word:0 for unique_word in vocabulary}

for word in vocabulary:

    n_word_given_spam = df_sspam[word].sum()   # spam_messages already defined in a cell above

    p_word_given_spam = (n_word_given_spam + 1) / (n_spam + 1*n_vocabulary)

    parameters_spam[word] = p_word_given_spam

    n_word_given_ham = df_hham[word].sum()   # ham_messages already defined in a cell above

    p_word_given_ham = (n_word_given_ham + 1) / (n_ham + 1*n_vocabulary)

    parameters_ham[word] = p_word_given_ham
def classifier(string):

    message=html.unescape(string)

    message=string.replace("\ufeff","")

    message=string.replace("(<a.+>)","htmllink")

    message=string.replace("\'|<.+>","")

    message=string.replace("\S*\.com\S*|\S*watch\?\S*","htmllink")

    message=string.replace("\W"," ").lower()

    p_string_s=1

    p_string_h=1

    for word in message.split():

        if word in parameters_spam:

            p_string_s*=parameters_spam[word]

            p_string_h*=parameters_ham[word]

    if (p_string_s*p_spam)>(p_string_h*p_ham):

        return(1)

    elif (p_string_s*p_spam)<(p_string_h*p_ham):

        return(0)

    else:

        return(-1)
# Reading the dataframe for testing model

df_shakira=pd.read_csv("../input/images/Youtube05-Shakira.csv")
df_shakira.head()
df_shakira["Pred_Class"]=df_shakira["CONTENT"].apply(classifier)
correct_predictions=0

total_rows=0

for row in df_shakira.iterrows():

    row=row[1]

    total_rows+=1

    if row["CLASS"]==row["Pred_Class"]:

        correct_predictions+=1

accuracy=correct_predictions/total_rows

print("accuracy=",accuracy)

classifier("This song gives me goosebumps!!")

classifier("Please subscribe to my channel as I'm approaching 1M subscribers")
classifier("If you want to be a mastercoder, consider buying my course for 50% off at www.buymycourse.com")