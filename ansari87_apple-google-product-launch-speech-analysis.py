from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

import pandas as pd

import re, spacy,markovify

from collections import Counter

import seaborn as sns
sw = set(STOPWORDS)

nlp = spacy.load("en_core_web_sm")



def show_word_cloud(title,content):

    wordcloud = WordCloud(stopwords=sw,background_color = 'white',width=1500,height=1000).generate(content)

    

    plt.figure(figsize=(15,10))

    plt.imshow(wordcloud,interpolation='bilinear')

    plt.axis('off')

    print(title)

    plt.show()



def find_adjectives(content):

    document = nlp(content)

    adjectives=[]

    for token in document:

        if(token.pos_=="ADJ" and token.lemma_ != "-"):

              adjectives.append(token.lemma_)

    return adjectives
#reading content of all the files

base_dir_of_subtitles="../input/sub-titles-of-presentations/"

file_names = ["Google-2018","Google-2019","Apple-2018","Apple-2019"]



file_content_array=[]

for file_name in file_names:

    file_content=open (base_dir_of_subtitles+file_name+".txt").read()



    #remove speaker names from content

    words_to_remove = re.findall(r'(\w+:)',file_content)



    for word in words_to_remove:

        file_content=file_content.replace(word,"")

        

    file_content_array.append(file_content)
#word cloud of google speech

show_word_cloud(file_names[0],file_content_array[0])

show_word_cloud(file_names[1],file_content_array[1])
#word cloud of apple speech

show_word_cloud(file_names[2],file_content_array[2])

show_word_cloud(file_names[3],file_content_array[3])
#Finding all adjectives in the speech using spaCy

adjectives_array=[]

for file_name in file_names:

    file_content=""

    file_content=open (base_dir_of_subtitles+file_name+".txt").read()

    adjectives_array.append(find_adjectives(file_content))
#word cloud of adjectives in google speech

adjective_google_2018=' '.join([str(elem) for elem in adjectives_array[0]])

show_word_cloud(file_names[0],adjective_google_2018)



adjective_google_2019=' '.join([str(elem) for elem in adjectives_array[1]])

show_word_cloud(file_names[1],adjective_google_2019)
#word cloud of adjectives in apple speech

adjective_apple_2018=' '.join([str(elem) for elem in adjectives_array[2]])

show_word_cloud(file_names[2],adjective_apple_2018)



adjective_apple_2019=' '.join([str(elem) for elem in adjectives_array[3]])

show_word_cloud(file_names[3],adjective_apple_2019)
#Plotting most used adjectives in each of the speeches

top_adjective_count=20



freq=Counter(adjectives_array[0])

df=pd.DataFrame(freq.most_common(top_adjective_count),columns=['adjective','count'])

df=df.apply(lambda x: x.astype(str).str.lower())

df['count'] = pd.to_numeric(df['count'])

df['company']='Google-2018'



freq1=Counter(adjectives_array[1])

df1=pd.DataFrame(freq1.most_common(top_adjective_count),columns=['adjective','count'])

df1['count'] = pd.to_numeric(df1['count'])

df1['company']='Google-2019'



freq2=Counter(adjectives_array[2])

df2=pd.DataFrame(freq2.most_common(top_adjective_count),columns=['adjective','count'])

df2['count'] = pd.to_numeric(df2['count'])

df2['company']='Apple-2018'



freq3=Counter(adjectives_array[3])

df3=pd.DataFrame(freq3.most_common(top_adjective_count),columns=['adjective','count'])

df3['count'] = pd.to_numeric(df3['count'])

df3['company']='Apple-2019'



final_df=pd.concat([df,df1,df2,df3])



plot_data = final_df.pivot("adjective", "company", "count")



chart_title="Top "+str(top_adjective_count)+" Adjectives in each of the Speeches"

f, ax = plt.subplots(figsize=(15, 12))

ax.set_title(chart_title)

sns.heatmap(plot_data, annot=True, fmt=".0f", linewidths=.2,linecolor='grey', ax=ax,cmap="YlGnBu")
#Plotting most used words in each of the speeches

file_content_array[0]=re.sub('\n',' ',file_content_array[0])

file_content_array[1]=re.sub('\n',' ',file_content_array[1])

file_content_array[2]=re.sub('\n',' ',file_content_array[2])

file_content_array[3]=re.sub('\n',' ',file_content_array[3])



file_content_array[0]=file_content_array[0].lower()

file_content_array[1]=file_content_array[1].lower()

file_content_array[2]=file_content_array[2].lower()

file_content_array[3]=file_content_array[3].lower()



added_stopwords=['.',',','-','3','!','4','_','"','--','?',"we're","we've","you're","yeah","tim"]

for w in added_stopwords:

    nlp.vocab[w].is_stop = True

    lex = nlp.vocab[w]

    lex.is_stop = True



doc = nlp(file_content_array[0])

tokens = [token.text for token in doc if not token.is_stop and token.text.strip()!='']



top_words_count=20

freq=Counter(tokens)

df=pd.DataFrame(freq.most_common(top_words_count),columns=['words','count'])

df=df.apply(lambda x: x.astype(str).str.lower())

df['count'] = pd.to_numeric(df['count'])

df['company']='Google-2018'



doc = nlp(file_content_array[1])

tokens = [token.text.strip() for token in doc if not token.is_stop and token.text.strip()!='']



freq=Counter(tokens)

df1=pd.DataFrame(freq.most_common(top_words_count),columns=['words','count'])

df1=df1.apply(lambda x: x.astype(str).str.lower())

df1['count'] = pd.to_numeric(df1['count'])

df1['company']='Google-2019'



doc = nlp(file_content_array[2])

tokens = [token.text for token in doc if not token.is_stop and token.text.strip()!='']



freq=Counter(tokens)

df2=pd.DataFrame(freq.most_common(top_words_count),columns=['words','count'])

df2=df2.apply(lambda x: x.astype(str).str.lower())

df2['count'] = pd.to_numeric(df2['count'])

df2['company']='Apple-2018'



doc = nlp(file_content_array[3])

tokens = [token.text for token in doc if not token.is_stop and token.text.strip()!='']



freq=Counter(tokens)

df3=pd.DataFrame(freq.most_common(top_words_count),columns=['words','count'])

df3=df3.apply(lambda x: x.astype(str).str.lower())

df3['count'] = pd.to_numeric(df3['count'])

df3['company']='Apple-2019'



final_df=pd.concat([df,df1,df2,df3])



plot_data=pd.pivot_table(final_df,values='count',index='words',columns='company')



chart_title="Top "+str(top_words_count)+" word in each of the Speeches"

f, ax = plt.subplots(figsize=(15, 12))

ax.set_title(chart_title)

sns.heatmap(plot_data, annot=True, fmt=".0f", linewidths=.2,linecolor='grey', ax=ax,cmap="YlGnBu")
#most frequent bi-grams - pair of words appearing most number of time in the speeches

doc = nlp(file_content_array[0])

tokens = [token.text for token in doc if not token.is_stop and token.text.strip()!='']



top_bigram_count=10

bigrams = zip(tokens, tokens[1:])

freq=Counter(bigrams)

df=pd.DataFrame(freq.most_common(top_words_count),columns=['words','count'])

df=df.apply(lambda x: x.astype(str).str.lower())

df['count'] = pd.to_numeric(df['count'])

df['company']='Google-2018'



doc = nlp(file_content_array[1])

tokens = [token.text.strip() for token in doc if not token.is_stop and token.text.strip()!='']



bigrams = zip(tokens, tokens[1:])

freq=Counter(bigrams)

df1=pd.DataFrame(freq.most_common(top_words_count),columns=['words','count'])

df1=df1.apply(lambda x: x.astype(str).str.lower())

df1['count'] = pd.to_numeric(df1['count'])

df1['company']='Google-2019'



doc = nlp(file_content_array[2])

tokens = [token.text for token in doc if not token.is_stop and token.text.strip()!='']



bigrams = zip(tokens, tokens[1:])

freq=Counter(bigrams)

df2=pd.DataFrame(freq.most_common(top_words_count),columns=['words','count'])

df2=df2.apply(lambda x: x.astype(str).str.lower())

df2['count'] = pd.to_numeric(df2['count'])

df2['company']='Apple-2018'



doc = nlp(file_content_array[3])

tokens = [token.text for token in doc if not token.is_stop and token.text.strip()!='']



bigrams = zip(tokens, tokens[1:])

freq=Counter(bigrams)

df3=pd.DataFrame(freq.most_common(top_words_count),columns=['words','count'])

df3=df3.apply(lambda x: x.astype(str).str.lower())

df3['count'] = pd.to_numeric(df3['count'])

df3['company']='Apple-2019'



final_df=pd.concat([df,df1,df2,df3])



plot_data=pd.pivot_table(final_df,values='count',index='words',columns='company')



chart_title="Top "+str(top_words_count)+" bi-grams in each of the Speeches"

f, ax = plt.subplots(figsize=(10, 20))

ax.set_title(chart_title)

sns.heatmap(plot_data, annot=True, fmt=".0f", linewidths=.2,linecolor='grey', ax=ax,cmap="YlGnBu")
#Now, with all the 4 speeches, lets try to generate random sentences using Markovify

#https://github.com/jsvine/markovify

with open(base_dir_of_subtitles+"Google-2018.txt") as f:

    text = f.read()



with open(base_dir_of_subtitles+"Apple-2018.txt") as f:

    text2 = f.read()



with open(base_dir_of_subtitles+"Google-2019.txt") as f:

    text3 = f.read()



with open(base_dir_of_subtitles+"Apple-2019.txt") as f:

    text4 = f.read()

    

text_model = markovify.Text(text)

text_model2 = markovify.Text(text2)

text_model3 = markovify.Text(text3)

text_model4 = markovify.Text(text4)



model_combo = markovify.combine([ text_model, text_model2,text_model3,text_model4 ], [1,1,1,1])



# Print five randomly-generated sentences

for i in range(5):

    print(str(i+1)+" : "+model_combo.make_sentence())