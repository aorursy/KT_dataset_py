#Import the different libraries 

import os

print(os.listdir("../input")) #display the available files for analysis

import pandas as pd

from pandas import DataFrame

import numpy as np

import seaborn as sns

from collections import defaultdict

import re

from bs4 import BeautifulSoup

#Code for wordcloud (adapted for removal of stop words)



#Code adpted from : https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc



#import the wordcloud package

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



#Define the word cloud function with a max of 200 words

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(10,10), 

                   title = None, title_size=20, image_color=False):

    stopwords = set(STOPWORDS)

    #define additional stop words that are not contained in the dictionary

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)

    #Generate the word cloud

    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    #set the plot parameters

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  
#ngram function

def ngram_extractor(text, n_gram):

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



# Function to generate a dataframe with n_gram and top max_row frequencies

def generate_ngrams(df, n_gram, max_row):

    temp_dict = defaultdict(int)

    for question in df:

        for word in ngram_extractor(question, n_gram):

            temp_dict[word] += 1

    temp_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda x: x[1])[::-1]).head(max_row)

    temp_df.columns = ["word", "wordcount"]

    return temp_df



#Function to construct side by side comparison plots

def comparison_plot(df_1,df_2,col_1,col_2, space):

    fig, ax = plt.subplots(1, 2, figsize=(20,10))

    

    sns.barplot(x=col_2, y=col_1, data=df_1, ax=ax[0], color="royalblue")

    sns.barplot(x=col_2, y=col_1, data=df_2, ax=ax[1], color="royalblue")



    ax[0].set_xlabel('Word count', size=14)

    ax[0].set_ylabel('Words', size=14)

    ax[0].set_title('Top words in sincere questions', size=18)



    ax[1].set_xlabel('Word count', size=14)

    ax[1].set_ylabel('Words', size=14)

    ax[1].set_title('Top words in insincere questions', size=18)



    fig.subplots_adjust(wspace=space)

    

    plt.show()
#Import and view information of the Professionals Dataset

prof = pd.read_csv('../input/professionals.csv')

prof.info()
#Print the start and end dates

print('Oldest Professional join date:',prof.professionals_date_joined.min(), '\n' +'Most Recent Professional join date:',prof.professionals_date_joined.max()) 
#Tabulate the count of missing values in the dataset

prof.isnull().sum()



#The location (~11% of missing values), industry (~9% of missing values) and headline (~7% of missing values)columns have missing values 

#missing value % has been calculated as [missing value count/ 28152]
#Select headlines from professionals dataset

prof_headlines = prof["professionals_headline"]

prof_headlines.replace('--', np.nan, inplace=True) 

prof_headlines_na = prof_headlines.dropna()

#run the function on the professional headlines and Remove NA values for clarity of visualisation

plot_wordcloud(prof_headlines_na, title="Word Cloud of Professionals Headlines")
#Define a barplot for each

#Below code adapted from: https://www.kaggle.com/anu0012/quick-start-eda-careervillage-org



headlines = prof_headlines_na.value_counts().head(20)

plt.figure(figsize=(12,8))

sns.barplot(headlines.values, headlines.index)

plt.xlabel("Count", fontsize=15)

plt.ylabel("Unique Headlines", fontsize=15)

plt.title("Top 20 Unique Professionals Headlines")

plt.show()
prof_industry_na = prof["professionals_industry"].dropna()       

industries = prof_industry_na.value_counts().head(20)

plt.figure(figsize=(12,8))

sns.barplot(industries.values, industries.index)

plt.xlabel("Count", fontsize=15)

plt.ylabel("Unique Industries", fontsize=15)

plt.title("Top 20 Unique Professionals Headlines")

plt.show()
prof_loc_na = prof["professionals_location"].dropna()       

top_loc = prof_loc_na.value_counts().head(20)

print(top_loc)
#Import and view information of the School membership Dataset

school = pd.read_csv('../input/school_memberships.csv')

school.info()

#Check for missing values

school.isnull().sum()
#Identify the top school IDs

school_id_na = school["school_memberships_school_id"] 

top_school_ids = school_id_na.value_counts().head(20)

print(top_school_ids)
#Import and view information of the Matches Dataset

match = pd.read_csv('../input/matches.csv')

match.info()

#Check for missing values

match.isnull().sum()
#Identify which email ids contained the most questions

top_match_emails = match["matches_email_id"] .value_counts().head(5)

print(top_match_emails)
#Import and view information of the Tags Dataset

tag_ques = pd.read_csv('../input/tag_questions.csv')

tag_ques.info()

#Check for missing values

tag_ques.isnull().sum()
#Identify which tag ids were the most used

tag_ques_pair = tag_ques["tag_questions_tag_id"] .value_counts().head(5)

print(tag_ques_pair)
#Import and view information of the Tags Dataset

tags = pd.read_csv('../input/tags.csv')

tags.info()

#Check for missing values

tags.isnull().sum()
#Identify which tags were the most used

tags_names = tags["tags_tag_name"] .value_counts().head(5)

print(tags_names)
#Import and view information of the Tags Dataset

ans = pd.read_csv('../input/answers.csv')

ans.info()

#Check for missing values

ans.isnull().sum()
#Print the oldest and most recent start date

print('Oldest answer date:',ans.answers_date_added.min(), '\n' +'Most recent answer date:',ans.answers_date_added.max()) 
ans_author_id = ans["answers_author_id"] .value_counts().tail(5) #switch to .head for the top n responses

print(ans_author_id)
#Obtain the unique counts of author ids 

ans_author = ans["answers_author_id"].value_counts().head(20)

#print(ans_author.tail(5))

plt.figure(figsize=(12,8))

sns.barplot(ans_author.values, ans_author.index)

plt.xlabel("Count", fontsize=15)

plt.ylabel("Unique author_id", fontsize=15)

plt.title("Top 20 Unique author_id")

plt.show()
#Select headlines from professionals dataset

ans_body = ans["answers_body"]

ans_body_na = ans_body.dropna()

#run the function on the professional headlines and Remove NA values for clarity of visualisation

plot_wordcloud(ans_body_na, title="Word Cloud for Answer body")
#Define empty list

ans_bod_cleaned = []

res = []

#Define for loop to iterate through the elements of the answer_body

for l in ans_body_na:

    #Parse the contents of the cell

    soup = BeautifulSoup(l, 'html.parser')

    #Find all instances of the text within the </p> tag

    for el in soup.find_all('p'):

        res.append(el.get_text())

    #concatenate the strings from the list    

    endstring = ' '.join(map(str, res))

    #reset list

    res = []

    #Append the concatenated string to the main list

    ans_bod_cleaned.append(endstring)
#convert list elements to lower case

ans_body_na_cleaned = [item.lower() for item in ans_bod_cleaned]

#remove html links from list 

ans_body_na_cleaned =  [re.sub(r"http\S+", "", item) for item in ans_body_na_cleaned]

#remove special characters left

ans_body_na_cleaned = [re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", item) for item in ans_body_na_cleaned]

#convert to dataframe and rename the column of the ans_body_na_cleaned list

ans_body_na_clean = pd.DataFrame(np.array(ans_body_na_cleaned).reshape(-1))

ans_body_na_clean.columns = ["ans"]

#Squeeze dataframe to obtain series

answers_cleaned = ans_body_na_clean.squeeze()
#generate unigram

ans_unigram = generate_ngrams(answers_cleaned, 1, 20)
#generate barplot for unigram

plt.figure(figsize=(12,8))

sns.barplot(ans_unigram["wordcount"],ans_unigram["word"])

plt.xlabel("Word Count", fontsize=15)

plt.ylabel("Unigrams", fontsize=15)

plt.title("Top 20 Unigrams for Answer body")

plt.show()
#generate bigram

ans_bigram = generate_ngrams(answers_cleaned, 2, 20)
#generate barplot for bigram

plt.figure(figsize=(12,8))

sns.barplot(ans_bigram["wordcount"],ans_bigram["word"])

plt.xlabel("Word Count", fontsize=15)

plt.ylabel("Bigrams", fontsize=15)

plt.title("Top 20 Bigrams for Answer body")

plt.show()
#generate trigram

ans_trigram = generate_ngrams(answers_cleaned, 3, 20)
#generate barplot for bigram

plt.figure(figsize=(12,8))

sns.barplot(ans_trigram["wordcount"],ans_trigram["word"])

plt.xlabel("Word Count", fontsize=15)

plt.ylabel("Trigrams", fontsize=15)

plt.title("Top 20 Trigrams for Answer body")

plt.show()
# Number of words in the answers

answers_cleaned["word_count"] = answers_cleaned.apply(lambda x: len(str(x).split()))



fig, ax = plt.subplots(figsize=(15,2))

sns.boxplot(x="word_count", data=answers_cleaned, ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')

ax.set_xlabel('Word Count', size=10, color="#0D47A1")

ax.set_ylabel('Answer body', size=10, color="#0D47A1")

ax.set_title('[Horizontal Box Plot] Word Count distribution for Answer body', size=12, color="#0D47A1")

plt.gca().xaxis.grid(True)

plt.show()
# Number of stopwords in answers

answers_cleaned["stop_words_count"] = answers_cleaned.apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

fig, ax = plt.subplots(figsize=(15,2))

sns.boxplot(x="stop_words_count", data=answers_cleaned, ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')

ax.set_xlabel('Stop Word Count', size=10, color="#0D47A1")

ax.set_ylabel('Answer body', size=10, color="#0D47A1")

ax.set_title('[Horizontal Box Plot] Number of Stop Words distribution for Answer body', size=12, color="#0D47A1")

plt.gca().xaxis.grid(True)

plt.show()
#Import and view information of the Tags Dataset

emails = pd.read_csv('../input/emails.csv')

emails.info()

#Check for missing values

emails.isnull().sum()
#Print the oldest and most recent start date

print('Oldest email date:',emails.emails_date_sent.min(), '\n' +'Most recent email date:',emails.emails_date_sent.max()) 
#The different frequencies at which the emails were received

emails["emails_frequency_level"].unique()    
#Identify which recipients were the most solicited

recipient_emails = emails["emails_recipient_id"] .value_counts().head(5)

print(recipient_emails)
#Import and view information of the Tags Dataset

students = pd.read_csv('../input/students.csv')

students.info()

#Check for missing values

students.isnull().sum()
#Identify where most students are from

students_loc = students["students_location"].dropna()

print(students_loc.value_counts().head(5))
#Print the oldest and most recent start date

print('Oldest student registration date:',students.students_date_joined.min(), '\n' +'Most recent student registration date:',students.students_date_joined.max()) 
#Import and view information of the Tags Dataset

ques = pd.read_csv('../input/questions.csv')

ques.info()

#Check for missing values

ques.isnull().sum()
#Print the oldest and most recent start date

print('Oldest question date:',ques.questions_date_added.min(), '\n' +'Most recent question date:',ques.questions_date_added.max()) 
#Identify who asked the most questions

ques_auth = ques["questions_author_id"].dropna()

print(ques_auth.value_counts().head(5))
#Select headlines from professionals dataset

ques_title = ques["questions_title"]

ques_title_na = ques_title.dropna()

#run the function on the professional headlines and Remove NA values for clarity of visualisation

plot_wordcloud(ques_title_na, title="Word Cloud for Question title")
#convert list elements to lower case

quest_title_na_cleaned = [item.lower() for item in ques_title_na]

#remove html links from list 

quest_title_na_cleaned =  [re.sub(r"http\S+", "", item) for item in quest_title_na_cleaned]

#remove special characters left

quest_title_na_cleaned = [re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", item) for item in quest_title_na_cleaned]
#generate unigram

ques_title_unigram = generate_ngrams(quest_title_na_cleaned, 1, 20)
#generate barplot for unigram

plt.figure(figsize=(12,8))

sns.barplot(ques_title_unigram["wordcount"],ques_title_unigram["word"])

plt.xlabel("Word Count", fontsize=15)

plt.ylabel("Unigrams", fontsize=15)

plt.title("Top 20 Unigrams for Question title")

plt.show()
#generate bigram

ques_title_bigram = generate_ngrams(quest_title_na_cleaned, 2, 20)
#generate barplot for bigram

plt.figure(figsize=(12,8))

sns.barplot(ques_title_bigram["wordcount"],ques_title_bigram["word"])

plt.xlabel("Word Count", fontsize=15)

plt.ylabel("Bigrams", fontsize=15)

plt.title("Top 20 Bigrams for Question title")

plt.show()
#generate trigram

ques_title_trigram = generate_ngrams(quest_title_na_cleaned, 3, 20)
#generate barplot for trigram

plt.figure(figsize=(12,8))

sns.barplot(ques_title_trigram["wordcount"],ques_title_trigram["word"])

plt.xlabel("Word Count", fontsize=15)

plt.ylabel("Bigrams", fontsize=15)

plt.title("Top 20 Trigrams for Question title")

plt.show()
#convert to dataframe and rename the column of the ans_body_na_cleaned list

ques_title_na_clean = pd.DataFrame(np.array(quest_title_na_cleaned).reshape(-1))

ques_title_na_clean.columns = ["ques_title"]

ques_title_na_clean = ques_title_na_clean.squeeze()



# Number of words in the question_title

ques_title_na_clean["word_count"] = ques_title_na_clean.apply(lambda x: len(str(x).split()))

fig, ax = plt.subplots(figsize=(15,2))

sns.boxplot(x="word_count", data= ques_title_na_clean, ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')

ax.set_xlabel('Word Count', size=10, color="#0D47A1")

ax.set_ylabel('Question title text', size=10, color="#0D47A1")

ax.set_title('[Horizontal Box Plot] Word Count distribution for Question title', size=12, color="#0D47A1")

plt.gca().xaxis.grid(True)

plt.show()
# Number of stopwords in question_title

ques_title_na_clean["stop_words_count"] = ques_title_na_clean.apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

fig, ax = plt.subplots(figsize=(15,2))

sns.boxplot(x="stop_words_count", data=answers_cleaned, ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')

ax.set_xlabel('Stop Word Count', size=10, color="#0D47A1")

ax.set_ylabel('Question title text', size=10, color="#0D47A1")

ax.set_title('[Horizontal Box Plot] Number of Stop Words distribution for Question title', size=12, color="#0D47A1")

plt.gca().xaxis.grid(True)

plt.show()
#Select headlines from professionals dataset

ques_bod = ques["questions_body"]

ques_bod_na = ques_bod.dropna()

#run the function on the professional headlines and Remove NA values for clarity of visualisation

plot_wordcloud(ques_bod_na, title="Word Cloud for Question Body")
#convert list elements to lower case

quest_bod_na_cleaned = [item.lower() for item in ques_bod_na]

#remove special characters left

quest_bod_na_cleaned = [re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", item) for item in quest_bod_na_cleaned]
#generate unigram

ques_bod_unigram = generate_ngrams(quest_bod_na_cleaned, 1, 20)
#generate barplot for unigram for question_body

plt.figure(figsize=(12,8))

sns.barplot(ques_bod_unigram["wordcount"],ques_bod_unigram["word"])

plt.xlabel("Word Count", fontsize=15)

plt.ylabel("Unigrams", fontsize=15)

plt.title("Top 20 Unigrams for Question Body")

plt.show()
#generate bigram

ques_bod_bigram = generate_ngrams(quest_bod_na_cleaned, 2, 20)
#generate barplot for bigram question_body

plt.figure(figsize=(12,8))

sns.barplot(ques_bod_bigram["wordcount"],ques_bod_bigram["word"])

plt.xlabel("Word Count", fontsize=15)

plt.ylabel("Bigrams", fontsize=15)

plt.title("Top 20 Bigrams for Question Body")

plt.show()
#generate trigram

ques_bod_trigram = generate_ngrams(quest_bod_na_cleaned, 3, 20)
#generate barplot for bigram question_body

plt.figure(figsize=(12,8))

sns.barplot(ques_bod_trigram["wordcount"],ques_bod_trigram["word"])

plt.xlabel("Word Count", fontsize=15)

plt.ylabel("Trigrams", fontsize=15)

plt.title("Top 20 Trigrams for Question Body")

plt.show()
#convert to dataframe and rename the column of the ans_body_na_cleaned list

ques_bod_na_clean = pd.DataFrame(np.array(quest_bod_na_cleaned).reshape(-1))

ques_bod_na_clean.columns = ["ques_body"]

ques_bod_na_clean = ques_bod_na_clean.squeeze()



# Number of words in the question_body

ques_bod_na_clean["word_count"] = ques_bod_na_clean.apply(lambda x: len(str(x).split()))

fig, ax = plt.subplots(figsize=(15,2))

sns.boxplot(x="word_count", data= ques_bod_na_clean, ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')

ax.set_xlabel('Word Count', size=10, color="#0D47A1")

ax.set_ylabel('Question Body text', size=10, color="#0D47A1")

ax.set_title('[Horizontal Box Plot] Word Count distribution for Question Body', size=12, color="#0D47A1")

plt.gca().xaxis.grid(True)

plt.show()
# Number of stopwords in question_body

ques_bod_na_clean["stop_words_count"] = ques_bod_na_clean.apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

fig, ax = plt.subplots(figsize=(15,2))

sns.boxplot(x="stop_words_count", data=ques_bod_na_clean, ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')

ax.set_xlabel('Stop Word Count', size=10, color="#0D47A1")

ax.set_ylabel('Question Body text', size=10, color="#0D47A1")

ax.set_title('[Horizontal Box Plot] Number of Stop Words distribution for Question Body', size=12, color="#0D47A1")

plt.gca().xaxis.grid(True)

plt.show()
#Import and view information of the Groups Dataset

groups = pd.read_csv('../input/groups.csv')

groups.info()

#Check for missing values

groups.isnull().sum()
#Identify who asked the most questions

group_type = groups["groups_group_type"].dropna()

print(group_type .value_counts().tail(5)) #switch to .head for the top results 
#Import and view information of the Groups membership Dataset

groups_mem = pd.read_csv('../input/group_memberships.csv')

groups_mem.info()

#Check for missing values

groups_mem.isnull().sum()
#Identify the most popular group_memberships_user_id     

groups_mem_id = groups_mem["group_memberships_user_id"].dropna()

print(groups_mem_id .value_counts().head(5))
#Identify the most popular group_memberships_group_id

groups_mem_g_id = groups_mem["group_memberships_group_id"].dropna()

print(groups_mem_g_id .value_counts().head(5))
#Import and view information of the comments Dataset

comms = pd.read_csv('../input/comments.csv')

comms.info()

#Check for missing values

comms.isnull().sum()
#Identify the most popular commenters (comments_author_id)

comms_authors = comms["comments_author_id"].dropna()

print(comms_authors .value_counts().head(5))
#Print the oldest and most recent comment dates

print('Oldest comment date:',comms.comments_date_added.min(), '\n' +'Most comment date:',comms.comments_date_added.max()) 
#select comments text from comments dataste

comms_bod = comms["comments_body"]

comms_bod_na = comms_bod.dropna()

#run the function on the professional headlines and Remove NA values for clarity of visualisation

plot_wordcloud(comms_bod_na, title="Word Cloud of Comments body")
#convert list elements to lower case

comms_bod_na_cleaned = [item.lower() for item in comms_bod_na]

#remove html links from list 

comms_bod_na_cleaned =  [re.sub(r"http\S+", "", item) for item in comms_bod_na_cleaned]

#remove special characters left

comms_bod_na_cleaned = [re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", item) for item in comms_bod_na_cleaned]
#generate unigram from comments body

comms_bod_unigram = generate_ngrams(comms_bod_na_cleaned, 1, 20)
#generate barplot for unigram from comments body

plt.figure(figsize=(12,8))

sns.barplot(comms_bod_unigram["wordcount"],comms_bod_unigram["word"])

plt.xlabel("Word Count", fontsize=15)

plt.ylabel("Unigrams", fontsize=15)

plt.title("Top 20 unigrams from Comments Body")

plt.show()
#generate bigram from comments body

comms_bod_bigram = generate_ngrams(comms_bod_na_cleaned, 2, 20)
#generate barplot for bigram

plt.figure(figsize=(12,8))

sns.barplot(comms_bod_bigram["wordcount"],comms_bod_bigram["word"])

plt.xlabel("Word Count", fontsize=15)

plt.ylabel("Bigrams", fontsize=15)

plt.title("Top 20 Bigrams from Comments Body")

plt.show()
#generate trigram from comments body

comms_bod_trigram = generate_ngrams(comms_bod_na_cleaned, 3, 20)
#generate barplot for bigram

plt.figure(figsize=(12,8))

sns.barplot(comms_bod_trigram["wordcount"],comms_bod_trigram["word"])

plt.xlabel("Word Count", fontsize=15)

plt.ylabel("Bigrams", fontsize=15)

plt.title("Top 20 Trigrams from Comments Body")

plt.show()
#convert to dataframe and rename the column of the ans_body_na_cleaned list

comms_bod_na_clean = pd.DataFrame(np.array(comms_bod_na_cleaned).reshape(-1))

comms_bod_na_clean.columns = ["ques_body"]

comms_bod_na_clean = comms_bod_na_clean.squeeze()



# Number of words in the question_body

comms_bod_na_clean["word_count"] = comms_bod_na_clean.apply(lambda x: len(str(x).split()))

fig, ax = plt.subplots(figsize=(15,2))

sns.boxplot(x="word_count", data= comms_bod_na_clean, ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')

ax.set_xlabel('Word Count', size=10, color="#0D47A1")

ax.set_ylabel('Comment Body text', size=10, color="#0D47A1")

ax.set_title('[Horizontal Box Plot] Word Count distribution for Comments Body', size=12, color="#0D47A1")

plt.gca().xaxis.grid(True)

plt.show()
# Number of stopwords in question_body

comms_bod_na_clean["stop_words_count"] = comms_bod_na_clean.apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

fig, ax = plt.subplots(figsize=(15,2))

sns.boxplot(x="stop_words_count", data=comms_bod_na_clean, ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')

ax.set_xlabel('Stop Word Count', size=10, color="#0D47A1")

ax.set_ylabel('Comments Body text', size=10, color="#0D47A1")

ax.set_title('[Horizontal Box Plot] Number of Stop Words distribution for Comments Body', size=12, color="#0D47A1")

plt.gca().xaxis.grid(True)

plt.show()
#Import and view information of the comments Dataset

tags = pd.read_csv('../input/tag_users.csv')

tags.info()

#Check for missing values

tags.isnull().sum()
#Identify the most popular tag_user_tag_id

tag_user = tags["tag_users_tag_id"].dropna()

print(tag_user.value_counts().head(5))
#Identify the most popular tag_user_tag_id

tag_user_id = tags["tag_users_user_id"].dropna()

print(tag_user_id.value_counts().head(5))