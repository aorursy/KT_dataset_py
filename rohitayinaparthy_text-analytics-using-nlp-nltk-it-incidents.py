### Download required packages

# import nltk
# nltk.download('gutenberg')
# nltk.download('genesis')
##### Imporing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

## Importing Textblob package
from textblob import TextBlob

# Importing CountVectorizer for sparse matrix/ngrams frequencies
from sklearn.feature_extraction.text import CountVectorizer

## Import datetime
import datetime as dt


import nltk.compat
import itertools



#### Checking on encoding
import chardet
##### Read the data file
filepath = "../input/ebi-finance-and-qlikview-incidents-data/Incident_2017_18_Final.csv"


## Checking the encoding factor
with open(filepath,"rb") as mydata:
    result = chardet.detect(mydata.read(1000000))
result
##### Read the data file
filepath = "../input/ebi-finance-and-qlikview-incidents-data/Incident_2017_18_Final.csv"
train_incidents = pd.read_csv(filepath,encoding="Windows-1252")

train_incidents["short_description_nwords"] = train_incidents["short_description"].apply(lambda x: len(str(x).split(" ")))

train_incidents[["short_description","short_description_nwords"]].sort_values(by = "short_description_nwords",ascending = True).head()


train_incidents[["short_description","short_description_nwords"]].sort_values(by = "short_description_nwords",ascending = False).head()
train_incidents["short_description_nchars"] = train_incidents["short_description"].str.len()

train_incidents[["short_description","short_description_nchars"]].sort_values(by = "short_description_nchars",ascending = False).head()

train_incidents[["short_description","short_description_nchars"]].sort_values(by = "short_description_nchars",ascending = True).head()
#sum of words/total words
def ave_word_len(sentence):
    words  = sentence.split(" ")
    return ((sum((len(word) for word in words))/len(words)))

train_incidents["short_description_avg_word_len"] = train_incidents["short_description"].apply(ave_word_len)
train_incidents[["short_description","short_description_avg_word_len"]].sort_values(by = "short_description_avg_word_len",ascending = True).head()
## Importing stop words from nltk.corpus
from nltk.corpus import stopwords
stop = stopwords.words("english")
train_incidents["short_description_nstopwords"] = train_incidents["short_description"].apply(lambda word: len([x for x in word.split(" ") if x in stop]))
train_incidents[["short_description","short_description_nstopwords"]].sort_values(by = "short_description_nstopwords",ascending = False).head()
train_incidents["short_description_ndigits"] = train_incidents["short_description"].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

train_incidents[["short_description","short_description_ndigits"]].sort_values(by = "short_description_ndigits",ascending = False).head()
train_incidents["short_description_nupper"] = train_incidents["short_description"].apply((lambda word: len([x for x in word.split() if x.isupper()])))
train_incidents[["short_description","short_description_nupper"]].sort_values(by = "short_description_nupper",ascending = False).head()
train_incidents["short_description"] = train_incidents["short_description"].apply(lambda x: x.lower())
train_incidents["short_description"].head()

train_incidents["short_description"] = train_incidents["short_description"].str.replace("qlik view","qlikview")
train_incidents["short_description"] = train_incidents["short_description"].str.replace("qv","qlikview")
train_incidents["short_description"] = train_incidents["short_description"].str.replace("wrongly","wrong")


train_incidents["short_description"] = train_incidents["short_description"].str.replace("[^\w\s]","")
train_incidents["short_description"].tail()
train_incidents["short_description"] = train_incidents["short_description"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

from textblob import Word
train_incidents["short_description"] = train_incidents["short_description"].apply(lambda x: " ".join([Word(myword).lemmatize() for myword in x.split()])  )
train_incidents["short_description"].head(5)
### Most frequent words in short description
Short_description_most_freq_words = pd.Series(" ".join(train_incidents["short_description"]).split()).value_counts()
Short_description_most_freq_words.head(20)

### Least frequent words in short description
short_description_least_freq_words =  pd.Series(" ".join(train_incidents["short_description"]).split()).value_counts().sort_values(ascending = True)
short_description_least_freq_words.head(10)
## Correction for top 10 sentences
##  train_incidents["short_description"] = train_incidents["short_description"].apply(lambda x: str(TextBlob(x).correct()))
TextBlob(train_incidents["short_description"][1]).words
train_incidents["short_description"][1]
train_incidents["short_description_tokens"] =  train_incidents["short_description"].apply(lambda x: TextBlob(x).words)
train_incidents["short_description_tokens"].head(10)
from nltk import word_tokenize,sent_tokenize
train_incidents["short_description"].apply(lambda x: word_tokenize(x))
from nltk.stem import PorterStemmer

st = PorterStemmer()

train_incidents["short_description"][:5].apply(lambda words: " ".join([st.stem(word) for word in words.split()]))


train_incidents["sys_created_on"] = (pd.to_datetime(train_incidents["sys_created_on"],format='%d/%m/%Y %H:%M'))
train_incidents["sys_updated_on"] = (pd.to_datetime(train_incidents["sys_updated_on"],format='%d/%m/%Y %H:%M'))
train_incidents["opened_at"] = (pd.to_datetime(train_incidents["opened_at"],format='%d/%m/%Y %H:%M'))
train_incidents["resolved_at"] = (pd.to_datetime(train_incidents["resolved_at"],format='%d/%m/%Y %H:%M'))

### Extracting dates from datetime object
train_incidents["opened_at_date"] = train_incidents["opened_at"].dt.date


## Creating Category GROUPBY Object
incidents_category = train_incidents.groupby("category")
## Creating sub Category GROUPBY Object
incidents_incident_subcategory = train_incidents.groupby("incident_subcategory")
## Creating priority GROUPBY Object
incidents_priority= train_incidents.groupby("priority")
## Creating priority GROUPBY Object
incidents_urgency= train_incidents.groupby("urgency")
## Creating re-open GROUPBY Object
incidents_reopen_count= train_incidents.groupby("reopen_count")
## Creating made_sla GROUPBY Object
incidents_made_sla= train_incidents.groupby("made_sla")
## Creating incident type GROUPBY Object
incidents_type= train_incidents.groupby("incident_type")


## Creating impact GROUPBY Object
incidents_impact= train_incidents.groupby("impact")

## Creating Escalations GROUPBY Object
incidents_escalation= train_incidents.groupby("escalation")

## Creating E2E resolution met Object
incidents_e2e_resolution_met= train_incidents.groupby("e2e_resolution_met")

## Creating location Object
incidents_location = train_incidents.groupby("current_location")

## Creating location Object
incidents_country = train_incidents.groupby("country")

## Creating contact type Object
incidents_contact_type = train_incidents.groupby("contact_type")

## Creating affected user Object
incidents_affected_user = train_incidents.groupby("affected_user")

## Creating assigned group Object
incidents_assignment_group = train_incidents.groupby("assignment_group")









### Analyzing top 20 frequent words


sd_freq_plot = Short_description_most_freq_words.head(20).sort_values(ascending = True).plot(kind="barh",title = "Top 20 Frequent Number Of Words")

plt.style.use("ggplot")
sd_freq_plot.set_xlabel("Frequency")
sd_freq_plot.set_ylabel("Terms")

totals = []
for i in sd_freq_plot.patches:
    totals.append(i.get_width())

for i in sd_freq_plot.patches:
    sd_freq_plot.text(i.get_width()+.3,i.get_y()+0.1,str(i.get_width()),fontsize = 8,color= 'black')
    


### Lets generate bigrams and store it in a bi_grams variable
### train_incidents["bi_grams"] = train_incidents["short_description"].apply(lambda x: TextBlob(x).ngrams(2))
### train_incidents["bi_grams"].head()

### Lets generate trigrams and store it in a tri_grams variable
### train_incidents["tri_grams"] = train_incidents["short_description"].apply(lambda x: TextBlob(x).ngrams(3))
### train_incidents["tri_grams"].head()
bigrams = TextBlob(" ".join(train_incidents["short_description"])).ngrams(2)
#bigrams = pd.Series(bigrams).apply(lambda x: list(x))
word_vectorizer = CountVectorizer(ngram_range=(2,2), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(train_incidents["short_description"])
frequencies = sum(sparse_matrix).toarray()[0]
bi_grams_df = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])
    
bi_grams_df.sort_values(by = "frequency",ascending=False).head(20)

#grams_df[grams_df.index.str.contains("reconciliation")]
### Analyzing top 20 frequent BI Gram words

plt.style.use("ggplot")
plt.xlabel("Frequency",)
plt.ylabel("Terms")
top20_bigrams = bi_grams_df["frequency"].sort_values(ascending = False).head(20)

top20_bigrams.head(20).sort_values(ascending = True).plot(kind="barh",title = "Top 20 Frequent Bi Grams")


train_incidents_word_issue = train_incidents[train_incidents["short_description"].str.contains("issue")]

train_incidents_word_issue["short_description"] = train_incidents_word_issue["short_description"].str.replace("issue qlikview","qliview issue")
train_incidents_word_issue["short_description"] = train_incidents_word_issue["short_description"].str.replace("issue data","data issue")
train_incidents_word_issue["short_description"] = train_incidents_word_issue["short_description"].str.replace("issue mrh","mrh issue")
train_incidents_word_issue["short_description"] = train_incidents_word_issue["short_description"].str.replace("issue dashboard","dashboard issue")
train_incidents_word_issue["short_description"] = train_incidents_word_issue["short_description"].str.replace("issue access","access issue")
train_incidents_word_issue["short_description"] = train_incidents_word_issue["short_description"].str.replace("issue query","query issue")
train_incidents_word_issue["short_description"] = train_incidents_word_issue["short_description"].str.replace("issue ebi","ebi issue")
train_incidents_word_issue["short_description"] = train_incidents_word_issue["short_description"].str.replace("issue report","report issue")
train_incidents_word_issue["short_description"] = train_incidents_word_issue["short_description"].str.replace("issue hr","hr issue")
train_incidents_word_issue["short_description"] = train_incidents_word_issue["short_description"].str.replace("issue mrh2","mrh2 issue")
train_incidents_word_issue["short_description"] = train_incidents_word_issue["short_description"].str.replace("issue master","master issue")
train_incidents_word_issue["short_description"] = train_incidents_word_issue["short_description"].str.replace("issue file","file issue")


word_vectorizer = CountVectorizer(ngram_range=(2,2), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(train_incidents_word_issue["short_description"])
frequencies = sum(sparse_matrix).toarray()[0]
bi_grams_issue_df = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])
    
bi_grams_issue_df[bi_grams_issue_df.index.str.contains("issue")].sort_values(by = "frequency",ascending=False).head(10)

### Analyzing top 20 frequent BI Gram words- word containing issue

plt.style.use("ggplot")
plt.xlabel("Frequency")
plt.ylabel("Terms")
plt.title("Top 10 Frequent Bi Grams contains word ""issue""")
top20_bigrams_issue = bi_grams_issue_df["frequency"].sort_values(ascending = False)

top20_bigrams_issue_plot = top20_bigrams_issue[top20_bigrams_issue.index.str.contains("issue")].head(10).sort_values(ascending = True).plot(kind="barh")

totals = []
for i in top20_bigrams_issue_plot.patches:
    totals.append(i.get_width())

for i in top20_bigrams_issue_plot.patches:
    top20_bigrams_issue_plot.text(i.get_width()+.3,i.get_y()+0.1,str(i.get_width()),fontsize = 10,color= 'black')


train_incidents_word_issue.head(1)
train_incidents_word_issue["data_issue_count"] = ""


word_vectorizer = CountVectorizer(ngram_range=(2,2), analyzer='word')
for each in (train_incidents_word_issue["short_description"].index):
    text_issue_list = [train_incidents_word_issue["short_description"][each]]
    sparse_matrix = word_vectorizer.fit_transform(text_issue_list)
    frequencies = sum(sparse_matrix).toarray()[0]
    bi_grams_issue_df = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])
    train_incidents_word_issue["data_issue_count"][each] = bi_grams_issue_df[bi_grams_issue_df.index.str.contains("^data issue$")]["frequency"].sum()
    

### Occurance of Word "data issue" in "short description" VS categories
issue_category_groupby = train_incidents_word_issue.groupby(by="category")
issue_category_groupby["data_issue_count"].sum().sort_values(ascending = False)
### Word "data issue" in "short description" VS sub categories
issue_subcategory_groupby = train_incidents_word_issue.groupby(by="incident_subcategory")
issue_subcategory_groupby["data_issue_count"].sum().sort_values(ascending = False)
### Word "data issue" in "short description" VS impact
issue_impact_groupby = train_incidents_word_issue.groupby(by="impact")
issue_impact_plot= issue_impact_groupby["data_issue_count"].sum()
issue_impact_plot
### Word "data issue" in "short description" VS source
issue_assignment_groupby = train_incidents_word_issue.groupby(by="assignment_group")
issue_assignment_freq= issue_assignment_groupby["data_issue_count"].sum()
issue_assignment_freq
### Word "data issue" in "short description" VS location
issue_location_groupby = train_incidents_word_issue.groupby(by="current_location")
issue_location_freq= issue_location_groupby["data_issue_count"].sum()
issue_location_freq
### Word "data issue" in "short description" VS Contact type
issue_contact_type_groupby = train_incidents_word_issue.groupby(by="contact_type")
issue_contact_type_freq= issue_contact_type_groupby["data_issue_count"].sum()
issue_contact_type_freq
### Word "data issue" in "short description" VS Syngenta location
issue_syn_loc_groupby = train_incidents_word_issue.groupby(by="syngenta_location")
issue_syn_loc_freq= issue_syn_loc_groupby["data_issue_count"].sum()
issue_syn_loc_freq.sort_values(ascending=False)

train_incidents_word_issue["top10_issue_count"] = ""

word_vectorizer = CountVectorizer(ngram_range=(2,2), analyzer='word')
for each in (train_incidents_word_issue["short_description"].index):
    text_issue_list = [train_incidents_word_issue["short_description"][each]]
    sparse_matrix = word_vectorizer.fit_transform(text_issue_list)
    frequencies = sum(sparse_matrix).toarray()[0]
    bi_grams_top_10_issue_df = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])
    train_incidents_word_issue["top10_issue_count"][each] = bi_grams_top_10_issue_df[bi_grams_top_10_issue_df.index.str.contains("^data issue$")]["frequency"].sum() + bi_grams_top_10_issue_df[bi_grams_top_10_issue_df.index.str.contains("^dashboard issue$")]["frequency"].sum() + bi_grams_top_10_issue_df[bi_grams_top_10_issue_df.index.str.contains("^query issue$")]["frequency"].sum() + bi_grams_top_10_issue_df[bi_grams_top_10_issue_df.index.str.contains("^mapping issue$")]["frequency"].sum() + bi_grams_top_10_issue_df[bi_grams_top_10_issue_df.index.str.contains("^access issue$")]["frequency"].sum() + bi_grams_top_10_issue_df[bi_grams_top_10_issue_df.index.str.contains("^mrh issue$")]["frequency"].sum() + bi_grams_top_10_issue_df[bi_grams_top_10_issue_df.index.str.contains("^file issue$")]["frequency"].sum() + bi_grams_top_10_issue_df[bi_grams_top_10_issue_df.index.str.contains("^ebi issue$")]["frequency"].sum() + bi_grams_top_10_issue_df[bi_grams_top_10_issue_df.index.str.contains("^report issue$")]["frequency"].sum() + bi_grams_top_10_issue_df[bi_grams_top_10_issue_df.index.str.contains("^qlikview issue$")]["frequency"].sum()
    
issue_impact_groupby["top10_issue_count"].sum()
plt.style.use("ggplot")
plt.title("Top 10 issues VS tickets Priority")
plt.ylabel("Top 10 Issues")
plt.xlabel("Tickets Priority")
issue_priority_groupby = train_incidents_word_issue.groupby("priority")
issue_priority_groupby["top10_issue_count"].sum().plot(kind = "bar")


train_incidents_word_issue["opened_at_date"].head(2)

train_incidents_word_issue["opened_at_year"] = train_incidents_word_issue["opened_at"].dt.year
# Top 10 data issues VS ticket Opened Year Analysis
plt.style.use("ggplot")
plt.title("Top 10 data issues VS ticket Opened Year Analysis")
plt.ylabel("Top 10 Issues")
issue_opened_at_year_grpby = train_incidents_word_issue.groupby("opened_at_year")
issue_opened_at_year_grpby["top10_issue_count"].sum().plot(kind ="bar")
plt.xlabel("Tickets Opened at Year")



word_vectorizer = CountVectorizer(ngram_range=(3,3), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(train_incidents["short_description"])
#sparse_matrix = word_vectorizer.fit_transform(train_incidents["short_description_tokens"])
frequencies = sum(sparse_matrix).toarray()[0]
tri_grams_df = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])



tri_grams_df.sort_values(by = "frequency",ascending=False).head(20)

##grams_df[grams_df.index.str.contains("reconciliation")]
### Analyzing top 20 frequent Tri Gram words

plt.style.use("ggplot")
plt.xlabel("Terms",)
plt.ylabel("Frequency")
trigrams_short_description = tri_grams_df["frequency"].sort_values(ascending = False)
top20_trigrams = tri_grams_df["frequency"].sort_values(ascending = False).head(20)

top5_trigrams_plot =  top20_trigrams.head(5).sort_values(ascending = False).plot(kind="bar",title = "Top 5 Frequent Tri Grams")
top5_trigrams_plot
plt.xticks(rotation=75)



#### Find N grams for category - incident

word_vectorizer = CountVectorizer(ngram_range=(2,2), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(incidents_category.get_group("Incident")["short_description"])
frequencies_Incident_cate = sum(sparse_matrix).toarray()[0]
grams_df_incident_cate = pd.DataFrame(frequencies_Incident_cate, index=word_vectorizer.get_feature_names(), columns=['Incident_category_frequency'])
grams_df_incident_cate.sort_values(by = "Incident_category_frequency",ascending= False).head(10)

#### Find N grams for category - Request

word_vectorizer = CountVectorizer(ngram_range=(2,2), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(incidents_category.get_group("Request")["short_description"])
frequencies_Request_cate = sum(sparse_matrix).toarray()[0]
grams_df_Request_cate = pd.DataFrame(frequencies_Request_cate, index=word_vectorizer.get_feature_names(), columns=['Request_category_frequency'])
grams_df_Request_cate.sort_values(by = "Request_category_frequency",ascending= False)


plt.style.use("ggplot")
plt.xlabel("Frequency",)
plt.ylabel("Bi Grams")
grams_Request_cate = grams_df_Request_cate["Request_category_frequency"].sort_values(ascending = False).head(20)
grams_Request_cate.sort_values(ascending = True).plot(kind="barh",title = "Request Category - Top 20 Frequent Number Of Words")
plt.style.use("ggplot")
plt.xlabel("Frequency",)
plt.ylabel("Bi Grams")
grams_df_incident_cate = grams_df_incident_cate["Incident_category_frequency"].sort_values(ascending = False).head(20)
grams_df_incident_cate.sort_values(ascending = True).plot(kind="barh",title = "Incident Category - Top 20 Frequent Number Of Words")


train_incidents["short_desc_report_count"]  = train_incidents["short_description_tokens"].apply(lambda x: list(x).count("report"))
incidents_category["short_desc_report_count"].sum()

train_incidents["short_desc_authorization_count"] = train_incidents["short_description_tokens"].apply(lambda x: list(x).count("authorization"))

incidents_category["short_desc_authorization_count"].sum()
incidents_escalation["short_desc_authorization_count"].sum()
incidents_type["short_desc_authorization_count"].sum()


train_incidents_sorted_opened_at_df  = train_incidents.sort_values(by = "opened_at")
train_incidents_sorted_opened_at_df.shape

#nltk.download('inaugural')
#nltk.download('nps_chat')
#nltk.download('webtext')
#nltk.download('treebank')

short_desc_tokens_series = train_incidents["short_description_tokens"].apply(lambda x: list(x))
short_desc_tokens_series = short_desc_tokens_series.tolist()
short_desc_tokens_series

#short_desc_tokens_list = list(itertools.chain.from_iterable(short_desc_tokens_series))   
#short_desc_tokens_list

short_desc_tokens_series = train_incidents["short_description_tokens"].apply(lambda x: list(x))
short_desc_tokens_series = short_desc_tokens_series.tolist()
short_desc_tokens_series

short_desc_tokens_list = list(itertools.chain.from_iterable(short_desc_tokens_series))   
short_desc_tokens_list

plt.figure(figsize=(16,5))
## Make the list as NLTK object
short_desc_tokens_list = nltk.Text(short_desc_tokens_list)

topics = ['authorization', 'access', 'reload',"qlikview","mismatch","reconciliation","ebi","mapping","report"]
short_desc_tokens_list.dispersion_plot(topics)




###
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# datetime parser
filepath2 = "../input/it-incidents-tokens-vs-date-fields-data/short_description_token_vs_open_at_dates.csv"

with open(filepath2,'rb') as filep:
    result2 = chardet.detect(filep.read(1000000))
    result2

result2

sd_token_timeseries = pd.read_csv(filepath2,encoding="Windows-1252")

sd_token_timeseries["Opened_at"] = (pd.to_datetime(sd_token_timeseries["Opened_at"],format = '%d/%m/%Y'))
## Delete duplicates value of all the rows
sd_token_timeseries = sd_token_timeseries.drop_duplicates()

sd_token_timeseries.head


### Filter the tokens data containing selected tokens to analyze with open dates
#selected_tokens_mask = sd_token_timeseries["Short_desc_tokens"].str.contains("access|reload",regex = True)

#sd_token_timeseries["Short_desc_selected_tokens"] = np.where(selected_tokens_mask,sd_token_timeseries["Short_desc_tokens"])

sd_token_timeseries["Short_desc_selected_tokens"] = sd_token_timeseries["Short_desc_tokens"].str.extract("("+'authorization|reload|mismatch|reconciliation|access|qlikview|ebi|query|report|mapping'+")",expand = False)

sd_token_timeseries_updated = sd_token_timeseries.dropna()
sd_token_timeseries_updated.drop_duplicates(subset=["Opened_at","Short_desc_selected_tokens"],keep="first")
# set size of figure
plt.figure(figsize=(16,10))

# use horizontal stripplot with x marker size of 5
sns.stripplot(y='Short_desc_selected_tokens',x='Opened_at', data=sd_token_timeseries_updated,
 orient='h', marker='^', color='navy', size=4)
# rotate x tick labels
plt.xticks(rotation=50,size= 15)
plt.yticks(size= 15)
# remover borders of plot
plt.style.use("ggplot")
plt.tight_layout()
plt.title("Tokens VS tickets Open Date - Time Series Analysis",size= 30)
plt.ylabel("Issues",size = 20)
plt.xlabel("Tickets Opened at",size = 20)
plt.show()

# set size of figure
plt.figure(figsize=(16,10))


# use horizontal stripplot with x marker size of 5
sns.stripplot(y='incident_subcategory',x='opened_at', data=train_incidents,
 orient='h', marker='X', color='navy', size=4)
# rotate x tick labels
plt.xticks(rotation=50)
# remover borders of plot
plt.style.use("ggplot")
plt.tight_layout()
plt.title("Incident Sub-Category VS tickets Open Date - Time Series Analysis")
plt.ylabel("Sub category")
plt.xlabel("Tickets Opened at")
plt.show()




# set size of figure

plt.figure(figsize=(16,10))

# use horizontal stripplot with x marker size of 5
sns.stripplot(y='priority',x='opened_at', data=train_incidents,
 orient='h', marker='X', color='navy', size=4)
# rotate x tick labels
plt.xticks(rotation=50)
# remover borders of plot
plt.style.use("ggplot")
plt.tight_layout()
plt.title("Incident Priority VS tickets Open Date - Time Series Analysis")
plt.ylabel("Priority")
plt.xlabel("Tickets Opened at")
plt.show()


# set size of figure

plt.figure(figsize=(16,10))

# use horizontal stripplot with x marker size of 5
sns.stripplot(y='close_code',x='opened_at', data=train_incidents,
 orient='h', marker='X', color='navy', size=4)
# rotate x tick labels
plt.xticks(rotation=50)
# remover borders of plot
plt.style.use("ggplot")
plt.tight_layout()
plt.title("Ticket closure status VS tickets Open Date - Time Series Analysis")
plt.ylabel("Tickets Closure Status")
plt.xlabel("Tickets Opened at")
plt.show()

train_incidents["selected_bi_grams_text"] = train_incidents["short_description"].str.extract("("+'net sale|ebi report|mrh report|sale report|qlikview dashboard|sale broadcast|tp tool|daily sale|mrh query|demand review|access request|review dashboard|data issue|edwh ebi|edwh report|complaint valid|sale data|fr qlikview|3rd party|incorrect data'+")",expand = False)
train_incidents["selected_bi_grams_text"].head()
train_incidents["selected_bi_grams_text"] = train_incidents["selected_bi_grams_text"].apply(lambda x: str(x))
#train_incidents["selected_bi_grams_text"].str.replace(np.nan,"",regex= True)
train_incidents.selected_bi_grams_text.fillna("",inplace=True)
train_incidents["top_20_bi_grams_list"] = " "

 
word_vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b",stop_words=None,ngram_range=(2,2), analyzer='word')
for each in (train_incidents["selected_bi_grams_text"].index):
    if (train_incidents["selected_bi_grams_text"][each] != 'nan'):
        text_list = [train_incidents["selected_bi_grams_text"][each]]
        sparse_matrix = word_vectorizer.fit_transform(text_list)
        df11 = pd.DataFrame(word_vectorizer.get_feature_names(), columns=['bi_grams'])
        train_incidents["top_20_bi_grams_list"][each] = list(df11.bi_grams)

## Lets check the format of the data
train_incidents["top_20_bi_grams_list"].head(1)
bi_grams_date_df =  train_incidents[["opened_at_date","top_20_bi_grams_list"]]
#bi_grams_date_df["top20_bi_grams_list"].apply(lambda x: str(x))
bi_grams_date_df["top_20_bi_grams_list"] = bi_grams_date_df["top_20_bi_grams_list"].apply(lambda x: "".join(x))
#bi_grams_date_df[bi_grams_date_df.isnull]
bi_grams_date_df = bi_grams_date_df[bi_grams_date_df["top_20_bi_grams_list"] != " "]

# set size of figure
plt.figure(figsize=(16,10))

# use horizontal stripplot with x marker size of 5
sns.stripplot(y='top_20_bi_grams_list',x='opened_at_date', data=bi_grams_date_df,
 orient='h', marker='^', color='navy', size=4)
# rotate x tick labels
plt.xticks(rotation=50,size= 15)
plt.yticks(size= 15)
# remover borders of plot
plt.style.use("ggplot")
plt.tight_layout()
plt.title("Bi Grams VS Tickets Open Date - Time Series Analysis",size= 30)
plt.ylabel("Bi Grams",size = 20)
plt.xlabel("Tickets Opened at",size = 20)
plt.show()

### Its observed majority of the bigrams piled up during 2018 compare to 2017 and few terms like demand review,FR Qlikview are not seen in 2017 at all
train_incidents["selected_tri_grams_text"] = train_incidents["short_description"].str.extract("("+'net sale broadcast|demand review dashboard|global edwh ebi|hr master file|3rd party net|daily sale report|party net sale|manual file upload|ea field crop|file upload qlikview|crop manual file|apac ea field|field crop manual|qlikview demand review|global qlikview demand|qlik demand dashboard|demand dashboard refresh|incident apac ea|related qlik demand|wd activity related'+")",expand = False)
train_incidents["selected_tri_grams_text"].head()
train_incidents["selected_tri_grams_text"] = train_incidents["selected_tri_grams_text"].apply(lambda x: str(x))
train_incidents.selected_tri_grams_text.fillna("",inplace=True)
train_incidents["top_20_tri_grams_list"] = " "

 
word_vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b",stop_words=None,ngram_range=(3,3), analyzer='word')
for each in (train_incidents["selected_tri_grams_text"].index):
    if (train_incidents["selected_tri_grams_text"][each] != 'nan'):
        text_list = [train_incidents["selected_tri_grams_text"][each]]
        sparse_matrix = word_vectorizer.fit_transform(text_list)
        df12 = pd.DataFrame(word_vectorizer.get_feature_names(), columns=['tri_grams'])
        train_incidents["top_20_tri_grams_list"][each] = list(df12.tri_grams)

## Lets check the format of the data
train_incidents["top_20_tri_grams_list"].tail(5)
tri_grams_date_df =  train_incidents[["opened_at_date","top_20_tri_grams_list"]]
tri_grams_date_df["top_20_tri_grams_list"] = tri_grams_date_df["top_20_tri_grams_list"].apply(lambda x: "".join(x))
tri_grams_date_df = tri_grams_date_df[tri_grams_date_df["top_20_tri_grams_list"] != " "]

# set size of figure
plt.figure(figsize=(16,10))

# use horizontal stripplot with x marker size of 5
sns.stripplot(y='top_20_tri_grams_list',x='opened_at_date', data=tri_grams_date_df,
 orient='h', marker='^', color='navy', size=4)
# rotate x tick labels
plt.xticks(rotation=50,size= 15)
plt.yticks(size= 15)
# remover borders of plot
plt.style.use("ggplot")
plt.tight_layout()
plt.title("Tri Grams VS Tickets Open Date - Time Series Analysis",size= 30)
plt.ylabel("Tri Grams",size = 20)
plt.xlabel("Tickets Opened at",size = 20)
plt.show()



train_incidents["bi_grams_contains_issue"] = train_incidents["short_description"].str.extract("("+'data issue|dashboard issue|mapping issue|query issue|access issue|mrh issue|file issue|ebi issue|report issue|mrh2 issue'+")",expand = False)
train_incidents["bi_grams_contains_issue"].head()
train_incidents["bi_grams_contains_issue"] = train_incidents["bi_grams_contains_issue"].apply(lambda x: str(x))
train_incidents.bi_grams_contains_issue.fillna("",inplace=True)
train_incidents["top_10_bi_grams_issue_list"] = " "

 
word_vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b",stop_words=None,ngram_range=(2,2), analyzer='word')
for each in (train_incidents["bi_grams_contains_issue"].index):
    if (train_incidents["bi_grams_contains_issue"][each] != 'nan'):
        text_list = [train_incidents["bi_grams_contains_issue"][each]]
        sparse_matrix = word_vectorizer.fit_transform(text_list)
        df13 = pd.DataFrame(word_vectorizer.get_feature_names(), columns=['bi_grams_contains_issues'])
        train_incidents["top_10_bi_grams_issue_list"][each] = list(df13.bi_grams_contains_issues)

## Lets check the format of the data
train_incidents["top_10_bi_grams_issue_list"][2274]
bi_grams_issue_date_df =  train_incidents[["opened_at_date","top_10_bi_grams_issue_list"]]
bi_grams_issue_date_df["top_10_bi_grams_issue_list"] = bi_grams_issue_date_df["top_10_bi_grams_issue_list"].apply(lambda x: "".join(x))
bi_grams_issue_date_df = bi_grams_issue_date_df[bi_grams_issue_date_df["top_10_bi_grams_issue_list"] != " "]


# set size of figure
plt.figure(figsize=(16,10))

# use horizontal stripplot with x marker size of 5
sns.stripplot(y='top_10_bi_grams_issue_list',x='opened_at_date', data=bi_grams_issue_date_df,
 orient='h', marker='^', color='navy', size=4)
# rotate x tick labels
plt.xticks(rotation=50,size= 15)
plt.yticks(size= 15)
# remover borders of plot
plt.style.use("ggplot")
plt.tight_layout()
plt.title("Bi Grams contains word 'Issue' VS Tickets Open Date - Time Series Analysis",size= 30)
plt.ylabel("Issue Bi Grams",size = 20)
plt.xlabel("Tickets Opened at",size = 20)
plt.show()

### Its observed data issues count were low in 2018 compared to 2017.However,report and mrh2 issue are occured more in 2018

train_incidents["sentiments"] = train_incidents["short_description"].apply(lambda x: TextBlob(x).sentiment[0])

train_incidents[["short_description","sentiments","short_description_tokens"]].sort_values(by = "sentiments",ascending = True)
incidents_impact["sentiments"].sum()
incidents_impact["sentiments"].sum().plot(kind= "bar")
plt.title("Sentiment Polarity VS Incident Impact")
plt.xlabel("Impact")
plt.ylabel("Polarity")
plt.xticks(rotation = "0.5")

incidents_category["sentiments"].sum().sort_values(ascending = True)
incidents_incident_subcategory["sentiments"].sum().sort_values(ascending = True)
incidents_assignment_group["sentiments"].sum()
incidents_assignment_group["sentiments"].sum().plot(kind ="bar",color= ["pink","brown"])
plt.title("Application Wise Sentiment Polarity Analysis")
plt.xlabel("Application")
plt.ylabel("Polarity")
plt.xticks(rotation = "0.5")

## Defining sentiment type based on polarity
def sentiment_type(value):
    if value >= 0.5:
        return "Positive"
    elif value <= -0.5:
        return "Negitive"
    else:
        return "Neutral"
train_incidents["sentiment_types"] = train_incidents["sentiments"].apply(sentiment_type)
train_incidents["sentiment_types"].value_counts()
train_incidents["sentiment_types"].value_counts().plot(kind = "bar",color = ["blue","red","green"])
plt.title("Sentiment types classification frequency")
plt.xlabel("Sentiment Types")
plt.ylabel("Frequency")
plt.xticks(rotation = "0.5")

#### Sentiment types Vs incident category analysis 
incident_cate_senti_type_grpby = train_incidents.groupby(["category","sentiment_types"])
df_ramdom = incident_cate_senti_type_grpby["number"].count().to_frame(name = "count")

df_ramdom.unstack(1)

# Importing SentimentIntensityAnalyzer  --- Method 2 -- in which we can calculate sentiment polarity
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# sia = SentimentIntensityAnalyzer()

## Function to hold positive sentiments and its words
def sentiment_type_words_fun_positive(words):
    mysentilist = []
    for word in words.split(" "):
        if (TextBlob(word).sentiment[0]) >= 0.5:
            mysentilist.append(word)
    return mysentilist    
    
## Function to hold negitive sentiments and its words
def sentiment_type_words_fun_negitive(words):
    mysentilist = []
    for word in words.split(" "):
        if (TextBlob(word).sentiment[0]) <= -0.5:
            mysentilist.append(word)
    return mysentilist    
    
## Function to hold neutral sentiments and its words
def sentiment_type_words_fun_neutral(words):
    mysentilist = []
    for word in words.split(" "):
        if ((TextBlob(word).sentiment[0]) > -0.5 and (TextBlob(word).sentiment[0]) < 0.5):
            mysentilist.append(word)
    return mysentilist    
    
### List of all postive sentimental words found in text "short description"
train_incidents["sentiment_types_postive_words"] = train_incidents["short_description"].apply(sentiment_type_words_fun_positive)

postive_senti_df = pd.DataFrame(train_incidents["sentiment_types_postive_words"].apply(lambda x: "".join(x)).value_counts())
postive_senti_df[postive_senti_df["sentiment_types_postive_words"] != 2271].plot(kind = "bar",color = "green")
plt.title("Most observed Positive words")
plt.xlabel("Positive Words")
plt.ylabel("Frequency")
plt.xticks(rotation = "0.5")


### List of all negitive sentimental words found in text "short description"
train_incidents["sentiment_types_negitive_words"] = train_incidents["short_description"].apply(sentiment_type_words_fun_negitive)

negitive_df = pd.DataFrame((train_incidents["sentiment_types_negitive_words"].apply(lambda x: "".join(x))).value_counts())
negitive_df[negitive_df["sentiment_types_negitive_words"] != 2199].head(5).plot(kind = "bar")
plt.title("Most observed negitive words")
plt.xlabel("Negitive Words")
plt.ylabel("Frequency")
plt.xticks(rotation = "0.5")

mask1 = (train_incidents["sentiment_types"] == "Negitive")

negitive_sentiment_data_df = train_incidents[["short_description","sentiments","sentiment_types_negitive_words"]][mask1].sort_values(by = "sentiments",ascending = True)

## Collecting top 5 negitive words into a list
top5_negitive_words_list = negitive_df.index[1:6]
top5_negitive_words_list = list(top5_negitive_words_list)
top5_negitive_words_list

negitive_sentiment_data_df.head()
word_vectorizer = CountVectorizer(ngram_range=(2,2), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(negitive_sentiment_data_df["short_description"])
frequencies = sum(sparse_matrix).toarray()[0]
bi_grams_negitive_df = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])

#### Bi grams of Negitive Sentiment Words
bi_grams_negitive_df = bi_grams_negitive_df[bi_grams_negitive_df.index.str.contains("wrong|unable|failed|bad|inconvenient")].sort_values(by = "frequency",ascending= False).head(20)
bi_grams_negitive_df
### Analyzing top 20 frequent Negitive BI Gram words

plt.style.use("ggplot")
plt.xlabel("Frequency")
plt.ylabel("Negitive Bi Grams")
top20_negitive_bigrams = bi_grams_negitive_df["frequency"].sort_values(ascending = False).head(20)

top20_negitive_bigrams.head(20).sort_values(ascending = True).plot(kind="barh",title = "Top 20 Frequent Negitive Bi Grams")
####### Negitive Sentiment Trigrams
word_vectorizer = CountVectorizer(ngram_range=(3,3), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(negitive_sentiment_data_df["short_description"])
frequencies = sum(sparse_matrix).toarray()[0]
tri_grams_negitive_df = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])


#### tri grams of Negitive Sentiment Words
tri_grams_negitive_df = tri_grams_negitive_df[tri_grams_negitive_df.index.str.contains("wrong|unable|failed|bad|inconvenient")].sort_values(by = "frequency",ascending= False).head(20)
tri_grams_negitive_df
### Analyzing top 20 frequent Negitive TRI Gram words

plt.style.use("ggplot")
plt.xlabel("Frequency")
plt.ylabel("Negitive Tri Grams")
top20_negitive_trigrams = tri_grams_negitive_df["frequency"].sort_values(ascending = False).head(20)

top20_negitive_trigrams.head(20).sort_values(ascending = True).plot(kind="barh",title = "Top 20 Frequent Negitive Tri Grams")

## SInce Tri grams above seems like have common grams revolving aroung top negitive words, 
## so higher grams would make sense to check any patterns
word_vectorizer = CountVectorizer(ngram_range=(4,4), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(negitive_sentiment_data_df["short_description"])
frequencies = sum(sparse_matrix).toarray()[0]
negitive_sentences_df = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])


#### tri grams of Negitive Sentiment Words
negitive_sentences_df = negitive_sentences_df[negitive_sentences_df.index.str.contains("wrong|unable|failed|bad|inconvenient")].sort_values(by = "frequency",ascending= False).head(20)
negitive_sentences_df
### Analyzing top 20 frequent Negitive TRI Gram words

plt.style.use("ggplot")
plt.xlabel("Frequency")
plt.ylabel("Negitive Tri Grams")
top20_negitive_trigrams = tri_grams_negitive_df["frequency"].sort_values(ascending = False).head(20)

top20_negitive_trigrams.head(20).sort_values(ascending = True).plot(kind="barh",title = "Top 20 Frequent Negitive Tri Grams")

mask1 = (train_incidents["sentiment_types"] == "Positive")

positive_sentiment_data_df = train_incidents[["short_description","sentiments","sentiment_types_postive_words"]][mask1].sort_values(by = "sentiments",ascending = True)

positive_sentiment_data_df.head()
word_vectorizer = CountVectorizer(ngram_range=(2,2), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(positive_sentiment_data_df["short_description"])
frequencies = sum(sparse_matrix).toarray()[0]
bi_grams_positive_df = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])

#### Bi grams of positive Sentiment Words
bi_grams_positive_df = bi_grams_positive_df[bi_grams_positive_df.index.str.contains("able|latest|good|successful|many|best|sure")].sort_values(by = "frequency",ascending= False).head(20)
bi_grams_positive_df
### Analyzing top 20 frequent Negitive BI Gram words

plt.style.use("ggplot")
plt.xlabel("Frequency")
plt.ylabel("Positive Bi Grams")
top20_positive_bigrams = bi_grams_positive_df["frequency"].sort_values(ascending = False).head(20)

top20_positive_bigrams.head(20).sort_values(ascending = True).plot(kind="barh",title = "Top 20 Frequent Positive Bi Grams")
####### Negitive Sentiment Trigrams
word_vectorizer = CountVectorizer(ngram_range=(3,3), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(positive_sentiment_data_df["short_description"])
frequencies = sum(sparse_matrix).toarray()[0]
tri_grams_positive_df = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])


''#### tri grams of Positive Sentiment Words
tri_grams_positive_df = tri_grams_positive_df[tri_grams_positive_df.index.str.contains("able|latest|good|successful|many|best|sure")].sort_values(by = "frequency",ascending= False).head(20)
tri_grams_positive_df
### Analyzing top 20 frequent Positive TRI Gram words

plt.style.use("ggplot")
plt.xlabel("Frequency")
plt.ylabel("Positive Tri Grams")
top20_positive_trigrams = tri_grams_positive_df["frequency"].sort_values(ascending = False).head(20)

top20_positive_trigrams.head(20).sort_values(ascending = True).plot(kind="barh",title = "Top 20 Frequent Positive Tri Grams")


# Lets analyze the correlation between top 20 bi/tri grams and the sentiment data
train_incidents["top_20_bi_grams_list"] = train_incidents["top_20_bi_grams_list"].apply(lambda x: "".join(x))
top20_bigrams_sentiment_sum = train_incidents[["top_20_bi_grams_list","sentiments"]]
top20_bigrams_sentiment_grpby = top20_bigrams_sentiment_sum.groupby(by = "top_20_bi_grams_list")

## Sentiments VS Bi grams
top20_bigrams_sentiment_series = (top20_bigrams_sentiment_grpby["sentiments"].sum()).sort_values(na_position= "last")
top20_bigrams_sentiment_series
top20_bigrams_sentiment_df = pd.DataFrame(top20_bigrams_sentiment_series.values,top20_bigrams_sentiment_series.index)
top20_bigrams_sentiment_df = top20_bigrams_sentiment_df[top20_bigrams_sentiment_df.index != " "]
top20_bigrams_sentiment_df

#plt.title("Top 20 Bi Grams VS Sentiment Correlation")
top20_bigrams_sentiment_df.sort_values(by = 0,ascending = True).plot(kind="bar",title = "Top 20 Bi Grams VS Sentiment Plot")
plt.style.use("ggplot")
plt.xlabel("Bi Grams")
plt.ylabel("Sentiment Polarity")
plt.show()

train_incidents["top_20_tri_grams_list"] = train_incidents["top_20_tri_grams_list"].apply(lambda x: "".join(x))
top20_trigrams_sentiment_sum = train_incidents[["top_20_tri_grams_list","sentiments"]]
top20_trigrams_sentiment_sum_grpby = top20_trigrams_sentiment_sum.groupby(by = "top_20_tri_grams_list")

## Sentiments VS Tri grams
top20_trigrams_sentiment_series = (top20_trigrams_sentiment_sum_grpby["sentiments"].sum()).sort_values(na_position= "last")
top20_trigrams_sentiment_series

top20_trigrams_sentiment_df = pd.DataFrame(top20_trigrams_sentiment_series.values,top20_trigrams_sentiment_series.index)
top20_trigrams_sentiment_df = top20_trigrams_sentiment_df[top20_trigrams_sentiment_df.index != " "]
top20_trigrams_sentiment_df
#plt.title("Top 20 Bi Grams VS Sentiment Correlation")
top20_trigrams_sentiment_df.sort_values(by = 0,ascending = True).plot(kind="bar",title = "Top 20 Tri Grams VS Sentiment Plot")
plt.style.use("ggplot")
plt.xlabel("Tri Grams")
plt.ylabel("Sentiment Polarity")
plt.show()


#incidents_category VS sentiments

incidents_category["sentiments"].sum().sort_values()

## incidents_incident_subcategory vs sentiments
incidents_incident_subcategory["sentiments"].sum().sort_values()
##incidents_priority vs sentiments
incidents_priority["sentiments"].sum().sort_values()
#incidents_urgency vs sentiments
incidents_urgency["sentiments"].sum().sort_values()
#incidents_made_sla vs sentiments
incidents_made_sla["sentiments"].sum().sort_values()
#incidents_type vs sentiments
incidents_type["sentiments"].sum().sort_values()
#incidents_impact vs sentiments
incidents_impact["sentiments"].sum().sort_values()
#incidents_escalation vs sentiments
incidents_escalation["sentiments"].sum().sort_values()
#incidents_e2e_resolution_met vs sentiments
incidents_e2e_resolution_met["sentiments"].sum().sort_values()
#incidents_location vs sentiments
incidents_location["sentiments"].sum().sort_values()
#incidents_country vs sentiments
country_sentiments_series = incidents_country["sentiments"].sum().sort_values()
country_sentiments_series
##create a dataframe of country series
country_sentiments_df = pd.DataFrame(country_sentiments_series.values,country_sentiments_series.index)
country_sentiments_df
#plt.title("Top 20 Bi Grams VS Sentiment Correlation")
country_sentiments_df.sort_values(by = 0,ascending = True).plot(kind="bar",title = "Country VS Sentiment Plot")
plt.style.use("ggplot")
plt.xlabel("Countries")
plt.ylabel("Sentiment Polarity")
plt.show()


#incidents_contact_type vs sentiments
incidents_contact_type["sentiments"].sum().sort_values()
#incidents_affected_user vs sentiments
incidents_affected_user["sentiments"].sum().sort_values()
#incidents_assignment_group vs sentiments
incidents_assignment_group["sentiments"].sum().sort_values()
close_code_grpby = train_incidents.groupby("close_code")
close_code_grpby["sentiments"].sum().sort_values()
resolved_by_grpby = train_incidents.groupby("resolved_by")
resolved_by_grpby["sentiments"].sum().sort_values()
from scipy.stats import linregress

### Lets check the correlation between infosys_e2e_resolution_duration and sentiments 
linregress(train_incidents["infosys_e2e_resolution_duration"],train_incidents["sentiments"])
# Correlation coefficient - 0.07(on positive side but very minimal correlation, hence can rule out the correlation
# between these 2 variables) 



### Lets check the correlation between infosys_e2e_response_duration and sentiments 
linregress(train_incidents["infosys_e2e_response_duration"],train_incidents["sentiments"])

### Lets check the correlation between e2e_response_duration and sentiments 
linregress(train_incidents["e2e_response_duration"],train_incidents["sentiments"])

### Lets check the correlation between e2e_resolution_duration and sentiments 
linregress(train_incidents["e2e_resolution_duration"],train_incidents["sentiments"])


### Extract numeric cols without null data
train_incidents.dtypes == "int64" 
int_float_cols = train_incidents.dtypes[(train_incidents.isna().sum() != 2301) & ((train_incidents.dtypes == "int64") | (train_incidents.dtypes == "float64"))].index
train_incidents[int_float_cols].head(5)

## Selecting only required numeric cols for correlation plot

numeric_cols_selected = train_incidents[["reopen_count","reassignment_count","infosys_e2e_response_duration","infosys_e2e_resolution_duration","followup_counter","e2e_response_duration","e2e_resolution_duration","calendar_duration","business_duration","sentiments"]]

pd.plotting.scatter_matrix(numeric_cols_selected,alpha = 0.8,figsize=(25, 25))
plt.tight_layout()


train_incidents["issue_turnaround_time_hours"] = (train_incidents["resolved_at"] - train_incidents["opened_at"]).astype('timedelta64[h]')
## Impute missing values with mean 
train_incidents["issue_turnaround_time_hours"][train_incidents["issue_turnaround_time_hours"].isna() == True] = train_incidents["issue_turnaround_time_hours"].mean(skipna= True)

train_incidents["issue_turnaround_time_days"] = (train_incidents["resolved_at"].dt.date - train_incidents["opened_at"].dt.date)
train_incidents["issue_turnaround_time_days"] = pd.to_numeric(train_incidents["issue_turnaround_time_days"].dt.days)
## Impute missing values with mean 
train_incidents["issue_turnaround_time_days"][train_incidents["issue_turnaround_time_days"].isna() == True] = train_incidents["issue_turnaround_time_days"].mean(skipna= True)

train_incidents["issue_turnaround_time_days"].head(2)

### Lets check the correlation between turenaround time and sentiments 
linregress(train_incidents["issue_turnaround_time_days"],train_incidents["sentiments"])

##  Incidents with high turnaround times have more negative sentiment polarity score. 
##  As we you can see in above plot, majority of turnaround times have sentiment polarity scores < -0.10.

train_incidents.plot.scatter("issue_turnaround_time_days","sentiments")
plt.show()






