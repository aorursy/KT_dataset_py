import pandas as pd
import numpy as np
from lda import guidedlda as glda

from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import string
df1=pd.read_csv("data.csv")
df1.head()
df1['text'] =df1['text'].str.replace("[^a-zA-Z#]", " ")

stopwords_list = stopwords.words('english')
punctuations = list(set(string.punctuation))

unwanted_list=punctuations+stopwords_list

def clean_text_initial(text):
    text = ' '.join([x.lower() for x in word_tokenize(text) if x.lower() not in unwanted_list and len(x)>1])
    text = ' '.join([x.lower() for x in word_tokenize(text) if nltk.pos_tag([x])[0][1].startswith("NN") or nltk.pos_tag([x])[0][1].startswith("JJ")])
    return text.strip()

df1["clean_text"]=df1.text.apply(lambda text:clean_text_initial(str(text)))
df1.head()
corpus=df1.clean_text.tolist()
vocab=list(set(word_tokenize(" ".join(df1.clean_text))))
vectorizer = CountVectorizer(ngram_range=(1,1),vocabulary=vocab)
X = vectorizer.fit_transform(corpus)
word2id=vectorizer.vocabulary_
house_words=["bedroom","room","house","home","airbnb","condo","bed","blocks","comfy","amenities","pool","gym",
             "cottage","min","minutes", "away","duplex","kitchen","stay","short","apartment","residential",
             "camping","distance","bungalow","walking","neighborhood","cozy","cabin","cabins","coziness",
             "families","property","courtyard","accommodate","living", "area","minutos"]

glassdoor_words=["pros"]

automobile_words=["odometer","automobile","car","engine","automatic", "transmissions","manual", "shift",
                      "automotive","chevrolet","transmission","accelerator","toyota","volvo","nissan",
                     "convertibles","convertible","drive"]

sports_words=["club","manager","player","championship","contract","stadium","players","season","score","scorer",
              "team","teammate","game","liverpool","football","nfl","victory","ravens","boxing","cricket","quarterback",
              "middleweight","arsenal","barcelona","welterweight","icc","ipl","bowlers","innings","bowler","lightweight",
              "knicks","match","matches","soccer","football","playoffs","premier league","drs","tournament","fan",
              "sports","boxer","fielders","ufc","linebacker","coach","nba","referee","champion","injury","races","points",
              "golf","arenas","pitching","receiver","champ","cornerback","mvp","jayhawks","quarterfinal","agent","ball",
              "comeback","shot","red sox","agency","wins","winners","warriors","gonzaga","race"]

tech_words=["snapchat","facebook","samsung","phone","smartphone","iphone","ai","uber","hewlettpackard","technology",
            "digitized","google","nintendo","economy","github","aws","tumblr","entrepreneur","business","ncaa",
            "tourney","amazon","startup","apps","android","crypto","develop","headset","intel","mwc","bitcoin",
            "spacex","processor","software","analytics","twitter","developer","microsoft","computer","study","research",
            "recession","plugin","youtube","netflix","marvel","logan","siri","electronic","tesla","programming","snap",
            "chromebooks","fitbit","insta","instagram","semiconductor","vpn","fintech","industry","system","systems",
            "att","ceo","alexa","computing","vr","nasa","technical","companies","hacker","blog","wireless","speaker",
            "screens","organization","photography","article","ebay","pandora","console","printer","pi","spacesuit",
            "movies","novel","bot","robot","nitrogen","hoverboard","conferences","online","data","images","biomimicry",
            "apple"]
house_words = [x for x in house_words if x in list(word2id.keys())]
glassdoor_words = [x for x in glassdoor_words if x in list(word2id.keys())]
automobile_words = [x for x in automobile_words if x in list(word2id.keys())]
sports_words = [x for x in sports_words if x in list(word2id.keys())]
tech_words = [x for x in tech_words if x in list(word2id.keys())]
seed_topic_list = [
    house_words,
    glassdoor_words,
    automobile_words,
    sports_words,
    tech_words
]
model = glda.GuidedLDA(n_topics=5, n_iter=2000, random_state=7, refresh=20,alpha=0.01,eta=0.01)
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id
model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)
n_top_words = 10
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
topic_num_name = {"Topic 0":"room_rentals",
                  "Topic 1":"glassdoor_reviews",
                  "Topic 2":"Automobiles",
                  "Topic 3":"sports_news",
                  "Topic 4":"tech_news"}    
def get_doc_topics(model_glda,X,num_topics,dataframe,col_name):
    """
    A function which creates dataframe with documents, their dominant topic, along with their probabilities
    
    Parameters
    -------------
    model_glda - Guided LDA trained model
    X - Document term frequency table
    num_topics - Number of topics the model was trained for
    dataframe - Dataframe consisting of cleaned text column
    col_name - Column name in dataframe holding cleaned text
    
    Returns
    -------------
    A dataframe with document number, topic, probability of topic
    """
    df_doc_top = pd.DataFrame()
    final_list = []
    for index in range(len(dataframe[col_name])):
        word_id_dict = dict((x,y) for x,y in zip([x for x in range(num_topics)],np.round(model.transform(X[index])*100,1).tolist()[0]))
        word_score_list = []
        for index in range(num_topics):
            try:
                value = word_id_dict[index]
            except:
                value = 0
            word_score_list.append(value)
        final_list.append(word_score_list)

    df_doc_top = pd.DataFrame(final_list)
    df_doc_top.columns = ['Topic ' + str(i) for i in range(num_topics)]
    df_doc_top.index = ['Document ' + str(i) for i in range(len(dataframe[col_name]))]

    df_doc_top["Dominant_Topic"] = df_doc_top.idxmax(axis=1).tolist()
    df_doc_top["Topic_Probability"] = df_doc_top.max(axis=1).tolist()
    document_df = df_doc_top.reset_index().rename(columns={"index":"Document"})[["Document","Dominant_Topic","Topic_Probability"]]

    return document_df
document_df=get_doc_topics(model,X,5,df1,"clean_text")
submission=pd.concat([df1.Id,document_df.Dominant_Topic],axis=1)
submission.Dominant_Topic=submission.Dominant_Topic.replace(topic_num_name)
submission=submission.set_index("Id").rename(columns={"Dominant_Topic":"topic"})
submission
