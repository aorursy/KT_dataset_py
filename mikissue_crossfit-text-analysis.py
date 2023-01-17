from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
import numpy as np
import pandas as pd 

#importing both CSV files
athletes_only = pd.read_csv("../input/athletes.csv")
leaderboard = pd.read_csv("../input/leaderboard.15.csv")
leaderboard = leaderboard[['athlete_id', 'score']]

#merging the two df to get the full data set
athletes = athletes_only.merge(leaderboard, left_on='athlete_id', right_on='athlete_id', how='inner')
athletes.drop(['name'], axis=1, inplace = True)

athletes.head()
#using the athletes only df, finding the numeric data and seting NaN to median of the column
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
athletesNum = athletes.select_dtypes(include=numerics)

#using the athletes only df, extracting the textual data
textCol = list(set(athletes.columns) - set(athletesNum.columns))
textCol.append('athlete_id')
textCol.append('score')
fulldf = athletes.loc[:,textCol]
#dropping the rows with missing data
tmData = fulldf.dropna()
#dropping duplicates based on athlete ID
tmData = tmData.drop_duplicates(subset=['athlete_id'])
#drop columns that we don't need
tmData.drop(['retrieved_datetime', 'team'], axis = 1, inplace = True)
tmData.head()
import re
def pre_process(text):
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    #lower case
    text=text.lower()
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    shortword = re.compile(r'\W*\b\w{1,3}\b')
    text = shortword.sub('', text)
    return text

text_df = tmData.iloc[:,2:-3]

for column in text_df:
    text_df.loc[:, column] = tmData.loc[:, column].apply(lambda x:pre_process(x))
text_df.head()
text_df.columns[1]
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    #get the feature names and tf-idf score of top n items 
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        fname = feature_names[idx]
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    return results
def get_stop_words(stop_file_path): 
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)
#load a set of stop words
stopwords=get_stop_words("../input/stopwords.txt")

def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
 
def get_keywords(df):
    col = text_df.columns
    topKeywords = []
    for i in col:
        topKeywords.append(i)
        docs=df[i].tolist()
        for j in range(len(docs)):
            temp = docs[j].split()
            for i in range(len(temp)):
                temp[i] = lemmatize_stemming(temp[i])
            docs[j] = " ".join(temp)

    #create a vocabulary of words, 
    #ignore words that appear in 85% of documents, 
    #eliminate stop words
        cv=CountVectorizer(max_df=0.85,stop_words=stopwords)
        word_count_vector=cv.fit_transform(docs)
        words = list(cv.vocabulary_.keys())


     #   for i in range(len(words)):
      #      words[i] = lemmatize_stemming(words[i])

        tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
        tfidf_transformer.fit(word_count_vector)

        # you only needs to do this once
        feature_names=cv.get_feature_names()

        # get the document that we want to extract keywords from
        doc = ''.join(map(str, docs))

        #generate tf-idf for the given document
        tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

        #sort the tf-idf vectors by descending order of scores
        sorted_items=sort_coo(tf_idf_vector.tocoo())

        #extract only the top n; n here is 5
        keywords=extract_topn_from_vector(feature_names,sorted_items,5)

        # now print the results

        for k in keywords:
            topKeywords.append((k, keywords[k]))
    return(topKeywords)
top_key = get_keywords(text_df)[:]
top_key
import matplotlib.pyplot as plt
gender = tmData.groupby(['gender']).count()
gender.rename({'--': 'Not specified'}, axis='index', inplace = True)
ax = gender.plot.bar(y='train', rot=0, figsize=(14,8), title='The number of female and male athletes', legend= False)
length = tmData.groupby(['howlong']).count()
length = length.iloc[:,0]
length = length[length > 1000]
ax1 = length.plot.bar(rot=0, figsize=(14,8), title = 'The length of training')