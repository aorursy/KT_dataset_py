import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json

import matplotlib.pyplot as plt
plt.style.use('ggplot')
### 1-Loading meta data ###

root_path = '/kaggle/input/CORD-19-research-challenge/'
metadata_path = f'{root_path}metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
meta_df.head()
### 2-Fetch All of JSON File Path ###


all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
len(all_json)
### helper function to extract body text, abstract, and paper id from json files ###

#file reader class
class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            try:
                self.paper_id = content['paper_id']
            except Exception as e:
                self.paper_id = ''
            self.abstract = []
            self.body_text= []
            
            # Abstract
            try:
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
            except Exception as e:
                pass
            # Body text
            
            try:
                for entry in content['body_text']:
                    self.body_text.append(entry['text'])
            except Exception as e:
                pass
            
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}:{self.abstract[:200]}... {self.body_text[:200]}...'
    
first_row = FileReader(all_json[0])
## 3- Load the Data into DataFrame ##

#Using the helper functions, let's read in the articles into a DataFrame that can be used easily:
    
dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [] }
for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 500) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    # no metadata, skip this paper
    if len(meta_data) == 0:
        continue
    
    dict_['paper_id'].append(content.paper_id)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
    
        
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    dict_['title'].append(meta_data['title'].values[0])
    dict_['authors'].append(meta_data['authors'].values[0])

    
    # add the journal information
    dict_['journal'].append(meta_data['journal'].values[0])
    
df_covid = pd.DataFrame(dict_, columns=['paper_id','abstract', 'body_text', 'authors', 'title', 'journal'])
df_covid.head()


## 4 -droping the duplicates ##
dict_ = None

df_covid.info()

df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)

df_covid['abstract'].describe(include='all')
df_covid['body_text'].describe(include='all')


## droping the null values##
df_covid.dropna(inplace=True)
#import sys
#!{sys.executable} -m pip install --upgrade pip

!pip install langdetect

## 1- labelling the languages of articles in a separate column ##

from langdetect import detect
df_covid['language'] = df_covid['title'].apply(detect)

#checking what languages are present in the dataset
uniqueValues = df_covid['language'].unique()

## plotting top 5 highest number of published languages ##
import seaborn as sns
plt.figure(figsize=(10,5))
sns.set_style('darkgrid')
sns.countplot(x='language',data=df_covid,order=pd.value_counts(df_covid['language']).iloc[:5].index,palette='viridis')
plt.title('language distribution')


## 2- keeping english language only ##

df_covid = df_covid.loc[df_covid['language'] == 'en']
df_covid.info()
## 3- finding papers takling about covid and sars and mers, proteins, genome ##


# a function that will work as a searching machine to find specific keywords in papers
import re
def pattern_searcher(search_str:str, search_list:str):

    search_obj = re.search(search_list, search_str)
    if search_obj :
        return True
    else:
        return False
    
###keywords related to covid that we are looking for in literatures  
covid_list = ['covid',
                    'coronavirus disease 19',
                    'sars cov 2', # Note that search function replaces '-' with ' '
                    '2019 ncov',
                    '2019ncov',
                    r'2019 n cov\b',
                    r'2019n cov\b',
                    'ncov 2019',
                    r'\bn cov 2019',
                    'novel corona',
                    'coronavirus 2019',
                    'wuhan pneumonia',
                    'wuhan virus',
                    'wuhan coronavirus',
                    r'coronavirus 2\b',
                    'chinese coronavirus', 
                    'sars-cov-2',
                    'novel coronavirus', 
                    'coronavirus', 
             ]
pattern_covid = '|'.join(covid_list)

#adding extra column that tags papers that are related to covid19
df_covid['covid_match'] = df_covid['body_text'].apply(lambda x: pattern_searcher(search_str=x, search_list=pattern_covid))


###keywords related to protein and receptor surface protein that we are looking for in literatures
protein_list = ['protein','surface protein', 'proteome', 'proteomics', 'proteomic','proteins']
pattern_protein = '|'.join(protein_list)

#adding extra column that tags papers that are related to Protein
df_covid['protein_match'] = df_covid['body_text'].apply(lambda x: pattern_searcher(search_str=x, search_list=pattern_protein))



###keywords related to genomic sequences that we are looking for in literatures
genom_list=['genomic','genomics', 'genome', 'genome sequence', 'genome-wide']
pattern_genom = '|'.join(genom_list)

#adding extra column that tags papers that are related to Genomic sequences
df_covid['genom_match'] = df_covid['body_text'].apply(lambda x: pattern_searcher(search_str=x, search_list=pattern_genom))


df_covid.head()
### only keeping the papers that should be most related to task3 and provides good results###

df_covid = df_covid.loc[(df_covid['covid_match'] == True) ] 
df_covid.info()

df_covid.head()
## 4-preparing the text to be ready for vectorization in the next steps ##

import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

lemma = WordNetLemmatizer()


stopword_set = set(stopwords.words('english')+['a','at','s','for','was', 'we', 'were', 'what', 'when', 'which', 'while', 'with', 'within', 'without', 'would', 'seem', 'seen','several', 'should','show', 'showed', 'shown', 'shows', 'significantly', 'since', 'so', 'some', 'such','obtained', 'of', 'often', 'on', 'our', 'overall','made', 'mainly', 'make', 'may', 'mg','might', 'ml', 'mm', 'most', 'mostly', 'must', 'each', 'either', 'enough', 'especially', 'etc','had', 'has', 'have', 'having', 'here', 'how', 'however', 'upon', 'use', 'used', 'using', 'perhaps', 'pmid','can', 'could', 'did', 'do', 'does', 'done', 'due', 'during', 'et al', 'found','study','observed','identified','fig','although','reported','group','result','include', 'figure', 'table'])

stop_set=['a','at','s','for','was', 'we', 'were', 'what', 'when', 'which', 'while', 'with', 'within', 'without', 'would', 'seem', 'seen','several', 'should','show', 'showed', 'shown', 'shows', 'significantly', 'since', 'so', 'some', 'such','obtained', 'of', 'often', 'on', 'our', 'overall','made', 'mainly', 'make', 'may', 'mg','might', 'ml', 'mm', 'most', 'mostly', 'must', 'each', 'either', 'enough', 'especially', 'etc','had', 'has', 'have', 'having', 'here', 'how', 'however', 'upon', 'use', 'used', 'using', 'perhaps', 'pmid','can', 'could', 'did', 'do', 'does', 'done', 'due', 'during', 'et al', 'found','study','observed','identified','fig','although','reported','group','result','include', 'figure', 'table']

#definig a function that cleans the string of full body text
def process(string):
    string=' '+string+' '
    string=' '.join([word if word not in stopword_set else '' for word in string.split()])
    string=re.sub('\@\w*',' ',string)
    string=re.sub('\.',' ',string)
    string=re.sub("[,#'-\(\):$;\?%]",' ',string)
    string=re.sub("\d",' ',string)
    string=re.sub(r'[^\x00-\x7F]+',' ', string)
    for i in stop_set:
        string = re.sub(' ' +i+' ', ' ', string)
    string=" ".join(lemma.lemmatize(word) for word in string.split())
    string=re.sub('( [\w]{1,2} )',' ', string)
    string=re.sub("\s+",' ',string)
    string=string.replace('[', '')
    string=string.replace(']', '')
    return string


df_covid['processed_text'] = df_covid['body_text'].apply(process)
## 1-Performing TF-IDF Vectorization on the data##

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_covid['processed_text'].values)
X = vectorizer.transform(df_covid['processed_text'].values)
## 2- Performing Principle component analysis (PCA) on the vectorized body text##

# Since the TF-IDF vector was sparse PCA was performed using TruncatedSVD 
from sklearn.decomposition import TruncatedSVD

PCA = TruncatedSVD(2)
Xpca = PCA.fit_transform(X)

## Helper function for plotting the clustering results ## 
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns


def plotting(datapoint, labels):
    # sns settings
    sns.set(rc={'figure.figsize':(15,15)})

    # colors
    palette = sns.color_palette("bright", len(set(labels)))

    # plot
    sns.scatterplot(datapoint[:, 0], datapoint[:, 1], hue=labels, legend='full', palette=palette)
    plt.title("Covid-19 Articles - Clustered(K-Means) - Using Vectorized body text")

    plt.savefig("/kaggle/working/covid19_label_TFIDF.png")
    plt.show()
## 3 & 4: visualization and evaluatoin ##
## Helper function for quantitatively comparing and measuring accuracy using a combination of clustering + KNN [5] ##

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score


def acc(data_vectors):
    
    # Cluster the vectors find their labels 
    kmeans_model = KMeans(n_clusters=9, init='k-means++', max_iter=100) 
    kmeans_model.fit(data_vectors)
    labels=kmeans_model.labels_.tolist()
    plotting(data_vectors, labels) # Plot the clusters 
    
    # Perform KNN on using the labels from clustering
    neigh = KNeighborsClassifier(n_neighbors=15)

    # performing cross validation 
    scores = cross_val_score(neigh, data_vectors, labels, cv=5)
    acc = np.average(scores)
    return acc 

print(acc(Xpca))
%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img=mpimg.imread('/kaggle/input/assets/Comparison of different algorithms(1).png')
plt.figure(figsize=(12,12))
imgplot = plt.imshow(img)
plt.show()
# Dropping irrelavant columns for further processing
try:
    df_covid = df_covid.drop([ "language", "processed_text", "protein_match", "sars_match", "mers_match", "covid_match", "genom_match", "paper_id"], axis=1) # keep "paper_id", "abstract", "authors", "title", "journal", "body_text"
except:
    print("Items already dropped")


# rearranging the data columns for further visualizations 
cols = ['title', 'abstract', 'authors', 'journal', 'body_text']
df_covid = df_covid[cols]
# Helper function for plotting word clouds for various questions related to task 3

from wordcloud import WordCloud

def word_cloud(df, filename):
    long_string = ','.join(list(df['body_text'].values[:20])) # Create a WordCloud object using the top 20 papers 
    wordcloud = WordCloud(background_color="white", max_words=50000, contour_width=5, contour_color='steelblue') # Generate a word cloud
    wordcloud.generate(long_string)
    wordcloud.to_image()
    wordcloud.to_file(filename + '.png')
    
    



from scipy import spatial
from sklearn.neighbors import NearestNeighbors
def get_closest_neighbours_table(question, vector_body_text, vectorizer_model, pca_model, df, filename):    
    
    # Vectorize the task vector using the pre-trained TF-IDF model
    task_X = vectorizer_model.transform(question)
    # Reduce the dimensionality of the task vector to 2 using pre-trained PCA model
    task_out = pca_model.transform(task_X)    
    filename = filename.replace(' ', '_')   
    
    # calculating the cosine similarity of the tasks between
    df['cosine_similarity'] = [1 - spatial.distance.cosine(a, task_out) for a in vector_body_text]    # Sorting the papers based on the cosine similarity scores
    df = df.sort_values(by=['cosine_similarity'], ascending=False)    
    
    # picking the top 10 papers based on highest cosine similarity
    df = df.iloc[:10]
    df['body_text'] = df['body_text'].apply(lambda x: x[:min(1000, len(x))])    
    
    # Save the dataframe to html for better visualization of data.
    html = df.to_html()
    #write html to file
    text_file = open( filename +".html", "w")
    text_file.write(html)
    text_file.close()    
    
    return df

# Increasing the maximum column width to 300 for better visualization of tables 

pd.set_option('display.max_colwidth', 300)
# Finding the most similar documents to questions related to task 3 as suggested by Dr. Hassan Vahidnezhad

question1 = 'How is the corona virus infecting the animals different from the covid-19 virus that is infecting humans'
question_preprocessed = [process(question1)] # Performing the same pre-processing step on the question document
df = get_closest_neighbours_table(question_preprocessed, Xpca, vectorizer, PCA, df_covid, question1) # Finding the closest neighbours 
print(question1)
df.head()


# Make of word cloud using the top 20 papers.

long_string = ','.join(list(df['body_text'].values[:20])) # Create a WordCloud object using the top 20 papers 
wordcloud = WordCloud(background_color="white", max_words=50000, contour_width=5, contour_color='steelblue') # Generate a word cloud
wordcloud.generate(long_string)
wordcloud.to_image()


question2 = 'What part of the covid-19 virus genome determines the suitable host for this coronavirus'
question_preprocessed = [process(question2)]
df = get_closest_neighbours_table(question_preprocessed, Xpca, vectorizer, PCA, df_covid, question2)
print(question2)
df.head()


# Make of word cloud using the top 20 papers.

long_string = ','.join(list(df['body_text'].values[:20])) # Create a WordCloud object using the top 20 papers 
wordcloud = WordCloud(background_color="white", max_words=50000, contour_width=5, contour_color='steelblue') # Generate a word cloud
wordcloud.generate(long_string)
wordcloud.to_image()
# Finding the most similar documents to complete task 3 desciption.  

task3 = 'Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time. Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged. Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over. Evidence of whether farmers are infected, and whether farmers could have played a role in the origin. Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia. Experimental infections to test host range for this pathogen. Animal host(s) and any evidence of continued spill-over to humans. Socioeconomic and behavioral risk factors for this spill-over. Sustainable risk reduction strategies.'
question_preprocessed = [process(task3)]
df = get_closest_neighbours_table(question_preprocessed, Xpca, vectorizer, PCA, df_covid, 'General task3 description')
print('General task3 description')
df.head()


# Make of word cloud using the top 20 papers.

long_string = ','.join(list(df['body_text'].values[:20])) # Create a WordCloud object using the top 20 papers 
wordcloud = WordCloud(background_color="white", max_words=50000, contour_width=5, contour_color='steelblue') # Generate a word cloud
wordcloud.generate(long_string)
wordcloud.to_image()
%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img=mpimg.imread('/kaggle/input/assets2/Assets/IMG_5188.JPG')
plt.figure(figsize=(20,20))
imgplot = plt.imshow(img)
plt.show()