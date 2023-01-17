# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json         #json file reading

import multiprocessing as mp

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

#ensure that the internet option is on (Kaggle) to install textract package

!pip install textract --upgrade

!pip install wordcloud

!pip install PIL

import textract

from wordcloud import WordCloud, STOPWORDS

def checkForDuplicates(List):

    ''' Check for any duplicates in a list'''

    if len(List) == len(set(List)):

        return False

    else:

        return True

    

    

def read_pdf(file):

    '''Reads "/kaggle/input/CORD-19-research-challenge/2020-03-13/COVID.DATA.LIC.AGMT.pdf" and makes it a bit more readable '''

    import textract

    text = str(textract.process(file))

    text = text.lstrip('b')

    text = text.strip("'")

    text = text.split('\\n')



    pdf_str = ''

    for t in text:

        if t != '':

            if t[0].isupper():

                pdf_str = pdf_str + t + '\n'

            else:

                pdf_str = pdf_str + t    

            

    return pdf_str





def load_json_files(directory_name):

    '''Loads json files into list'''

    articles = []

    

    articlenames = os.listdir(directory_name)



    for articlename in tqdm(articlenames):

        articlename = directory_name + articlename

        article = json.load(open(articlename, 'rb'))

        articles.append(article)

    

    return articles







def article_title_list(article,source):

    '''Create a list of a paper's ID, title and source''' 



    row = [article["paper_id"],article["metadata"]["title"],source]

    return  row









def article_author_list(article):

    '''Create a list of a paper's ID, authors and source'''

    authors = []



    for idx in range(len(article["metadata"]["authors"])):

        author = [article["paper_id"],article["metadata"]["authors"][idx]["first"], article["metadata"]["authors"][idx]["last"]]

        authors.append(author)

    

    return authors











def article_body_list(article):

    '''Create a list of a paper's ID and body'''

    body = []

    for idx in range(len(article["body_text"])):

        bod = [article["paper_id"],article["body_text"][idx]["section"],article["body_text"][idx]["text"]]

        body.append(bod)

    

    return body











def article_abstract_list(article):

    '''Create a list of a paper's ID, and abstracts'''

    abstracts = []

    for idx in range(len(article["abstract"])):

        abstract = [article["paper_id"], article["abstract"][idx]["text"]]

        abstracts.append(abstract)

    

    return abstracts





def article_text(article):

    '''Create a list of a paper's ID, and abstracts'''

    text = ''

    for idx in range(len(article["abstract"])):

        text = text + '\n\n' + article["abstract"][idx]["text"]

        

    for idx in range(len(article["body_text"])):

        text = text + '\n\n' + article["body_text"][idx]["section"] + '\n\n' + article["body_text"][idx]["text"]

    

    text = [article["paper_id"],text]

    

    

    return text









def create_df(articles,source):

    '''Creates dataframes for Kaggle Covid-19'''

    

    titles_list = []

    authors_list = []

    text = []

     



    

    for article in tqdm(articles):

        

        '''Create a list of lists containing a paper's ID, title and source'''

        titles_list.append(article_title_list(article,source))



        '''Create a list of lists containing a paper's ID, authors and source'''    

        authors_list.extend([*[item for item in article_author_list(article)]])           

 



        '''Create a list of lists containing a paper's abstracts and body'''    

        text.append(article_text(article)) 

        

    text = pd.DataFrame(text,columns = ["Paper_Id",'Text'])   

    title_df = pd.DataFrame(titles_list,columns = ["Paper_Id","Title",'Source'])

    author_df = pd.DataFrame(authors_list,columns = ["Paper_Id","First_Name","Last_Name"])



        

    return title_df,author_df,text

    
noncomm_use_subset_Dir = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/'

biorxiv_medrxiv_Dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'

comm_use_subset_Dir = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/'

custom_license_Dir = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/'



noncomm_pmc_Dir = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pmc_json/'

biorxiv_pmc_Dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pmc_json/'

comm_pmc_Dir = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pmc_json/'

custom_pmc_Dir = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pmc_json/'
listOfFileNames = []



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        listOfFileNames.append(os.path.join(dirname, filename))



checkForDuplicates(listOfFileNames)



del listOfFileNames
with open("/kaggle/input/CORD-19-research-challenge/metadata.readme",'r') as f:

    file = f.read()

    print(file)
print(read_pdf("/kaggle/input/CORD-19-research-challenge/COVID.DATA.LIC.AGMT.pdf"))
meta_data = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')

meta_data.head()
with open("/kaggle/input/CORD-19-research-challenge/json_schema.txt",'r') as f:

    file = f.read()

    print(file)
# Create stopword list:



stopwords = set(STOPWORDS)

stopwords.update(["et al",'et', 'al',"addition", "respectively", "found", "although",'present',

                  'identified','Thu','Finally','either','suggesting','include',"well", 

                  "including", "associated", "method", "result",'used','doi','display',

                  'https','copyright', 'holder','org','author','available','made','peer',

                  'reviewed','without','permission','license','rights','reserverd','Furthermore'

                  'using','preprint','allowed','following','may','thus','funder','International',

                 'granted','compared','will','one','two','use','different','likely','Discussion',

                 'medRexiv','Introduction','Moreover','known','funder','granted'])
articles = load_json_files(comm_use_subset_Dir)

comm_subset_title,comm_subset_author,comm_subset_text = create_df(articles,'comm_use_subset')



comm_subset_title.to_csv('comm_subset_title.csv',index = False)

comm_subset_author.to_csv('comm_subset_author.csv',index = False)

comm_subset_text.to_csv('comm_subset_text.csv',index = False)





body_text = " ".join(text for text in comm_subset_text.Text)





#del comm_subset_title

#del comm_subset_author

#del comm_subset_text





print ("There are {} words in the bodies of the articles.".format(len(body_text)))







# Generate a word cloud image

wordcloud = WordCloud(stopwords=stopwords,max_words=200, background_color="white").generate(body_text)



# Display the generated image:

# the matplotlib way:

print('World Cloud for Bodies of articles')

plt.figure(figsize=(20,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()



#Generate Wordclouds

'''

articles = load_json_files(biorxiv_medrxiv_Dir)

bio_title,bio_author,bio_text = create_df(articles,'biorxiv_medrxiv')



bio_title.to_csv('biorxiv_medrxiv_title.csv',index = False)

bio_author.to_csv('biorxiv_medrxiv_author.csv',index = False)

bio_text.to_csv('biorxiv_medrxiv_text.csv',index = False)





body_text = " ".join(review for review in bio_text.Text)





del bio_title

del bio_author

del bio_text





print ("There are {} words in the abstracts of articles.".format(len(body_text)))





# Generate a word cloud image

wordcloud = WordCloud(stopwords=stopwords,max_words=200, background_color="white").generate(body_text)



# Display the generated image:

# the matplotlib way:

print('World Cloud for Bodies of articles')

plt.figure(figsize=(20,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()



# Save the image:

#wordcloud.to_file("wordcloud_biorxiv.png")

###############################################################



articles = load_json_files(noncomm_use_subset_Dir)

noncomm_subset_title,noncomm_subset_author,noncomm_subset_text= create_df(articles,'noncomm_use_subset')

  



noncomm_subset_title.to_csv('noncomm_subset_title.csv',index = False)

noncomm_subset_author.to_csv('noncomm_subset_author.csv',index = False)

noncomm_subset_text.to_csv('noncomm_subset_text.csv',index = False)





body_text = " ".join(review for review in noncomm_subset_text.Text)





del noncomm_subset_title

del noncomm_subset_author

del noncomm_subset_text





print ("There are {} words in the combination of all review.".format(len(body_text)))





# Generate a word cloud image

wordcloud = WordCloud(stopwords=stopwords,max_words=200, background_color="white").generate(body_text)



# Display the generated image:

print('World Cloud for Bodies of articles')

plt.figure(figsize=(20,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()



# Save the image:

#wordcloud.to_file("wordcloud_noncom.png")



##########################################################################



articles = load_json_files(custom_license_Dir)

custom_license_title,custom_license_author,custom_license_text= create_df(articles,'custom_license')

    



custom_license_title.to_csv('custom_license_title.csv',index = False)

custom_license_author.to_csv('custom_license_author.csv',index = False)

custom_license_text.to_csv('custom_license_text.csv',index = False)





body_text = " ".join(review for review in custom_license_text.Text)





del custom_license_title

del custom_license_author

del custom_license_text



print ("There are {} words in the combination of all review.".format(len(body_text)))





# Generate a word cloud image

wordcloud = WordCloud(stopwords=stopwords,max_words=200, background_color="white").generate(body_text)



# Display the generated image:

print('World Cloud for Bodies of articles')

plt.figure(figsize=(20,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()



# Save the image:

#wordcloud.to_file("wordcloud_custom.png")

'''
comm_subset_text.head()
def search(word,df_text):

    from nltk.tokenize import word_tokenize

    papers = []

    

    for idx in tqdm(range(len(df_text))):

        if word in word_tokenize(df_text.loc[idx,'Text']):

            papers.append(df_text.loc[idx,'Paper_Id'])

            

    return papers

        

looking_for = 'pregnancy'



search(looking_for,comm_subset_text)