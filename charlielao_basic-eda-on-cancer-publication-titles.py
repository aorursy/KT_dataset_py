%%capture
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords#to filter out stop words to scale down the data 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
#the last two are used for NLP 
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
#this is for ipython to display all the results of cell
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
raw = pd.read_csv('../input/publications-from-roswell-park-cancer-institute-beginning-2006.csv')
raw.columns #show what informations are available
raw.head()
%%capture
#see what each column contain
for feature in list(raw.columns):
    print(feature + str(raw[feature].unique()) + '\n')
'''
year: looks fine, can be used
type: can be used, there is a 967-9753 can take out
journal name: could be useful
title: essential for my purpose
arthor: maybe can use
journal volume: definitely can drop
issue number: also useless
range: could be used to calculate the length of the paper
ISSN: useless
peer reviewed: important?
Impact: the most important prediction here
'''
#first stage cleaning: dropping the stuff definitely irrelevant to analysis
DF = raw.drop(['ISSN', 'Journal Issue Number', 'Journal Volume'], axis = 1)
DF = DF[DF['Publication Type']!='967-9753']
DF.head()
DF['Publication Type'].unique()#no more 967-963
DF['Year Published'].value_counts().plot(kind='bar');
#see the number of journals each year
#2018 doesn't really count but we still see a gradual decrease
%%capture
#from here, if the analysis has to do with impact score, use DF_impact, unless use DF as there are more data
#some are non-rated so replace the string: 'Not Rated' with nan
DF['Impact Factor'] = DF['Impact Factor'].replace(to_replace = 'Not Rated',value=np.nan)
#ignore the data without rating
DF_impact = DF[DF['Impact Factor'].notnull()]
#covert the score/year from string to a number
DF_impact['Impact Factor'] = DF_impact['Impact Factor'].astype('float64')
DF_impact['Year Published'] = DF_impact['Year Published'].astype('int64')
#ignore years from 2015 since impact score shouldn't be calculated before three years after prublication
DF_impact = DF_impact[DF_impact['Year Published'].apply(lambda x: x <=2015)]
#check the most impactful publication 
DF_impact.sort_values(by=['Impact Factor'], ascending=False).head(10)
#trying to combine all the titles from all the articles
combined_long =''
#adding them
for _ in DF['Publication Title']:
    combined_long+=_
    #combined_long+=' '.join(list(set(_.split(' '))))
#filtered=' '.join([words for words in combined_long.split(" ") if words not in uninformation_words])

#build a bag of words for each title with NLP toolbox
#second one is more robust and will be used from now on
count_vectorizer = CountVectorizer(lowercase=True)
tfidf_vectorizer = TfidfVectorizer(lowercase=True)
#build bag of words with the vectorizer
bag_of_words = count_vectorizer.fit_transform([combined_long])
bag_of_words2 = tfidf_vectorizer.fit_transform([combined_long])
#get name of the feature i.e. the key words
feature_names = count_vectorizer.get_feature_names()
feature_names2 = tfidf_vectorizer.get_feature_names()
#the words we don't want
customed_uselesswords=set(['cancer','study','cell','cells','analysis','tumor','risk','phase','human','group',
                           'advanced','expression','thearpy','treatment','patients','non','based','survival'
                          'small','gene','trial','results','novel'])
uninformation_words = customed_uselesswords|set(stopwords.words('english')) 
uninformation_words=list(uninformation_words&set(feature_names))
#convert our results for the bag of words into a data frame for the normal method
BoW=pd.DataFrame(bag_of_words.toarray(), columns = feature_names)
BoW= BoW.transpose()
BoW.columns = BoW.columns.astype(str)
BoW.columns = ['counts']
#filter the less important ones(less frequent)
BoW = BoW.drop(uninformation_words).loc[BoW['counts']>200].sort_values(by=['counts'], ascending=False)
#same thing for the more robust method
BoW2=pd.DataFrame(bag_of_words2.toarray(), columns = feature_names2)
BoW2= BoW2.transpose()
BoW2.columns = BoW2.columns.astype(str)
BoW2.columns = ['frequency']
BoW2 = BoW2.drop(uninformation_words).loc[BoW2['frequency']>0.015].sort_values(by=['frequency'], ascending=False)    
BagWords=BoW2

BoW.plot.bar();
BoW2.plot.bar();

#the two happened to be the same 
BagWords.head(20)
#we can pick out the main cancer research area to invertigate: breast, prostate, ovarian, carcinoma, lung, leukemia,lymphoma,myeloid
%%capture
#next stage will be to investigate how the key words of cancer has raised popularity 
grouped_df = DF_impact.groupby(['Year Published'])

for key, item in grouped_df:
    print(grouped_df.get_group(key), "\n\n")
#define a class contain the methods for the type of cancer we're focusing on

class cancer():
    #initiate the cancer type such that we have a data frame that doesn't contain the type of cancer we interesested
    def __init__(self,cancer_type):
        self.cancer_type=cancer_type
        self.df = DF[DF['Publication Title'].apply(lambda x: self.cancer_type in x)]
        self.df_i = DF_impact[DF_impact['Publication Title'].apply(lambda x: self.cancer_type in x)]
    #define the method for the verious things we might be interested from the data
    def summary(self,h,t):
        def counts_year(t):
            self.f_y = self.df.groupby('Year Published')['Publication Title'].count()
            self.f_y.plot.bar(title=self.cancer_type, ax=ax[t], rot =0);
        def impact_year(t):
            self.i_y = self.df_i.groupby('Year Published')['Impact Factor'].agg('mean')
            self.i_y.plot.bar(title=self.cancer_type, ax=ax[t], rot =0);
        def average_impact():
            return self.df_i['Impact Factor'].mean() 
        def correlation():
            self.f_y = self.df_i.groupby('Year Published')['Publication Title'].count()
            self.i_y = self.df_i.groupby('Year Published')['Impact Factor'].agg('mean')
            return self.f_y.corr(self.i_y)
        def box(t):
            self.df_i[self.df_i['Year Published'] == int(t)+2006].boxplot(column=['Impact Factor'], ax=ax[t])
        def box_year():
            self.df_i.groupby('Year Published')['Impact Factor'].agg('mean').plot(kind='line', ax = ax);
            self.df_i.boxplot(column = ['Impact Factor'], by = 'Year Published', ax=ax,  rot=0 );
           
        if h=='f':#the publication counts by year
            counts_year(t)
        elif h=='i':#the average impact factor by year
            impact_year(t)
        elif h=='a':#find the average impact score for each type of cancer
            return average_impact()
        elif h=='c':#find the correlation between publication number and impact factor
            return correlation()
        elif h=='b':#the distribution of impact factor score by year
            box(t)
        elif h=='b_y':#the distribution of impact factor score by year
            box_year()
#the most frequently appeared type of cancer in this order
all_cancer = ['breast', 'prostate', 'ovarian', 'carcinoma', 'lung', 'leukemia', 'lymphoma', 'myeloid']
#define a function that tells us the cancer and its associate information
def find_summary(Cancer, s, t):
    _ = cancer(Cancer)
    return _.summary(s,t)
#find the average impact score according to year by cancer
fig, ax = plt.subplots(8,1,figsize=(20,20))
plt.subplots_adjust(hspace=1)
for i in range(8):
    find_summary(all_cancer[i],'i',i)
#find the publication counts according to year by cancer
fig, ax = plt.subplots(8,1,figsize=(20,20))
plt.subplots_adjust(hspace=1)
for i in range(8):
    find_summary(all_cancer[i],'f',i)
#find the distribution of impact score with boxplot for breast cancer from 2006 to 2015(left to right)
fig, ax = plt.subplots(1,10,figsize=(20,5))
for i in range(10):
    find_summary('breast','b',i)
fig, ax = plt.subplots(1,1,figsize=(20,20))
find_summary('breast','b_y',2)
