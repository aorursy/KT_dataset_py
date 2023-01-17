# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Imports

from datetime import datetime

import matplotlib.pyplot as plt

data=pd.read_csv("../input/HN_posts_year_to_Sep_26_2016.csv")



# We will be converting the created_at column to hour, minutes etc

data['created_at']=data['created_at'].map(lambda x:datetime.strptime(x,'%m/%d/%Y %H:%M'))

data['hour']=data['created_at'].map(lambda x:x.hour)

data['year']=data['created_at'].map(lambda x:x.year)

data['month']=data['created_at'].map(lambda x:x.month)

data['title']=data['title'].map(lambda x:x.lower())
### Simple Plots

# Plot for Hourly distribution. We will loop for plotting it for 12 months



plt.figure()

monthLabel=['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov','Dec']

for monthVal in range(1,13):

    data[data['month']==monthVal].groupby(['hour'])['id'].count().plot(figsize=(10,10),label=monthLabel[monthVal-1])

plt.xlabel("Hour of the day")

plt.ylabel("Number of Posts")

plt.legend()

plt.show()

# We can see that there is definitely a trend

# But how does it relate to our problem statement?

# I think that topics initiated during the peak hours is likely to get more traction that topics initiated during off hours

# To verify this, let us plot comments. It should follow the same pattern



plt.figure()

for monthVal in range(1,13):

    data[data['month']==monthVal].groupby(['hour'])['num_comments'].sum().plot(figsize=(10,10),label=monthLabel[monthVal-1])

plt.xlabel("Hour of the day")

plt.ylabel("Number of Comments")

plt.legend()

plt.show()



# I think that topics initiated during the peak hours is likely to get more traction that topics initiated during off hours

# To verify this, let us plot comments. It should follow the same pattern



plt.figure()

for monthVal in range(1,13):

    data[data['month']==monthVal].groupby(['hour'])['num_points'].sum().plot(figsize=(10,10),label=monthLabel[monthVal-1])

plt.xlabel("Hour of the day")

plt.ylabel("Number of Upvotes")

plt.legend()

plt.show()
# Translating each row into a TOPIC

# This will be challenging



# COUNT VECTORIZER

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

titleCounts = count_vect.fit_transform(data['title'])

titleCounts.shape

# 80076 features



# The number of unique words created is 80076. Not that helpful

# We need to do a lot of cleaning before this

# Removing StopWords

# Lemmatization

# Stemming



# We need to do some straight away replacements

charsToBeReplaced=['.', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']

def replaceText(Text,CharsToBeReplaced):

    for x in CharsToBeReplaced:

        Text=Text.replace(x,"")

    return Text



data['CleanedTitle']=data['title'].map(lambda x:replaceText(x,charsToBeReplaced))





# Let us use NLTK to remove the Stop Words and Stem and Lemmatize the data

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

porter_stemmer = PorterStemmer()

data['CleanedTitle']=data['CleanedTitle'].map(lambda z: ' '.join([wordnet_lemmatizer.lemmatize(porter_stemmer.stem(x)) for x in z.split(' ') if x not in stop_words]))



# We will now work on this data set

titleCounts=count_vect.fit_transform(data['CleanedTitle'])

# 71623 features. Not much reduction, but something is better than nothing

#from sklearn.feature_extraction.text import TfidfTransformer

#transformer=TfidfTransformer(smooth_idf=False)

#tfidf = transformer.fit_transform(titleCounts)

#tfidf.toarray() 



# This caused some memory error, so let us try something else

# Let us analyse at a month level at least
# We are choosing a smaller data set to avoid memory issues

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer



# We will now be running this for all the months and then take a weighted average

TotalCoefficientsWeight=pd.DataFrame()



#transformer=TfidfTransformer(smooth_idf=False)

#titleCounts=count_vect.fit_transform(data[(data['month']==1) & (data['year']==2016)]['CleanedTitle'])

#tfidf = transformer.fit_transform(titleCounts)

def analysisMonthYear(Month,Year,ngramLength,evaluationColumn):

    if len(data[(data['month']==Month) & (data['year']==Year)]['CleanedTitle']) >0:

        vectorizer=TfidfVectorizer(max_df=0.95, min_df=2,max_features=100,stop_words='english',ngram_range=(1,ngramLength))

        tfidf_vect=vectorizer.fit_transform(data[(data['month']==Month) & (data['year']==Year)]['CleanedTitle'])

        topFeatures=vectorizer.get_feature_names()

        # This was for debugging

        #print("Year is " + str(Year) + " Month is " + str(Month))

        # This is very time consuming, so i am commenting it for now

        #topFeaturesCount=[sum([1 if f1 in text1.split(' ') else 0 for text1 in data['CleanedTitle']]) for f1 in topFeatures]



        # Now we can plot the frequency of posts having these words

        # e.g. object



        # We will now be creating a matrix X, that consists of the word vectors( only the important features)

        # The correponding Y matrix will be the respective counts and we will then optimize it

        # We might have to select fewer features in the beginning to check our logic. LEt us being with 100 features

        # So the matrix will be numberOfRows X 100

    

        # Running the count vectorizer

        from sklearn.feature_extraction.text import CountVectorizer

        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,max_features=100,vocabulary=topFeatures,ngram_range=(1,ngramLength))

        tf = tf_vectorizer.fit_transform(data[(data['month']==Month) & (data['year']==Year)]['CleanedTitle'])



        # Now that we have got our features, let us perform some linear algebra functions to extract the important relations

        #AX=Y



        # Y is the comment count for each row of MONTH=1 and YEAR=2016

        # A is the TF matrix

        # Let us find the X, which will tell us the weightage of each word

        from scipy.sparse.linalg import lsqr

        x = lsqr(tf, data[(data['month']==Month) & (data['year']==Year)][evaluationColumn])



        # Since we have an asymmetric matrix, we will have to take help of LEast Squares Solution



        # The following is now the final coefficients

        FinalCoefficients=[[topFeatures[i],x[0][i]] for i in range(0,len(topFeatures))]

        FinalCoefficients=pd.DataFrame(FinalCoefficients)

        FinalCoefficients.columns=['Word','Weight']

        return FinalCoefficients

    else:

        return pd.DataFrame(columns=['Word','Weight'])
# Temporary Script to create a wordcloud

# Now we will create a wordCloud of the top 30 words

from wordcloud import WordCloud

from sklearn.preprocessing import LabelEncoder

import pandas as pd

import matplotlib 

matplotlib.rc('xtick', labelsize=20) 

# We will now be creating a weighted list

FinalCoefficients=pd.DataFrame()

for year in data['year'].unique():

    for month in data['month'].unique():

        global FinalCoefficients

        tempCoefficients=analysisMonthYear(month,year,1,'num_comments')

        FinalCoefficients=FinalCoefficients.append(tempCoefficients)



# Sum up across months

df=pd.DataFrame(columns=['Word','Values'])

df['Values']=FinalCoefficients.groupby(['Word'])['Weight'].sum().order(ascending=0)[1:30].values

df['Word']=FinalCoefficients.groupby(['Word'])['Weight'].sum().order(ascending=0)[1:30].index.values



# Create the word cloud

wordcloud = WordCloud().generate_from_frequencies([(row['Word'],row['Values']) for index, row in df.iterrows()])

plt.imshow(wordcloud)

plt.axis("off")

plt.suptitle("Important words in the data by Weight based on Num Comments",fontsize=20)

plt.show()



# WordCloud is just fancy, we want a bar chart

fig=plt.figure(figsize=(20,10))

y_pos = np.arange(len(df['Word']))

plt.bar(y_pos,df['Values'])

plt.xticks(y_pos, df['Word'],rotation=90)

plt.suptitle("Important words in the data by Weight based on Num Comments",fontsize=20)

plt.show()
## ANALYSIS 2 ###

# We will now create a timeline of the most important words

df=pd.DataFrame(columns=['Word','Values'])

df['Values']=FinalCoefficients.groupby(['Word'])['Weight'].count().order(ascending=0)[1:30].values

df['Word']=FinalCoefficients.groupby(['Word'])['Weight'].count().order(ascending=0)[1:30].index.values



# We will now create a plot based on its presence in each month's list of most important words

plt.figure(figsize=(20,10))

y_pos = np.arange(len(df['Word']))

plt.bar(y_pos,df['Values'])

plt.xticks(y_pos, df['Word'],rotation=90)

plt.suptitle("Important words in the data by count based on Num comments",fontsize=20)

plt.show()



# We can see that there is a lot of difference in the words that came here and the last bar plot

# This is because our TFID and LS on the matrix is being done over a month. We need to do it over the entire data set

# to get the complete picture



# So we can conclude that the above chart conains words which attract maximum responses
# Let us now do the analysis for upvotes

# We will now be creating a weighted list

FinalCoefficients=pd.DataFrame()

for year in data['year'].unique():

    for month in data['month'].unique():

        global FinalCoefficients

        tempCoefficients=analysisMonthYear(month,year,1,'num_points')

        FinalCoefficients=FinalCoefficients.append(tempCoefficients)



# Sum up across months

df=pd.DataFrame(columns=['Word','Values'])

df['Values']=FinalCoefficients.groupby(['Word'])['Weight'].count().order(ascending=0)[1:30].values

df['Word']=FinalCoefficients.groupby(['Word'])['Weight'].count().order(ascending=0)[1:30].index.values





# WordCloud is just fancy, we want a bar chart

plt.figure(figsize=(20,10))

y_pos = np.arange(len(df['Word']))

plt.bar(y_pos,df['Values'])

plt.xticks(y_pos, df['Word'],rotation=90)

plt.suptitle("Important words in the data by count based on Num Upvotes",fontsize=20)

plt.show()