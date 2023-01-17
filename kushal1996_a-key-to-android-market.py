from IPython.display import HTML
HTML('''
<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }

  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from collections import Counter as cntr
import os 
df = pd.read_csv(r"../input/googleplaystore.csv")

'''Treating duplicate values'''
df["seq"] =  np.arange(0 , df.shape[0])
df = pd.DataFrame.copy(df.sort_values(by = "Reviews", ascending = True))
df.drop_duplicates(subset = "App" , keep ="first" , inplace = True)
df = pd.DataFrame.copy(df.sort_values(by = "seq" , ascending = True))
df.set_index( np.arange(0 , df.shape[0]) , inplace = True)
del df["seq"]

'''Treating missing values'''
df["Rating"].fillna( round(df["Rating"].mean() , 1) , inplace = True )
df["Type"].fillna(df["Type"].mode()[0] , inplace = True )
df["Content Rating"].fillna(df["Content Rating"].mode()[0] , inplace = True)
df["Current Ver"].fillna(df["Current Ver"].mode()[0] , inplace = True)
df["Android Ver"].fillna(df["Android Ver"].mode()[0] , inplace = True)

'''Droping noisy data'''
df.drop(df.index[9298] , inplace = True)
df.drop(df.index[8025] , inplace = True)

'''Converting data types of variale Review , size , price and installs to numeric'''
df["Reviews"] = df["Reviews"].astype("float")

size_mb = [] 
for i in df["Size"].values:
    if "M" in i :
        i = i.replace("M","")
        size_mb.append(i)
    elif "k" in i :
        i = i.replace("k","")
        size_mb.append(round(float(i)*0.001 , 3 ))
    else:
        size_mb.append(0)
df["Size_mb"] = size_mb
df["Size_mb"] = df["Size_mb"].astype("float")

def dollar_to_rupees(df,x):
    price = []
    for i in df[x].values:
        i = i.replace("$", "")
        i = float(i)
        price.append(i)
    
    df["price_rupees"]  = price
    df["price_rupees"]  = df["price_rupees"] * 72.55
dollar_to_rupees(df , "Price")

installs_numeric = []
for i in df["Installs"].values :
    x = i.replace("+" , "")
    x = x.replace("," , "")
    installs_numeric.append(float(x))
df["installs_numeric"] =  installs_numeric

'''Cleaning and preparing data of second dataset.'''
df_reviews = pd.read_csv(r"../input/googleplaystore_user_reviews.csv")

'''treating missing values'''
df_reviews.dropna(inplace = True )

'''Cleaning'''
import re
import string
def clean_txt(text):
    text = text.lower()
    text = text.replace("(" , "")
    text = text.replace(")" , "")
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = text.replace( " \ " , "" )
    text = text.replace("/" , "")
    return text

cleaning1 = lambda x : clean_txt(x)

d = {"App" : df_reviews.App , 
    "Translated_Review" :df_reviews.Translated_Review.apply(cleaning1)}

df_sentiments_topic = pd.DataFrame(data = d)
    
def clean_text_round2(text):
    text = re.sub('[‘’“”…]', '', text)
    return text

round2 = lambda x: clean_text_round2(x)

d = {"App" : df_sentiments_topic.App , 
    "Translated_Review" :df_sentiments_topic.Translated_Review.apply(round2)}
df_sentiments_topic = pd.DataFrame(data = d)

def clean_text_round3(text):
    text = re.sub(r'[^\x00-\x7f]', '', text)
    return text

round3= lambda x: clean_text_round3(x)

d = {"App" : df_sentiments_topic.App , 
    "Translated_Review" :df_sentiments_topic.Translated_Review.apply(round3)}
df_sentiments_topic = pd.DataFrame(data = d)

'''Creating a corpus for every application by combinig every single review belonging to that application.'''
combined_review_dict = {}
for i in df_sentiments_topic["App"].unique():
    
    combined_review = "" 
    
    for review in df_sentiments_topic["Translated_Review"][df_sentiments_topic["App"] == i]:
        combined_review = combined_review +" "+review

    combined_review_dict[i] = combined_review

df_sentiments_topic = pd.DataFrame(data=combined_review_dict , index = [0])
df_sentiments_topic = pd.DataFrame.copy(df_sentiments_topic.T)
df_sentiments_topic.columns = ["Translated_Review"]
df_sentiments_topic["App"] = df_sentiments_topic.index

'''Sentiment analysis'''
from textblob import TextBlob

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

df_sentiments_topic["polarity"] = df_sentiments_topic["Translated_Review"].apply(pol)
df_sentiments_topic["subjectivity"] = df_sentiments_topic["Translated_Review"].apply(sub)

'''Document Term matrix'''
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(df_sentiments_topic.Translated_Review)
df_documentTermMatrix = pd.DataFrame(data_cv.toarray() , columns=cv.get_feature_names())
df_documentTermMatrix.index = df_sentiments_topic.index

df_documentTermMatrix = df_documentTermMatrix.transpose()
df_documentTermMatrix.head()

'''Merging datasets'''
df_2  = pd.merge(df , df_sentiments_topic)

'''Creating class for basic plots which can been easy to use rather then typing same code again again.'''
class fast_plot():
    def __init__(self):
        return None

    def count_plot_horizontal(self , y , data  , rotation , title):


        sns.countplot(y = y  , data = data  , order = data[y].value_counts().index)
        plt.yticks(rotation = rotation)
        plt.title(title)
        plt.show()

    def count_plot_verticale(self , x , data , rotation , title):

        sns.countplot(x = x  , data = data  , order = data[x].value_counts().index)

        plt.xticks(rotation = rotation)
        plt.title(title)
        plt.show()
        
    def basic(self , x , data):
        
        plt.subplot(121)
        data[x].plot(kind = "density")
        plt.title("Density plot")
        plt.subplot(122)
        plt.title("Histogram")
        data[x].plot(kind = "hist" , bins = 50)
        plt.show()
        
    def dist_hist(self , x , data , kind , row , col , colors ):

        ''' x and y should be list of variables to plot .'''
        count = 0
        for variable , color in zip(x,colors) :
            count += 1 
            plt.subplot(row , col , count )
            
            if kind == "hist":
                data[variable].plot(kind =  kind, bins = 50 , color = color )
                plt.title(variable)
            else:
                data[variable].plot(kind = kind , color = color)
                    
                plt.title(variable)
        plt.show()

    def scatter_plot(self , x  , y , data , row , col , title , colors ):

        count = 0
        for variable , color in zip(y , colors):
            count += 1 
            
            plt.subplot(row , col , count)
            plt.scatter(x = x  , y = variable , data = data , color = color , alpha = 0.5 )
            plt.xlabel(x)
            plt.ylabel(variable)
            
        plt.show()
        
easy_plot = (fast_plot())
plt.figure(1 , figsize = (15 , 10))
easy_plot.count_plot_horizontal(y = "Category" , data = df , rotation = 360 , title = "Count plot")
plt.figure(1 , figsize = (15 , 10))
plt.pie( [1861 , 940 , 828 , df.shape[0] - 1861 - 940 - 828] , explode = [0.1 , 0 , 0 , 0 ] ,
        labels = ["Family (1861)" , "Game (940)" , "Tools (828)" , "others"] , autopct='%1.1f%%', shadow=True,
        startangle=140 )
plt.axis("equal")
plt.title("")
plt.legend()
plt.show()
correlation_matrix = df.corr()
a4_dimens = (11.7 , 8.27)
fig , ax = plt.subplots(figsize = a4_dimens)
sns.heatmap(correlation_matrix, cmap="Blues", vmax=.8, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5} , 
            xticklabels=correlation_matrix.columns.values ,
            yticklabels=correlation_matrix.columns.values,
            ax = ax)

plt.xticks(rotation = 90)
plt.yticks(rotation = 360)
plt.show()
plt.figure(1 , figsize = (15 , 5))
easy_plot.basic(x = "Rating" , data = df)

plt.figure(2 , figsize = (15 , 8))
easy_plot.count_plot_verticale(x = "Rating" , data = df , rotation = 90 , title = "Count Plot")
print("Average rating for an app is ", round(df["Rating"].mean(),1) )
plt.figure(1 , figsize = (15 , 5))
df["Rating"][df["Type"] == "Free"].plot(kind = "density" , label = "Free")
df["Rating"][df["Type"] == "Paid"].plot(kind = "density" , label = "Paid")
plt.legend()
plt.show()
y_vals = ["Size_mb" ,"Reviews" ,"installs_numeric" , "price_rupees"]
colors = ["b" , "r" , "c" , "g"]
plt.figure(1  , figsize = (15 , 8))

easy_plot.scatter_plot(x = "Rating" , y = y_vals , data = df , row = 2 , col = 2 , title = "" , colors = colors)
plt.figure(1 , figsize = (15 , 5))
for i in df["Content Rating"].unique()[:5]:
    df["Rating"][df["Content Rating"] == i].plot(kind = "density" , label = i)
plt.legend()
plt.xlabel("Ratings")
plt.show()
for i in df["Content Rating"].unique():
    print("Average rating by",i, df["Rating"][df["Content Rating"] == i].mode()[0])
plt.figure(1 , figsize = (15 , 5))
easy_plot.count_plot_verticale(x = "Content Rating" , data = df , rotation = 360 , title = "Count Plot")
plt.figure(1 , figsize = (15 , 5))
easy_plot.basic(x = "Size_mb" , data = df)
print("Mean Size of an App", df["Size_mb"][df["Size_mb"] > 0].mean(),"MB\n",
     df["Size_mb"][df["Size_mb"] == 0].shape[0]," applications size varies from device to device")
y_vals = ["Rating" ,"Reviews" ,"installs_numeric" , "price_rupees"]
colors = ["b" , "r" , "c" , "g"]
plt.figure(1  , figsize = (15 , 8))
easy_plot.scatter_plot(x = "Size_mb" , y = y_vals , data = df , row = 2 , col = 2 , title = "" , colors = colors  )
plt.figure(1 , figsize = (15 , 5))
df["Size_mb"][df["Type"] == "Free"].plot(kind = "density" , label = "Free")
df["Size_mb"][df["Type"] == "Paid"].plot(kind = "density" , label = "Paid")
plt.legend()
plt.show()
plt.figure(1 , figsize = (15 , 5))
np.sqrt(df["Reviews"]).plot(kind = "hist" , bins = 100)
plt.title("Histogram")
plt.show()
print("Average reviews" , df["Reviews"].mean())
y_vals = ["Rating","Size_mb" ,"installs_numeric" , "price_rupees"]
colors = ["b" , "r" , "c" , "g"]
plt.figure(1  , figsize = (15 , 8))
for i , j , clrs in zip(np.arange(221 , 225) , y_vals , colors):
    plt.subplot(i)
    plt.scatter(x = np.sqrt(df["Reviews"]) , y = df[j]  , alpha = 0.5  , c = clrs )
    plt.xlabel("Reviews")
    plt.ylabel(j)
plt.show()
plt.figure(1 , figsize = (15 , 8))
easy_plot.count_plot_verticale(x = "Installs" , data = df , title = "Count Plot" , rotation = 90)
print("Average installs",df["installs_numeric"].mean())
y_vals = ["Rating","Size_mb" ,"Reviews" , "price_rupees"]
colors = ["b" , "r" , "c" , "g"]
plt.figure(1  , figsize = (15 , 8))
easy_plot.scatter_plot(x = "installs_numeric" , y = y_vals , data = df , row = 2 , col = 2 , title = "" , colors = colors)
plt.figure(1 , figsize = (15 , 5))
easy_plot.basic(x = "price_rupees" , data = df[df["price_rupees"] > 0])
print("Minimum price of an App $",df["price_rupees"][df["price_rupees"] > 0].min()*0.014,
     "\nAverage price of an App $",df["price_rupees"][df["price_rupees"] > 0].mean()*0.014,
     "\nMaximum price of an App $",df["price_rupees"].max()*0.014)
plt.figure(2 , figsize = (15 , 5))
easy_plot.count_plot_verticale(x = "Type" , data = df , rotation = 360 , title = "Count Plot")
print( (df["Type"][df["Type"] == "Paid"].shape[0] / df.shape[0])*100,"% of Apps are paid on play store" )
plt.figure(2 , figsize = (15 , 8))
sns.countplot(x = "Android Ver" , data = df , order = df["Android Ver"].value_counts().index)
plt.title("Android Version Count plot")
plt.xticks(rotation = 90)
plt.show()
plt.figure(1 , figsize = (15 , 8))
sns.boxplot(x = "Android Ver" , y = "Rating" , data = df)
plt.plot( np.arange(df["Android Ver"].unique().shape[0]) , 
         np.ones((df["Android Ver"].unique().shape[0] , 1 ))*np.mean(df["Rating"]) , "r--")
plt.xticks(rotation = 90)
plt.show()

plt.figure(1  , figsize = (20 , 10))
sns.violinplot(x = "Category" , y = "Rating" , data = df)
plt.plot( np.arange(df["Category"].unique().shape[0]) , 
         np.ones((df["Category"].unique().shape[0] , 1 ))*np.mean(df["Rating"]) , "black")
plt.xticks(rotation= 90)
plt.show()
plt.figure(1 , figsize = (15 , 10))
sns.stripplot( x = "price_rupees" , y = "Category"  , data = df , jitter = True)
plt.plot( np.ones((df["Category"].unique().shape[0] , ))*np.mean(df["price_rupees"][df["price_rupees"] > 0]), np.arange(df["Category"].unique().shape[0] ,) , "black")
plt.show()
plt.figure(1 , figsize = (15 , 5))
easy_plot.count_plot_verticale(x = "Sentiment" , data = df_reviews , rotation = 360 , title ="Sentiments")
plt.figure(1 , figsize = (15 , 5))
easy_plot.basic(data = df_reviews , x  = "Sentiment_Polarity")
plt.figure(1 , figsize  = (15 , 5))
plt.subplot(121)
df_reviews["Sentiment_Polarity"][df_reviews["Sentiment_Polarity"] < 0 ].plot(kind = "hist" , bins = 50 , color = "red")
plt.title("Negative Sentiments")

plt.subplot(122)
df_reviews["Sentiment_Polarity"][df_reviews["Sentiment_Polarity"] > 0 ].plot(kind = "hist" , bins = 50 , color = "green")
plt.title("Postive Sentiments")

plt.show()
plt.figure(1 , figsize = (15 , 5))
easy_plot.basic(data = df_reviews , x  = "Sentiment_Subjectivity")
plt.figure(1 , figsize  = (15 , 5))
easy_plot.dist_hist(x = ["polarity" , "subjectivity"] , data = df_sentiments_topic , kind = "hist", 
                    row = 1 , col = 2 , colors = ["red" , "green"])
avg_pol = []
avg_subj = []
cats = []

for cat in df_2["Category"].unique():
    
    avg_pol.append( df_2["polarity"][df_2["Category"] == cat].mean() )
    avg_subj.append( df_2["subjectivity"][df_2["Category"] == cat].mean() )
    cats.append(cat)
    
plt.figure(1 , figsize = (15 , 5 ) )

plt.plot(np.arange(len(cats)) , avg_pol  , "b" )
plt.plot(np.arange(len(cats)) , avg_pol  , "ro" )
plt.plot(np.arange(len(cats)) ,  np.arange(len(cats)) * 0  , "r" , label = "Neutral" )

plt.title("Average Polarity Category wise")
plt.ylabel("<---Negative------Neutral(0)------Positive--->")
plt.legend()
plt.xticks(np.arange(len(cats)) , cats , rotation = 90)
plt.show()
plt.figure(2 , figsize = (15 , 5 ) )
plt.plot(np.arange(len(cats)) , avg_subj  , "b" )
plt.plot(np.arange(len(cats)) , avg_subj  , "go" )
plt.plot(np.arange(len(cats)) ,  np.ones((len(cats) ,)) * 0.5  , "g" )

plt.title("Average Subjectivity Category wise")
plt.ylabel("<---Fact------Neutral(0.50)------opinion--->")
plt.xticks(np.arange(len(cats)) , cats , rotation = 90)

plt.show()
from wordcloud import WordCloud
from sklearn.feature_extraction import text
wc = WordCloud(stopwords = text.ENGLISH_STOP_WORDS.union(["game" , "app" , "time" , "play" , "like"])
               , background_color = "white" , colormap = "Dark2" , 
               max_font_size = 150 , random_state = 42)


plt.rcParams['figure.figsize'] = [16 , 10]
corpus = " "
ind = 0
for cat  in df_2["Category"].unique():
    ind += 1
    for review in df_2["Translated_Review"][df_2["Category"] ==  cat ]:
        corpus = corpus+" "+review
        
    plt.subplot(6 , 6 , ind )
    wc.generate(corpus)
    plt.imshow(wc , interpolation="bilinear")
    coprus = ""
    plt.axis("off")
    plt.title(cat)
plt.show()
df[df["installs_numeric"] == df["installs_numeric"].max()][["App" , "Category" , "Rating" , "Installs" , "Reviews"]]
print("There are "+str(df[df["installs_numeric"] == df["installs_numeric"].max()].shape[0])+
      " Applications which have downloads more than 1,000,000,000+. Most of the apps in the most downloaded Apps belong to Google.")
df_top_Apps =  pd.DataFrame.copy(df[df["installs_numeric"] == df["installs_numeric"].max()])
value = []
label = []
for i in np.arange(len(cntr(df_top_Apps["Category"]))):
    value.append(cntr(df_top_Apps["Category"]).most_common()[i][1])
    label.append(cntr(df_top_Apps["Category"]).most_common()[i][0])
plt.figure(1, figsize = (15 ,7))
plt.pie( value ,labels = label , autopct='%1.1f%%', shadow=True)
plt.axis("equal")
plt.legend()
plt.title("")
plt.show()
plt.figure(1 , figsize = (15 , 5) )

plt.plot(np.arange(df_top_Apps.shape[0]) , df_top_Apps["Rating"] , "g-")
plt.plot(np.arange(df_top_Apps.shape[0]) , df_top_Apps["Rating"] , "bo")
plt.plot(np.arange(df_top_Apps.shape[0]) , np.ones((df_top_Apps.shape[0],))*df["Rating"].mean() , "b-" ,
         label = "Average Rating")
for i , j in zip(np.arange(df_top_Apps.shape[0]) , df_top_Apps["Rating"]):
    plt.annotate( round(j,1) , xy = (i , j) , xytext = (i , j + 0.01) )
plt.xticks(np.arange(df_top_Apps.shape[0]),df_top_Apps["App"] , rotation =90)
plt.title("Ratings of Applications with 1,000,000,000+ downloads")
plt.legend
plt.show()
plt.figure(1 , figsize = (15 , 7) )
plt.plot(np.arange(df_top_Apps.shape[0]) , df_top_Apps["Reviews"] , "g-")
plt.plot(np.arange(df_top_Apps.shape[0]) , df_top_Apps["Reviews"] , "ro")
plt.plot(np.arange(df_top_Apps.shape[0]) , np.ones((df_top_Apps.shape[0],))*df["Reviews"].mean() , "b-" ,
         label = "Average Reviews")
for i , j in zip(np.arange(df_top_Apps.shape[0]) , df_top_Apps["Reviews"]):
    if j > df_top_Apps["Reviews"].mean():
        plt.annotate( round(j,1) , xy = (i , j) , xytext = (i , j + 1000000) )
plt.xticks(np.arange(df_top_Apps.shape[0]),df_top_Apps["App"] , rotation =90)
plt.title("Reviews of Applications with 1,000,000,000+ downloads")
plt.legend()
plt.show()
plt.figure(1 , figsize = (15 , 5))
easy_plot.count_plot_verticale(x = "Content Rating" , data = df_top_Apps , rotation = 360 , title = "")
df_most_expensive_apps = pd.DataFrame.copy(df.sort_values(by = "price_rupees" , ascending=False).head(20))
df_most_expensive_apps[["App" , "Category" , "Rating" , "price_rupees" , "Price" , "Installs"]]
games_df = pd.DataFrame.copy(df[df["Category"] == "GAME"])

def four_density(data):
    plt.figure(1 , figsize = (15 , 8))
    vals = ["Rating" , "Reviews" , "Size_mb" , "price_rupees"]
    colr = ["b" , "red" , "y" , "g"]
    for i , n , c in zip(vals,np.arange(221,225) ,colr):
        plt.subplot(n)
        if i == "Size_mb" or i == "price_rupees":
            data[i][data[i] > 0].plot(kind = "density" , color= c)
        else:
            data[i].plot(kind = "density" , color= c)

        plt.xlabel(i)
    plt.show()

four_density(games_df)
vals = ["Rating" , "Reviews" , "Size_mb" , "price_rupees"]
for i in vals :
    if i == "Size_mb" or i == "price_rupees":
        print("Average "+str(i)+" "+str(games_df[i][games_df[i] > 0].mean()) )
    else:
        print("Average "+str(i)+" "+str(games_df[i].mean()) )
plt.figure(1 , figsize = (15 , 8))
easy_plot.count_plot_horizontal(y = "Genres" , data = games_df , rotation = 360 , title = "Genres")
print((299/games_df.shape[0])*100 ,"% are Action games.\n")
plt.figure(1 , figsize = (15 , 9))
sns.stripplot( x = "price_rupees" , y = "Genres"  , data = games_df , jitter = True)
plt.plot( np.ones((games_df["Genres"].unique().shape[0] , ))*np.mean(df["price_rupees"]), 
         np.arange(games_df["Genres"].unique().shape[0] ,) , "black" , label = "Average price line")
plt.title("Price of games according to genres")
plt.legend()
plt.show()
plt.figure(1  , figsize = (20 , 10))
sns.violinplot(y = "Rating" , x = "Genres" , data = games_df)
plt.plot( np.arange(games_df["Genres"].unique().shape[0]) , 
         np.ones((games_df["Genres"].unique().shape[0] , 1 ))*np.mean(df["Rating"]) , "blue" , label = "Average rating line")
plt.xticks(rotation = 90)
plt.legend()
plt.title("Ratings of games according to genres")
plt.show()
value = []
label = []
for i in np.arange(len(cntr(games_df["Content Rating"]))):
    value.append(cntr(games_df["Content Rating"]).most_common()[i][1])
    label.append(cntr(games_df["Content Rating"]).most_common()[i][0])
plt.figure(1, figsize = (15 ,7))
plt.pie( value ,labels = label , autopct='%1.1f%%', shadow=True)
plt.axis("equal")
plt.legend()
plt.show()
games_df[games_df["installs_numeric"] == games_df["installs_numeric"].max()][["App" , "Rating" , "Installs" , 
                                                                              "Genres" , "Size" ,"Type", "Last Updated"]]
games_df2 = pd.DataFrame.copy(df_2[df_2["Category"] == "GAME"])
plt.figure(1 , figsize = (15 , 5))
plt.subplot(121)
games_df2["polarity"].plot(kind = "hist" , bins = 50 , color = "red")
games_df2["polarity"].plot(kind = "density")
plt.title("Polarity")

plt.subplot(122)
games_df2["subjectivity"].plot(kind = "hist" , bins = 50 , color = "green")
games_df2["subjectivity"].plot(kind = "density")
plt.title("Subjectivity")
plt.show()

def word_cloud_combined(x , data , stop_word):
    

    combined_review = ""
    for review in data[x]:
        combined_review = combined_review +" "+review
    
    wc = WordCloud(stopwords = text.ENGLISH_STOP_WORDS.union(stop_word) , background_color = "white" , colormap = "Dark2" , 
               max_font_size = 150 , random_state = 42)
    wc.generate(combined_review)
    plt.rcParams['figure.figsize'] = [15 , 6]
    plt.imshow(wc , interpolation="bilinear")
    plt.axis("off")
    plt.show()

word_cloud_combined(x = "Translated_Review" , data = games_df2 , stop_word = ["game" , "like" , "play"])
def word_cloud(x , data , vals  , row , col , stop_word ):
    
    count = 0
    for v in vals:
        count += 1
    
    
        wc = WordCloud(stopwords = text.ENGLISH_STOP_WORDS.union(stop_word) ,
                   background_color = "white" , colormap = "Dark2" , 
               max_font_size = 150 , random_state = 42)
    
        corpus = [ w for w in data[x][data["App"] == v]]
        wc.generate(str(corpus))
    
        plt.rcParams['figure.figsize'] = [15 , 6]
        plt.subplot(row , col, count)
        plt.imshow(wc , interpolation="bilinear")
        plt.title(v)
        plt.axis("off")

    plt.show()
    
top_games = ["Extreme Car Driving Simulator" , "Fruit Ninja®" , "Asphalt 8: Airborne" , 
             "Angry Birds Rio" , "Candy Crush Soda Saga" , "DEAD TARGET: FPS Zombie Apocalypse Survival Games"]
    
word_cloud(x = "Translated_Review" , data = games_df2 , vals = top_games ,
           row = 2 , col = 3 , stop_word =["game" , "like" , "play" , "said"])
communication_df = pd.DataFrame.copy(df[df["Category"] == "COMMUNICATION"])
communication_df2 = pd.DataFrame.copy(df_2[df_2["Category"] == "COMMUNICATION"])

four_density(data = communication_df)
vals = ["Rating" , "Reviews" , "Size_mb" , "price_rupees"]
for i in vals :
    if i == "Size_mb" or i == "price_rupees":
        print("Average "+str(i)+" "+str(communication_df[i][communication_df[i] > 0].mean()) )
    else:
        print("Average "+str(i)+" "+str(communication_df[i].mean()) )
plt.figure(1 , figsize = (15 , 5))
plt.subplot(121)
communication_df2["polarity"].plot(kind = "hist" , bins = 50 , color = "red")
communication_df2["polarity"].plot(kind = "density")
plt.title("Polarity")

plt.subplot(122)
communication_df2["subjectivity"].plot(kind = "hist" , bins = 50 , color = "green")
communication_df2["subjectivity"].plot(kind = "density")
plt.title("Subjectivity")
plt.show()
word_cloud_combined(x = "Translated_Review" , data = communication_df2 , stop_word = ["message" , "app" , "time" , "like" , "work" , "phone"])
top_com = pd.DataFrame.copy(communication_df.sort_values(by = "installs_numeric" , ascending = False).head(10))
top_com[["App" ,"Rating","Size","Reviews","Installs"]]
plt.figure(1 , figsize = (15 , 5))
plt.title("Rating and Size of top 10 communication Apps")
plt.subplot(121)
plt.plot(np.arange(top_com["App"].shape[0]) , top_com["Rating"],"b")
plt.plot(np.arange(top_com["App"].shape[0]) , top_com["Rating"],"go")

plt.plot(np.arange(top_com["App"].shape[0]) , np.ones( (top_com.shape[0], ))*df["Rating"].mean() , "r" , 
         label = "Average Rating line")

plt.xticks(np.arange(top_com.shape[0]) , top_com["App"] , rotation = 90 )
plt.ylabel("<___________________Ratings___________________>")
plt.title("Ratings of top communication apps")
plt.legend()

plt.subplot(122)
plt.plot(np.arange(top_com["App"].shape[0]) , top_com["Size_mb"],"b")
plt.plot(np.arange(top_com["App"].shape[0]) , top_com["Size_mb"],"go")

plt.plot(np.arange(top_com["App"].shape[0]) , np.ones( (top_com.shape[0], ))*df["Size_mb"][df["Size_mb"] > 0].mean() , "r" , 
         label = "Average Size line")

plt.xticks(np.arange(top_com.shape[0]) , top_com["App"] , rotation = 90 )
plt.ylabel("<___________________Size in MB___________________>")
plt.title("Size of top communication apps")
plt.legend()


plt.show()
plt.figure(1 , figsize = (15 , 5) )

easy_plot.count_plot_verticale(x = "Content Rating" , data = top_com , rotation = 360 , title = "Age group target of Communication Apps" )
top_com2 = communication_df2.sort_values(by = "installs_numeric" , ascending = False).head(10)
from nltk.corpus import words
vals = ["Hangouts" , "Gmail" , "Google Duo - High Quality Video Calls" , "Firefox Browser fast & private"]
top_dict = {}
for c in df_documentTermMatrix.columns:
    if c in vals:
        
        top = df_documentTermMatrix[c].sort_values(ascending=False).head(30)
        top_dict[c]= list(zip(top.index, top.values))

stop_words = [ "thank" , "say" , "need" , "mail" , "sure" , "thing" , "opens" , "thanks" , "able" , "thats" , "doesnt"]
for app in vals:
    top =  [word for (word , count) in top_dict[app] if count > 8 or word not in words.words() ]
    for t in top :
        stop_words.append(t)

word_cloud(x = "Translated_Review" , data = top_com2 , vals = vals ,
           row = 2 , col = 2 , stop_word = stop_words)
df_social = pd.DataFrame.copy(df[df["Category"] == "SOCIAL"])
df_social2 = pd.DataFrame.copy(df_2[df_2["Category"] == "SOCIAL"])

four_density(df_social)
vals = ["Rating" , "Reviews" , "Size_mb" , "price_rupees"]
for i in vals :
    if i == "Size_mb" or i == "price_rupees":
        print("Average "+str(i)+" "+str(df_social[i][df_social[i] > 0].mean()) )
    else:
        print("Average "+str(i)+" "+str(df_social[i].mean()) )
plt.figure(1 , figsize = (15 , 5))
plt.subplot(121)
df_social2["polarity"].plot(kind = "hist" , bins = 50 , color = "red")
df_social2["polarity"].plot(kind = "density")
plt.title("Polarity")

plt.subplot(122)
df_social2["subjectivity"].plot(kind = "hist" , bins = 50 , color = "green")
df_social2["subjectivity"].plot(kind = "density")
plt.title("Subjectivity")
plt.show()
word_cloud_combined(x = "Translated_Review" , data = df_social2 , stop_word = ["app" , "like" , "time" , "work" , "able" , "thing" , 
                                                         "need" , "want" , "im" , "know" , "really"])
top_social = pd.DataFrame.copy(df_social.sort_values(by = "installs_numeric" , ascending = False).head(10))
top_social[["App" ,"Rating","Size","Reviews","Installs"]]
plt.figure(1 , figsize = (15 , 5))
plt.title("Rating and Size of top 10 communication Apps")
plt.subplot(121)
plt.plot(np.arange(top_social["App"].shape[0]) , top_social["Rating"],"b")
plt.plot(np.arange(top_social["App"].shape[0]) , top_social["Rating"],"go")

plt.plot(np.arange(top_social["App"].shape[0]) , np.ones( (top_social.shape[0], ))*df["Rating"].mean() , "r" , 
         label = "Average Rating line")

plt.xticks(np.arange(top_social.shape[0]) , top_social["App"] , rotation = 90 )
plt.ylabel("<___________________Ratings___________________>")
plt.title("Ratings of top social media apps")
plt.legend()

plt.subplot(122)
plt.plot(np.arange(top_social["App"].shape[0]) , top_social["Size_mb"],"b")
plt.plot(np.arange(top_social["App"].shape[0]) , top_social["Size_mb"],"go")

plt.plot(np.arange(top_social["App"].shape[0]) , np.ones( (top_social.shape[0], ))*df["Size_mb"][df["Size_mb"] > 0].mean() , "r" , 
         label = "Average Size line")

plt.xticks(np.arange(top_social.shape[0]) , top_social["App"] , rotation = 90 )
plt.ylabel("<___________________Size in MB___________________>")
plt.title("Size of top social media apps")
plt.legend()

plt.show()
plt.figure(1 , figsize = (15 , 7) )
plt.plot(np.arange(top_social.shape[0]) , top_social["Reviews"] , "b-")
plt.plot(np.arange(top_social.shape[0]) , top_social["Reviews"] , "ro")
plt.plot(np.arange(top_social.shape[0]) , np.ones((top_social.shape[0],))*df["Reviews"].mean() , "r-" ,
         label = "Average Review line")
for i , j in zip(np.arange(top_social.shape[0]) , top_social["Reviews"]):
    if j > df["Reviews"].mean():
        plt.annotate( round(j,1) , xy = (i , j) , xytext = (i , j + 1000000) )
plt.xticks(np.arange(top_social.shape[0]),top_social["App"] , rotation =90)
plt.legend()
plt.title("Reviews on top Social media apps")
plt.show()
top_social2 = pd.DataFrame.copy(df_social2.sort_values(by = "installs_numeric" , ascending = False).head(10))
vals = ["Google+" , "Facebook" , "Facebook Lite" , "Badoo - Free Chat & Dating App"]

word_cloud(x = "Translated_Review" , data = top_social2 , vals = vals ,
           row = 2 , col = 2 , stop_word = ["thing","im", "like" , "time"])

df_tool = pd.DataFrame.copy(df[df["Category"] == "TOOLS"])
df_tool2 = pd.DataFrame.copy(df_2[df_2["Category"] == "TOOLS"])

four_density(data = df_tool)
vals = ["Rating" , "Reviews" , "Size_mb" , "price_rupees"]
for i in vals :
    if i == "Size_mb" or i == "price_rupees":
        print("Average "+str(i)+" "+str(df_tool[i][df_tool[i] > 0].mean()) )
    else:
        print("Average "+str(i)+" "+str(df_tool[i].mean()) )
plt.figure(1 , figsize = (15 , 5))
plt.subplot(121)
df_tool2["polarity"].plot(kind = "hist" , bins = 50 , color = "red")
df_tool2["polarity"].plot(kind = "density")
plt.title("Polarity")

plt.subplot(122)
df_tool2["subjectivity"].plot(kind = "hist" , bins = 50 , color = "green")
df_tool2["subjectivity"].plot(kind = "density")
plt.title("Subjectivity")
plt.show()
word_cloud_combined(x = "Translated_Review" , data = df_tool2 , stop_word = ["phone" , "work" , "like" , "really" ,
                                                                             "using" , "time", "app" , "want"])
top_tools = pd.DataFrame.copy(df_tool.sort_values(by = "installs_numeric" , ascending = False).head(10) )
top_tools[["App" , "Rating" , "Size" , "Installs" , "Android Ver"]]
plt.figure(1 , figsize = (15 , 5))
plt.title("Rating and Size of top 10 communication Apps")
plt.subplot(121)
plt.plot(np.arange(top_tools["App"].shape[0]) , top_tools["Rating"],"b")
plt.plot(np.arange(top_tools["App"].shape[0]) , top_tools["Rating"],"go")

plt.plot(np.arange(top_tools["App"].shape[0]) , np.ones( (top_tools.shape[0], ))*df["Rating"].mean() , "r" , 
         label = "Average Rating line")

plt.xticks(np.arange(top_tools.shape[0]) , top_tools["App"] , rotation = 90 )
plt.ylabel("<___________________Ratings___________________>")
plt.title("Ratings of top tool apps")
plt.legend()

plt.subplot(122)
plt.plot(np.arange(top_tools["App"].shape[0]) , top_tools["Size_mb"],"b")
plt.plot(np.arange(top_tools["App"].shape[0]) , top_tools["Size_mb"],"go")

plt.plot(np.arange(top_tools["App"].shape[0]) , np.ones( (top_tools.shape[0], ))*df["Size_mb"][df["Size_mb"] > 0].mean() , "r" , 
         label = "Average Size line")

plt.xticks(np.arange(top_tools.shape[0]) , top_tools["App"] , rotation = 90 )
plt.ylabel("<___________________Size in MB___________________>")
plt.title("Size of top tool apps")
plt.legend()

plt.show()
plt.figure(1 , figsize = (15 , 7) )
plt.plot(np.arange(top_tools.shape[0]) , top_tools["Reviews"] , "b-")
plt.plot(np.arange(top_tools.shape[0]) , top_tools["Reviews"] , "ro")
plt.plot(np.arange(top_tools.shape[0]) , np.ones((top_tools.shape[0],))*df["Reviews"].mean() , "r-" ,
         label = "Average Review line")
for i , j in zip(np.arange(top_tools.shape[0]) , top_tools["Reviews"]):
    if j > df["Reviews"].mean():
        plt.annotate( round(j,1) , xy = (i , j) , xytext = (i , j + 1000000) )
plt.xticks(np.arange(top_tools.shape[0]),top_tools["App"] , rotation =90)
plt.title("Reviews on top tool apps")
plt.legend()
plt.show()
top_tools2 = pd.DataFrame.copy(df_tool2.sort_values(by = "installs_numeric" , ascending = False).head(10) )

vals = ["Google" , "Google Translate" , "Gboard - the Google Keyboard" , "AppLock"]

word_cloud(x = "Translated_Review" , data = top_tools2 , vals = vals ,
           row = 2 , col = 2 , stop_word = ["thing","im", "like" , "time"])
df_ed = pd.DataFrame.copy(df[df["Category"] == "EDUCATION"])
df_ed2 = pd.DataFrame.copy(df_2[df_2["Category"] == "EDUCATION"])

four_density(data = df_ed)
vals = ["Rating" , "Reviews" , "Size_mb" , "price_rupees"]
for i in vals :
    if i == "Size_mb" or i == "price_rupees":
        print("Average "+str(i)+" "+str(df_ed[i][df_ed[i] > 0].mean()) )
    else:
        print("Average "+str(i)+" "+str(df_ed[i].mean()) )
plt.figure(1 , figsize = (15 , 5))
plt.subplot(121)
df_ed2["polarity"].plot(kind = "hist" , bins = 50 , color = "red")
df_ed2["polarity"].plot(kind = "density")
plt.title("Polarity")

plt.subplot(122)
df_ed2["subjectivity"].plot(kind = "hist" , bins = 50 , color = "green")
df_ed2["subjectivity"].plot(kind = "density")
plt.title("Subjectivity")
plt.show()
word_cloud_combined(x = "Translated_Review" , data = df_ed2 , stop_word = ["like" , "really" , "need" , "app" , "thank" ,
                                                                          "learn"] )
top_ed = pd.DataFrame.copy(df_ed.sort_values(by = "installs_numeric" , ascending = False).head(10) )
top_ed[["App" , "Rating" , "Size" , "Installs" , "Android Ver"]]
plt.figure(1 , figsize = (15 , 5))
plt.title("Rating and Size of top 10 communication Apps")
plt.subplot(121)
plt.plot(np.arange(top_ed["App"].shape[0]) , top_ed["Rating"],"b")
plt.plot(np.arange(top_ed["App"].shape[0]) , top_ed["Rating"],"go")

plt.plot(np.arange(top_ed["App"].shape[0]) , np.ones( (top_ed.shape[0], ))*df["Rating"].mean() , "r" , 
         label = "Average Rating line")

plt.xticks(np.arange(top_ed.shape[0]) , top_ed["App"] , rotation = 90 )
plt.ylabel("<___________________Ratings___________________>")
plt.title("Ratings of top educational apps")
plt.legend()

plt.subplot(122)
plt.plot(np.arange(top_ed["App"].shape[0]) , top_ed["Size_mb"],"b")
plt.plot(np.arange(top_ed["App"].shape[0]) , top_ed["Size_mb"],"go")

plt.plot(np.arange(top_ed["App"].shape[0]) , np.ones( (top_ed.shape[0], ))*df["Size_mb"][df["Size_mb"] > 0].mean() , "r" , 
         label = "Average Size line")

plt.xticks(np.arange(top_ed.shape[0]) , top_ed["App"] , rotation = 90 )
plt.ylabel("<___________________Size in MB___________________>")
plt.title("Size of top educational apps")
plt.legend()

plt.show()
top_ed2 = pd.DataFrame.copy(df_ed2.sort_values(by = "installs_numeric" , ascending = False).head(10) )

vals = ["Duolingo: Learn Languages Free" , "Edmodo" , "C++ Programming" , "Brilliant"]

word_cloud(x = "Translated_Review" , data = top_ed2 , vals = vals ,
           row = 2 , col = 2 , stop_word = ["thing","im", "like" , "time"])