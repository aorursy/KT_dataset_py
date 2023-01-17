import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import string

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from wordcloud import WordCloud

from nltk.sentiment.vader import SentimentIntensityAnalyzer



from PIL import Image

import os

print(os.listdir("../input"))
clothes_data = pd.read_csv("../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv")

clothes_data.head().T
clothes_data.info()
clothes_data[clothes_data['Title'].isnull()]
clothes_data.isnull().sum()
clothes_data['Title'] = clothes_data['Title'].replace('nan',np.NaN)

clothes_data['Review Text'] = clothes_data['Review Text'].replace('nan',np.NaN)



clothes_data.dropna(subset=['Title'], how='all', inplace = True)

clothes_data.dropna(subset=['Review Text'], how='all', inplace = True)

clothes_data = clothes_data[clothes_data['Department Name'] != 'General']
mode_division = clothes_data['Division Name'].mode()[0]

mode_department = clothes_data['Department Name'].mode()[0]

mode_class = clothes_data['Class Name'].mode()[0]



clothes_data['Division Name'] = clothes_data['Division Name'].fillna(mode_division)

clothes_data['Department Name'] = clothes_data['Department Name'].fillna(mode_division)

clothes_data['Class Name'] = clothes_data['Class Name'].fillna(mode_division)
clothes_data[["Title", "Division Name","Department Name","Class Name","Review Text"]].describe(include=["O"]).T.drop("count",axis=1)
sns.set(rc={"font.style":"normal",

            "axes.grid":False,

            'axes.labelsize':25,

            'figure.figsize':(12.0,12.0),

            'xtick.labelsize':15,

            'ytick.labelsize':15}) 
SentAnalyzer = SentimentIntensityAnalyzer()



clothes_data['Title'] = clothes_data['Title'].astype(str)



clothes_data['Polarity Score'] = clothes_data['Title'].apply(lambda x:SentAnalyzer.polarity_scores(x)['compound'])



clothes_data['Sentiment'] = ''



clothes_data.loc[clothes_data['Polarity Score'] > 0,'Sentiment'] = 'Positive'



clothes_data.loc[clothes_data['Polarity Score'] < 0,'Sentiment'] = 'Negative'

clothes_data.loc[clothes_data['Polarity Score'] == 0,'Sentiment'] = 'Neutral'

clothes_data.head()
xvar = "Sentiment"

huevar ="Rating"

rowvar =  "Department Name"



sns.countplot(x=xvar,hue=rowvar,data=clothes_data,

         order=["Negative","Neutral","Positive"])

plt.title("Sentiment Vs Rating of each Department",fontsize=20,fontweight='bold')

plt.yticks(fontsize=15,fontweight='bold')

plt.xticks(fontsize=15,fontweight='bold')

plt.ylabel("")

plt.show()
age = pd.cut(clothes_data['Age'],bins = [0, 25, 45, 60, 75, 100],labels = ['<=25', '<=45', '<=60', '<=75', '<=100',])

print("Different Age Group Count:")

print(age.value_counts().sort_values())
sns.countplot(x=age,hue=clothes_data['Rating'], palette = sns.color_palette('bright', 5))

plt.title("Rating by different age groups",fontsize=20,fontweight='bold')

plt.ylabel("")

plt.xlabel('Age',fontsize=15,fontweight='bold')

plt.xticks(fontsize=15,fontweight='bold')

plt.show()
sns.heatmap(pd.crosstab(clothes_data['Class Name'], 

        clothes_data["Department Name"]),

            annot=True,fmt='g', cmap="Pastel1_r")

plt.title("Class Name Count Vs Department Name",fontsize=20,fontweight='bold')

plt.show()



sns.heatmap(pd.crosstab(clothes_data['Class Name'], clothes_data["Division Name"]),

            annot=True,fmt='g', cmap="Pastel1")

plt.title("Class Name Count Vs Division Name",fontsize=20,fontweight='bold')



plt.show()



sns.heatmap(pd.crosstab(clothes_data['Department Name'], clothes_data["Division Name"]),

            annot=True,fmt='g', cmap="Pastel2_r")

plt.title("Department Name Count Vs Division Name",fontsize=20,fontweight='bold')



plt.show()
def wordcloud(text,my_mask=None):

    wordcloud = WordCloud(width=800,height=800,max_words=50,collocations=False,

    min_font_size=10,contour_width=2, contour_color='cadetblue',mask=my_mask,background_color='white').generate(text)



    plt.imshow(wordcloud,interpolation="bilinear")

    plt.axis('off')

    plt.show()



def tokenize(text):

    

    stop_words = set(stopwords.words('english'))

    

    token =word_tokenize(text)

    

    word_token = []



    for w in token:

        if w not in stop_words and not w.isdigit() and w.isalnum() :

            word_token.append(str(w))

    

    freq = nltk.FreqDist(word_token)



    new_list = []

    for k,v in freq.items():

        if v >= 2:

            new_list.append(k)

    

    freq.plot(10,cumulative=False)

    plt.show()

    return (str(new_list))
positive = clothes_data[(clothes_data['Rating'] >3) & 

                        (clothes_data['Recommended IND'] ==1) & 

                       (clothes_data['Sentiment'] =='Positive')]



plt.title("Positive Titles Plot",fontsize=25,fontstyle='oblique')

ret_text = tokenize(str(positive['Title']).lower())



plt.title("Positive Titles",fontsize=25,fontstyle='oblique')

wordcloud(ret_text)

negative = clothes_data[(clothes_data['Rating'] <=2) & 

                        (clothes_data['Recommended IND'] ==0) & 

                       (clothes_data['Sentiment'] =='Negative')]



plt.title("Negative Titles Plot",fontsize=25,fontstyle='oblique')

ret_text = tokenize(str(negative['Title']).lower())



plt.title("Negative Titles",fontsize=25,fontstyle='oblique')

wordcloud(ret_text)
my_mask = np.array(Image.open("../input/dressdress/dress.jpg"))

ret_text = tokenize(str(clothes_data['Review Text']).lower())

wordcloud(ret_text,my_mask)