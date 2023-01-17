import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import wordcloud
import re
import matplotlib.pyplot as plt 


with open("/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv") as amazon_alexa_tsv:
    tsv = pd.read_csv(amazon_alexa_tsv, delimiter="\t")
    

def process_wordcloud(df):
    regex = re.compile('[^a-zA-Z]')
    words = ''
    for review in df['verified_reviews']:
        review = review.split()
        for word in review:
            word = regex.sub('', word).lower() #remove non-alphabet characters and convert to lowercase
            words += word + " "
    wc = wordcloud.WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = set(wordcloud.STOPWORDS), 
                    min_font_size = 10).generate(words)
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wc) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show() 
print(f"The data contain {tsv.shape[0]} reviews between {tsv.date.min()} and {tsv.date.max()}.")
print(f"The average rating is {tsv['rating'].mean():0.2f}.\nHere is how the ratings are distributed: ")
hist = tsv['rating'].hist(bins=np.arange(0.5, 6.5, 1))
print("Here is a breakdown of the average rating for each model:")
by_variation = tsv.groupby(['variation'])['rating'].mean().sort_values().plot.bar()
print("Here is what the best reviews mention:")
process_wordcloud(tsv[tsv['rating'] == 5])
print("Here is a wordcloud for poor reviews only (1-3 stars): ")
process_wordcloud(tsv[tsv['rating'] <= 3])

