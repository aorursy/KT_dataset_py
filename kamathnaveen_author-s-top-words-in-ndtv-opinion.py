import pandas as pd # Load author & links to DataFrame 
import requests # Getting Webpage content
from bs4 import BeautifulSoup as bs # Scraping webpages
import matplotlib.pyplot as plt # Plotting 
%matplotlib inline
#Display all content's in DataFrame
pd.set_option('display.max_rows', 500) 
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
def b_soup(url):
    """Check website if active"""
    response = requests.get(url)
    if response.status_code == 200:                             #Get data if website is active
        return bs(response.content, 'html.parser')               
    else:
        print('Bad URL Pls check')    
def author(soup):
    """ Get all autors who have opinion """
    data = author_soup.findAll('ul',attrs={'class':'all_author'}) # all autors are in class = all_author
    authors = {}
    for div in data:
        links = div.findAll('a')
        for item in links:
            response = requests.get(item['href']) 
            if response.status_code == 200:                       # update only if opinion is active
                authors.update({item['alt']:item['href']}) 
    return authors
        
def b_soup_opinion_pages(url):
    """Get all opinion pages for author"""
    author_opinion_soup = b_soup(url)
    if author_opinion_soup != None:
        author_pages = author_opinion_soup.findAll('div',attrs={'class':'new_pagination'})
        pages = [author_opinion]
        for page in author_pages:
            links = page.findAll('a')
            for item in links:
                response = requests.get(item['href'])
                if response.status_code == 200:
                    pages.append((item['href']))                  # update all opinion pages
    return pages
def opinion_urls(url):
    """Get all Opinion Links for author"""
    page_link = []
    #url = pg[3]
    if len(pg)>2:                            # update all individual opinion links for authors with >1 page
        for i in range(len(pg)-1):           # Ignore >>> link's at the end of page to navigate to next page.
            opinion_soup = b_soup(pg[i])
            for item in opinion_soup:
                links = opinion_soup.findAll('div',attrs={'class':'opinion_blog_header'})

            for page in links:
                l = page.findAll('a')
                for item in l:
                    page_link.append((item['href']))
    else:                                       # update all individual opinion links for authors with one page
        opinion_soup = b_soup(pg[0])
        for item in opinion_soup:
            links = opinion_soup.findAll('div',attrs={'class':'opinion_blog_header'})
        for page in links:
            l = page.findAll('a')
            for item in l:
                page_link.append((item['href']))                   # update all individual opinion links
        
    return page_link
authorurl = 'https://www.ndtv.com/opinion/authors'
author_soup = b_soup(authorurl)
all_author =author(author_soup)
author_df = pd.DataFrame.from_dict(all_author,orient='index',columns=['Author'])
author_df.reset_index(inplace=True)
author_df.columns = ['Author','Author_links']
author_df = author_df[author_df['Author'] != '']
author_df['total_opinions'] =""
for i in range(len(author_df)):
    author_opinion = author_df['Author_links'].iloc[i]
    pg = b_soup_opinion_pages(author_opinion)
    author_opinion_links = opinion_urls(pg)
    author_df['total_opinions'].iloc[i] = len(author_opinion_links)
author_df = author_df.sort_values(by='total_opinions', ascending=False).reset_index(drop=True)
author_df.head(10)
author_op = input('Enter the Author to find the Most used words ')
author_op_link= author_df.loc[author_df['Author'] == author_op, 'Author_links'].iloc[0]
pg = b_soup_opinion_pages(author_op_link)
author_opinion_links = opinion_urls(pg)
ndtv_author_opinion = ""
for i in range(len(author_opinion_links)):
    author_opinion_data = b_soup(author_opinion_links[i])
    for author_opinion_text in author_opinion_data.findAll('div',attrs={'itemprop':'articleBody'}):
        ndtv_author_opinion+= ". "+author_opinion_text.text
ndtv_author_opinion= ndtv_author_opinion.lower()
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
stop_words = stopwords.words('english')
stop_words[:5]
disclaimer_text = """Disclaimer: The opinions expressed within this article are the personal opinions of the author. NDTV is not responsible for the accuracy, completeness, suitability, or validity of any information on this article. All information is provided on an as-is basis. The information, facts or opinions appearing in the article do not reflect the views of NDTV and NDTV does not assume any responsibility or liability for the same"""
credits = """Dr Shashi Tharoor is a two-time MP from Thiruvananthapuram, the Chairman of the Parliamentary Standing Committee on External Affairs, the former Union Minister of State for External Affairs and Human Resource Development and the former UN Under-Secretary-General. He has written 16 books, including, most recently, 'An Era of Darkness: The British Empire in India"""
disclaimer_word = nltk.word_tokenize(disclaimer_text.lower())+nltk.word_tokenize(credits.lower())
stop_words = stop_words + disclaimer_word
blob = TextBlob(ndtv_author_opinion.lower())
items = blob.word_counts.items()
items = [item for item in items if item[0] not in stop_words]
from operator import itemgetter
sorted_items = sorted(items,key=itemgetter(1),reverse=True)
sorted_items[0:20]
from pathlib import Path
import imageio
from wordcloud import WordCloud
#wordcloud = WordCloud(width=2000,height=2000,colormap='prism',mask = mask_image, background_color = 'white')
#wordcloud = wordcloud.generate(ndtv_author_opinion)

clean_text = [word for word in ndtv_author_opinion.split() if word not in stop_words]

# Converting the list to string
text = ' '.join([str(elem) for elem in clean_text])

# Generating a wordcloud
wordcloud = WordCloud(background_color = "white").generate(text)

# Display the generated image:
plt.figure(figsize = (15, 10))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.title(f'Top words used by {author_op}',fontdict = {'fontsize' : 20})
plt.show()