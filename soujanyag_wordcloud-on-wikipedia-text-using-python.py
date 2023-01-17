import wikipediaapi

#Add below lines of code in case of SSL errors
import os
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['CURL_CA_BUNDLE'] = ""
os.environ['PYTHONWARNINGS']="ignore:Unverified HTTPS request"


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# set language
wiki_set_ln = wikipediaapi.Wikipedia('en')

# type any search word 
page = wiki_set_ln.page('Artificial Intelligence')
textdata =page.text
# define stop words and wordcloud 
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'white',
                      stopwords = stopwords,
                      min_font_size = 10,
                      max_font_size=80).generate(textdata)
# plot the diagram on wordcloud text
plt.figure(figsize=(100,20))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()