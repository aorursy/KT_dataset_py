# Make sure Internet is turned "on" under settings
!pip install -U newspaper3k
# We will use Beautiful Soup to pull out website URLs
from bs4 import BeautifulSoup  

# We will use Newspaper3k to get information from the article and our input url
from newspaper import Article  
url = 'https://www.google.com/search?q=black+lives+matter&source=lnms&tbm=nws&sa=X&ved=2ahUKEwi7wK78tJnrAhUDvFkKHYu7AcsQ_AUoAXoECBkQAw&biw=1016&bih=559&dpr=2'

article = Article(url)  # Now we use the newspaper library to pull out the article
article.download()
html_page = article.html
print(html_page)
soup = BeautifulSoup(html_page, features="lxml")
sources_file = open('sources.txt', 'w')
for link in soup.findAll('a'):
    # We find all the links on the page using the beautiful soup library
    current_link = link.get('href')  
    print(current_link)
    sources_file.write(current_link + '\n')
sources_file.close()
try:
    sources_file = open('sources.txt', 'r')
except:
    print('Please make a file called sources.txt and populate it with article urls.')
contents = sources_file.read()
sources_file.close()  # remember to close our file again
all_urls_set = set(contents.split("\n"))  # The split command generates a list
substring_to_remove_from_beginning = "/url?q="
substring_to_remove_from_end = "&sa"  # And everything that comes after is also junk

cleaned_urls_list = [] # We will store our cleaned URLs here
for url in all_urls_set:
    # First let's check for the substring "/url?q=" which we can use to identify relevant urls
    if substring_to_remove_from_beginning in url:
        # We want to remove the substring at the beginning and replace it with an empty string
        cleaned_url = url.replace(substring_to_remove_from_beginning, "") 
        
        # We want to pull everything before "&qa" so we split on "&qa" and take the first part
        cleaned_url_split = cleaned_url.split(substring_to_remove_from_end)
        print(cleaned_url_split)
        cleaned_url = cleaned_url_split[0]
        print(cleaned_url)
        
        cleaned_urls_list.append(cleaned_url)
# Solution: Join all the urls on the newline character ("\n)
all_cleaned_urls = "\n".join(cleaned_urls_list)  # This is a string
cleaned_urls_file = open('cleaned_sources.txt', 'w')
cleaned_urls_file.write(all_cleaned_urls)
cleaned_urls_file.close()
all_keywords = []

for news_article_url in cleaned_urls_list:
    try:
        # Let's get some basics from the article after we cleaned it
        print(news_article_url)
        current_article = Article(news_article_url)
        current_article.download()
        current_article.parse()
        print(current_article.title)
        print(current_article.authors)
        print(current_article.text)

        # On your own: What line do I need to add to get the publish date of the article?
        
#     except:
#         print("No information extracted from the following url " + news_article_url)
        
        
        
        
        
        
        
        
        
        # Solution:
        print(current_article.publish_date)

        # Now let's get use Newspaper3k to get some fancier information from the article
        current_article.nlp()

        # This is where it gets interesting when we pull out the keywords from each article
        current_keywords = current_article.keywords
        print(current_keywords)
        all_keywords.extend(current_keywords)

        # Let's see what the discussion is about and compare
        # From here we can save keywords to a CSV file and make graphs in Excel or a simple word cloud

    except:
        print("No information extracted from the following url " + news_article_url)

from collections import Counter
dictionary_of_keyword_counts = dict(Counter(all_keywords))
print(dictionary_of_keyword_counts)
import csv
keywords_csv = open('keywords.csv', 'w')
for key in dictionary_of_keyword_counts.keys():
    keywords_csv.write("%s,%s\n"%(key, dictionary_of_keyword_counts[key]))

keywords_csv.close()
import csv
keywords_csv_more_words = open('keywords_more_words.csv', 'w')
for key in dictionary_of_keyword_counts.keys():
    # The code in this if-statement will only execute if we have more than 3 of the same keywords
    if dictionary_of_keyword_counts[key] > 3:
        keywords_csv_more_words.write("%s,%s\n"%(key, dictionary_of_keyword_counts[key]))
        
keywords_csv_more_words.close()