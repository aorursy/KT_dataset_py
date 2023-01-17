# First we need to make sure our internet is on so we can install the following package
# Go under "Settings" on the right side and toggle "Internet" on. You will need to authenicate with your phone. 
!pip install -U newspaper3k
# We will use Beautiful Soup to pull out website URLs
from bs4 import BeautifulSoup  

# We will use Newspaper3k to get information from the article and our input url
from newspaper import Article  

# We will use collections to count how many times a keywords appear
from collections import Counter
# Paste url here
url = ''
# Let's get the html from the page using newspaper3k

# Now we will use Beautiful Soup

# We create our file sources.txt here

# We find all the links on the page using the beautiful soup library and a for-loop

# Close sources.txt here

# Read from sources.txt file

# Create cleaned_urls_list
cleaned_urls_list = [] # We will store our cleaned URLs here

all_urls_set = set(contents.split('\n'))  # The split command generates a list

# FILL IN THESE STRINGS
substring_to_remove_from_beginning = '' 
substring_to_remove_from_end = ''  # And everything that comes after is also junk

# Now let's iterate through the urls with a for-loop

# String of all urls to save to a new file
all_cleaned_urls = "\n".join(cleaned_urls_list)  # This is a string
# Save the string all_cleaned_urls to a file called cleaned_sources.txt

# All of our keywords will be stored in a list called all_keywords

# Count the keywords
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
    # Write an if-statement will only execute if we have more than 3 of the same keywords
    
    keywords_csv_more_words.write("%s,%s\n"%(key, dictionary_of_keyword_counts[key]))
keywords_csv_more_words.close()