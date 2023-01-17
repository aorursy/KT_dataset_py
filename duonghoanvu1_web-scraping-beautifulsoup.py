# Data Processing
import pandas as pd

# Data Scraping
import requests
from bs4 import BeautifulSoup
# Define the URL of the site
base_site = "https://editorial.rottentomatoes.com/guide/140-essential-action-movies-to-watch-now/2/"
# sending a request to the webpage
response = requests.get(base_site)
response.status_code
# get the HTML from the webpage
html = response.content
# convert the HTML to a Beautiful Soup object
soup = BeautifulSoup(html, 'html.parser')

# Exporting the HTML to a file
with open('Rotten_tomatoes_page_2_HTML_Parser.html', 'wb') as file:
    file.write(soup.prettify('utf-8'))
# convert the HTML to a BeatifulSoup object
soup = BeautifulSoup(html, 'lxml')

# Exporting the HTML to a file
with open('Rotten_tomatoes_page_2_LXML_Parser.html', 'wb') as file:
    file.write(soup.prettify('utf-8'))
# Find all div tags on the webpage containing the information we want to scrape
divs = soup.find_all("div", {"class": "col-sm-18 col-full-xs countdown-item-content"})
divs[:2]
# for instance, let's explore the first div
divs[0].find("h2")
# Extracting all 'h2' tags
headings = [div.find("h2") for div in divs]
headings[:5]
# Inspecting the text inside the headings
[heading.text for heading in headings][:5]
headings[0]
# Let's check all heading links
[heading.find('a') for heading in headings][:5]
# Obtaining the movie titles from the links
movie_names = [heading.find('a').string for heading in headings]
movie_names[:5]
# Filtering only the spans containing the year
[heading.find("span", class_ = 'start-year') for heading in headings] [:5]
# Extracting the year string
years = [heading.find("span", class_ = 'start-year').string for heading in headings]
years [:5]
years[0]
years[0][1:-1]
# Removing '('
print(years[0].strip('('))

# Removing ')'
print(years[0].strip(')'))

# Combining both
print(years[0].strip('()'))
# Updating years with stripped values
years = [year.strip('()') for year in years]
years [:5]
# Converting all the strings to integers
years = [int(year) for year in years]
years [:5]
# Filtering only the spans containing the score
[heading.find("span", class_ = 'tMeterScore') for heading in headings] [:5]
# Extracting the score string
scores = [heading.find("span", class_ = 'tMeterScore').string for heading in headings]
scores [:5]
# Removing the '%' sign
scores = [s.strip('%') for s in scores]
scores [:5]
# Converting each score to an integer
scores = [int(s) for s in scores]
scores [:5]
# The critics consensus is located inside a 'div' tag with the class 'info critics-consensus'
# This can be found inside the original 'div's we scraped
divs [:1]
# Getting the 'div' tags containing the critics consensus
consensus = [div.find("div", {"class": "info critics-consensus"}) for div in divs]
consensus [:5]
# Inspecting the text inside these tags
[con.text for con in consensus] [:5]
# The simplest (but not necessarily the best) way of achieving it is by taking the substring after the common phrase

# Defining the phrase to be removed (note the space at the end)
common_phrase = 'Critics Consensus: '

# Finding how long is the common phrase
len(common_phrase)
consensus[0].text
# Taking only the part of the text after the common phrase
consensus[0].text[19:]
# Define a variable to store the length
common_len = len(common_phrase)

# Cleaning the list of the common phrase
consensus_text = [con.text[common_len:] for con in consensus]
consensus_text [:5]
# We can add if-else logic to only truncate the string in case it starts with the common phrase
consensus_text = [con.text[common_len:] if con.text.startswith(common_phrase) else con.text for con in consensus ]
consensus_text [:5]
consensus[0]
# We can use .contents to obtain a list of all children of the tag
consensus[0].contents
# The second element of that list is the text we want
consensus[0].contents[1]
# We can remove the extra whitespace (space at the beginning) with the .strip() method
consensus[0].contents[1].strip()
# Processing all texts
consensus_text = [con.contents[1].strip() for con in consensus]
consensus_text [:5]
# Extracting all director divs
directors = [div.find("div", class_ = 'director') for div in divs]
directors [:5]
# Inspecting a div
directors[0]
# The director's name can be found as the string of a link

# Obtaining all director links
[director.find("a") for director in directors] [40:45]
# We can use if-else to deal with the None value

final_directors = [None if director.find("a") is None else director.find("a").string for director in directors]
final_directors [:5]
cast_info = [div.find("div", class_ = 'cast') for div in divs]
cast_info [:5]
cast_info[0]
# Let's first practice with a single movie

# Obtain all the links to different cast members
cast_links = cast_info[0].find_all('a')
cast_links
cast_names = [link.string for link in cast_links]
cast_names
# OPTIONALLY: We can stitch all names together as one string

# This can be done using the join method
# To use join, pick a string to use as a separator (in our case a comma, followed with a space) and
# pass the list of strings you want to merge to the join method

cast = ", ".join(cast_names)
cast
# Initialize the list of all cast memners
cast = []

# Just put all previous operations inside a for loop
for c in cast_info:
    cast_links = c.find_all('a')
    cast_names = [link.string for link in cast_links]
    
    cast.append(", ".join(cast_names)) # Joining is optional

cast [:5]
# As you can see this can be done in just one line using nested list comprehension
# However, the code is harded to understand

cast = [", ".join([link.string for link in c.find_all("a")]) for c in cast_info]
cast [:5]
# The adjusted scores can be found in a div with class 'info countdown-adjusted-score'
adj_scores = [div.find("div", {"class": "info countdown-adjusted-score"}) for div in divs]
adj_scores [:5]
# Inspecting an element
adj_scores[0]
# By inspection we see that the string we are looking for is the second child of the 'div' tag
adj_scores[0].contents[1]  # Note the extra whitespace at the end
# Extracting the string (without '%' sign and extra space)
adj_scores_clean = [score.contents[1].strip('% ') for score in adj_scores]
adj_scores_clean [:5]
# Converting the strings to numbers
final_adj = [float(score) for score in adj_scores_clean] # Note that this time the scores are float, not int!
final_adj [:5]
# The synopsis is located inside a 'div' tag with the class 'info synopsis'
synopsis = [div.find('div', class_='synopsis') for div in divs]
synopsis [:5]
# Inspecting the element
synopsis[0]
# The text is the second child
synopsis[0].contents[1]
# Extracting the text
synopsis_text = [syn.contents[1] for syn in synopsis]
synopsis_text [:5]
# A dataframe is a tabular data type, frequently used in data science

movies_info = pd.DataFrame()
movies_info  # The dataframe is still empty, we need to fill it with the info we gathered
movies_info["Movie Title"] = movie_names
movies_info["Year"] = years
movies_info["Score"] = scores
movies_info["Adjusted Score"] = final_adj
movies_info["Director"] = final_directors
movies_info["Synopsis"] = synopsis_text
movies_info["Cast"] = cast
movies_info["Consensus"] = consensus_text
movies_info.head()
# By default pandas abbreviates any text beyond a certain length (as seen in the Cast and Consensus columns)

# We can change that by setting the maximum column width to -1,
# which means the column would be as wide as to display the whole text
pd.set_option('display.max_colwidth', -1)
movies_info.head()
# Write data to CSV file
movies_info.to_csv("movies_info.csv", index = False, header = True)

# Write data to excel file
#movies_info.to_excel("movies_info.xlsx", index = False, header = True)