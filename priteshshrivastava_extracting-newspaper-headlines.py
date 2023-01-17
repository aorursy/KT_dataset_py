! pip3 install newspaper3k
import json

import newspaper

from newspaper import Article

from time import mktime

from datetime import datetime

import csv

import os



os.listdir("../input/")
# Set the limit for number of articles to download

LIMIT = 50



data = {}

data['newspapers'] = {}



# Loads the JSON files with news sites

with open('../input/newspapers.json') as data_file:

    companies = json.load(data_file)

    

count = 1



csv_articles  = []
# Iterate through each news company

for company, value in companies.items():

    # It uses the python newspaper library to extract articles

    print("Building site for ", company)

    paper = newspaper.build(value['link'], memoize_articles=False)

    newsPaper = {

           "link": value['link'],

           "articles": []

        }

    noneTypeCount = 0

    for content in paper.articles:

        if count > LIMIT:

            break

        try:

            content.download()

            content.parse()

        except Exception as e:

            print(e)

            print("continuing...")

            continue

            # Again, for consistency, if there is no found publish date the article will be skipped.

            # After 10 downloaded articles from the same newspaper without publish date, the company will be skipped.

        if content.publish_date is None:

            print(count, " Article has date of type None...")

            noneTypeCount = noneTypeCount + 1

            if noneTypeCount > 10:

                print("Too many noneType dates, aborting...")

                noneTypeCount = 0

                break

            count = count + 1

            continue

        article = {}

        article['title'] = content.title

        article['text'] = content.text

        article['link'] = content.url

        try:

            article['published'] = content.publish_date.isoformat()

        except:

            print("bad date")



        csv_article  = {}

        csv_article['headline'] = content.title.encode('utf-8')

        try:

            csv_article['date'] = content.publish_date.isoformat()

        except:

            print("bad date")



        csv_article['link'] = content.url



        csv_articles.append(csv_article)





        newsPaper['articles'].append(article)

        print(count, "articles downloaded from", company, " using newspaper, url: ", content.url)

        count = count + 1

        noneTypeCount = 0

    count = 1

    data['newspapers'][company] = newsPaper
# Finally, save the articles as a JSON-file

try:

    with open('scraped_articles.json', 'w') as outfile:

        json.dump(data, outfile)

except Exception as e: print(e)



OUTFILE = "headlines.csv"

with open(OUTFILE, 'w') as output_file:

    keys = ['headline', 'date', 'link']

    dict_writer = csv.DictWriter(output_file, keys)

    dict_writer.writeheader()

    for row in csv_articles:

        dict_writer.writerow(row)