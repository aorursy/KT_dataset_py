# Setup



import os

import nltk

import csv

import html

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter

from pprint import PrettyPrinter

from scipy.stats import zscore

from datetime import datetime

from textblob import TextBlob

from textblob.np_extractors import ConllExtractor



%matplotlib inline



pp = PrettyPrinter(indent=4)



nltk.download('punkt')

nltk.download('conll2000')

nltk.download('stopwords')

nltk.download('wordnet')



pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)
# Shared utilities



class Utils():

    def __init__(self):

        return

    

    def prettyPercent(self, f: float):

        return "%.2f%%" % f



    def toDateTime(self, x: str):

        return datetime.strptime(x, '%d-%b-%y')

    

    def read_tsv(self):

        files = self.getFilePaths()

        tsv_file = open(files["amazon_alexa.tsv"])

        return csv.reader(tsv_file, delimiter="\t")



    def getFilePaths(self):

        files = {}

        for dirname, _, filenames in os.walk('/kaggle/input'):

            for filename in filenames:

                path = os.path.join(dirname, filename)

                files[filename] = path

        return files

    

utils = Utils()
class Ratings():

    def __init__(self):

        self.ratings = Counter()

        self.count()

        

    def count(self):

        for i, row in enumerate(utils.read_tsv()):

            if i is 0:

                continue

            rating = int(row[0])

            self.ratings.update({ rating: 1 })

            

    def get(self):

        return self.ratings

    

    def distribution(self, pretty=True):

        ratingsByPercentage = {}

        totalRatings = self.totalReviews()

        for stars in self.ratings:

            rating = int(stars)

            count = self.ratings[rating]

            percent = (count/totalRatings) * 100

            

            if pretty:

                percent = utils.prettyPercent(percent)

                

            ratingsByPercentage[stars] = percent

        return ratingsByPercentage



    def groupedDistribution(self, pretty=True):

        ratingsByPercentage = self.distribution(pretty=False)

        grouped = {

            'bad': 0,

            'good': 0,

            'great': 0,

        }

        

        for stars, percentage in ratingsByPercentage.items():

            if stars is 1 or stars is 2:

                grouped['bad'] += percentage

            elif stars is 3:

                grouped['good'] += percentage

            elif stars is 4 or stars is 5:

                grouped['great'] += percentage

        

        if pretty:

            groupedPretty = {}

            for group, percentage in grouped.items():

                groupedPretty[group] = utils.prettyPercent(percentage)

            return groupedPretty    

            

        return grouped

    

    def totalReviews(self):

        counts = self.ratings.values()

        return sum(counts)



    def average(self):

        collectedStars = 0

        

        for item in self.ratings.items():

            stars = item[0]

            count = item[1]

            collectedStars += stars * count

        

        

        total = self.totalReviews()

        avg = collectedStars / total

        return avg

            



class RatingsBarGraph():

    def __init__(self, ratings):

        self.ratings = ratings        



    def plot(self):

        x = []

        height = []

        for stars in self.ratings:

            count = self.ratings[stars]

            x.append(stars)

            height.append(count)

        return plt.bar(x=x, height=height, align='center')



        

ratings = Ratings()



barGraph = RatingsBarGraph(ratings.get())

barGraph.plot()



print('Ratings Distribution:', ratings.distribution())

print('Grouped Distribution:', ratings.groupedDistribution())

print('Average Rating:',ratings.average())
class RatingsOverTime():

    def __init__(self):

        self.calc()

    

    def calc(self):

        self.ratingsCountByDate = {}

        

        dates = set()

        for i, row in enumerate(utils.read_tsv()):

            if i is 0:

                continue

            rating = row[0]

            date = row[1]

            if date not in self.ratingsCountByDate.keys():

                self.ratingsCountByDate[date] = Counter()

            self.ratingsCountByDate[date].update({rating: 1})

            dates.add(date)



        self.dates = sorted(dates, key=utils.toDateTime)

                

    def getLines(self):

        ratingsLines = [

            [],

            [],

            [],

            [],

            []

        ]



        for date in self.dates:

            counter = self.ratingsCountByDate[date]

            for i in range(1, 6):

                index = str(i)

                rating = counter[index] or 0

                ratingsLines[i - 1].append(rating)

                

        return ratingsLines

                

    def plot(self):        

        lines = self.getLines()

        SIZE_FACTOR = 2

        FIG_SIZE = plt.gcf().get_size_inches()

        plt.gcf().set_size_inches(SIZE_FACTOR * FIG_SIZE)

        handles = []

        for i, line in enumerate(lines):

            rating = i + 1

            l, = plt.plot_date([utils.toDateTime(date) for date in self.dates], line, ls='-', label=str(rating) + ' star(s)')

            handles.append(l)

        plt.legend(handles)

            

        

ratingsOverTime = RatingsOverTime()

ratingsOverTime.plot()
class Variants():

    def __init__(self):

        self.ratingsByVariant = {}

        # only include variants that have more than 50 reviews.         

        self.minRatingsCutoff = 50

        self.starsRange = range(1, 6)

        self.calcVariantRatings()

    

    def calcVariantRatings(self):

        ratingsByVariant = self.ratingsByVariant

        

        for i, row in enumerate(utils.read_tsv()):

            if i is 0:

                continue

            rating = row[0]

            variant = row[2]

            

            if variant not in ratingsByVariant.keys():

                ratingsByVariant[variant] = {

                    "1": 0,

                    "2": 0,

                    "3": 0,

                    "4": 0,

                    "5": 0,

                }

                ratingsByVariant[variant][rating] += 1

            else:

                ratingsByVariant[variant][rating] += 1



    def table(self):

        table = {}

        ratingsByVariant = self.ratingsByVariant

        

        for variant in ratingsByVariant:

            ratings = ratingsByVariant[variant]

            row = []

            total = 0

            for stars in self.starsRange:

                count = 0

                key = str(stars)

                if key in ratings.keys():

                    count = ratings[key]

                row.append(count)

                total += count

            row.append(total)

            table[variant] = row



        df = pd.DataFrame(data=table, index=(list(self.starsRange) + ["total"]))

        dft = df.transpose()

        return dft

    

    def tablePercentages(self):

        tablePercentages = {}

        ratingsByVariant = self.ratingsByVariant

        

        for variant in ratingsByVariant:

            ratings = ratingsByVariant[variant]

            total = sum(ratings.values())



            if total < self.minRatingsCutoff:

                continue



            row = []

            for stars in self.starsRange:

                count = 0

                key = str(stars)

                if key in ratings.keys():

                    count = ratings[key]

                percent = (count/total) * 100

                row.append(utils.prettyPercent(percent))

            tablePercentages[variant] = row



        df = pd.DataFrame(data=tablePercentages, index=self.starsRange)

        dft = df.transpose()

        return dft

    

    def tableGroupedPercentage(self, pretty=True):

        tableGrouped = {}

        tableGroupedPercentages = {}

        ratingsByVariant = self.ratingsByVariant

        

        for variant in ratingsByVariant:

            ratings = ratingsByVariant[variant]

            total = sum(ratings.values())



            if total < self.minRatingsCutoff:

                continue



                # great, good, bad

            row = [0, 0, 0]

            rowPercentages = [0, 0, 0]

            for stars in self.starsRange:

                count = 0

                key = str(stars)

                if key in ratings.keys():

                    count = ratings[key]

                    if stars is 1 or stars is 2:

                        row[2] += count

                    elif stars is 3:

                        row[1] += count

                    elif stars is 4 or stars is 5:

                        row[0] += count



            for i, count in enumerate(row):

                percent = (count/total) * 100

                if pretty:

                    rowPercentages[i] = utils.prettyPercent(percent)

                else:

                    rowPercentages[i] = percent



            tableGrouped[variant] = row

            tableGroupedPercentages[variant] = rowPercentages



        df = pd.DataFrame(data=tableGroupedPercentages, index=['great', 'good', 'bad'])

        dft = df.transpose(copy = True)

        return dft

    

    def groupedZScore(self):

        df = self.tableGroupedPercentage(pretty=False)

        df = df.apply(zscore)

        

        def greatColor(val):

            if val > 2:

                return "color: green"

            return "color: black"

        

        def badColor(val):

            if val > 2:

                return "color: red"

            return "color: black"

        

        df = df.style.applymap(greatColor, subset=['great']).applymap(badColor, subset=['bad'])

        return df



v = Variants()

print('Average ratings:', ratings.groupedDistribution())

v.tableGroupedPercentage()
v.groupedZScore()
class Reviews():

    def __init__(self):

        self.extremeReviews = {}

        self.extractor = ConllExtractor()

        self.gatherExtremes()



    def gatherExtremes(self):

        for i, row in enumerate(utils.read_tsv()):

            if i is 0:

                continue

            rating = int(row[0])

            review = row[3]



            if rating is 1 or rating is 5:

                if rating not in self.extremeReviews.keys():

                    self.extremeReviews[rating] = []

                self.extremeReviews[rating].append(review)

                

    def wordsFromTopReviews(self):

        noun_phrases = Counter()

        for review in self.extremeReviews[5]:

            cleaned = html.unescape(review)

            cleaned.lower()

            blob = TextBlob(cleaned, np_extractor=self.extractor)

            phrases = blob.noun_phrases

            for phrase in phrases:

                noun_phrases.update({phrase: 1})

        return noun_phrases

    

    def wordsFromWorstReviews(self):

        noun_phrases = Counter()

        for review in self.extremeReviews[1]:

            cleaned = html.unescape(review)

            cleaned.lower()

            blob = TextBlob(cleaned, np_extractor=self.extractor)

            phrases = blob.noun_phrases

            for phrase in phrases:

                noun_phrases.update({phrase: 1})

        return noun_phrases

    

    def topPhrasesFromPositiveReviews(self, amount=20):

        phrases = self.wordsFromTopReviews()

        return phrases.most_common(amount)



    def topPhrasesFromNegativeReviews(self, amount=20):

        phrases = self.wordsFromWorstReviews()

        return phrases.most_common(amount)

    

reviews = Reviews()

print('Top 20 positive phrases:')

pp.pprint(reviews.topPhrasesFromPositiveReviews())



print('Top 20 negative phrases:')

pp.pprint(reviews.topPhrasesFromNegativeReviews())
