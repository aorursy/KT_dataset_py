# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib.pyplot as plt; plt.rcdefaults()
import sklearn
import csv
%matplotlib inline 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
filename = check_output(["ls", "../input"]).decode("utf8").strip()


# Any results you write to the current directory are saved as output.
filename
def read_csv(filename):
    lines = []
    with open(filename, 'r') as t:
        spamreader = csv.reader(t, delimiter=',', quotechar='"')
        for row in spamreader:
            lines.append(row)
    t.close()
    return lines

all_reviews = read_csv('../input/br.csv')

print(all_reviews[0:3])
print(f"Number of reviews: {len(all_reviews)}")
negative_reviews = [row for row in all_reviews[1:] if float(row[3]) < 2.5]
neutral_reviews = [row for row in all_reviews[1:] if float(row[3]) >= 2.5 and float(row[3]) < 4.0]
positive_reviews = [row for row in all_reviews[1:] if float(row[3]) >= 4.0]

print(f"Negative reviews: {len(negative_reviews)}")
print(f"Neutral reviews: {len(neutral_reviews)}")
print(f"Positive reviews: {len(positive_reviews)}")

objects = ('Negative', 'Neutral', 'Positive')
y_pos = np.arange(len(objects))
performance = [len(negative_reviews),len(neutral_reviews),len(positive_reviews)]
 
plt.bar(y_pos, performance, align='center', alpha=0.25)
plt.xticks(y_pos, objects)
plt.ylabel('Number of reviews')
plt.title('Rating quality')
 
plt.show()

bookid_review_tuples = [(row[0], float(row[3]), row[8]) for row in all_reviews[1:]]
review_per_book = {}
for bookid,rating,review in bookid_review_tuples:
    try:
        new_entry = review_per_book[bookid]
        new_entry.append((rating,review))
        review_per_book[bookid] = new_entry
    except:
        review_per_book[bookid] = [(rating,review)]
print(f"Number of books found in dictionary: {len(review_per_book)}")
MIN_REVIEW_THRESHOLD = 1000
flop_books = [review for review in negative_reviews if int(review[4]) > MIN_REVIEW_THRESHOLD]
top_books = [review for review in positive_reviews if float(review[3]) > 4.6 and int(review[4]) > MIN_REVIEW_THRESHOLD]
print(f"Number of flop books found in dictionary: {len(flop_books)}")
print(f"Number of top books found in dictionary: {len(top_books)}")
for book in top_books[-10:]:
    print(f"BookID: {book[0]} - Title: {book[1]}. Written By: {book[2]}. Rating: {float(book[3])}")
print(review_per_book['330586'])
print(review_per_book['330587'])
print(review_per_book['330588'])