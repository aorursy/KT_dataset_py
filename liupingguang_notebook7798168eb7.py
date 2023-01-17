import re

import csv

import math

from collections import defaultdict, OrderedDict

from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))





def clean_html(raw_html):

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, '', raw_html)

    return cleantext





def get_words(text):

    word_split = re.compile('[^a-zA-Z0-9_\\+\\-/]')

    return [word.strip().lower() for word in word_split.split(text)]





def process_text(doc, idf, text):

    tf = OrderedDict()

    word_count = 0.



    for word in get_words(text):

        if word not in stop_words and word.isalpha():



            if word not in tf:

                tf[word] = 0

            tf[word] += 1

            idf[word].add(doc)

            word_count += 1.



    for word in tf:

        tf[word] = tf[word] / word_count



    return tf, word_count





def main():

    data_path = "../input/"

    in_file = open(data_path + "test.csv")

    out_file = open("tf_idf.csv", "w")



    reader = csv.DictReader(in_file)

    writer = csv.writer(out_file)

    writer.writerow(['id', 'tags'])



    docs = []



    # Calculate TF and IDF per document

    idf = defaultdict(set)

    tf = {}

    word_counts = defaultdict(float)



    print("Counting words..")

    for row in reader:

        doc = int(row['id'])

        docs.append(doc)



        text = clean_html(row["title"]) + ' ' + clean_html(row["content"])

        tf[doc], word_counts[doc] = process_text(doc, idf, text)



    in_file.close()



    # Calculate TF-IDF

    nr_docs = len(docs)

    for doc in docs:



        for word in tf[doc]:

            tf[doc][word] *= math.log(nr_docs / len(idf[word]))



    # Write predictions

    print("Writing predictions..")

    for doc in docs:



        # Sort words with frequency from high to low.

        pred_tags = sorted(tf[doc], key=tf[doc].get, reverse=True)[:3]



        # Write predictions

        writer.writerow([doc, " ".join(sorted(pred_tags))])



    in_file.close()

    out_file.close()





if __name__ == "__main__":

    print("Starting program.")

    main()
