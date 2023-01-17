ls
import json



import xml



import xml.etree.cElementTree as cElementTree



import os

os.chdir('../input')



def process_xml():



    '''



    file_name = Name of XML file to process



    country_dict = object to write data to



    '''

    for event, elem in cElementTree.iterparse('china.xml'):



        if elem.tag == "title":

            for item in elem:

                print(item.tag,item.attrib)

            print(elem)

            elem.clear()  # It's crucial to clear the elem in order to free up memory

            

process_xml()
import xml.etree.ElementTree as ET

tree = ET.parse('sample.xml')

root = tree.getroot()
root
import xml.etree.ElementTree as ET

tree = ET.parse('sample.xml')

root = tree.getroot()
root
#root = ET.fromstring(country_data_as_string)
for child in root:

    print(child.tag,child.attrib)
import re

from nltk.corpus import stopwords

#with open('china.xml') as f:

with open('sport.xml') as f:

    f = f.read()

    these_regex="<title.*?>(.+?)</title>"

    pattern=re.compile(these_regex)

    china_titles=re.findall(pattern,f)

    china_titles.remove('South China Morning Post')

    china_titles.remove('South China Morning Post')

    

    china_words = []

    

for t in china_titles:

    t = t.split(" ")

    for w in t:

        if w not in stopwords.words('english'):

            china_words.append(w)

    print(t)

print(china_words)

    

#with open('us_canada.xml') as f:

with open('food.xml') as f:

    f = f.read()

    these_regex="<title.*?>(.+?)</title>"

    pattern=re.compile(these_regex)

    us_canada_titles=re.findall(pattern,f)

    us_canada_titles.remove('South China Morning Post')

    us_canada_titles.remove('South China Morning Post')

    

us_canada_words = []

for t in us_canada_titles:

    t = t.split(" ")

    for w in t:

        if w not in stopwords.words('english'):

            us_canada_words.append(w)

    print(t)





europe_words = []

#with open('europe.xml') as f:

with open('tech.xml') as f:

    f = f.read()

    these_regex="<title.*?>(.+?)</title>"

    pattern=re.compile(these_regex)

    europe_titles=re.findall(pattern,f)

    europe_titles.remove('South China Morning Post')

    europe_titles.remove('South China Morning Post')



for t in europe_titles:

    t = t.split(" ")

    for w in t:

        if w not in stopwords.words('english'):

            europe_words.append(w)

    print(t)

    

print(len(europe_titles+us_canada_titles+china_titles))

len(set(europe_titles+us_canada_titles+china_titles))
import nltk

nltk.download('stopwords')
print(len(set(us_canada_words+china_words+europe_words)))

unique_words = set(us_canada_words+china_words+europe_words)
feature_hashing = [[] for i in range(20)]

for word in unique_words:

    rem = hash(word) % 20

    feature_hashing[rem].append(word)
for i in feature_hashing:

    print(i)
def feature_rep(titles, size=100):

    feature_representations = []

    for title in titles:

        current_feature = [0 for i in range(size)]

        for word in title:

            rem = hash(word) % size

            current_feature[rem] += 1

        feature_representations.append(current_feature)

    return feature_representations



china_feature_representation = feature_rep(china_titles)

us_canada_feature_representation = feature_rep(us_canada_titles)

europe_feature_representation = feature_rep(europe_titles)
china_feature_representation = feature_rep(china_titles)

us_canada_feature_representation = feature_rep(us_canada_titles)

europe_feature_representation = feature_rep(europe_titles)
china_feature_representation

all_representations = china_feature_representation + us_canada_feature_representation + europe_feature_representation
print(len(china_feature_representation))

print(len(us_canada_feature_representation))

print(len(europe_feature_representation))
import numpy as np  

from sklearn.cluster import KMeans 
kmeans = KMeans(n_clusters=3)  

kmeans.fit(all_representations) 
kmeans.predict(china_feature_representation)
kmeans.predict(us_canada_feature_representation)
kmeans.predict(europe_feature_representation)
kmeans.labels_