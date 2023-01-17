#importing all dependencies.

import os

import json

import pandas as pd

from tqdm import tqdm

import numpy as np

from nltk.corpus import wordnet

import re

import matplotlib.pyplot as plt

from nltk.corpus import stopwords 

#from geotext import GeoText

#/kaggle/input
#lets say we are searching for 'transmission'

synonyms=["spread","transmit","transmission","transmitted"]

for syn in wordnet.synsets("transmission"):

    for name in syn.lemma_names():

        if name not in synonyms:

        	synonyms.append(name)



for syn in wordnet.synsets("spread"):

    for name in syn.lemma_names():

        if name not in synonyms:

        	synonyms.append(name)



#print(synonyms)
#load a file

dirs_=["/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv",

"/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset",

"/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset",

"/kaggle/input/CORD-19-research-challenge/custom_license/custom_license"]



data_=list()

for dir_ in dirs_:

	for filename in tqdm(os.listdir(dir_)):



		x=str(dir_)+'/'+str(filename)

        

		with open(x) as file:

			data=json.loads(file.read())

		

		#take out the data from the json format

		paper_id=data['paper_id']

		meta_data=data['metadata']

		abstract=data['abstract']

		abstract_text=""

		for text in abstract:

			abstract_text+=text['text']+" "

		body_text=data['body_text']

		full_text=""

		for text in body_text:

			full_text+=text['text']+" "

		back_matter=data['back_matter']

		#store everything to a dataframe

		data_.append([paper_id,abstract_text,full_text])



df=pd.DataFrame(data_,columns=['paper_id','abstract','full_text'])

print(df.head())

#save as a csv

#df.to_csv('biorxiv_medrxiv.csv', index = True)

#a data frame for my complete body.
synonyms_=["coronavirus","Covid-19","COVID 19","Coronavirus","Corona virus","COVID 19"]

#transmission and its synonyms related body are taken for consideration

transmission_synonym=df[df['full_text'].str.contains('|'.join(synonyms_))]

keys_=["transmitted","transmits"]

my_data=list()

#looping through each paper

for texts in transmission_synonym['full_text']:



	#looping through each sentence

	for tex in texts.split(". "):



		#checking if my keyswords is present in those sentences or not

		for wor_ in keys_:

			if wor_ in tex:

				#print(tex)

				#print()

				#print()

				my_data.append(tex)



###############################################################################################

from stop_words import get_stop_words

#print(len(my_data))

#create a word bucket

#remove stop words and punctions

punctuations='''!()[,]{\};:'"<>./?@#$%^&*_~'''

stopwords=get_stop_words('english')

#print(stopwords)

word_bucket=dict()

string=""

#loop through lines

for lines in my_data:

	#loop through words of each line

	for words in lines.split(" "):



		#for all non-alphanumeric words

		if words.isalpha() == False:



			#remove punctions

			for chrs in words:

				if chrs in punctuations:

					words=words.replace(chrs,"")



			#convert to lower case

			words=words.lower()



			#removing names of citiesfrom geotext import GeoText

			#places = GeoText(words)

			#city=places.cities

			#country=places.countries and (len(city)+len(country))==0



			#remove the stopwords

			if words not in stopwords and len(re.findall("\d|transmi|also",words))==0 and len(words)>3:



				lexemes=re.findall(words,string)

				#print(lexemes)



				if words not in word_bucket.keys() and len(lexemes)==0:

					#add new elemnts

					word_bucket[words]=1

					string=string+" "+words

				else:

					#freq of occurances

					try:

						word_bucket[words]=word_bucket[words]+1

					except:

						pass







analytics=dict()

for keys in word_bucket:

	#print(keys)

	if int(word_bucket[keys])>4:

		analytics[keys]=word_bucket[keys]

		#print(keys)



#print(analytics)

############################################################################################################

common_words=["population","unknown","illness","sars-cov","fecal-oral","mers-cov","family","contact","coronavirus","droplets","zoonotic","note",

"faecal-oral","route","countries","influenza","coronaviruses","animals","human-to-human","food","bats","hosts","humans","fever","aerosols","patients",

"fluids","infections","arthropod-borne","chikv","flaviviridae","ticks","denv","tick-borne","pathogens","vector-borne","bites","person-to-person",

"ferrets","zikv","water","malaria","dengue","individuals","camels","bat-borne","dogs","swine","stis","arthropods","flaviviruses","mosquitoes",

"transfusion","within-host","rodents","schistosomiasis","healthcare-associated","workers","henipaviruses","horses","saliva","pigs","fomites","agents",

"persons","genus"]

#bar graphs

freq=[]

cw=[]

for w_ in common_words:

    if w_ in analytics.keys():

        cw.append(w_)

for w_ in cw:

    if w_ in analytics.keys():

        freq.append(analytics[w_])

    else:

        common_words.remove(w_)



plt.barh(cw[0:20],freq[0:20])

plt.xlabel("Frequency")

plt.ylabel("Factors of transmission")

plt.title("Probable causes of Transmission")

plt.show()
plt.barh(cw[20:40],freq[20:40])

plt.xlabel("Frequency")

plt.ylabel("Factors of transmission")

plt.title("Probable causes of Transmission")

plt.show()
plt.barh(cw[39:61],freq[39:61])

plt.xlabel("Frequency")

plt.ylabel("Factors of transmission")

plt.title("Probable causes of Transmission")

plt.show()

#COVID 19 incubation period

synonyms=list()

for syn in wordnet.synsets("incubation"):

    for name in syn.lemma_names():

        if name not in synonyms:

        	synonyms.append(name)

incubation_synonym=df[df['full_text'].str.contains('|'.join(synonyms))]



#incubation.to_csv('biorxiv_medrxiv_incubation.csv', index = True)

#incubation_synonym.to_csv('biorxiv_medrxiv_incubation_synonyms.csv', index = True)



all_details=list()

texts=incubation_synonym['full_text']

for tex in texts:

	#print(tex)

	for t in tex.split(". "):

		if 'incubation' in t:

			regex=re.findall("\d{1,2} days| \d{1,2}-\d{1,2} days| \d{1,2} to \d{1,2} days| \d+\.\d+ days|\d+\.\d+-\d+\.\d+ days|\d+\.\d+ to \d+\.\d+ days",t)

			if len(regex)!=0:

				#print(t)

				#print(regex)

				#print('\n')

				all_details.append(regex)



#extracting incubation periods from data

incubation_period=list()

for samples in all_details:

	for sam in samples:

		for data in sam.split(" "):

			try:

				incubation_period.append(float(data))

			except:

				#ingore the days n all

				pass

#incubation_period=np.array(incubation_period)

#print(incubation_period)

dataf=pd.DataFrame(incubation_period)

print(dataf.describe())



#datadescription and plots

hist=dataf.hist(bins=50)

plt.ylabel('Frequency as per research paper')

plt.xlabel('No of days')

plt.title('COVID 19 incubation period')

plt.show()
#COVID 19 infected age groups

wards=["years","infected","age","heath"]

body_=df[df['full_text'].str.contains('|'.join(wards))]



all_details=list()

for bod in body_['full_text']:

	for b in bod.split(". "):

		if 'age' in b or 'infected' in b or 'health' in b or 'years' in b:

			regex=re.findall(" \d{1,2} years| \d{1,2}-\d{1,2} years| \d{1,2} to \d{1,2} years| \d+\.\d+ years| \d+\.\d+-\d+\.\d+ years| \d+\.\d+ to \d+\.\d+ years",b)

			if len(regex)!=0:

				#print(b)

				#print(regex)

				#print("\n")

				all_details.append(regex)



#extracting age groups from data

age_group=list()

for samples in all_details:

	for sam in samples:

		for data in sam.split(" "):

			try:

				age_group.append(float(data))

			except:

				#ingore the days n all

				pass



#print(age_group)

#datadescription and plots

for x in age_group:

	if x>=0 and x<=100:

		pass

	else:

		age_group.remove(x)

dataf=pd.DataFrame(age_group)

print(dataf.describe())

plt.hist(age_group,bins=50)

plt.ylabel('Frequency as per research paper')

plt.xlabel('Age of Human in years')

plt.title('COVID 19 infected age groups')

plt.show()