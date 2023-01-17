

'''

Script:  thesaurusGet.py 

Purpose:  Gets synonyms using the BIG HUGE THESAURUS at http://words.bighugelabs.co



API response format sample:

examination <Response [200]>

{'noun': {'syn': ['scrutiny', 'exam', 'test', 'interrogation', 'interrogatory', 'examen', 'testing', 'communicating', 'communication', 'inquiring', 'introspection', 'investigating', 'investigation', 'questioning', 'self-contemplation', 'self-examination']}}

'''



import requests

import json



#### add a list of words here.  Remember to follow the Big Huge Thesaurus guide and don't submit too many words

longWords = ['examination']

            

fs = open('synonyms.txt', 'w')



#### get synonyms from Big Huge Thesaurus - add your API key

for longWord in longWords:

    url = 'http://words.bighugelabs.com/api/YOUR API KEY HERE/' + longWord + '/json'

    if "YOUR API KEY" in url:

        print("Get your API KEY at http://words.bighugelabs.com")

    else:

        r = requests.get(url)

        print(longWord, r)

        try:

            synonyms = json.loads(r.content.decode())

        except:

            continue

        print(longWord,",",synonyms, file=fs)

fs.close()

'''

Script:  thesaurusProcessV2.py



Purpose: to find the shortest two synonyms for a list of long words



Format of input file called synonyms.txt:



Long Word from

Job Bulletin     List of synonyms from Big Huge Thesaurus



examination , {'noun': {'syn': ['scrutiny', 'exam', 'test', 'interrogation', 'interrogatory', 'examen', 'testing', 'communicating', 'communication', 'inquiring', 'introspection', 'investigating', 'investigation', 'questioning', 'self-contemplation', 'self-examination']}}

'''



import re

import json



f = open('../input/jobbulletindata/JBR_Output/synonyms.txt', 'r')           # file of synonyms downloaded from Big Huge Thesaurus using their API

fo = open('wordRecommend.txt', 'w')     # output file with recommendations for simpler words.  used to process native_words

for line in f:

    shorterWords = []                   # stores simpler words

    recommend = []                      # stores recommendations



    line = line.strip("\n\r")           # get rid of carriage return and line break

    if line[0:1] != '{':                # if line contains a word from the Job Bulletins then strip the comma and store the word

        longWord = line.strip(',')

    elif "<" not in line:               # if line contains Big Huge Thesaurus synonyms then find the shortest synonym

        line = re.sub("'", '"', line)   # change the formatting to expected json format



        try:

            if "adjective" in line:         # find the Big Huge Thesaurus recommendation for the simplest adjective

                synonyms = json.loads(line)

                if "sim" in line:

                    for synonym in synonyms['adjective']['sim']:

                        shorterWords.append((len(synonym), synonym))

                elif "syn" in line:

                    for synonym in synonyms['adjective']['syn']:

                        shorterWords.append((len(synonym), synonym))

                elif "rel" in line:

                    for synonym in synonyms['adjective']['rel']:

                        shorterWords.append((len(synonym), synonym))



            elif "noun" in line:

                synonyms = json.loads(line)

                for synonym in synonyms['noun']['syn']:

                    shorterWords.append((len(synonym), synonym))

            

            elif "adverb" in line:

                synonyms = json.loads(line)

                if 'syn' in line:

                    for synonym in synonyms['adverb']['syn']:

                        shorterWords.append((len(synonym), synonym))

                elif 'ant' in line:

                    shorterWords.append([0,'none'])

            

            elif "verb" in line:

                synonyms = json.loads(line)

                for synonym in synonyms['verb']['syn']:

                    shorterWords.append((len(synonym), synonym))

        except:

            print("EXCEPTION: ", line)

            continue



        shorterWords.sort()

        recommend.append(longWord)      # add the Job Bulletin term to wordRecommend.txt file

        if len(shorterWords) == 1:      # use the Big Huge Thesaurus recommendation for the most common term

            recommend.append(shorterWords[0][1])

            recommend.append('none')

        else:                           # if no recommendation then find the shortest two words

            for x in range(2):

                if int(shorterWords[x][0]) < len(longWord): 

                    recommend.append(shorterWords[x][1])

                else:

                    recommend.append('none')    # if all the synonyms are longer than the Job Bulletin term, use the Job Bulletin term

        print(recommend, file=fo)

        print("\nRECOMMEND ", recommend)



fo.close()
