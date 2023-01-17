import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from nltk.tokenize import word_tokenize

from nltk.probability import FreqDist

from nltk.corpus import stopwords

import nltk

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

print(os.listdir("../input"))

pd.options.mode.chained_assignment = None
dataset1=pd.read_csv(r"../input/GBvideos.csv")

dataset1=dataset1.sort_values(by=['title','views']).reset_index(drop=True)
dataset1.category_id[dataset1.category_id==1]="Film & Animation"

dataset1.category_id[dataset1.category_id==2]="Autos & Vehicles"

dataset1.category_id[dataset1.category_id==10]="Music"

dataset1.category_id[dataset1.category_id==15]="Pets & Animals"

dataset1.category_id[dataset1.category_id==17]="Sports"

dataset1.category_id[dataset1.category_id==18]="Short Movies"

dataset1.category_id[dataset1.category_id==19]="Travel & Events"

dataset1.category_id[dataset1.category_id==20]="Gaming"

dataset1.category_id[dataset1.category_id==21]="Videoblogging"

dataset1.category_id[dataset1.category_id==22]="People & Blogs"

dataset1.category_id[dataset1.category_id==23]="Comedy"

dataset1.category_id[dataset1.category_id==24]="Entertainment"

dataset1.category_id[dataset1.category_id==25]="News & Politics"

dataset1.category_id[dataset1.category_id==26]="Howto & Style"

dataset1.category_id[dataset1.category_id==27]="Education"

dataset1.category_id[dataset1.category_id==28]="Science & Technology"

dataset1.category_id[dataset1.category_id==30]="Movies"

dataset1.category_id[dataset1.category_id==31]="Anime/Animation"

dataset1.category_id[dataset1.category_id==32]="Action/Adventure"

dataset1.category_id[dataset1.category_id==33]="Classics"

dataset1.category_id[dataset1.category_id==34]="Comedy"

dataset1.category_id[dataset1.category_id==35]="Documentary"

dataset1.category_id[dataset1.category_id==36]="Drama"

dataset1.category_id[dataset1.category_id==37]="Family"

dataset1.category_id[dataset1.category_id==38]="Foreign"

dataset1.category_id[dataset1.category_id==39]="Horror"

dataset1.category_id[dataset1.category_id==40]="Sci-Fi/Fantasy"

dataset1.category_id[dataset1.category_id==41]="Thriller"

dataset1.category_id[dataset1.category_id==42]="Shorts"

dataset1.category_id[dataset1.category_id==43]="Shows"

dataset1.category_id[dataset1.category_id==44]="Trailers"

dataset1.category_id[dataset1.category_id==29]='Non-profits & Activism'
Q1 = ['WDT','WRB','WP','WRB'] # wh- phrases



dataset1.title = dataset1.title.str.lower()

dataset1['IsTitleAQuestion'] = dataset1.title.map(lambda x: 1 if any([True if word in ''.join(np.array(nltk.pos_tag(word_tokenize(x))).reshape(-1)[1::2]) else False for word in Q1])==True or '?' in x else 0)

# tokenizing titles, getting their parts of speech and indicating that if it contains parts of speech from Q1 or if it has question mark it should be marked as question

dataset1.IsTitleAQuestion[dataset1.category_id=='Music'] = 0 # I assume that music titles don't stimulate curiosity
Eng_personal = ['i', 'he','she']

Personal_classification = dataset1[['title','category_id']]

PoS = ['PRPVB','PRPVBD','PRPVBG','PRPVBN','PRPVBZ']



# Detecting parts of speech from PoS

Personal_classification['PartsOfSpeech'] = Personal_classification.title.map(lambda x: 1 if any([True if word in ''.join(

                                                                                    np.array(nltk.pos_tag(word_tokenize(x.lower()))).reshape(-1)[1::2])

                                                                                    else False for word in PoS])==True else 0) 



# Checks whether title contains question marks or equivalent

Personal_classification['QuotationMarks'] = Personal_classification.title.map(lambda x: 1 if sum([1 if '\'' in title or '"' in title else 0 for title in x])>=2 or

                                                                  sum([1 if ':' in title or '|' in title else 0 for title in x])>=1 else 0) 



# Detecting phrases 'official video' and 'official music'

Personal_classification['Officialness'] = Personal_classification.title.map(lambda x: 1 if 'official video' in x.lower() or 'official music' in x.lower() else 0)



# Classifies as having personal aspect if title contains words from Eng_personal list and doesn't belong to 'Music' category

PersonalAspect = []

for i, s in enumerate(dataset1.title.map(lambda x: x.replace('"','').replace('\'','')),start=0):

    if any(word in word_tokenize(s.lower()) for word in Eng_personal) and dataset1.category_id.iloc[i]!='Music':

        PersonalAspect.append(1)

    else:

        PersonalAspect.append(0)

Personal_classification['PersonalAspect'] = pd.Series(PersonalAspect)







data = Personal_classification[['PartsOfSpeech','QuotationMarks','category_id','title']][(Personal_classification.PersonalAspect==1) & (Personal_classification.Officialness==0)]



# Standardizing variables

training_data = pd.get_dummies(data[['PartsOfSpeech','QuotationMarks','category_id']])

scaler = StandardScaler()

scaler.fit(training_data)

training_data = scaler.transform(training_data)



data1 = pd.DataFrame(data['title'])

model = KMeans(n_clusters=2,n_init=100).fit(training_data)

#     data1['Labels{}'.format(i)]= model.labels_

data['IsThisPersonal'] = model.predict(training_data)



dataset1['IsThisPersonal'] = data.IsThisPersonal
dataset1.title[dataset1.IsThisPersonal==1].unique()
MainAnimalClassification = dataset1[['title','category_id','tags','description']]

MainAnimalClassification['NoOfMentionsInTags'] = dataset1.tags.map(lambda x: x.lower().replace('|',' ').split(' ').count('cat') +

                                                            x.lower().replace('|',' ').split(' ').count('kitten') +

                                                            x.lower().replace('|',' ').split(' ').count('kitty') +

                                                            x.lower().replace('|',' ').split(' ').count('dog') +

                                                            x.lower().replace('|',' ').split(' ').count('doggy') +

                                                            x.lower().replace('|',' ').split(' ').count('hound') +

                                                            x.lower().replace('|',' ').split(' ').count('pup') +

                                                            x.lower().replace('|',' ').split(' ').count('puppy') +

                                                            x.lower().replace('|',' ').split(' ').count('pussy') +

                                                            x.lower().replace('|',' ').split(' ').count('cats') +

                                                            x.lower().replace('|',' ').split(' ').count('kitties') +

                                                            x.lower().replace('|',' ').split(' ').count('kittens') +

                                                            x.lower().replace('|',' ').split(' ').count('dogs') +

                                                            x.lower().replace('|',' ').split(' ').count('doggies') +

                                                            x.lower().replace('|',' ').split(' ').count('hounds') +

                                                            x.lower().replace('|',' ').split(' ').count('pups') +

                                                            x.lower().replace('|',' ').split(' ').count('puppies') +

                                                            x.lower().replace('|',' ').split(' ').count('pussies'))



MainAnimalClassification['NoOfMentionsInTitle'] = dataset1.title.map(lambda x: x.lower().split(' ').count('cat') +

                                                                x.lower().split(' ').count('kitten') +

                                                                x.lower().split(' ').count('kitty') +

                                                                x.lower().split(' ').count('dog') +

                                                                x.lower().split(' ').count('doggy') +

                                                                x.lower().split(' ').count('hound') +

                                                                x.lower().split(' ').count('pup') +

                                                                x.lower().split(' ').count('puppy') +

                                                                x.lower().split(' ').count('pussy') +

                                                                x.lower().split(' ').count('cats') +

                                                                x.lower().split(' ').count('kitties') +

                                                                x.lower().split(' ').count('kittens') +

                                                                x.lower().split(' ').count('dogs') +

                                                                x.lower().split(' ').count('doggies') +

                                                                x.lower().split(' ').count('hounds') +

                                                                x.lower().split(' ').count('pups') +

                                                                x.lower().split(' ').count('puppies') +

                                                                x.lower().split(' ').count('pussies'))







MainAnimalClassification['NoOfMentionsOverall'] = MainAnimalClassification['NoOfMentionsInTags'] + MainAnimalClassification['NoOfMentionsInTitle']

# MainAnimalClassification['NoOfMentionsInTitle']



training_data = pd.get_dummies(MainAnimalClassification[['NoOfMentionsInTitle','NoOfMentionsOverall']])

scaler = StandardScaler()

scaler.fit(training_data)

training_data = scaler.transform(training_data)



model = KMeans(n_clusters=2).fit(training_data)

dataset1['AreAnimalsInvolved'] = model.predict(training_data)

HasCapitalWordItTitle = []

NumberOfCapitalWordsInTitle = []

NumberOfWordsInTitle = []

for title in dataset1.title:

    tit1 = [word for word in word_tokenize(title) if len(word)>2]

    NumberOfCapitalWordsInTitle.append(sum([z.isupper() for z in word_tokenize(title)]))

    NumberOfWordsInTitle.append(len(tit1))



dataset1['CapitalWordsInTitle'] = NumberOfCapitalWordsInTitle

dataset1['NumberOfWordsInTitle'] = NumberOfWordsInTitle