# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df_amazon = pd.read_json('../input/Amazon_Instant_Video_5.json', lines = True)
df_app = pd.read_json('../input/Apps_for_Android_5.json', lines = True)
df_automotive = pd.read_json('../input/Automotive_5.json', lines = True)
df_baby = pd.read_json('../input/Baby_5.json', lines = True)
df_beauty = pd.read_json('../input/Beauty_5.json', lines = True)
df_cd = pd.read_json('../input/CDs_and_Vinyl_5.json', lines = True)
df_cell = pd.read_json('../input/Cell_Phones_and_Accessories_5.json', lines = True)
df_cloth = pd.read_json('../input/Clothing_Shoes_and_Jewelry_5.json', lines = True)
df_digital = pd.read_json('../input/Digital_Music_5.json', lines = True)
df_electronic = pd.read_json('../input/Electronics_5.json', lines = True)
df_grocery = pd.read_json('../input/Grocery_and_Gourmet_Food_5.json', lines = True)
df_health = pd.read_json('../input/Health_and_Personal_Care_5.json', lines = True)
df_home = pd.read_json('../input/Home_and_Kitchen_5.json', lines = True)
df_kindle = pd.read_json('../input/Kindle_Store_5.json', lines = True)
df_musical = pd.read_json('../input/Musical_Instruments_5.json', lines = True)
df_office = pd.read_json('../input/Office_Products_5.json', lines = True)
df_patio = pd.read_json('../input/Patio_Lawn_and_Garden_5.json', lines = True)
df_pet = pd.read_json('../input/Pet_Supplies_5.json', lines = True)
df_sport = pd.read_json('../input/Sports_and_Outdoors_5.json', lines = True)
df_tool = pd.read_json('../input/Tools_and_Home_Improvement_5.json', lines = True)
df_toy = pd.read_json('../input/Toys_and_Games_5.json', lines = True)
df_video = pd.read_json('../input/Video_Games_5.json', lines = True)
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
def extract_features(word_list):
    return dict([(word, True) for word in word_list])
if __name__=='__main__':
   positive_fileids = movie_reviews.fileids('pos')
   negative_fileids = movie_reviews.fileids('neg')
features_positive = [(extract_features(movie_reviews.words(fileids=[f])), 
           'Positive') for f in positive_fileids]
features_negative = [(extract_features(movie_reviews.words(fileids=[f])), 
           'Negative') for f in negative_fileids]
threshold_factor = 0.8
threshold_positive = int(threshold_factor * len(features_positive))
threshold_negative = int(threshold_factor * len(features_negative))
features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]  
print ("\nNumber of training datapoints:", len(features_train))
print ("Number of test datapoints:", len(features_test))
classifier = NaiveBayesClassifier.train(features_train)
print ("\nAccuracy of the classifier:"), nltk.classify.util.accuracy(classifier, features_test)
print("\nTop 20 most informative words:")
for item in classifier.most_informative_features()[:20]:
    print (item[0])
def reviewAna(review) :
    probdist = classifier.prob_classify(extract_features(review.split()))
    pred_sentiment = probdist.max()
    if (pred_sentiment == 'Negative'):
        return False
    else : return True
df_amazon['Prola'] = df_amazon.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_app['Prola'] = df_app.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_automotive['Prola'] = df_automotive.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_baby['Prola'] = df_baby.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_beauty['Prola'] = df_beauty.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_cd['Prola'] = df_cd.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_cell['Prola'] = df_cell.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_cloth['Prola'] = df_cloth.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_digital['Prola'] = df_digital.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_electronic['Prola'] = df_electronic.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_grocery['Prola'] = df_grocery.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_health['Prola'] = df_health.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_home['Prola'] = df_home.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_kindle['Prola'] = df_kindle.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_musical['Prola'] = df_musical.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_office['Prola'] = df_office.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_patio['Prola'] = df_patio.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_pet['Prola'] = df_pet.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_sport['Prola'] = df_sport.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_tool['Prola'] = df_tool.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_toy['Prola'] = df_toy.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_video['Prola'] = df_video.apply(lambda row : reviewAna(row['reviewText']),axis = 1)
df_amazon_prola = df_amazon.groupby('Prola').size()
df_app_prola = df_app.groupby('Prola').size()
df_automotive_prola = df_automotive.groupby('Prola').size()
df_baby_prola = df_baby.groupby('Prola').size()
df_beauty_prola = df_beauty.groupby('Prola').size()
df_cd_prola = df_cd.groupby('Prola').size()
df_cell_prola = df_cell.groupby('Prola').size()
df_cloth_prola = df_cloth.groupby('Prola').size()
df_digital_prola = df_digital.groupby('Prola').size()
df_electronic_prola = df_electronic.groupby('Prola').size()
df_grocery_prola = df_grocery.groupby('Prola').size()
df_health_prola = df_health.groupby('Prola').size()
df_home_prola = df_home.groupby('Prola').size()
df_kindle_prola = df_kindle.groupby('Prola').size()
df_musical_prola = df_musical.groupby('Prola').size()
df_office_prola = df_office.groupby('Prola').size()
df_patio_prola = df_patio.groupby('Prola').size()
df_pet_prola = df_pet.groupby('Prola').size()
df_sport_prola = df_sport.groupby('Prola').size()
df_tool_prola = df_tool.groupby('Prola').size()
df_toy_prola = df_toy.groupby('Prola').size()
df_video_prola = df_video.groupby('Prola').size()
amazon_negative = df_amazon_prola.loc[False]
amazon_positive = df_amazon_prola.loc[True]
amazon_max_rating = df_amazon['overall'].max()
amazon_trust = (amazon_positive/(amazon_positive+amazon_negative))*amazon_max_rating

app_negative = df_app_prola.loc[False]
app_positive = df_app_prola.loc[True]
app_max_rating = df_app['overall'].max()
app_trust = (app_positive/(app_positive+app_negative))*app_max_rating

automotive_negative = df_automotive_prola.loc[False]
automotive_positive = df_automotive_prola.loc[True]
automotive_max_rating = df_automotive['overall'].max()
automotive_trust = (automotive_positive/(automotive_positive+automotive_negative))*automotive_max_rating

baby_negative = df_baby_prola.loc[False]
baby_positive = df_baby_prola.loc[True]
baby_max_rating = df_baby['overall'].max()
baby_trust = (baby_positive/(baby_positive+baby_negative))*baby_max_rating

beauty_negative = df_beauty_prola.loc[False]
beauty_positive = df_beauty_prola.loc[True]
beauty_max_rating = df_beauty['overall'].max()
beauty_trust = (beauty_positive/(beauty_positive+beauty_negative))*beauty_max_rating

cd_negative = df_cd_prola.loc[False]
cd_positive = df_cd_prola.loc[True]
cd_max_rating = df_cd['overall'].max()
cd_trust = (cd_positive/(cd_positive+cd_negative))*cd_max_rating

cell_negative = df_cell_prola.loc[False]
cell_positive = df_cell_prola.loc[True]
cell_max_rating = df_cell['overall'].max()
cell_trust = (cell_positive/(cell_positive+cell_negative))*cell_max_rating

cloth_negative = df_cloth_prola.loc[False]
cloth_positive = df_cloth_prola.loc[True]
cloth_max_rating = df_cloth['overall'].max()
cloth_trust = (cloth_positive/(cloth_positive+cloth_negative))*cloth_max_rating

digital_negative = df_digital_prola.loc[False]
digital_positive = df_digital_prola.loc[True]
digital_max_rating = df_digital['overall'].max()
digital_trust = (digital_positive/(digital_positive+digital_negative))*digital_max_rating

electronic_negative = df_electronic_prola.loc[False]
electronic_positive = df_electronic_prola.loc[True]
electronic_max_rating = df_electronic['overall'].max()
electronic_trust = (electronic_positive/(electronic_positive+electronic_negative))*electronic_max_rating

grocery_negative = df_grocery_prola.loc[False]
grocery_positive = df_grocery_prola.loc[True]
grocery_max_rating = df_grocery['overall'].max()
grocery_trust = (grocery_positive/(grocery_positive+grocery_negative))*grocery_max_rating

health_negative = df_health_prola.loc[False]
health_positive = df_health_prola.loc[True]
health_max_rating = df_health['overall'].max()
health_trust = (health_positive/(health_positive+health_negative))*health_max_rating

home_negative = df_home_prola.loc[False]
home_positive = df_home_prola.loc[True]
home_max_rating = df_home['overall'].max()
home_trust = (home_positive/(home_positive+home_negative))*home_max_rating

kindle_negative = df_kindle_prola.loc[False]
kindle_positive = df_kindle_prola.loc[True]
kindle_max_rating = df_kindle['overall'].max()
kindle_trust = (kindle_positive/(kindle_positive+kindle_negative))*kindle_max_rating

musical_negative = df_musical_prola.loc[False]
musical_positive = df_musical_prola.loc[True]
musical_max_rating = df_musical['overall'].max()
musical_trust = (musical_positive/(musical_positive+musical_negative))*musical_max_rating

office_negative = df_office_prola.loc[False]
office_positive = df_office_prola.loc[True]
office_max_rating = df_office['overall'].max()
office_trust = (office_positive/(office_positive+office_negative))*office_max_rating

patio_negative = df_patio_prola.loc[False]
patio_positive = df_patio_prola.loc[True]
patio_max_rating = df_patio['overall'].max()
patio_trust = (patio_positive/(patio_positive+patio_negative))*patio_max_rating

pet_negative = df_pet_prola.loc[False]
pet_positive = df_pet_prola.loc[True]
pet_max_rating = df_pet['overall'].max()
pet_trust = (pet_positive/(pet_positive+pet_negative))*pet_max_rating

sport_negative = df_sport_prola.loc[False]
sport_positive = df_sport_prola.loc[True]
sport_max_rating = df_sport['overall'].max()
sport_trust = (sport_positive/(sport_positive+sport_negative))*sport_max_rating

tool_negative = df_tool_prola.loc[False]
tool_positive = df_tool_prola.loc[True]
tool_max_rating = df_tool['overall'].max()
tool_trust = (tool_positive/(tool_positive+tool_negative))*tool_max_rating

toy_negative = df_toy_prola.loc[False]
toy_positive = df_toy_prola.loc[True]
toy_max_rating = df_toy['overall'].max()
toy_trust = (toy_positive/(toy_positive+toy_negative))*toy_max_rating

video_negative = df_video_prola.loc[False]
video_positive = df_video_prola.loc[True]
video_max_rating = df_video['overall'].max()
video_trust = (video_positive/(video_positive+video_negative))*video_max_rating
raw_data = {'Trust Score(max = 5)': [amazon_trust,app_trust,automotive_trust,baby_trust,beauty_trust,cd_trust,cell_trust,cloth_trust,digital_trust,electronic_trust,
                                    grocery_trust,health_trust,home_trust,kindle_trust,musical_trust,office_trust,patio_trust,pet_trust,sport_trust,tool_trust,
                                    toy_trust,video_trust]}
df = pd.DataFrame(raw_data, index = ['Amazon', 'App', 'Automotive', 'Baby', 'Beauty', 'CDs & Vinyl', 'Cell Phone','Clothing Shoes','Digital Music','Electronic',
                                    'Grocery & Gourment','Health Care','Home & Kitchen','Kindle Store','Musical Ins','Office Product','Patio & Garden','Pet Supplie','Sport & Outdoor','Tool & Home',
                                    'Toy & Game','Video Game'])
df
