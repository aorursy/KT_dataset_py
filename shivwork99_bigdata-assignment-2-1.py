import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
dfInstantVideo = pd.read_json('../input/Amazon_Instant_Video_5.json', lines=True)
dfAppAndroid = pd.read_json('../input/Apps_for_Android_5.json', lines=True)
dfAuto = pd.read_json('../input/Automotive_5.json', lines=True)
dfBaby = pd.read_json('../input/Baby_5.json', lines=True)
dfBeauty = pd.read_json('../input/Beauty_5.json', lines=True)
dfCd = pd.read_json('../input/CDs_and_Vinyl_5.json', lines=True)
dfPhone = pd.read_json('../input/Cell_Phones_and_Accessories_5.json', lines=True)
dfClothing = pd.read_json('../input/Clothing_Shoes_and_Jewelry_5.json', lines=True)
dfDigital = pd.read_json('../input/Digital_Music_5.json', lines=True)
dfElectonics = pd.read_json('../input/Electronics_5.json', lines=True)
dfGrocery = pd.read_json('../input/Grocery_and_Gourmet_Food_5.json', lines=True)
dfHealth = pd.read_json('../input/Health_and_Personal_Care_5.json', lines=True)
dfHome = pd.read_json('../input/Home_and_Kitchen_5.json', lines=True)
dfKindle = pd.read_json('../input/Kindle_Store_5.json', lines=True)
dfMusical = pd.read_json('../input/Musical_Instruments_5.json', lines=True)
dfOffice = pd.read_json('../input/Office_Products_5.json', lines=True)
dfPatio = pd.read_json('../input/Patio_Lawn_and_Garden_5.json', lines=True)
dfPet = pd.read_json('../input/Pet_Supplies_5.json', lines=True)
dfSport = pd.read_json('../input/Sports_and_Outdoors_5.json', lines=True)
dfTool = pd.read_json('../input/Tools_and_Home_Improvement_5.json', lines=True)
dfToy = pd.read_json('../input/Toys_and_Games_5.json', lines=True)
dfVideo = pd.read_json('../input/Video_Games_5.json', lines=True)
import nltk
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize 

def word_feats(words):
    return dict([(word, True) for word in words])
 
negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

negcutoff = len(negfeats)*3//4
poscutoff = len(posfeats)*3//4

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
#dictionary = set(word.lower() for passage in trainfeats for word in word_tokenize(passage[0])) 
print ('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))
 
classifier = NaiveBayesClassifier.train(trainfeats)
print ('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))

#classifier.show_most_informative_features()
input_reviews = [ 
    "Great place to be when you are in Bangalore."
    
]
print ("\nPredictions:")
for review in input_reviews:
    print ("\nReview:", review)
    probdist = classifier.prob_classify(word_feats(review.split()))
    pred_sentiment = probdist.max()
    print ("Predicted sentiment:", pred_sentiment) 
    print ("Probability:", round(probdist.prob(pred_sentiment), 2))
def review_scoring(review):
    probdist = classifier.prob_classify(word_feats(review.split()))
    pred_sentiment = probdist.max()
    if pred_sentiment == 'pos':
        return "positive"
    else :
        return "negative"
    
    
    

dfInstantVideo['Score'] = dfInstantVideo.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfAppAndroid['Score'] = dfAppAndroid.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfAuto['Score'] = dfAuto.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfBaby['Score'] = dfBaby.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfBeauty['Score'] = dfBeauty.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfCd['Score'] = dfCd.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfPhone['Score'] = dfPhone.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfClothing['Score'] = dfClothing.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfDigital['Score'] = dfDigital.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfElectonics['Score'] = dfElectonics.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfGrocery['Score'] = dfGrocery.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfHealth['Score'] = dfHealth.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfHome['Score'] = dfHome.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfKindle['Score'] = dfKindle.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfMusical['Score'] = dfMusical.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfOffice['Score'] = dfOffice.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfPatio['Score'] = dfPatio.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfPet['Score'] = dfPet.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfSport['Score'] = dfSport.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfTool['Score'] = dfTool.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfToy['Score'] = dfToy.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
dfVideo['Score'] = dfVideo.apply(lambda row : review_scoring(row['reviewText']),axis = 1)
trustValue = []
count_In = dfInstantVideo.groupby('Score').size()
trust_In = (count_In.loc['positive']/(count_In.loc['positive']+count_In.loc['negative']))*dfInstantVideo['overall'].max()
trustValue.append(trust_In)

count_App = dfAppAndroid.groupby('Score').size()
trust_App = (count_App.loc['positive']/(count_App.loc['positive']+count_App.loc['negative']))*dfAppAndroid['overall'].max()
trustValue.append(trust_App)

count_Auto = dfAuto.groupby('Score').size()
trust_Auto = (count_Auto.loc['positive']/(count_Auto.loc['positive']+count_Auto.loc['negative']))*dfAuto['overall'].max()
trustValue.append(trust_Auto)

count_Baby = dfBaby.groupby('Score').size()
trust_Baby = (count_Baby.loc['positive']/(count_Baby.loc['positive']+count_Baby.loc['negative']))*dfBaby['overall'].max()
trustValue.append(trust_Baby)

count_Beauty = dfBeauty.groupby('Score').size()
trust_Beauty = (count_Beauty.loc['positive']/(count_Beauty.loc['positive']+count_Beauty.loc['negative']))*dfBeauty['overall'].max()
trustValue.append(trust_Beauty)

count_Cd = dfCd.groupby('Score').size()
trust_Cd = (count_Cd.loc['positive']/(count_Cd.loc['positive']+count_Cd.loc['negative']))*dfCd['overall'].max()
trustValue.append(trust_Cd)

count_Phone = dfPhone.groupby('Score').size()
trust_Phone = (count_Phone.loc['positive']/(count_Phone.loc['positive']+count_Phone.loc['negative']))*dfPhone['overall'].max()
trustValue.append(trust_Phone)

count_Clothing = dfClothing.groupby('Score').size()
trust_Clothing = (count_Clothing.loc['positive']/(count_Clothing.loc['positive']+count_Clothing.loc['negative']))*dfClothing['overall'].max()
trustValue.append(trust_Clothing)

count_Digital = dfDigital.groupby('Score').size()
trust_Digital = (count_Digital.loc['positive']/(count_Digital.loc['positive']+count_Digital.loc['negative']))*dfDigital['overall'].max()
trustValue.append(trust_Digital)

count_Elect = dfElectonics.groupby('Score').size()
trust_Elect = (count_Elect.loc['positive']/(count_Elect.loc['positive']+count_Elect.loc['negative']))*dfElectonics['overall'].max()
trustValue.append(trust_Elect)

count_Grocery = dfGrocery.groupby('Score').size()
trust_Grocery = (count_Grocery.loc['positive']/(count_Grocery.loc['positive']+count_Grocery.loc['negative']))*dfGrocery['overall'].max()
trustValue.append(trust_Grocery)

count_Health = dfHealth.groupby('Score').size()
trust_Health = (count_Health.loc['positive']/(count_Health.loc['positive']+count_Health.loc['negative']))*dfHealth['overall'].max()
trustValue.append(trust_Health)

count_Home = dfHome.groupby('Score').size()
trust_Home = (count_Home.loc['positive']/(count_Home.loc['positive']+count_Home.loc['negative']))*dfHome['overall'].max()
trustValue.append(trust_Home)

count_Kindle = dfKindle.groupby('Score').size()
trust_Kindle = (count_Kindle.loc['positive']/(count_Kindle.loc['positive']+count_Kindle.loc['negative']))*dfKindle['overall'].max()
trustValue.append(trust_Kindle)

count_Musical = dfMusical.groupby('Score').size()
trust_Musical = (count_Musical.loc['positive']/(count_Musical.loc['positive']+count_Musical.loc['negative']))*dfMusical['overall'].max()
trustValue.append(trust_Musical)

count_Office = dfOffice.groupby('Score').size()
trust_Office = (count_Office.loc['positive']/(count_Office.loc['positive']+count_Office.loc['negative']))*dfOffice['overall'].max()
trustValue.append(trust_Office)

count_Patio = dfPatio.groupby('Score').size()
trust_Patio = (count_Patio.loc['positive']/(count_Patio.loc['positive']+count_Patio.loc['negative']))*dfPatio['overall'].max()
trustValue.append(trust_Patio)

count_Pet = dfPet.groupby('Score').size()
trust_Pet = (count_Pet.loc['positive']/(count_Pet.loc['positive']+count_Pet.loc['negative']))*dfPet['overall'].max()
trustValue.append(trust_Pet)

count_Sport = dfSport.groupby('Score').size()
trust_Sport = (count_Sport.loc['positive']/(count_Sport.loc['positive']+count_Sport.loc['negative']))*dfSport['overall'].max()
trustValue.append(trust_Sport)

count_Tool = dfTool.groupby('Score').size()
trust_Tool = (count_Tool.loc['positive']/(count_Tool.loc['positive']+count_Tool.loc['negative']))*dfTool['overall'].max()
trustValue.append(trust_Tool)

count_Toy = dfToy.groupby('Score').size()
trust_Toy = (count_Toy.loc['positive']/(count_Toy.loc['positive']+count_Toy.loc['negative']))*dfToy['overall'].max()
trustValue.append(trust_Toy)

count_Video = dfVideo.groupby('Score').size()
trust_Video = (count_Video.loc['positive']/(count_Video.loc['positive']+count_Video.loc['negative']))*dfVideo['overall'].max()
trustValue.append(trust_Video)
column = ['Product list','Trust vulue']
ResultDict = {'Product list':  ['Amazon Instant Video','Apps for Android','Automotive','Baby','Beauty','CDs and Vinyl','Cell Phones and Accessories','Clothing, Shoes and Jewelry','Digital Music','Electronics','Grocery and Gourmet Food','Health and Personal Care','Home and Kitchen','Kindle Store','Musical Instruments','Office Products','Patio, Lawn and Garden','Pet Supplies','Sports and Outdoors','Tools and Home Improvement','Toys and Games','Video Games'],'Trust vulue':trustValue}
ResultDf = pd.DataFrame(ResultDict, columns=column)
ResultDf
import seaborn
seaborn.set() 
from itertools import cycle, islice
import matplotlib.pylab as plt



df_class = ResultDf
df_class.index = ['Amazon Instant Video','Apps for Android','Automotive','Baby','Beauty','CDs and Vinyl','Cell Phones and Accessories','Clothing, Shoes and Jewelry','Digital Music','Electronics','Grocery and Gourmet Food','Health and Personal Care','Home and Kitchen','Kindle Store','Musical Instruments','Office Products','Patio, Lawn and Garden','Pet Supplies','Sports and Outdoors','Tools and Home Improvement','Toys and Games','Video Games']
df_class.plot(kind='barh', figsize=(15,7), colormap='PRGn',title=" Each Category Rating")
plt.ylabel('Value of Trust',Fontsize = 15)