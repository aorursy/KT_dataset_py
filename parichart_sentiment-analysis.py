import numpy as np 
import pandas as pd 
import os
Instant_Video_df = pd.read_json('../input/Amazon_Instant_Video_5.json', lines=True)

Apps_for_Android_df = pd.read_json('../input/Apps_for_Android_5.json', lines=True)

Automotive_df = pd.read_json('../input/Automotive_5.json', lines=True)

Baby_df = pd.read_json('../input/Baby_5.json', lines=True)

Beauty_df = pd.read_json('../input/Beauty_5.json', lines=True)

CDs_and_Vinyl_df = pd.read_json('../input/CDs_and_Vinyl_5.json', lines=True)

Cell_Phones_and_Accessories_df = pd.read_json('../input/Cell_Phones_and_Accessories_5.json', lines=True)

Clothing_Shoes_and_Jewelry_df = pd.read_json('../input/Clothing_Shoes_and_Jewelry_5.json', lines=True)

Digital_Music_df = pd.read_json('../input/Digital_Music_5.json', lines=True)

Electronics_df = pd.read_json('../input/Electronics_5.json', lines=True)

Grocery_and_Gourmet_Food_df = pd.read_json('../input/Grocery_and_Gourmet_Food_5.json', lines=True)

Health_and_Personal_Care_df = pd.read_json('../input/Health_and_Personal_Care_5.json', lines=True)

Home_and_Kitchen_df = pd.read_json('../input/Home_and_Kitchen_5.json', lines=True)

Kindle_Store_df = pd.read_json('../input/Kindle_Store_5.json', lines=True)

Musical_Instruments_df = pd.read_json('../input/Musical_Instruments_5.json', lines=True)

Office_Products_df = pd.read_json('../input/Office_Products_5.json', lines=True)

Patio_Lawn_and_Garden_df = pd.read_json('../input/Patio_Lawn_and_Garden_5.json', lines=True)

Pet_Supplies_df = pd.read_json('../input/Pet_Supplies_5.json', lines=True)

Sports_and_Outdoors_df = pd.read_json('../input/Sports_and_Outdoors_5.json', lines=True)

Tools_and_Home_Improvement_df = pd.read_json('../input/Tools_and_Home_Improvement_5.json', lines=True)

Toys_and_Games_df = pd.read_json('../input/Toys_and_Games_5.json', lines=True)

Video_Games_df = pd.read_json('../input/Video_Games_5.json', lines=True)
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
def extract_features(word_list):
    return dict([(word, True) for word in word_list])
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
print ("\nAccuracy of the classifier:", nltk.classify.util.accuracy(classifier, features_test))
print ("\nTop 10 most informative words:")
for item in classifier.most_informative_features()[:10]:
    print (item[0])
def sentence_Scoring(review):
    probdist = classifier.prob_classify(extract_features(review.split()))
    pred_sentiment = probdist.max()
    if pred_sentiment == "Negative":
        return 0
    else:
        return 1
Instant_Video_df['Score'] = Instant_Video_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Apps_for_Android_df['Score'] = Apps_for_Android_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Automotive_df['Score'] = Automotive_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Baby_df['Score'] = Baby_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Beauty_df['Score'] = Beauty_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

CDs_and_Vinyl_df['Score'] = CDs_and_Vinyl_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Cell_Phones_and_Accessories_df['Score'] = Cell_Phones_and_Accessories_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Clothing_Shoes_and_Jewelry_df['Score'] = Clothing_Shoes_and_Jewelry_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Digital_Music_df['Score'] = Digital_Music_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Electronics_df['Score'] = Electronics_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Grocery_and_Gourmet_Food_df['Score'] = Grocery_and_Gourmet_Food_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Health_and_Personal_Care_df['Score'] = Health_and_Personal_Care_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Home_and_Kitchen_df['Score'] = Home_and_Kitchen_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Kindle_Store_df['Score'] = Kindle_Store_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Musical_Instruments_df['Score'] = Musical_Instruments_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Office_Products_df['Score'] = Office_Products_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Patio_Lawn_and_Garden_df['Score'] = Patio_Lawn_and_Garden_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Pet_Supplies_df['Score'] = Pet_Supplies_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Sports_and_Outdoors_df['Score'] = Sports_and_Outdoors_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Tools_and_Home_Improvement_df['Score'] = Tools_and_Home_Improvement_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Toys_and_Games_df['Score'] = Toys_and_Games_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)

Video_Games_df['Score'] = Video_Games_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Trust_Values = [] 
count_score_IV = Instant_Video_df.groupby('Score').size()
total_neg_IV = count_score_IV.loc[0]
total_pos_IV = count_score_IV.loc[1]
max_rate_IV = Instant_Video_df['overall'].max()
trust_IV = (total_pos_IV/(total_pos_IV+total_neg_IV))*max_rate_IV
Trust_Values.append(trust_IV.round(2))

count_score_AA = Apps_for_Android_df.groupby('Score').size()
total_neg_AA = count_score_AA.loc[0]
total_pos_AA = count_score_AA.loc[1]
max_rate_AA = Apps_for_Android_df['overall'].max()
trust_AA = (total_pos_AA/(total_pos_AA+total_neg_AA))*max_rate_AA
Trust_Values.append(trust_AA.round(2))

count_score_A = Automotive_df.groupby('Score').size()
total_neg_A = count_score_A.loc[0]
total_pos_A = count_score_A.loc[1]
max_rate_A = Automotive_df['overall'].max()
trust_A = (total_pos_A/(total_pos_A+total_neg_A))*max_rate_A
Trust_Values.append(trust_A.round(2))

count_score_B = Baby_df.groupby('Score').size()
total_neg_B = count_score_B.loc[0]
total_pos_B = count_score_B.loc[1]
max_rate_B = Baby_df['overall'].max()
trust_B = (total_pos_B/(total_pos_B+total_neg_B))*max_rate_B
Trust_Values.append(trust_B.round(2))

count_score_Be = Beauty_df.groupby('Score').size()
total_neg_Be = count_score_Be.loc[0]
total_pos_Be = count_score_Be.loc[1]
max_rate_Be = Beauty_df['overall'].max()
trust_Be = (total_pos_Be/(total_pos_Be+total_neg_Be))*max_rate_Be
Trust_Values.append(trust_Be.round(2))

count_score_CD = CDs_and_Vinyl_df.groupby('Score').size()
total_neg_CD = count_score_CD.loc[0]
total_pos_CD = count_score_CD.loc[1]
max_rate_CD = CDs_and_Vinyl_df['overall'].max()
trust_CD = (total_pos_CD/(total_pos_CD+total_neg_CD))*max_rate_CD
Trust_Values.append(trust_CD.round(2))

count_score_Ce = Cell_Phones_and_Accessories_df.groupby('Score').size()
total_neg_Ce = count_score_Ce.loc[0]
total_pos_Ce = count_score_Ce.loc[1]
max_rate_Ce = Cell_Phones_and_Accessories_df['overall'].max()
trust_Ce = (total_pos_Ce/(total_pos_Ce+total_neg_Ce))*max_rate_Ce
Trust_Values.append(trust_Ce.round(2))

count_score_Cl = Clothing_Shoes_and_Jewelry_df.groupby('Score').size()
total_neg_Cl = count_score_Cl.loc[0]
total_pos_Cl = count_score_Cl.loc[1]
max_rate_Cl = Clothing_Shoes_and_Jewelry_df['overall'].max()
trust_Cl = (total_pos_Cl/(total_pos_Cl+total_neg_Cl))*max_rate_Cl
Trust_Values.append(trust_Cl.round(2))

count_score_DM = Digital_Music_df.groupby('Score').size()
total_neg_DM = count_score_DM.loc[0]
total_pos_DM = count_score_DM.loc[1]
max_rate_DM = Digital_Music_df['overall'].max()
trust_DM = (total_pos_DM/(total_pos_DM+total_neg_DM))*max_rate_DM
Trust_Values.append(trust_DM.round(2))

count_score_E = Electronics_df.groupby('Score').size()
total_neg_E = count_score_E.loc[0]
total_pos_E = count_score_E.loc[1]
max_rate_E = Electronics_df['overall'].max()
trust_E = (total_pos_E/(total_pos_E+total_neg_E))*max_rate_E
Trust_Values.append(trust_E.round(2))

count_score_G = Grocery_and_Gourmet_Food_df.groupby('Score').size()
total_neg_G = count_score_G.loc[0]
total_pos_G = count_score_G.loc[1]
max_rate_G = Grocery_and_Gourmet_Food_df['overall'].max()
trust_G = (total_pos_G/(total_pos_G+total_neg_G))*max_rate_G
Trust_Values.append(trust_G.round(2))

count_score_H = Health_and_Personal_Care_df.groupby('Score').size()
total_neg_H = count_score_H.loc[0]
total_pos_H = count_score_H.loc[1]
max_rate_H = Health_and_Personal_Care_df['overall'].max()
trust_H = (total_pos_H/(total_pos_H+total_neg_H))*max_rate_H
Trust_Values.append(trust_H.round(2))

count_score_Ho = Home_and_Kitchen_df.groupby('Score').size()
total_neg_Ho = count_score_Ho.loc[0]
total_pos_Ho = count_score_Ho.loc[1]
max_rate_Ho = Home_and_Kitchen_df['overall'].max()
trust_Ho = (total_pos_Ho/(total_pos_Ho+total_neg_Ho))*max_rate_Ho
Trust_Values.append(trust_Ho.round(2))

count_score_K = Kindle_Store_df.groupby('Score').size()
total_neg_K = count_score_K.loc[0]
total_pos_K = count_score_K.loc[1]
max_rate_K = Kindle_Store_df['overall'].max()
trust_K = (total_pos_K/(total_pos_K+total_neg_K))*max_rate_K
Trust_Values.append(trust_K.round(2))

count_score_MI = Musical_Instruments_df.groupby('Score').size()
total_neg_MI = count_score_MI.loc[0]
total_pos_MI = count_score_MI.loc[1]
max_rate_MI = Musical_Instruments_df['overall'].max()
trust_MI = (total_pos_MI/(total_pos_MI+total_neg_MI))*max_rate_MI
Trust_Values.append(trust_MI.round(2))

count_score_OP = Office_Products_df.groupby('Score').size()
total_neg_OP = count_score_OP.loc[0]
total_pos_OP = count_score_OP.loc[1]
max_rate_OP = Office_Products_df['overall'].max()
trust_OP = (total_pos_OP/(total_pos_OP+total_neg_OP))*max_rate_OP
Trust_Values.append(trust_OP.round(2))

count_score_PL = Patio_Lawn_and_Garden_df.groupby('Score').size()
total_neg_PL = count_score_PL.loc[0]
total_pos_PL = count_score_PL.loc[1]
max_rate_PL = Patio_Lawn_and_Garden_df['overall'].max()
trust_PL = (total_pos_PL/(total_pos_PL+total_neg_PL))*max_rate_PL
Trust_Values.append(trust_PL.round(2))

count_score_Pe = Pet_Supplies_df.groupby('Score').size()
total_neg_Pe = count_score_Pe.loc[0]
total_pos_Pe = count_score_Pe.loc[1]
max_rate_Pe = Pet_Supplies_df['overall'].max()
trust_Pe = (total_pos_Pe/(total_pos_Pe+total_neg_Pe))*max_rate_Pe
Trust_Values.append(trust_Pe.round(2))

count_score_Sp = Sports_and_Outdoors_df.groupby('Score').size()
total_neg_Sp = count_score_Sp.loc[0]
total_pos_Sp = count_score_Sp.loc[1]
max_rate_Sp = Sports_and_Outdoors_df['overall'].max()
trust_Sp = (total_pos_Sp/(total_pos_Sp+total_neg_Sp))*max_rate_Sp
Trust_Values.append(trust_Sp.round(2))

count_score_TH = Tools_and_Home_Improvement_df.groupby('Score').size()
total_neg_TH = count_score_TH.loc[0]
total_pos_TH = count_score_TH.loc[1]
max_rate_TH = Tools_and_Home_Improvement_df['overall'].max()
trust_TH = (total_pos_TH/(total_pos_TH+total_neg_TH))*max_rate_TH
Trust_Values.append(trust_TH.round(2))

count_score_TG = Toys_and_Games_df.groupby('Score').size()
total_neg_TG = count_score_TG.loc[0]
total_pos_TG = count_score_TG.loc[1]
max_rate_TG = Toys_and_Games_df['overall'].max()
trust_TG = (total_pos_TG/(total_pos_TG+total_neg_TG))*max_rate_TG
Trust_Values.append(trust_TG.round(2))

count_score_VG = Video_Games_df.groupby('Score').size()
total_neg_VG = count_score_VG.loc[0]
total_pos_VG = count_score_VG.loc[1]
max_rate_VG = Video_Games_df['overall'].max()
trust_VG = (total_pos_VG/(total_pos_VG+total_neg_VG))*max_rate_VG
Trust_Values.append(trust_VG.round(2))
Category_Products = ['Amazon Instant Video','Apps for Android','Automotive','Baby','Beauty','CDs and Vinyl','Cell Phones and Accessories','Clothing, Shoes and Jewelry','Digital Music','Electronics','Grocery and Gourmet Food','Health and Personal Care','Home and Kitchen','Kindle Store','Musical Instruments','Office Products','Patio, Lawn and Garden','Pet Supplies','Sports and Outdoors','Tools and Home Improvement','Toys and Games','Video Games']
print(Category_Products,Trust_Values)
Column = ['Category of Product','Vulue of Trust']
Complete_dict = {'Category of Product': Category_Products,'Vulue of Trust':Trust_Values}
Complete_df = pd.DataFrame(Complete_dict, columns=Column)
Complete_df
import seaborn
seaborn.set() 
from itertools import cycle, islice
import matplotlib.pylab as plt

df_class = Complete_df
df_class.index = Category_Products
df_class.plot(kind='bar', figsize=(13,5),title="Rating of each product's category")
plt.ylabel('Value of Trust',Fontsize = 13)
plt.bar(range(len(df_class)), df_class['Vulue of Trust'], color=plt.cm.tab20b(np.arange(len(df_class))))
x = range(len(df_class))
for a,b in zip(x, Trust_Values):
    plt.text(a, b, str(b),ha='center', va='bottom',color= 'brown')
plt.show()
