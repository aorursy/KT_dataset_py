# Import Packages



#Dataframe packages

import json

import glob

import pandas as pd

import numpy as np

import os

from PIL import Image

import requests

from io import BytesIO

import cv2

import numpy as np

from collections import Counter

from functools import partial

import scipy as sp



#Plot packages

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from wordcloud import WordCloud



# LightGBM

import lightgbm as lgb

import scipy as sp



# Load scikit's classifier library

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, StratifiedKFold,RandomizedSearchCV

from sklearn.feature_selection import SelectFromModel



from sklearn.metrics import cohen_kappa_score,mean_squared_error, accuracy_score, confusion_matrix, f1_score,classification_report



import xgboost as xgb





#Oversampling

from imblearn.over_sampling import RandomOverSampler

from sklearn.decomposition import PCA
sentimental_analysis = sorted(glob.glob('../input/train_sentiment/*.json'))

print('num of train sentiment files: {}'.format(len(sentimental_analysis)))
# Define Empty lists

score=[]

magnitude=[]

petid=[]



for filename in sentimental_analysis:

         with open(filename, 'r') as f:

            sentiment_file = json.load(f)

         file_sentiment = sentiment_file['documentSentiment']

         file_score =  np.asarray(sentiment_file['documentSentiment']['score'])

         file_magnitude =np.asarray(sentiment_file['documentSentiment']['magnitude'])

        

            

         score.append(file_score)

         magnitude.append(file_magnitude)

        

         petid.append(filename.replace('.json','').replace('../input/train_sentiment/', ''))



# Output with sentiment data for each pet

# Output with sentiment data for each pet

sentimental_analysis = pd.concat([ pd.DataFrame(petid, columns =['PetID']) ,pd.DataFrame(score, columns =['sentiment_document_score']),

                                                pd.DataFrame(magnitude, columns =['sentiment_document_magnitude'])],axis =1)
image_metadata =  sorted(glob.glob('../input/train_metadata/*.json'))

print('num of train metadata: {}'.format(len(image_metadata)))
description=[]

topicality=[]

imageid=[]

# Read Zip File and Export a Dataset with the Score and the ID

for filename in image_metadata:

         with open(filename, 'r') as f:

            d = json.load(f)

            file_keys = list(d.keys())

         if  'labelAnnotations' in file_keys:

            file_annots = d['labelAnnotations']

            file_topicality = np.asarray([x['topicality'] for x in file_annots])

            file_description = [x['description'] for x in file_annots]

            #Create a list of all descriptions and topicality

            description.append(file_description)

            topicality.append(file_topicality)

            #Create a list with all image id name

            imageid.append(filename.replace('.json','').replace('../input/train_metadata/',''))





# Prepare the output by renaming all variables

description=pd.DataFrame(description)

topicality=pd.DataFrame(topicality)



new_names = [(i,'metadata_description_'+str(i)) for i in description.iloc[:, 0:].columns.values]

description.rename(columns = dict(new_names), inplace=True)



new_names = [(i,'metadata_topicality_'+str(i)) for i in topicality.iloc[:, 0:].columns.values]

topicality.rename(columns = dict(new_names), inplace=True)



# Output with sentiment data for each pet

image_labelannot = pd.concat([ pd.DataFrame(imageid, columns =['ImageId']) ,topicality,description],axis =1)



# create the PetId variable

image_labelannot['PetID'] = image_labelannot['ImageId'].str.split('-').str[0]





##############

# TOPICALITY #

##############



image_labelannot['metadata_topicality_mean'] = image_labelannot.iloc[:,1:10].mean(axis=1)

image_labelannot['metadata_topicality_mean']  = image_labelannot.groupby(['PetID'])['metadata_topicality_mean'].transform('mean') 



image_labelannot['metadata_topicality_max'] = image_labelannot.iloc[:,1:10].max(axis=1)

image_labelannot['metadata_topicality_max'] = image_labelannot.groupby(['PetID'])['metadata_topicality_max'].transform(max)



image_labelannot['metadata_topicality_min'] = image_labelannot.iloc[:,1:10].min(axis=1)

image_labelannot['metadata_topicality_min'] = image_labelannot.groupby(['PetID'])['metadata_topicality_min'].transform(min)





image_labelannot['metadata_topicality_0_mean']  = image_labelannot.groupby(['PetID'])['metadata_topicality_0'].transform('mean')

image_labelannot['metadata_topicality_0_max'] = image_labelannot.groupby(['PetID'])['metadata_topicality_0'].transform(max)

image_labelannot['metadata_topicality_0_min'] = image_labelannot.groupby(['PetID'])['metadata_topicality_0'].transform(min)





###############

# DESCRIPTION #

###############



# Create Features from the Images

image_labelannot['L_metadata_0_cat']=image_labelannot['metadata_description_0'].str.contains("cat").astype(int)

image_labelannot['L_metadata_0_dog'] =image_labelannot['metadata_description_0'].str.contains("dog").astype(int)



image_labelannot['L_metadata_any_cat']=image_labelannot.apply(lambda row: row.astype(str).str.contains('cat').any(), axis=1)

image_labelannot['L_metadata_any_dog']=image_labelannot.apply(lambda row: row.astype(str).str.contains('dog').any(), axis=1)



image_labelannot['L_metadata_0_cat_sum'] = image_labelannot.groupby(image_labelannot['PetID'])['L_metadata_0_cat'].transform('sum')

image_labelannot['L_metadata_0_dog_sum'] = image_labelannot.groupby(image_labelannot['PetID'])['L_metadata_0_dog'].transform('sum')



image_labelannot['L_metadata_any_cat_sum'] = image_labelannot.groupby(image_labelannot['PetID'])['L_metadata_any_cat'].transform('sum')

image_labelannot['L_metadata_any_dog_sum'] = image_labelannot.groupby(image_labelannot['PetID'])['L_metadata_any_dog'].transform('sum')



image_labelannot = image_labelannot[['PetID','metadata_topicality_max','metadata_topicality_mean','metadata_topicality_min','metadata_topicality_0_mean','metadata_topicality_0_max','metadata_topicality_0_min','L_metadata_0_cat_sum','L_metadata_0_dog_sum','L_metadata_any_cat_sum','L_metadata_any_dog_sum']]

image_labelannot=image_labelannot.drop_duplicates('PetID')
color_score_mean=[]

color_score_min=[]

color_score_max=[]



color_pixelfrac_mean=[]

color_pixelfrac_min=[]

color_pixelfrac_max=[]



imageid=[]



# Read Zip File and Export a Dataset with the Score and the ID

for filename in image_metadata:

         with open(filename, 'r') as f:

              d = json.load(f)

              file_keys = list(d.keys())

              if  'imagePropertiesAnnotation' in file_keys:

                  file_colors = d['imagePropertiesAnnotation']['dominantColors']['colors']

               

                  file_color_score_mean = np.asarray([x['score'] for x in file_colors]).mean()

                  file_color_pixelfrac_mean = np.asarray([x['pixelFraction'] for x in file_colors]).mean()



                  file_color_score_min = np.asarray([x['score'] for x in file_colors]).min()

                  file_color_pixelfrac_min = np.asarray([x['pixelFraction'] for x in file_colors]).min()





                  file_color_score_max = np.asarray([x['score'] for x in file_colors]).max()

                  file_color_pixelfrac_max = np.asarray([x['pixelFraction'] for x in file_colors]).max()





              #Create a list with all image id name

              imageid.append(filename.replace('.json','').replace('../input/train_metadata/', ''))



              color_score_mean.append(file_color_score_mean)

              color_score_min.append(file_color_score_min)

              color_score_max.append(file_color_score_max)





              color_pixelfrac_mean.append(file_color_pixelfrac_mean)

              color_pixelfrac_min.append(file_color_pixelfrac_min)

              color_pixelfrac_max.append(file_color_pixelfrac_max)



      

image_properties = pd.concat([pd.DataFrame({'ImageId':imageid}),pd.DataFrame({'metadata_color_pixelfrac_mean':color_pixelfrac_mean}), pd.DataFrame({'metadata_color_pixelfrac_min':color_pixelfrac_min}),pd.DataFrame({'metadata_color_pixelfrac_max':color_pixelfrac_max}),pd.DataFrame({'metadata_color_score_mean':color_score_mean}),pd.DataFrame({'metadata_color_score_min':color_score_min}),pd.DataFrame({'metadata_color_score_max':color_score_max})],axis=1)





# create the PetId variable

image_properties['PetID'] = image_properties['ImageId'].str.split('-').str[0]





##############

# COLOR INFO #

##############

image_properties['metadata_color_pixelfrac_mean']  = image_properties.groupby(['PetID'])['metadata_color_pixelfrac_mean'].transform('mean') 

image_properties['metadata_color_pixelfrac_min']  = image_properties.groupby(['PetID'])['metadata_color_pixelfrac_min'].transform(min) 

image_properties['metadata_color_pixelfrac_max']  = image_properties.groupby(['PetID'])['metadata_color_pixelfrac_max'].transform(max) 



image_properties['metadata_color_score_mean']  = image_properties.groupby(['PetID'])['metadata_color_score_mean'].transform('mean') 

image_properties['metadata_color_score_min']  = image_properties.groupby(['PetID'])['metadata_color_score_min'].transform(min) 

image_properties['metadata_color_score_max']  = image_properties.groupby(['PetID'])['metadata_color_score_max'].transform(max)



image_properties=image_properties.drop_duplicates('PetID')

image_properties = image_properties.drop(['ImageId'], 1)
image_quality =sorted(glob.glob('../input/train_images/*.jpg'))



blur=[]

image_pixel=[]

imageid =[]



for filename in image_quality:

              #Blur 

              image = cv2.imread(filename)

              gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

              result = cv2.Laplacian(gray, cv2.CV_64F).var() 

              # Pixels

              with Image.open(filename) as pixel:

                  width, height = pixel.size

              

              pixel = width*height

              

              #image pixel size for each image

              

              image_pixel.append(pixel)

              #blur for each image

              blur.append(result)

              #image id

              imageid.append(filename.replace('.jpg','').replace('../input/train_images/', ''))

                

# Join Pixel, Blur and Image ID

image_quality = pd.concat([ pd.DataFrame(imageid, columns =['ImageId']) ,pd.DataFrame(blur, columns =['blur']),

                                        pd.DataFrame(image_pixel,columns=['pixel'])],axis =1)



# create the PetId variable

image_quality['PetID'] = image_quality['ImageId'].str.split('-').str[0]



#Mean of the Mean

image_quality['pixel_mean'] = image_quality.groupby(['PetID'])['pixel'].transform('mean')

image_quality['blur_mean'] = image_quality.groupby(['PetID'])['blur'].transform('mean') 



image_quality['pixel_min'] = image_quality.groupby(['PetID'])['pixel'].transform('min') 

image_quality['blur_min'] = image_quality.groupby(['PetID'])['blur'].transform('min')



image_quality['pixel_max'] = image_quality.groupby(['PetID'])['pixel'].transform('max') 

image_quality['blur_max'] = image_quality.groupby(['PetID'])['blur'].transform('max')



image_quality['pixel_sum'] = image_quality.groupby(['PetID'])['pixel'].transform('sum')

image_quality['blur_sum'] = image_quality.groupby(['PetID'])['blur'].transform('sum')





image_quality = image_quality.drop(['blur','pixel','ImageId'], 1)

image_quality=image_quality.drop_duplicates('PetID')
from math import copysign, log10



huMoments0=[]

huMoments1=[]

huMoments2=[]

huMoments3=[]

huMoments4=[]

huMoments5=[]

huMoments6=[]

imageid =[]



image_info_train =sorted(glob.glob('../input/train_images/*.jpg'))



for filename in image_info_train:

            if filename.endswith("-1.jpg"): # Take only the moments of picture 1

                image = cv2.imread(filename)

                im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

                # Calculate Moments

                moments = cv2.moments(im)



                # Calculate Hu Moments

                huMoments = cv2.HuMoments(moments)

                # Log scale hu moments

                for i in range(0,7):

                      huMoments[i] = round(-1* copysign(1.0, huMoments[i]) * log10(abs(huMoments[i])),2)



                #image id

                imageid.append(filename.replace('.jpg','').replace('../input/train_images/', ''))

                huMoments0.append(huMoments[0])



                huMoments1.append(huMoments[1])

                huMoments2.append(huMoments[2])

                huMoments3.append(huMoments[3])

                huMoments4.append(huMoments[4])

                huMoments5.append(huMoments[5])

                huMoments6.append(huMoments[6])



image_moments= pd.concat([pd.DataFrame({'ImageId':imageid}),pd.DataFrame({'huMoments0':np.concatenate(huMoments0,axis=0)}), 

                                     pd.DataFrame({'huMoments1':np.concatenate(huMoments1,axis=0)}),

                                     pd.DataFrame({'huMoments2':np.concatenate(huMoments2,axis=0)}),

                                     pd.DataFrame({'huMoments3':np.concatenate(huMoments3,axis=0)}),

                                     pd.DataFrame({'huMoments4':np.concatenate(huMoments4,axis=0)}),

                                     pd.DataFrame({'huMoments5':np.concatenate(huMoments5,axis=0)}),pd.DataFrame({'huMoments6':np.concatenate(huMoments6,axis=0)})],axis=1)

            



# create the PetId variable

image_moments['PetID'] = image_moments['ImageId'].str.split('-').str[0]

image_moments = image_moments[image_moments['ImageId'].apply(lambda x:x.endswith(("-1")))]

#Load Data

train = pd.read_csv('../input/train/train.csv')

test = pd.read_csv('../input/test/test.csv')

sample_submission = pd.read_csv('../input/test/sample_submission.csv')



breed =pd.read_csv('../input/breed_labels.csv',usecols=["BreedID", "BreedName"]) #A pet could have multiple breed

color =pd.read_csv('../input/color_labels.csv') #A pet could have multiple colors

state =pd.read_csv('../input/state_labels.csv')



# Add information about color, breed, state and sentiment data

train = (pd.merge(train, breed.rename(columns={"BreedName": "BreedName1"}),  how='left', left_on=['Breed1'], right_on = ['BreedID']).drop('BreedID', axis=1))

train = (pd.merge(train, breed.rename(columns={"BreedName": "BreedName2"}),  how='left', left_on=['Breed2'], right_on = ['BreedID']).drop('BreedID', axis=1))



train = (pd.merge(train, color.rename(columns={"ColorName": "ColorName1"}),  how='left', left_on=['Color1'], right_on = ['ColorID']).drop('ColorID', axis=1))

train = (pd.merge(train, color.rename(columns={"ColorName": "ColorName2"}),  how='left', left_on=['Color2'], right_on = ['ColorID']).drop('ColorID', axis=1))

train = (pd.merge(train, color.rename(columns={"ColorName": "ColorName3"}),  how='left', left_on=['Color3'], right_on = ['ColorID']).drop('ColorID', axis=1))



train = (pd.merge(train, state,  how='inner', left_on=['State'], right_on = ['StateID']).drop('StateID', axis=1))



# Add information about sentimental analysis

train = (pd.merge(train, sentimental_analysis,  how='left', left_on=['PetID'], right_on = ['PetID']))



# Add information about Metadata Images

train = (pd.merge(train, image_properties,  how='left', left_on=['PetID'], right_on = ['PetID']))

train = (pd.merge(train, image_labelannot,  how='left', left_on=['PetID'], right_on = ['PetID']))

train = (pd.merge(train, image_moments,  how='left', left_on=['PetID'], right_on = ['PetID']))



# Add information about quality Images

train = (pd.merge(train, image_quality,  how='left', left_on=['PetID'], right_on = ['PetID']))

## Using the Kernel:https://www.kaggle.com/bibek777/stacking-kernels

## Using the Kernel:https://www.kaggle.com/bibek777/stacking-kernels



# state GDP: https://en.wikipedia.org/wiki/List_of_Malaysian_states_by_GDP

state_gdp = {

    41336: 116.679,

    41325: 40.596,

    41367: 23.02,

    41401: 190.075,

    41415: 5.984,

    41324: 37.274,

    41332: 42.389,

    41335: 52.452,

    41330: 67.629,

    41380: 5.642,

    41327: 81.284,

    41345: 80.167,

    41342: 121.414,

    41326: 280.698,

    41361: 32.270

}



# state population: https://en.wikipedia.org/wiki/Malaysia

state_population = {

    41336: 33.48283,

    41325: 19.47651,

    41367: 15.39601,

    41401: 16.74621,

    41415: 0.86908,

    41324: 8.21110,

    41332: 10.21064,

    41335: 15.00817,

    41330: 23.52743,

    41380: 2.31541,

    41327: 15.61383,

    41345: 32.06742,

    41342: 24.71140,

    41326: 54.62141,

    41361: 10.35977

}



state_area ={

    41336:19102,

41325:9500,

41367:15099,

41401:243,

41415:91,

41324:1664,

41332:6686,

41335:36137,

41330:21035,

41380:821,

41327:1048,

41345:73631,

41342:124450,

41326:8104,

41361:13035}



state_unemployment ={

    41336 : 3.6,

41325 :2.9,

41367: 3.8,

41324: 0.9,

41332 : 2.7,

41335: 2.6,

41330: 3.4,

41380: 2.9,

41327: 2.1,

41345 : 5.4,

41342 : 3.3,

41326: 3.2,

41361: 4.2,

41415: 7.8,

41401: 3.3

}



# per 1000 population

state_birth_rate = {

 41336:16.3,

41325:17.0,

41367:21.4,

41401:14.4,

41415:18.1,

41324:16.0,

41332:16.4,

41335:17.0,

41330:14.4,

41380:17.5,

41327:12.7,

41345:13.7,

41342:13.9,

41326:16.6,

41361:23.3,     

}



train["state_gdp"] = train.State.map(state_gdp)

train["state_population"] = train.State.map(state_population)

train["state_area"] = train.State.map(state_area)

train['state_unemployment']=train.State.map(state_unemployment)

train['state_birth_rate']=train.State.map(state_birth_rate)
# Color (Create a Flag pet has 1 color, 2 colors, 3 colors)

train['L_Color1'] = (pd.isnull(train['ColorName3']) & pd.isnull(train['ColorName2']) & pd.notnull(train['ColorName1'])).astype(int)

train['L_Color2'] = (pd.isnull(train['ColorName3']) & pd.notnull(train['ColorName2']) & pd.notnull(train['ColorName1'])).astype(int)

train['L_Color3'] = (pd.notnull(train['ColorName3']) & pd.notnull(train['ColorName2']) & pd.notnull(train['ColorName1'])).astype(int)



# Breed (create a flag if the pet has 1 breed or 2)

train['L_Breed1'] = (pd.isnull(train['BreedName2']) & pd.notnull(train['BreedName1'])).astype(int)

train['L_Breed2'] = (pd.notnull(train['BreedName2']) & pd.notnull(train['BreedName1'])).astype(int)



#Name (create a flag if the name is missing, with less than two letters)

train['Name_Length']=train['Name'].str.len()

train['L_Name_missing'] =  (pd.isnull(train['Name'])).astype(int)



# Breed create columns

train['L_Breed1_Siamese'] =(train['BreedName1']=='Siamese').astype(int)

train['L_Breed1_Persian']=(train['BreedName1']=='Persian').astype(int)

train['L_Breed1_Labrador_Retriever']=(train['BreedName1']=='Labrador Retriever').astype(int)

train['L_Breed1_Terrier']=(train['BreedName1']=='Terrier').astype(int)

train['L_Breed1_Golden_Retriever ']=(train['BreedName1']=='Golden Retriever').astype(int)



#Description 

train['Description_Length']=train['Description'].str.len() 



# Fee Amount

train['L_Fee_Free'] =  (train['Fee']==0).astype(int)



#Add the Number of Pets per Rescuer 

pets_total = train.groupby(['RescuerID']).size().reset_index(name='N_pets_total')

train= pd.merge(train, pets_total, left_on='RescuerID', right_on='RescuerID', how='inner')

train.count()



# No photo

train['L_NoPhoto'] =  (train['PhotoAmt']==0).astype(int)



#No Video

train['L_NoVideo'] =  (train['VideoAmt']==0).astype(int)



#Log Age 

train['Log_Age']= np.log(train.Age+1) 



#Negative Score 

train['L_scoreneg'] =  (train['sentiment_document_score']<0).astype(int)



#Quantity Amount >5

train.loc[train['Quantity'] > 5, 'Quantity'] = 5
# Normalize the Variable Description

train['Description'] =train['Description'].fillna("<MISSING>")

train['Description'] = train['Description'].str.replace('\d+', '')

train['Description'] = train['Description'].str.lower()

train["Description"] = train['Description'].str.replace('[^\w\s]','')



# Stop Words 

from nltk.corpus import stopwords



stop = stopwords.words('english')

pat = r'\b(?:{})\b'.format('|'.join(stop))

train['Description'] = train['Description'].str.replace(pat, '')

train['Description'] = train['Description'].str.replace(r'\s+', ' ')



# Stem Words

train['Description'] = train['Description'].astype(str).str.split()



from nltk.stem import PorterStemmer, WordNetLemmatizer

porter_stemmer = PorterStemmer()

train['Description']=train['Description'].apply(lambda x : [porter_stemmer.stem(y) for y in x])



train['Description']=train['Description'].apply(lambda x : " ".join(x))



def get_top_n_words(corpus, n=None):

    from sklearn.feature_extraction.text import CountVectorizer



    vec = CountVectorizer().fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    test=pd.DataFrame(words_freq[:n], columns=['words','freq']) 

    

    sns.barplot(x='words', y='freq', data=test)



get_top_n_words(train['Description'],10)



from sklearn.decomposition import TruncatedSVD, NMF

# Matrix Factorization for dimensionality reduction

from sklearn.feature_extraction.text import TfidfVectorizer



svd_ = TruncatedSVD(

    n_components=5, random_state=1337)

nmf_ = NMF(

    n_components=5, random_state=1337)



tfidf_col = TfidfVectorizer().fit_transform(train['Description'])

svd_col = svd_.fit_transform(tfidf_col)

svd_col = pd.DataFrame(svd_col)

svd_col = svd_col.add_prefix('SVD_')



nmf_col = nmf_.fit_transform(tfidf_col)

nmf_col = pd.DataFrame(nmf_col)

nmf_col = nmf_col.add_prefix('NMF_')



# Concatenate all dataframes

train = pd.concat([train,nmf_col,svd_col],axis=1)

train.head(2)
total = float(len(train)) # one person per row 

ax =sns.countplot(x="AdoptionSpeed", data=train,palette="Set3")

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height,

            '{:1.2f}'.format(height/total),

            ha="center") 

images_train = sorted(glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg'))



for img in images_train[:30]:

    image= Image.open(img)

    pet_id = img.replace("../input/petfinder-adoption-prediction/train_images/","").replace(".jpg","")

    pet_id= pet_id.split("-")[0]

    

    print(pet_id)

    plt.imshow(image)

    plt.title(("Name: {}\nAdoptionSpeed: {}".format(*list(map(str, train[train.PetID==pet_id][["Name", "AdoptionSpeed"]].values.tolist()[0])))))

    plt.show()
def graphics (train, target, features, ncat):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))



    sns.countplot(x=target,hue=i, data=train,palette=('#00a1ff','#f9bc86','#f85cc2')).set_title('Distribution of Adoption Speed per '+ i)

 

    # CrossTab

    cross = pd.crosstab(train[target],train[i],margins=True)

    # Delete the All obs. per variable y

    cross = cross.drop(cross.index[len(cross)-1])

    

    for j in ncat:

        # Stacked Bar Plot

        if j ==0:

            Type1 = pd.DataFrame(cross.iloc[:,j]/cross['All'])

            ax1.bar(Type1.index.values, Type1[0], color='#00a1ff', label=cross.columns[0])

        if j==1:

            Type2 = pd.DataFrame(cross.iloc[:,j]/cross['All'])

            ax1.bar(Type2.index.values, Type2[0], bottom=Type1[0], color='#f9bc86', label=cross.columns[1])

        if j ==2:

            Type3 = pd.DataFrame(cross.iloc[:,j]/cross['All'])

            ax1.bar(Type3.index.values, Type3[0], bottom=[z+t for z,t in zip(Type1[0],Type2[0])], color='#f85cc2', label=cross.columns[2])

        if j ==3:

            Type4 = pd.DataFrame(cross.iloc[:,j]/cross['All'])

            ax1.bar(Type4.index.values, Type4[0], bottom=[n+z+t for n,z,t in zip(Type1[0],Type2[0],Type3[0])], color='#ef53c7', label=cross.columns[4])

 

    # Add title and axis names

    ax1.set_title('Adoption Speed vs '+i)

    ax1.set(xlabel='Adoption Speed', ylabel=i)

    ax1.legend(loc='upper right')
def graphics_num (train, target, features):



    num_analysis =train.groupby([features, target]).size().reset_index(name='counts')



    fig, ax = plt.subplots(1, 2, figsize=(17,7))

    # Add title and axis names

    sns.lineplot(x=features, y="counts", hue = target,data=num_analysis, ax=ax[0]).set_title(target+" vs "+features)

    sns.boxplot(x=target, y=features, data=train, palette="Set1", ax=ax[1]).set_title(target+" vs "+features)
x= 'AdoptionSpeed'

features =["Type"]



for i in features:

    graphics(train,'AdoptionSpeed','Type',ncat=range(0,2))    #1. Dog 2.Cat
features=["Gender"]

for i in features:

    graphics(train,"AdoptionSpeed",i,ncat=range(0,3))    #1. Male, 2.Female 3. Mixed
# Draw a nested analysis per Target

features =['FurLength','Vaccinated','Dewormed','Sterilized','Health']

target = ['AdoptionSpeed']

for i in features:

    graphics(train,"AdoptionSpeed",i,ncat=range(0,3))   
# FurLength (0 = Not Specified) - Health (0 = Not Specified) No missing values



train.loc[train['FurLength'] == 0]

train.loc[train['Health'] == 0 ]
#Count how many pets per Age (months) and Adoption Speed group

grouped_data = train.groupby(['AdoptionSpeed'])

grouped_data['Age'].describe()
age_analysis =train.groupby(['Age', 'AdoptionSpeed']).size().reset_index(name='counts')



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,10))



sns.lineplot(x="Age", y="counts", hue = "AdoptionSpeed",data=age_analysis).set_title("Adoption Speed vs Age")

# CrossTab

cross = pd.crosstab(train["AdoptionSpeed"],train['MaturitySize'],margins=True)

# Delete the All obs. per variable y

cross = cross.drop(cross.index[len(cross)-1])

print(cross.columns)

# Stacked Bar Plot

Type1 = pd.DataFrame(cross[1]/cross['All'])

Type2 = pd.DataFrame(cross[2]/cross['All'])

Type3 = pd.DataFrame(cross[3]/cross['All'])

Type4 = pd.DataFrame(cross[4]/cross['All'])



ax1.bar(Type1.index.values, Type1[0], label='Small', color='#6f07f9')

ax1.bar(Type2.index.values, Type2[0], bottom=Type1[0], color='#d6adf7', label='Medium')

ax1.bar(Type3.index.values, Type3[0], bottom=[z+t for z,t in zip(Type1[0], Type2[0])], label='Large', color='#ef53c7')

ax1.bar(Type4.index.values, Type4[0], bottom=[s+t+z for s,z,t in zip(Type1[0], Type2[0],Type3[0])], label='Extra Large', color='#3b006b')



# Add title and axis names

ax1.set_title('Adoption Speed vs Maturity Size')

ax1.set(xlabel='Adoption Speed', ylabel="Maturity Size")

ax1.legend(loc='upper right')

plt.figure(figsize=(14,6))

sns.countplot(x="AdoptionSpeed",hue="ColorName1", data=train,palette=('#393938','#ffd700','#d87c01',"#febf09","#f9dfb0","#707070","#befbfc")).set_title('Distribution of Adoption Speed per Color')
# Draw a nested analysis per Target

features =['L_Color1','L_Color2','L_Color3']

target = ['AdoptionSpeed']

for i in features:

    graphics(train,"AdoptionSpeed",i,ncat=range(0,2))   
# Draw a nested analysis per Target

features =['L_Breed1','L_Breed2']

target = ['AdoptionSpeed']

for i in features:

    graphics(train,"AdoptionSpeed",i,ncat=range(0,2))   
# Free Adoption 

features =['L_Fee_Free']

target = ['AdoptionSpeed']

for i in features:

    graphics(train,"AdoptionSpeed",i,ncat=range(0,2))   
# Free Adoption Dogs 

features =['L_Fee_Free']

target = ['AdoptionSpeed']

for i in features:

    graphics(train.query('Type==1'),"AdoptionSpeed",i,ncat=range(0,2))   
# Free Adoption Cats 

features =['L_Fee_Free']

target = ['AdoptionSpeed']

for i in features:

    graphics(train.query('Type==2'),"AdoptionSpeed",i,ncat=range(0,2))   
fee_analysis =train.query('Fee>0').groupby(['AdoptionSpeed'])

fee_analysis['Fee'].describe()

fig= plt.subplots(figsize=(18,8))

ax = sns.countplot(x="StateName", data=train, order = train["StateName"].value_counts().index)

# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 

            fontsize=12, color='green', ha='center', va='bottom')
fig= plt.subplots(figsize=(24,12))

ax = sns.countplot(x="StateName",  hue="AdoptionSpeed",data=train, palette="Set2")
rescuer_analysis =train.groupby(['RescuerID']).size().reset_index(name='counts')

rescuer_analysis['counts'].describe()
Top4 =rescuer_analysis.sort_values('counts',ascending=False).head(4)

Top4
train_toprescuer = train.loc[train['RescuerID'].isin(Top4['RescuerID'].values.tolist())]



fig = plt.subplots(figsize=(24,10))



ax = sns.countplot(x="RescuerID",data=train_toprescuer ,hue="AdoptionSpeed" ,palette="Set3")

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 

            fontsize=12, color='blue', ha='center', va='bottom') 
# Best Rescuer for minimum 10 pets

rescuer_speed =train.groupby(['RescuerID','AdoptionSpeed']).size().reset_index(name='N_pets_speed')



pets_total = train.groupby(['RescuerID']).size().reset_index(name='N_pets_total')

rescuer_analysis= pd.merge(pets_total, rescuer_speed, left_on='RescuerID', right_on='RescuerID', how='inner')

rescuer_analysis['pct_pets'] = rescuer_analysis['N_pets_speed']/rescuer_analysis['N_pets_total']



#Big Rescuer

rescuer_analysis.query("N_pets_total>10").groupby("AdoptionSpeed")['pct_pets'].describe().reset_index()

#Small Rescuer

rescuer_analysis.query("N_pets_total<10").groupby("AdoptionSpeed")['pct_pets'].describe().reset_index()
train['PhotoAmt'].describe()
fig = plt.subplots(figsize=(12,5))

sns.distplot(train['PhotoAmt'])
train.groupby('AdoptionSpeed')['PhotoAmt'].mean()
# Photo? 

features =['L_NoPhoto']

target = ['AdoptionSpeed']

for i in features:

    graphics(train,"AdoptionSpeed",i,ncat=range(0,2))   
train['VideoAmt'].describe()
fig = plt.subplots(figsize=(12,5))

sns.distplot(train['VideoAmt'])
# Video? 

features =['L_NoVideo']

target = ['AdoptionSpeed']

for i in features:

    graphics(train,"AdoptionSpeed",i,ncat=range(0,2))   
train.groupby('AdoptionSpeed')['sentiment_document_score'].mean()
train.groupby('AdoptionSpeed')['sentiment_document_magnitude'].mean()
graphics_num(train,'AdoptionSpeed','sentiment_document_magnitude')
features =['L_scoreneg']

target = ['AdoptionSpeed']

for i in features:

    graphics(train,"AdoptionSpeed",i,ncat=range(0,2))   
#Impact on missing name or incorrect name



features =['L_Name_missing']

target = ['AdoptionSpeed']

for i in features:

    graphics(train,"AdoptionSpeed",i,ncat=range(0,2))   
graphics_num(train,'AdoptionSpeed','Name_Length')
train_dog=train.loc[train['Type'] == 1]

train_cat=train.loc[train['Type'] == 2]



text_cat = ','.join(str(v) for v in train_cat.Name)

text_dog = ','.join(str(v) for v in train_dog.Name)

print ("There are {} words in the combination of all Name.".format(len(text_dog)))
def word_cloud(text):

    # Create a word cloud image

    wc = WordCloud(background_color="white", max_words=300,

                   contour_width=3, contour_color='firebrick')



    # Generate a wordcloud

    wc.generate(text)



    # show

    plt.figure(figsize=[15,8])

    plt.imshow(wc, interpolation='bilinear')

    plt.axis("off")

    plt.show()



word_cloud(text_dog)
word_cloud(text_cat)
text_cat = ','.join(str(v) for v in train_cat.Description)

text_dog = ','.join(str(v) for v in train_dog.Description)

print ("There are {} words in the combination of all Description for Cats.".format(len(text_cat)))
graphics_num(train,'AdoptionSpeed','Description_Length')
# Draw a nested analysis per Target

features =['metadata_topicality_max','metadata_topicality_mean','metadata_topicality_min','metadata_topicality_0_mean','metadata_topicality_0_max','metadata_topicality_0_min','L_metadata_0_cat_sum','L_metadata_0_dog_sum','L_metadata_any_cat_sum','L_metadata_any_dog_sum']

target = ['AdoptionSpeed']

for i in features:

    graphics_num(train,"AdoptionSpeed",i)   
# Draw a nested analysis per Target

features =['pixel_mean','blur_mean','pixel_min','blur_min','pixel_max','blur_max','pixel_sum','blur_sum']

for i in features:

    graphics_num(train,"AdoptionSpeed",i)
def plot_correlation_matrix(df):

    corr = df.corr()



    # Generate a mask for the upper triangle

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True



    # Set up the matplotlib figure

    f, ax = plt.subplots(figsize=(30, 15))



    # Generate a custom diverging colormap

    cmap = sns.diverging_palette(220, 10, as_cmap=True)



    # Draw the heatmap with the mask and correct aspect ratio

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

                square=True, linewidths=.5, cbar_kws={"shrink": .5})
plot_correlation_matrix(train)
columns = train.columns

percent_missing = train.isnull().sum() * 100 / len(train)

missing_value_df = pd.DataFrame({'column_name': columns,

                                 'percent_missing': percent_missing})



missing_value_df =missing_value_df[missing_value_df['percent_missing']>0]

missing_value_df





plt.figure(figsize=(20, 10))

ax = sns.barplot(x="column_name", y="percent_missing", data=missing_value_df, label='Sales')

ax.set_xticklabels(ax.get_xticklabels(),rotation=75)

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % round(p.get_height(),2) + '%', 

            fontsize=12, color='grey', ha='center', va='bottom') 

    

plt.show()
# Cannot be used for this analysis (IDs, Texts...)

train_analysis = train.drop(["Name","Description","BreedName2","ColorName3",'Name','Breed1','Breed2','RescuerID','Description',

                            'BreedName1','Color1', 'Color2', 'Color3','Age','State','ImageId'],axis=1)



for col in ['sentiment_document_score', 'sentiment_document_magnitude','metadata_topicality_max','metadata_topicality_mean','metadata_topicality_min',

           'metadata_topicality_0_mean','metadata_topicality_0_max','metadata_topicality_0_min','L_metadata_0_dog_sum',

           'L_metadata_0_cat_sum','L_metadata_any_dog_sum','L_metadata_any_cat_sum','pixel_mean','pixel_min','pixel_max','pixel_sum',

           'blur_min','blur_max','blur_sum','blur_mean','metadata_color_pixelfrac_mean','metadata_color_pixelfrac_min',

           'metadata_color_pixelfrac_max','metadata_color_score_mean','metadata_color_score_min','metadata_color_score_max',

           'Description_Length','Name_Length','huMoments0','huMoments1','huMoments2','huMoments3','huMoments4','huMoments5','huMoments6']:

    train_analysis[col].fillna((train_analysis[col].median()), inplace=True)

    

# replacing na values with No Color 

train_analysis["ColorName2"].fillna("No Color", inplace = True) 
#Label Encoding Breed

#One Hot Encoding: ColorName1,ColorName2,StateName

train_analysis = pd.concat([train_analysis.drop('StateName', axis=1),pd.get_dummies(train_analysis['StateName'], prefix='State')], axis=1)



col=['ColorName1','ColorName2','Health', 'Gender', 'Dewormed','Type','MaturitySize', 'Sterilized','Vaccinated','FurLength']

for i in col:

    train_analysis = pd.concat([train_analysis.drop(i, axis=1),pd.get_dummies(train_analysis[i], prefix=i)], axis=1)
# FROM: https://www.kaggle.com/myltykritik/simple-lgbm-image-features



# The following 3 functions have been taken from Ben Hamner's github repository

# https://github.com/benhamner/Metrics

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):

    """

    Returns the confusion matrix between rater's ratings

    """

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:

        min_rating = min(rater_a + rater_b)

    if max_rating is None:

        max_rating = max(rater_a + rater_b)

    num_ratings = int(max_rating - min_rating + 1)

    conf_mat = [[0 for i in range(num_ratings)]

                for j in range(num_ratings)]

    for a, b in zip(rater_a, rater_b):

        conf_mat[a - min_rating][b - min_rating] += 1

    return conf_mat





def histogram(ratings, min_rating=None, max_rating=None):

    """

    Returns the counts of each type of rating that a rater made

    """

    if min_rating is None:

        min_rating = min(ratings)

    if max_rating is None:

        max_rating = max(ratings)

    num_ratings = int(max_rating - min_rating + 1)

    hist_ratings = [0 for x in range(num_ratings)]

    for r in ratings:

        hist_ratings[r - min_rating] += 1

    return hist_ratings





def quadratic_weighted_kappa(y, y_pred):

    """

    Calculates the quadratic weighted kappa

    axquadratic_weighted_kappa calculates the quadratic weighted kappa

    value, which is a measure of inter-rater agreement between two raters

    that provide discrete numeric ratings.  Potential values range from -1

    (representing complete disagreement) to 1 (representing complete

    agreement).  A kappa value of 0 is expected if all agreement is due to

    chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b

    each correspond to a list of integer ratings.  These lists must have the

    same length.

    The ratings should be integers, and it is assumed that they contain

    the complete range of possible ratings.

    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating

    is the minimum possible rating, and max_rating is the maximum possible

    rating

    """

    rater_a = y

    rater_b = y_pred

    min_rating=None

    max_rating=None

    rater_a = np.array(rater_a, dtype=int)

    rater_b = np.array(rater_b, dtype=int)

    assert(len(rater_a) == len(rater_b))

    

    if min_rating is None:

        min_rating = min(min(rater_a), min(rater_b))

    if max_rating is None:

        max_rating = max(max(rater_a), max(rater_b))

    conf_mat = confusion_matrix(rater_a, rater_b,

                                min_rating, max_rating)

    num_ratings = len(conf_mat)

    num_scored_items = float(len(rater_a))



    hist_rater_a = histogram(rater_a, min_rating, max_rating)

    hist_rater_b = histogram(rater_b, min_rating, max_rating)



    numerator = 0.0

    denominator = 0.0



    for i in range(num_ratings):

        for j in range(num_ratings):

            expected_count = (hist_rater_a[i] * hist_rater_b[j]

                              / num_scored_items)

            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)

            numerator += d * conf_mat[i][j] / num_scored_items

            denominator += d * expected_count / num_scored_items



    return (1.0 - numerator / denominator)

class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4



        ll = quadratic_weighted_kappa(y, X_p)

        return -ll



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.6, 1.7, 2.6, 3.6]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4

        return X_p



    def coefficients(self):

        return self.coef_['x']

    

def rmse(actual, predicted):

    return sqrt(mean_squared_error(actual, predicted))
def evaluate(y_pred, y_true):

  

    cohen_kappa= cohen_kappa_score(y_true, y_pred)

    accuracy=accuracy_score(y_true,y_pred)

    f1=f1_score(y_true,y_pred,average='micro')

    classification=classification_report(y_true,y_pred)

    

    #Confusion Matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(20,6))

    

    sns.heatmap(cm, annot=True)

    plt.title('Confusion matrix')

    plt.figure(figsize = (5,4))

    plt.show()

    #Evaluation Metrics

    print('Cohen Kappa: {:0.2f}.'.format(cohen_kappa))

    print('Accuracy Score: {:0.2f}%.'.format(accuracy))

    print('F1 Score: {:0.2f}%.'.format(f1))

    
print(train_analysis.columns)
#Extracting Features and Output

ids=train_analysis[['PetID']]

train_analysis=train_analysis.drop(['PetID','State_Labuan'],axis=1)
X, y = train_analysis.loc[:, train_analysis.columns != 'AdoptionSpeed'], train_analysis['AdoptionSpeed']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
plt.subplot(1, 2, 1)

prob_train  =y_train.value_counts(normalize=True)

prob_train.plot(kind='barh',figsize =(15,6))

plt.title('Adoption Speed Repartition for the Training Set (75% Data)')

plt.subplot(1, 2, 2)



prob  =y_test.value_counts(normalize=True)

prob.plot(kind='barh', figsize =(15,6))

plt.title('Adoption Speed Repartition for the Test Set (25% Data)')

plt.show()
model = lgb.LGBMRegressor()

model.fit(X_train, y_train)



feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,X_train.columns)), columns=['Value','Feature'])



plt.figure(figsize=(10, 17))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()



features_selection = SelectFromModel(model, threshold='1.25*median') # The Threshold is the median of features importance*1.25 

features_selection.fit(X_train, y_train)



features_selection_support = features_selection.get_support()

features_selection = X_train.loc[:,features_selection_support].columns.tolist()

features_selection
X_train =X_train.loc[:,features_selection]

X_test = X_test.loc[:,features_selection]
print('Original dataset shape %s' % Counter(y_train))



sampling_strategy= {4: 3148, 2: 3028, 3: 2444, 1: 2317, 0: 1000}

ros = RandomOverSampler(sampling_strategy= sampling_strategy, random_state=42)



X_res, y_res = ros.fit_resample(X_train, y_train)

print('Resampled dataset shape %s' % Counter(y_res))
from sklearn import preprocessing



min_max_scaler = preprocessing.MinMaxScaler()

X_scaled = min_max_scaler.fit_transform(X_train)

X_res_scaled = min_max_scaler.fit_transform(X_res)



# Instanciate a PCA object for the sake of easy visualisation

pca = PCA(n_components=2)

# Fit and transform x to visualise inside a 2D feature space

X_vis = pca.fit_transform(X_scaled)

X_res_vis = pca.transform(X_res_scaled)



plt.figure()



# sp1

plt.subplot(121)

plt.scatter(X_vis[y_train == 0, 0],  X_vis[y_train == 0, 1],  c="navy", alpha=0.5,label="Class 0")

plt.legend(loc='upper left')



plt.subplot(122)

plt.scatter(X_res_vis[y_res == 0, 0],  X_res_vis[y_res == 0, 1],  c="navy", alpha=0.5,label="Class 0")



plt.legend(loc='upper left')

plt.show()
def cross_val(model,X_train,y_train):

    X = X_train

    y = y_train

    coeff = np.empty((1,4))

    cv_scores=[]

    fold=1

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    print(skf.get_n_splits(X, y))



    for train_index, val_index in skf.split(X, y):

        xtrain, xvalid = X[train_index], X[val_index]

        ytrain, yvalid = y[train_index], y[val_index]



        model.fit(

            xtrain, ytrain,

            eval_set=[(xvalid, yvalid)],

            verbose=100,

            early_stopping_rounds=100

        )



        #model.fit(xtrain, ytrain)

        valid_preds = model.predict(xvalid, num_iteration=model.best_iteration_)

        yvalid = np.array(yvalid).tolist()

        optR = OptimizedRounder()

        optR.fit(valid_preds, yvalid)



        coefficients = optR.coefficients()

        valid_p = optR.predict(valid_preds, coefficients)



        scr = quadratic_weighted_kappa(yvalid, valid_p)

        cv_scores.append(scr)



        print("QWK = {}. Coef = {}".format(scr, coefficients))

        coefficients.reshape((4, 1))



        coeff = np.vstack([coeff, coefficients])

        fold += 1





    coeff = np.delete(coeff, (0), axis=0)

    global coefficient_mean

    coefficient_mean = coeff.mean(axis=0)

    print("Coef Mean ={}".format(coefficient_mean))
lgb_params = {

'boosting_type': 'gbdt',

'objective': 'regression',

'learning_rate': 0.005,

'subsample': .8,

'colsample_bytree': 0.8,

'min_split_gain': 0.006,

'min_child_samples': 150,

'min_child_weight': 0.1,

'n_estimators': 1000,

'num_leaves': 80,

'silent': -1,

'verbose': -1,

'max_depth': 11,

'random_state': 2018

}

    

lgb_model = lgb.LGBMRegressor(**lgb_params)



cross_val(lgb_model,X_res,y_res)



#Prediction

y_pred=lgb_model.predict(X_train.values)

print(coefficient_mean)
optR=OptimizedRounder()

y_true = pd.DataFrame(y_train)

y_true.reset_index(inplace=True,drop=False)



predictions = optR.predict(y_pred, coefficient_mean).astype(int)

pred_lgb = pd.concat([pd.DataFrame(y_pred),y_true,pd.DataFrame(predictions)],axis=1,ignore_index=True)

pred_lgb.columns = ['y_pred','index','y_true','y_pred_class']



pred_lgb['y_pred_class'].value_counts()
evaluate(pred_lgb['y_pred_class'].tolist(), pred_lgb['y_true'].tolist())
y_pred=lgb_model.predict(X_test.values)

predictions = optR.predict(y_pred, coefficient_mean).astype(int)

y_test =pd.DataFrame(y_test)

y_test.reset_index(drop=False, inplace=True)





pred_lgb = pd.concat([pd.DataFrame(y_pred),y_test,pd.DataFrame(predictions)],axis=1,ignore_index=True)

pred_lgb.columns = ['y_pred','index','y_true','y_pred_class']

pred_lgb['y_pred_class'].value_counts()
evaluate(pred_lgb['y_pred_class'].tolist(), pred_lgb['y_true'].tolist())
lgb_params = {

'boosting_type': 'gbdt',

'objective': 'regression',

'learning_rate': 0.005,

'subsample': .8,

'colsample_bytree': 0.8,

'min_split_gain': 0.006,

'min_child_samples': 150,

'min_child_weight': 0.1,

'n_estimators': 1000,

'num_leaves': 80,

'silent': -1,

'verbose': -1,

'max_depth': 11,

'random_state': 20

}

    

lgb_model2 = lgb.LGBMRegressor(**lgb_params)



lgb_model2.fit(X_res, y_res)



#Prediction

y_pred2=lgb_model2.predict(X_test)
optR=OptimizedRounder()



predictions2 = optR.predict(y_pred2, coefficient_mean).astype(int)



pred_lgb2 = pd.concat([pd.DataFrame(y_pred2),y_true,pd.DataFrame(predictions2)],axis=1,ignore_index=True)

pred_lgb2.columns = ['y_pred_lgb2','index','y_true','y_pred_class']



pred_lgb2['y_pred_class'].value_counts()
lgb_params = {

'boosting_type': 'gbdt',

'objective': 'regression',

'learning_rate': 0.005,

'subsample': .8,

'colsample_bytree': 0.8,

'min_split_gain': 0.006,

'min_child_samples': 150,

'min_child_weight': 0.1,

'n_estimators': 1000,

'num_leaves': 80,

'silent': -1,

'verbose': -1,

'max_depth': 11,

'random_state': 2019

}

   

lgb_model3 = lgb.LGBMRegressor(**lgb_params)



lgb_model3.fit(X_res, y_res)



#Prediction

y_pred=lgb_model3.predict(X_test.values)
optR=OptimizedRounder()



predictions = optR.predict(y_pred, coefficient_mean).astype(int)

pred_lgb3 = pd.concat([pd.DataFrame(y_pred),y_true,pd.DataFrame(predictions)],axis=1,ignore_index=True)

pred_lgb3.columns = ['y_pred_lgb3','index','y_true','y_pred_class']



pred_lgb3['y_pred_class'].value_counts()