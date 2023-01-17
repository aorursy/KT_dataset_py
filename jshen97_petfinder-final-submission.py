import cv2

import gc

import glob

import os

import json

import matplotlib.pyplot as plt

import warnings

import re



import numpy as np

import pandas as pd



from PIL import Image



%matplotlib inline

%config InlineBackend.figure_format = 'svg'



warnings.filterwarnings("ignore")



plt.rcParams['figure.figsize'] = (12, 9)

plt.style.use('ggplot')



pd.options.display.max_rows = 64

pd.options.display.max_columns = 512
train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')

train['AdoptionSpeed'].astype(np.int32)

test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
from keras.applications.densenet import preprocess_input, DenseNet121

from tqdm import tqdm, tqdm_notebook



img_size = 256

batch_size = 16

pet_ids = train.index

n_batches = len(pet_ids) // batch_size + 1



def resize_to_square(im):

    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(img_size)/max(old_size)

    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = img_size - new_size[1]

    delta_h = img_size - new_size[0]

    top, bottom = delta_h//2, delta_h-(delta_h//2)

    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]

    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)

    return new_im



def load_image(path, pet_id):

    image = cv2.imread(f'{path}{pet_id}-1.jpg')

    new_image = resize_to_square(image)

    new_image = preprocess_input(new_image)

    return new_image



from keras.models import Model

from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D

import keras.backend as K

inp = Input((256,256,3))

backbone = DenseNet121(input_tensor = inp, 

                       weights="../input/densenet-keras/DenseNet-BC-121-32-no-top.h5",

                       include_top = False)

x = backbone.output

x = GlobalAveragePooling2D()(x)

x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)

x = AveragePooling1D(4)(x)

out = Lambda(lambda x: x[:,:,0])(x)



m = Model(inp,out)



pet_ids = train['PetID'].values

n_batches = len(pet_ids) // batch_size + 1



features = {}

for b in tqdm_notebook(range(n_batches)):

    start = b*batch_size

    end = (b+1)*batch_size

    batch_pets = pet_ids[start:end]

    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))

    for i,pet_id in enumerate(batch_pets):

        try:

            batch_images[i] = load_image("../input/petfinder-adoption-prediction/train_images/", pet_id)

        except:

            pass

    batch_preds = m.predict(batch_images)

    for i,pet_id in enumerate(batch_pets):

        features[pet_id] = batch_preds[i]



train_id = train['PetID'].values

test_id = test['PetID'].values



vertex_xs = []

vertex_ys = []

bounding_confidences = []

bounding_importance_fracs = []

dominant_blues = []

dominant_greens = []

dominant_reds = []

dominant_pixel_fracs = []

dominant_scores = []

label_descriptions = []

label_scores = []

nf_count = 0

nl_count = 0

for pet in train_id:

    try:

        with open('../input/petfinder-adoption-prediction/train_metadata/' + pet + '-1.json', 'r') as f:

            data = json.load(f)

        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']

        vertex_xs.append(vertex_x)

        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']

        vertex_ys.append(vertex_y)

        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']

        bounding_confidences.append(bounding_confidence)

        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)

        bounding_importance_fracs.append(bounding_importance_frac)

        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']

        dominant_blues.append(dominant_blue)

        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']

        dominant_greens.append(dominant_green)

        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']

        dominant_reds.append(dominant_red)

        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']

        dominant_pixel_fracs.append(dominant_pixel_frac)

        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']

        dominant_scores.append(dominant_score)

        if data.get('labelAnnotations'):

            label_description = data['labelAnnotations'][0]['description']

            label_descriptions.append(label_description)

            label_score = data['labelAnnotations'][0]['score']

            label_scores.append(label_score)

        else:

            nl_count += 1

            label_descriptions.append('nothing')

            label_scores.append(-1)

    except FileNotFoundError:

        nf_count += 1

        vertex_xs.append(-1)

        vertex_ys.append(-1)

        bounding_confidences.append(-1)

        bounding_importance_fracs.append(-1)

        dominant_blues.append(-1)

        dominant_greens.append(-1)

        dominant_reds.append(-1)

        dominant_pixel_fracs.append(-1)

        dominant_scores.append(-1)

        label_descriptions.append('nothing')

        label_scores.append(-1)



print(nf_count)

print(nl_count)

train.loc[:, 'vertex_x'] = vertex_xs

train.loc[:, 'vertex_y'] = vertex_ys

train.loc[:, 'bounding_confidence'] = bounding_confidences

train.loc[:, 'bounding_importance'] = bounding_importance_fracs

train.loc[:, 'dominant_blue'] = dominant_blues

train.loc[:, 'dominant_green'] = dominant_greens

train.loc[:, 'dominant_red'] = dominant_reds

train.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs

train.loc[:, 'dominant_score'] = dominant_scores

train.loc[:, 'label_score'] = label_scores





vertex_xs = []

vertex_ys = []

bounding_confidences = []

bounding_importance_fracs = []

dominant_blues = []

dominant_greens = []

dominant_reds = []

dominant_pixel_fracs = []

dominant_scores = []

label_descriptions = []

label_scores = []

nf_count = 0

nl_count = 0

for pet in test_id:

    try:

        with open('../input/petfinder-adoption-prediction/test_metadata/' + pet + '-1.json', 'r') as f:

            data = json.load(f)

        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']

        vertex_xs.append(vertex_x)

        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']

        vertex_ys.append(vertex_y)

        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']

        bounding_confidences.append(bounding_confidence)

        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)

        bounding_importance_fracs.append(bounding_importance_frac)

        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']

        dominant_blues.append(dominant_blue)

        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']

        dominant_greens.append(dominant_green)

        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']

        dominant_reds.append(dominant_red)

        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']

        dominant_pixel_fracs.append(dominant_pixel_frac)

        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']

        dominant_scores.append(dominant_score)

        if data.get('labelAnnotations'):

            label_description = data['labelAnnotations'][0]['description']

            label_descriptions.append(label_description)

            label_score = data['labelAnnotations'][0]['score']

            label_scores.append(label_score)

        else:

            nl_count += 1

            label_descriptions.append('nothing')

            label_scores.append(-1)

    except FileNotFoundError:

        nf_count += 1

        vertex_xs.append(-1)

        vertex_ys.append(-1)

        bounding_confidences.append(-1)

        bounding_importance_fracs.append(-1)

        dominant_blues.append(-1)

        dominant_greens.append(-1)

        dominant_reds.append(-1)

        dominant_pixel_fracs.append(-1)

        dominant_scores.append(-1)

        label_descriptions.append('nothing')

        label_scores.append(-1)



print(nf_count)

test.loc[:, 'vertex_x'] = vertex_xs

test.loc[:, 'vertex_y'] = vertex_ys

test.loc[:, 'bounding_confidence'] = bounding_confidences

test.loc[:, 'bounding_importance'] = bounding_importance_fracs

test.loc[:, 'dominant_blue'] = dominant_blues

test.loc[:, 'dominant_green'] = dominant_greens

test.loc[:, 'dominant_red'] = dominant_reds

test.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs

test.loc[:, 'dominant_score'] = dominant_scores

test.loc[:, 'label_score'] = label_scores
df = pd.concat([train,test],ignore_index=True)
train_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_sentiment/*.json'))

test_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_sentiment/*.json'))

sentimental_analysis = train_sentiment_files + test_sentiment_files
score=[]

magnitude=[]

petid=[]

for filename in sentimental_analysis:

    with open(filename, 'r') as f:

        sentiment_file = json.load(f)

        file_sentiment = sentiment_file['documentSentiment']

        file_score =  sentiment_file['documentSentiment']['score']

        file_magnitude = sentiment_file['documentSentiment']['magnitude']

        score.append(file_score)

        magnitude.append(file_magnitude)

        petid.append(filename.replace('.json','').replace('../input/petfinder-adoption-prediction/train_sentiment/', '').replace('../input/petfinder-adoption-prediction/test_sentiment/', ''))
score_dict = dict(zip(petid,score))

magnitude_dict = dict(zip(petid,magnitude))
df['Score'] = df['PetID'].map(score_dict)

df['Score'][df.Score.isnull()] = 0

df['Magnitude'] = df['PetID'].map(magnitude_dict)

df['Magnitude'][df.Magnitude.isnull()] = 0

df.set_index('PetID',inplace=True)
df.isnull().sum()
def namevaild(name):

    if name == np.nan:

        return 0

    elif len(str(name)) < 3:

        return 1

    elif re.match(u'[0-9]', str(name).lower()):

        return 1

    elif len(set(str(name).lower().split(' ')+['no','not','yet','male','female','unnamed'])) != len(set(str(name).lower().split(' ')))+6:

        return 1

    else:

        return 2

df['Name_state'] = df['Name'].apply(namevaild)
df['Fee_per_pet'] = df.Fee/df.Quantity



df['Fee_Bin']=pd.factorize(pd.cut(df.Fee_per_pet,bins=[0,0.01,50,100,200,500,3000],right=False))[0]

fee_bin_dummies_df = pd.get_dummies(df['Fee_Bin']).rename(columns=lambda x: 'Fee_Bin_' + str(x))

df = pd.concat([df, fee_bin_dummies_df], axis=1)
df['Quantity_Bin']=pd.factorize(pd.cut(df.Quantity,bins=[1,2,4,22],right=False))[0]

quantity_bin_dummies_df = pd.get_dummies(df['Quantity_Bin']).rename(columns=lambda x: 'Quantity_Bin_' + str(x))

df = pd.concat([df, quantity_bin_dummies_df], axis=1)
df.VideoAmt = df.VideoAmt.apply(lambda x: 1 if x > 0 else 0)



df['PhotoAmt_Bin']=pd.factorize(pd.cut(df.PhotoAmt,bins=[0,1,2,4,31],right=False))[0]

photo_bin_dummies_df = pd.get_dummies(df['PhotoAmt_Bin']).rename(columns=lambda x: 'PhotoAmt_Bin_' + str(x))

df = pd.concat([df, photo_bin_dummies_df], axis=1)
def map_state(state):

    if state == 41326:

        return 'Selangor'

    elif state == 41401:

        return 'Kuala_Lumpur'

    else:

        return 'Other_State'

df['State_Bin'] = df.State.apply(map_state)

state_bin_dummies_df = pd.get_dummies(df['State_Bin']).rename(columns=lambda x: 'State_' + str(x))

df = pd.concat([df, state_bin_dummies_df], axis=1)
rescuer_dict = df.RescuerID.value_counts().to_dict()

df['Rescuer_Num'] = df.RescuerID.map(rescuer_dict)

#df['Rescuer_Bin']=pd.factorize(pd.cut(df.Rescuer_Num,bins=[1,2,5],right=False))[0]

#df['Rescuer_Bin'].value_counts()

#rescuer_bin_dummies_df = pd.get_dummies(df['Rescuer_Bin']).rename(columns=lambda x: 'Rescuer_Bin_' + str(x))

#df = pd.concat([df, rescuer_bin_dummies_df], axis=1)
breeds = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')

breeds_dict = {k: v for k, v in zip(breeds['BreedID'], breeds['BreedName'])}

df['Breed1_name'] = df['Breed1'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'NA')

df['Breed2_name'] = df['Breed2'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'NA')
df['Breed'] = df['Breed1_name'] + '--' + df['Breed2_name']

def mix_breed(string):

    breed = string.split('--')

    if breed[0] in ['Mixed_Breed','NA']:

        return 1

    elif breed[1] == 'Mixed_Breed':

        return 1

    elif breed[1] == 'NA':

        return 0

    elif breed[0] != breed[1]:

        return 1

    else:

        return 0

df['Mixed_Breed'] = df.Breed.apply(mix_breed)

df[df.Mixed_Breed == 0].Breed.value_counts()
df.Description[df.Description.isnull()] = ''

des_list = df.Description.values.tolist()
import unicodedata

import re



from nltk import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from nltk.stem import LancasterStemmer, WordNetLemmatizer

from nltk.corpus import stopwords



def remove_non_ascii(words):

    """Remove non-ASCII characters from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        new_words.append(new_word)

    return new_words



def to_lowercase(words):

    """Convert all characters to lowercase from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = word.lower()

        new_words.append(new_word)

    return new_words



def remove_punctuation(words):

    """Remove punctuation from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = re.sub(r'[^\w\s]', '', word)

        if new_word != '':

            new_words.append(new_word)

    return new_words



'''

def replace_numbers(words):

    """Replace all interger occurrences in list of tokenized words with textual representation"""

    p = inflect.engine()

    new_words = []

    for word in words:

        if word.isdigit():

            new_word = p.number_to_words(word)

            new_words.append(new_word)

        else:

            new_words.append(word)

    return new_words 

'''



def remove_stopwords(words):

    """Remove stop words from list of tokenized words"""

    new_words = []

    for word in words:

        if word not in stopwords.words('english'):

            new_words.append(word)

    return new_words



def stem_words(words):

    """Stem words in list of tokenized words"""

    stemmer = LancasterStemmer()

    stems = []

    for word in words:

        stem = stemmer.stem(word)

        stems.append(stem)

    return stems



def lemmatize_verbs(words):

    """Lemmatize verbs in list of tokenized words"""

    lemmatizer = WordNetLemmatizer()

    lemmas = []

    for word in words:

        lemma = lemmatizer.lemmatize(word, pos='v')

        lemmas.append(lemma)

    return lemmas



def normalize(words):

    words = remove_non_ascii(words)

    words = to_lowercase(words)

    words = remove_punctuation(words)

    #words = replace_numbers(words)

    words = remove_stopwords(words)

    #words = stem_words(words)

    words = lemmatize_verbs(words)

    return words



'''

'''

word_bag = []



for i,item in enumerate(des_list):

    words = word_tokenize(item)

    words = normalize(words)

    word_bag.append(words)

df['Word_bag'] = word_bag



def wordjoin(x):

    return ' '.join(x)



df['Word_list'] = df['Word_bag'].apply(wordjoin)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



vectorizer = CountVectorizer(min_df = 0.02)

transformer=TfidfTransformer()

tfidf = transformer.fit_transform(vectorizer.fit_transform(df.Word_list))

weight=tfidf.toarray()

from sklearn.decomposition import PCA



n_components = 25

pca = PCA(n_components=n_components, random_state=42)

pca.fit(weight)

text_feature = pca.transform(weight)



columns = []

for i in range(n_components):

    columns.append('text_feature_'+str(i+1))
df = pd.concat([df,pd.DataFrame(text_feature, index = df.index, columns = columns)],axis = 1)
df.head()
df_copy = df.drop(columns=['Description','Fee','Fee_per_pet','Name','PhotoAmt','Quantity','RescuerID','State','State_Bin','Fee_Bin','Quantity_Bin','PhotoAmt_Bin','Breed','Breed1_name','Breed2_name','Word_bag','Word_list'])



train = df_copy[df.AdoptionSpeed.notnull()]

test  = df_copy[df.AdoptionSpeed.isnull()]

print(train.shape, test.shape)
train.head()
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.metrics import cohen_kappa_score, make_scorer



kappa_scorer = make_scorer(cohen_kappa_score,weights='quadratic')



X_train = train.drop(columns = ['AdoptionSpeed'])

Y_train = train['AdoptionSpeed']

X_test = test.drop(columns = ['AdoptionSpeed'])
def test_rf_model(n_splits,params):

    X = train.drop(columns=['AdoptionSpeed'])

    y = train.AdoptionSpeed

    kf = KFold(n_splits=n_splits, shuffle=True)

    kf.get_n_splits([X,y])

    score = []



    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        y_train, y_test = y[train_index], y[test_index]

        rf = RandomForestClassifier(random_state=42)

        rf.set_params(**params)

        rf.fit(X_train,y_train)

        y_test = y_test.astype(np.int32)

        y_learned = rf.predict(X_train).astype(np.int32)

        y_predict = rf.predict(X_test).astype(np.int32)

        print('On training set: ', cohen_kappa_score(y_train,y_learned,weights='quadratic'))

        score.append(cohen_kappa_score(y_test,y_predict,weights='quadratic'))

        print('On testing set: ', score[-1])

    print('The final score: ', np.mean(score))

    return rf,np.mean(score)
def test_gb_model(n_splits,params):

    X = train.drop(columns=['AdoptionSpeed'])

    y = train.AdoptionSpeed

    kf = KFold(n_splits=n_splits, shuffle=True)

    kf.get_n_splits([X,y])

    score = []

    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        y_train, y_test = y[train_index], y[test_index]

        gb = GradientBoostingClassifier(random_state=42)

        gb.set_params(**params)

        gb.fit(X_train,y_train)

        y_test = y_test.astype(np.int32)

        y_learned = gb.predict(X_train).astype(np.int32)

        y_predict = gb.predict(X_test).astype(np.int32)

        print('On training set: ', cohen_kappa_score(y_train,y_learned,weights='quadratic'))

        score.append(cohen_kappa_score(y_test,y_predict,weights='quadratic'))

        print('On testing set: ', score[-1])

    print('The final score: ', np.mean(score))

    return gb,np.mean(score)
def test_ada_model(n_splits,params):

    X = train.drop(columns=['AdoptionSpeed'])

    y = train.AdoptionSpeed

    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)

    kf.get_n_splits([X,y])

    score = []



    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        y_train, y_test = y[train_index], y[test_index]

        ada = AdaBoostClassifier(random_state=42)

        ada.set_params(**params)

        ada.fit(X_train,y_train)

        y_test = y_test.astype(np.int32)

        y_learned = ada.predict(X_train).astype(np.int32)

        y_predict = ada.predict(X_test).astype(np.int32)

        print('On training set: ', cohen_kappa_score(y_train,y_learned,weights='quadratic'))

        score.append(cohen_kappa_score(y_test,y_predict,weights='quadratic'))

        print('On testing set: ', score[-1])



    print('The final score: ', np.mean(score))

    return ada,np.mean(score)
# Deep rf

rf1,rf1_score = test_rf_model(4,{'criterion': 'gini', 'max_depth': 13, 'n_estimators': 1800})

rf1_train= rf1.predict(X_train).astype(np.int32)

rf1_pred = rf1.predict(X_test).astype(np.int32)



# Shallow rf

rf2,rf2_score = test_rf_model(4,{'criterion': 'gini', 'max_depth': 10, 'n_estimators': 1500})

rf2_train= rf2.predict(X_train).astype(np.int32)

rf2_pred = rf2.predict(X_test).astype(np.int32)



# Deep gb

gb1,gb1_score = test_gb_model(4,{'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5})

gb1_train= gb1.predict(X_train).astype(np.int32)

gb1_pred = gb1.predict(X_test).astype(np.int32)



# Shallow gb

gb2,gb2_score = test_gb_model(4,{'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 4})

gb2_train= gb2.predict(X_train).astype(np.int32)

gb2_pred = gb2.predict(X_test).astype(np.int32)
# Deep ada

ada1,ada1_score = test_ada_model(4,{'base_estimator':DecisionTreeClassifier(max_depth=4),'n_estimators': 150, 'learning_rate':0.05})

ada1_train= ada1.predict(X_train).astype(np.int32)

ada1_pred = ada1.predict(X_test).astype(np.int32)



# Shallow ada

ada2,ada2_score = test_ada_model(4,{'base_estimator':DecisionTreeClassifier(max_depth=3),'n_estimators': 150, 'learning_rate':0.05})

ada2_train= ada2.predict(X_train).astype(np.int32)

ada2_pred = ada2.predict(X_test).astype(np.int32)
train_list= [rf1_train,rf2_train,gb1_train,gb2_train,ada1_train,ada2_train]

pred_list = [rf1_pred,rf2_pred,gb1_pred,gb2_pred,ada1_pred,ada2_pred]
prediction = pd.DataFrame({'PetID': test.index})



for item in pred_list:

    prediction = pd.concat([prediction,pd.DataFrame({'AdoptionSpeed': item})],axis=1,ignore_index=True)

prediction.set_index(0,inplace=True)
validation = pd.DataFrame({'PetID': train.index})



for item in train_list:

    validation = pd.concat([validation,pd.DataFrame({'AdoptionSpeed': item})],axis=1,ignore_index=True)

validation.set_index(0,inplace=True)

validation['AdoptionSpeed'] = train['AdoptionSpeed']
rf = RandomForestClassifier(random_state=42,max_depth=6,n_estimators=100)

rf.fit(validation[[x for x in range(1,7)]],validation['AdoptionSpeed'])

prediction['AdoptionSpeed'] = rf.predict(prediction).astype(np.int32)
submission = prediction.drop(columns=[x for x in range(1,7)])

submission['PetID'] = submission.index

submission.reset_index(inplace=True)

submission = submission[['PetID','AdoptionSpeed']]

submission.head()
submission.to_csv('submission.csv', index=False)