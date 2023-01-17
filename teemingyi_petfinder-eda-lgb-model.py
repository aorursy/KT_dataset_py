import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
print(os.listdir("../input"))
train_df = pd.read_csv('../input/train/train.csv')
train_df.info()
test_df = pd.read_csv('../input/test/test.csv')
test_df.info()
train_df.head()
train_df.Type.value_counts().plot(kind='bar')
train_df.Name.value_counts().head()
train_df.Age.plot(kind='hist')
plt.figure(figsize=(20,10))
sns.boxplot(x='AdoptionSpeed', y='Age', data=train_df, showfliers=False)
breed_label = pd.read_csv('../input/breed_labels.csv')
breed_label.head()
train_df.Breed1.value_counts().reset_index().join(breed_label.set_index('BreedID'),on='index').rename(columns={'index':'Breed1','Breed1':'Count'}).head()
train_df.Breed2.value_counts().reset_index().join(breed_label.set_index('BreedID'),on='index').rename(columns={'index':'Breed2','Breed2':'Count'}).head()
train_df.Breed2.loc[train_df.Breed1==train_df.Breed2] = 0
train_df['Mixed_Breed'] = train_df.apply(lambda x: 0 if x.Breed2==0 and x.Breed1!=307 else 1, axis=1)
test_df['Mixed_Breed'] = test_df.apply(lambda x: 0 if x.Breed2==0 and x.Breed1!=307 else 1, axis=1)
train_df.Mixed_Breed.value_counts().plot(kind='bar')
color_label = pd.read_csv('../input/color_labels.csv')
color_label
train_df.Color1.value_counts().reset_index().join(color_label.set_index('ColorID'),on='index').rename(columns={'index':'Color1','Color1':'Count'})
train_df.Color2.value_counts().reset_index().join(color_label.set_index('ColorID'),on='index').rename(columns={'index':'Color2','Color2':'Count'})
train_df.Color3.value_counts().reset_index().join(color_label.set_index('ColorID'),on='index').rename(columns={'index':'Color3','Color3':'Count'})
train_df['Num_Color'] = train_df.apply(lambda x:  3-sum([y==0 for y in [x.Color1, x.Color2, x.Color3]]), axis=1)
test_df['Num_Color'] = test_df.apply(lambda x:  3-sum([y==0 for y in [x.Color1, x.Color2, x.Color3]]), axis=1)
train_df.Num_Color.value_counts().plot(kind='bar')
train_df.MaturitySize.value_counts().plot(kind='bar')
train_df.FurLength.value_counts().plot(kind='bar')
train_df.Vaccinated.value_counts().plot(kind='bar')
train_df.Dewormed.value_counts().plot(kind='bar')
train_df.Sterilized.value_counts().plot(kind='bar')
train_df.Health.value_counts().plot(kind='bar')
train_df.Quantity.value_counts().plot(kind='bar')
train_df.Fee.plot(kind='hist')
state_label = pd.read_csv('../input/state_labels.csv')
state_label
train_df.State.value_counts().reset_index().join(state_label.set_index('StateID'),on='index').rename(columns={'index':'State','State':'Count'})
train_df.RescuerID.value_counts().head(10).plot(kind='bar')
train_df.VideoAmt.value_counts().plot(kind='bar')
train_df.PhotoAmt.value_counts().plot(kind='bar')
plt.figure(figsize=(20,10))
sns.boxplot(x='AdoptionSpeed', y='PhotoAmt', data=train_df)
train_df.AdoptionSpeed.value_counts().sort_index().plot(kind='bar')
train_df['Description'].fillna("", inplace=True)
test_df['Description'].fillna("", inplace=True)
train_df['Description_Length'] = train_df.Description.map(len)
test_df['Description_Length'] = test_df.Description.map(len)
plt.figure(figsize=(20,10))
sns.boxplot(x='AdoptionSpeed', y='Description_Length', data=train_df, showfliers=False)
sentiment_list = os.listdir('../input/train_sentiment')
sentiment = {}
for x in sentiment_list:
    sentiment[x[:9]] = pd.read_json('../input/train_sentiment/{}'.format(x), orient='index', typ='series').documentSentiment
sentiment_df = pd.DataFrame.from_dict(sentiment).transpose()
train_df = train_df.join(sentiment_df, on='PetID')
train_df.magnitude.fillna(0, inplace=True)
train_df.score.fillna(0, inplace=True)
sentiment_list_test = os.listdir('../input/test_sentiment')
sentiment_test = {}
for x in sentiment_list_test:
    sentiment_test[x[:9]] = pd.read_json('../input/test_sentiment/{}'.format(x), orient='index', typ='series').documentSentiment
sentiment_df_test = pd.DataFrame.from_dict(sentiment_test).transpose()
test_df = test_df.join(sentiment_df_test, on='PetID')
test_df.magnitude.fillna(0, inplace=True)
test_df.score.fillna(0, inplace=True)
from sklearn.feature_extraction.text import TfidfVectorizer
import re
pattern = re.compile('[\W_]+', re.UNICODE)
texts = [pattern.sub(' ', x) for x in train_df.Description]
texts_test = [pattern.sub(' ', x) for x in test_df.Description]
Tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1,3), max_features=20000,max_df=0.95,min_df=5)
train_x_tfidf_full = Tfidf.fit_transform(texts)
test_x_tfidf = Tfidf.transform(texts_test)
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=100, random_state=42)
train_x_svd = svd.fit_transform(train_x_tfidf_full)
train_df = train_df.join(pd.DataFrame(train_x_svd, columns=['svd_'+str(x) for x in np.arange(100)]))
test_x_svd = svd.transform(test_x_tfidf)
test_df = test_df.join(pd.DataFrame(test_x_svd, columns=['svd_'+str(x) for x in np.arange(100)]))
photo_list = os.listdir('../input/train_metadata')
profile_photo = [x for x in photo_list if "-1." in x]
example = pd.read_json('../input/train_metadata/{}'.format(profile_photo[0]), orient='index', typ='series')
example.labelAnnotations[0]
example.cropHintsAnnotation['cropHints'][0]
example.imagePropertiesAnnotation
def get_dominant_color(photo_json):
    example = pd.read_json('../input/train_metadata/{}'.format(photo_json), orient='index', typ='series')
    max_index = np.argmax([x['pixelFraction'] for x in example.imagePropertiesAnnotation['dominantColors']['colors']])
    result = example.imagePropertiesAnnotation['dominantColors']['colors'][max_index]['color']
    result['score'] = example.imagePropertiesAnnotation['dominantColors']['colors'][max_index]['score']
    result['pixelFraction'] = example.imagePropertiesAnnotation['dominantColors']['colors'][max_index]['pixelFraction']
    try:
        result['image_description'] = example.labelAnnotations[0]['description']
        result['image_description_score'] = example.labelAnnotations[0]['score']
        
    except AttributeError:
        result['image_description'] = -1
        result['image_description_score'] = -1
        
    result['image_confidence'] = example.cropHintsAnnotation['cropHints'][0]['confidence']
    try:
        result['image_importanceFraction'] = example.cropHintsAnnotation['cropHints'][0]['importanceFraction']
    except KeyError:
        result['image_importanceFraction'] = -1
    return result
get_dominant_color(profile_photo[1])
dominant_color_train = {}
for x in profile_photo:
    dominant_color_train[x[:9]] = get_dominant_color(x)
dominant_color_df = pd.DataFrame(dominant_color_train).transpose()
dominant_color_df.columns = ['photo_'+x for x in dominant_color_df.columns]
train_df = train_df.join(dominant_color_df, on='PetID', rsuffix='_color')
train_df[pd.DataFrame(dominant_color_df).columns.tolist()] = train_df[pd.DataFrame(dominant_color_df).columns.tolist()].fillna(-1)
photo_list_test = os.listdir('../input/test_metadata')
profile_photo_test = [x for x in photo_list_test if "-1." in x]

def get_dominant_color_test(photo_json):
    example = pd.read_json('../input/test_metadata/{}'.format(photo_json), orient='index', typ='series')
    max_index = np.argmax([x['pixelFraction'] for x in example.imagePropertiesAnnotation['dominantColors']['colors']])
    result = example.imagePropertiesAnnotation['dominantColors']['colors'][max_index]['color']
    result['score'] = example.imagePropertiesAnnotation['dominantColors']['colors'][max_index]['score']
    result['pixelFraction'] = example.imagePropertiesAnnotation['dominantColors']['colors'][max_index]['pixelFraction']
    try:
        result['image_description'] = example.labelAnnotations[0]['description']
        result['image_description_score'] = example.labelAnnotations[0]['score']
        
    except AttributeError:
        result['image_description'] = -1
        result['image_description_score'] = -1
        
    result['image_confidence'] = example.cropHintsAnnotation['cropHints'][0]['confidence']
    try:
        result['image_importanceFraction'] = example.cropHintsAnnotation['cropHints'][0]['importanceFraction']
    except KeyError:
        result['image_importanceFraction'] = -1
    return result

dominant_color_test = {}
for x in profile_photo_test:
    dominant_color_test[x[:9]] = get_dominant_color_test(x)
    
dominant_color_df_test = pd.DataFrame(dominant_color_test).transpose()
dominant_color_df_test.columns = ['photo_'+x for x in dominant_color_df_test.columns]
test_df = test_df.join(dominant_color_df_test, on='PetID', rsuffix='_color')

test_df[pd.DataFrame(dominant_color_df_test).columns.tolist()] = test_df[pd.DataFrame(dominant_color_df_test).columns.tolist()].fillna(-1)
photo_image_description = {}
for i,x in enumerate(pd.concat([train_df.photo_image_description, test_df.photo_image_description]).value_counts().index):
    photo_image_description[x] = i
    
train_df.photo_image_description = train_df.photo_image_description.map(lambda x: photo_image_description[x])
test_df.photo_image_description = test_df.photo_image_description.map(lambda x: photo_image_description[x])
import lightgbm as lgb
train_df.columns
features = [x for x in train_df.columns if x not in ['Name', 'Type','RescuerID','AdoptionSpeed','Description','PetID']]
df_train, df_val = train_test_split(train_df, test_size=0.3, random_state=420)
df_train.columns
from sklearn.metrics import cohen_kappa_score
def kappa_scorer(pred, train_data):
    length = len(train_data.get_label())
    pred_results = [[pred[x], pred[x+length*1], pred[x+length*2], pred[x+length*3], pred[x+length*4]] for x in np.arange(length)]
    
    return 'kappa', cohen_kappa_score([np.argmax(x) for x in pred_results],train_data.get_label(), weights='quadratic'), True
d_train = lgb.Dataset(df_train[features], label=df_train['AdoptionSpeed'],feature_name=features, 
                      categorical_feature=['Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'State','Mixed_Breed','photo_image_description'])
d_val = lgb.Dataset(df_val[features], label=df_val['AdoptionSpeed'], reference=d_train,feature_name=features, 
                    categorical_feature=['Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'State', 'Mixed_Breed','photo_image_description'])

params = {"objective" : "multiclass",
              "num_class": 5,
              "metric" : "None",
              "learning_rate" : 0.1,
              "bagging_seed" : 420,
              "feature_fraction" : 0.4,
              "early_stopping_rounds": 100
             }

evals_result = {}
model = lgb.train(params, d_train, num_boost_round=1000, valid_sets=[d_train, d_val], feval=kappa_scorer, evals_result=evals_result, verbose_eval=10)
d_train = lgb.Dataset(df_train[features], label=df_train['AdoptionSpeed'],feature_name=features, 
                      categorical_feature=['Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'State','Mixed_Breed','photo_image_description'])

cv_dict = lgb.cv(params, d_train, num_boost_round=1000, feval=kappa_scorer, verbose_eval=10)
from sklearn.dummy import DummyClassifier

dummy_model = DummyClassifier(random_state=1)
dummy_model.fit(df_train[features], df_train['AdoptionSpeed'])
cohen_kappa_score(dummy_model.predict(df_val[features]), df_val['AdoptionSpeed'])
fig, ax = plt.subplots(figsize=(12,10))
lgb.plot_importance(model, height=0.8, ax=ax)
ax.grid(False)
plt.ylabel('Feature', size=12)
plt.xlabel('Importance', size=12)
plt.title("Importance of the Features of LightGBM Model", fontsize=15)
plt.show()
pred_test = model.predict(test_df[features], num_iteration=model.best_iteration)
submission = pd.concat([test_df.PetID,pd.DataFrame(pred_test, columns=['A','B','C','D','E'])],axis=1)
submission['AdoptionSpeed'] = submission.apply(lambda x: np.argmax([x.A,x.B,x.C,x.D,x.E]), axis=1)
submission = submission[['PetID','AdoptionSpeed']]
submission.head()
submission.to_csv('submission.csv',index=False)
submission.AdoptionSpeed.value_counts()
