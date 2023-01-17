BALANCING = False

MODEL_USE = 3

# 0 is run all model(it takes quite long)

# 1 is Model from original train dataset,

# 2 is Model from original train dataset and description,

# 3 Model with images features and above, yet to be implemented
import os

print(os.listdir("../input"))
from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import make_scorer

def kappa(y_true, y_pred):

    return cohen_kappa_score(y_true, y_pred, weights='quadratic')
import pandas as pd

import numpy as np



breeds = pd.read_csv('../input/breed_labels.csv')

colors = pd.read_csv('../input/color_labels.csv')

train = pd.read_csv('../input/train/train.csv')

test = pd.read_csv('../input/test/test.csv')

sub = pd.read_csv('../input/test/sample_submission.csv')

states = pd.read_csv('../input/state_labels.csv')



# train['dataset_type'] = 'train'

# test['dataset_type'] = 'test'
train.head(10)
train.info()
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')
train['AdoptionSpeed'].value_counts()
train['AdoptionSpeed'].value_counts().sort_index().plot('barh')

plt.title('Adoption speed classes comparison');
train['Type'].value_counts().sort_index().plot('barh')
train['Age'].describe()
plt.hist(train['Age'],bins=list(range(0,60,1)))
# Gender distribution

train['Gender'].value_counts().rename({1:'Male',2:'Female', 3:'Mixed (Group of pets)'}).plot(kind='barh')

# plt.yticks(fontsize='xx-large')

plt.title('Gender distribution', fontsize='xx-large')
states
states_to_ID = states.set_index('StateName')

state_value_counts = train['State'].value_counts(ascending=False)

state_distribution = states_to_ID['StateID'].map(state_value_counts).sort_values(ascending=False)

state_distribution

train['State'] = train['State'].replace(41401, 41326)# convert Kuala Lumpur to Selangor 
train['PhotoAmt'].describe()
train['PhotoAmt'].plot(kind='hist', 

                          bins=30, 

                          xticks=list(range(31)))

plt.title('Photo Amount distribution')

plt.xlabel('Photos')
train['VideoAmt'].describe()
train['VideoAmt'].plot(kind='hist', 

                          bins=8, 

                          xticks=list(range(9)))

plt.title('Video Amount distribution')

plt.xlabel('Video')
train['Description'] = train['Description'].fillna('')

test['Description'] = test['Description'].fillna('')

train['desc_len'] = train['Description'].apply(lambda x: len(x))
train['desc_len'].describe()
test['desc_len'] = test['Description'].apply(lambda x: len(x))

test['desc_len'].describe()
# Clean up DataFrames

# Will try to implement these into the model later

target_train = train['AdoptionSpeed']

cleaned_train = train.drop(columns=['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'])

test_pet_ID = test['PetID']

test_X = test.drop(columns=['Name', 'RescuerID', 'Description', 'PetID'])

cleaned_train.head()
target_train.isnull().values.any()
test_X.isnull().values.any()
from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import cohen_kappa_score, make_scorer

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

import xgboost as xgb

import lightgbm as lgb



seed = 42
class EnsembleModel:

    

    def __init__(self,balancing=False):

        self.balance_ratio = 5 if balancing else 1

        self.rf_model = RandomForestClassifier()

        self.lgb_model = lgb.LGBMClassifier()

        self.rand_forest_params= {

            'bootstrap': [True, False],

            'max_depth': [30,50],

            'min_samples_leaf': [20, 30],

            'min_samples_split': [10,15],

            'n_estimators': [250,300],

            'random_state' : [seed]

        }

        self.lgb_params = {'objective' : ['multi:softprob'],

              'eta' : [0.01],

              'max_depth' : [7,8],

              'num_class' : [5],

              'num_leaves':[50],

              'lambda' : [0.75],

              'reg_alpha':[1e-5, 1e-2],

              'silent': [1]

        }

        self.rf_best_param = None

        self.lgb_best_param = None

        self.columns = None

        

    

    def set_scorer(self,kappa):

        self.kappa = kappa

        self.scorer = make_scorer(kappa)

        

    def set_param(self,rf_param,lgb_param):

        self.rf_best_param = rf_param

        self.lgb_best_param = lgb_param

    

    def tune_best_param(self,x_train,y_train):

        weights_train = [self.balance_ratio if i==0 else 1 for i in y_train.tolist()]

        rf_gridsearch = GridSearchCV(estimator = self.rf_model, 

                                      param_grid = self.rand_forest_params, 

                                      cv = 5, 

                                      n_jobs = -1, 

                                      verbose = 1, 

                                      scoring=self.scorer)

        rf_gridsearch.fit(x_train, y_train, sample_weight = weights_train)

        print('tuning for rf finished')

        self.rf_model = rf_gridsearch.best_estimator_

        self.rf_best_param = rf_gridsearch.best_params_

        

        lgb_gridsearch = GridSearchCV(self.lgb_model, self.lgb_params, n_jobs=-1, 

                   cv=5, 

                   scoring=self.scorer,

                   verbose=1, refit=True)

        lgb_gridsearch.fit(x_train, y_train, sample_weight = weights_train)

        print('tuning for lgb finished')

        self.lgb_model = lgb_gridsearch.best_estimator_

        self.lgb_best_param = lgb_gridsearch.best_params_

        print('best param for rf is:')

        print(self.rf_best_param)

        print('best param for lgb is:')

        print(self.lgb_best_param)

    

    # let's try combining the 2 models together by averging

    def _avg(self,y_1,y_2):

        return np.rint((y_1 + y_2)/2.0).astype(int)



    def re_fit_with_best_param(self,X,y):

        if self.rf_best_param == None or self.lgb_best_param == None:

            print('use tune_best_param() method to get best param first')

            return

        weights_train = [self.balance_ratio if i==0 else 1 for i in y.tolist()]

        self.rf_model = RandomForestClassifier()

        self.lgb_model =  lgb.LGBMClassifier()

        self.rf_model.set_params(**self.rf_best_param)

        self.lgb_model.set_params(**self.lgb_best_param)

        self.rf_model.fit(X,y,sample_weight=weights_train)

        self.lgb_model.fit(X,y,sample_weight=weights_train)

        print('refit finished')

    

    def validate(self,x_valid, y_valid):

        rf_score = self.kappa(self.rf_model.predict(x_valid), y_valid)

        print('{} score: {}'.format('rf', round(rf_score, 4)))

        lgb_score = self.kappa(self.lgb_model.predict(x_valid), y_valid)

        print('{} score: {}'.format('lgb', round(lgb_score, 4)))

        score = kappa(self._avg(self.lgb_model.predict(x_valid), self.rf_model.predict(x_valid)) , y_valid)

        print('{} score on validation set: {}'.format('combiner', round(score, 4)))

        self.columns = x_valid.columns



    def predict(self,test_X):

        rf_result = self.rf_model.predict(test_X)

        lgb_result = self.lgb_model.predict(test_X)

        final_result = self._avg(rf_result,lgb_result)

        return final_result

    

    def get_feature_importance(self):

        rf_feature_importances = pd.DataFrame({'Feature':self.columns.tolist(),'importance':self.rf_model.feature_importances_.tolist()})

        lgb_feature_importances = pd.DataFrame({'Feature':self.columns.tolist(),'importance':self.lgb_model.feature_importances_.tolist()})

        overall_feature_importance = pd.merge(rf_feature_importances, lgb_feature_importances, on='Feature', how='outer')

        overall_feature_importance['avg_importance'] = (overall_feature_importance['importance_x'] + overall_feature_importance['importance_y'])/2

        overall_feature_importance = overall_feature_importance.sort_values(by=['avg_importance'], ascending=False)

        return overall_feature_importance

# class EnsembleModel:



#     def __init__(self,balancing=False):

#         self.balance_ratio = 5 if balancing else 1

#         self.rf_model = RandomForestClassifier()

#         self.xgb_model = xgb.XGBClassifier()

#         self.rand_forest_params= {

#             'bootstrap': [True, False],

#             'max_depth': [30,50],

#             'min_samples_leaf': [20, 30],

#             'min_samples_split': [10,15],

#             'n_estimators': [250,300],

#             'random_state' : [seed]

#         }

#         self.xgb_params = {'objective' : ['multi:softprob'],

#               'eval_metric' : ['mlogloss'],

#               'eta' : [0.01],

#               'max_depth' : [6,7],

#               'num_class' : [5],

#               'lambda' : [0.75],

#               'reg_alpha':[1e-5, 1e-2],

#               'silent': [1]

#         }

#         self.rf_best_param = None

#         self.xgb_best_param = None

#         self.columns = None

        

    

#     def set_scorer(self,kappa):

#         self.kappa = kappa

#         self.scorer = make_scorer(kappa)

        

#     def set_param(self,rf_param,xgb_param):

#         self.rf_best_param = rf_param

#         self.xgb_best_param = xgb_param

    

#     def tune_best_param(self,x_train,y_train):

#         weights_train = [self.balance_ratio if i==0 else 1 for i in y_train.tolist()]

#         rf_gridsearch = GridSearchCV(estimator = self.rf_model, 

#                                       param_grid = self.rand_forest_params, 

#                                       cv = 5, 

#                                       n_jobs = -1, 

#                                       verbose = 1, 

#                                       scoring=self.scorer)

#         rf_gridsearch.fit(x_train, y_train, sample_weight = weights_train)

#         print('tuning for rf finished')

#         self.rf_model = rf_gridsearch.best_estimator_

#         self.rf_best_param = rf_gridsearch.best_params_

        

#         xgb_gridsearch = GridSearchCV(self.xgb_model, self.xgb_params, n_jobs=-1, 

#                    cv=5, 

#                    scoring=self.scorer,

#                    verbose=1, refit=True)

#         xgb_gridsearch.fit(x_train, y_train, sample_weight = weights_train)

#         print('tuning for xfb finished')

#         self.xgb_model = xgb_gridsearch.best_estimator_

#         self.xgb_best_param = xgb_gridsearch.best_params_

#         print('best param for rf is:')

#         print(self.rf_best_param)

#         print('best param for xgb is:')

#         print(self.xgb_best_param)

    

#     # let's try combining the 2 models together by averging

#     def _avg(self,y_1,y_2):

#         return np.rint((y_1 + y_2)/2.0).astype(int)



#     def re_fit_with_best_param(self,X,y):

#         if self.rf_best_param == None or self.xgb_best_param == None:

#             print('use tune_best_param() method to get best param first')

#             return

#         weights_train = [self.balance_ratio if i==0 else 1 for i in y.tolist()]

#         self.rf_model = RandomForestClassifier()

#         self.xgb_model =  xgb.XGBClassifier()

#         self.rf_model.set_params(**self.rf_best_param)

#         self.xgb_model.set_params(**self.xgb_best_param)

#         self.rf_model.fit(X,y,sample_weight=weights_train)

#         self.xgb_model.fit(X,y,sample_weight=weights_train)

#         print('refit finished')

    

#     def validate(self,x_valid, y_valid):

#         rf_score = self.kappa(self.rf_model.predict(x_valid), y_valid)

#         print('{} score: {}'.format('rf', round(rf_score, 4)))

#         xgb_score = self.kappa(self.xgb_model.predict(x_valid), y_valid)

#         print('{} score: {}'.format('xgb', round(xgb_score, 4)))

#         score = kappa(self._avg(self.xgb_model.predict(x_valid), self.rf_model.predict(x_valid))\

#                       , y_valid)

#         print('{} score on validation set: {}'.format('combiner', round(score, 4)))

#         self.columns = x_valid.columns



#     def predict(self,test_X):

#         rf_result = self.rf_model.predict(test_X)

#         xgb_result = self.xgb_model.predict(test_X)

#         final_result = self._avg(rf_result,xgb_result)

#         return final_result

    

#     def get_feature_importance(self):

#         rf_feature_importances = pd.DataFrame({'Feature':self.columns.tolist(),'importance':self.rf_model.feature_importances_.tolist()})

#         xgb_feature_importances = pd.DataFrame({'Feature':self.columns.tolist(),'importance':self.xgb_model.feature_importances_.tolist()})

#         overall_feature_importance = pd.merge(rf_feature_importances, xgb_feature_importances, on='Feature', how='outer')

#         overall_feature_importance['avg_importance'] = (overall_feature_importance['importance_x'] + overall_feature_importance['importance_y'])/2

#         overall_feature_importance = overall_feature_importance.sort_values(by=['avg_importance'], ascending=False)

#         return overall_feature_importance

# Clean up DataFrames

# Will try to implement these into the model later

target_train = train['AdoptionSpeed']

cleaned_train = train.drop(columns=['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'])

test_pet_ID = test['PetID']

test_X = test.drop(columns=['Name', 'RescuerID', 'Description', 'PetID'])

x_train, x_valid, y_train, y_valid = train_test_split(cleaned_train, 

                                                      target_train, 

                                                      test_size=0.2, 

                                                      random_state=seed)



if MODEL_USE == 1 or MODEL_USE==0:

    first_model = EnsembleModel(balancing=True)

    first_model.set_scorer(kappa)

    first_model.tune_best_param(x_train, y_train)

    first_model.validate(x_valid, y_valid)
import json

filename = os.listdir("../input/train_sentiment")[1]

filename = "../input/train_sentiment/"+filename

with open(filename, 'r') as f:

    sentiment = json.load(f)

sentiment  
def load_desc_sentiment(path):

    all_desc_sentiment_files = os.listdir(path)

    count_file = len(all_desc_sentiment_files)

    desc_sentiment_df = pd.DataFrame(columns=['PetID','desc_senti_magnitude','desc_senti_score'])

    current_file_index = 1

    for filename in all_desc_sentiment_files:

        with open(path+filename, 'r') as f:

            sentiment_json = json.load(f)

            petID = filename.split('.')[0]

            magnitude = sentiment_json['documentSentiment']['magnitude']

            score = sentiment_json['documentSentiment']['score']

            desc_sentiment_df = desc_sentiment_df.append({'PetID': petID, 'desc_senti_magnitude':magnitude,'desc_senti_score':score}, \

                                                         ignore_index=True)

            if current_file_index % 1000 == 0 or current_file_index == count_file :

                print('current progress: %d file of %d loaded' %(current_file_index,count_file))

            current_file_index += 1

    return desc_sentiment_df
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD,PCA

tfv = TfidfVectorizer(min_df=2,  max_features=None,

        strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',

        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,

        )

tfv.fit(train['Description'])

desc_X_train =  tfv.transform(train['Description'])

desc_X_test = tfv.transform(test['Description'])

print(desc_X_train.shape)

print(desc_X_test.shape)
svd = TruncatedSVD(n_components=5)

svd.fit(desc_X_train)

# print(svd.explained_variance_ratio_.sum())

# print(svd.explained_variance_ratio_)

desc_X_train = svd.transform(desc_X_train)

desc_X_test = svd.transform(desc_X_test)

print("desc_X_train (svd):", desc_X_train.shape)

print("desc_X_test (svd):", desc_X_test.shape)
train_desc_sentiment_df = load_desc_sentiment("../input/train_sentiment/")

test_desc_sentiment_df = load_desc_sentiment("../input/test_sentiment/")
train_desc_sentiment_df['score_times_mag'] = train_desc_sentiment_df['desc_senti_magnitude'] * train_desc_sentiment_df['desc_senti_score']

test_desc_sentiment_df['score_times_mag'] = test_desc_sentiment_df['desc_senti_magnitude'] * test_desc_sentiment_df['desc_senti_score']
train_desc_sentiment_df.head(5)
desc_X_train = pd.DataFrame(desc_X_train, columns=['desc_{}'.format(i) for i in range(svd.n_components)])

desc_X_test = pd.DataFrame(desc_X_test, columns=['desc_{}'.format(i) for i in range(svd.n_components)])

train_with_desc = pd.concat([train,desc_X_train],axis=1)

test_with_desc = pd.concat([test,desc_X_test],axis=1)
target_train = train_with_desc['AdoptionSpeed']

joint_train = train_with_desc.merge(train_desc_sentiment_df, how='left',left_on=['PetID'],right_on=['PetID'])

cleaned_train = joint_train.drop(columns=['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'])

cleaned_train.fillna(0.0,inplace=True)



test_pet_ID = test_with_desc['PetID']

joint_test = test_with_desc.merge(test_desc_sentiment_df, how='left',left_on=['PetID'],right_on=['PetID'])

test_X = joint_test.drop(columns=['Name', 'RescuerID', 'Description', 'PetID'])

test_X.fillna(0.0, inplace=True)



x_train, x_valid, y_train, y_valid = train_test_split(cleaned_train, 

                                                      target_train, 

                                                      test_size=0.2, 

                                                      random_state=seed)
if MODEL_USE == 2 or MODEL_USE==0:

    second_model = EnsembleModel(balancing=True)

    second_model.set_scorer(kappa)

    second_model.tune_best_param(x_train, y_train)

    second_model.validate(x_valid,y_valid)
def add_meta_feature(path,df):

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

    pet_id = df['PetID']

    for pet in pet_id:

        try:

            with open(path + pet + '-1.json', 'r') as f:

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

    df.loc[:, 'vertex_x'] = vertex_xs

    df.loc[:, 'vertex_y'] = vertex_ys

    df.loc[:, 'bounding_confidence'] = bounding_confidences

    df.loc[:, 'bounding_importance'] = bounding_importance_fracs

    df.loc[:, 'dominant_blue'] = dominant_blues

    df.loc[:, 'dominant_green'] = dominant_greens

    df.loc[:, 'dominant_red'] = dominant_reds

    df.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs

    df.loc[:, 'dominant_score'] = dominant_scores

    df.loc[:, 'label_description'] = label_descriptions

    df.loc[:, 'label_score'] = label_scores

#     df = df.drop(['label_description'])

    return df







if MODEL_USE == 3 or MODEL_USE==0:

    train_with_meta = add_meta_feature('../input/train_metadata/', train_with_desc)

    target_train = train_with_meta['AdoptionSpeed']

    cleaned_train = train_with_meta.drop(columns=['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed', 'label_description'])

    cleaned_train.fillna(0.0,inplace=True)

    

    test_with_meta = add_meta_feature('../input/test_metadata/', test_with_desc)

    test_pet_ID = test_with_desc['PetID']

    test_X = test_with_meta.drop(columns=['Name', 'RescuerID', 'Description', 'PetID', 'label_description'])

    test_X.fillna(0.0, inplace=True)



    x_train, x_valid, y_train, y_valid = train_test_split(cleaned_train, 

                                                          target_train, 

                                                          test_size=0.2, 

                                                          random_state=seed)



x_train.head()
# Metadata:

# train_df_ids = train[['PetID']]

# train_df_metadata = pd.DataFrame(train_metadata_files)

# train_df_metadata.columns = ['metadata_filename']

# train_metadata_pets = train_df_metadata['metadata_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

# train_df_metadata = train_df_metadata.assign(PetID=train_metadata_pets)

# print(len(train_metadata_pets.unique()))



# pets_with_metadatas = len(np.intersect1d(train_metadata_pets.unique(), train_df_ids['PetID'].unique()))

# print('fraction of pets with metadata: {:.3f}'.format(pets_with_metadatas / train_df_ids.shape[0]))
if MODEL_USE == 3 or MODEL_USE==0:

    third_model = EnsembleModel(balancing=True)

    third_model.set_scorer(kappa)

    third_model.tune_best_param(x_train, y_train)

    third_model.validate(x_valid,y_valid)
model = None

if MODEL_USE == 1:

    model = first_model

if MODEL_USE == 2: 

    model = second_model

if MODEL_USE == 0 or MODEL_USE == 3: # if all 3 model is enabled, we just use the 3rd model

    pass # yet to be implemented

    model = third_model
overall_feature_importance = model.get_feature_importance()

overall_feature_importance.head(5)

overall_feature_importance.drop(['importance_x','importance_y'],axis=1).set_index('Feature').plot(kind='bar')

model.re_fit_with_best_param(cleaned_train,target_train)
final_result = model.predict(test_X)
submission_df = pd.DataFrame(data={'PetID' : test_pet_ID.tolist(), 

                                   'AdoptionSpeed' : final_result})

submission_df.head(5)
submission_df.to_csv('submission.csv', index=False)