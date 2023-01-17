import pandas as pd

import numpy as np

import json

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm

from sklearn import metrics

import hypertools as hyp

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier, DMatrix, cv

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV
pets = pd.read_csv('../input/train/train.csv')

pets2 = pd.read_csv('../input/test/test.csv')
petID = np.asarray(pets2.PetID)
def add_pure(dataframe):

    #create new column and set values to 0 by default

    dataframe['pure_bred'] = 0



    #For loop to check if pet is pure bred and change the values accordingly

    for i in range(0,(len(pets))):

        try:

            if dataframe.Breed2[i] == 0 or (dataframe.Breed1[i] == dataframe.Breed2[i]):

                dataframe.iat[i, dataframe.columns.get_loc('pure_bred')] = 1

        except:

            continue

    return (dataframe)

    #pets.pure_bred.value_counts(dropna = False)

#'../input/test/test_sentiment/'

def add_sentiments(dataframe, path):

    dataframe['sentiment_score'] = float(0)

    dataframe['sentiment_magnitude'] = float(0)



    for i in range(0,(len(dataframe))):

        filename = path + dataframe.PetID[i] + '.json'

        try:

            with open(filename, 'r') as f:

                sentiments = json.load(f)

                dataframe.iat[i, dataframe.columns.get_loc('sentiment_score')] = sentiments['documentSentiment']['score']

                dataframe.iat[i, dataframe.columns.get_loc('sentiment_magnitude')] = sentiments['documentSentiment']['magnitude']

        except FileNotFoundError:

            continue

    return (dataframe)

    #dataframe.sentiment_score.value_counts(dropna = False)

    

def create_cats(dataframe):

    for i in dataframe.columns:

        #if i is not dataframe.columns[2] and i is not dataframe.columns[10] and i is not dataframe.columns[15] and i is not dataframe.columns[18] and i is not dataframe.columns[22] and i is not dataframe.columns[25] and i is not dataframe.columns[26]:

        if i not in ['Age', 'AdoptionSpeed', 'FurLength', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'sentiment_score', 'sentiment_magnitude', 'desc_len', 'vertex_x', 'vertex_y','bounding_confidence', 'bounding_importance','dominant_blue', 'dominant_green', 'dominant_red', 'dominant_pixel_frac', 'dominant_score', 'label_score']:

            dataframe[i] = dataframe[i].astype('category')

            #dataframe[i] = pd.get_dummies(dataframe[i])



def sentiment_to_float(dataframe):

    dataframe.sentiment_score = dataframe.sentiment_score.astype(float)

    dataframe.sentiment_magnitude = dataframe.sentiment_magnitude.astype(float)
#code taken from  Bojan Tunguz; who forked Abhishek's kernel (Taken on 31 Jan, 2019)



def image_data(dataframe, ext):

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

    for pet in dataframe.PetID:

        try:

            with open(ext + pet + '-1.json', 'r') as f:

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



#    print(nf_count)

#    print(nl_count)

    dataframe.loc[:, 'vertex_x'] = vertex_xs

    dataframe.loc[:, 'vertex_y'] = vertex_ys

    dataframe.loc[:, 'bounding_confidence'] = bounding_confidences

    dataframe.loc[:, 'bounding_importance'] = bounding_importance_fracs

    dataframe.loc[:, 'dominant_blue'] = dominant_blues

    dataframe.loc[:, 'dominant_green'] = dominant_greens

    dataframe.loc[:, 'dominant_red'] = dominant_reds

    dataframe.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs

    dataframe.loc[:, 'dominant_score'] = dominant_scores

    dataframe.loc[:, 'label_description'] = label_descriptions

    dataframe.loc[:, 'label_score'] = label_scores
def fillna(dataframe):

    for i in dataframe.columns:

        if i not in ['Name', 'Description', 'PetID', 'RescuerID']:

            if i not in ['Age', 'Quantity', 'PhotoAmt', 'VideoAmt', 'sentiment_score', 'sentiment_magnitude', 'Fee']:

                dataframe[i].fillna(dataframe[i].mode()[0], inplace = True)

            else:

                dataframe[i].fillna(dataframe[i].mean(), inplace = True)
def add_length(dataframe):

    dataframe['desc_len'] = 0

    for i in range(0, len(dataframe)):

        dataframe.iat[i, dataframe.columns.get_loc('desc_len')] = len(str(dataframe.Description[i]))



        

def sqrt_age(dataframe):

    dataframe.Age = np.sqrt(dataframe.Age)
def find_weights(dataframe): 

    weights = []

    count = 0

    for i in range(0,5):

        count = 0

        weight = 0.0

        for j in dataframe.AdoptionSpeed:

            if j == i:

                count += 1

        weight = count/len(dataframe.AdoptionSpeed)

        weights.append(weight)

    return weights
add_pure(pets)

add_pure(pets2)

add_sentiments(pets, '../input/train_sentiment/')

add_sentiments(pets2, '../input/test_sentiment/')

image_data(pets, '../input/train_metadata/')

image_data(pets2, '../input/test_metadata/')

pets
create_cats(pets)

create_cats(pets2)

sentiment_to_float(pets)

sentiment_to_float(pets2)
add_length(pets)

add_length(pets2)

#sqrt_age(pets)

#sqrt_age(pets2)

#fillna(pets)

#fillna(pets2)

pets = pets.drop(['Name', 'PetID', 'Description','RescuerID'], axis = 1)

pets2 = pets2.drop(['Name', 'PetID', 'Description','RescuerID'], axis = 1)

pets = pd.get_dummies(pets)

pets2 = pd.get_dummies(pets2)

#pets
pets
def mat_interactions(dataframe):

    dataframe['mat_int_1_1'] = dataframe.MaturitySize_1 * dataframe.Type_1

    dataframe['mat_int_1_2'] = dataframe.MaturitySize_2 * dataframe.Type_1

    dataframe['mat_int_1_3'] = dataframe.MaturitySize_3 * dataframe.Type_1

    dataframe['mat_int_1_4'] = dataframe.MaturitySize_4 * dataframe.Type_1

    dataframe['mat_int_2_1'] = dataframe.MaturitySize_1 * dataframe.Type_2

    dataframe['mat_int_2_2'] = dataframe.MaturitySize_2 * dataframe.Type_2

    dataframe['mat_int_2_3'] = dataframe.MaturitySize_3 * dataframe.Type_2

    dataframe['mat_int_2_4'] = dataframe.MaturitySize_4 * dataframe.Type_2
def age_interactions(dataframe):

    dataframe['age_int_1_1'] = dataframe.Age * dataframe.Health_1

    dataframe['age_int_1_2'] = dataframe.Age * dataframe.Health_2

    dataframe['age_int_1_3'] = dataframe.Age * dataframe.Health_3
mat_interactions(pets)

mat_interactions(pets2)

age_interactions(pets)

age_interactions(pets2)
sns.countplot(x='AdoptionSpeed', data = pets)
#sns.pairplot(pets, hue = 'AdoptionSpeed')
pets_vars = pets.columns.tolist()

adopt = ['AdoptionSpeed']

X=[i for i in pets_vars if i not in adopt]
k=find_weights(pets)

balance_weight = {0 :int((max(k)*len(pets))-(min(k)*len(pets)))}
train_y = pets[adopt]

#new_XX = 

smote = SMOTE(sampling_strategy = 'not majority', k_neighbors = 100)

new_XX, train_y = smote.fit_resample(pets[X], train_y.values.ravel())



#smote = SMOTE(sampling_strategy = balance_weight, k_neighbors = 20,)

#pets[X], pets[adopt] = smote.fit_resample(np.asarray(pets[X]), np.asarray(pets[adopt].values.ravel()))
sns.countplot(x=train_y)
#pca = PCA(n_components=3)

#fit = pca.fit(new_XX)

#print("Explained Variance: %s", fit.explained_variance_ratio_)

#print((fit.components_[0]))



model = ExtraTreesClassifier()

model.fit(new_XX, train_y)

#print(model.feature_importances_)



feature_list = {}



for i in range(0, len(model.feature_importances_)):

    feature_list.update({i:model.feature_importances_[i]})

    

#sort values

feature_list = sorted(feature_list.items(), key=lambda kv: kv[1], reverse = True)
#feature_list[0:35]
j = feature_list[0:50]



new_X = []

for i in j:

    #print(i[0])

    #var = i[]

    new_X.append(X[i[0]])

    

new_X
#new_X

train_X = pd.DataFrame(data = new_XX, columns=X)

train_x = train_X[new_X]

test_x = pets2[new_X]
rf_model = RandomForestClassifier(class_weight = 'balanced', n_estimators = 1000, random_state = 42)

rf_model.fit(train_x, train_y)
#importance = list(rf_model.feature_importances_)

#importance = sorted(importance,  reverse = True)

#importance
#for i in X:

    #if i not in pets2.columns.tolist():

        #pets2[i] = 0

        #print(i)

#pets2
# pred = rf_model.predict(test_x)
#test_x.info()
# count = 1

# # results_array = np.array([0,0,0])

# for i in np.linspace(0.05, 0.3, 15):

#     for j in [6,7,8,9]:

#         parameters = {'learning_rate':i, 'max_depth':j, 'objective':'multi:softmax', 'num_class':4}

#         #model = XGBClassifier(**parameters)

#         model = XGBClassifier(**parameters)

#         #clf = RandomizedSearchCV(model, param_distributions = parameters, n_iter = 25, scoring = 'f1', error_score = 0, verbose = 3, n_jobs = -1)

#         numFolds = 5

#         folds = StratifiedKFold(shuffle = True, n_splits = numFolds)

#         results = cross_val_score(model, train_x, train_y, cv=folds)

#         if count == 1:

#             results_array = np.array([[i, j, np.mean(results)]])

#             count=0

#         else:

#             results_array = np.append(results_array, [[i, j , np.mean(results)]], axis = 0)

#         print('done')

# print(results_array)
# results_array = results_array[results_array[:,2].argsort()]

# results_array
parameters = {'learning_rate':0.15, 'max_depth':6,'subsample':0.5,'objective':'multi:softmax', 'num_class':5}

model = XGBClassifier(**parameters)








# estimators = []

# results = np.zeros(len(train_x))

# score = 0.0

# for train_index, test_index in folds:

#     X_train, X_test = X[train_index], X[test_index]

#     y_train, y_test = y[train_index], y[test_index]

#     clf.fit(X_train, y_train)



#     estimators.append(clf.best_estimator_)

#     results[test_index] = clf.predict(X_test)

#     score += f1_score(y_test, results[test_index])

# score /= numFolds
np.unique(train_y, return_counts = True)
#xg_train = DMatrix(data = new_XX, label = train_y)

#parameters = {'learning_rate':np.linspace(0.05, 0.3, 10), 'max_depth':[3,4,5,6,7],

              #'subsample':np.linspace(0.35, 0.8, 9), 'eta':np.linspace(0.01, 0.3, 40), 

              #'gamma':np.arange(0,10,1), 'min_child_weight':np.arange(0,10,1), 

              #'colsample_bytree':np.linspace(0.2, 1.0, 5), 

              #'objective':'multi:softmax', 'num_class':5}



#cv(params = parameters, dtrain = xg_train,num_boost_round=10, nfold=5, stratified=True,as_pandas=True)
model.fit(train_x,train_y, eval_metric = 'merror', verbose = True)

pred_xg = model.predict(test_x)



pred_xg = np.round(pred_xg)

pred_xg = pred_xg.astype(int)
df = {'PetID': petID, 'AdoptionSpeed': pred_xg}

df = pd.DataFrame(df)

df.AdoptionSpeed = df.AdoptionSpeed.astype('int32')
df.AdoptionSpeed.value_counts(dropna= False)
df.to_csv('submission.csv', index = False)
#k = np.array([1,2,3])
#k = np.append([k], [[1,2,3]], axis = 0)
#k