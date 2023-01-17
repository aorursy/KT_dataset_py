import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import lightgbm as lgb

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns
tp_df = pd.read_csv("../input/recognizing-faces-in-the-wild/train_true_positive_features.csv")

tn_df = pd.read_csv("../input/recognizing-faces-in-the-wild/train_true_negative_features.csv")

df = pd.concat([tp_df, tn_df])

df = df.reset_index(drop = True)
df.shape
df.tail()
df['is_related'].value_counts()
df['age_diff'] = (df['p1_age'] - df['p2_age']).abs()



df['age_ratio'] = df['p1_age'] /df['p2_age']

df[df['age_ratio'] < 1]['age_ratio'] = df['p2_age'] / df['p1_age']



df['different_gender'] = (df['p1_gender'] - df['p2_gender']).abs()



df['same_emotion'] = 0

df.loc[df[df['p1_dominant_emotion'] == df['p2_dominant_emotion']].index, 'same_emotion'] = 1



#--------------------------------------



df['cosine_avg'] = (df['vgg_cosine'] + df['facenet_cosine'] + df['openface_cosine'])/3

df['euclidean_l2_avg'] = (df['vgg_euclidean_l2'] + df['facenet_euclidean_l2'] + df['openface_euclidean_l2'])/3



df['vgg_ratio'] = df['vgg_euclidean_l2'] / df['vgg_cosine']

df['facenet_ratio'] = df['facenet_euclidean_l2'] / df['facenet_cosine'] 

df['openface_ratio'] = df['openface_euclidean_l2'] / df['openface_cosine']



df['vgg_over_facenet_cosine'] = df['vgg_cosine'] / df['facenet_cosine']

df['vgg_over_facenet_euclidean'] = df['vgg_euclidean_l2'] / df['facenet_euclidean_l2']



df['vgg_over_openface_cosine'] = df['vgg_cosine'] / df['openface_cosine']

df['vgg_over_openface_cosine'] = df['vgg_euclidean_l2'] / df['openface_euclidean_l2']



df['facenet_over_openface_cosine'] = df['facenet_cosine'] / df['openface_cosine']

df['facenet_over_openface_euclidean'] = df['facenet_euclidean_l2'] / df['openface_euclidean_l2']
df = df.drop(columns=[ 'person1', 'person2'

                      , 'p1_age', 'p2_age', 'p1_dominant_age', 'p2_dominant_age', 'p1_gender', 'p2_gender'

                      , 'p1_dominant_emotion', 'p2_dominant_emotion'

                      , 'p1_angry', 'p2_angry'

                      , 'p1_disgust', 'p2_disgust'

                      , 'p1_fear', 'p2_fear'

                      , 'p1_happy', 'p2_happy'

                      , 'p1_sad', 'p2_sad'

                      , 'p1_surprise', 'p2_surprise'

                      , 'p1_neutral', 'p2_neutral'

                      , 'vgg_euclidean', 'facenet_euclidean', 'openface_euclidean'

                     ])
x = df.drop(columns=['is_related'])

y = df['is_related']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
x_test, x_cross_val, y_test, y_cross_val = train_test_split(x_test, y_test, test_size=0.50)
print("Distributions for train, validation and cross validation sets")

print("Train:\n",y_train.value_counts()/y_train.value_counts().sum())

print("Validation:\n",y_test.value_counts()/y_test.value_counts().sum())

print("Cross Validation:\n",y_cross_val.value_counts()/y_cross_val.value_counts().sum())
train_data = lgb.Dataset(x_train, label=y_train)

test_data = lgb.Dataset(x_test, label=y_test)
params = {

    'boosting_type': 'gbdt',

    'objective': 'multiclass',

    'num_class': 2,

    'metric': 'multi_logloss',

    'learning_rate': 0.1,

    'num_leaves': 64,

    'verbose': 2

}
model = lgb.train(params

                  , train_data

                  , valid_sets=test_data, early_stopping_rounds=50

                  , num_boost_round=500 

                 )
y_pred = model.predict(x_train)

predictions = []

for i in range(0,y_pred.shape[0]):

    predictions.append(np.argmax(y_pred[i]))



accuracy = accuracy_score(predictions, y_train)

print("accuracy on train set: ",accuracy)



cm = confusion_matrix(y_train, predictions)

print("confusion matrix: \n",cm)
y_pred = model.predict(x_test)

predictions = []

for i in range(0,y_pred.shape[0]):

    predictions.append(np.argmax(y_pred[i]))

    

accuracy = accuracy_score(predictions, y_test)

print("accuracy on test set: ",accuracy)



cm = confusion_matrix(y_test, predictions)

print("confusion matrix: \n",cm)
y_pred = model.predict(x_cross_val)



predictions = []

for i in range(0,y_pred.shape[0]):

    predictions.append(np.argmax(y_pred[i]))



accuracy = accuracy_score(predictions, y_cross_val)

print("accuracy on cross val set: ",accuracy)



cm = confusion_matrix(y_cross_val, predictions)

print("confusion matrix: \n",cm)
ax = lgb.plot_importance(model, max_num_features=10)

plt.show()



fig_size = [50, 30]

plt.rcParams["figure.figsize"] = fig_size



plt.show()
test_df = pd.read_csv("../input/recognizing-faces-in-the-wild/testset_features.csv")
tmp = test_df.drop(columns=['img_pair', 'is_related'])
tmp['age_diff'] = (tmp['p1_age'] - tmp['p2_age']).abs()



tmp['age_ratio'] = tmp['p1_age'] /tmp['p2_age']

tmp[tmp['age_ratio'] < 1]['age_ratio'] = tmp['p2_age'] / tmp['p1_age']



tmp['different_gender'] = (tmp['p1_gender'] - tmp['p2_gender']).abs()



tmp['same_emotion'] = 0

tmp.loc[tmp[tmp['p1_dominant_emotion'] == tmp['p2_dominant_emotion']].index, 'same_emotion'] = 1



#--------------------------------------



tmp['cosine_avg'] = (tmp['vgg_cosine'] + tmp['facenet_cosine'] + tmp['openface_cosine'])/3

tmp['euclidean_l2_avg'] = (tmp['vgg_euclidean_l2'] + tmp['facenet_euclidean_l2'] + tmp['openface_euclidean_l2'])/3



tmp['vgg_ratio'] = tmp['vgg_euclidean_l2'] / tmp['vgg_cosine']

tmp['facenet_ratio'] = tmp['facenet_euclidean_l2'] / tmp['facenet_cosine'] 

tmp['openface_ratio'] = tmp['openface_euclidean_l2'] / tmp['openface_cosine']



tmp['vgg_over_facenet_cosine'] = tmp['vgg_cosine'] / tmp['facenet_cosine']

tmp['vgg_over_facenet_euclidean'] = tmp['vgg_euclidean_l2'] / tmp['facenet_euclidean_l2']



tmp['vgg_over_openface_cosine'] = tmp['vgg_cosine'] / tmp['openface_cosine']

tmp['vgg_over_openface_cosine'] = tmp['vgg_euclidean_l2'] / tmp['openface_euclidean_l2']



tmp['facenet_over_openface_cosine'] = tmp['facenet_cosine'] / tmp['openface_cosine']

tmp['facenet_over_openface_euclidean'] = tmp['facenet_euclidean_l2'] / tmp['openface_euclidean_l2']
tmp = tmp.drop(columns=['vgg_euclidean', 'facenet_euclidean', 'openface_euclidean'

                      , 'p1_age', 'p2_age'

                      , 'p1_dominant_age', 'p2_dominant_age'

                      , 'p1_gender', 'p2_gender'

                      , 'p1_dominant_emotion', 'p2_dominant_emotion'

                      , 'p1_angry', 'p2_angry'

                      , 'p1_disgust', 'p2_disgust'

                      , 'p1_fear', 'p2_fear'

                      , 'p1_happy', 'p2_happy'

                      , 'p1_sad', 'p2_sad'

                      , 'p1_surprise', 'p2_surprise'

                      , 'p1_neutral', 'p2_neutral'

                       ])
tmp.shape[1] == x_train.shape[1]
predictions = model.predict(tmp)
prediction_classes = []

for i in predictions:

    #prediction_classes.append(np.argmax(i))

    

    is_related = i[1]

    prediction_classes.append(is_related)
test_df['is_related'] = prediction_classes
result_set = test_df[['img_pair', 'is_related']]
result_set.head()
result_set.to_csv("submission.csv", index=False)