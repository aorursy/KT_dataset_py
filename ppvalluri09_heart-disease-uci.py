import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns



df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df.head()
# To ignore nasty warnings

def warn(*args, **kwargs):

    pass

import warnings

warnings.warn = warn
# Checking for distribution of values in the dataframe

colors = ['r', 'g', 'b', 'c', 'm', 'y'] * 3

plt.figure(figsize=(32, 8))

for i in range(13):

    if df.columns[i] not in ['cp', 'sex', 'restecg', 'fbs', 'exang', 'slope', 'ca']:

        plt.subplot(2,7, i + 1)

        sns.distplot(df.iloc[:, i], color=colors[i])

plt.plot()

# We check for box or dist plots to check whether data is normalized or not
df.isna().sum()

# Our dataset does not have any null values, which is good
# df['cp'][df['cp'] == 1] = 'typical angina'

# df['cp'][df['cp'] == 2] = 'atypical angina'

# df['cp'][df['cp'] == 3] = 'non-anginal pain'

# df['cp'][df['cp'] == 4] = 'asymptomatic'



# df = pd.get_dummies(df, drop_first=True)

# print(df.shape)

# df.head()
corr = df.corr().round(2)

plt.figure(figsize=(16, 16))

sns.heatmap(corr, annot=True)

plt.show()



# though from correlation heatmap we see that some features have very less correlation with the target

# we know from personal knowledge that those features are a crucial factors in deciding whether a person has heart disease

# or not, therefore we won't be dropping any columns
# Splitting the training features and labels

y = df['target']

train = df.copy()

train.drop(['target'], axis=1, inplace=True)

dropped_categorical_features = train['cp']

train.drop(['cp'], axis=1, inplace=True)

# print(y.head())
# Feature scaling the data using StandardScaler (also can use MinMaxScaler)

from sklearn.preprocessing import StandardScaler

scl = StandardScaler()

train = pd.DataFrame(scl.fit_transform(train), columns=train.columns)

train.head()
# adding the categorical values to the feature set

train['cp'] = dropped_categorical_features

train['cp'][train['cp'] == 1] = 'typical angina'

train['cp'][train['cp'] == 2] = 'atypical angina'

train['cp'][train['cp'] == 3] = 'non-anginal pain'

train['cp'][train['cp'] == 4] = 'asymptomatic'



train = pd.get_dummies(train, drop_first=True)

train.head()
# Splitting our data into training and testing data

from sklearn.model_selection import train_test_split

x_t, x_v, y_t, y_v = train_test_split(train.values, y, test_size=0.25, random_state=42)

print('Train Data Size', x_t.shape, y_t.shape)

print('Test Data Shape', x_v.shape, y_v.shape)
# Function to train our model



def train_model(model, x_t, y_t, x_v=None, y_v=None):

    model.fit(x_t, y_t)

    print('Training Accuracy', model.score(x_t, y_t))

    

    if x_v is not None:

        print('Validation Accuracy', model.score(x_v, y_v))

        print('F1_Score', f1_score(model.predict(x_v), y_v))

        print('Confusion Matrix\n', confusion_matrix(y_v, model.predict(x_v)))

    

    return model
# We are using different models for training our data

# XGBoost, Logistic, DecisionTrees, KMeans

from sklearn.metrics import f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier



xgb = XGBClassifier(n_estimators=500, max_depth=5,learning_rate=0.1,scale_pos_weight=1.4266790777602751)

lgs = LogisticRegression(n_jobs=1)

dtc = DecisionTreeClassifier()

rfc = RandomForestClassifier(n_estimators=100)

mlp = MLPClassifier(hidden_layer_sizes=(32, 32, 16))



print('XGBoost...')

xgb = train_model(xgb, x_t, y_t, x_v, y_v)



print('Logistic Regression...')

lgs = train_model(lgs, x_t, y_t, x_v, y_v)



print('Decision Tree...')

dtc = train_model(dtc, x_t, y_t, x_v, y_v)



print('Random Forest...')

rfc = train_model(rfc, x_t, y_t, x_v, y_v)



print('MLP...')

mlp = train_model(mlp, x_t, y_t, x_v, y_v)
# Though f1_scores are higher for Random Forest and Logistic Regression the FNs are higher for these, so we will

# apply grid search to reduce the FNs, since recall is more important in this case.
# We can see that the training accuracy is close to 1 or 1 in some cases but the validation accuracy is pretty bad

# implying a classical case of overfitting, hence we actually included features which donot correlate much with 

# the label, hence let's use a process known as Information Gain
%matplotlib inline



def train_model_with_info_gain(model, x_t, y_t, x_v, y_v):

    scores = []

    included = [0]

    model.fit(x_t[:, included], y_t)

    score_prev = model.score(x_v[:, included], y_v)

    scores.append(score_prev)

    for i in range(1, x_t.shape[1]):

        data = x_t[:, included + [i]]

        model.fit(data, y_t)

        score = model.score(x_v[:, included + [i]], y_v)

        scores.append(score)

        if score > score_prev:

            score_prev = score

            included.append(i)

    

    plt.figure(figsize=(8, 4))

    plt.plot(scores, 'yo--', alpha=0.7)

    plt.xticks(range(x_t.shape[1]))

    plt.xlabel('Columns Included')

    plt.ylabel('Validation Scores')

    model.fit(x_t[:, included], y_t)

    print('Validation Score', model.score(x_v[:, included], y_v))

    return (model, included)
from sklearn.metrics import f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier



xgb = XGBClassifier(n_estimators=500, max_depth=5,learning_rate=0.1,scale_pos_weight=1.4266790777602751)

lgs = LogisticRegression(n_jobs=1, solver='lbfgs')

dtc = DecisionTreeClassifier()

rfc = RandomForestClassifier(n_estimators=100)

mlp = MLPClassifier(hidden_layer_sizes=(32, 32, 16))



print('\nXGBoost...')

xgb, included_xgb = train_model_with_info_gain(xgb, x_t, y_t, x_v, y_v)

print('Columns Included', included_xgb)

print('F1 Score', f1_score(xgb.predict(x_v[:, included_xgb]), y_v))



print('\nLogistic Regression...')

lgs, included_lgs = train_model_with_info_gain(lgs, x_t, y_t, x_v, y_v)

print('Columns Included', included_lgs)

print('F1 Score', f1_score(lgs.predict(x_v[:, included_lgs]), y_v))



print('\nDecision Tree...')

dtc, included_dtc = train_model_with_info_gain(dtc, x_t, y_t, x_v, y_v)

print('Columns Included', included_dtc)

print('F1 Score', f1_score(dtc.predict(x_v[:, included_dtc]), y_v))



print('\nRandom Forest...')

rfc, included_rfc = train_model_with_info_gain(rfc, x_t, y_t, x_v, y_v)

print('Columns Included', included_rfc)

print('F1 Score', f1_score(rfc.predict(x_v[:, included_rfc]), y_v))



print('\nNeural Networks...')

mlp, included_mlp = train_model_with_info_gain(mlp, x_t, y_t, x_v, y_v)

print('Columns Included', included_mlp)

print('F1 Score', f1_score(mlp.predict(x_v[:, included_mlp]), y_v))
# We have an improved Validation accuracy and also the f1_score in case of every model which is a good thing, 

# moreover we now have a clear picture of which features are actually necessary which knowledge we were void of in 

# the correlation matrix, out of all these models Logistic Regression seemed to have performed well, but we have 

# another aim which is to reduce the number of False Negatives

print('XGBoost\n', confusion_matrix(xgb.predict(x_v[:, included_xgb]), y_v))

print('\nLogistic Regression\n', confusion_matrix(lgs.predict(x_v[:, included_lgs]), y_v))

print('\nDecision Tree\n', confusion_matrix(dtc.predict(x_v[:, included_dtc]), y_v))

print('\nRandom Forest\n', confusion_matrix(rfc.predict(x_v[:, included_rfc]), y_v))

print('\nNeural Networks\n', confusion_matrix(mlp.predict(x_v[:, included_mlp]), y_v))