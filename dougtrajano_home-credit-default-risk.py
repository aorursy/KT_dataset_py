# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Graphics

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

from plotly.offline import init_notebook_mode, iplot

import plotly.offline as py

import plotly.graph_objs as go

import plotly.offline as offline

py.init_notebook_mode(connected=True)

init_notebook_mode(connected=True)

offline.init_notebook_mode()

%matplotlib inline



# Sklearn and TensorFlow

from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report



# Models

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.utils.testing import all_estimators

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

import tensorflow as tf



import warnings

warnings.filterwarnings("ignore",category=RuntimeWarning)

warnings.filterwarnings("ignore",category=DeprecationWarning)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_application_train = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")

print("Shape:", df_application_train.shape)

df_application_train.head()
df_temp = df_application_train["TARGET"].value_counts()



trace = go.Pie(

    labels = df_temp.index,

    values = df_temp.values,

)



data = [trace]



layout = go.Layout(

    title = "Loan Repayed or not",

    xaxis=dict(

        tickfont=dict(

            size=14,

            color='rgb(107, 107, 107)'

        )

    ),

    yaxis=dict(

        titlefont=dict(

            size=16,

            color='rgb(107, 107, 107)'

        ),

        tickfont=dict(

            size=14,

            color='rgb(107, 107, 107)'

        )

)

)



fig = go.Figure(data=data, layout=layout)

fig.update_layout(template="seaborn")

py.iplot(fig)
df_temp = df_application_train["NAME_CONTRACT_TYPE"].value_counts()



fig = {

  "data": [

    {

      "values": df_temp.values,

      "labels": df_temp.index,

      "domain": {"x": [0, .48]},

      "hole": .7,

      "type": "pie"

    },

    

    ],

  "layout": {

        "title":"Contract type",

    }

}



fig = go.Figure(fig)

fig.update_layout(template="seaborn")

iplot(fig)
df_temp = df_application_train["NAME_FAMILY_STATUS"].value_counts()



fig = {

  "data": [

    {

      "y": df_temp.values,

      "x": df_temp.index,

      "type": "bar"

    },

    

    ],

  "layout": {

        "title":"Family Status",

    }

}



fig = go.Figure(fig)

fig.update_layout(template="seaborn")

iplot(fig)
df_temp = df_application_train["OCCUPATION_TYPE"].value_counts()



fig = {

  "data": [

    {

        "x": df_temp.index,

        "y": df_temp.values,

        "type": "bar"

    },

    

    ],

  "layout": {

        "title":"Occupation of applicant\'s"

    }

}



fig = go.Figure(fig)

fig.update_layout(template="seaborn")

iplot(fig)
df_temp = pd.DataFrame(df_application_train.isnull().sum().sort_values(ascending=False)/len(df_application_train), columns=["MISSING_VALUES"])

df_temp = df_temp[df_temp["MISSING_VALUES"] > 0]



fig = {

  "data": [

    {

        "x": df_temp.index,

        "y": df_temp.MISSING_VALUES.values,

        "type": "bar"

    },

    

    ],

  "layout": {

        "title":"Columns with missing values (%)"

    }

}



fig = go.Figure(fig)

fig.update_layout(template="seaborn")

iplot(fig)
# Missing values

## For low number of missing values we'll apply the mean for float64 objects and "Other" for string columns.



low_missing_values_col = ["DAYS_LAST_PHONE_CHANGE", "CNT_FAM_MEMBERS", "AMT_ANNUITY", "AMT_GOODS_PRICE", "EXT_SOURCE_2",

                          "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE",

                          "OBS_30_CNT_SOCIAL_CIRCLE", "NAME_TYPE_SUITE"]



for col in low_missing_values_col:

    if df_application_train[col].dtype == "object":

        df_application_train[col].fillna("Other", inplace=True)

    elif df_application_train[col].dtype == "float64":

        df_application_train[col].fillna(df_application_train[col].mean(), inplace=True)

    else:

        print("{} has a different dtype.".format(col))

    print("{}:".format(col), df_application_train[col].dtype)
# Medium quantity

## The strategy for medium quantity of missing values will be remove rows with np.nan.



medium_missing_values_col = ["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_WEEK",

                             "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_YEAR", "AMT_REQ_CREDIT_BUREAU_QRT",

                            "EXT_SOURCE_3"]



print("df_application_train shape before remove missing rows:", df_application_train.shape)

    

for col in medium_missing_values_col:

    print(col, df_application_train[col].dtype)

    if col == "EXT_SOURCE_3":

        print("Mean:", df_application_train[col].mean(), "\n")

    else:

        print(df_application_train[col].unique(), "\n")



df_application_train.dropna(subset=medium_missing_values_col, inplace=True)

print("df_application_train shape after remove missing rows:", df_application_train.shape)
# For high quantity of missing values, we'll drop these columns

df_application_train.dropna(axis="columns", how="any", inplace=True)

df_application_train.shape
# Check missing values

missing_values = df_application_train.isnull().sum().sum()

print("Quantiy of missing values:", missing_values)
# Categorical features

df_application_train_dtypes = pd.DataFrame(df_application_train.dtypes, columns=["dtypes"])

cat_features = df_application_train_dtypes[(df_application_train_dtypes["dtypes"] == "object") | (df_application_train_dtypes["dtypes"] == "category")].index.tolist()

print("Categorical features ({}):".format(len(cat_features)), cat_features)
from sklearn.preprocessing import LabelEncoder



# instantiate labelencoder object

le = LabelEncoder()



# apply le on categorical feature columns

df_application_train[cat_features] = df_application_train[cat_features].apply(lambda col: le.fit_transform(col))

df_application_train.loc[:][cat_features].head(10)
#Using Pearson Correlation

plt.figure(figsize=(12,10))

cor = df_application_train.corr()

sns.heatmap(cor, cmap=plt.cm.Reds)

plt.show()
#Correlation with TARGET variable

cor_target = abs(cor["TARGET"])



#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.05].sort_values(ascending=False)

relevant_features[1:]
# Pairplot only for relevant_features

sns.pairplot(df_application_train, hue="TARGET", vars=relevant_features.index.tolist())

plt.show()
# Split in X and Y and apply MinMaxScaler (0, 1)



X = df_application_train.drop(columns=["SK_ID_CURR", "TARGET"])

y = df_application_train["TARGET"].values



scaler = MinMaxScaler()

X = scaler.fit_transform(X.values)



print("X shape:", X.shape)
def do_pca(n_components, data):

    '''

    Transforms data using PCA to create n_components, and provides back the results of the

    transformation.



    INPUT: n_components - int - the number of principal components to create

           data - the data you would like to transform



    OUTPUT: pca - the pca object created after fitting the data

            X_pca - the transformed X matrix with new number of components

    '''

    X = StandardScaler().fit_transform(data)

    pca = PCA(n_components)

    X_pca = pca.fit_transform(X)

    return pca, X_pca



def scree_plot(pca):

    '''

    Creates a scree plot associated with the principal components 

    

    INPUT: pca - the result of instantian of PCA in scikit learn

            

    OUTPUT:

            None

    '''

    num_components=len(pca.explained_variance_ratio_)

    ind = np.arange(num_components)

    vals = pca.explained_variance_ratio_

 

    plt.figure(figsize=(10, 6))

    ax = plt.subplot(111)

    cumvals = np.cumsum(vals)

    ax.bar(ind, vals)

    ax.plot(ind, cumvals)

    #for i in range(num_components):

    #    ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)

 

    ax.xaxis.set_tick_params(width=0)

    ax.yaxis.set_tick_params(width=2, length=12)

 

    ax.set_xlabel("Principal Component")

    ax.set_ylabel("Variance Explained (%)")

    plt.title('Explained Variance Per Principal Component')

    

pca, X_pca = do_pca(X.shape[1], X)

X_pca = None #just cleaning memory

scree_plot(pca)
n_components = 40

pca, X_pca = do_pca(n_components, X)



print("Explained variance for {} components: {:.2f}%".format(n_components, sum(pca.explained_variance_ratio_)*100))
print("PCA: Top 10 Explained variance", "\n")

for i in range(len(pca.explained_variance_ratio_[:10])):

    print("component {}: {:.2f}%".format(i+1, pca.explained_variance_ratio_[i]*100))
# Split the dataset in train and test.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



print("train size:", len(X_train))

print("test size:", len(X_test))
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)



class_weights = {

    0: class_weights[0],

    1: class_weights[1]

}



class_weights
print("Sklearn's classifier algorithms with class_weight attribute:", "\n")

estimators = all_estimators(type_filter='classifier')

for name, class_ in estimators:

    try:

        if hasattr(class_(), 'class_weight'): 

            print(name)

    except:

        pass
%%time

clf_log = LogisticRegression(solver="liblinear", class_weight=class_weights)

clf_log.fit(X_train, y_train)



y_pred = clf_log.predict(X_test)



print(classification_report(y_test, y_pred, digits=4))
%%time

clf_rf = RandomForestClassifier(n_estimators=50, class_weight=class_weights, n_jobs=-1, min_samples_leaf=200, max_depth=30)

clf_rf.fit(X_train, y_train)



y_pred = clf_rf.predict(X_test)



print(classification_report(y_test, y_pred, digits=4))
%%time

# GridSearch for RandomForestClassifier

model = RandomForestClassifier(class_weight=class_weights)



parameters = {

    "n_estimators": [10, 30, 50, 100],

    "criterion": ["gini", "entropy"],

    "max_depth": [10, 20, 30],

    "min_samples_leaf": [50, 100, 200]    

}



clf_rf = GridSearchCV(model, parameters, cv=5, n_jobs=-1, scoring=make_scorer(roc_auc_score), verbose=1)

clf_rf.fit(X_train, y_train)



y_pred = clf_rf.predict(X_test)

print(clf_rf.best_estimator_)

print("\n")

print(classification_report(y_test, y_pred, digits=4))
%%time

from lightgbm import LGBMClassifier

clf_lgbmc = LGBMClassifier(objective='binary', metric='auc', class_weight=class_weights, n_estimators=1000, max_depth=20)



clf_lgbmc.fit(X_train, y_train,

              eval_set = [(X_test, y_test)],

              early_stopping_rounds=50,

              verbose=0)



y_pred = clf_lgbmc.predict(X_test, num_iteration=clf_lgbmc.best_iteration_)



print(classification_report(y_test, y_pred, digits=4))
%%time

# GridSearch for LGBMClassifier

model = LGBMClassifier(class_weight=class_weights, objective='binary', metric='auc')

    

parameters = {

    "n_estimators": [100, 300, 500, 1000],

    "boosting_type": ["gbdt", "dart", "goss"],

    "max_depth": [10, 20, 30] 

}



clf_lgbmc = GridSearchCV(model, parameters, cv=5, n_jobs=-1, scoring=make_scorer(roc_auc_score), verbose=1)

clf_lgbmc.fit(X_train, y_train)



y_pred = clf_lgbmc.predict(X_test)



print(classification_report(y_test, y_pred, digits=4))
%%time

import keras.backend as K



X_train_k = X_train

y_train_k = np.array(y_train)



X_test_k = X_test

y_test_k = np.array(y_test)



def f1_keras(y_true, y_pred):

    y_pred = K.round(y_pred)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)

    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)



    precision = tp / (tp + fp + K.epsilon())

    recall = tp / (tp + fn + K.epsilon())



    f1 = 2*precision*recall / (precision+recall+K.epsilon())

    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)



# TensorFlow/Keras

clf_keras = tf.keras.models.Sequential([

    tf.keras.layers.Dense(512, input_dim=X_train_k.shape[1], activation="relu"),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(256, activation="relu"),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

    

clf_keras.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy", f1_keras])



clf_keras.fit(X_train_k, y_train_k, epochs=50, class_weight=class_weights, use_multiprocessing=True, batch_size=128)



model_evals = clf_keras.evaluate(X_test_k, y_test_k)



print("\n")

print("Evaluate Model")

print("Loss: {}".format(model_evals[0]))

print("Accuracy: {}".format(model_evals[1]))

print("F1-Score: {}".format(model_evals[2]))
clf_keras.summary()
def shame_function(y_pred):

    """

    This function update y_pred for 1 or 0.

    

    I think that have a better solution for that, but I don't find anything at this time.

    """

    result = []

    for n in y_pred:

        if n[0] >= 0.5:

            result.append(1)

        else:

            result.append(0)

    return result



y_pred = shame_function(clf_keras.predict(X_test))



print("y_test")

print("% of target == 1: {:.2f}%".format((len(y_test.tolist())/sum(y_test.tolist()))))

print(y_test.tolist()[:10])

print("\n")

print("y_pred")

print("% of target == 1: {:.2f}%".format((len(y_pred)/sum(y_pred))))

print(y_pred[:10])
def preprocessing(df, n_components=None, return_pca=False):

    # Missing values

    low_missing_values_col = ["DAYS_LAST_PHONE_CHANGE", "CNT_FAM_MEMBERS", "AMT_ANNUITY", "AMT_GOODS_PRICE", "EXT_SOURCE_2",

                              "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE",

                              "OBS_30_CNT_SOCIAL_CIRCLE", "NAME_TYPE_SUITE"]



    for col in low_missing_values_col:

        if df[col].dtype == "object":

            df[col].fillna("Other", inplace=True)

        elif df[col].dtype == "float64":

            df[col].fillna(df[col].mean(), inplace=True)





    medium_missing_values_col = ["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_WEEK",

                                 "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_YEAR", "AMT_REQ_CREDIT_BUREAU_QRT",

                                "EXT_SOURCE_3"]



    for col in medium_missing_values_col:

        if df[col].dtype == "object":

            df[col].fillna("Other", inplace=True)

        elif df[col].dtype == "float64":

            df[col].fillna(df[col].mean(), inplace=True)

            

    df.dropna(axis="columns", how="any", inplace=True)

    

    # Split in X and Y

    X = df.dropna(axis="columns", how="any")

    X.drop(columns=["SK_ID_CURR", "TARGET"], inplace=True, errors="ignore")



    try:

        y = df["TARGET"].values

    except:

        y = None

        print("TARGET column not found.")

    

    X_dtypes = pd.DataFrame(X.dtypes, columns=["dtypes"])

    cat_features = X_dtypes[(X_dtypes["dtypes"] == "object") | (X_dtypes["dtypes"] == "category")].index.tolist()

    

    # instantiate labelencoder object

    le = LabelEncoder()

    # apply le on categorical feature columns

    X[cat_features] = X[cat_features].apply(lambda col: le.fit_transform(col))

    X.loc[:][cat_features].head(10)



    scaler = MinMaxScaler()

    X = scaler.fit_transform(X.values)



    if n_components == None:

        n_components = X.shape[1]

        

    pca, X_pca = do_pca(n_components, X)

    

    if y != None and return_pca != False:

        return X_pca, y

    elif y != None and return_pca == True:

        return X_pca, y, pca

    elif y == None and return_pca == True:

        return X_pca, pca

    else:

        return X_pca



def submission_file(model, keras_model=False, n_components=70, filename="submission.csv"):

    """

    Args

    

    model: Model to predict

    keras_model: True/False if we are using Keras Model or not.

    n_components: To preprocessing PCA.

    

    Return

    Link to submission file download.

    """

    from IPython.display import FileLink

    

    df_application_test = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")

    submission = preprocessing(df_application_test, n_components=n_components)



    if keras_model:

        predicts = shame_function(model.predict(submission))

    else:

        predicts = model.predict(submission)



    df_sample_submission = pd.read_csv("/kaggle/input/home-credit-default-risk/sample_submission.csv")

    df_sample_submission["TARGET"] = predicts



    df_sample_submission.to_csv(filename, index=False)

    print("{} salved.".format(filename))

    return FileLink(r'{}'.format(filename))
# LogisticRegression

submission_file(clf_log, n_components=70,

                filename="submission_LogisticRegression.csv")
# RandomForest

submission_file(clf_rf, n_components=70,

                filename="submission_RandomForest.csv")
# LGBMClassifier

submission_file(clf_lgbmc, n_components=70,

                filename="submission_LGBMClassifier.csv")
# TensorFlow/Keras

submission_file(clf_keras, keras_model=True, n_components=70,

                filename="submission_TensorFlow-Keras.csv")