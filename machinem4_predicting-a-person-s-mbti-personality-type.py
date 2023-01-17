import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import string

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from nltk.tokenize import word_tokenize
# Filter warnings out of outputs

import warnings

warnings.filterwarnings('ignore')
# Import Kaggle MBTI data from your local folder

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_train.info()

df_test.info()
f, ax = plt.subplots(figsize=(10, 10))

sns.countplot(df_train['type'].sort_values(ascending=False))

plt.title("Count of Personality Types")

plt.xlabel("Personality Type")

plt.ylabel("Count")
# Save the 'Id' column for later use in model predictions

df_test_Id = df_test['id']



# Now drop the 'Id' column from the base dataframe

df_test.drop("id", axis=1, inplace=True)



# Lambda expressions written to convert the personality type into the correct attribute encodings

df_train['E/I'] = df_train['type'].apply(lambda x: x[0] == 'E').astype('int')

df_train['S/N'] = df_train['type'].apply(lambda x: x[1] == 'N').astype('int')

df_train['T/F'] = df_train['type'].apply(lambda x: x[2] == 'T').astype('int')

df_train['J/P'] = df_train['type'].apply(lambda x: x[3] == 'J').astype('int')



# Check encodings

df_train.head()
# Split off personality attributes from train data into y_train for later use in the modelling section

y_train = df_train[['E/I', 'S/N', 'T/F', 'J/P']]



# Create mask varaibles for test and train subsetting later on

ntrain = df_train.shape[0]

ntest = df_test.shape[0]



# Concatenate train and test dataframes

all_data = pd.concat((df_train, df_test)).reset_index(drop=True)



# Check all data shape

print("all_data size is : {}".format(all_data.shape))
# Split posts within the posts column on the triple pipe (|||)

all_data['split_posts'] = all_data['posts'].str.split('\|\|\|')

all_data['split_posts'] = all_data['split_posts'].apply(', '.join)



# Transform all text to lowercase

all_data['split_posts'] = all_data['split_posts'].str.lower()



# Detect and replace any urls with the string 'url-web'

pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

subs_url = r'url-web'

all_data['split_posts'] = all_data['split_posts'].replace(to_replace=pattern_url, value=subs_url, regex=True)





# Create and apply a function for removing punctuation from posts

def remove_punctuation(post):

    '''

    Strips all punctuation tokens present in the string packages punctuation object from the desired colunm.



    Parameters

    ----------



    post: str

        str object containing the text to be stripped of punctuation.



    Returns

    -------



    method: remove_punctuation

        method of removing punctuations from a given dataframe column

   '''

    return ''.join([l for l in post if l not in string.punctuation])



all_data['posts_no_punct'] = all_data['split_posts'].apply(remove_punctuation)



# Tokenise the posts text into individual words

all_data['words'] = all_data['posts_no_punct'].apply(word_tokenize)



# Check preprossesing steps were successful

all_data.head()
# Subset all_data 

train_wordclouds = all_data[:ntrain]



# Group data by personality type

grouped_wordclouds = train_wordclouds[['type','words']]

grouped_wordclouds = grouped_wordclouds.groupby('type').sum()

grouped_wordclouds = grouped_wordclouds.reset_index()



# Check grouped personality type words

grouped_wordclouds.head(20)
# Instatiate figure and axis and the number of subplots to use

fig, ax = plt.subplots(nrows=4, ncols=4)

fig.set_size_inches(22, 10)



# Create a list containing all the words for all the personalities then loop through these creating a wordcloud for each one

random = grouped_wordclouds['words']

for i, j in grouped_wordclouds.iterrows():

    text = ', '.join(random[i])



    # Create and generate a word cloud image:

    wordcloud = WordCloud().generate(text)



    # Display the generated images:

    plt.subplot(4, 4, (i+1))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.title(str(grouped_wordclouds['type'].iloc[i]))
# Create a list containing all the words for all the personalities then loop through these creating a wordcloud for the total dataset

grouped_wordclouds = grouped_wordclouds['words']



vocab = []

for i in random:

    vocab.append(i)



flat_vocab = []

for sublist in vocab:

    for item in sublist:

        flat_vocab.append(item)



text = ', '.join(word for word in flat_vocab)



# Create and generate a word cloud image:

wordcloud = WordCloud().generate(text)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.title('Total vocab Wordcloud')

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer
# Create a TfidfVectorizer and apply it to the data

TFIDF_vect = TfidfVectorizer()

all_data_TFIDF = TFIDF_vect.fit_transform(all_data['posts'])



# Check the TfidfVectorizer shape

all_data_TFIDF.shape
# Create a TfidfVectorizer with better parameter usage and apply it to the data

TFIDF_vect = TfidfVectorizer(lowercase=True, stop_words='english', max_df=0.5, min_df=0.01, max_features=10000)

all_data_TFIDF = TFIDF_vect.fit_transform(all_data['posts'])



# Check the new TfidfVectorizer shape

all_data_TFIDF.shape
# Split into train and test and check that shapes match

train = all_data_TFIDF[:ntrain]

test = all_data_TFIDF[ntrain:]

print(train.shape)

print(test.shape)

print(y_train.shape)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import AdaBoostClassifier

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

import optuna
# Define fuction to calculate the log loss over 5 cross validation sets

def log_loss_cv(model, category):

    '''

    Gets the average log loss score for a model across a given number of cross validation sets.



    Parameters

    ----------



    model: model_object

        model object containing a trained sklearn model on which the score can be calculated.



    category: dataframe

        dataframe object containing the specific response variable to use as the response.



    Returns

    -------



    log_loss: int

        average log loss score for a given model and response variable.



    '''



    log_loss = -cross_val_score(model, train, y_train[category], scoring="neg_log_loss", cv=5)

    return(log_loss)
# Create base Logistic Regression models for personality attributes

logreg_EI = make_pipeline(LogisticRegression())

logreg_SN = make_pipeline(LogisticRegression())

logreg_TF = make_pipeline(LogisticRegression())

logreg_JP = make_pipeline(LogisticRegression())



# Check the cross-validation scores of the Logistic Regression base models on the train data

EI_score = log_loss_cv(logreg_EI, 'E/I')

SN_score = log_loss_cv(logreg_SN, 'S/N')

TF_score = log_loss_cv(logreg_TF, 'T/F')

JP_score = log_loss_cv(logreg_JP, 'J/P')



# Print out model score for each category

print('Extrovert/Introvert Score: ', EI_score.mean())

print('Sensing/Intuition Score: ', SN_score.mean())

print('Thinking/Feeling Score: ', TF_score.mean())

print('Judging/Percieving Score: ', JP_score.mean())
# Create base Multinomial Naive Bayes models for personality attributes

MultiNB_EI = make_pipeline(MultinomialNB())

MultiNB_SN = make_pipeline(MultinomialNB())

MultiNB_TF = make_pipeline(MultinomialNB())

MultiNB_JP = make_pipeline(MultinomialNB())



# Check the cross-validation scores of the Multinomial Naive Bayes base models on the train data

EI_score = log_loss_cv(MultiNB_EI, 'E/I')

SN_score = log_loss_cv(MultiNB_EI, 'S/N')

TF_score = log_loss_cv(MultiNB_EI, 'T/F')

JP_score = log_loss_cv(MultiNB_EI, 'J/P')



# Print out model score for each category

print('Extrovert/Introvert Score: ', EI_score.mean())

print('Sensing/Intuition Score: ', SN_score.mean())

print('Thinking/Feeling Score: ', TF_score.mean())

print('Judging/Percieving Score: ', JP_score.mean())
# Create base AdaBoost models for personality attributes

AdaB_EI = make_pipeline(AdaBoostClassifier())

AdaB_SN = make_pipeline(AdaBoostClassifier())

AdaB_TF = make_pipeline(AdaBoostClassifier())

AdaB_JP = make_pipeline(AdaBoostClassifier())



# Check the cross-validation scores of the AdaBoost base models on the train data

EI_score = log_loss_cv(AdaB_EI, 'E/I')

SN_score = log_loss_cv(AdaB_EI, 'S/N')

TF_score = log_loss_cv(AdaB_EI, 'T/F')

JP_score = log_loss_cv(AdaB_EI, 'J/P')



# Print out model score for each category

print('Extrovert/Introvert Score: ', EI_score.mean())

print('Sensing/Intuition Score: ', SN_score.mean())

print('Thinking/Feeling Score: ', TF_score.mean())

print('Judging/Percieving Score: ', JP_score.mean())
# Define an objective function to be minimized.

def objective(trial):



    # Invoke suggest methods of a Trial object to generate hyperparameters.

    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])

    tol = trial.suggest_loguniform('tol', 1e-10, 1)

    C = trial.suggest_loguniform('C', 1e-10, 1)

    random_state = trial.suggest_int('random_state', 1, 10)

    max_iter = trial.suggest_int('max_iter', 1000, 10000)

    warm_start = trial.suggest_categorical('warm_start', [True, False])



    # Create a variable containing the model and a set of selected hyperparameter values

    classifier_obj = LogisticRegression(penalty=penalty,

                                        tol=tol,

                                        C=C,

                                        random_state=random_state,

                                        max_iter=max_iter,

                                        warm_start=warm_start)



    # Define x and y variables

    x = train

    y = y_train['E/I']



    # Check cross validation score of the model based on x and y values

    score = cross_val_score(classifier_obj, x, y, scoring="neg_log_loss")

    accuracy = score.mean()



    # A objective value linked with the Trial object.

    return 1.0 - accuracy



# Create a new study and invoke optimization of the objective function

study = optuna.create_study()

study.optimize(objective, n_trials=1000)
# Used to print the optimal hyperparameters found by the objective function

study.best_params
# Run Logistic Regression models using optimised hyperparameters

logreg_EI = make_pipeline(LogisticRegression(penalty='l1',

                                             tol=0.003850701503405173,

                                             C=0.9981811566847507,

                                             random_state=1,

                                             max_iter=1762,

                                             warm_start=True))

logreg_SN = make_pipeline(LogisticRegression(penalty='l1',

                                             tol=0.003850701503405173,

                                             C=0.9981811566847507,

                                             random_state=1,

                                             max_iter=1762,

                                             warm_start=True))

logreg_TF = make_pipeline(LogisticRegression(penalty='l1',

                                             tol=0.003850701503405173,

                                             C=0.9981811566847507,

                                             random_state=1,

                                             max_iter=1762,

                                             warm_start=True))

logreg_JP = make_pipeline(LogisticRegression(penalty='l1',

                                             tol=0.003850701503405173,

                                             C=0.9981811566847507,

                                             random_state=1,

                                             max_iter=1762,

                                             warm_start=True))
# Check the cross-validation score of the model from the train data

EI_score = log_loss_cv(logreg_EI, 'E/I')

SN_score = log_loss_cv(logreg_SN, 'S/N')

TF_score = log_loss_cv(logreg_TF, 'T/F')

JP_score = log_loss_cv(logreg_JP, 'J/P')



# Print out model score for each category

print('Extrovert/Introvert Score: ' + EI_score.mean())

print('Sensing/Intuition Score: ' + SN_score.mean())

print('Thinking/Feeling Score: ' + TF_score.mean())

print('Judging/Percieving Score: ' + JP_score.mean())
# Define an objective function to be minimized.

def objective(trial):



    # Invoke suggest methods of a Trial object to generate hyperparameters.

    alpha = trial.suggest_loguniform('alpha', 1e-10, 1)

    fit_prior = trial.suggest_categorical('fit_prior', [True, False])



    # Create a variable containing the model and a set of selected hyperparameter values

    classifier_obj = MultinomialNB(alpha=alpha,

                                   fit_prior=fit_prior)



    # Define x and y variables

    x = train

    y = y_train['E/I']



    # Check cross validation score of the model based on x and y values

    score = cross_val_score(classifier_obj, x, y, scoring="neg_log_loss")

    accuracy = score.mean()



    # A objective value linked with the Trial object.

    return 1.0 - accuracy



# Create a new study and invoke optimization of the objective function

study = optuna.create_study()

study.optimize(objective, n_trials=1000)
# Used to print the optimal hyperparameters found by the objective function

study.best_params
# Run Multinomial Naive Bayes models using optimised hyperparameters

MultiNB_EI = make_pipeline(MultinomialNB(alpha=0.08874918773669986,

                                         fit_prior=True))

MultiNB_SN = make_pipeline(MultinomialNB(alpha=0.08874918773669986,

                                         fit_prior=True))

MultiNB_TF = make_pipeline(MultinomialNB(alpha=0.08874918773669986,

                                         fit_prior=True))

MultiNB_JP = make_pipeline(MultinomialNB(alpha=0.08874918773669986,

                                         fit_prior=True))
# Check the cross-validation score of the model from the train data

EI_score = log_loss_cv(MultiNB_EI, 'E/I')

SN_score = log_loss_cv(MultiNB_EI, 'S/N')

TF_score = log_loss_cv(MultiNB_EI, 'T/F')

JP_score = log_loss_cv(MultiNB_EI, 'J/P')



# Print out model score for each category

print('Extrovert/Introvert Score: ' + EI_score.mean())

print('Sensing/Intuition Score: ' + SN_score.mean())

print('Thinking/Feeling Score: ' + TF_score.mean())

print('Judging/Percieving Score: ' + JP_score.mean())
# Define an objective function to be minimized.

def objective(trial):



    # Invoke suggest methods of a Trial object to generate hyperparameters.

    n_estimators = trial.suggest_int('n_estimators', 1, 100)

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-10, 1)

    algorithm = trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R'])

    random_state = trial.suggest_int('random_state', 1, 10)



    # Create a variable containing the model and a set of selected hyperparameter values

    classifier_obj = AdaBoostClassifier(n_estimators=n_estimators,

                                        learning_rate=learning_rate,

                                        algorithm=algorithm,

                                        random_state=random_state)



    # Define x and y variables

    x = train

    y = y_train['E/I']



    # Check cross validation score of the model based on x and y values

    score = cross_val_score(classifier_obj, x, y, scoring="neg_log_loss")

    accuracy = score.mean()



    # A objective value linked with the Trial object.

    return 1.0 - accuracy



# Create a new study and invoke optimization of the objective function

study = optuna.create_study()

study.optimize(objective, n_trials=100)
# Used to print the optimal hyperparameters found by the objective function

study.best_params
# Run AdaBoost models using optimised hyperparameters

AdaB_EI = make_pipeline(AdaBoostClassifier(n_estimators=84,

                                           learning_rate=0.0025576981225485613,

                                           algorithm='SAMME.R',

                                           random_state=4))

AdaB_SN = make_pipeline(AdaBoostClassifier(n_estimators=84,

                                           learning_rate=0.0025576981225485613,

                                           algorithm='SAMME.R',

                                           random_state=4))

AdaB_TF = make_pipeline(AdaBoostClassifier(n_estimators=84,

                                           learning_rate=0.0025576981225485613,

                                           algorithm='SAMME.R',

                                           random_state=4))

AdaB_JP = make_pipeline(AdaBoostClassifier(n_estimators=84,

                                           learning_rate=0.0025576981225485613,

                                           algorithm='SAMME.R',

                                           random_state=4))
# Check the cross-validation score of the model from the train data

EI_score = log_loss_cv(AdaB_EI, 'E/I')

SN_score = log_loss_cv(AdaB_EI, 'S/N')

TF_score = log_loss_cv(AdaB_EI, 'T/F')

JP_score = log_loss_cv(AdaB_EI, 'J/P')



# Print out model score for each category

print('Extrovert/Introvert Score: ' + EI_score.mean())

print('Sensing/Intuition Score: ' + SN_score.mean())

print('Thinking/Feeling Score: ' + TF_score.mean())

print('Judging/Percieving Score: ' + JP_score.mean())
# Fit final models to training data

logreg_EI.fit(train, y_train['E/I'])

logreg_SN.fit(train, y_train['S/N'])

logreg_TF.fit(train, y_train['T/F'])

logreg_JP.fit(train, y_train['J/P'])
# Generate predictions

EI_y_pred_test = logreg_EI.predict(test)

SN_y_pred_test = logreg_SN.predict(test)

TF_y_pred_test = logreg_TF.predict(test)

JP_y_pred_test = logreg_JP.predict(test)
# Create submission dataframe and add predictions to it

sub = pd.DataFrame()

sub['id'] = df_test_Id

sub['mind'] = EI_y_pred_test

sub['energy'] = SN_y_pred_test

sub['nature'] = TF_y_pred_test

sub['tactics'] = JP_y_pred_test



# Write submission dataframe to a csv for submission

sub.to_csv('submission.csv', index=False)