!wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
!unzip v0.9.2.zip
!cd fastText-0.9.2 && pip install .
import pandas as pd

# "The cleaning supply"

import re

import string

# "NLP Supply"

import nltk

from nltk.stem import WordNetLemmatizer

from nltk.stem.snowball import SnowballStemmer

from nltk.corpus import stopwords
train = pd.read_csv('../input/student-shopee-code-league-sentiment-analysis/train.csv')

test = pd.read_csv('../input/student-shopee-code-league-sentiment-analysis/test.csv')
train.head()
test.head()
# Initialize the lemmarizer and stemmer

wordnet_lemmatizer = WordNetLemmatizer()

englishStemmer=SnowballStemmer("english")



def clean_text(x):

    # Remove zero width space from the string and lower it

    temp_text = x.lower().replace('\u200b', '')

    # Remove punctuation of the string

    temp_text = temp_text.translate(str.maketrans('', '', string.punctuation))

    # Remove new line from string

    temp_text = temp_text.replace('\n', '')

    # Remove double space or more

    temp_text = re.sub(' +', ' ', temp_text).strip()

    # Tokenized the text

    temp_text = nltk.word_tokenize(temp_text)

    stop_words = set(stopwords.words('english'))

    

    filtered_word = []

    

    for word in temp_text:

        # Lemmanize and stem word

        lemma_word = wordnet_lemmatizer.lemmatize(word)

        stemmed_word = englishStemmer.stem(lemma_word)

        

        # Do not add stop words into the the final cleaned sentence

        if stemmed_word in stop_words:

            continue

        else:

            filtered_word.append(stemmed_word)

            

    return " ".join(filtered_word).strip()
# Clean all of the review in training, and test

train['review'] = train['review'].apply(clean_text)

test['review'] = test['review'].apply(clean_text)
VAL_PERCENTAGE = 0.2

N_VAL = int(len(train) * VAL_PERCENTAGE)



# Shuffle train DataFrame also reset the shuffled index

train = train.sample(frac=1).reset_index(drop=True)



# Set validation DataFrame as having the first N_VAL row

val_data = train[:N_VAL]



# Set train DataFrame as the rest

train_data = train[N_VAL:]
val_data.describe()
train_data.describe()
validation_file_path = "./review.val"

training_file_path = "./review.train"



def append_fasttext_dataset(row, file_writer):

    def convert_row_to_dataset_string(row):

        return "__label__" + str(row['rating']) + " " + row['review']

    

    file_writer.write(convert_row_to_dataset_string(row) + '\n')
# Validation file

with open(validation_file_path, 'a+') as writer:

    val_data.apply(lambda x: append_fasttext_dataset(x, writer), axis=1)

    

# Training file

with open(training_file_path, 'a+') as writer:

    train_data.apply(lambda x: append_fasttext_dataset(x, writer), axis=1)
# Look at number of label (1-5) of the training data

train_data['rating'].value_counts().plot.bar()
# Look at number of label (1-5) of the validation data

val_data['rating'].value_counts().plot.bar()
import fasttext
hyper_params = {

    "lr": 0.01,

    "epoch": 15,

    "wordNgrams": 2,

    "dim": 20,

    "verbose": 1

}
model = fasttext.train_supervised(input=training_file_path, **hyper_params)
import matplotlib.pyplot as plt
# Get model accuracy and accuracy of the validation

result = model.test(training_file_path)

validation = model.test(validation_file_path)



print("Result : ", result)

print("Validation : ", validation)



# Plot the result

accuracy_data = [result[1], validation[1]]

labels = ['Model Accuracy', 'Validation Accuracy']



plt.title("Model accuracy")

plt.bar(labels, accuracy_data)

plt.show()
def get_predicted_rating(x, model):

    return int(model.predict(x)[0][0].split('__label__')[1])
val_data['predicted'] = val_data['review'].apply(lambda x: get_predicted_rating(x, model))
import numpy as np

import seaborn as sns

from sklearn.metrics import confusion_matrix
confusion_labels = [1, 2, 3, 4, 5]

confusion_matrix_data = confusion_matrix(val_data["rating"], val_data["predicted"], labels=confusion_labels)

normalised_confusion_matrix = confusion_matrix_data.astype('float') / confusion_matrix_data.sum(axis=1)[:, np.newaxis]
# Plot the normalised confusion matrix

ax = plt.subplot()

sns.heatmap(normalised_confusion_matrix, annot=True, ax=ax, fmt='.2f');



ax.set_title('Normalized Confusion Matrix')



ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels')



ax.xaxis.set_ticklabels(confusion_labels)

ax.yaxis.set_ticklabels(confusion_labels)
hyper_params_25_epoch = {

    "lr": 0.01,

    "epoch": 25,

    "wordNgrams": 2,

    "dim": 20,

    "verbose": 1

}
model_25_epoch = fasttext.train_supervised(input=training_file_path, **hyper_params_25_epoch)
# Get model accuracy and accuracy of the validation

result = model_25_epoch.test(training_file_path)

validation = model_25_epoch.test(validation_file_path)



print("Result : ", result)

print("Validation : ", validation)



# Plot the result

accuracy_data = [result[1], validation[1]]

labels = ['Model Accuracy', 'Validation Accuracy']



plt.title("Model accuracy with 25 epoch")

plt.bar(labels, accuracy_data)

plt.show()
hyper_params_bigger_lr = {

    "lr": 0.6,

    "epoch": 15,

    "wordNgrams": 2,

    "dim": 20,

    "verbose": 1

}
model_bigger_lr = fasttext.train_supervised(input=training_file_path, **hyper_params_bigger_lr)
# Get model accuracy and accuracy of the validation

result = model_bigger_lr.test(training_file_path)

validation = model_bigger_lr.test(validation_file_path)



print("Result : ", result)

print("Validation : ", validation)



# Plot the result

accuracy_data = [result[1], validation[1]]

labels = ['Model Accuracy', 'Validation Accuracy']



plt.title("Model accuracy with learning rate 0.6")

plt.bar(labels, accuracy_data)

plt.show()
hyper_params_autotuning = {

    "lr": 0.06,

    "epoch": 20,

    "wordNgrams": 2,

    "dim": 20,

    "verbose": 1,

    "autotuneValidationFile": validation_file_path

}
model_autotuning = fasttext.train_supervised(input=training_file_path, **hyper_params_autotuning)
# Get model accuracy and accuracy of the validation

result = model_autotuning.test(training_file_path)

validation = model_autotuning.test(validation_file_path)



print("Result : ", result)

print("Validation : ", validation)



# Plot the result

accuracy_data = [result[1], validation[1]]

labels = ['Model Accuracy', 'Validation Accuracy']



plt.title("Model accuracy with autotuning")

plt.bar(labels, accuracy_data)

plt.show()
val_data['predicted'] = val_data['review'].apply(lambda x: get_predicted_rating(x, model_autotuning))
confusion_labels = [1, 2, 3, 4, 5]

confusion_matrix_data = confusion_matrix(val_data["rating"], val_data["predicted"], labels=confusion_labels)

normalised_confusion_matrix = confusion_matrix_data.astype('float') / confusion_matrix_data.sum(axis=1)[:, np.newaxis]
# Plot the normalised confusion matrix

ax = plt.subplot()

sns.heatmap(normalised_confusion_matrix, annot=True, ax=ax, fmt='.2f');



ax.set_title('Normalized Confusion Matrix (Autotuning)')



ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels')



ax.xaxis.set_ticklabels(confusion_labels)

ax.yaxis.set_ticklabels(confusion_labels)
hyper_params_autotuning_metrics = {

    "lr": 0.1,

    "epoch": 20,

    "wordNgrams": 2,

    "dim": 50,

    "verbose": 2,

    "autotuneValidationFile": validation_file_path,

    "autotuneMetric": "f1:__label__4"

}
model_autotuning_metrics = fasttext.train_supervised(input=training_file_path, **hyper_params_autotuning_metrics)
# Get model accuracy and accuracy of the validation

result = model_autotuning_metrics.test(training_file_path)

validation = model_autotuning_metrics.test(validation_file_path)



print("Result : ", result)

print("Validation : ", validation)



# Plot the result

accuracy_data = [result[1], validation[1]]

labels = ['Model Accuracy', 'Validation Accuracy']



plt.title("Model accuracy with autotuning")

plt.bar(labels, accuracy_data)

plt.show()
val_data['predicted'] = val_data['review'].apply(lambda x: get_predicted_rating(x, model_autotuning_metrics))
confusion_labels = [1, 2, 3, 4, 5]

confusion_matrix_data = confusion_matrix(val_data["rating"], val_data["predicted"], labels=confusion_labels)

normalised_confusion_matrix = confusion_matrix_data.astype('float') / confusion_matrix_data.sum(axis=1)[:, np.newaxis]
# Plot the normalised confusion matrix

ax = plt.subplot()

sns.heatmap(normalised_confusion_matrix, annot=True, ax=ax, fmt='.2f');



ax.set_title('Normalized Confusion Matrix (Autotuning Metrics f1:__label__4)')



ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels')



ax.xaxis.set_ticklabels(confusion_labels)

ax.yaxis.set_ticklabels(confusion_labels)
hyper_params_autotuning_metrics_5 = {

    "lr": 0.1,

    "epoch": 20,

    "wordNgrams": 2,

    "dim": 50,

    "verbose": 2,

    "autotuneValidationFile": validation_file_path,

    "autotuneMetric": "f1:__label__5"

}
model_autotuning_metrics_5 = fasttext.train_supervised(input=training_file_path, **hyper_params_autotuning_metrics_5)
# Get model accuracy and accuracy of the validation

result = model_autotuning_metrics_5.test(training_file_path)

validation = model_autotuning_metrics_5.test(validation_file_path)



print("Result : ", result)

print("Validation : ", validation)



# Plot the result

accuracy_data = [result[1], validation[1]]

labels = ['Model Accuracy', 'Validation Accuracy']



plt.title("Model accuracy with autotuning metrics")

plt.bar(labels, accuracy_data)

plt.show()
val_data['predicted'] = val_data['review'].apply(lambda x: get_predicted_rating(x, model_autotuning_metrics_5))
confusion_labels = [1, 2, 3, 4, 5]

confusion_matrix_data = confusion_matrix(val_data["rating"], val_data["predicted"], labels=confusion_labels)

normalised_confusion_matrix = confusion_matrix_data.astype('float') / confusion_matrix_data.sum(axis=1)[:, np.newaxis]
# Plot the normalised confusion matrix

ax = plt.subplot()

sns.heatmap(normalised_confusion_matrix, annot=True, ax=ax, fmt='.2f');



ax.set_title('Normalized Confusion Matrix (Autotuning Metrics f1:__label__5)')



ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels')



ax.xaxis.set_ticklabels(confusion_labels)

ax.yaxis.set_ticklabels(confusion_labels)
submission = test.copy()

submission['rating'] = submission['review'].apply(lambda x: get_predicted_rating(x, model_autotuning_metrics))



del submission['review']
submission.head()
submission.to_csv('submission.csv', index=False)