import os

import json

from pathlib import Path

import re

import random
for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!pip install fasttext
def transform_text(rating, text):

    fasttext_line = "__label__{} {}".format(rating, text)

    return fasttext_line    



transform_text('5','This restaurant is great!')
def string_formatting(string):

    """This function will convert input text to lowercase and also add space before punctuation symbol."""

    string = string.lower()

    string = re.sub(r"([.!?,'/()])", r" \1 ", string) # The sub() function replaces the matches with the text of your choice

    return string



string_formatting('This restaurant is great!')
# Defining the path of  training and test files

reviews_data = Path('/kaggle/input/yelp-dataset/yelp_academic_dataset_review.json')

fasttext_dataset_train = Path('/kaggle/working/fasttext_dataset_train.txt') # Text file to store data in required format

fasttext_dataset_test = Path('/kaggle/working/fasttext_dataset_test.txt') # Text file to store data in required format



# We will keep 10% data for testing

percent_test_data = 0.10



with reviews_data.open() as input, fasttext_dataset_train.open("w") as train_output, fasttext_dataset_test.open("w") as test_output:



    for line in input:

        review_data = json.loads(line)



        rating = review_data['stars']

        text = review_data['text'].replace("\n", " ")

        text = string_formatting(text)



        fasttext_line = "__label__{} {}".format(rating, text)



        # Return the next random floating point number in the range [0.0, 1.0)

        if random.random() <= percent_test_data:

            test_output.write(fasttext_line + "\n")

        else:

            train_output.write(fasttext_line + "\n")
# Print file size in GB

file_size = os.stat('/kaggle/working/fasttext_dataset_train.txt').st_size/1e+9

print(f'fasttext_dataset_train, file size is: {file_size} GB \n')



file_size = os.stat('/kaggle/working/fasttext_dataset_test.txt').st_size/1e+9

print(f'fasttext_dataset, file size is: {file_size} GB \n')
import fasttext



model = fasttext.train_supervised('fasttext_dataset_train.txt')
# Once the model is trained, we can retrieve the list of words and labels

print(model.words[:20]) # Printing first 20 words

print(model.labels)
def print_results(N, p, r):

    print("N\t" + str(N))

    print("P@{}\t{:.3f}".format(1, p))

    print("R@{}\t{:.3f}".format(1, r))



print_results(*model.test('fasttext_dataset_test.txt'))
print_results(*model.test('fasttext_dataset_test.txt', 2))
input_text = "This is a terrible restaurant. I hate it so much."

print(model.predict(string_formatting(input_text)))



input_text = "This is a very good restaurant."

print(model.predict(string_formatting(input_text)))



input_text = "This is the best restaurant I have ever tried."

print(model.predict(string_formatting(input_text)))

# Predict 3 label by specifying the parameter k = 3

input_text = "This is a terrible restaurant. I hate it so much."

print(model.predict(string_formatting(input_text), k =3))
# If you want to predict more than one sentence you can pass an array of strings 

input_text = ["This is a terrible restaurant. I hate it so much." , "This is a very good restaurant.", "This is the best restaurant I have ever tried."]

print(model.predict(input_text))