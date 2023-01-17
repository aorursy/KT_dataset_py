import numpy as np

import pandas as pd

import requests

from flashtext.keyword import KeywordProcessor

from nltk.corpus import stopwords



# let's read in a couple of forum posts

forum_posts = pd.read_csv("../input/meta-kaggle/ForumMessages.csv")



# get a smaller sub-set for playing around with

sample_posts = forum_posts.Message[0:3]



# get data from list of top 5000 pypi packages (last 30 days)

url = 'https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.json'

data = requests.get(url).json()



# get just the list of package names

list_of_packages = [data_item['project'] for data_item in data['rows']]



# create a KeywordProcess

keyword_processor = KeywordProcessor()

keyword_processor.add_keywords_from_list(list_of_packages)



# remove english stopwords

keyword_processor.remove_keywords_from_list(stopwords.words('english'))



# remove custom stopwords

keyword_processor.remove_keywords_from_list(['http','kaggle'])



# test our keyword processor

for post in sample_posts:

    keywords_found = keyword_processor.extract_keywords(post, span_info=True)

    print(keywords_found)
# I'm not going to read in the packages & data again since it's 

# already in our current environment.



def create_keywordProcessor(list_of_terms, remove_stopwords=True, 

                            custom_stopword_list=[""]):

    """ Creates a new flashtext KeywordProcessor and opetionally 

    does some lightweight text cleaning to remove stopwords, including

    any provided by the user.

    """

    # create a KeywordProcessor

    keyword_processor = KeywordProcessor()

    keyword_processor.add_keywords_from_list(list_of_terms)



    # remove English stopwords if requested

    if remove_stopwords == True:

        keyword_processor.remove_keywords_from_list(stopwords.words('english'))



    # remove custom stopwords

    keyword_processor.remove_keywords_from_list(custom_stopword_list)

    

    return(keyword_processor)



def apply_keywordProcessor(keywordProcessor, text, span_info=True):

    """ Applies an existing keywordProcessor to a given piece of text. 

    Will return spans by default. 

    """

    keywords_found = keywordProcessor.extract_keywords(text, span_info=span_info)

    return(keywords_found)

    



# create a keywordProcessor of python packages    

py_package_keywordProcessor = create_keywordProcessor(list_of_packages, 

                                                      custom_stopword_list=["kaggle", "http"])



# apply it to some sample posts (with apply_keywordProcessor function, omitting

# span information)

for post in sample_posts:

    text = apply_keywordProcessor(py_package_keywordProcessor, post, span_info=False)

    print(text)
import pickle



# save our file (make sure our file permissions are "wb", 

# which will let us _w_rite a _b_inary file)

pickle.dump(py_package_keywordProcessor, open("processor.pkl", "wb"))



# check our current directory to make sure it saved

!ls
# read in a processor from our pickled file. Don't forget to 

# include "rb", which lets us _r_ead a _b_inary file.

pickle_keywordProcessor = pickle.load(open("processor.pkl", "rb"))



# apply it to some sample text to make sure it works

apply_keywordProcessor(pickle_keywordProcessor, "I like pandas numpy and seaborn") 

# your code here :)



# get list of R packages

r_packages = pd.read_csv("../input/list-of-r-packages/r_packages.csv")



r_processor = create_keywordProcessor(list(r_packages["Package"]))



# test our keyword processor

for post in sample_posts:

    keywords_found = r_processor.extract_keywords(post, span_info=True)

    print(keywords_found)
# your code here :)



# already did it :)
# your code here :)



import pickle 



pickle.dump(r_processor, open("r_processor.pkl", "wb"))

!ls
from flashtext.keyword import KeywordProcessor

import pickle



# Function that takes loads in our pickled word processor

# and defines a function for using it. This makes it easy

# to do these steps together when serving our model.

def get_keywords_api():

    

    # read in pickled word processor. You could also load in

    # other models as this step.

    keyword_processor = pickle.load(open("processor.pkl", "rb"))

    

    # Function to apply our model & extract keywords from a 

    # provided bit of text

    def keywords_api(keywordProcessor, text, span_info=True): 

        keywords_found = keywordProcessor.extract_keywords(text, span_info=True)      

        return keywords_found

    

    # return the function we just defined

    return keywords_api
import json

from flask import Flask, request

#from serve import get_keywords_api

# I've commented out the last import because it won't work in kernels, 

# but you should uncomment it when we build our app tomorrow



# create an instance of Flask

app = Flask(__name__)



# load our pre-trained model & function

keywords_api = get_keywords_api()



# Define a post method for our API.

@app.route('/extractpackages', methods=['POST'])

def extractpackages():

    """ 

    Takes in a json file, extracts the keywords &

    their indices and then returns them as a json file.

    """

    # the data the user input, in json format

    input_data = request.json



    # use our API function to get the keywords

    output_data = keywords_api(input_data)



    # convert our dictionary into a .json file

    # (returning a dictionary wouldn't be very

    # helpful for someone querying our API from

    # java; JSON is more flexible/portable)

    response = json.dumps(output_data)



    # return our json file

    return response
# save pickles with names that work for the function we define below

pickle.dump(py_package_keywordProcessor, open("python_processor.pkl", "wb"))

pickle.dump(r_processor, open("r_processor.pkl", "wb"))

!ls
# your code here :) (feel free to copy & paste my code and then modify it for your project -- R)



from flashtext.keyword import KeywordProcessor

import pickle



# Function that takes loads in the pickled word processor 



# for a specific language. HT to neomatrix369 for the refactor!

def get_keywords_api(language="python"):

    

    # read in pickled word processor. You could also load in

    # other models as this step.

    keyword_processor = pickle.load(open(language + "_processor.pkl", "rb"))

    

    # Function to apply our model & extract keywords from a 

    # provided bit of text

    def keywords_api(keywordProcessor, text, span_info=True): 

        keywords_found = keywordProcessor.extract_keywords(text, span_info=True)      

        return keywords_found

    

    # return the function we just defined

    return keywords_api
# your code here :)



import json

from flask import Flask, request

#from serve import get_keywords_api

# I've commented out the last import because it won't work in kernels, 

# but you should uncomment it when we build our app tomorrow



# create an instance of Flask

app = Flask(__name__)



# load our pre-trained model & function

keywords_api_python = get_keywords_api()

keywords_api_r = get_keywords_api(language="r")



# Define a post method for our API.

@app.route('/extractpackages_python', methods=['POST'])

def extractpackages_python():

    """ 

    Takes in a json file, extracts the keywords &

    their indices and then returns them as a json file.

    """

    # the data the user input, in json format

    input_data = request.json



    # use our API function to get the keywords

    output_data = keywords_api_python(input_data)



    # convert our dictionary into a .json file

    # (returning a dictionary wouldn't be very

    # helpful for someone querying our API from

    # java; JSON is more flexible/portable)

    response = json.dumps(output_data)



    # return our json file

    return response



# Define a post method for our API.

@app.route('/extractpackages_r', methods=['POST'])

def extractpackages_r():

    """ 

    Takes in a json file, extracts the keywords &

    their indices and then returns them as a json file.

    """

    # the data the user input, in json format

    input_data = request.json



    # use our API function to get the keywords

    output_data = keywords_api_r(input_data)



    # convert our dictionary into a .json file

    # (returning a dictionary wouldn't be very

    # helpful for someone querying our API from

    # java; JSON is more flexible/portable)

    response = json.dumps(output_data)



    # return our json file

    return response