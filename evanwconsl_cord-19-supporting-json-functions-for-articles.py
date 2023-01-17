# This notebook contains a function to return the keys of an article in the dataset.



# def findArticlePath(article_sha_id)

# Given the article_sha_id, it first returns the article path.

# 2DO: Look in more than one directory



# def returnJson(article_sha_id)

# Given the article_sha_id (same as above) return the JSON text of the article



# def retrieveKey(json_input,key_input)

# Then given a string containing one key_input value

# Return a list of results of values for that key

# Otherwise an empty array []



# Example below:

# List all of the title that contain the word 'vaccine'
!pip install pyspark
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import json

from pyspark.sql import SparkSession

from pyspark.sql.functions import from_json,col
spark = SparkSession.builder.appName("SimpleApp").getOrCreate()

sc = spark.sparkContext
def findArticlePath(article_sha_id):

    """

    This function finds the full path given an article_sha_id

    """

    # Let's start by opening the file.

    ROOT_PATH = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'

    FILE_SUFFIX = '.json'

    article_path = ROOT_PATH + article_sha_id + FILE_SUFFIX

    return article_path
def retrieveJson(article_sha_id):

    """

    Given a 1-word string containing a JSON key, return the data for those keys.

    Also return the location of those keys?

    """

    article_file_path = findArticlePath(article_sha_id)

    with open(article_file_path, "r") as read_file:

        json_string = json.load(read_file)

    return(json_string)
def saveJson(article_sha_id,json_save_file):

    """

    Save Json Value

    """

    article_file_path = findArticlePath(article_sha_id)

    with open(json_save_file,"w") as write_file:

        read_file = open(article_file_path,"r")

        c = read_file.read()

        while c != 0:

            write_file.write(c)

            c = read_file.read()

        read_file.close()

    write_file.close()
def retrieveTopKey(json_string,key_input):

    """

    Retrieve value of top key in json_string

    or None if it does not exist

    """

    foundkey = False

    

    keys_returned = []

    for key in json_string:

        if key == key_input:

            keyvalue = json_string[key]

            keys_returned.append(keyvalue)

            foundkey = True

    if foundkey == True:

        return keys_returned

    else:

        return None

    

            
def retrieveSubKey(intermediate_result,key_input):

    """

    Retrieve value of top key in json_string

    or None if it does not exist

    """

    foundkey = False

    keys_returned = []

    for item in intermediate_result:

        for key in item:

            if key == key_input:

                keyvalue = item[key]

                keys_returned.append(keyvalue)

                foundkey = True

        if foundkey == True:

            return keys_returned

        else:

            return None
def retrieveKey(json_input,key_input):

    """

    Uses retrieveTopKey and retrieveSubKey to return key values from anywhere in the JSON data string.

    """

    result = []

    # First decide if this is the top or not!

    istopitem = False

    if not isinstance(json_input,list):

        json_string = json_input

        json_input = []

        json_input.append(json_string)

        istopitem = True

    # If this is the top value check to see if the key is in the Top values

    # This is the Json string the top, not our intermediate result!

    if istopitem:

        #print("Top")

        for key in json_string:

            if key == key_input:

                top_key_value = json_string[key]

                result.append(top_key_value)

                return result

    # It was not found in the top value so recursively check the sub keys

    #print("Sub")

    #print(json_input)

    for json_string in json_input:

        #print(json_string)

        if isinstance(json_string,dict):

            for key in json_string:

                top_key_value = json_string[key]

                # Is a sub key currently valid?

                if isinstance(top_key_value,dict):

                    #print("is dict")

                    #print(top_key_value)

                    for sub_key in top_key_value:

                        sub_key_value = top_key_value[sub_key]

                        if sub_key == key_input:

                            result.append(sub_key_value)

                        else:

                            # Sub Key is not currently valid but a sub-sub key might be!

                            # So recursively call the sub Key

                            sub_result = retrieveKey(sub_key_value,key_input)

                            if len(sub_result) != 0:

                                result.append(sub_result)

                elif isinstance(top_key_value,list):

                    #print("is list")

                    #print(top_key_value)

                    for top_key_value_item in top_key_value:

                        #print(top_key_value_item)

                        if isinstance(top_key_value_item,dict):

                            for sub_key in top_key_value_item:

                                sub_key_value = top_key_value_item[sub_key]

                                if sub_key == key_input:

                                    result.append(sub_key_value)

                                else:

                                    # Sub Key is not currently valid but a sub-sub key might be!

                                    # So recursively call the sub Key

                                    sub_result = retrieveKey(sub_key_value,key_input)

                                    if len(sub_result) != 0:

                                        result.append(sub_result)

                        else: # Must be a string

                            result.append(top_key_value_item)

    return result
def searchWordInTitle(word):

    article_list = []

    for files in os.listdir(path='/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'):

        article_sha_id = files.split(".")[0]

        article_json = retrieveJson(article_sha_id)

        title = retrieveKey(article_json,"title")

        if isinstance(title,list):

            for item in title:

                if isinstance(item,list):

                    for it in item:

                        find_result = it.find(word)

                        if find_result >= 0:

                            if not article_sha_id in article_list:

                                article_list.append(article_sha_id)

                else:

                    find_result = item.find(word)

                    if find_result >= 0:

                        if not article_sha_id in article_list:

                            article_list.append(article_sha_id)

        else:

            find_words = title.find(word)

            if find_words >= 0:

                if not article_sha_id in article_list:

                    article_list.append(article_sha_id)

    return article_list
def printArticleListResult(article_list):

    for article_sha_id in article_list:

        article_json = retrieveJson(article_sha_id)

        article_title = retrieveKey(article_json,"title")

        for item in article_title:

            if isinstance(item,list):

                for it in item:

                    find_result = it.find("vaccine")

                    if find_result >=0:

                        print(it)

            else:

                print(item)
def abstract(data):

    """ Abstract """

    #print("Abstract")

    data_item = data[0]

    row_entries = data_item[0]

    for row in row_entries:

        #print(row)

        cite_span = row[0]

        ref_span = row[1]

        section = row[2]

        text = row[3]

        #print(cite_span)

        #print(ref_span)

        #print(section)

        #print(text)

    return (cite_span,ref_span,section,text)
def back_matter(data):

    """ Back Matter """

    #print("Back Matter")

    data_item = data[0]

    row_entries = data_item[0]

    for row in row_entries:

        #print(row)

        cite_span = row[0]

        ref_span = row[1]

        section = row[2]

        text = row[3]

        #print(cite_span)

        #print(ref_span)

        #print(section)

        #print(text)

    return(cite_span,ref_span,section,text)
def bib_entries(data):

    """ Bib Entries """

    #print("Bib Entries")

    data_item = data[0]

    row_entries = data_item[0]

    for row in row_entries:

        #print(row)

        cite_span = row[0]

        ref_span = row[1]

        section = row[2]

        text = row[3]

        #print(cite_span)

        #print(ref_span)

        #print(section)

        #print(text)

    return (cite_span,ref_span,section,text)
def body_text(data):

    """ Body Text """

    #print("Body Text")

    data_item = data[0]

    row_entries = data_item[0]

    for row in row_entries:

        cite_span = row[0]

        ref_span = row[1]

        section = row[2]

        text = row[3]

        #print(cite_span)

        #print(ref_span)

        #print(section)

        #print(text)

    return (cite_span,ref_span,section,text)
def metadata(data):

    """ Metadata """

    #print("Metadata")

    data_item = data[0]

    row_entries = data_item[0]

    num_entries = len(row_entries)

    i = 0

    for row in row_entries:

        if i < num_entries - 1:

            rowitems = row_entries[i][0]

            affiliation = rowitems[0]

            institution = affiliation[0]

            laboratory = affiliation[1]

            location = affiliation[2]

            #country = location[0]

            #postcode = location[1]

            #settlement = location[2]

            email = rowitems[1]

            first = rowitems[2]

            last = rowitems[3]

            middle = rowitems[4]

            suffix = rowitems[5]

            #print(affiliation)

            #print(institution)

            #print(laboratory)

        else:

            title = row

        i += 1

    return (affiliation,institution,laboratory,location,email,first,last,middle,suffix)
def paper_id(data):

    """ Paper Id """

    #print("Paper Id")

    data_item = data[0]

    paper_id = data_item[0]

    return (paper_id)

def ref_entries(data):

    """ Ref Entries """

    #print("Ref Entries")

    data_item = data[0]

    row_entries = data_item[0]

    #print(row_entries)

    for row in row_entries:

        latex = row[0]

        text = row[1]

        type_data = row[2]

        #print(latex)

        #print(text)

        #print(type_data)

    return (latex,text,type_data)

column_functions = {0:abstract,1:back_matter,2:bib_entries,3:body_text,4:metadata,5:paper_id,6:ref_entries}
def returnTextualReferences(article_list):

    for files in os.listdir(path='/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'):

        article_sha_id = files.split(".")[0]

        for article in article_list:

            if article == article_sha_id:

                save_json_file = '/kaggle/working/' + article + '.json';

                article_json = saveJson(article_sha_id,save_json_file)

                article_df = spark.read.json(save_json_file)

                article_df.createOrReplaceTempView("article")

                column_results = []

                column_names = []

                print(article_df)

#                for column_name in article_df.schema.names:

#                    strSQL = "SELECT " + column_name + " from article"

#                    column_result = spark.sql(strSQL)

#                    print(column_result)

#                    column_results.append(column_result)

#                    column_names.append(column_name)

#                i = 0

#                for column_result in column_results:

#                    column_name = column_names[i]

#                    column_schema = column_result.schema

#                    column_rdd = column_result.rdd

#                    column_result.show()

#                    result = column_result.collect()

#                    func = column_functions.get(i)

#                    func(result)

#                    print(result)

#                    i += 1

vaccine_in_title = searchWordInTitle("vaccine")

printArticleListResult(vaccine_in_title)

#returnTextualReferences(try_one)