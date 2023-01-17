# -*- coding: utf-8 -*-

"""

Created on Wed Apr 12 10:46:36 2017



@author: IN0055

"""

import csv



import re, math

from collections import Counter



WORD = re.compile(r'\w+')



def get_cosine(vec1, vec2):

     intersection = set(vec1.keys()) & set(vec2.keys())

     numerator = sum([vec1[x] * vec2[x] for x in intersection])



     sum1 = sum([vec1[x]**2 for x in vec1.keys()])

     sum2 = sum([vec2[x]**2 for x in vec2.keys()])

     denominator = math.sqrt(sum1) * math.sqrt(sum2)



     if not denominator:

        return 0.0

     else:

        return float(numerator) / denominator



def text_to_vector(text):

     words = WORD.findall(text)

     return Counter(words)



all_result=[]

with open('../input/test.csv', 'r') as csvfile:

    reader = csv.DictReader(csvfile)

    for row in reader:

        new_dict={}

        

        new_dict=row

        text1 = row['question1']

        text2 = row['question2']

        vector1 = text_to_vector(text1)

        vector2 = text_to_vector(text2)



        cosine = get_cosine(vector1, vector2)

        

        new_dict['similarity']=cosine

        all_result.append(new_dict)

        



   

sub_result=[]

for item in all_result:

    sub_result.append({'test_id':item['test_id'],'is_duplicate':item['similarity']})





keys = sub_result[0].keys()

with open('submission_test_1.csv', 'w') as output_file:

    dict_writer = csv.DictWriter(output_file, keys)

    dict_writer.writeheader()

    dict_writer.writerows(sub_result)

    
