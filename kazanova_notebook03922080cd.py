import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import chardet

import pandas as pd

import csv



data=[] #will hold text

target=[] #will hold label

with open("../input/training.1600000.processed.noemoticon.csv", encoding='latin1') as csvfile:

     spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')

     for row in spamreader:

            data.append(row[5])

            target.append(0 if row[0]=="0" else 1)    

#print top rows            

for i in range (10):

    print ("row: %d label: %d text: %s" % (i,target[i],data[i]))    

#print average target

print ("average target %f " % (np.mean(target)))

#print len of data

print ("len of text data is %d  " % (len(data)))





# Any results you write to the current directory are saved as output.
#we transform text sentences into numbers using tfidf from sklearn

#link : http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html



from sklearn.feature_extraction.text import TfidfVectorizer



tfv=TfidfVectorizer(min_df=0, max_features=None, strip_accents='unicode',lowercase =True,

                            analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1,1),

                            use_idf=True,smooth_idf=True, sublinear_tf=True, stop_words = "english")   



#we fit the TfidfVectorizer tarnsform the dataset

transformed_data=tfv.fit_transform(data)

print (" dataset trandformed")

print (" dataset shape ", transformed_data.shape)
#fit a classifier (Logistic regression) to differentiate between tweets with negative and positive intent.

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

#specify model and parameters

model=LogisticRegression(C=1.)

#fit model

model.fit(transformed_data,target)

#make prediction on the same (train) data

probability_to_be_positive=model.predict_proba(transformed_data)[:,1]

#chcek AUC(Area Undet the Roc Curve) to see how well the score discriminates between negative and positive

print (" auc " , roc_auc_score(target,probability_to_be_positive))

#print top 10 scores as a sanity check

print (probability_to_be_positive[:10])
#sort results to show some positive ans some negative tweets

import operator

array_with_all_elements=[]

#create a new 2dimensional array to add probability and original text

for i in range (len(probability_to_be_positive)):

    array_with_all_elements.append([data[i],probability_to_be_positive[i] ])



#sort in ascending manner based on prediction

array_with_all_elements=sorted(array_with_all_elements, key=operator.itemgetter(1)) 



print ("===============Printing top negative comments===============")

#print top negative comments            

for i in range (10):

    print ("probability: %f negative comment text: %s" % (array_with_all_elements[i][1],array_with_all_elements[i][0]))  



print ("===============Printing top positive comments===============")

#print top positive comments            

for i in range (len(array_with_all_elements)-1,len(array_with_all_elements)-11,-1 ):

    print ("probability: %f positive comment text: %s" % (array_with_all_elements[i][1],array_with_all_elements[i][0]))  
