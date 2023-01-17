# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import xlsxwriter

from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC, LinearSVC

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import SGDClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.



## Loading Training data

df = pd.read_excel('../input/training/train.xlsx')



## Loading Testing data

df_test = pd.read_excel('../input/testing1/testData.xlsx')



df1 = df.loc[:,"Title":"State"]

df1_test = df_test.loc[:,"Title":"State"]





df_x = df1["Title"]

df_test_x = df1_test["Title"]

df_y = df1["State"]

df_test_y = df1_test["State"]





df_z = df1["Assigned To"]

df_test_z =  df1_test["Assigned To"]



count = 0;

arr = []

employee = set()



for i in range(len(df_z)):

    if df_z[i] not in employee:

        employee.add(df_z[i])

        arr.append(df_z[i])

       

    

for i in range(len(df_y)):

    df_y[i] =int(arr.index(df_z[i]))

    

for i in range(len(df_test_y)):

    df_test_y[i] =int(arr.index(df_test_z[i]))    

    

# Employee list

df_y = df_y.astype('int64')

df_test_y = df_test_y.astype('int64')







# Bifurcated the training & testing set using train_test_split

x_train , x_test , y_train , y_test = train_test_split(df_x,df_y, test_size = 0.33,random_state = 4)



 

 # Function to split the title into token   

#def split_into_token(Title):

#    Title = str(Title)

#    words = Title.split()

#    array =[]

 #   for word in words:

 #       array.append(word)

 #   print(array)    

    

#df1.Title.apply(split_into_token)



# split_into_token(Title):

#    return TextBlob(str(Title)).words



#df1.Title.head().apply(split_into_token)



def split_into_lemmas(Title):

    Title = str(Title)

    Title = Title.lower()

    words = TextBlob(Title).words

    return [word.lemma for word in words]







bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(df_x)

#print (len(bow_transformer.vocabulary_))





messages_bow = bow_transformer.transform(df_x)

messages_bowtest = bow_transformer.transform(df_test_x)



tfidf_transformer = TfidfTransformer().fit(messages_bow)

tfidf_testtransformer = TfidfTransformer().fit(messages_bowtest)



messages_tfidf = tfidf_transformer.transform(messages_bow)

testmessages_tfidf = tfidf_transformer.transform(messages_bowtest)



#print (messages_tfidf.shape)



#mnb = MultinomialNB()

## SUPPORT VECTOR mACHINE -----------

svm = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42)

## TRAIN THE MODEL 

t_model = svm.fit(messages_tfidf, df_y)



## Predicted value

pred = svm.predict(testmessages_tfidf)



## Actual value

actual = np.array(df_test_y)



## Accuracy

acc = np.mean(pred == actual)



workbook = xlsxwriter.Workbook('output.xlsx')

worksheet = workbook.add_worksheet()



bold = workbook.add_format({'bold': True})

#num = workbook.add_format({'num_format': '##'})



worksheet.write('A1','Predicted',bold)

worksheet.write('D1','Actual',bold)





row  = 1

col = 0



for i in range(len(y_test)):

    worksheet.write(row,col,str(arr[pred[i]]))

    worksheet.write(row,col+3,str(arr[actual[i]]))

    row += 1

    







print("Accuracy ------  " + str(acc*100))

##print("                       ")

##print("Predictions" + "                                   " + " Actual")

##print("                       ")

##for i in range(len(y_test)):

 ##   print(str(arr[pred[i]]) + "    -----------------------   " + str(arr[actual[i]]))







#print ('accuracy', accuracy_score(df1['label'], all_predictions))

     