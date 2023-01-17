# The input files are Sheet_1 & Sheet_2.



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Sheet_1 is the responses to the therapy bot



df = pd.read_csv('../input/Sheet_1.csv', encoding='latin-1')



# Print a good looking dataframe



df = df.drop(["Unnamed: 3", "Unnamed: 4", "Unnamed: 5", "Unnamed: 6","Unnamed: 7"], axis = 1)

df = df.rename(columns={"v1":"class", "v2":"Responses"})



df.head()
df["class"].value_counts()
# Sheet_2 is the resumes.



df2 = pd.read_csv('../input/Sheet_2.csv', encoding='latin-1')



df2 = df2.rename(columns={"v1":"class", "v2":"Resumes"})



df2.head()
df2["class"].value_counts()




## Bag of response words



from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



def wordcloud(dataframe):

    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(background_color="white",stopwords=stopwords).generate(" ".join([i for i in dataframe.str.upper()]))

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.title("bag_composition")



wordcloud(df['response_text'])



# Looks positive overall, 'friend' is largest, because the bot said: 'Describe a time when you

# have acted as a resource for someone else.'
# Bag of resume words



wordcloud(df2['resume_text'])  



# Looks reasonable, 'data' and 'research' are largest
import operator

from pprint import pprint



def word_freq_bag(dataframe):



    counts = dict()

    bag = []

    counter = 0

    for line in dataframe:

        # print(line)

        words = line.split()

        # print(words)

        for each in words:

            bag.append(each)

    # print(bag)

    for word in bag:

        # print(word)

        # print(counts.get(word,0))

        counts[bag[counter]] = counts.get(word,0)+1        

        #print(counts)

        counter += 1

        

        # if counter == 50: # if you uncomment the print statements, use break (limited memory)

         #   break 



    key = None

    value = None

    keyvalue = dict() # would like to sort by count, dicts are orderless

    for key in counts:

        value = counts[key]

        keyvalue[key] = value

        # rank in descending order of frequency, get a list of tuples sorted by the second

        # element in each tuple

        valueSort1 = sorted(keyvalue.items(), key=operator.itemgetter(1), reverse = True)

        valueSort0 = sorted(keyvalue.items(), key=operator.itemgetter(0), reverse = False)

    # print(counter) # total number of non-unique words

    # print(len(valueSort1)) # every key has a value

    # print(len(valueSort0)) # so these return the same number

    return(valueSort1, valueSort0, counts) 



    

response_word_freq_bag = word_freq_bag(df['response_text'])

# take a look at the words picked out in ascending order of 1st letter 

print(response_word_freq_bag[1])
# Print the word frequencies in the bag in descending order. 

print(response_word_freq_bag[0]) # you can confirm with a find (& replace) tool. E.g. R IDE
# Compute & view for the resumes, if you have a few minutes

# resume_word_freq_bag = word_freq_bag(df2['resume_text']) # Compute the word frequences  

# print(resume_word_freq_bag[0]) # See the most common words (this is commented out, because

# there are 93177 non-unique words (and 1/5th of that number, ~17k, unique words)



# Here is the output down to occurences of 50, if you don't want to uncomment it and run it - 

# [('and', 4629), ('to', 2522), ('of', 2430), ('-', 2021), 

#  ('the', 1706), ('\x8a\x97¢', 1685), ('in', 1683), ('for', 1381), 

#  ('a', 735), ('with', 709), ('on', 637), ('VT', 448), ('as', 354), 

#  ('University', 353), ('data', 346), ('*', 308), ('Research', 275), 

#  ('by', 238), ('I', 235), ('research', 221), ('at', 214), 

#  ('from', 212), ('&', 192), ('including', 185), ('May', 181), 

#  ('analysis', 178), ('using', 173), ('all', 172), ('work', 171), 

#  ('Vermont', 170), ('June', 166), ('an', 165), ('development', 163), 

#  ('2013', 162), ('August', 161), ('new', 151), ('Assistant', 150), 

#  ('Science', 149), ('me', 147), ('Burlington', 143), ('Developed', 143), 

#  ('2014', 142), ('that', 142), ('EXPERIENCE', 132), ('management', 131), 

#  ('College', 128), ('New', 127), ('laboratory', 127), ('software', 126), 

#  ('Email', 125), ('Indeed:', 125), ('WORK', 125), ('Present', 125), 

#  ('EDUCATION', 123), ('2011', 123), ('Environmental', 120), ('was', 119), 

#  ('Engineering', 118), ('o', 114), ('quality', 114), ('2012', 112), 

#  ('support', 112), ('2009', 112), ('January', 111), ('project', 111), 

#  ('students', 111), ('Laboratory', 108), ('team', 107), ('2007', 107), 

#  ('or', 105), ('Scientist', 105), ('2010', 103), ('design', 99), 

#  ('equipment', 99), ('technical', 99), ('September', 99), ('2015', 97), 

#  ('Department', 96), ('2008', 96), ('NY', 95), ('testing', 94), 

#  ('process', 93), ('2005', 93), ('December', 89), ('State', 89), 

#  ('years)', 89), ('other', 88), ('Data', 88), ('Center', 88), 

#  ('The', 87), ('2006', 86), ('experience', 85), ('my', 83), ('is', 81), 

#  ('system', 81), ('water', 81), ('use', 80), ('Biology', 80), ('lab', 79), 

#  ('skills', 78), ('2016', 78), ('MA', 78), ('Medical', 77), ('projects', 76), 

#  ('systems', 76), ('2004', 76), ('March', 75), ('US', 75), 

#  ('ADDITIONAL', 74), ('field', 74), ('business', 74), ('INFORMATION', 73), 

#  ('Development', 73), ('2001', 73), ('reports', 72), ('through', 72), 

#  ('production', 72), ('test', 72), ('product', 71), ('School', 70), 

#  ('April', 70), ('various', 69), ('training', 69), ('materials', 68), 

#  ('samples', 68), ('environmental', 68), ('Engineer', 67), ('SKILLS', 67), 

#  ('October', 66), ('NH', 66), ('Responsible', 65), ('time', 65), 

#  ('July', 64), ('Health', 63), ('into', 62), ('Manager', 62), ('2003', 62), 

#  ('/', 62), ('Assisted', 61), ('Microsoft', 61), ('Performed', 61), 

#  ('customer', 60), ('any', 60), ('years', 59), ('Senior', 59), ('National', 59), 

#  ('2000', 58), ('based', 58), ('well', 57), ('Provided', 57), ('IBM', 57), 

#  ('working', 57), ('Analysis', 57), ('technology', 57), ('techniques', 57), 

#  ('included', 57), ('February', 56), ('were', 56), ('their', 56), ('Project', 56),

#  ('control', 56), ('this', 56), ('study', 56), ('cell', 56), ('information', 55), 

#  ('Inc', 55), ('multiple', 55), ('applications', 55), ('program', 55),

#  ('Design', 55), ('company', 54), ('Technician', 53), ('Management', 53),

#  ('products', 53), ('develop', 53), ('1999', 53), ('during', 52), 

#  ('scientific', 52), ('Managed', 51), ('be', 51), ('Worked', 51), 

#  ('Chemistry', 51), ('staff', 51), ('Responsibilities', 51), 

#  ('professional', 50), ('manufacturing', 50), ('such', 50),
# This is a little bit of an unconventional training set. The bag of words model does not

# pick out features of individual training examples that caused them to be identified as a

# particular class. Instead, it has an aggregate collection of words from all training examples

# identified as a particular class. It classifieds new examples based on this aggregate.



train_df = df.sample(frac = 0.5, axis=0)

test_df = df.sample(frac = 0.5, axis=0)



train_df2 = df2.sample(frac = 0.5, axis=0)

test_df2 = df2.sample(frac = 0.5, axis=0)



train_df_flagged = train_df.loc[train_df['class'] == 'flagged']

train_df_flagged_bag = word_freq_bag(train_df_flagged['response_text'])

print(train_df_flagged_bag[0]) # for example, here you can see the flagged responses' words

train_df_not_flagged = train_df.loc[train_df['class'] == 'not_flagged']

train_df_not_flagged_bag = word_freq_bag(train_df_not_flagged['response_text'])

train_df2_flagged = train_df2.loc[train_df2['class'] == 'flagged']

train_df2_flagged_bag = word_freq_bag(train_df2_flagged['resume_text'])

train_df2_not_flagged = train_df2.loc[train_df2['class'] == 'not_flagged']

train_df_not_flagged.tail() # it's a random sample, so the entries are not ordered, but

# you can see that you are indexing what you want - here's a subset of not_flagged responses

train_df2_not_flagged_bag = word_freq_bag(train_df2_not_flagged['resume_text'])



# You can see printed out first is the number of non-unique words & the number of unique words

# The number of unique words is printed twice, this was to check that each list was the same

# length returned from the word_freq_bag function.

# All good.
# You can use the print statements in this code below here to get a clearer picture.



# to convert a string to a dataframe with pandas, in order to reuse word_freq_bag from above

import io

import sys



# mydict1={'response:2,'response2:2,'response6':2,'response8':1}

# mydict2={'response':1,'response2':5,'response8':7}

# mykey=[sum(value * mydict1[key] for key,value in mydict2.items())]

# print(mykey)



def score(a, b):

    mykey=[sum(value * a.get(key,0) for key,value in b.items())]

    # print(b.items())

    # print(a)

    # print(b)

    # global score_counter

    # score_counter += 1

    # if score_counter == 10: # to limit memory so you can run without needing to restart

    #     sys.exit()

    return(mykey)



def classifier_accuracy(text_type, text_id, test_dataframe, train_dataframe_flagged,

                        train_dataframe_flagged_bag, 

                        train_dataframe_not_flagged_bag, length):

    counter = 0

    false_positive = 0

    false_negative = 0

    true_positive = 0

    true_negative = 0

    random_guess = 1-len(train_dataframe_flagged)/length

    test_dataframe_flagged = test_dataframe.loc[test_dataframe['class'] == 'flagged']

    positive_text = len(test_dataframe_flagged)

    print(positive_text)

    test_dataframe_not_flagged = test_dataframe.loc[test_dataframe['class'] == 'not_flagged']

    negative_text = len(test_dataframe_not_flagged)

    print(negative_text)

    

    # print(train_dataframe_flagged_bag[2])

    for i in test_dataframe[text_id]:

        split_char = i.split('_') # get id from resume_id

        integer = int(float(split_char[1])) -1 # acess by index, one less that resume_id

        text_class = test_dataframe['class'][integer]

        # print(text_class)

        # print(text_class)

        text = test_dataframe[text_type][integer]

        # convert string to dataframe, make more than wide enough for the text column

        df3 = pd.read_fwf(io.StringIO(text), header=None, widths=[1000000], names=[text_type])

        # df3 # if you want to see the dataframe for the response/resume

        text_bag = word_freq_bag(df3[text_type]) 

        # print(text_bag[2])

        # print("ok, here goes:")

        

        

        score_flagged = score(text_bag[2], train_dataframe_flagged_bag[2]) # 2 is counts dict

        #print(score_flagged)

        score_not_flagged = score(text_bag[2], train_dataframe_not_flagged_bag[2])

        #print(score_not_flagged)

        if score_flagged > score_not_flagged and text_class == 'flagged':

            counter += 1

            true_positive += 1

        elif score_flagged <= score_not_flagged and text_class == 'not_flagged':

            counter += 1

            true_negative += 1

        elif score_flagged > score_not_flagged and text_class == 'not_flagged':

            counter += 0

            false_positive += 1

        elif score_flagged <= score_not_flagged and text_class == 'flagged':

            counter += 0

            false_negative += 1

    return('number of test examples: ' + str(length), 'correct identifications: ' + str(counter), 'pick_randomly_performace: ' + str(random_guess), 'model_performance: ' + str(counter/length), 'false negatives: ' + str((positive_text-false_negative)/positive_text), 'false positives: ' + str((positive_text-true_positive)/positive_text), 'false positive count: ' + str(false_positive) , 'false negative count: ' + str(false_negative))



score_counter = 0

print(classifier_accuracy('response_text', 'response_id', test_df, train_df_flagged, train_df_flagged_bag, train_df_not_flagged_bag, 40))

print(classifier_accuracy('resume_text', 'resume_id', test_df2, train_df2_flagged, train_df2_flagged_bag, train_df2_not_flagged_bag, 62))

    
# converting string of words in a response, for example, into a data frame (Appendix work)



# text = test_df['response_text'][0]

# df3 = pd.read_fwf(io.StringIO(text), header=None, widths=[1000000], names=['response_text'])

# df3

# text_bag = word_freq_bag(df3['response_text'])

# text_bag



# This is the summing in the bag



mydict1={'response':2,'response2':2,'response6':2,'response8':1}

mydict2={'respose':1,'respone':5,'response8':7}

mykey=[sum(value * mydict1.get(key,0) for key,value in mydict2.items())]

print(mykey)
# To see the effect of CountVectorizer, you can note the number of words 

# now in the pool of words versus seperating words by hand:

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()

train_df = df.sample(frac = 1, axis=0)  # set this to 0.5, you'll have a 

                                        # test set of 40 responses

train_df_mat = vect.fit_transform(train_df['response_text'])

print(train_df_mat.shape) 

# to split into a train and test set, make the sample frac = 0.5, and you will get

# ~ 250-500 words in a sum total of half of the responses

# list of words represented in this train set

print(vect.get_feature_names())

# confirm number of words is 660 (and note the omission of the 832 - 660 = 172 words that were 

# caught with CountVectorizer. For example, below "themselves" & "themselves," are rolled

# together into one "themselves".) 

# This is a benefit of sklearn.

print(len(vect.get_feature_names()))
# repeat for resume set

# Again, CountVectorizer, significantly decreases the number of words (down from ~17k to ~11k).

vect2 = CountVectorizer()

train_df2 = df2.sample(frac = 1, axis=0) # if the frac is changed to 0.5, 

                                         # get 62 resumes (0.5(floor(125))) ✔️️

                                         # ~ 6-9k words in a sum total of half 

                                         # of the resumes, depending on the sample  

df2_mat = vect2.fit_transform(train_df2['resume_text'])

print(df2_mat.shape)  



# With the full pool of resumes, here we see CountVectorizer finds 11466 unique words.