import pandas as pd

import nltk

import re



from nltk import sent_tokenize

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
df=pd.read_csv(r"../input/amazon-music-reviews/Musical_instruments_reviews.csv")
df.head()
df_use=df[['summary']]
result = [] 

for value in df["overall"]: 

    if value == 1.0: 

        result.append("Negative") 

    elif value==2.0: 

        result.append("Negative") 

    elif value==3.0: 

        result.append("Neutral")

    elif value==4.0: 

        result.append("Positive")

    elif value==5.0: 

        result.append("Positive")

       

df_use["Result"] = result    
df_use.shape
#df_use=df_use.loc[df_use['Result']!='Neutral']
df_use.groupby('Result').describe()
df_use.head()
df_use.shape
df_use.tail(30)
train_df=df_use.iloc[0:-1]
train_df.shape
train_df.tail()
#test_df=df_df.iloc[[-1]]

#test_df.shape
#test_df.head()
data=[['The musical instruments are very good with nice quality','positve']]

test_df=pd.DataFrame(data,columns=['summary','Result'])

test_df.head()


number_of_class_labels=len(train_df['Result'].unique())

number_of_class_labels


train_df['Result'].unique()
Count_Row=train_df.shape[0] 

Count_Col=train_df.shape[1] 
train_df.shape


class_prob_df = pd.DataFrame(columns=['Result', 'probability'], index=range(number_of_class_labels))

class_prob_df
train_df['Result'].value_counts()

    
i=0

for val, cnt in train_df['Result'].value_counts().iteritems():

    print ('value', val, 'was found', cnt, 'times')

    class_prob_df.loc[i].Result = val

    class_prob_df.loc[i].probability = cnt/Count_Row

    i = i +1

    

class_prob_df
import nltk

nltk.download('punkt')
#row.summary
train_df['summary'].dtype


import re

#nltk.download('punkt')

from nltk import sent_tokenize

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



all_tokens = []



for idx, row in train_df.iterrows():

    for word in word_tokenize(row.summary):

        all_tokens.append(word)

    

print(len(all_tokens), all_tokens)

# get only unique tokens

all_tokens_unique = set(all_tokens)

print(len(all_tokens_unique), all_tokens_unique)


stop_words = set(stopwords.words('english'))



tokens = [w for w in all_tokens_unique if not w in stop_words]

print(len(tokens), tokens)



tokens1=[]

tokens = [word for word in tokens if word.isalpha()]

print(len(tokens), tokens)
word = ['@', 'rr', '!', '$', '@', 'jfjf', '&','(', ')', ',']

for word in word:

    if word.isalpha():

        print("yes it is alpha: ", word)
train_df.values
# merge documents for each category

merged_train_df = train_df.groupby('Result')['summary'].apply(' '.join).reset_index()



merged_train_df
for idx, row in merged_train_df.iterrows():

    

    temp1_tokens = []

    for word in word_tokenize(row.summary):

        temp1_tokens.append(word)

    

    temp1_tokens = set(temp1_tokens)

         

    temp2_tokens = []

    for word in temp1_tokens:

        if not word in stop_words:

            temp2_tokens.append(word)           

    

    temp3_tokens = []

    for word in temp2_tokens:

        if word.isalpha():

            temp3_tokens.append(word)

            

    print(temp3_tokens)

    temp4_tokens = " ".join(temp3_tokens)

    print(temp4_tokens)

    

    merged_train_df.at[idx, 'summary'] = temp4_tokens

    merged_train_df.at[idx, 'no_of_words_in_category'] = len(temp3_tokens)
merged_train_df
# merge to get catgory probability

# merged_train_df

# class_prob_df

merged_train_df = pd.merge(merged_train_df, class_prob_df[['Result', 'probability']], on='Result')
merged_train_df
final_df = pd.DataFrame()



row_counter = 0



for idx, row in merged_train_df.iterrows():

    for token in tokens:

        # find the number of occurances of the token in the current category of documents

        no_of_occurances = row.summary.count(token)

        no_of_words_in_category = row.no_of_words_in_category

        no_unique_words_all = len(tokens)

        

        prob_of_token = (no_of_occurances+ 1)/ (no_of_words_in_category+ no_unique_words_all)

        #print(row.class_label, token, no_of_occurances, prob_of_token)

        final_df.at[row_counter, 'Result'] = row.Result

        final_df.at[row_counter, 'token'] = token

        final_df.at[row_counter, 'no_of_occurances'] = no_of_occurances

        final_df.at[row_counter, 'no_of_words_in_category'] = no_of_words_in_category

        final_df.at[row_counter, 'no_unique_words_all'] = no_unique_words_all

        final_df.at[row_counter, 'prob_of_token_category'] = prob_of_token

        

        row_counter = row_counter + 1
final_df
# Calculate P(Category/Document) 

#      = P(Category) * P(Word1/Category) * P(Word2/Category) * P(Word3/Category)



# P(Auto/D6) = P(Auto) * P(Engine/Auto) * P(Noises/Auto) * P(Car/Auto)

for idx, row in test_df.iterrows():

    

    # tokenize & unique words

    temp1_tokens = []

    for word in word_tokenize(row.summary):

        temp1_tokens.append(word)

        #temp1_tokens = set(temp1_tokens)

        

    # remove stop words

    temp2_tokens = []

    for word in temp1_tokens:

        if not word in stop_words:

            temp2_tokens.append(word)

          

    # remove punctuations

    temp3_tokens = []

    for word in temp2_tokens:

        if word.isalpha():

            temp3_tokens.append(word)

            

    #temp4_tokens = " ".join(temp3_tokens)

    #print(temp4_tokens)

    

    prob = 1 

    

    # process for each class_label

    for idx1, row1 in merged_train_df.iterrows():

        print("class: "+ row1.Result)

        for token in temp3_tokens:

            # find the token in final_df for the given category, get the probability

            # row1.class_label & token

        

            print("      : "+ token)  

        

            temp_df = final_df[(final_df['Result'] == row1.Result) & (final_df['token'] == token)]



            # process for exception

            if (temp_df.shape[0] == 0):

                token_prob = 1/(row1.no_of_words_in_category+ no_unique_words_all)

                print("       no token found prob :", token_prob)

                prob = prob * token_prob

            else:

                token_prob = temp_df.get_value(temp_df.index[0],'prob_of_token_category')

                print("       token prob          :", token_prob)

                prob = prob * token_prob



            prob = prob * row1.probability



        col_at = 'prob_'+row1.Result



        test_df.at[idx, col_at] = prob





test_df