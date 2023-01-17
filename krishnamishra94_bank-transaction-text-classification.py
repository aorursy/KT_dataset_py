import pandas as pd

import numpy as np

import os



############### LOADING THE EXCEL FILE ####################



data = pd.read_excel('../input/DA_Test.xlsx')

print('----> Data Loaded Successfully')

############################################################





################# DISPLAYING THE DATA ###############



data.head(10)



####################################################







################# CHECK FOR MISSING VALUES FOR INPUT & OUTPUT COLUMN ###########



def draw_missing_values_table(df):

    nullCount  = df.isnull().sum().sort_values(ascending=False)

    percentage = (df.isnull().sum().sort_values(ascending=False))*100/df.shape[0]

    missingTable = pd.concat([nullCount,percentage],axis=1,keys=['Total','Percentage'])

    return missingTable







input_column  = 'Particulars'

output_column = 'First Level Classification'

draw_missing_values_table(data.loc[:,[input_column,output_column]])



##############################################################################



################ PERFORMING PREPROCESSING OF INPUT TEXT ###################

def preprocess_text(df,column):

    import re

    for i in range(len(df)):

        ######  REMOVING SPECIAL CHARACTERS

        df.loc[i,column]  = re.sub(r'\W',' ',str(df.loc[i,column]))

    

        ######  REMOVING ALL SINGLE CHARACTERS

        df.loc[i,column]  = re.sub(r'\s+[a-zA-Z]\s+',' ',str(df.loc[i,column]))

    

        ######  REMOVING MULTIPLE SPACES WITH SINGLE SPACE

        df.loc[i,column]  = re.sub(r'\s+',' ',str(df.loc[i,column]))

        

    return df





data = preprocess_text(data,input_column)



data[input_column].head()



################# DIVIDING DATA INTO INPUT OUTPUT ######################



X = data.loc[:,input_column]

y = data.loc[:,output_column]

print('Data Divided Successfully Into Input & Output')

#######################################################################



############### USING BAG OF WORDS MODEL TO CONVERT FEATURES INTO NUMBERS ############

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

X_counts   = count_vect.fit_transform(X).toarray()

print(X_counts.shape)

######################################################################################



############### TRAIN - TEST DATA SPLIT #################



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size=0.3, random_state=0)

print('Data Divided Into Train & Test')

########################################################



################ PERFORMING NAIVE BAYESIAN CLASSIFICATION ##############



from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train, y_train)



predict = clf.predict(X_test)



print('Accuracy of The Model is =====> '+str(round(np.mean(predict == y_test)*100,2))+'%')



#######################################################################
