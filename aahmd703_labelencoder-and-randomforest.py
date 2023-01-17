# importing libraries



import numpy as np

import pandas as pd

from IPython.display import display, HTML



from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier
# reading in data



people = pd.read_csv('../input/people.csv')

activity_train = pd.read_csv('../input/act_train.csv')

activity_test = pd.read_csv('../input/act_test.csv')
# merging the dataframes into train, test



df = activity_train.merge(people, how='left', on='people_id' )

df_test = activity_test.merge(people, how='left', on='people_id' )
# the shape of the dataframes



print (df.shape)

print (df_test.shape)
# filling NaN values first



df = df.fillna('0', axis=0)

df_test = df_test.fillna('0', axis=0)
# taking a look at the first few rows



df.head()
df_test.head()
# a multi-column LabelEncoder()



# this solution for applying LabelEncoder() across multiple columns was suggested in the following thread

# http://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn



# I like this solution but is it the most efficient?  Would another method be more practical, particularly if 

# applied to different type of model 



class MultiColumnLabelEncoder:

    def __init__(self,columns = None):

        self.columns = columns 



    def fit(self,X,y=None):

        return self



    def transform(self,X):

        output = X.copy()

        if self.columns is not None:

            for col in self.columns:

                output[col] = LabelEncoder().fit_transform(output[col])

        else:

            for colname,col in output.iteritems():

                output[colname] = LabelEncoder().fit_transform(col)

        return output



    def fit_transform(self,X,y=None):

        return self.fit(X,y).transform(X)
# defining a processor 



def processor(data):

    data = MultiColumnLabelEncoder(columns = ['people_id','activity_id', 'activity_category', 'date_x', 'char_1_x', 'char_2_x',

                                        'char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x',

                                        'char_10_x', 'char_1_y', 'group_1', 'char_2_y', 'date_y', 'char_3_y', 'char_4_y',

                                        'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y']).fit_transform(df)

    

    bool_map = {True:1, False:0}



    data = data.applymap(lambda x: bool_map.get(x,x))

    

    return data
# applying processor to training data



df_encoded = processor(df)
df_encoded.head()
df_encoded.dtypes
# applying processor to test data



df_test_encoded = processor(df_test)
df_test_encoded.head()
df_test_encoded.dtypes
# defining X and y (features and target label)



X = df_encoded

y = X.pop('outcome')
# shape of X and y



print (X.shape)

print (y.shape)
'''



train, test, split the data.  hold out 25% for test



generally if not provided a test set, this would be the way to move forward.

yet I feel something is off in my process and would love feedback!



'''



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
# random forest classifier



model = RandomForestClassifier(77, n_jobs=-1, random_state=7)

model.fit(X_train, y_train)

print ("model score ", model.score(X_test, y_test))
# predicting test data



pred = model.predict(X_test)

pred