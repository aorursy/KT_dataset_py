import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk

import riiideducation # feather dataset 

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
 #used to make feather files easier to load.

 #dtypes = {

 #    "row_id": "int64",

 #    "timestamp": "int64",

 #    "user_id": "int32",

 #    "content_id": "int16",

 #   "content_type_id": "boolean",

 #   "task_container_id": "int16",

 #   "user_answer": "int8",

 #   "answered_correctly": "int8",

 #   "prior_question_elapsed_time": "float32", 

 #   "prior_question_had_explanation": "boolean"

 #}



 #files = ['train', 'questions', 'lectures', 'example_test', 'example_sample_submission']



 #for file in files:

 #    if file=='train':

 #        data = pd.read_csv("../input/riiid-test-answer-prediction/{0}.csv".format(file), dtype=dtypes)

 #    else:

 #        data = pd.read_csv("../input/riiid-test-answer-prediction/{0}.csv".format(file))

 #    data.to_feather("{0}.feather".format(file))

 #    print("File: {0} - size: {1}".format(file,data.shape))
# train_dataframe = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', low_memory=False, nrows=10**5,)

# questions_dataframe = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv', low_memory=False, nrows=10**5,)

# example_test = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/example_test.csv', low_memory=False, nrows=10**5,)



train_dataframe = pd.read_feather('../input/feathers/train.feather')

questions_dataframe = pd.read_feather('../input/feathers/questions.feather')

lectures = pd.read_feather('../input/feathers/lectures.feather')

example_test = pd.read_feather('../input/feathers/example_test.feather')

example_sample_submission = pd.read_feather('../input/feathers/example_sample_submission.feather')

#train_dataframe = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',usecols = ['user_id','content_id','answered_correctly','content_type_id'])

train_dataframe.shape
# questions_dataframe = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv')

# example_test = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/example_test.csv')
questions_dataframe.shape
print(train_dataframe['user_id'])
#trainWhereContentIdIsZero = train_dataframe[train_dataframe.content_type_id==0]



#print(trainWhereContentIdIsZero)



#trainDFWithContentIdAndAnsweredCorrectly = trainWhereContentIdIsZero[['content_id','answered_correctly']].groupby('content_id')



# Question I have is grouping by ContentId even necessary? We already filtered the train_dataframe to 

# only show where the Content_Type_Id is equal to Zero so this may not even be needed. (N.I.)



# Did you want to group by the answered correctly values instead? (N.I.)





# This will print the first values in each group 

#trainDFWithContentIdAndAnsweredCorrectly.first()



def prepare_features(col_name):

    df = train_dataframe[train_dataframe.content_type_id==0][[col_name,'answered_correctly']].groupby(col_name).agg(['count','sum'])

    df.columns=['total', 'positive']

    df = df.astype('uint64')

    df['negative'] = df['total']-df['positive']

    return df

    
questions_dataframe = prepare_features('content_id')

questions_dataframe
train_dataframe.content_id.unique()
users_dataframe = prepare_features('user_id')

users_dataframe.head()
class NaiveBayes:

    def __init__(self, features, threshold=20):

        assert type(features)==dict, 'parameter features is not a dictionary!'

        for f in features.keys():

            assert type(features[f])==pd.core.frame.DataFrame, 'Wrong datatype for {0}. Each entry of the dictionary must contain a pandas DataFrame'.format(f)

            assert list(features[f].columns)==['total', 'positive', 'negative'], 'wrong columns in {0} DataFrame'.format(f)

        self.THRESHOLD = threshold

        self.features = features

        self.prior_probability = {}

        one_feature = list(features.keys())[0]

        self.prior_probability['negative'] = features[one_feature]['negative'].sum()/features[one_feature]['total'].sum()

        self.prior_probability['positive'] = features[one_feature]['positive'].sum()/features[one_feature]['total'].sum()

        

    def predict(self, data):

        assert data.keys()==self.features.keys(), "Keys doesn't match!"

        data_len = len(data[list(data.keys())[0]])

        # pos and neg are the priors for positive and negative classes

        pos = np.array([self.prior_probability['positive'] for _ in range(data_len)])

        neg = np.array([self.prior_probability['negative'] for _ in range(data_len)])

        # multiply the prior probability by the likelihood of each feature

        for d in data.keys():

            feature = pd.DataFrame({'id':data[d]})

            counts=pd.merge(feature,self.features[d],left_on='id',right_index=True,how='left').fillna(0).astype('uint64').values

            # counts.shape == (sample_len,4)

            # counts[:,0]==id ; counts[:,1]==total ; counts[:,2]==positive ; counts[:,3]==negative

            # e.g.: counts == array([[115,46,32,14],[124,10,7,3],[115,46,32,14]],dtype=uint64)

            updatable = np.where(counts[:,1]>self.THRESHOLD)[0]

            # e.g.: updatable == array([True,False,True])

            pos[updatable] *= counts[updatable,2]/counts[updatable,1]

            neg[updatable] *= counts[updatable,3]/counts[updatable,1]

        return pos/(pos+neg)
naivebayes = NaiveBayes({'questions': questions_dataframe, 'users':users_dataframe})
test_questions = example_test['content_id']

test_users = example_test['user_id']

test_rowids = example_test['row_id']



answered_correctly = naivebayes.predict({'questions':test_questions, 'users':test_users})

prediction = pd.DataFrame({'row_id':test_rowids, 'answered_correctly': answered_correctly})



prediction
env = riiideducation.make_env()

iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:

    test_questions = test_df['content_id']

    test_users = test_df['user_id']

    answered_correctly = naivebayes.predict({'questions':test_questions, 'users':test_users})

    test_df['answered_correctly'] = answered_correctly

    env.predict(test_df.loc[test_df['content_type_id']==0,['row_id','answered_correctly']])