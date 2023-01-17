import pandas as pd

import numpy as np

import riiideducation

import matplotlib.pyplot as plt
%%time

train = pd.read_feather('../input/riiid-feather-dataset/train.feather')

questions = pd.read_feather('../input/riiid-feather-dataset/questions.feather')

lectures = pd.read_feather('../input/riiid-feather-dataset/lectures.feather')

example_test = pd.read_feather('../input/riiid-feather-dataset/example_test.feather')

example_sample_submission = pd.read_feather('../input/riiid-feather-dataset/example_sample_submission.feather')
class NaiveBayes:

    def __init__(self, features, threshold=10):

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
def prepare_features(col_name):

    df = train[train.content_type_id==0][[col_name,'answered_correctly']].groupby(col_name).agg(['count','sum'])

    df.columns=['total', 'positive']

    df = df.astype('uint64')

    df['negative'] = df['total']-df['positive']

    return df
question_df = prepare_features('content_id')

question_df.head()
plt.hist((question_df['positive']/question_df['total']).values, bins =30)

plt.show()
user_df = prepare_features('user_id')

user_df.head()
plt.hist((user_df['positive']/user_df['total']).values, bins =30)

plt.show()
nb = NaiveBayes({'question':question_df, 'user':user_df})
test_questions = example_test['content_id']

test_users = example_test['user_id']

test_rowids = example_test['row_id']



answered_correctly = nb.predict({'question':test_questions, 'user':test_users})

prediction = pd.DataFrame({'row_id':test_rowids, 'answered_correctly': answered_correctly})



prediction
env = riiideducation.make_env()

iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:

    test_questions = test_df['content_id']

    test_users = test_df['user_id']

    answered_correctly = nb.predict({'question':test_questions, 'user':test_users})

    test_df['answered_correctly'] = answered_correctly

    env.predict(test_df.loc[test_df['content_type_id']==0,['row_id','answered_correctly']])