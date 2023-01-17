# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import xgboost as xgb



from sklearn.preprocessing import LabelEncoder





data_dir = '/kaggle/input/allstate-claims-severity/'



train_data_path = data_dir+'train.csv'

test_data_path = data_dir+'test.csv'

submission_csv_path = data_dir+'sample_submission.csv'


# Label Encode all categorical features

def get_labelEncoded_dataframes(data_dir):

    '''

    creates a label encoded dataframe out of the categorical features using sklearns's LabelEncoder

    saves new dataframe in object_dir

    skips creating new dataframe if already exists

    '''

    print('Label Encoding categorical features . . .')

    train_data = pd.read_csv(data_dir+'train.csv')

    test_data = pd.read_csv(data_dir+'test.csv')

    cat_cols = [x for x in train_data.columns if x.startswith('cat')]



    for col in cat_cols:

        le = LabelEncoder()

        train_data[col] = le.fit_transform(train_data[col])

        # update::

        # Test data had some values in some cateogorical features that were unseen in train data

        # the next 2 lines fix that :|

        test_data[col] = test_data[col].map(lambda s: 'UNK' if s not in le.classes_ else s)

        le.classes_ = np.append(le.classes_, 'UNK')

        test_data[col] = le.transform(test_data[col])

    return train_data, test_data





train_data, test_data = get_labelEncoded_dataframes(data_dir)

submission = pd.read_csv(submission_csv_path)
X = train_data.iloc[:,1:-1]

Y = train_data.iloc[:,-1]





# get categorical and continuous features names

cat_cols = [x for x in train_data.columns if x.startswith('cat')]

cont_cols = [x for x in train_data.columns if x.startswith('cont')]
    

from sklearn.feature_selection     import    f_regression, mutual_info_regression



# f_regression

##############

f_reg_res = {}

fval, pval = f_regression(X, Y)

for i,c in enumerate(X.columns):

  f_reg_res[c] = fval[i]



# sort the features according to f_regression scores

sorted_res = [[k,v] for k, v in sorted(f_reg_res.items(), key=lambda item: item[1])]

sorted(sorted_res, key = lambda x: x[1])



# remove features that scored too low

high_score_features_F = [x[0] for x in list(filter(lambda x: x[1]>100, sorted_res))]

print("features with f_regression score > 100")

print(high_score_features_F)







# mutual_information

####################

# sampling a subset of data, as mutual_info calculation is intensive

sample = train_data.sample(10000)

x = sample.iloc[:,:-1]

y = sample.iloc[:,-1]



mutinf_res = {}

mi = mutual_info_regression(x, y)

for i,c in enumerate(X.columns):

  mutinf_res[c] = mi[i]



# sort the features according to mutual_information scores

sorted_res = [[k,v] for k, v in sorted(mutinf_res.items(), key=lambda item: item[1])]

sorted(sorted_res, key = lambda x: x[1])



# remove features that scored too low

high_score_features_MI = [x[0] for x in list(filter(lambda x: x[1]>0.001, sorted_res))]

print("features with mutual_information score > 100")

print(high_score_features_MI)



# get intersection of features which score high on both of these tests

# i.e. we are discarding features that did not do well in both the tests

common_features_union = list(set(high_score_features_F).union(set(high_score_features_MI)))

print("# feautres selected: ", common_features_union.__len__())
print("Features that would be used: ", common_features_union)

print("# features: ", common_features_union.__len__())



import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, LeakyReLU

from keras.preprocessing import text

from keras import utils



# set hyperparameters for MLP

class NN:

    def __init__(self):

        self.in_shape = common_features_union.__len__()

        self.num_layers = 3

        self.nodes = [2048,1024, 1]

        self.activations = ['relu', 'relu', 'relu']

        self.dropouts = [0.2,0.15,0]

        self.loss = 'mean_squared_logarithmic_error'

        self.optimizer = keras.optimizers.RMSprop(5e-4)







def sequential_MLP(nn):

    model = Sequential()

    for i in range(nn.num_layers):

        if i==0: # add input shape if first layer

            model.add(Dense(nn.nodes[i], activation=nn.activations[i], input_shape=(nn.in_shape,) ))

        else:

            model.add(Dense(nn.nodes[i], activation=nn.activations[i]))

        if(nn.dropouts[i] != 0): # skip adding dropout if dropout == 0

            model.add(Dropout(rate=nn.dropouts[i]))            

    model.compile(optimizer=nn.optimizer, loss=nn.loss, metrics=['mae'])



    return model




nn = NN()

model = sequential_MLP(nn)





for i in range(45):

  if i%5 == 0: verbose=True

  else: verbose = False

  model.fit(X[common_features_union], Y, epochs=1, batch_size=512, validation_split=0.25, verbose=verbose)
test_predictions = model.predict(test_data[common_features_union])





submission = pd.read_csv('/kaggle/input/allstate-claims-severity/sample_submission.csv')

submission['loss'] = test_predictions

submission.to_csv('submission.csv', index=False)