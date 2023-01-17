import pandas as pd

from keras import Sequential

from keras import layers

from keras import regularizers

import numpy as np

from string import punctuation

import pyprind

from collections import Counter

from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from numpy.random  import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(1)
def read_caps(fname):

    """Load the captions into a dataframe"""

    vn = []

    cap = []

    df = pd.DataFrame();

    with open(fname) as f:

        for line in f:

            pairs = line.split()

            vn.append(pairs[0])

            cap.append(pairs[1])

        df['video']=vn

        df['caption']=cap

    return df





# load the captions

cap_path = 'C:/Users/Monisri/Documents/sem 2/ML/dev-set/dev-set/dev-set_video-captions.txt'

df_cap=read_caps(cap_path)



# load the ground truth values

label_path = 'C:/Users/Monisri/Documents/sem 2/ML/dev-set/dev-set/'

labels=pd.read_csv(label_path+'dev-set_ground-truth.csv')
print(df_cap)
print(labels)
counts = Counter()

pbar = pyprind.ProgBar(len(df_cap['caption']), title='Counting word occurrences')

for i, cap in enumerate(df_cap['caption']):

    # replace punctuations with space

    # convert words to lower case 

    text = ''.join([c if c not in punctuation else ' ' for c in cap]).lower()

    df_cap.loc[i,'caption'] = text

    pbar.update()

    counts.update(text.split())
print(counts)

print(len(counts))
df_cap.head()
len_token = len(counts)

tokenizer = Tokenizer(num_words=len_token)

print(len_token)
tokenizer.fit_on_texts(list(df_cap.caption.values)) #fit a list of captions to the tokenizer

#the tokenizer vectorizes a text corpus, by turning each text into either a sequence of integers 
print(len(tokenizer.word_index))
one_hot_res = tokenizer.texts_to_matrix(list(df_cap.caption.values),mode='binary')

sequences = tokenizer.texts_to_sequences(list(df_cap.caption.values))
#Just to visualise some stuff in sequences and counts

print(sequences[0]) # prints location of words from caption 0 'blonde woman is massaged tilt down'

print(counts['blonde']) # no. of occurences of 'blonde'

n=3

print('Least Common: ', counts.most_common()[:-n-1:-1])       # n least common elements

print('Most Common: ',counts.most_common(n))                     # n most common elements
max_len = 50
print(sequences[0]) # length of 1st sequence
X_seq = np.zeros((len(sequences),max_len))

for i in range(len(sequences)):

    n = len(sequences[i])

    if n==0:

        print(i)

    else:

        X_seq[i,-n:] = sequences[i]

X_seq.shape
print(X_seq[5999,:])
print(X_seq[0,:]) # length of 1st sequence after padding the caption with zeros.
from numpy import genfromtxt

C3D_data = genfromtxt('C:/Users/Monisri/Documents/sem 2/ML/dev-set/dev-set/dev-set_features/C3D/C3D.csv', delimiter=',')

ColorHistogram = genfromtxt('C:/Users/Monisri/Documents/sem 2/ML/dev-set/dev-set/dev-set_features/ColorHistogram/ColorHistogram.csv', delimiter=',')

HMP = genfromtxt('C:/Users/Monisri/Documents/sem 2/ML/dev-set/dev-set/dev-set_features/HMP/HMP.csv', delimiter=',')

IV3 = genfromtxt('C:/Users/Monisri/Documents/sem 2/ML/dev-set/dev-set/dev-set_features/InceptionV3/IV3.csv', delimiter=',')

LBP = genfromtxt('C:/Users/Monisri/Documents/sem 2/ML/dev-set/dev-set/dev-set_features/LBP/LBP.csv', delimiter=',')

C3D_data[np.isnan(C3D_data)] = 0

ColorHistogram[np.isnan(ColorHistogram)] = 0

HMP[np.isnan(HMP)] = 0

IV3[np.isnan(IV3)] = 0

LBP[np.isnan(LBP)] = 0

#ORB[np.isnan(ORB)] = 0

IV3 = pd.read_csv('C:/Users/Monisri/Documents/sem 2/ML/dev-set/dev-set/dev-set_features/InceptionV3/IV3.csv')

IV3.head(5)

IV3['1'] = IV3['1'].str.split('-').str[0]
from pandas import DataFrame



IV3 = DataFrame(IV3.values [1::3], index=IV3.index[1::3])

IV3.shape

IV3 = IV3.values



def Get_score(Y_pred,Y_true):

    '''Calculate the Spearmann"s correlation coefficient'''

    Y_pred = np.squeeze(Y_pred)

    Y_true = np.squeeze(Y_true)

    if Y_pred.shape != Y_true.shape:

        print('Input shapes don\'t match!')

    else:

        if len(Y_pred.shape) == 1:

            Res = pd.DataFrame({'Y_true':Y_true,'Y_pred':Y_pred})

            score_mat = Res[['Y_true','Y_pred']].corr(method='spearman',min_periods=1)

            print('The Spearman\'s correlation coefficient is: %.5f' % score_mat.iloc[1][0])

        else:

            for ii in range(Y_pred.shape[1]):

                Get_score(Y_pred[:,ii],Y_true[:,ii])
from sklearn.preprocessing import StandardScaler

Y = labels[['short-term_memorability','long-term_memorability']].values # targets



X = HMP 

#HMP = 3992

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42) # random state for reproducability



print('X_train', X_train.shape)

print('X_test', X_test.shape)

print('Y_train', Y_train.shape)

print('Y_test', Y_test.shape)



from sklearn.kernel_ridge import KernelRidge

rng = np.random.RandomState(0)

clf = KernelRidge(alpha=1)

model = clf.fit(X_train, Y_train) 
predictions = model.predict(X_test)

print(predictions.shape)

Get_score(predictions, Y_test)
from sklearn.preprocessing import StandardScaler

Y = labels[['short-term_memorability','long-term_memorability']].values # targets

#scaler = StandardScaler()

#print(scaler.fit(C3D_data))

X = C3D_data # sequences





X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=50) # random state for reproducability



print('X_train', X_train.shape)

print('X_test', X_test.shape)

print('Y_train', Y_train.shape)

print('Y_test', Y_test.shape)



from sklearn.kernel_ridge import KernelRidge

rng = np.random.RandomState(0)

clf = KernelRidge(alpha=1)

model = clf.fit(X_train, Y_train) 
predictions = model.predict(X_test)

print(predictions.shape)

Get_score(predictions, Y_test)
from sklearn.preprocessing import StandardScaler

Y = labels[['short-term_memorability','long-term_memorability']].values # targets

X = one_hot_res # sequences



X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=30) # random state for reproducability



print('X_train', X_train.shape)

print('X_test', X_test.shape)

print('Y_train', Y_train.shape)

print('Y_test', Y_test.shape)



from sklearn.kernel_ridge import KernelRidge

rng = np.random.RandomState(0)

clf = KernelRidge(alpha=1)

model = clf.fit(X_train, Y_train) 
predictions = model.predict(X_test)

print(predictions.shape)

Get_score(predictions, Y_test)
from sklearn.preprocessing import StandardScaler

Y = labels[['short-term_memorability','long-term_memorability']].values # targets

X = one_hot_res # sequences



X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42) # random state for reproducability



print('X_train', X_train.shape)

print('X_test', X_test.shape)

print('Y_train', Y_train.shape)

print('Y_test', Y_test.shape)



from sklearn.linear_model import Ridge

clf = Ridge()



clf.set_params(alpha=20)

model = clf.fit(X_train, Y_train) 



predictions = model.predict(X_test)

print(predictions.shape)

Get_score(predictions, Y_test)
from sklearn.preprocessing import StandardScaler

Y = labels[['short-term_memorability']].values # targets

X = one_hot_res # sequences



X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42) # random state for reproducability



print('X_train', X_train.shape)

print('X_test', X_test.shape)

print('Y_train', Y_train.shape)

print('Y_test', Y_test.shape)



from sklearn.linear_model import Ridge

clf = Ridge()



clf.set_params(alpha=20)

model = clf.fit(X_train, Y_train) 







predictions = model.predict(X_test)

print(predictions.shape)

Get_score(predictions, Y_test)







#To download predictions data

predictions = model.predict(one_hot_res)

print(predictions)

#Get_score(predictions, Y_test)

arr = np.array(predictions)



df = pd.DataFrame(data=arr.flatten())

#predictions = pd.DataFrame({'Column1':predictions[:,0],'Column2':predictions[:,1]})



#predictions = pd.DataFrame(predictions, index=1)



df_out = pd.merge(labels,df,how = 'left', left_index = True , right_index = True)

print(df_out)

df_out.to_csv('Predictions_final.csv')

from sklearn.preprocessing import StandardScaler

Y = labels[['long-term_memorability']].values # targets

X = one_hot_res # sequences



X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42) # random state for reproducability



print('X_train', X_train.shape)

print('X_test', X_test.shape)

print('Y_train', Y_train.shape)

print('Y_test', Y_test.shape)



from sklearn.linear_model import Ridge

clf = Ridge()



clf.set_params(alpha=20)

model = clf.fit(X_train, Y_train) 





predictions = model.predict(X_test)

print(predictions.shape)

Get_score(predictions, Y_test)
#To download predictions data

predictions = model.predict(one_hot_res)

print(predictions)

#Get_score(predictions, Y_test)

arr = np.array(predictions)



df = pd.DataFrame(data=arr.flatten())

#predictions = pd.DataFrame({'Column1':predictions[:,0],'Column2':predictions[:,1]})



#predictions = pd.DataFrame(predictions, index=1)



df_out = pd.merge(labels,df,how = 'left', left_index = True , right_index = True)

print(df_out)

df_out.to_csv('Predictions_final2.csv')

from sklearn.preprocessing import StandardScaler

Y = labels[['short-term_memorability','long-term_memorability']].values # targets

X = HMP # sequences



X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42) # random state for reproducability



print('X_train', X_train.shape)

print('X_test', X_test.shape)

print('Y_train', Y_train.shape)

print('Y_test', Y_test.shape)



from sklearn.linear_model import Ridge

clf = Ridge()



clf.set_params(alpha=1)

model = clf.fit(X_train, Y_train) 





predictions = model.predict(X_test)

print(predictions.shape)

Get_score(predictions, Y_test)
from sklearn.preprocessing import StandardScaler

Y = labels[['short-term_memorability','long-term_memorability']].values # targets

X = C3D_data # sequences



X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42) # random state for reproducability



print('X_train', X_train.shape)

print('X_test', X_test.shape)

print('Y_train', Y_train.shape)

print('Y_test', Y_test.shape)



from sklearn.linear_model import Ridge

clf = Ridge()



clf.set_params(alpha=100)

model = clf.fit(X_train, Y_train) 



predictions = model.predict(X_test)

print(predictions.shape)

Get_score(predictions, Y_test)
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression



Y = labels[['short-term_memorability','long-term_memorability']].values # targets

X = one_hot_res # sequences



#HMP = 3992

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42) # random state for reproducability





print('X_train', X_train.shape)

print('X_test', X_test.shape)

print('Y_train', Y_train.shape)

print('Y_test', Y_test.shape)

clf = LinearRegression()

model = clf.fit(X_train, Y_train) 

predictions = model.predict(X_test)

print(predictions.shape)

Get_score(predictions, Y_test)
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression



Y = labels[['short-term_memorability','long-term_memorability']].values # targets

X = HMP # sequences



#HMP = 3992

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42) # random state for reproducability

#scaler = StandardScaler()

#scaler.fit(X_train)

#X_train = scaler.transform(X_train)

print('X_train', X_train.shape)

print('X_test', X_test.shape)

print('Y_train', Y_train.shape)

print('Y_test', Y_test.shape)

clf = LinearRegression()

model = clf.fit(X_train, Y_train) 

predictions =(model.predict(X_test))

print(predictions.shape)

Get_score(predictions, Y_test)
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression



Y = labels[['short-term_memorability','long-term_memorability']].values # targets

X = C3D_data # sequences



#HMP = 3992

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42) # random state for reproducability

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

print('X_train', X_train.shape)

print('X_test', X_test.shape)

print('Y_train', Y_train.shape)

print('Y_test', Y_test.shape)

clf = LinearRegression()

model = clf.fit(X_train, Y_train) 
predictions = model.predict(X_test)

print(predictions.shape)

Get_score(predictions, Y_test)
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor



Y = labels[['short-term_memorability','long-term_memorability']].values # targets

X = one_hot_res # sequences



#HMP = 3992

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42) # random state for reproducability



print('X_train', X_train.shape)

print('X_test', X_test.shape)

print('Y_train', Y_train.shape)

print('Y_test', Y_test.shape)

clf = RandomForestRegressor()

model = clf.fit(X_train, Y_train) 
predictions = model.predict(X_test)

print(predictions.shape)

Get_score(predictions, Y_test)
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor



Y = labels[['short-term_memorability','long-term_memorability']].values # targets

X = HMP # sequences



#HMP = 3992

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42) # random state for reproducability



print('X_train', X_train.shape)

print('X_test', X_test.shape)

print('Y_train', Y_train.shape)

print('Y_test', Y_test.shape)

clf = RandomForestRegressor()

model = clf.fit(X_train, Y_train) 
predictions = model.predict(X_test)

print(predictions.shape)

Get_score(predictions, Y_test)
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor



Y = labels[['short-term_memorability','long-term_memorability']].values # targets

X = C3D_data # sequences



#HMP = 3992

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42) # random state for reproducability



print('X_train', X_train.shape)

print('X_test', X_test.shape)

print('Y_train', Y_train.shape)

print('Y_test', Y_test.shape)

clf = RandomForestRegressor()

model = clf.fit(X_train, Y_train) 
predictions = model.predict(X_test)

print(predictions.shape)

Get_score(predictions, Y_test)