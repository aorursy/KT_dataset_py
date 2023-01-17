import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import random, os



%matplotlib inline

plt.style.use("dark_background")
def seed_everything(seed=1234): 

    random.seed(seed) 

    os.environ['PYTHONHASHSEED'] = str(seed) 

    tf.random.set_seed(seed)

    np.random.seed(seed) 

    

seed_everything(2020)
# kaggle 

main_dir = "../input/biobytes-contest"



# Jupyter lab

# main_dir = "."
!ls {main_dir}
with open(f"{main_dir}/Main_data.txt") as f:

    temp = f.read()

    temp = temp[:temp.index('>', 1)]

    print ("When printed it looks like so:\n\n".upper() + temp)

    print ("Raw string:\n\n".upper() + repr(temp))
data = pd.read_csv(

    f"{main_dir}/Main_data.txt", 

    # each 'unit' internally is seperated by a \n

    sep='\n', names=['Name', 'P_Seq', 'Target'], 

    # Each new 'unit' starts with a '>'

    lineterminator='>', 

    # The txt file doesn't contain an index

    # first column is the name of protein

    index_col=False

)



# let's see if we were successful in reading to pandas DF

for name, value in zip(['Name', 'P_seq', 'Target'], data.iloc[0]):

    print (f"{name}: {repr(value)}\n")
# Apply a function to a Dataframe elementwise.

data = data.applymap(lambda x: x.rstrip('\r'))



for name, value in zip(['Name', 'P_seq', 'Target'], data.iloc[0]):

    print (f"{name}: {repr(value)}\n")
test = (

    pd.read_csv(

        f"{main_dir}/Test_data.txt", 

        sep='\n', names=['Name', 'P_Seq'], 

        lineterminator='>', index_col=False)

    .applymap(lambda x: x.rstrip('\r'))

)



for name, value in zip(['Name', 'P_seq', 'Target'], test.iloc[0]):

    print (f"{name}: {repr(value)}\n")
sample_sub = pd.read_csv(f"{main_dir}/Sample_Solution.csv")

sample_sub.head()
# shapes of loaded Frames

data.shape, test.shape, sample_sub.shape
# places where our above assumption is false

(data.apply(lambda x: len(x[1]) - len(x[2]), axis=1) != 0).sum()
# The Protein Class, Dunno if something like this 

# indeed exists. Correct me if I am wrong ofc!

data['P_Main_Class'] = data.Name.str.extract(":(\w)")

data['P_Sub_Class'] = data.Name.str.extract("(.+):\w")



# length of sequence

data['Seq_len'] = data.P_Seq.apply(len)



# Number of unique peptides per sequence

data['Uniq_seq_Count'] = data.P_Seq.apply(lambda x: len(set(x)))



# Number of peptides bound

data['B_Site_Count'] = data.Target.str.count('1')



# Percentage of peptides that are bound

data['B_Site_percent'] = data['B_Site_Count'] / data['Seq_len']



# Count of peptides that are unbound

data['Non_B_Site_Count'] = data['Seq_len'] - data['B_Site_Count']



data.head()
uniq = set()

for _, seq in data.P_Seq.iteritems():

    uniq |= set(seq)

    

print (uniq)

len(uniq)
# some basic stats

data.describe()
f, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))

(data[['Seq_len', 'Uniq_seq_Count', 'B_Site_Count', 'Non_B_Site_Count']]

 .plot(kind='kde', subplots=True, ax=ax, grid=True));
data.B_Site_percent.plot(kind='kde', figsize=(15, 5), title='Percentage Bound for sequences');
data.plot(kind='box', figsize=(20, 10), subplots=True);
temp = data.P_Main_Class.value_counts()

print ("Unique classes per Total Classes: {}/{}".format(len(temp), len(data)))



plt.figure(figsize=(15, 5))

plt.yticks(range(0, 21, 2))

plt.bar(temp.index, temp.values)



for index, value in temp.iteritems():

    plt.text(index, value, value)
pmc_mapper = dict(zip(temp[temp > 1].index, range(len(temp))))

print (pmc_mapper)
temp = data['P_Sub_Class'].value_counts()

print ("Unique classes per Total Classes: {}/{}".format(len(temp), len(data)))



plt.figure(figsize=(15, 5))

plt.bar(temp.index, temp.values)

plt.xticks(rotation=90)



for index, value in temp.iteritems():

    plt.text(index, value, value)
psc_mapper = dict(zip(temp[temp > 1].index, range(len(temp))))

print (psc_mapper)
temp = data.apply(lambda x: np.array(list(x[1]))[np.array(list(x[2])).astype(bool)], axis=1)

temp = temp.apply(pd.Series).stack().reset_index(level=1, drop=True)



freq_occured = data.P_Seq.apply(lambda x: pd.Series(list(x)).value_counts()).sum()

freq_bound = temp.value_counts()



freq = pd.merge(

    pd.DataFrame(freq_occured).reset_index().rename({"index": "Peptide", 0: "Occured"}, axis=1), 

    pd.DataFrame(freq_bound).reset_index().rename({"index": "Peptide", 0: "Bound"}, axis=1),

    on='Peptide'

)



freq['Percent'] = freq['Bound'] / freq['Occured'] * 100



freq.head()
(freq[['Peptide', 'Percent']]

 .set_index('Peptide')

 .sort_values('Percent')

 .plot(kind='bar', figsize=(20, 5), title='Percentage Plot (SORTED)', rot=0)

);
ax = freq.set_index("Peptide")['Occured'].plot(

    kind='bar', figsize=(20, 5), 

    title='Peptide Occuring vs Being Bound',

    legend=True,

)



freq.set_index("Peptide")['Bound'].plot(kind='bar', ax=ax, color='r', legend=True)



for index, percent, value in freq[['Percent', 'Occured']].itertuples():

    ax.text(index-0.2, value, f"{percent:.0f}%")
freq['Percent_bin'] = pd.cut(freq['Percent'], bins=5).cat.codes

freq = freq.sort_values(['Percent_bin', 'Occured'], ascending=[False, True])



(freq.set_index("Peptide")[['Occured', 'Bound']]

 .plot(kind='bar', 

       stacked=True,

       title='Peptides sorted by Importance (Percent & Rareness)',

      figsize=(20, 5), rot=0)

);
len(sample_sub), test.P_Seq.map(len).sum()
sub = test['P_Seq'].apply(list).explode().reset_index()

sub = sub.rename({'index': 'Seq_No', "P_Seq": 'Peptide'}, axis=1)

sub['Id'] = sub.index

sub = sub.iloc[:, [-1, 0, 1]]

sub.head()
sub["Expected"] = sub.groupby("Seq_No")['Id'].transform(lambda x: (np.random.random(len(x)) < 0.35).astype(int))

sub.head()
ax = sub.Expected.value_counts().plot(

    kind='bar', figsize=(10, 5), 

    color=['g', 'r'], title='Naive Prediction'

)



ax.set_xticks([0, 1])

ax.set_xticklabels(['Unbound', 'Bound'], rotation=0);
sub[['Id', 'Expected']].to_csv("Naive_submission.csv", index=False)
# mapper containing the frequencies for each peptide

mapper = dict(zip(freq.Peptide, freq.Percent))

print (mapper)
sub['Expected'] = (

    sub.Peptide.map(mapper) / 100 

    # calculate random for each peptide, say 'A' together

    > sub.groupby('Peptide')['Id'].transform(lambda x: np.random.random(len(x)))

).astype(int)
# let's simply check if the random function has been used properly

# I was confused with using >, < 

# here both should match for any value of temp (Nearly)

temp = np.random.choice(list(mapper.keys()))

mapper[temp] / 100, sub.loc[sub.Peptide == temp, 'Expected'].mean()
ax = sub.Expected.value_counts().plot(

    kind='bar', figsize=(10, 5), 

    color=['g', 'r'], 

    title='Peptide Freq based Prediction'

)



ax.set_xticklabels(['Unbound', 'Bound'], rotation=0);
sub[['Id', 'Expected']].to_csv("Peptide_Based_Fprediction.csv", index=False)
from sklearn.metrics import roc_auc_score, accuracy_score



Y = data.Target.apply(list).explode().values.astype(int)

y_hat_naive = (np.random.random(len(Y)) < 0.35).astype(int)



y_hat_freq_based = (

    data.P_Seq.apply(list).explode().reset_index(drop=True).map(mapper) / 100 

    > 

    (data.P_Seq.apply(list).explode().to_frame().reset_index().groupby("P_Seq")

     .transform(lambda x: np.random.random(len(x)))['index'])

).astype(int)



print ("Our estimates on TRAINING DATA:")

print ("\n\tNaive model accuracy score: {:12.2f}\n\tNaive model ROC score: {:17.2f}"

       .format(

           accuracy_score(Y, y_hat_naive), 

           roc_auc_score(Y, y_hat_naive))

      )



print ("\n\tNuanced Naive model accuracy score: {:.2f}\n\tNuanced Naive model ROC score: {:9.2f}"

       .format(

           accuracy_score(Y, y_hat_freq_based), 

           roc_auc_score(Y, y_hat_freq_based))

      )
flat_data = pd.DataFrame({

    "P_Seq": data.P_Seq.apply(list).explode(),

    "Target": data.Target.apply(list).explode().astype(int)

})



flat_data.head()
peptide_mapper = dict(zip(list(mapper.keys()), range(len(mapper))))

print (peptide_mapper)
# simplest possible linear model

tf.keras.backend.clear_session()

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape=(len(peptide_mapper),)))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=tf.keras.metrics.AUC())

model.summary()
hist = model.fit(

    tf.one_hot(flat_data.P_Seq.map(peptide_mapper), depth=20), 

    flat_data.Target, 

    validation_split=0.2, 

    callbacks=tf.keras.callbacks.EarlyStopping(patience=10),

    epochs=100, 

    verbose=0)



print("Train Best ROC_AUC Score: {:.2f}".format(hist.history['auc'][-1]))

print ("Val Best ROC_AUC Score: {:6.2f}".format(hist.history['val_auc'][-1]))



pd.DataFrame(hist.history).iloc[:, [1, 3]].plot(figsize=(20, 5), title='Model Performance');
sub['Expected'] = (

    model.predict(tf.one_hot(

        sub.Peptide.map(peptide_mapper), 

        depth=len(peptide_mapper)))

)
sub[['Id', 'Expected']].to_csv("Linear_model_one_hot.csv", index=False)
flat_data['Percent'] = flat_data.P_Seq.map(mapper).astype(float) / 100

flat_data = flat_data.merge(data.iloc[:, 3:7], right_index=True, left_index=True)

flat_data['Position'] = flat_data.groupby(flat_data.index)['P_Seq'].transform(lambda x: np.arange(len(x)) / len(x))

flat_data.head()
# Main and sub class for the test data as well

test['P_Main_Class'] = test.Name.str.extract(":(\w)")

test['P_Sub_Class'] = test.Name.str.extract("(.+):\w")



# length of sequence

test['Seq_len'] = test.P_Seq.apply(len)



# Number of unique peptides per sequence

test['Uniq_seq_Count'] = test.P_Seq.apply(lambda x: len(set(x)))



# merge the test data with Sub dataFrame

sub = sub.set_index("Seq_No").merge(test.iloc[:, 2:], right_index=True, left_index=True)



# we reuse column 'expected' as our percentage expected

sub['Expected'] = sub.Peptide.map(mapper) / 100



# rename the columns accordingly

sub = sub.rename({"Peptide": "P_Seq", "Expected": "Percent"}, axis=1)



# the position of peptide in sequence

sub['Position'] = sub.groupby(sub.index)['P_Seq'].transform(lambda x: np.arange(len(x)) / len(x))



# how does it look?

sub.head(3)
def process_flat_df(

    data, 

    ohc=[('P_Sub_Class', len(psc_mapper)), ('P_Main_Class', len(pmc_mapper))], 

    dc=['P_Main_Class', 'P_Sub_Class'], 

    seq_shift=0,

    pep_freq=None, 

    as_df=False):

    

    '''One hot enocdes the categorical data after mapping them to the respective mappers.

    The numeric columns, futher more are scaled.

    ohc      -> One hot columns, other than `P_Seq`

    dc       -> columns to drop

    pep_freq -> does the dataframe contain the peptide frequency, (we pass in the series)

    as_Df    -> Return output as a df or tensor

    '''

    

    df = data.copy()

    

    # copy and create mapper for null values

    pmapper = peptide_mapper.copy()

    pmapper['0'] = max(peptide_mapper.values()) + 1

    

    # reusing pipeline in the future, skip for now

    if pep_freq is not None:

        freq_mat = get_freq(pep_freq[0], fit=pep_freq[1])

        df = df.merge(freq_mat, left_index=True, right_index=True, how='left')

        df.iloc[:, -freq_mat.shape[1]:] = df.iloc[:, -freq_mat.shape[1]:] / df['Seq_len'].values.reshape(-1, 1)

    

    # normalize numeric values

    df['Seq_len'] = df['Seq_len'] / max(sub.Seq_len.max(), flat_data.Seq_len.max())

    df['Uniq_seq_Count'] = df['Uniq_seq_Count'] / 20

    

    # ordinal encoding the categorical data

    # df['P_Seq'] = df['P_Seq'].map(pmapper).fillna(len(pmapper))

    df['P_Main_Class'] = df['P_Main_Class'].map(pmc_mapper).fillna(len(pmc_mapper))

    df['P_Sub_Class'] = df['P_Sub_Class'].map(psc_mapper).fillna(len(psc_mapper))  



    one_hot = []

    

    for i in range(-seq_shift, seq_shift+1):

        df[f'P_Seq_{i}'] = df.groupby(df.index)['P_Seq'].shift(-i).fillna('0').map(pmapper)

        

    seq_one_hot = tf.reduce_sum(tf.one_hot(df.iloc[:, -((seq_shift*2)+1):], depth=len(pmapper)), axis=1)

    seq_one_hot = seq_one_hot / ((seq_shift * 2) + 1)

    one_hot.append(seq_one_hot)

    

    for key, depth in ohc:

#         one_hot.append(tf.one_hot(df[key], depth=depth+1))

        one_hot.append(tf.one_hot(df[key], depth=depth))

    

    one_hot = tf.concat(one_hot, axis=1)

    

    # drop the columns we had one hot encoded to

    # always drop ['P_Seq'] since it is 1hencoded

    dc = dc + df.columns[df.columns.str.contains("P_Seq", na=False)].tolist()

    df.drop(dc, axis=1, inplace=True)

    

    if as_df:

        return (df.reset_index(drop=True).merge(

                pd.DataFrame(one_hot.numpy()), right_index=True, left_index=True))

    

    else:

        return tf.concat([df, one_hot], axis=1)
# working good?

shape = process_flat_df(

    flat_data.drop("Target", 1), 

    ohc=[('P_Sub_Class', len(psc_mapper))]

).shape[1]



shape
# linear model with more features added

tf.keras.backend.clear_session()

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape=(shape,)))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=tf.keras.metrics.AUC())



hist = model.fit(

    process_flat_df(flat_data.drop("Target", 1), 

                    ohc=[('P_Sub_Class', len(psc_mapper))]

    ),

    flat_data.Target, 

    validation_split=0.2, 

    callbacks=tf.keras.callbacks.EarlyStopping(patience=10),

    epochs=100, 

    verbose=0)



print("Train Best ROC_AUC Score: {:.2f}".format(hist.history['auc'][-1]))

print ("Val Best ROC_AUC Score: {:6.2f}".format(hist.history['val_auc'][-1]))



pd.DataFrame(hist.history).iloc[:, [1, 3]].plot(figsize=(20, 5), title='Model Performance');
predictions = model.predict(

    process_flat_df(

        sub.drop(['Id'], axis=1), 

        ohc=[('P_Sub_Class', len(psc_mapper))])

)



sub['Expected'] = predictions

sub.head()
sub[['Id', 'Expected']].to_csv("Linear_model_with_meta.csv", index=False)
def get_freq(series, min_thresh=0.65, fit=False):

    'Returns a 20 * n matrix containing frequencies for each sequence'

    from sklearn.feature_extraction.text import CountVectorizer

    global cnt

    

    if fit:

        cnt = CountVectorizer(analyzer='char', ngram_range=(1, 2), min_df=min_thresh, lowercase=False)

        return pd.DataFrame(cnt.fit_transform(series).todense(), columns=cnt.get_feature_names())

    else:

        return pd.DataFrame(cnt.transform(series).todense(), columns=cnt.get_feature_names())



    # pervious code, counts only 1 character

    '''

    matrix = pd.DataFrame()

    for key in peptide_mapper.keys():

        matrix[key] = series.str.count(key)

    return matrix

    '''
# defining the columns we would be dropping

dc=[

    'P_Sub_Class', 

#     'P_Main_Class', 

#     'Seq_len', 

#     'Uniq_seq_Count', 

#     'Percent',

   ]



# defining columns we would be one hot encoding along with their mappers

ohc=[

    ('P_Sub_Class', len(psc_mapper)), 

#     ('P_Main_Class', len(pmc_mapper)),

]



# working good?

temp = process_flat_df(

    flat_data.drop("Target", 1), 

    ohc=ohc,

    dc=dc,

    pep_freq=(data.P_Seq, True),

    as_df=True

).head(3)



shape = temp.shape[1]

temp
# linear model with more features added

tf.keras.backend.clear_session()

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape=(shape,)))

model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', 

              optimizer=tf.keras.optimizers.Adam(0.0025), 

              metrics=tf.keras.metrics.AUC())



hist = model.fit(

    process_flat_df(

        flat_data.drop("Target", 1), 

        ohc=ohc,

        dc=dc,

        pep_freq=(data.P_Seq, True)

    ),

    

    flat_data.Target, 

    batch_size=128,

    validation_split=0.2, 

    callbacks=[

        tf.keras.callbacks.EarlyStopping(patience=8, monitor='val_auc', mode='max'),

        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', patience=5, factor=0.5, mode='max')],

    epochs=100, 

    verbose=0)



print("Train Best ROC_AUC Score: {:.2f}".format(hist.history['auc'][-1]))

print ("Val Best ROC_AUC Score: {:6.2f}".format(hist.history['val_auc'][-1]))



pd.DataFrame(hist.history).iloc[:, [1, 3]].plot(figsize=(20, 5), title='Model Performance');
predictions = model.predict(

    process_flat_df(

        sub.drop(["Expected", "Id"], 1), 

        ohc=ohc,

        dc=dc,

        pep_freq=(test.P_Seq, False)),

)



sub['Expected'] = predictions

sub.head(3)
sub[['Id', 'Expected']].to_csv("Linear_model_with_Seq_freq.csv", index=False)
# Seq in future (and in past) to consider as input

SHIFT = 2



# defining the columns we would be dropping

dc=[

#     'P_Sub_Class', 

    'P_Main_Class',

#     'Seq_len',

    'Uniq_seq_Count',

#     'Percent',

#     'Position',

   ]



# defining columns we would be one hot encoding along with their mappers

ohc=[

#     ('P_Sub_Class', len(psc_mapper)),

#     ('P_Main_Class', len(pmc_mapper)),

]



# working good?

temp = process_flat_df(

    flat_data.drop("Target", 1), 

    ohc=ohc,

    dc=dc,

    # we can tune the value of seq_shift 

    # to see which one performs better

    pep_freq=(data.P_Seq, True),

    seq_shift=SHIFT,

    as_df=True

).head(5)



shape = temp.shape[1]

print ("The number of columns in data that would be fit to our model is:", shape)



# how does the one hot encoded labels look?

temp.iloc[:3, 29:29+21]
flat_data['P_Seq'].head(5).map(peptide_mapper).values
# resetting the shift value

SHIFT = 3



# linear model with more features added

tf.keras.backend.clear_session()

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape=(shape,)))

model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', 

              optimizer=tf.keras.optimizers.Adam(0.005), 

              metrics=tf.keras.metrics.AUC())



hist = model.fit(

    process_flat_df(

        flat_data.drop("Target", 1), 

        ohc=ohc,

        dc=dc,

        pep_freq=(data.P_Seq, True),

        seq_shift=SHIFT),

    flat_data.Target,

    

    validation_split=0.25,

    batch_size=128,

    callbacks=[

        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_auc', mode='max'),

        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', patience=5, factor=0.25, mode='max')],

    epochs=100, 

    verbose=0)



print("Train Best ROC_AUC Score: {:.2f}".format(hist.history['auc'][-1]))

print ("Val Best ROC_AUC Score: {:6.2f}".format(hist.history['val_auc'][-1]))



pd.DataFrame(hist.history).iloc[:, [1, 3]].plot(figsize=(20, 5), title='Model Performance');
predictions = model.predict(

    process_flat_df(

        sub.drop(["Expected", "Id"], 1), 

        ohc=ohc,

        dc=dc,

        pep_freq=(test.P_Seq, False)

    ),

)



sub['Expected'] = predictions

sub.head(3)
sub[['Id', 'Expected']].to_csv("Linear_model_with_hist.csv", index=False)