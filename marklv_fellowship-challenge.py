# magics 

%reload_ext autoreload

%autoreload 2

%matplotlib inline
# install nlp data augmentation library and dependencies

!pip install git+https://github.com/makcedward/nlpaug.git numpy matplotlib python-dotenv; pip install nltk
# fastai library

from fastai import *

from fastai.text import *

# plotting confusion matrix

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels



# data visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns



# nlp word augmentation libraries

import nlpaug.augmenter.word as naw

from nlpaug.util import Action



# classification report

from sklearn.metrics import classification_report
# set data path

path = Path('../input/twitter-airline-sentiment')

path.ls()
# create pandas dataframe and 

df = pd.read_csv(path/'Tweets.csv')
# check dataframe shape

df.shape
# look at first 5 data samples

df.head()
# Check number of posible lables

senti_types = df['airline_sentiment'].unique()

print('No. of sentiment labels: {}'.format(len(senti_types)))
# set graph style

sns.set(palette='Pastel2', style='dark')
# Plot class distribution

dist = df.groupby('airline_sentiment')



# plot label distribution to see if dataset is un-balanced

sns.countplot(df.airline_sentiment)

plt.xlabel('Sentiment Distribution')

plt.show()



dist.size()
# plot how often an airline has been mentioned

sns.countplot(df.airline)

plt.title('Airline mention distribution')

plt.tick_params(axis='x', rotation=45)

plt.xlabel('')

plt.show()
print('Tweet distribution by airline \n', df.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False))
df.groupby(['airline', 'airline_sentiment']).size().unstack().plot(kind='bar', stacked=True, colormap='Pastel1' )
# sentiment distribution by airline

plt.figure(1, figsize=(15,10))

airlines = list(df['airline'].unique())

for i in df.airline.unique():

    idxs = airlines.index(i)

    plt.subplot(2,3,idxs+1)

    pl_df = df[df.airline==i]

    count = pl_df['airline_sentiment'].value_counts()

    index = [1,2,3]

    plt.bar(index, count, color='lightblue')

    plt.xticks(index, ['negative','neutral','positive'])

    plt.ylabel('')

    plt.title('Sentiment Distribution of ' + i)
#get the number of negative reasons

df['negativereason'].nunique()



nr_count = dict(df['negativereason'].value_counts(sort=False))

def reason_count(airline):

    if airline=='All':

        a = df

    else:

        a = df[df['airline']==airline]

    count = dict(a['negativereason'].value_counts())

    unique_reason=list(df['negativereason'].unique())

    unique_reason=[x for x in unique_reason if str(x) != 'nan']

    reason_frame=pd.DataFrame({'reasons':unique_reason})

    reason_frame['count']=reason_frame['reasons'].apply(lambda x: count[x])

    return reason_frame



def plot_reason(airline):

    

    a = reason_count(airline)

    count = a['count']

    index = range(1,(len(a)+1))

    plt.bar(index,count, color='lightblue')

    plt.xticks(index,a['reasons'],rotation=90)

    plt.ylabel('count')

    plt.xlabel(' ')

    plt.title('Cause of complaint for ' + airline)

    

plot_reason('All')

plt.figure(2,figsize=(15, 12))

for i in airlines:

    indices= airlines.index(i)

    plt.subplot(2,3,indices+1)

    plt.subplots_adjust(hspace=0.9)

    plot_reason(i)
# first we ensure that we're only working with the columns we need from the dataframe

df = df[['airline_sentiment', 'text']]
# determine split point

train_len = int(len(df)*0.85)



# set training dataframe

df_train = df[:train_len]



# set test dataframe

df_test = df[train_len:]



# test if split left samples behind

assert len(df) == sum([len(df_train), len(df_test)])



# print lengths

print(len(df_train), len(df_test), sum([len(df_train), len(df_test)]))
df_pos = df_train.loc[df_train.airline_sentiment == 'positive']

df_neu = df_train.loc[df_train.airline_sentiment == 'neutral']

df_neg = df_train.loc[df_train.airline_sentiment == 'negative']



print('There are {} positive tweets, {} neutral tweets and {} negative tweets'.format(len(df_pos), len(df_neu), len(df_neg)))
# calculate how many positive and neutral augmented tweets we want to create to balance the datasets

pos_aug = len(df_neg)-len(df_pos)

neu_aug = len(df_neg)-len(df_neu)



print('Tweets to create: {} postive tweets, {} neutral tweets'.format(pos_aug,neu_aug))
# data augmentation function



def train_aug(n_aug, source_df, train_df):

    # print and store initial df length

    train_len = len(train_df)

    print('Data length before augmentation: {} \n'.format(train_len))

    

    for n in range(n_aug):

        # get a random tweet from the source dataframe

        randn = random.randint(0, len(source_df)-1)

        row = source_df.iloc[randn].values

        # do synonym augmentation

        aug = naw.SynonymAug(aug_src='wordnet')

        augmented_text = aug.augment(row[1])

        # add new row to training dataframe

        new_row = {'airline_sentiment': row[0], 'text': augmented_text}

        train_df = train_df.append(new_row, ignore_index=True)

    

    new_added = len(train_df) - train_len

    # print 

    print('{} items added to training datafrane. Training dataframe now has {} rows'.format(new_added, len(train_df)))

    

    return train_df
# add postive tweets to training data

df_train = train_aug(pos_aug, df_pos, df_train)
# add neutral tweets to training data

df_train = train_aug(neu_aug, df_neu, df_train)
len_df_all = len(df_train) + len(df_test)

print('Train length: {}, Test length: {}, Combined data length {}'.format(len(df_train), len(df_test), len_df_all))
# concat train and test set for language model

df_lm = pd.concat([df_train, df_test])

print(len(df_lm))
# set batchsize and backprop through time

bs = 256

bptt = 80
# prepare/pre-process langauge model data

data_lm = (TextList.from_df(df_lm, cols='text')

                   .split_by_rand_pct(0.05, seed=42) # default for valid_pct=0.2, we're using 0.05, so we have more data to train lm

                   .label_for_lm() # the text is the label

                   .databunch())
# look at dataset length after split and pre-processing

print('Vocab size: {}, training set length: {}, validation set length: {}'.format(len(data_lm.vocab.itos), len(data_lm.train_ds), len(data_lm.valid_ds)))
# save data

data_lm.save('data_lm.pkl')
# look at data batch

data_lm.show_batch()
# backward lm data - *the ../../working is specific to working in a Kaggle notebook

data_lm_bwd = load_data(path, '../../working/data_lm.pkl', bs=bs, bptt=bptt, backwards=True)
# let's look at the data in reverse

data_lm_bwd.show_batch()
# initiate learner

lm_learner = language_model_learner(data_lm, AWD_LSTM, model_dir='../../working', metrics=[accuracy, error_rate])
# find the learning rate

lm_learner.lr_find()

lm_learner.recorder.plot(suggestion=True)
# use mixed-precision training so that we can train faster and use a higher batch-size

lm_learner = lm_learner.to_fp16(clip=0.1)
# train 

lm_learner.fit_one_cycle(1, 4e-2, moms=(0.8,0.7), wd=0.1)
lm_learner.recorder.plot_lr()
# unfreeze model

lm_learner.unfreeze()
lm_learner.fit_one_cycle(10, 2e-3, moms=(0.8,0.7), wd=0.1)
lm_learner.save_encoder('fwd_enc_sg')
# init backwards model

lm_backward = language_model_learner(data_lm_bwd, AWD_LSTM, metrics=[accuracy, error_rate]).to_fp16(clip=0.1)
# train for 1 epoch

lm_backward.fit_one_cycle(1, 4e-2, moms=(0.8,0.7), wd=0.1)

# unfreeze model 

lm_backward.unfreeze()

# train for another 10 epochs

lm_backward.fit_one_cycle(10, 2e-3, moms=(0.8,0.7), wd=0.1)
# set model directory so that encoder saves in the right directory

lm_backward.model_dir = '../../working'
lm_backward.save_encoder('bwd_enc_sg')
# load the data for the forward classifier model using the language model vocabulary

fwd_data_clas = (TextList.from_df(df_train, vocab=data_lm.vocab, cols='text')

                          .split_by_rand_pct(0.15, seed=42) # 20% goes to validation set

                          .label_from_df(cols='airline_sentiment')

                          .databunch(bs=128))
fwd_data_clas.save('fwd_data_clas.pkl')
# look at a batch of the data

fwd_data_clas.show_batch()
# saving the backward data for the backwards model

bwd_data_clas = load_data(path, '../../working/fwd_data_clas.pkl', bs=bs, backwards=True)
# look at a batch of the backward model data

bwd_data_clas.show_batch()
# initiate the classifier for the forward data

fwd_clas = text_classifier_learner(fwd_data_clas, AWD_LSTM, drop_mult=0.5, pretrained=False, metrics=[accuracy, error_rate])
# move encoder to correct directory

!mv ../../working/fwd_enc_sg.pth ./fwd_enc_sg.pth
# create models directory and move encoder there

!mkdir models; mv ./fwd_enc_sg.pth models/fwd_enc_sg.pth
# load the language model encoder into the classifer

fwd_clas.load_encoder('fwd_enc_sg')
# train last layer

lr = 1e-1

fwd_clas.fit_one_cycle(1, lr, moms=(0.8,0.7), wd=0.1)
# train last two layers

fwd_clas.freeze_to(-2)

lr /= 2

fwd_clas.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)
# train last three layers

fwd_clas.freeze_to(-3)

lr /= 2

fwd_clas.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)
# train all layers

fwd_clas.unfreeze()

lr /= 5

fwd_clas.fit_one_cycle(2, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)
fwd_clas.save('fwd_clas_cp')
# train the classifer a little more

fwd_clas.fit_one_cycle(3, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)
# train the classifer a little more

#fwd_clas.fit_one_cycle(3, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)
fwd_clas.save('fwd_clas_fin')
bwd_clas = text_classifier_learner(bwd_data_clas, AWD_LSTM, drop_mult=0.5, pretrained=False, metrics=[accuracy, error_rate])
bwd_clas.model_dir = '../../working'
bwd_clas.load_encoder('bwd_enc_sg')
# train only the last layer

bwd_clas.fit_one_cycle(1, lr, moms=(0.8,0.7), wd=0.1)
# train the last two layers

bwd_clas.freeze_to(-2)

lr /= 2

bwd_clas.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)
# train the last 3 layers

bwd_clas.freeze_to(-3)

lr /= 2

bwd_clas.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)
# train all the layers

bwd_clas.unfreeze()

lr /= 5

bwd_clas.fit_one_cycle(2, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)
# train a little bit more

bwd_clas.fit_one_cycle(3, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)
# and a little bit more

bwd_clas.fit_one_cycle(3, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)
# train an extra 15 epochs

bwd_clas.fit_one_cycle(15, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)
# checkpoint

bwd_clas.save('bwd_clas_fin')
# get forward classifier predictions and accuracy for validation set

fwd_preds, fwd_targets = fwd_clas.get_preds(ordered=True)

print('Forward classifier results (validation set): \nValidation accuracy: {:.2f}, Validation error rate: {:.2f}'.format(accuracy(fwd_preds, fwd_targets), error_rate(fwd_preds, fwd_targets)))
# get backward classifier predictions and accuracy for validation set

bwd_preds, bwd_targets = bwd_clas.get_preds(ordered=True)

print('Backward classifier results (validation set): \nValidation accuracy: {:.2f}, Validation error rate: {:.2f}'.format(accuracy(bwd_preds, bwd_targets), error_rate(bwd_preds, bwd_targets)))
# get combined(mean) predictions

ensemble_preds = (fwd_preds + bwd_preds)/2

# get combined(mean) accuracy on validation set

print('Ensemble classifier results (validation set): \nValidation accuracy: {:.2f}, Validation error rate: {:.2f}'.format(accuracy(ensemble_preds, fwd_targets), error_rate(ensemble_preds, fwd_targets)))
# load the test data using datablock api - a somewhat hacky but quick way to load the test set into the model

data_test = (TextList.from_df(df_test, vocab=data_lm.vocab, cols='text')

                          .split_subsets(train_size=0.01, valid_size=0.99) # the hacky part

                          .label_from_df(cols='airline_sentiment')

                          .databunch(bs=128))
# load data into forward classifier

fwd_clas.data = data_test
# hacky solution comes at a price - sometimes not all labels are loaded into the model

assert fwd_clas.data.c == 3
# save test data

data_test.save('test_data.pkl')
# save backward test data

bwd_test = load_data(path, '../../working/test_data.pkl', bs=bs, backwards=True)
# load data into backward classifier

bwd_clas.data = bwd_test
# hacky solution comes at a price - sometimes not all labels are loaded into the model

assert bwd_clas.data.c == 3
# get foward model predictions on test set

test_preds, test_targs = fwd_clas.get_preds(ds_type=DatasetType.Valid)

print('Forward classifier results (test set): \nTest accuracy: {:.2f}, Test error rate: {:.2f}'.format(accuracy(test_preds, test_targs), error_rate(test_preds,test_targs)))
# get backward model predictions on test set

bwdt_preds, bwdt_targs = bwd_clas.get_preds(ds_type=DatasetType.Valid)

print('Backward classifier results (test set): \nTest accuracy: {:.2f}, Test error rate: {:.2f}'.format(accuracy(bwdt_preds, bwdt_targs),error_rate(bwdt_preds, bwdt_targs)))
# get bi-directional model predictions

ensemble_preds = (test_preds + bwdt_preds)/2

# Get ensemble error

# higher precision reporting - to be able to inspect model performance closer

print('Ensemble classifier results (test set): \nTest accuracy: {:.4f}, Test error rate: {:.4f}'.format(accuracy(ensemble_preds, test_targs),error_rate(ensemble_preds, test_targs)))
# prepare data for confusion matrix

y_pred = np.array(np.argmax(ensemble_preds, axis=1))

y_true = np.array(test_targs)
# quick check if all classes are present

print(y_pred)

print(y_true)
# from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

cmap = 'Reds'

cm = confusion_matrix(y_true, y_pred)



def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=cmap, normalize=False):

  acc = np.trace(cm) / float(np.sum(cm))

  err_rate = 1 - acc

  plt.figure(figsize=(8, 6))

  plt.imshow(cm, interpolation='nearest', cmap=cmap)

  plt.title(title)

  plt.colorbar()

  if target_names is not None:

      tick_marks = np.arange(len(target_names))

      plt.xticks(tick_marks, target_names, rotation=45)

      plt.yticks(tick_marks, target_names)

  if normalize:

      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  thresh =  cm.max() / 1.5 if normalize else cm.max() / 2

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):      

      if normalize:

          plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                   horizontalalignment="center",

                   color="grey" if cm[i, j] > thresh else "grey")

      else:

          plt.text(j, i, "{:,}".format(cm[i, j]),

                   horizontalalignment="center",

                   color="grey" if cm[i, j] > thresh else "grey")

  plt.tight_layout()

  plt.ylabel('True label')

  plt.xlabel('Predicted label\naccuracy={:0.4f}; error_rate={:0.4f}'.format(acc, err_rate))

  plt.show()

    

    

plot_confusion_matrix(cm=cm, target_names=['negative', 'neutral', 'positive'], 

                      title='Classifation Confusion Matrix')
plot_confusion_matrix(cm=cm, target_names=['negative', 'neutral', 'positive'], 

                      title='Classifation Distribution in Percent(Normalized)', normalize=True)