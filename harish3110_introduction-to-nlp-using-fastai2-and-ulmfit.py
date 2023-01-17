# Installing and importing the necessary libraries 

!pip install fastai2 --quiet

!pip install kaggle --quiet



from fastai2.text.all import *



import warnings

warnings.filterwarnings('ignore')
path = Path('../input/nlp-getting-started/')

Path.BASE_PATH = path

path.ls()
train = pd.read_csv(path/'train.csv')

test = pd.read_csv(path/'test.csv')
train.head() 
train['target'].value_counts()
test.head()
print(f'The training set has {len(train)} records.')

print(f'The test set has {len(test)} records.')
# Let's take an example text from our training set to show a tokenization example



txt = train['text'].iloc[0]

txt
# Initializing the default tokenizer used in Fastai which is that of Spacy called `WordTokenizer`

spacy = WordTokenizer() 



# Wrapping the Spacy tokenizer with a custom Fastai function to make some custom changes to the tokenizer

tkn = Tokenizer(spacy) 



tkn(txt)
txts = L([i for i in train['text']])
# Setting up a tokenizer on the entire dataframe 'train'

tok = Tokenizer.from_df(train)

tok.setup(train)



toks = txts.map(tok)

toks[0]
tok.encodes(toks[0])
tok.decode(toks[0])
num = Numericalize()

num.setup(toks)

nums = toks.map(num)

nums[0][:10]
num.encodes(toks[0])
num.decode(nums[0][:10])
# dataset for fine-tuning language model which only needs the text data



df_lm = pd.concat([train, test], axis=0)[['text']]

df_lm.head()
dls_lm = DataBlock(

    blocks=TextBlock.from_df('text', is_lm=True),

    get_x=ColReader('text'), 

    splitter=RandomSplitter(0.1) 

    # using only 10% of entire comments data for validation inorder to learn more

)
dls_lm = dls_lm.dataloaders(df_lm, bs=64, seq_len=72)
dls_lm.show_batch(max_n=3)
# Saving the dataloader for fast use in the future



# torch.save(dls_lm, path/'disaster_tweets_dls_lm.pkl')
# To load the Dataloaders in the future



# dls_lm = torch.load(path/'disaster_tweets_dls_lm.pkl')
#fine-tuning wikitext LM to disaster tweets dataset



learn = language_model_learner(

    dls_lm, AWD_LSTM,

    metrics=[accuracy, Perplexity()]).to_fp16()
learn.model
learn.lr_find()
learn.fine_tune(5, 1e-2)
# Saving the encoder



learn.save_encoder('finetuned')
blocks = (TextBlock.from_df('text', seq_len=dls_lm.seq_len, vocab=dls_lm.vocab), CategoryBlock())

dls = DataBlock(blocks=blocks,

                get_x=ColReader('text'),

                get_y=ColReader('target'),

                splitter=RandomSplitter(0.2))
dls = dls.dataloaders(train, bs=64)
dls.show_batch(max_n=3)
len(dls.train_ds), len(dls.valid_ds)
learn = text_classifier_learner(dls, AWD_LSTM, metrics=[accuracy, FBeta(beta=1)]).to_fp16()

learn.load_encoder('finetuned')
learn.model
learn.fit_one_cycle(1, 1e-2)
# Applying gradual unfreezing of one layer after another



learn.freeze_to(-2)

learn.fit_one_cycle(1, slice(1e-3/(2.6**4),1e-2))
learn.freeze_to(-3)

learn.fit_one_cycle(1, slice(5e-3/(2.6**4),1e-2))
learn.unfreeze()

learn.fit_one_cycle(2, slice(1e-3/(2.6**4),3e-3))
learn.save('final_model')
learn.export()
sub = pd.read_csv(path/'sample_submission.csv')

sub.head()
dl = learn.dls.test_dl(test['text'])
preds = learn.get_preds(dl=dl)
# Let's view the output of a single row of data



preds[0][0].cpu().numpy()
# Since it's a multi-class problem and it uses softmax on the binary classes, 

# Need to calculate argmax of the output to get the best class as follows 



preds[0][0].cpu().argmax(dim=-1)
sub['target'] = preds[0].argmax(dim=-1)
sub.head()