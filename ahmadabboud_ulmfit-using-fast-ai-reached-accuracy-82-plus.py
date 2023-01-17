# prepare the notebook
%reload_ext autoreload
%autoreload 2
%matplotlib inline
# import necessary libraries
import numpy as np
from fastai.text import *
from pathlib import Path
# View current working directory
print(f"Current directory: {Path.cwd()}")
print(f"Home directory: {Path.home()}")
path=Path.cwd()
path=path
path
# Download Dataset
#import kaggle
#kaggle.api.authenticate()
#kaggle.api.dataset_download_files('crowdflower/twitter-airline-sentiment', path=path, unzip=True)
#Prepare Dataframe
df_org = pd.read_csv('../input/twitter-airline-sentiment/Tweets.csv')
df_org.rename(columns={'airline_sentiment':'label'},inplace=True)               
df=df_org[['label','text']]
np.random.seed(2020)
df['is_valid']=np.random.choice([True, False], len(df_org), p=[0.9,0.1 ]) # Seperate 10% for test
df.head()
df.info()
# Save to clean version CSV
df[['label','text','is_valid']].to_csv(path/'Tweets.csv',index=False)
#Show the first text item 
df['text'][1]
import seaborn as sns
df.label.value_counts().plot(kind='pie',autopct='%1.0f')
# Sentiment distribution
sns.countplot(x='label',data=df,palette='viridis')
#sentiment distribution over airelines 
plt.figure(figsize=(12,7))
sns.countplot(x='airline',hue='label',data=df_org,palette='rainbow')
# selecting bunch size depends on the memory size of your PC
bs=48
data_lm = (TextList.from_csv(path, 'Tweets.csv', cols='text') 
            .split_by_rand_pct(0.1,seed=2020)
           #We randomly split and keep 10% (10,000 reviews) for validation
            .label_for_lm()           
           #We want to do a language model so we label accordingly
            .databunch(bs=bs))
#Save DataBunch object
data_lm.save('data_lm.pkl')
data_lm = load_data(path, 'data_lm.pkl', bs=bs)
# Lets have a look at the first item of the training set
data_lm.train_ds[0][0]
data_lm.train_ds[0][0].data[:10]
data_lm.show_batch()
data_lm.vocab.itos[:20]
# Slanted triangular learning rates (STLR), which first linearly increases the learning rate and then linearly decays it
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

learn.lr_find()
learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7)) # lr should be 4*1e-2 at the stiffest slope
#The momentum is the first beta in Adam (or the momentum in SGD/RMSProp). When you pass along (0.8,0.7) it means going from 0.8 to0.7 during the warmup then from 0.8 to 0.7 in the annealing, but it only changes the first beta in Adam
#fit_one_cycle equivalent to the Adam optimizer’s (beta_2, beta_1) (notice the order) parameters, where beta_1 is the decay rate for the first moment, and beta_2 for the second
learn.save('fit_head')
learn.load('fit_head');
learn.unfreeze()
learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
learn.save('fine_tuned')
learn.load('fine_tuned');
TEXT = "I liked this airline because"
N_WORDS = 40
N_SENTENCES = 2
print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
learn.save_encoder('fine_tuned_enc')
data_clas=TextClasDataBunch.from_csv(path, 'Tweets.csv',vocab=data_lm.vocab)

data_clas.save('data_clas.pkl')
data_clas = load_data(path, 'data_clas.pkl', bs=bs)
data_clas.show_batch()
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)

#Show the learner structure 
learn
# Transfer learned encoder from previous language model
learn.load_encoder('fine_tuned_enc')
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7)) 
learn.save('first')
learn.load('first')
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))   #?? why 1e-2/(2.6**4)
learn.save('second')
learn.load('second');
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
learn.save('third')
learn.load('third');
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
learn.save('Fourth')
learn.load('Fourth');
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
learn.predict("I really loved that airline, it was awesome!")
# Prepare Interpreter
interp = ClassificationInterpretation.from_learner(learn)
# Confusion Matrix
interp.plot_confusion_matrix()
