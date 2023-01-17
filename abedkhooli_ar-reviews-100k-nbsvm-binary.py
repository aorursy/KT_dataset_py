# Please see the dataset description for additional info 

# current accuracy (balanced): see last run (this is the NBSVM binary implementation using ktrain) [not optimized]
import time 

print(f'Started: {time.ctime()}')
!pip install -U pip

!pip install "tensorflow_gpu>=2.0.0"  # cpu: pip3 install "tensorflo

!pip install transformers==2.5.1

!pip install ktrain 
# forked versions of eli5 and stellargraph (required by ktrain)

!pip install git+https://github.com/amaiya/eli5@tfkeras_0_10_1 

!pip install git+https://github.com/amaiya/stellargraph@no_tf_dep_082
import pandas as pd

pd.set_option('display.max_colwidth',200)

pd.set_option('display.max_rows',50)

df = pd.read_csv('/kaggle/input/arabic-100k-reviews/ar_reviews_100k.tsv',sep='\t')

df['label'].value_counts()
dfm = df[df['label']=='Mixed']

df = df.drop(dfm.index)

len(df)
df = df.sample(frac=1).reset_index(drop=True)

df.head()
df = pd.concat([df, df.label.astype('str').str.get_dummies()], axis=1, sort=False)

df.head()
df = df[['text','Negative','Positive']]
import ktrain

from ktrain import text 
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_df(df, 

                                                                   'text', # name of column containing review text

                                                                   label_columns=['Negative','Positive'],

                                                                   maxlen=150, 

                                                                   max_features=150000,

                                                                   random_state = 42,

                                                                   preprocess_mode='standard',

                                                                   val_pct=0.15,

                                                                   ngram_range=3)
model = text.text_classifier('nbsvm', (x_train, y_train) , preproc=preproc)

learner = ktrain.get_learner(model, 

                             train_data=(x_train, y_train), 

                             val_data=(x_test, y_test), 

                             batch_size=32)
# takes ssome time

#learner.lr_find(show_plot=True, max_epochs=2)
learner.autofit(4e-3, 3)
print(learner.validate(class_names=['Negative','Positive']))
print(learner.view_top_losses(n=1, preproc=preproc))
p = ktrain.get_predictor(learner.model, preproc)

p.predict('كان هذا الفندق باهظ الثمن والموظفين غير مهذبين.')
p.explain('كان هذا الفندق باهظ الثمن والموظفين غير مهذبين.')
print(f'Finished: {time.ctime()}')