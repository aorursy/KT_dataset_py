!pip install fastai -U
from fastai.text.all import *
path = '../input/nlp-getting-started'
train_path = path + '/train.csv'
test_path = path + '/test.csv'
train = pd.read_csv(train_path)
train.head()
train.info()
train.keyword.value_counts()
train.location.describe()
train.location.value_counts()
dls = TextDataLoaders.from_df(
    train, 
    path, 
    valid_pct=0.2, 
    text_col = ['text', 'keyword', 'location'],
    label_col = 'target'
)
dls.show_batch()
learn = text_classifier_learner(dls, 
                                AWD_LSTM, 
                                drop_mult=0.5, 
                                metrics=accuracy, 
                                cbs=ShowGraphCallback())
learn.fine_tune(4, 1e-2)
test = pd.read_csv(test_path)
test.head()
learn.predict(test.iloc[0])
results = pd.DataFrame(columns=['id', 'target'])
for i, row in test.iterrows():
    print(i)
    pass_id = row['id']
    pred = learn.predict(row);
    
    
    results.loc[i] = [pass_id, pred[0]]
        
results
results
results.to_csv('submission.csv', index=False)