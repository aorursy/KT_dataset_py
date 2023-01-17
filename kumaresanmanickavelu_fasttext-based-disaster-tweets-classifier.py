import fasttext 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import re
def preprocess_text(text) :

    

    # replace regular charecters

    text = text.replace('\n','').replace('\t','').replace('#','')

    

    # replace twitter handles 

    text = re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z0-9]+[A-Za-z0-9-_]+)','USER',text)

    

    # replace twitter links

    text = re.sub("http\S+", "URL", text)

    

    return text



preprocess_text("VICTORINOX SWISS ARMY DATE WOMEN'S RUBBER MOP WATCH 241487 http://t.co/yFy3nkkcoH http://t.co/KNEhVvOHVK  ")

 




# load training data 

train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")





# fast text expects train data in the following format. 



# Labels       Document

# __label__one document_one

# __label__two document_two



# prepare fasttext label & document.

train_df["label"] = train_df["target"].apply(lambda x: "__label__" + str(x))

train_df["processed_text"] = train_df["text"].apply(lambda x: preprocess_text(x))



fasttext_input = train_df[["label","processed_text"]]



fasttext_input.head()
# prepare fasttext train and valid set. 

np.random.seed(0)

msk = np.random.rand(len(train_df)) < 0.8

fasttext_train = fasttext_input[msk]

fasttext_valid = fasttext_input[~msk]
fasttext_train.to_csv("fasttext_train.csv", sep=" ", quotechar=" ", header=False, index=False)

fasttext_valid.to_csv("fasttext_valid.csv", sep=" ", quotechar=" ", header=False, index=False)



# train the fast text model.

model = fasttext.train_supervised(input="fasttext_train.csv")



#validate the model

model.test("fasttext_valid.csv")
fasttext_valid["predicted"] = fasttext_valid["processed_text"].apply(lambda x: model.predict(x)[0][0])



fasttext_errors = fasttext_valid[fasttext_valid['label'] != fasttext_valid['predicted']]

fasttext_errors.describe()
#false negative

# disaster-tweet predicted as not-a-disaster

pd.set_option('display.max_colwidth', -1)

fasttext_errors[fasttext_errors['label'] == "__label__1"]["processed_text"].head(25)
#false positive

# non-disaster-tweet predicted as a disaster-tweet

fasttext_errors[fasttext_errors['label'] == "__label__0"]["processed_text"].head(25)
def predict_target(model, text) :

    text = preprocess_text(text)

    return (0,1)[model.predict(text)[0][0]=='__label__1']
#read files 

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")



#predict

sample_submission["target"] = test_df["text"].apply(lambda x: predict_target(model,x))



#submit

sample_submission.to_csv("submission.csv", index=False)

!head submission.csv