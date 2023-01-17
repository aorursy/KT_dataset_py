# Pick any one of the following Twi models
#MODEL = "Ghana-NLP/abena-base-akuapem-twi-cased" # (Akuapem ABENA) mBERT fine-tuned on JW300 Akuapem Twi, cased
#MODEL = "Ghana-NLP/abena-base-asante-twi-uncased" # (Asante ABENA) Akuapem ABENA fine-tuned on Asante Twi Bible, uncased
MODEL = "Ghana-NLP/distilabena-base-akuapem-twi-cased" # (Akuapem DistilABENA) DistilmBERT fine-tuned on JW300 Akuapem Twi, cased
#MODEL = "Ghana-NLP/distilabena-base-v2-akuapem-twi-cased" # (Akuapem DistilABENA V2) DistilmBERT fine-tuned on JW300 Akuapem Twi with Twi-only tokenizer trained from scratch, cased
#MODEL = "Ghana-NLP/distilabena-base-asante-twi-uncased" # (Asante DistilABENA) Akuapem DistilABENA fine-tuned on Asante Bible, uncased
#MODEL = "Ghana-NLP/distilabena-base-v2-asante-twi-uncased" # (Asante DistilABENA V2) Akuapem DistilABENA V2 fine-tuned on Asante Bible, uncased
#MODEL = "Ghana-NLP/robako-base-akuapem-twi-cased" # (Akuapem RoBAKO) RoBERTa trained from scratch on JW300 Akuapem Twi, cased [note - use <mask> not [MASK] to represent blank in sentence]
#MODEL = "Ghana-NLP/robako-base-asante-twi-uncased" # (Asante RoBAKO) Akuapem RoBAKO fine-tuned on Asante Twi Bible, uncased [note - use <mask> not [MASK] to represent blank in sentence]
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model=MODEL,
    tokenizer=MODEL
)

print(fill_mask("kwame yɛ panyin [MASK].")) # if using ABENA

#print(fill_mask("Saa tebea yi maa me papa <mask>.")) # if using BAKO

print(fill_mask(" Mayɛ basaa, da mu no nyinaa mede [MASK] nenam ")) # if using ABENA

#print(fill_mask("Eyi de ɔhaw kɛse baa <mask> hɔ.")) # if using BAKO
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForMaskedLM.from_pretrained(MODEL)
input_ids = tokenizer("Mayɛ basaa, da mu no nyinaa mede awerɛhow nenam", return_tensors="pt")["input_ids"] # these are indices of tokens in the vocabulary
print(input_ids)
decoded_tokens = [tokenizer.decode(int(el)) for el in input_ids[0]]
print(decoded_tokens)
import torch
import numpy as np

def get_embedding(in_str,model):
    input_ids = torch.tensor(tokenizer.encode(in_str)).unsqueeze(0)  # Batch has size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[1]  # The embedding vectors are a tuple of length equal to number of layers
    embedding_vecs = last_hidden_states[-1].detach().numpy()[0] # these vectors are n_tokens by 768 in size
    CLS_embedding_vec = embedding_vecs[0] # the CLS token is usually used as the average representation for classification
    average_vec = np.average(embedding_vecs[1:],axis=0) # averaging remaining vectors instead for similarity task yields slightly better results
    return average_vec

from transformers import AutoConfig
config = AutoConfig.from_pretrained(MODEL)
config.output_hidden_states=True
model = AutoModelForMaskedLM.from_pretrained(MODEL,config=config)
vec = get_embedding("Eyi de ɔhaw kɛse baa fie hɔ",model)
print("The vector representation of the sentence is:")
print(vec)
print("The shape of the vector is:")
print(vec.shape)
import pandas

data_df = pandas.read_csv("../input/twi-sentiment-analysis-unit-dataset/sentiment_analysis_unit_dataset.csv")
data_df = data_df.sample(frac=1) # shuffle
print(data_df)
train_data = data_df[:14]["Sentence"].values # use 14 out of the 20 as training, i.e., val ratio of 30%
train_labels = data_df[:14]["Label (1 is +ve)"].values
test_data = data_df[14:]["Sentence"].values # use 6 out of the 20 as testing
test_labels = data_df[14:]["Label (1 is +ve)"].values
print(test_data)
print("Checking testing data:")
print(test_data)
print(test_labels)
X_train_list = [get_embedding(sent,model) for sent in train_data] # vectorize/generate features for training
X_train = np.asarray(X_train_list)
y_train = train_labels
print("Training data shape is:")
print(X_train.shape)
X_test_list = [get_embedding(sent,model) for sent in test_data] # vectorize/generate features for testing
X_test = np.asarray(X_test_list)
y_test = test_labels
print("Testing data shape is:")
print(X_test.shape)
from sklearn.neighbors import KNeighborsClassifier # use a simple sklearn nearest neighbor classifier

CLF = KNeighborsClassifier(n_neighbors=1)
CLF.fit(X_train, y_train)

y_pred = CLF.predict(X_test)
print(y_pred)
np.average(y_test==y_pred)