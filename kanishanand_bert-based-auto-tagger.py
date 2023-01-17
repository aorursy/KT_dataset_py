!pip install -q "tensorflow_gpu>=2.0.0"
!pip install -q transformers
!pip install -q ktrain 
!pip3 install -q tornado==5
import eli5
import ktrain
import pandas as pd
import seaborn as sns
from ktrain import text
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
input_path = '../input/books-data.csv'
input = pd.read_csv(input_path)
input = input.drop_duplicates(subset=['isbn'])
input = input.fillna(" ")
len(input)
input.head(10)
input['text'] = input.title + " " + input.author + " " + input.description
input[input['text'].isnull()]
input.head()
len(input['tag'].unique())
label_list = input['tag'].unique()
label_list
plt.figure(figsize = (25,10))
pd.value_counts(input['tag']).plot(kind="bar")
train, val =  train_test_split(input, test_size = 0.2, random_state = 100)
len(train['tag'].unique())
len(val['tag'].unique())
len(train)
len(val)
train.head(10)
val_copy = val
MAX_LEN = 200
BATCH_SIZE = 6
train_text = train.text.tolist()
val_text = val.text.tolist()
train_tag = train.tag.tolist()
val_tag = val.tag.tolist()
MODEL_NAME = 'distilbert-base-uncased'
t = text.Transformer(MODEL_NAME, maxlen=MAX_LEN, classes=label_list)
trn = t.preprocess_train(train_text, train_tag)
val = t.preprocess_test(val_text, val_tag)
model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=BATCH_SIZE)
LR = 5e-5
EPOCHS = 10
history = learner.autofit(LR, EPOCHS)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
a = t.get_classes()
conf = learner.validate(class_names = a)
plt.figure(figsize = (20,20))
sns.heatmap(conf)
learner.view_top_losses(n=10, preproc=t)
predictor = ktrain.get_predictor(learner.model, preproc=t)
predictor.predict("harry potter")
check = val_copy.iloc[10]
check
check.tag
predictor.predict(check.text)
predictor.explain(check.text)
check = val_copy.iloc[830]
check
check.tag
predictor.predict(check.text)
predictor.explain(check.text)
predictor.save('model')