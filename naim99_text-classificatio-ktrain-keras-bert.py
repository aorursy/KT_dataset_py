!pip3 install ktrain==0.2.2
from sklearn.model_selection import train_test_split
import pandas as pd

train_df = pd.read_csv("../input/nlp-getting-started/train.csv")
train_df.head()
random_seed = 12342

x_train, x_val, y_train, y_val = train_test_split(train_df['text'], train_df['target'], shuffle=True, test_size = 0.2, random_state=random_seed, stratify=train_df['target'])
import ktrain

from ktrain import text
(x_train_bert,  y_train_bert), (x_val_bert, y_val_bert), preproc = text.texts_from_array(x_train=x_train, y_train=y_train,

                                                                                         x_test = x_val, y_test=y_val,

                                                                                          class_names= ["0", "1"],

                                                                                          preprocess_mode='bert',

                                                                                          

                                                                                          maxlen=65, 

                                                                                          max_features=35000)
model = text.text_classifier('bert', train_data=(x_train_bert, y_train_bert), preproc=preproc)

learner = ktrain.get_learner(model, train_data=(x_train_bert, y_train_bert), val_data=(x_val_bert, y_val_bert), batch_size=16)
learner.lr_find()             # briefly simulate training to find good learning rate

   
learner.lr_plot()
learner.autofit(1e-5)
learner.validate(val_data=(x_val_bert, y_val_bert), class_names=['No Disaster', 'Disaster'])
# getting predictor variable

predictor = ktrain.get_predictor(learner.model, preproc)
learner.print_layers()
test_df = pd.read_csv("../input/nlp-getting-started/test.csv")

test_df["target"] = predictor.predict(test_df["text"].tolist())



test_df = test_df[["id", "target"]]

test_df.head()
test_df.to_csv("submisssions.csv", index=False)