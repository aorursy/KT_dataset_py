# data preprocessing for two classes: classes=['history', 'tech']

!mkdir data && wget https://www.dropbox.com/s/7qimiz727clhj6s/ua_wiki.zip && unzip ua_wiki.zip -d data
!rm -rf ./ua_wiki

!mkdir ./ua_wiki

!mv ./data/train ./ua_wiki/

!mv ./data/test ./ua_wiki/

!ls -GFlash --color ./ua_wiki/
!ls ./ua_wiki/train/history|head
!ls ./ua_wiki/train/tech|head
!ls ./ua_wiki/test/history|head
!ls ./ua_wiki/test/tech|head
!pip install --upgrade tensorflow

!pip install ktrain
import ktrain

from ktrain import text

ktrain.__version__
#

(x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder('./ua_wiki/', 

                                                                       maxlen=75, 

                                                                       max_features=10000,

                                                                       preprocess_mode='bert',

                                                                       train_test_names=['train', 'test'], 

                                                                       #val_pct=0.1,

                                                                       classes=['history', 'tech'])
# Create a Model and Wrap in Learner Object

model = text.text_classifier('bert', (x_train, y_train) , preproc=preproc)

learner = ktrain.get_learner(model, 

                             train_data=(x_train, y_train), 

                             val_data=(x_test, y_test), 

                             batch_size=32)
# STEP 3: Train the Model

learner.fit_onecycle(2e-5, 3, checkpoint_folder='../kaggle/working/saved_weights')
learner.validate(val_data=(x_test, y_test))
# Inspecting Misclassifications

learner.view_top_losses(n=3, preproc=preproc)
# Making Predictions on New Data

p = ktrain.get_predictor(learner.model, preproc)
p.get_classes()
# Predicting label for the text

p.predict("Козацькі повстання були формою боротьби за волю українського народу.")
# Predicting label for the text

p.predict("У моєму компьютері встановлено потужний процессор.")
p.save('../kaggle/working/ua_wiki_pred')
fin_bert_model = ktrain.load_predictor('../kaggle/working/ua_wiki_pred')
# still works

fin_bert_model.predict('Програмування є досить важливою галуззю ІТ індустрії.')
# still works

fin_bert_model.predict('Польська шляхта досить часто не враховувала інтереси козаків.')
# still works

fin_bert_model.predict('Чи мала Україна шанси на успіх у визвольній боротьбі?')
# still works

fin_bert_model.predict('Футбол не є нашим національним спортом.')
# still works

fin_bert_model.predict("Чи можеми ми виробляти компьютери.")