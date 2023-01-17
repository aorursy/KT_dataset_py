# install ktrain on Google Colab
!pip3 install ktrain
# import ktrain and the ktrain.text modules
import ktrain
from ktrain import text
ktrain.__version__
# fetch the dataset using scikit-learn
categories = ['alt.atheism', 'soc.religion.christian',
             'comp.graphics', 'sci.med']
from sklearn.datasets import fetch_20newsgroups
train_b = fetch_20newsgroups(subset='train',
   categories=categories, shuffle=True, random_state=42)
test_b = fetch_20newsgroups(subset='test',
   categories=categories, shuffle=True, random_state=42)

print('size of training set: %s' % (len(train_b['data'])))
print('size of validation set: %s' % (len(test_b['data'])))
print('classes: %s' % (train_b.target_names))

x_train = train_b.data
y_train = train_b.target
x_test = test_b.data
y_test = test_b.target
(x_train,  y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=x_train, y_train=y_train,
                                                                       x_test=x_test, y_test=y_test,
                                                                       class_names=train_b.target_names,
                                                                       preprocess_mode='bert',
                                                                       maxlen=350, 
                                                                       max_features=35000)
# you can disregard the deprecation warnings arising from using Keras 2.2.4 with TensorFlow 1.14.
model = text.text_classifier('bert', train_data=(x_train, y_train), preproc=preproc)
learner = ktrain.get_learner(model, train_data=(x_train, y_train), batch_size=6)
learner.fit_onecycle(2e-5, 4)
learner.validate(val_data=(x_test, y_test), class_names=train_b.target_names)
predictor = ktrain.get_predictor(learner.model, preproc)
predictor.get_classes()
predictor.predict(test_b.data[0:1])
# we can visually verify that our prediction of 'sci.med' for this document is correct
print(test_b.data[0])
# we predicted the correct label
print(test_b.target_names[test_b.target[0]])
# let's save the predictor for later use
predictor.save('/tmp/my_predictor')
# reload the predictor
reloaded_predictor = ktrain.load_predictor('/tmp/my_predictor')
# make a prediction on the same document to verify it still works
reloaded_predictor.predict(test_b.data[0:1])