### load dataset & dependencies
import os,math,io
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
#bureau_data = pd.read_csv('../input/bureau.csv')
train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')

train['DOCS'] = train['FLAG_DOCUMENT_2']+train['FLAG_DOCUMENT_3']+train['FLAG_DOCUMENT_4']+train['FLAG_DOCUMENT_5']+train['FLAG_DOCUMENT_6']+train['FLAG_DOCUMENT_7']+train['FLAG_DOCUMENT_8']+train['FLAG_DOCUMENT_9']+train['FLAG_DOCUMENT_10']+train['FLAG_DOCUMENT_11']+train['FLAG_DOCUMENT_12']+train['FLAG_DOCUMENT_13']+train['FLAG_DOCUMENT_14']+train['FLAG_DOCUMENT_15']+train['FLAG_DOCUMENT_16']+train['FLAG_DOCUMENT_17']+train['FLAG_DOCUMENT_18']+train['FLAG_DOCUMENT_19']+train['FLAG_DOCUMENT_20']+train['FLAG_DOCUMENT_21']
train.DOCS.unique()
for i in train.DOCS.unique():
    D1 = train.DOCS==i
    D3 = train[D1]
    print(i,D3.TARGET.value_counts(normalize=True),train.DOCS.mean())
train.NAME_CONTRACT_TYPE.fillna(0,inplace=True)
train.NAME_CONTRACT_TYPE.unique()
train.replace({'Cash loans', 'Revolving loans'},{1,2},inplace=True)
for i in train.NAME_CONTRACT_TYPE.unique():
    D1 = train.NAME_CONTRACT_TYPE==i
    D3 = train[D1]
    print(i,D3.TARGET.value_counts(normalize=True),D3.TARGET.value_counts())
train.NAME_INCOME_TYPE.unique()
train.replace({'Working', 'State servant', 'Commercial associate', 'Pensioner',
       'Unemployed', 'Student', 'Businessman', 'Maternity leave'},{1,8,6,4,7,2,3,4,-1},inplace=True)
for i in train.NAME_INCOME_TYPE.unique():
    V1 = train.NAME_INCOME_TYPE==i
    V2 = train[V1]
    print(i,V2.TARGET.value_counts(),i,V2.TARGET.value_counts(normalize=True))
train.NAME_EDUCATION_TYPE.unique()
train.replace({'Secondary / secondary special', 'Higher education',
       'Incomplete higher', 'Lower secondary', 'Academic degree'},{1,5,4,3,2},inplace=True)
for i in train.NAME_EDUCATION_TYPE.unique():
    edu1 = train.NAME_EDUCATION_TYPE==i
    edu = train[edu1]
    print(i,edu.TARGET.value_counts(),i,edu.TARGET.value_counts(normalize=True))
    
#clean the data, dont forget the soap 

train.replace(to_replace={'M','F','XNA'},value={1,2,3},inplace=True)
train.DEF_30_CNT_SOCIAL_CIRCLE.fillna(value=train['DEF_30_CNT_SOCIAL_CIRCLE'].mean(),inplace=True)
train.DEF_60_CNT_SOCIAL_CIRCLE.fillna(value=train['DEF_60_CNT_SOCIAL_CIRCLE'].mean(),inplace=True)
train.AMT_REQ_CREDIT_BUREAU_DAY.fillna(train['AMT_REQ_CREDIT_BUREAU_DAY'].mean(),inplace=True)
train.replace(to_replace={'Y','N'},value={0,1},inplace=True)
train.AMT_INCOME_TOTAL.fillna(value=168797.9192969845,inplace=True)
train.AMT_INCOME_TOTAL.fillna(value=168797.9192969845,inplace=True)
train.fillna(0,inplace=True)
train.dtypes
'FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','DAYS_BIRTH','DAYS_LAST_PHONE_CHANGE'
#SET INPUTS
#make 
target = train.TARGET.values

features = train[['DAYS_EMPLOYED','REGION_POPULATION_RELATIVE','DAYS_BIRTH','CODE_GENDER','AMT_REQ_CREDIT_BUREAU_DAY','NAME_EDUCATION_TYPE','NAME_CONTRACT_TYPE']].values

#clean the data, dont forget the soap 
from sklearn import model_selection
import sklearn 
from sklearn import linear_model
from sklearn import preprocessing
target = train.TARGET.values
features = train[['CODE_GENDER','AMT_REQ_CREDIT_BUREAU_DAY','NAME_EDUCATION_TYPE','NAME_CONTRACT_TYPE']].values
x_test,x_train,y_test,y_train = model_selection.train_test_split(features,target,test_size=.6,random_state=11)
classifier = linear_model.LogisticRegression()
classifier_ = classifier.fit(x_train,y_train)
cross_val_score = model_selection.cross_val_score
print(cross_val_score(classifier,x_test,y_test))
#SET INPUTS
#make inputs and targets np arrays with the .values function
target = train.TARGET.values
train.FLAG_EMP_PHONE
unscaled_inputs = train[['DAYS_BIRTH','CNT_FAM_MEMBERS','CODE_GENDER','AMT_REQ_CREDIT_BUREAU_DAY','NAME_EDUCATION_TYPE','NAME_CONTRACT_TYPE','FLAG_EMP_PHONE','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','DAYS_BIRTH','DAYS_LAST_PHONE_CHANGE']].values
unscaled_targets = train['TARGET'].values

#balancing data half defaulted other half of samples did not default
one_targets = int(np.sum(unscaled_targets))
to_remove = []
zero_counter = 0
for i in range(unscaled_targets.shape[0]):
    if unscaled_targets[i]==0:
        zero_counter += 1
        if zero_counter > one_targets:
            to_remove.append(i)
unscaled_inputs_equal_priors = np.delete(unscaled_inputs,to_remove,axis=0)
targets_new = np.delete(unscaled_targets,to_remove,axis=0)
###standarize and shuffle data
from sklearn import preprocessing
inputs = preprocessing.scale(unscaled_inputs_equal_priors)
shuffled = np.arange(inputs.shape[0])
np.random.shuffle(shuffled)
shuffled_inputs = inputs[shuffled]
shuffled_targets = targets_new[shuffled]

sample_count = shuffled_inputs.shape[0]
train_samples_count = int(0.8*sample_count)
validation_samples_count = int(0.1*sample_count)
test_sample_count = sample_count - train_samples_count - validation_samples_count
#
train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]
#
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]
#
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]
test_inputs.shape
np.savez('loan_data_train',inputs=train_inputs,targets=train_targets)
np.savez('loan_data_validation',inputs=validation_inputs,targets=validation_targets)
np.savez('loan_data_test',inputs=test_inputs,targets=test_targets)

class loan_data_reader():
    
    def __init__(self,dataset,batch_size=None):
        npz = np.load('loan_data_{0}.npz'.format(dataset))
        self.inputs, self.targets = npz['inputs'].astype(np.float),npz['targets'].astype(np.int)
        
        if batch_size is None:
            self.batch_size = self.inputs.shape[0]
        else:
            self.batch_size = batch_size
        self.curr_batch = 0
        self.batch_count = self.inputs.shape[0]//self.batch_size
    
    def __next__(self):
        if self.curr_batch >= self.batch_count:
            self.curr_batch = 0
            raise StopIteration()
            
        batch_slice = slice(self.curr_batch * self.batch_size,(self.curr_batch + 1)* self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self.curr_batch += 1
        
        classes_num = 2
        targets_one_hot = np.zeros((targets_batch.shape[0],classes_num))
        targets_one_hot[range(targets_batch.shape[0]),targets_batch] = 1
        
        return inputs_batch,targets_one_hot
    def __iter__(self):
        return self


import tensorflow as tf


input_size = 15

output_size = 2

hidden_layer_size = 25


tf.reset_default_graph()

#placeholders
inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.int32, [None, output_size])

weights_1 = tf.get_variable("weights_1", [input_size, hidden_layer_size])
biases_1 = tf.get_variable("biases_1", [hidden_layer_size])
outputs_1 = tf.nn.softmax(tf.matmul(inputs, weights_1) + biases_1)

weights_2 = tf.get_variable("weights_2", [hidden_layer_size, hidden_layer_size])
biases_2 = tf.get_variable("biases_2", [hidden_layer_size])
outputs_2 = tf.nn.sigmoid(tf.matmul(outputs_1, weights_2) + biases_2)

weights_3 = tf.get_variable("weights_3", [hidden_layer_size, output_size])
biases_3 = tf.get_variable("biases_3", [output_size])

outputs = tf.matmul(outputs_2, weights_3) + biases_3

# softmax cross entropy loss with logits
loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets)
mean_loss = tf.reduce_mean(loss)


out_equals_target = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))


optimize = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(mean_loss)


sess = tf.InteractiveSession()


initializer = tf.global_variables_initializer()
sess.run(initializer)


batch_size = 100

max_epochs = 500
prev_validation_loss = 9999999.

train_data = loan_data_reader('train', batch_size)
validation_data = loan_data_reader('validation')

for epoch_counter in range(max_epochs):
    
    curr_epoch_loss = 0.
    
    for input_batch, target_batch in train_data:
        _, batch_loss = sess.run([optimize, mean_loss], 
            feed_dict={inputs: input_batch, targets: target_batch})
        

        curr_epoch_loss += batch_loss
    

    curr_epoch_loss /= train_data.batch_count
    

    validation_loss = 0.
    validation_accuracy = 0.
    

    for input_batch, target_batch in validation_data:
        validation_loss, validation_accuracy = sess.run([mean_loss, accuracy],
            feed_dict={inputs: input_batch, targets: target_batch})
    

    print('Epoch '+str(epoch_counter+1)+
          '. Training loss: '+'{0:.3f}'.format(curr_epoch_loss)+
          '. Validation loss: '+'{0:.3f}'.format(validation_loss)+
          '. Validation accuracy: '+'{0:.2f}'.format(validation_accuracy * 100.)+'%')
    
 
    if validation_loss > prev_validation_loss:
        break
        
    prev_validation_loss = validation_loss
    
print('End of training.')




