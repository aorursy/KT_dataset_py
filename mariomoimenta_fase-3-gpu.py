import os
import tensorflow as tf
%matplotlib inline
import tensorflow as tf
from tensorflow.keras.layers import * # para facilitar a escrita
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time

print(tf.__version__)
tfrecordsDir = '/kaggle/input/house-numbers'

trainfile = tfrecordsDir + '/images_train' + '.tfrecords'
train_raw_image_dataset = tf.data.TFRecordDataset(trainfile)

validfile = tfrecordsDir + '/images_validation' + '.tfrecords'
valid_raw_image_dataset = tf.data.TFRecordDataset(validfile)

testfile = tfrecordsDir + '/images_test' + '.tfrecords'
test_raw_image_dataset = tf.data.TFRecordDataset(testfile)
# Create a dictionary describing the features.
image_feature_description = {'label':  tf.io.FixedLenFeature([6], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  aux = tf.io.parse_single_example(example_proto, image_feature_description)
  imageraw = aux['image_raw']
  imagemat = tf.io.decode_jpeg(imageraw, channels=3)
  imagemat = tf.cast(imagemat,dtype=tf.float32)
  imagemat = 1/255*imagemat
  label =  aux['label']
  
  #return tf.image.decode_jpeg(imageraw) #,label
  return (imagemat,label)

trainDataset = train_raw_image_dataset.map(_parse_image_function)
validDataset = valid_raw_image_dataset.map(_parse_image_function)
testDataset = test_raw_image_dataset.map(_parse_image_function)
iterator = iter(trainDataset.batch(2)) #trainDataset.batch(2)

image, label = next(iterator)
plt.imshow(image[0], interpolation='nearest')
plt.show()
plt.imshow(image[1], interpolation='nearest')
plt.show()
print(label)
class Encoder(tf.keras.layers.Layer):

  def __init__(self):
    #aqui é só para inicializar e não se aplica nenhuns inputs
    super(Encoder, self).__init__()

    self.layer_1_conv = Conv2D( filters=64, kernel_size=(5,5), strides=(1,1), padding='same')
    self.layer_1_act = Activation( LeakyReLU(0.1) )
    self.layer_1_pool = MaxPooling2D( pool_size=(2,2), strides=(2,2), padding='same')
    self.layer_1_norm = BatchNormalization()

    self.layer_2_conv = Conv2D( filters=128, kernel_size=(4,4), strides=(1,1), padding='same')
    self.layer_2_act = Activation( LeakyReLU(0.1) )
    self.layer_2_pool = MaxPooling2D( pool_size=(2,2), strides=(2,2), padding='same')
    self.layer_2_norm = BatchNormalization()
    
    self.layer_3_conv = Conv2D( filters=256, kernel_size=(3,3), strides=(1,1), padding='same')
    self.layer_3_act = Activation( LeakyReLU(0.1) )
    self.layer_3_pool = MaxPooling2D( pool_size=(2,2), strides=(2,2), padding='same')
    self.layer_3_norm = BatchNormalization()
  
  #no call é que recebe os inputs para o que foi inicializado possa ser utilizado
  def call(self, inputs):

    x1 = inputs
  
    x1 = self.layer_1_conv(x1)
    x1 = self.layer_1_act(x1)
    x1 = self.layer_1_pool(x1)
    x1 = self.layer_1_norm(x1)

    x2 = self.layer_2_conv(x1)
    x2 = self.layer_2_act(x2)
    x2 = self.layer_2_pool(x2)
    x2 = self.layer_2_norm(x2)
    
    x3 = self.layer_3_conv(x2)
    x3 = self.layer_3_act(x3)
    x3 = self.layer_3_pool(x3)
    x3 = self.layer_3_norm(x3)

    outputs = x3

    return outputs
#inicialização da classe Encoder
encoder = Encoder()

batch_size=2
iterator = iter(trainDataset.batch(batch_size))
sample_input = next(iterator)[0] #so estou a ir buscar as imagens
sample_output = encoder(sample_input)

print ('Encoder input shape: (batch size, altura, largura, profundidade) {}'.format(sample_input.shape))
print ('Encoder output shape: (batch size, altura, largura, profundidade) {}'.format(sample_output.shape))
def split_tensor(t,n):
  # t: tensor (resultado da convulsao 2D)
  # n: larguras das fatias (colunas), tem de ser um divisor da largura
  # t.shape = (batch size, altura, largura, profundidade)
  
  # se quisermos uma solução mais complicada pode-se fazer o split numa outra dimensão
  return tf.transpose( tf.split(t, n, axis=2, num=None, name='split'), perm=[1,0,2,3,4] )
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.W3 = tf.keras.layers.Dense(units)
    #nota o número de filtros 12 e o kernel size de 2 são parâmetros
    # que são definidos por nós
    self.conv = tf.keras.layers.Conv1D(64,2,padding="same")

    #devemos colocar no bias o tamanho do batch?????????????????
    self.b = tf.Variable(tf.zeros(units),trainable=True, name="bias")

    self.V = tf.keras.layers.Dense(1)


  def call(self, dec_hidden_state, enc_output_reshape_data, attention_weights_b):

 
    dec_hidden_state_with_time_axis = tf.expand_dims(dec_hidden_state, 1)

     # cálculo da matriz F que é uma matriz de convolução, de acordo com o paper
    # esta matriz vai depender do formato dos attention weights
    conv_attent = self.conv(attention_weights_b)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(dec_hidden_state_with_time_axis) + self.W2(enc_output_reshape_data) + self.W3(conv_attent) + self.b)
                  )

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * enc_output_reshape_data
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
class Decoder(tf.keras.Model):
  def __init__(self,number_house_possible,attention_units,dec_units,batch_size,number_split_images):
    super(Decoder, self).__init__()
    # dec_units: Positive integer, dimensionality of the output space.

    self.dec_units = dec_units
    self.batch_sz = batch_size
    self.attention_units=attention_units
    self.number_split_images=number_split_images
    self.number_house_possible=number_house_possible
    #o número de neurónios da unidade GRU é igual ao decoder units
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    
    #nota: pode-se colocar outra gru, tipo fazer um stack, uma em cima de outra GRU
    
    # temos aqui 11 porque temos 11 possíveis números que vão de 0 até 10
    # portanto o resultado depois é um vector com 11 colunas
    self.fc = tf.keras.layers.Dense(number_house_possible)

    # used for attention
    self.attention = BahdanauAttention(self.attention_units)

  def call(self, dec_prev_output, decoder_hidden, enc_output_r, attention_weights_prev):
    
    context_vector, attention_weights = self.attention(decoder_hidden, enc_output_r, attention_weights_prev)

    dec_prev_output = tf.expand_dims(dec_prev_output, -2)
    x = tf.concat([tf.expand_dims(context_vector, 1), dec_prev_output], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    output = tf.reshape(output, (-1, output.shape[2]))

    dec_output = self.fc(output)

    return dec_output, state, attention_weights

  def initialize_dec_hidden_state(self):
    return tf.zeros((self.batch_sz, self.dec_units))
  
  def initialize_attetion_weights(self):
    return tf.zeros((self.batch_sz, self.number_split_images,1))
  
  def initialize_dec_output(self):
    return tf.zeros((self.batch_sz,self.number_house_possible))
def mask_acc(real,pred,k):

  for i in range(BATCH_SIZE):
    if real[i] == 10:
      continue
    else:
      if k == 1:
        _ = train_acc.update_state(real,pred)
      elif k == 2:
        _ = test_acc.update_state(real,pred)
      else: #k==0
        _ = valid_acc.update_state(real,pred)

  return real
@tf.function
def train_step(inp, targ):
  loss = 0
  
  with tf.GradientTape() as tape:
    
    # o encoder vai apenas receber a imagem e vai apenas retornar o enc_output
    enc_output = encoder(inp)

    result = split_tensor(enc_output,NUM_FATIAS)

    lines = result.shape[2]
    columns = result.shape[3]
    filters = result.shape[4]

    #agora vamos fazer o reshape do encoder output
    enc_out_reshape = tf.reshape(result,[BATCH_SIZE, NUM_FATIAS, lines*columns*filters])

    #inicialização decoder hidden state
    dec_hidden_state = decoder.initialize_dec_hidden_state()

    # devemos inicializar também os attention_weights iniciais
    attention_weights_prev = decoder.initialize_attetion_weights()

    # devemos inicializar também o output previous do decoder
    decoder_output = decoder.initialize_dec_output()

    # primeiro timestep
    decoder_output, dec_hidden_state, attention_weights_p = decoder(decoder_output,
                                      dec_hidden_state, enc_out_reshape, attention_weights_prev)

    expected_output = decoder_output#tf.one_hot(targ[:, 0],11)
    
    loss += loss_object(tf.one_hot(targ[:, 0],11), decoder_output)    
    _ = mask_acc(targ[:,0],tf.argmax(decoder_output,axis=-1),1)

    for t in range(1, targ.shape[1]):
      
      # passing enc_output to the decoder
      decoder_output, dec_hidden_state, attention_weights_p = decoder(expected_output, dec_hidden_state, 
                                                                   enc_out_reshape,attention_weights_p)
      #calcula a loss function para todas as fotos de um batch, por exemplo se tivermos 
      # 32 imagens, vamos o targ[:, 3] vai ver a posição 3 de todas as imagens do batch
      loss += loss_object(tf.one_hot(targ[:, t],11), decoder_output)
      _ = mask_acc(targ[:,t],tf.argmax(decoder_output,axis=-1),1)
      
      # using teacher forcing
      # passa de um vector linha com uma dimensão, para duas dimensões
      expected_output = decoder_output#tf.one_hot(targ[:, t],11)


  batch_loss = (loss / int(targ.shape[1]))

  # é aqui que se vai buscar os pesos para depois serem optimizados
  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss
#função que determina o loss e accuracy validation set
def evaluate(inp, targ, k):
  loss = 0
 
  # o encoder vai apenas receber a imagem e vai apenas retornar o enc_output
  enc_output = encoder(inp)
  
  result = split_tensor(enc_output,NUM_FATIAS)
    
  lines=result.shape[2]
  columns=result.shape[3]
  filters=result.shape[4]

  #agora vamos fazer o reshape do encoder output
  enc_out_reshape = tf.reshape(result,[BATCH_SIZE, NUM_FATIAS, lines*columns*filters])

  #inicialização decoder hidden state
  dec_hidden_state = decoder.initialize_dec_hidden_state()

  # devemos inicializar também os attention_weights iniciais
  attention_weights_prev = decoder.initialize_attetion_weights()

  # devemos inicializar também o output previous do decoder
  decoder_output = decoder.initialize_dec_output()

  #primeiro timestep
  decoder_output, dec_hidden_state, attention_weights_p = decoder(decoder_output,
                                      dec_hidden_state, enc_out_reshape, attention_weights_prev)
  
  loss += loss_object(tf.one_hot(targ[:, 0],11), decoder_output)
  _ = mask_acc(targ[:,0],tf.argmax(decoder_output,axis=-1),k)

  for t in range(1, targ.shape[1]):
      
    # passing enc_output to the decoder
    decoder_output, dec_hidden_state, attention_weights_p = decoder(decoder_output, dec_hidden_state, 
                                                                   enc_out_reshape,attention_weights_p)
    #calcula a loss function para todas as fotos de um batch, por exemplo se tivermos 
    # 32 imagens, vamos o targ[:, 3] vai ver a posição 3 de todas as imagens do batch
    
    loss += loss_object(tf.one_hot(targ[:, t],11), decoder_output)
    _ = mask_acc(targ[:,t],tf.argmax(decoder_output,axis=-1),k)
    
  batch_loss = (loss / int(targ.shape[1]))

  return batch_loss
print(f'Possíveis valores para a largura das fatias: {[x for x in range(1, sample_output.shape[2]+1) if sample_output.shape[2] % x == 0]}')
EPOCHS = 20
BATCH_SIZE = 32
SIZE_VECT_DECODER_OUT = 11 #é sempre 11
ATT_UNITS = 128
DEC_UNITS = 128
NUM_FATIAS = 8

optimizer = tf.keras.optimizers.Adamax()
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

encoder = Encoder()
decoder = Decoder(SIZE_VECT_DECODER_OUT, ATT_UNITS, DEC_UNITS, BATCH_SIZE, NUM_FATIAS)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
train_acc = tf.keras.metrics.Accuracy()
valid_acc = tf.keras.metrics.Accuracy()
best_acc = float('-Inf')
best_epoch = -1

train_acc_list = []
train_loss_list = []
valid_acc_list = []
valid_loss_list = []
epoch_list = []

for epoch in range(EPOCHS):

  #inicializar os contadores
  start = time.time()
  train_acc.reset_states()
  valid_acc.reset_states()
  total_loss_train = 0
  total_loss_valid = 0
  count_batch_train = 0
  count_batch_valid = 0

  #train loop
  for (batch, (inp, targ)) in enumerate(trainDataset.shuffle(130000).batch(BATCH_SIZE,drop_remainder=True)):
    batch_loss = train_step(inp, targ)
    total_loss_train += batch_loss
    count_batch_train += 1
    
    if batch % 500 == 0:
      print('Epoch {}; Batch {}; Loss_train {:.4f}; Acc_train {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy(),train_acc.result().numpy()))
    
  print('Time taken to train 1 epoch {:.4f} sec.\nCalculating validation accuracy now.'.format(time.time() - start))

  #validation loop
  for (batch, (inp, targ)) in enumerate(validDataset.batch(BATCH_SIZE,drop_remainder=True)):
    batch_loss_valid = evaluate(inp, targ, 0)
    total_loss_valid += batch_loss_valid
    count_batch_valid += 1
    
  epoch_list.append(epoch+1)
  train_loss_list.append(total_loss_train.numpy() / count_batch_train)
  valid_loss_list.append(total_loss_valid.numpy() / count_batch_valid)
  train_acc_list.append(train_acc.result().numpy())
  valid_acc_list.append(valid_acc.result().numpy())
  print('>>>>>> Epoch {} Loss_train {:.4f} Acc_train {:.4f} Loss_valid {:.4f} Acc_valid {:.4f} <<<<<<'.format(epoch + 1,
                                                                                                (total_loss_train / count_batch_train), train_acc.result().numpy(),
                                                                                                (total_loss_valid / count_batch_valid), valid_acc.result().numpy()
                                                                                               )
       )
  
  # saving best checkpoint so far
  if best_acc < valid_acc.result().numpy():
      print("Accuracy improved from {:.4f} to {:.4f}".format(best_acc, valid_acc.result().numpy()))
      best_acc = valid_acc.result().numpy()
      best_epoch = epoch
  else:
      print("Accuracy has not improved from {:.4f}".format(best_acc))

  checkpoint.save(f"/kaggle/working/check_epoch_{str(epoch + 1)}/ckpt")
  print(f"Saved checkpoint of epoch {epoch+1} into /kaggle/working/check_epoch_{str(epoch + 1)}/")
 
  print('Time taken for 1 epoch {:.4f} sec\n'.format(time.time() - start))
import matplotlib.pyplot as plt
plt.plot(epoch_list,train_loss_list)
plt.plot(epoch_list,valid_loss_list)
plt.axvline(x=12, color='green')
plt.title('Loss evolution')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation', 'Best checkpoint'], loc='upper right')
plt.grid()
fig = plt.gcf()
fig.set_size_inches(8, 4.5)
fig.savefig('loss_evolution.png', dpi=200)
plt.show()
import matplotlib.pyplot as plt
plt.plot(epoch_list,train_acc_list)
plt.plot(epoch_list,valid_acc_list)
plt.axvline(x=12, color='green')
plt.title('Accuracy evolution')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train accuracy', 'Validation accuracy', 'Best checkpoint'], loc='lower right')
plt.grid()
fig = plt.gcf()
fig.set_size_inches(8, 4.5)
fig.savefig('accuracy_evolution.png', dpi=200)
plt.show()
load_checkpoint = True
load_checkpoint_dir = '/kaggle/working/check_epoch_12/ckpt-12'
if load_checkpoint:
    checkpoint.restore(load_checkpoint_dir)
#ATENÇÂO O MODELO TEM QUE ESTAR TREINADO ANTES DE INICAR ESTA CÉLULA DE CÓDIGO
#calculo da accuracy do teste
start = time.time()
test_acc = tf.keras.metrics.Accuracy()

total_loss_test = 0
count_batch_test = 0

### steps por epoch -> número de imagens que pretendemos processar, pode ser 32
for (batch, (inp, targ)) in enumerate(testDataset.batch(BATCH_SIZE,drop_remainder=True)):
  batch_loss = evaluate(inp, targ, 2)
  total_loss_test += batch_loss
  count_batch_test +=1

  if batch % 100 == 0:
    print( 'Batch {} Loss_test {} Acc_test {:.4f}'.format(batch, batch_loss.numpy(),test_acc.result().numpy()) )
       
print('Loss_teste {:.4f} Acc_test {:.4f}'.format((total_loss_test / count_batch_test), test_acc.result().numpy()))
print('Time taken {} sec\n'.format(time.time() - start))
import pickle
obj_to_save = [epoch_list, train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, total_loss_test.numpy() / count_batch_test,test_acc.result().numpy()]
# Saving the objects:
with open('train_info.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(obj_to_save, f)
iterator = iter(trainDataset.batch(BATCH_SIZE))
image, label = next(iterator)

In = image
Out = label
def predict(inp):

  # o encoder vai apenas receber a imagem e vai apenas retornar o enc_output
  enc_output = encoder(inp)
 
  result = split_tensor(enc_output,NUM_FATIAS)

  lines = result.shape[2]
  columns = result.shape[3]
  filters = result.shape[4]

  #agora vamos fazer o reshape do encoder output
  enc_out_reshape = tf.reshape(result,[BATCH_SIZE, NUM_FATIAS, lines*columns*filters])

  #inicialização decoder hidden state
  dec_hidden_state = decoder.initialize_dec_hidden_state()

  # devemos inicializar também os attention_weights iniciais
  attention_weights_prev = decoder.initialize_attetion_weights()

  # devemos inicializar também o output previous do decoder
  #decoder_prev_output=decoder.initialize_dec_output()
  decoder_output = decoder.initialize_dec_output()

  decoder_output, dec_hidden_state, attention_weights_p = decoder(decoder_output,
                                      dec_hidden_state, enc_out_reshape, attention_weights_prev)

  prediction_labels = tf.reshape(tf.argmax(decoder_output,axis=1),[BATCH_SIZE,1])

  for t in range(1, 6):
      
    # passing enc_output to the decoder
    decoder_output, dec_hidden_state, attention_weights_p = decoder(decoder_output, dec_hidden_state, 
                                                                   enc_out_reshape,attention_weights_p)
    
    prediction_labels = tf.concat([prediction_labels,tf.reshape(tf.argmax(decoder_output,axis=1),[BATCH_SIZE,1])],axis=1)

  prediction_labels_total = tf.reshape(prediction_labels,[BATCH_SIZE,6]) 

  return prediction_labels_total
predict_labels = predict(In)
predict_labels
import pandas as pd
RealL=pd.DataFrame(np.asarray(Out))
PredictL=pd.DataFrame(np.asarray(predict_labels))
print(" Predicted Labels          vs           Real Labels")
versus=pd.DataFrame(np.reshape(["vs"]*BATCH_SIZE,(BATCH_SIZE,1)))
PredictL_vs_RealL=pd.concat((PredictL,versus,RealL), axis=1)
PredictL_vs_RealL
accuracy_predict = np.sum(predict_labels.numpy()==Out.numpy())/(6*BATCH_SIZE)
accuracy_predict
