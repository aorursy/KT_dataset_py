#Imports for the entire Kernel
import pandas as pd
import numpy as np
import sklearn.preprocessing
import keras as ke

separator_char = "|"
sample_length = 64
batch_size = 64

df = pd.read_csv("../input/winemag-data_first150k.csv")
df = df["description"]

train_string = df.str.cat(sep=separator_char)

train_string = train_string.lower()



vocab_size = len(set(train_string))

print ("Data Sanity Check:")
print (train_string[:1000])
print("\n\n")
print ("Training Set Total Lenght:", len(train_string))
print ("Total Unique Characters:", vocab_size)
#Mapping Vocabulary characters to integers and also creating a reverse dict for later:
mapping_dict = {}
mapping_dict_rev = {}
for i, c in enumerate(set(train_string)): 
    mapping_dict[c] = i
    mapping_dict_rev[i] = c 

#Preprocessing the Data inside of a Generator
def training_batch_generator(trainstring, mapping_dict, batchsize, sample_length, vocab_size):
    enc = sklearn.preprocessing.OneHotEncoder(n_values=vocab_size)
    
    fitarray = np.array(list(trainstring[:batchsize*1000]), dtype=np.str)
    fitarray = np.vectorize(mapping_dict.__getitem__)(fitarray)
    fitarray = fitarray.reshape((-1, 1))
    enc.fit(fitarray)
    del fitarray
    
    trainlength = len(trainstring)-1
    
    while True:
        seed = np.random.randint(0, trainlength-sample_length-batchsize-1)
        
        batch_x = []
        batch_y = []
        
        for i in range(batchsize):
            sample_in =  np.array(list(trainstring[seed+i:seed+i+sample_length]), dtype=np.str) 
            sample_in = np.vectorize(mapping_dict.__getitem__)(sample_in)
            sample_in = sample_in.reshape((-1, 1))
            x = enc.transform(sample_in)
            
            sample_out = np.array(list(trainstring[seed+i+sample_length]), dtype=np.str) 
            sample_out = np.vectorize(mapping_dict.__getitem__)(sample_out)
            sample_out = sample_out.reshape((-1, 1))
            y= enc.transform(sample_out)
            
            batch_x.append(x.toarray())
            batch_y.append(y.toarray())
        
        batch_x = np.array(batch_x, dtype=np.bool)
        batch_x = batch_x.reshape((batchsize, sample_length, vocab_size))
        
        batch_y = np.array(batch_y, dtype=np.bool)
        batch_y = batch_y.reshape((batchsize, vocab_size))
        
        yield (batch_x, batch_y)
        
#Generator object for later use
generator_object = training_batch_generator(train_string, mapping_dict, batch_size, sample_length, vocab_size)
#Define the LSTM
print("Generating Model")
lstm_model = ke.models.Sequential()
lstm_model.add(ke.layers.LSTM(256, return_sequences=True, input_shape=(sample_length, vocab_size)))
lstm_model.add(ke.layers.Dropout(0.3))
lstm_model.add(ke.layers.LSTM(128, return_sequences=True, input_shape=(sample_length, vocab_size)))
lstm_model.add(ke.layers.Dropout(0.3))
lstm_model.add(ke.layers.LSTM(128,input_shape=(sample_length, vocab_size)))
lstm_model.add(ke.layers.Dropout(0.3))
lstm_model.add(ke.layers.Dense(vocab_size, activation='softmax'))
lstm_model.compile(loss='categorical_crossentropy', optimizer=ke.optimizers.Adam(lr=0.001))

lstm_model.summary()


class LSTMGeneration(ke.callbacks.Callback):
    def __init__(self, generator):
        #The generator should be the same as the one used for training in order to have consistent encoding
        self.generator = generator 
        
    
    def on_epoch_end(self, epoch, logs={}):
        #Settings for the Generator
        file = "lstm_generator.txt"
        generate_n = 1
        
        for n in range(generate_n):
            start = next(self.generator)         
            current_predict = start[0][0]
            
            predicted = 0
            fallback_string = ""
            predict_string = ""
            started = False
            finished = False
            
            while not finished:
                x_predict = np.reshape(current_predict, (1, sample_length, vocab_size))
                pred_raw = self.model.predict(x_predict)
                
                pred = np.zeros((1, vocab_size))
                pred[0, np.argmax(pred_raw)] = 1
                
                
                current_predict = np.vstack([current_predict, pred])
                current_predict = current_predict[1:]
                
                predict_char = mapping_dict_rev[np.argmax(pred_raw)]
                
                fallback_string += predict_char
                if started:
                    predict_string += predict_char
                    
                if predict_char == separator_char:
                    if not started:
                        started = True
                    else:
                        finished = True
                
                predicted += 1
                if (predicted >= 1000 and not started) or (predicted >= 3000):
                    break
                
            with open(file, "a+") as f:
                f.write("Epoch "+str(epoch)+" , Text "+str(n)+"\n")
                if len(predict_string) > 0:
                    f.write(predict_string+"\n\n")
                    print(predict_string+"\n\n")
                else:
                    f.write(fallback_string+"\n\n")
                    print(fallback_string+"\n\n")
#Change Steps per Epoch if you want to have denser or sparser monitoring of learning progress 
lstm_model.fit_generator(generator_object, steps_per_epoch=1000, epochs=1, callbacks=[LSTMGeneration(generator_object)], verbose=1)