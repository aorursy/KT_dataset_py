import tensorflow as tf, numpy as np



# hyperparameters here

divisor = 7



# convert a number into binary

def int2bits(i,fill=21): 

    return list(map(int,bin(i)[2:].zfill(fill)))



def bits2int(b):

    return sum(i*2**n for n,i in enumerate(reversed(b)))



# Data. 

I = np.random.randint(0,2**20,size=(250000,))

X = np.array(list(map(int2bits,I)))

Y = np.array([int2bits(2**i,divisor) for i in I % divisor])



# Test Data. 

It = np.random.randint(2**20,2**21,size=(10000,))

Xt = np.array(list(map(int2bits,It)))

Yt = np.array([int2bits(2**i,divisor) for i in It % divisor])
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (21))



x = Dense(1000, 'relu')(inputs)

outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,20,validation_data=(Xt,Yt))
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (21))

# x = Dense(1000, 'relu')(inputs)



# # Do a ResNet style skip connection

# layer1 = Concatenate()([x, inputs])

# x = Dense(1000, 'relu')(layer1)



# # Do a double skip connection

# layer2 = Concatenate()([x, layer1])

# x = Dense(1000, 'relu')(layer2)



# # Do a triple skip connection

# layer3 = Concatenate()([x, layer2])

# x = Dense(1000, 'relu')(layer3)



x = Dense(1000, 'relu')(inputs)

x = Dense(1000, 'relu')(x)

outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,20,validation_data=(Xt,Yt))
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (21))

# x = Dense(1000, 'relu')(inputs)



# # Do a ResNet style skip connection

# layer1 = Concatenate()([x, inputs])

# x = Dense(1000, 'relu')(layer1)



# # Do a double skip connection

# layer2 = Concatenate()([x, layer1])

# x = Dense(1000, 'relu')(layer2)



# # Do a triple skip connection

# layer3 = Concatenate()([x, layer2])

# x = Dense(1000, 'relu')(layer3)



x = Dense(1000, 'relu')(inputs)

x = Dense(1000, 'relu')(x)

x = Dense(1000, 'relu')(x)

outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,20,validation_data=(Xt,Yt))
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (21))

x = Dense(1000, 'relu')(inputs)



# Do a ResNet style skip connection

layer1 = Concatenate()([x, inputs])

x = Dense(1000, 'relu')(layer1)



# Do a double skip connection

layer2 = Concatenate()([x, layer1])

x = Dense(1000, 'relu')(layer2)



# Do a triple skip connection

layer3 = Concatenate()([x, layer2])

x = Dense(1000, 'relu')(layer3)



# x = Dense(1000, 'relu')(inputs)

# x = Dense(1000, 'relu')(x)

# x = Dense(1000, 'relu')(x)

outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,20,validation_data=(Xt,Yt))
import tensorflow as tf, numpy as np



# hyperparameters here

divisor = 7



# convert a number into binary

def int2bits(i,fill=24): 

    return list(map(int,bin(i)[2:].zfill(fill)))



def bits2int(b):

    return sum(i*2**n for n,i in enumerate(reversed(b)))



# Data. 

I = np.random.randint(0,2**20,size=(250000,))

X = np.array(list(map(int2bits,I*8+divisor)))

Y = np.array([int2bits(2**i,divisor) for i in I % divisor])



# Test Data. 

It = np.random.randint(2**20,2**21,size=(10000,))

Xt = np.array(list(map(int2bits,It*8+divisor)))

Yt = np.array([int2bits(2**i,divisor) for i in It % divisor])
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (24))



x = Dense(100, 'relu')(inputs)

outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,20,validation_data=(Xt,Yt))
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (21))

# x = Dense(1000, 'relu')(inputs)



# # Do a ResNet style skip connection

# layer1 = Concatenate()([x, inputs])

# x = Dense(1000, 'relu')(layer1)



# # Do a double skip connection

# layer2 = Concatenate()([x, layer1])

# x = Dense(1000, 'relu')(layer2)



# # Do a triple skip connection

# layer3 = Concatenate()([x, layer2])

# x = Dense(1000, 'relu')(layer3)



x = Dense(1000, 'relu')(inputs)

outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,20,validation_data=(Xt,Yt))
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (24))



x = Dense(1000, 'relu')(inputs)

outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,20,validation_data=(Xt,Yt))
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (24))



x = Dense(1000, 'relu')(inputs)

x = Dense(1000, 'relu')(x)

outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,20,validation_data=(Xt,Yt))
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (24))

x = Dense(1000, 'relu')(inputs)



# Do a ResNet style skip connection

layer1 = Concatenate()([x, inputs])

x = Dense(1000, 'relu')(layer1)



# Do a double skip connection

layer2 = Concatenate()([x, layer1])

x = Dense(1000, 'relu')(layer2)



# Do a triple skip connection

layer3 = Concatenate()([x, layer2])

x = Dense(1000, 'relu')(layer3)



outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,20,validation_data=(Xt,Yt))
import tensorflow as tf, numpy as np



# hyperparameters here

divisor = 7



# Data. 

I = np.random.randint(0,2**20,size=(250000,))

X = np.array([[i, divisor] for i in I])

Y = np.array([[1 if element%divisor == i else 0 for i in range(7)] for element in I])



# Test Data. 

It = np.random.randint(0,2**20,size=(250000,))

Xt = np.array([[i, divisor] for i in It])

Yt = np.array([[1 if element%divisor == i else 0 for i in range(7)] for element in It])
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model



### Change the model architecture here

########################################################

inputs = Input(shape = (2))



x = Dense(1000, 'relu')(inputs)

outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,20,validation_data=(Xt,Yt))
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model



### Change the model architecture here

########################################################

inputs = Input(shape = (2))



x = Dense(1000, 'relu')(inputs)

x = Dense(1000, 'relu')(inputs)

outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,20,validation_data=(Xt,Yt))
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (2))

x = Dense(1000, 'relu')(inputs)



# Do a ResNet style skip connection

layer1 = Concatenate()([x, inputs])

x = Dense(1000, 'relu')(layer1)



# Do a double skip connection

layer2 = Concatenate()([x, layer1])

x = Dense(1000, 'relu')(layer2)



# Do a triple skip connection

layer3 = Concatenate()([x, layer2])

x = Dense(1000, 'relu')(layer3)



outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,20,validation_data=(Xt,Yt))
import tensorflow as tf, numpy as np



# hyperparameters here

divisor = 7



# convert a number into binary

def int2bits(i,fill=21): 

    return list(map(int,bin(i)[2:].zfill(fill)))



def bits2int(b):

    return sum(i*2**n for n,i in enumerate(reversed(b)))



# Data. 

I = np.append(np.random.randint(0,2**20,size=(250000,)), np.random.randint(2**20, 2**21, size = (100,)))

X = np.array(list(map(int2bits,I)))

Y = np.array([int2bits(2**i,divisor) for i in I % divisor])



# Test Data. 

It = np.random.randint(2**20,2**21,size=(10000,))

Xt = np.array(list(map(int2bits,It)))

Yt = np.array([int2bits(2**i,divisor) for i in It % divisor])
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (21))



x = Dense(1000, 'relu')(inputs)

outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,100,validation_data=(Xt,Yt))
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (21))



x = Dense(1000, 'relu')(inputs)

x = Dense(1000, 'relu')(x)

outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,100,validation_data=(Xt,Yt))
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (21))

x = Dense(1000, 'relu')(inputs)



# Do a ResNet style skip connection

layer1 = Concatenate()([x, inputs])

x = Dense(1000, 'relu')(layer1)



# Do a double skip connection

layer2 = Concatenate()([x, layer1])

x = Dense(1000, 'relu')(layer2)



# Do a triple skip connection

layer3 = Concatenate()([x, layer2])

x = Dense(1000, 'relu')(layer3)



outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,100,validation_data=(Xt,Yt))
import tensorflow as tf, numpy as np



# hyperparameters here

divisor = 7



# convert a number into binary

def int2bits(i,fill=21): 

    return list(map(int,bin(i)[2:].zfill(fill)))



def bits2int(b):

    return sum(i*2**n for n,i in enumerate(reversed(b)))



# Data. 

I = np.append(np.random.randint(0,2**20,size=(250000,)), np.random.randint(2**20, 2**21, size = (1,)))

X = np.array(list(map(int2bits,I)))

Y = np.array([int2bits(2**i,divisor) for i in I % divisor])



# Test Data. 

It = np.random.randint(2**20,2**21,size=(10000,))

Xt = np.array(list(map(int2bits,It)))

Yt = np.array([int2bits(2**i,divisor) for i in It % divisor])
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (21))



x = Dense(1000, 'relu')(inputs)

outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,100,validation_data=(Xt,Yt))
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (21))



x = Dense(1000, 'relu')(inputs)

x = Dense(1000, 'relu')(x)

outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,100,validation_data=(Xt,Yt))
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (21))

x = Dense(1000, 'relu')(inputs)



# Do a ResNet style skip connection

layer1 = Concatenate()([x, inputs])

x = Dense(1000, 'relu')(layer1)



# Do a double skip connection

layer2 = Concatenate()([x, layer1])

x = Dense(1000, 'relu')(layer2)



# Do a triple skip connection

layer3 = Concatenate()([x, layer2])

x = Dense(1000, 'relu')(layer3)



outputs = Dense(divisor, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10_000,100,validation_data=(Xt,Yt))
import tensorflow as tf, numpy as np



# hyperparameters here

divisor = 7



# convert a number into binary

def int2bits(i,fill=21): 

    return list(map(int,bin(i)[2:].zfill(fill)))



def bits2int(b):

    return sum(i*2**n for n,i in enumerate(reversed(b)))



# Data. 

I = np.random.randint(0,2**20,size=(250000,))

X = np.array(list(map(int2bits, I))) 

Y = np.array([int2bits(2**i,divisor) for i in I % divisor])



# Test Data. 

It = np.random.randint(2**20,2**21,size=(10000,))

Xt = np.array(list(map(int2bits,It)))

Yt = np.array([int2bits(2**i,divisor) for i in It % divisor])
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate, Reshape, Conv1D, Flatten

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (21))



x = Reshape((-1, 1))(inputs)

print(x.shape)



x = Conv1D(filters = 1, kernel_size = 3, strides = 3)(x)

print(x.shape)

x = Flatten()(x)

x = Dense(1000, 'relu')(x)

outputs = Dense(7, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10000,100,validation_data=(Xt,Yt))
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate, Reshape, Conv1D, Flatten

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (21))



x = Reshape((-1, 1))(inputs)

print(x.shape)



x = Conv1D(filters = 1, kernel_size = 3, strides = 3)(x)

print(x.shape)

x = Flatten()(x)

x = Dense(1000, 'relu')(x)

x = Dense(1000, 'relu')(x)

outputs = Dense(7, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10000,100,validation_data=(Xt,Yt))
import tensorflow as tf, numpy as np



# hyperparameters here

divisor = 7



# convert a number into binary

def int2bits(i,fill=24): 

    return list(map(int,bin(i)[2:].zfill(fill)))



def bits2int(b):

    return sum(i*2**n for n,i in enumerate(reversed(b)))



# Data. 

I = np.random.randint(0,2**21,size=(250000,))

X = np.array(list(map(int2bits, I))) 

Y = np.array([int2bits(2**i,divisor) for i in I % divisor])



# Test Data. 

It = np.random.randint(2**21,2**22,size=(10000,))

Xt = np.array(list(map(int2bits,It)))

Yt = np.array([int2bits(2**i,divisor) for i in It % divisor])
# Model.

from tensorflow.keras.layers import Dense, Input, Concatenate, Reshape, Conv1D, Flatten

from tensorflow.keras import Model





### Change the model architecture here

########################################################

inputs = Input(shape = (24))



x = Reshape((-1, 1))(inputs)

print(x.shape)



x = Conv1D(filters = 1, kernel_size = 6, strides = 6)(x)

print(x.shape)

x = Flatten()(x)

x = Dense(1000, 'relu')(x)

x = Dense(1000, 'relu')(x)

outputs = Dense(7, 'softmax')(x)



model = Model(inputs=inputs, outputs=outputs)



########################################################



model.compile('adam','categorical_crossentropy',['accuracy'])



# Train (report the final score at the 20th epoch)

model.fit(X,Y,10000,100,validation_data=(Xt,Yt))