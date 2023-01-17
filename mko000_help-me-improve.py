# Definitions etc.
# Lots of functions to convert drawing to a static binary image
# Generator class that loads chunks from pandas dataframes

import os
import numpy as np
import pandas as pd
import random

from PIL import Image, ImageDraw
from keras.utils import np_utils
from keras.utils import Sequence

#%%

def ClassFromFileName(filename):
    dot = filename.find('.')
    return filename[:dot].replace(' ', '_')

#%%

def MakeImage(drawing, line_width, orig_size=(256, 256), target_size=(128, 128)):
    img = Image.new('1', orig_size)
    draw = ImageDraw.Draw(img)
    for stroke in drawing:
        stroke_seq = list(zip(stroke[0], stroke[1]))
        draw.line(stroke_seq, fill=1, width=line_width)
    img_small = img.resize(target_size)
    return img_small

#def SaveImage(drawing, file_path, line_width, orig_size=(256, 256), target_size=(128, 128)):
    #img = MakeImage(drawing, line_width, orig_size, target_size)
    #img.save(file_path)
    
def DrawingToImage(drawing, line_width, orig_size=(256, 256), target_size=(128, 128)):
    img = MakeImage(drawing, line_width, orig_size, target_size)
    return np.reshape(np.array(list(img.getdata()), dtype=np.uint8), target_size)

#%%
    
def MyRead(folder, file, count, header_index=0, start_index=0):
    skip = list(range(1, start_index))
    if len(skip) > 0:
        return pd.read_csv(os.path.join(folder, file), nrows=count, header=header_index, skiprows=skip)
    else:
        return pd.read_csv(os.path.join(folder, file), nrows=count)
            

def DrawingsToImages(drawings, line_width, target_size=(128, 128)):
    return [DrawingToImage(drawing, line_width, target_size=target_size) for drawing in drawings]

def ExtractData(dataframes, column_name, line_width, target_size=(128, 128)):
    column_data = [df[column_name].map(eval).tolist() for df in dataframes]
    return [DrawingsToImages(drawings, line_width, target_size=target_size) for drawings in column_data]

def MakeAnswers(classes, count):
    return [[c for i in range(count)] for c in classes]



#%%
    
def LoadData(folder, dim_2d, channels, samples_per_class, column='drawing', line_width=3, shuffle=True, header=0, start_index=0):
    files = os.listdir(folder)
    dataframes = [MyRead(folder, file, samples_per_class, header, start_index) for file in files]
    image_data = ExtractData(dataframes, column, line_width, dim_2d)
    
    classes = [ClassFromFileName(file) for file in files]
    classes_num = [i for i in range(len(files))]
    answers = MakeAnswers(classes_num, samples_per_class)
    
    total_samples = len(classes) * samples_per_class
    image_data = np.reshape(image_data, (total_samples, 1, dim_2d[0], dim_2d[1]))
    answers = np.reshape(answers, (total_samples))
    answers = np_utils.to_categorical(answers, len(classes), dtype='uint8')
    
    if shuffle:
        images_answers = list(zip(image_data, answers))
        random.shuffle(images_answers)
        image_data, answers = zip(*images_answers)
    
    image_data = np.array(image_data, dtype=np.uint8)
    answers = np.array(answers)
    class_mapping = dict(zip(classes_num, classes))
    
    return (image_data, answers, class_mapping)

class DoodleGenerator(Sequence):
    
    def __init__(self, files, class_count, batchsize, chunksize, batches_per_epoch, line_width=3, target_size=(64, 64), skip=0):
        self.answers = self.MakeAnswers(class_count)
        self.batch = batchsize
        self.chunksize = chunksize
        #self.shuffle = shuffle
        self.batches_per_epoch = batches_per_epoch
        #self.max_chunks = max_chunks
        self.line_width=line_width
        self.target_size=target_size
        
        self.dfiterators = self.LoadDataframes(files)
        self.chunks = self.LoadChunks(self.dfiterators, chunksize)
        self.chunkpositions = [0 for d in files]
        self.chunkcounters = [0 for d in files]
    
    def MakeAnswers(self, class_count):
        indexes = [i for i in range(class_count)]
        return np_utils.to_categorical(indexes, dtype='uint8')
    
    def LoadDataframes(self, files, skip=0):
        return [self.LoadDataframe(f) for f in files]
    
    def LoadDataframe(self, file, skip=0):
        return pd.read_csv(file, iterator=True, skiprows=lambda x: x > 0 and x < skip)
    
    def LoadChunks(self, dataframes, chunksize):
        return [self.LoadChunk(d, chunksize) for d in dataframes]
    
    def LoadChunk(self, dataframe, chunksize):
        return dataframe.get_chunk(chunksize)        
    
    def __len__(self):
        return self.batches_per_epoch
    
    def DoGetOne(self, chunk, chunkposition):
        dfrow = self.chunks[chunk].iloc[chunkposition]
        drawing = eval(dfrow['drawing'])
        return DrawingToImage(drawing, self.line_width, target_size=self.target_size)
    
    def GetOne(self):
        chosen = random.randrange(0, len(self.answers))
        x = self.DoGetOne(chosen, self.chunkpositions[chosen])
        self.chunkpositions[chosen] = self.chunkpositions[chosen] + 1
        if self.chunkpositions[chosen] == self.chunksize:
            self.chunks[chosen] = self.LoadChunk(self.dfiterators[chosen], self.chunksize)
            self.chunkcounters[chosen] = self.chunkcounters[chosen] + 1
            self.chunkpositions[chosen] = 0
        y = self.answers[chosen]
        return (x, y)
    
    def UnzipReshape(self, tuples):
        xs = np.reshape(np.array([t[0] for t in tuples]), (self.batch, 1, self.target_size[0], self.target_size[1]))
        ys = np.array([t[1] for t in tuples])
        return xs, ys
    
    def __getitem__(self, idx):
        xytuples = [self.GetOne() for i in range(self.batch)]
        return self.UnzipReshape(xytuples)
    
    def GetSamplesServed(self, i):
        return self.chunkcounters[i] * self.chunksize + self.chunkpositions[i]
    
    def GetStatus(self):
        return [self.GetSamplesServed(i) for i in range(len(self.chunkcounters))]
#class end
print("defs done")
# Load validation data
#

folder = '../input/train_simplified'
files = os.listdir(folder)
paths = [os.path.join(folder, file) for file in files]

dim_1 = 64
dim_2 = 64
items_per_class = 70
val_start_index = 0

# paint images with this line width, also used for generator
line_width = 4


val_data, val_answers, class_map = LoadData(folder, (dim_1, dim_2), 1, items_per_class, 'drawing', line_width, False, start_index=val_start_index)
class_count = len(class_map)

print('Data shape: {}'.format(np.shape(val_data)))
print('Answers shape: {}'.format(np.shape(val_answers)))
print('Classes: {}'.format(class_count))

# Network architecture
#

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Activation, LeakyReLU
from keras.metrics import top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

from keras import backend as K
K.set_image_dim_ordering('th')

net = Sequential()
net.add(Convolution2D(64, (3, 3), padding='same', input_shape=(1, dim_1, dim_2)))
net.add(LeakyReLU(alpha=0.1))

net.add(Convolution2D(128, (3, 3), padding='same'))
net.add(LeakyReLU(alpha=0.2))

net.add(Convolution2D(256, (3, 3), padding='same'))
net.add(LeakyReLU(alpha=0.2))

net.add(MaxPooling2D((2, 2)))

net.add(Convolution2D(256, (3, 3), padding='same'))
net.add(LeakyReLU(alpha=0.3))

net.add(MaxPooling2D((2, 2)))

net.add(Convolution2D(384, (3, 3)))
net.add(LeakyReLU(alpha=0.3))

net.add(Convolution2D(512, (3, 3)))
net.add(LeakyReLU(alpha=0.2))

net.add(MaxPooling2D((2, 2)))

net.add(Convolution2D(384, (1, 1)))
net.add(LeakyReLU(alpha=0.2))

net.add(MaxPooling2D((2, 2)))

net.add(Flatten())

net.add(Dense(600))
net.add(LeakyReLU(alpha=0.1))

net.add(Dropout(0.25))

net.add(Dense(class_count, activation='softmax'))

net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top_3_accuracy])

try:
    os.stat('models/')
except:
    os.mkdir('models/')

net_json = net.to_json()
with open("./models/model10.json", "w") as json_file:
    json_file.write(net_json)
    
print(os.listdir('models'))
# Print net summary
#
print(net.metrics_names)
print(net.summary())
# Prepare generator and fit model
#
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

train_model_file = './models/model10.traing.{epoch:02d}.hdf5'
val_model_file = './models/model10.test.{epoch:02d}.hdf5'
final_model_file = './models/model10.final.hdf5'
log_file = './models/model10-log.csv'

train_checkpointer = ModelCheckpoint(filepath=train_model_file, monitor='top_3_accuracy', save_best_only=True, save_weights_only=True, mode='max')
val_checkpointer = ModelCheckpoint(filepath=val_model_file, monitor='val_top_3_accuracy', save_best_only=True, save_weights_only=True, mode='max')
logger = CSVLogger(log_file)
val_loss_stop = EarlyStopping(monitor='val_loss', patience=8, mode='min', baseline=4.0)

epoch_count = 30
batch_size = 128
batches_per_epoch = 900
num_classes = 340 # same as class_count somewhere above, but whatever

# files, class_count, batchsize, chunksize, batches_per_epoch
doodlegen = DoodleGenerator(paths, num_classes, batch_size, 128, batches_per_epoch, line_width=line_width, skip=items_per_class+1)
print('generator line width: {}'.format(doodlegen.line_width))

imgs_per_epoch = batch_size * batches_per_epoch
imgs_total = imgs_per_epoch * epoch_count
imgs_per_class = imgs_total / num_classes
print('Images per epoch: {}'.format(imgs_per_epoch))
print('Total images processed: {}'.format(imgs_total))
print('Expected images per class: {}'.format(imgs_per_class))

mycallbacks = [logger, train_checkpointer, val_checkpointer, val_loss_stop]
net.fit_generator(doodlegen, epochs=epoch_count, validation_data=(val_data, val_answers), callbacks=mycallbacks)
net.save(final_model_file)
# Print generator status
# I used this to verify that there is roughly the same number of samples per class when training

status = doodlegen.GetStatus()
print('status:\n', status)
print('max: ', np.amax(status))
print('avg: ', np.mean(status))
print('min: ', np.amin(status))

# load test data

test_file = '../input/test_simplified.csv'
test_df = pd.read_csv(test_file)
test_data = ExtractData([test_df], 'drawing', 3, (dim_1, dim_2))

test_keys = test_df['key_id'].tolist()
test_data = np.reshape(test_data[0], (len(test_data[0]), 1, dim_1, dim_2))
print(np.shape(test_data))
# Get net output on test set and make a submission file

def Top3(vec):
    top1, top2, top3 = -float('inf'), -float('inf'), -float('inf')
    itop1, itop2, itop3 = 0, 0, 0
    for idx, val in enumerate(vec):
        if val > top1:
            top3 = top2
            itop3 = itop2
            top2 = top1
            itop2 = itop1
            top1 = val
            itop1 = idx
        elif val > top2:
            top3 = top2
            itop3 = itop2
            top2 = val
            itop2 = idx
        elif val > top3:
            top3 = val
            itop3 = idx
    return ((itop1, top1), (itop2, top2), (itop3, top3))

def ToClasses(top, class_map):
    return [class_map[c] for c, conf in top]

def GetOutput(net, test_data, test_keys, class_map):
    net_output = net.predict(test_data, batch_size=128)
    classes_output = [ToClasses(Top3(x), class_map) for x in net_output]
    return list(zip(test_keys, classes_output))

print('Getting output')
output = GetOutput(net, test_data, test_keys, class_map)
print(output[0])

def SubmissionLine(item):
    return str(item[0]) + ', ' + ' '.join(item[1]) + '\n'

print('Writing submission file')
def MakeSubmission(output, file_name, header):
    with open(file_name, 'w', newline='\n') as file:
        file.write(header + '\n')
        for item in output:
            file.write(SubmissionLine(item))

submission_file = './submission.csv'
MakeSubmission(output, submission_file, 'key_id,word')

print('Done')