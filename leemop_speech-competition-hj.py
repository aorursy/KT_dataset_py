import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd
import random
# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa

from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display

import pandas as pd

import csv
import h5py
import math
from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pydub import AudioSegment
%%time
#!cp ../input/tensorflow-speech-recognition-challenge/train.7z .
#!cp ../input/tensorflow-speech-recognition-challenge/sample*.7z .
#!cp ../input/train.7z .
#!cp ../input/test.7z .
!apt -y install libsndfile1
!pip3 install pydub
# ON-TPU
#!apt install p7zip
#!/usr/bin/p7zip -h
#!p7zip -d train.7z
#!p7zip -d sample*.7z
#!ls train/audio/
!rm train.7z
#!rm -rf train/audio/_back*
!find train/audio/. | grep "\.wav" > trainwav.txt
#!ls train/
#!df . -h
class Preprocess:	

	def __init__(self):		
		self.pathlist = "kaggle.txt"
		self.base_path = "dataset/train/audio/"
		self.frame_length = 0.025
		self.frame_stride = 0.010
		self.sample_rate = 16000
		self.input_nfft = int(round(self.sample_rate * self.frame_length))
		self.input_stride = int(round(self.sample_rate * self.frame_stride))
		self.labels = {'yes':0, 'no':1, 'up':2, 'down':3, 'left':4, 'right':5, 'on':6, 'off':7, 'stop':8, 'go':9, 'silence':10, 'unknown':11}


	def getwavelist(self):
		f = open(self.base_path + self.pathlist,'r')
		self.filelist = f.readlines()
		random.shuffle(self.filelist)

		self.numofdata = len(self.filelist)
		print(self.numofdata, "is num of data")


	def split_silence(self):
		basic_path = "dataset/train/audio/_background_noise_/"
		dest_path = "dataset/train/audio/silence/"

		sil_fl = os.listdir(basic_path)
		sil_fl.remove("README.md")

		numof_silfile = 0
		for fidx, fl in enumerate(sil_fl):    
			s, sr = librosa.load(basic_path + fl, sr=16000)
			filelen = int(len(s)/sr)

			audio = AudioSegment.from_wav(basic_path + fl)

			for sidx in range(0, (filelen-1)*1000, 250):
				newAudio = audio[sidx:sidx+1000]
				newAudio.export(dest_path + '%d_%d.wav' % (fidx, sidx) , format="wav")
				numof_silfile += 1

		print("%d silence file is created." % numof_silfile)


	def preprocessing(self):
		train_data = np.zeros([self.numofdata, 40, 101], dtype=float)
		train_label = np.zeros([self.numofdata], dtype=int)

		unknown_cnt = 0
		total_cnt = 0
		for fl in self.filelist:
		    fl = fl[:-1]
		    lab = self.labels.get(fl.split("/")[1])    
		    
		    if str(lab) == "None":
		        if unknown_cnt >= 4000:
		            continue
		            
		        train_label[total_cnt] = 11
		        unknown_cnt += 1        
		    else:        
		        train_label[total_cnt] = lab
		        
		    samples, sample_rate = librosa.load(self.base_path + fl, sr=16000)
		    S = librosa.feature.melspectrogram(y=samples, n_mels=40, n_fft=self.input_nfft, hop_length=self.input_stride)
		    if S.shape[1] != 101:
		        S = librosa.util.fix_length(S, 101, axis=1) # zero-paddings
		        
		    train_data[total_cnt] = S
		    total_cnt += 1
		    if total_cnt%2000 == 0 :
		        print(total_cnt)

		self.numofdata = total_cnt
		print("%d file processed. Done." % self.numofdata)

		train_data = train_data[:self.numofdata]
		t_data  = train_data.reshape(self.numofdata, 40, 101, 1)
		t_label = keras.utils.to_categorical(train_label, 12)

		return t_data, t_label
class Modeling:

	def __init(self):
		pass


	def build_model(self):  
		self.model = keras.Sequential([
			layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=[40,101,1]),
		    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
		    layers.Dropout(0.2),
		    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
		    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
		    layers.Dropout(0.3),
		    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
		    layers.Flatten(),
		    layers.Dense(256, activation='relu'),
		    layers.Dropout(0.5),
		    layers.Dense(12, activation='softmax')
		])
  		
		optimizer = tf.keras.optimizers.Adam()
		self.model.compile(loss='categorical_crossentropy',
                	optimizer=optimizer,
                	metrics=['acc'])

		print(self.model.summary())


	def load_model(self, weight):
		self.model.load_weights(weight)


	def save_model(self, filename):
		self.model.save_weights(filename)


	def train_model(self, train_data, train_label, bs=512, epoch=5, v_split=0.1):
		history = self.model.fit(train_data, train_label, shuffle=True, batch_size=bs,
								 epochs=epoch, validation_split=v_split, verbose=1)
		return history


	def predict_model(self, test_data, bs=512):
		predict_result = self.model.predict(test_data, batch_size=bs, verbose=1)
		pred_res = np.argmax(predict_result, axis=-1)		
		return pred_res
class Postprocess:

	def __init__(self):
		self.pathlist = "kaggle.txt"
		self.base_path = "dataset/test/audio/"
		self.submission_file = "hj_submission.csv"
		self.frame_length = 0.025
		self.frame_stride = 0.010
		self.sample_rate = 16000
		self.input_nfft = int(round(self.sample_rate * self.frame_length))
		self.input_stride = int(round(self.sample_rate * self.frame_stride))
		self.labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']

		IdLookupTable = pd.read_csv("dataset/sample_submission.csv", na_values = "?", comment='\t', sep=",", skipinitialspace=True)
		self.Fname = IdLookupTable["fname"]
		self.numofdata = len(self.Fname)		


	def makehdf5(self, f_towrite):		
		f = open(self.base_path + self.pathlist, 'r')
		filelist = f.readlines()
		test_data = np.zeros([self.numofdata, 40, 101], dtype=float)

		for idx, fl in enumerate(filelist):
		    fl = fl[:-1]

		    samples, sample_rate = librosa.load(self.base_path + fl, sr=self.sample_rate)
		    S = librosa.feature.melspectrogram(y=samples, n_mels=40, n_fft=self.input_nfft, hop_length=self.input_stride)
		    if S.shape[1] != 101:
		        S = librosa.util.fix_length(S, 101, axis=1)  # zero-paddings

		    test_data[idx] = S
		    if idx % 2000 == 0:
		        print(idx)

		with h5py.File(f_towrite, 'w') as hf:
		    dy_str = h5py.special_dtype(vlen=str)
		    fn_data = hf.create_dataset('feature', [self.numofdata, 40, 101], dtype=float)
		    fn_data[:self.numofdata] = test_data[:self.numofdata]

		f.close()

	def loadhdf5(self, f_toread):
		with h5py.File(f_toread, 'r') as hf:
			for i in range(math.ceil(self.numofdata / 10000)):
				if (i+1)*10000 > self.numofdata:
					feature = hf.get('feature')[i*10000 : self.numofdata]
					yield feature.reshape(self.numofdata - i*10000, 40, 101, 1)
				else:
		  			feature = hf.get('feature')[i*10000 : (i+1)*10000]
		  			yield feature.reshape(10000, 40, 101, 1)


	def postprocessing(self, predict_res):
		f = open(self.submission_file, 'w', newline='')		
		wr = csv.writer(f)
		wr.writerow(['fname','label'])

		for i in range(self.numofdata):
		    pred_label = self.labels[int(predict_res[i])]
		    wr.writerow([self.Fname[i], pred_label]) 

		f.close()
def main():

	# PREPROCESSING
	pre_process = Preprocess()
	#pre_process.split_silence() # 1568 silence file is created.
	#pre_process.getwavelist() # 66289 is num of data
	#t_data, t_label = pre_process.preprocessing() # but only 29500 file is used in here

	# MODELING AND TRAINING
	model = Modeling()
	model.build_model()
	#history = model.train_model(t_data, t_label, epoch=30)

	# save weights
	#model.save_model("dataset/train/" + "hj_weights.h5")
	model.load_model("dataset/train/hj_weights.h5")

	#result = model.predict_model(t_data[:512])
	#print(result)

	# POSTPROCESSING
	post_process = Postprocess()
	#post_process.makehdf5("dataset/test/postprocess.hdf5") # only once is enough

	post_feature = post_process.loadhdf5("dataset/test/postprocess.hdf5")
	test_predict = np.zeros([post_process.numofdata], dtype=int)
	for idx, pf in enumerate(post_feature):
		pred_res = model.predict_model(pf)		
		test_predict[idx*10000:idx*10000+len(pred_res)] = pred_res
	
	post_process.postprocessing(test_predict)
model = Modeling()
model.build_model()
#model.load_model("weights/hj_weights.h5")
!ls
if __name__ == "__main__":    
    pass
    #main()
