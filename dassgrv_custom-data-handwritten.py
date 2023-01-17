import numpy as np 
import pandas as pd 
import os
import h5py
import matplotlib.pylab as plt
from matplotlib import cm
%matplotlib inline
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing import image as keras_image
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
import scipy
from scipy import misc


def get_max_pad(I,TI,R,C,M,rr,cc):	
	lt = 0; lb = 0;	ll = 0;	lr = 0;	
	for i in range(0,len(R)):		
		if( (R[i]-M < 0) ):
			if(np.abs(R[i]-M) > lt):
				lt = np.abs(R[i]-M);				
		if( (C[i]-M < 0)):
			if(np.abs(C[i]-M) > ll):
				ll = np.abs(C[i]-M);				
		if(R[i]+M > I.shape[0]): 
			if(np.abs(I.shape[0] - (R[i]+M) + 1) > lb):
				lb = np.abs(I.shape[0] - (R[i]+M) + 1);
		if(C[i]+M > I.shape[1]):
			if(np.abs(I.shape[1] - (C[i]+M) + 1) > lr):
				lr = np.abs(I.shape[1] - (C[i]+M) + 1);					
	I = cv2.copyMakeBorder(I,lt,lb,ll,lr,cv2.BORDER_CONSTANT,0);
	TI = cv2.copyMakeBorder(TI,lt,lb,ll,lr,cv2.BORDER_CONSTANT,0);
	R = list(np.asarray(R) + lt);
	C = list(np.asarray(C) + ll);
	rr = list(np.asarray(rr) + lt);
	cc = list(np.asarray(cc) + ll);
	return I,TI,R,C,rr,cc,lt,ll;
def add_rows_cols(I,N):
	lr = np.abs(N - I.shape[0]);
	lc = np.abs(N - I.shape[1]);
	I = cv2.copyMakeBorder(I,0,lr,lc,0,cv2.BORDER_CONSTANT,0);
	return I;
def get_data(fname,mode,N,reshap):
	if(N%2!=0):
		N = N-1;
	if (N <= 0):
		print('Please make sure window size >=2, window size too small, exiting');
		sys.exit(1);
	print('\nProcessing:',fname,'\t');
	print('Window Size:', N+1,'x',N+1,'\n');
	def wind_new(TI,I,r,c,R,C,mode,K,window_size):
		M = int(window_size/2);
		final_i = None;	
		for i,g in enumerate(R):
			if (R[i] == r) & (C[i] == c):
				final_i = i;
				break;
		if(final_i == None):
			print('indexing error, skipping');
			return I[1:32,1:32],0,[],[];
		lor = final_i-M;
		upp = final_i+M;
		
		if(lor < 0):
			lor = 0;
		
		newr = R[lor:upp];
		newc = C[lor:upp];
		colo = list(np.random.choice(range(256),size=3));
		while(colo==[255,255,255]):
			colo = list(np.random.choice(range(256),size=3));
		K[newr,newc,:] = colo;
		
		nr = []; nc = [];
		nr = newr[round(len(newr)/2)];
		nc = newc[round(len(newr)/2)];
		
		if mode == 'stroke':
			mask = np.zeros(TI.shape,'uint8');
			mask[newr,newc] = 255;
			#globi[0] = globi[0] + 1;
			#if(mask[r-M:r+M,c-M:c+M].shape[0] == 99):
				#pdb.set_trace();
			return mask[r-M:r+M,c-M:c+M],1,nr,nc;
		elif mode == 'bgr':
			mask = np.zeros(I.shape,'bool');
			mask[newr,newc,:] = True;
			l = np.zeros(I.shape,I.dtype) + 255;
			np.copyto(l,I,'same_kind',mask);
			return l[r-M:r+M,c-M:c+M],1,nr,nc;
		else:
			print("Some error!")
			return I[1:32,1:32],0,[],[];
	I = cv2.imread(fname);
	IG = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY);
	th2,TG = cv2.threshold(IG,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	r = [];
	c = [];	
	for i in range(0,TG.shape[1]):
		if(sum(TG[:,i] != 0) != 0):
			tem = list(np.where(TG[:,i] != 0)[0]);
			r+= tem;
			c += [i]*len(tem);
	rr = [];
	cc = [];
	for i,C in enumerate(c):
		if(i%N == 0):
			rr.append(r[i]);
			cc.append(C);
	# pdb.set_trace();		
	I,TG,rr,cc,r,c,lt,ll = get_max_pad(I,TG,rr,cc,int(N/2),r,c);
	K = np.zeros(I.shape,I.dtype) + [255,255,255];
	L = np.zeros(I.shape,'uint8');
	L[rr,cc,:] = [255,255,255];
	NR = [];
	NC = [];
	
	# pdb.set_trace();
	if mode == 'stroke':
		fin_arr = [];
		for i,m in enumerate(rr):
			iii,mm,nr,nc = wind_new(TG,I,m,cc[i],r,c,mode,K,N);
			nr = nr - lt;
			nc = nc - ll;
			
			if((iii.shape[0] != N) | (iii.shape[1] != N)):
				iii = add_rows_cols(iii,N);				
			iii = cv2.resize(iii,(reshap,reshap));
			#iii = np.zeros((32,32)); mm = 1;
			if(mm == 0):
				continue;
			if i == 0:
				fin_arr = np.expand_dims(iii,2);
				NR.append(nr);
				NC.append(nc);
				continue;
			if ( (i > 0) & (len(fin_arr) == 0)):
				fin_arr = np.expand_dims(iii,2);
				continue;
			#if(iii.shape[0] == 0):
				# pdb.set_trace();
			fin_arr = np.concatenate((fin_arr,np.expand_dims(iii,2)),2);
			NR.append(nr);
			NC.append(nc);
		cv2.imwrite('strokes'+ str(N) +'_.jpg',K);	
		fin_arr = np.rollaxis(fin_arr,2,0);
		fin_arr = np.expand_dims(fin_arr,3);
		return fin_arr/255,NR,NC;
	if mode == 'bgr':
		fin_arr = [];
		for i,m in enumerate(rr):
			iii,mm,nr,nc = wind_new(TG,I,m,cc[i],r,c,mode,K,N);
			nr = nr - lt;
			nc = nc - ll;
			
			if((iii.shape[0] != N) | (iii.shape[1] != N)):
				iii = add_rows_cols(iii,N);	
			iii = cv2.resize(iii,(reshap,reshap));
			if(mm == 0):
				continue;
			if i == 0:
				fin_arr = np.expand_dims(iii,3);
				NR.append(nr);
				NC.append(nc);
				continue;
			if ((i > 0) & (len(fin_arr) == 0)):
				fin_arr = np.expand_dims(iii,3);
				continue;		
			fin_arr = np.concatenate((fin_arr,np.expand_dims(iii,3)),3);
			NR.append(nr);
			NC.append(nc);			
		cv2.imwrite('strokes'+ str(N) +'_.jpg',K);
		fin_arr = np.rollaxis(fin_arr,3,0);
		return fin_arr/255,NR,NC;
    
    
def draw_labels(labs,P,NR,NC,char):
    I = cv2.imread(P);
    fnam, ext = os.path.splitext(os.path.basename(P));
    print(fnam);
    for i,m in enumerate(labs):
        cv2.putText (I,str(ord(m)),(NC[i],NR[i]),1, 0.8, (0,0,255),1,2); 
    cv2.imwrite(fnam+char + '_labels.jpeg',I);
    plt.imshow(I)
    plt.show()
    


    
    
def vgg16_model():
    model = Sequential()
    # Define a model architecture    
    model.add(GlobalAveragePooling2D(input_shape=(1,1,512)))
    
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.02))
    model.add(Dropout(0.5))
        
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.02))
    model.add(Dropout(0.5))
    
    model.add(Dense(33, activation='softmax'))
    # Compile the model     
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

vgg16_model = vgg16_model()

def vgg19_model():
    model = Sequential()
    # Define a model architecture    
    model.add(GlobalAveragePooling2D(input_shape=(1,1,512)))
    
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.02))
    model.add(Dropout(0.5))
        
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.02))
    model.add(Dropout(0.5))
    model.add(Dense(33, activation='softmax'))
    # Compile the model     
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

vgg19_model = vgg19_model()

def resnet50_model():
    model = Sequential()
    # Define a model architecture    
    model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.02))
    model.add(Dropout(0.5))
        
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.02))
    model.add(Dropout(0.5))
    model.add(Dense(33, activation='softmax'))
    # Compile the model     
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

resnet50_model = resnet50_model()


vgg19_model.load_weights('/kaggle/input/weights/weights.best.vgg19.hdf5');
vgg16_model.load_weights('/kaggle/input/weights/weights.best.vgg16.hdf5');
resnet50_model.load_weights('/kaggle/input/weights/weights.best.resnet50.hdf5');


dname = "../input/plots/";
vgg16_base_model = VGG16(weights='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',  include_top=False);
vgg19_base_model = VGG19(weights='../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',  include_top=False);
resnet50_base_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',  include_top=False);

labels = ['а','б','в','г','д','е','ё','ж','з','и','й', 'к','л','м','н','о','п','р','с','т','у','ф', 'х','ц','ч','ш','щ','ъ','ы','ь','э','ю','я'];

def get_letters(preds,labels):
    finlab = [];
    for i in preds:
       finlab.append(labels[i]);
    return finlab;

for fname in os.listdir(dname):
    ttvlist = []
    dat,nr,nc = get_data(dname+fname,'bgr',200,32)
    resize_x_test = np.array([scipy.misc.imresize(dat[i], (48,48,3)) for i in range(0, len(dat))]).astype('float32')
    # vgg 16
    vgg16_x_test = vgg16_preprocess_input(resize_x_test)
    x_test_bn_16 = vgg16_base_model.predict(vgg16_x_test)
    vgg16pred = vgg16_model.predict(x_test_bn_16)
    l16 = list(vgg16pred.argmax(1))
    fin16 = get_letters(l16,labels);
    draw_labels(fin16,dname+fname,nr,nc,'_16');
    pd.DataFrame.from_records(np.asarray(fin16)).to_csv(fname + '_resnet16_results.csv',',');
    
    #vgg 19
    vgg19_x_test = vgg16_preprocess_input(resize_x_test)
    x_test_bn_19 = vgg19_base_model.predict(vgg19_x_test)
    vgg19pred = vgg19_model.predict(x_test_bn_19)
    l19 = list(vgg19pred.argmax(1))
    fin19 = get_letters(l19,labels);
    draw_labels(fin19,dname+fname,nr,nc,'_19');
    pd.DataFrame.from_records(np.asarray(fin19)).to_csv(fname + '_resnet19_results.csv',',');
    
    #resnet 50
    resize_x_test = np.array([scipy.misc.imresize(dat[i], (197,197,3)) for i in range(0, len(dat))]).astype('float32')
    resnet50_x_test = resnet50_preprocess_input(resize_x_test)
    x_test_bn_50 = resnet50_base_model.predict(resnet50_x_test)
    resnet50pred = resnet50_model.predict(x_test_bn_50)
    l50 = list(resnet50pred.argmax(1))
    fin50 = get_letters(l50,labels);
    draw_labels(fin50,dname+fname,nr,nc,'_50');
    pd.DataFrame.from_records(np.asarray(fin50)).to_csv(fname + '_resnet50_results.csv',',');