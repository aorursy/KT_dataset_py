import pandas as pd

import numpy as np

import tensorflow as tf

import tensorflow_addons as tfa

from tensorflow.keras.models import Sequential

import tensorflow.keras.layers as L

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from tensorflow.keras.models import load_model

from sklearn.metrics import log_loss

from sklearn.decomposition import PCA

import random
random.seed(100)
train_data = pd.read_csv('../input/lishmy/train_data_exc.csv') 

test_data = pd.read_csv('../input/lish-moa/test_features.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
train_data.head()
top_feats = [  0,   1,   2,   3,   5,   6,   8,   9,  10,  11,  12,  14,  15,

        16,  18,  19,  20,  21,  23,  24,  25,  27,  28,  29,  30,  31,

        32,  33,  34,  35,  36,  37,  39,  40,  41,  42,  44,  45,  46,

        48,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,

        63,  64,  65,  66,  68,  69,  70,  71,  72,  73,  74,  75,  76,

        78,  79,  80,  81,  82,  83,  84,  86,  87,  88,  89,  90,  92,

        93,  94,  95,  96,  97,  99, 100, 101, 103, 104, 105, 106, 107,

       108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,

       121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133, 134,

       135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,

       149, 150, 151, 152, 153, 154, 155, 157, 159, 160, 161, 163, 164,

       165, 166, 167, 168, 169, 170, 172, 173, 175, 176, 177, 178, 180,

       181, 182, 183, 184, 186, 187, 188, 189, 190, 191, 192, 193, 195,

       197, 198, 199, 202, 203, 205, 206, 208, 209, 210, 211, 212, 213,

       214, 215, 218, 219, 220, 221, 222, 224, 225, 227, 228, 229, 230,

       231, 232, 233, 234, 236, 238, 239, 240, 241, 242, 243, 244, 245,

       246, 248, 249, 250, 251, 253, 254, 255, 256, 257, 258, 259, 260,

       261, 263, 265, 266, 268, 270, 271, 272, 273, 275, 276, 277, 279,

       282, 283, 286, 287, 288, 289, 290, 294, 295, 296, 297, 299, 300,

       301, 302, 303, 304, 305, 306, 308, 309, 310, 311, 312, 313, 315,

       316, 317, 320, 321, 322, 324, 325, 326, 327, 328, 329, 330, 331,

       332, 333, 334, 335, 338, 339, 340, 341, 343, 344, 345, 346, 347,

       349, 350, 351, 352, 353, 355, 356, 357, 358, 359, 360, 361, 362,

       363, 364, 365, 366, 368, 369, 370, 371, 372, 374, 375, 376, 377,

       378, 379, 380, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391,

       392, 393, 394, 395, 397, 398, 399, 400, 401, 403, 405, 406, 407,

       408, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422,

       423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435,

       436, 437, 438, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450,

       452, 453, 454, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465,

       466, 468, 469, 471, 472, 473, 474, 475, 476, 477, 478, 479, 482,

       483, 485, 486, 487, 488, 489, 491, 492, 494, 495, 496, 500, 501,

       502, 503, 505, 506, 507, 509, 510, 511, 512, 513, 514, 516, 517,

       518, 519, 521, 523, 525, 526, 527, 528, 529, 530, 531, 532, 533,

       534, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547,

       549, 550, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563,

       564, 565, 566, 567, 569, 570, 571, 572, 573, 574, 575, 577, 580,

       581, 582, 583, 586, 587, 590, 591, 592, 593, 595, 596, 597, 598,

       599, 600, 601, 602, 603, 605, 607, 608, 609, 611, 612, 613, 614,

       615, 616, 617, 619, 622, 623, 625, 627, 630, 631, 632, 633, 634,

       635, 637, 638, 639, 642, 643, 644, 645, 646, 647, 649, 650, 651,

       652, 654, 655, 658, 659, 660, 661, 662, 663, 664, 666, 667, 668,

       669, 670, 672, 674, 675, 676, 677, 678, 680, 681, 682, 684, 685,

       686, 687, 688, 689, 691, 692, 694, 695, 696, 697, 699, 700, 701,

       702, 703, 704, 705, 707, 708, 709, 711, 712, 713, 714, 715, 716,

       717, 723, 725, 727, 728, 729, 730, 731, 732, 734, 736, 737, 738,

       739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751,

       752, 753, 754, 755, 756, 758, 759, 760, 761, 762, 763, 764, 765,

       766, 767, 769, 770, 771, 772, 774, 775, 780, 781, 782, 783, 784,

       785, 787, 788, 790, 793, 795, 797, 799, 800, 801, 805, 808, 809,

       811, 812, 813, 816, 819, 820, 821, 822, 823, 825, 826, 827, 829,

       831, 832, 833, 834, 835, 837, 838, 839, 840, 841, 842, 844, 845,

       846, 847, 848, 850, 851, 852, 854, 855, 856, 858, 860, 861, 862,

       864, 867, 868, 870, 871, 873, 874]

print(f'length of top features: {len(top_feats)}')

print('top features df: ')

train_data.drop('sig_id',axis=1).iloc[:,top_feats].head()
genes = [col for col in train_data if col[0:2]=='g-']

cells = [col for col in train_data if col[0:2]=='c-']
temp = pd.read_csv('../input/lish-moa/train_features.csv')

features = temp.drop(['sig_id'],axis=1).columns

temp = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

targets = temp.drop('sig_id',axis=1).columns



del temp
test_data.head(3)
train_data.head(3)
from sklearn.preprocessing import StandardScaler
# Class to prepare data for training.

class MoaDataPrepare:

    def __init__(self,train,test):

        self.train_data = train.copy()

        self.test_data = test.copy()

        

        self.train_data["cp_time"]= self.train_data["cp_time"].map({24:'24hr',48:'48hr',72:'72hr'})

        self.train_data["cp_dose"]= self.train_data["cp_dose"].map({'D1':0,'D2':1})

        

      

        self.test_data["cp_time"]= self.test_data["cp_time"].map({24:'24hr',48:'48hr',72:'72hr'})

        self.test_data["cp_dose"]= self.test_data["cp_dose"].map({'D1':0,'D2':1})

        

        """self.train_targets = self.train_data.iloc[:,876:]

        self.train_data = self.train_data.drop('sig_id',axis=1).iloc[:,top_feats]

        self.test_data = self.test_data.drop('sig_id',axis=1).iloc[:,top_feats]"""

        

        genes = [col for col in self.train_data if col[0:2]=='g-']

        cells = [col for col in self.train_data if col[0:2]=='c-']

        

        #gene_data = self.train_data.loc[:,genes]

        cell_data = self.train_data.loc[:,cells]

        #gene_data_test = self.test_data.loc[:,genes]

        cell_data_test = self.test_data.loc[:,cells]

        

        pca=PCA(0.98)

        tran = pca.fit_transform(cell_data,self.train_data.loc[:,targets])

        conv_cell_data = pd.DataFrame(tran,columns=['pca-'+str(i) for i in range(tran.shape[1])])

        tran = pca.transform(cell_data_test)

        conv_cell_test = pd.DataFrame(tran,columns=['pca-'+str(i) for i in range(tran.shape[1])])

        

        self.train_data = self.train_prepare()

        self.test_data = self.test_prepare()

        

        self.train_data = pd.concat([self.train_data.iloc[:,:3+len(genes)],conv_cell_data,self.train_data.iloc[:,3+len(genes)+conv_cell_data.shape[1]:]],axis=1)

        self.test_data = pd.concat([self.test_data.iloc[:,:3+len(genes)],conv_cell_test],axis=1)

        

        sc = StandardScaler()

        self.train_data.loc[:,genes] = sc.fit_transform(self.train_data.loc[:,genes])

        self.test_data.loc[:,genes] = sc.transform(self.test_data.loc[:,genes])

        

        self.features = self.train_data.columns[:3+len(genes)+conv_cell_data.shape[1]]

        self.targets = targets

        

    def train_prepare(self):

        self.train_data = self.train_data.drop(['sig_id','cp_type'],axis=1)

        cp_time = pd.get_dummies(self.train_data.cp_time,drop_first = True)

    

        self.train_data = pd.concat([cp_time,self.train_data.drop(['cp_time'],axis=1)],axis=1)

        

        return self.train_data

    

    def test_prepare(self):

        self.test_data = self.test_data.drop(['sig_id','cp_type'],axis=1)

        cp_time = pd.get_dummies(self.test_data.cp_time,drop_first = True)

        

    

        self.test_data = pd.concat([cp_time,self.test_data.drop(['cp_time'],axis=1)],axis=1)

        return self.test_data

    

    

    def getFold(self,fold):

        X_train = self.train_data[self.train_data.kfold!=fold].loc[:,self.get_test().columns]

        X_val = self.train_data[self.train_data.kfold==fold].loc[:,self.get_test().columns]

        

        Y_train = self.train_data[self.train_data.kfold!=fold].loc[:,self.targets]

        Y_val = self.train_data[self.train_data.kfold==fold].loc[:,self.targets]

        

        return X_train,X_val,Y_train,Y_val

    

    def get_train(self):

        return self.train_data.loc[:,self.get_test().columns]

    

    def get_test(self):

        return self.test_data

    

    def __del__(self):

        del self.train_data

        del self.test_data
# Class for building the model and implementing the score metric

class ModelBuilder:

    def __init__(self,random_state=666):

        self.random_state=random_state

        

    def SimpleNeuralNet(self,shape=None,learning_rate=0.001,output_shape=206):

        """shape=None, lr=0.001"""

        model = tf.keras.models.Sequential([

                L.InputLayer(input_shape=shape),

                L.BatchNormalization(),

                L.Dropout(0.5),

                tfa.layers.WeightNormalization(L.Dense(256,kernel_initializer="he_normal")),

                L.BatchNormalization(),

                L.Activation(tf.nn.leaky_relu),

                L.Dropout(0.5),

                tfa.layers.WeightNormalization(L.Dense(128,kernel_initializer="he_normal")),

                L.BatchNormalization(),

                L.Activation(tf.nn.leaky_relu),

                L.Dropout(0.3),

                tfa.layers.WeightNormalization(L.Dense(output_shape,activation="sigmoid",kernel_initializer="he_normal"))

            ])

        

        model.compile(optimizer= tfa.optimizers.AdamW(lr=learning_rate,weight_decay=1e-5, clipvalue=900),loss="binary_crossentropy",metrics=["binary_crossentropy"])

        return model

    

    def transfer_weight(self,model_source,model_dest):

        for i in range(len(model_source.layers[:-1])):

            model_dest.layers[i].set_weights(model_source.layers[i].get_weights())

        return model_dest

    

    def metric(self,train,predict):

        metrics=[]

        for col in range(train.shape[1]):

            metrics.append(log_loss(train.iloc[:,col],predict[:,col],labels=[0,1]))

        return np.mean(metrics)
data_prepare = MoaDataPrepare(train_data,test_data)
data_prepare.get_train()
model_builder = ModelBuilder(random_state=606)
transfer_data = data_prepare.get_train()

transfer_data.loc[:,'sig_id'] = train_data.sig_id

transfer_data = transfer_data.merge(train_targets_nonscored,how='inner',on='sig_id')

transfer_data = transfer_data.drop('sig_id',axis=1)

Y_transfer = transfer_data.loc[:,train_targets_nonscored.columns[1:]]

transfer_data = transfer_data.loc[:,data_prepare.get_train().columns]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(transfer_data,Y_transfer,test_size=0.2,random_state = 101)
model = model_builder.SimpleNeuralNet(shape = (x_train.shape[1],),output_shape = 402)
save_weight = tf.keras.callbacks.ModelCheckpoint('model.learned.hdf5',save_best_only=True,save_weights_only=True,monitor = 'val_loss',mode='min')

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')

early = EarlyStopping(monitor='val_loss',patience=5,mode='min')
model.fit(x_train,y_train,

          epochs=30,

          batch_size=128,

          validation_data=(x_test,y_test),

          callbacks=[early,save_weight,reduce_lr_loss]

         )
model.load_weights('model.learned.hdf5')
ss = pd.read_csv('../input/lish-moa/sample_submission.csv')

ss.loc[:,'sig_id'] = test_data['sig_id'].values

ss.iloc[:,1:]=0
histories=[]

scores = []



    

random.seed(100)

for fold in range(5):



    x_train,x_val,y_train,y_val = data_prepare.getFold(fold)

    

    print(f'Fold: {fold}\n')



    tf.keras.backend.clear_session()

    print('training with transfered weights')

    model_fin = model_builder.SimpleNeuralNet(shape=(x_train.shape[1],),learning_rate=0.001,output_shape=206)

    model_fin = model_builder.transfer_weight(model,model_fin)

    for layer in model_fin.layers:

        layer.trainable=True



    checkpoint_path = f'best_model_{fold}.hdf5'



    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')



    checkpt = ModelCheckpoint(checkpoint_path, monitor = 'val_loss', verbose = 0, save_best_only = True,

                                  mode = 'min')



    early = EarlyStopping(monitor='val_loss',patience=5,mode='min')



    history=model_fin.fit(x_train.values,

              y_train.values,

              validation_data=(x_val.values,y_val.values),

              epochs=50, batch_size=128,

              callbacks=[early,reduce_lr_loss,checkpt]

             )

    print('')

    histories.append(history)



    model_fin= tf.keras.models.load_model(checkpoint_path, custom_objects={'leaky_relu': tf.nn.leaky_relu})

    score= model_builder.metric(y_val,model_fin.predict(x_val).astype(float))

    scores.append(score)

    print(f'Validation metric: {score}')

    test_predict = model_fin.predict(data_prepare.get_test().values)



    ss.loc[:, y_train.columns] += test_predict

    print('')

ss.loc[:, train_targets_scored.columns[1:]] /= 5
ss.loc[test_data[test_data.cp_type=="ctl_vehicle"].index,train_targets_scored.drop('sig_id',axis=1).columns]=0
np.mean(scores)
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))

plt.title('training_curve')

for h in histories:

    plt.plot(h.history['val_loss'],color='red',label='val')

    plt.plot(h.history['loss'],color="green",label='train')

plt.legend()
ss.to_csv('./submission.csv',index=False)