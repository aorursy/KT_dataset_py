!pip install deepctr --no-deps

from deepctr.utils import SingleFeat

from deepctr.models import DeepFM, FNN, PNN, xDeepFM, NFM, NFFM, AFM

from sklearn.metrics import roc_auc_score

import numpy as np

import tensorflow.keras.optimizers

import tensorflow as tf

import gc



feature_size = 645195

field_size = 24

size = 100

test_size = 100

true_sample_size = int(size * 0.75)

train_xi = []

train_xv = []

data_xi = []

data_xv = []

label = []



def auc(y_true, y_pred):

    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)



def main():

    gc.collect()

    global train_xi

    global train_xv

    global data_xi

    global data_xv

    global label

    feature_dict =  {'uid':101449, 'day':7, 'hour':24, 'dev': 5, 'devmodel':5925, 'ip':523672,'domain':3487,'cct':4,'sid':3135,'stp':24,'slotp':7, 'iid':4002,

                    'f0':28,'f1':252,'f2':10, 'f3':67, 'f4':8, 'f5':7, 'f6':426, 'f7':9, 'f8':166, 'f9':4, 'f10':60, 'f11':2417}

    #feature_dict =  {'uid':101449, 'day':7, 'hour':24, 'dev': 5, 'devmodel':5925,'domain':3487,'cct':4,'sid':3135,'stp':24,'slotp':7, 'iid':4002,

    #                'f0':28,'f1':252,'f2':10, 'f3':67, 'f4':8, 'f5':7, 'f6':426, 'f7':9, 'f8':166, 'f9':4, 'f10':60, 'f11':2417}

    feature_base = [0, 101449, 101456, 101480, 101485, 107410, 631082, 634569, 634573, 637708, 637732, 637739, 641741, 641769, 642021, 642031, 

                    642098, 642106, 642113, 642539, 642548, 642714, 642718, 642778]

    print(len(feature_base))

    lst = [SingleFeat(k, feature_dict[k]) for k in feature_dict]

    model = FNN({'sparse':lst},hidden_size=(128, 128), embedding_size=8)

    model.compile('adam', "binary_crossentropy", metrics=[auc])

    model.summary()

    with open("../input/train.data") as file:

    #with open("../input/sample/train_sample") as file:

        for line in file:

            line = line.strip()

            if line != "":

                tmpdidx = []

                ls = line.split()

                label.append(int(ls[0]))

                for i in range(1, len(ls)):

                    #if i != 6:

                    res = ls[i].split(":")

                    tmpdidx.append(int(res[0]) - feature_base[i-1])

                train_xi.append(tmpdidx)

    model.fit(np.transpose(train_xi).tolist(), label, batch_size = 256, epochs=10, verbose=0, validation_split=0)

    del lst

    gc.collect()

    with open("../input/test.data") as file:

    #with open("../input/sampletest/test_sample") as file:

        for line in file:

            line = line.strip()

            if line != "":

                tmpdidx = []

                ls = line.split()

                for i in range(1, len(ls)):

                    #if i != 6:

                    res = ls[i].split(":")

                    tmpdidx.append(int(res[0]) - feature_base[i-1])

                data_xi.append(tmpdidx)

    pred = model.predict(np.transpose(data_xi).tolist(), batch_size=512)

    file = open("submitFNN.csv", "w")

    file.write("Id,Predicted\n")

    for i in range(len(pred)):

        file.write("%d,%.10f\n" % (i, pred[i]))

    file.close()

    gc.collect()

                

if __name__=='__main__':

    main()