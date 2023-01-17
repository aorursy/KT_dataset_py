import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/train.csv')

df.head()
import re

def istype(name,_type):

    match = re.search('^.*'+_type+'.*$',name)

    if match:

        return True

    else:

        return False

    

def notBinOrCat(name):

    match = re.search('^.*bin.*$',name)

    if match:

        return False

    else:

        match = re.search('^.*cat.*$',name)

        if match:

            return False

        else:

            return True
ind_cols = [col for col in df.columns if istype(col,'ind')]

reg_cols = [col for col in df.columns if istype(col,'reg')]

car_cols = [col for col in df.columns if istype(col,'car')]

calc_cols= [col for col in df.columns if istype(col,'calc')]



ind_cat = [col for col in ind_cols if istype(col,'cat')]

reg_cat = [col for col in reg_cols if istype(col,'cat')]

car_cat = [col for col in car_cols if istype(col,'cat')]

calc_cat= [col for col in calc_cols if istype(col,'cat')]



ind_bin = [col for col in ind_cols if istype(col,'bin')]

reg_bin = [col for col in reg_cols if istype(col,'bin')]

car_bin = [col for col in car_cols if istype(col,'bin')]

calc_bin= [col for col in calc_cols if istype(col,'bin')]



ind_con = [col for col in ind_cols if not (istype(col,'bin') or istype(col,'cat'))]

reg_con = [col for col in reg_cols if not (istype(col,'bin') or istype(col,'cat'))]

car_con = [col for col in car_cols if not (istype(col,'bin') or istype(col,'cat'))]

calc_con= [col for col in calc_cols if not (istype(col,'bin') or istype(col,'cat'))]
zeros = df[df['target']==0][0:25000]

ones = df[df['target']==1]

rdf = pd.concat([zeros,ones],axis=0)



thist = rdf.groupby(['target'],as_index=False).count()['id']

fig,axarr = plt.subplots(1,1,figsize=(12,6))

sns.barplot(x=thist.index,y=thist.values)

rdf = rdf.sample(frac=0.5)

plt.show()
#rdf = rdf[rdf['ps_reg_03']!=-1]

#rdf = rdf[rdf['ps_car_14']!=-1]



f,axarr = plt.subplots(1,2,figsize=(15,6))

ldf = rdf.copy()



#ldf['ps_reg_03'] = np.log(rdf['ps_reg_03'])

ldf['ps_car_13'] = np.log(rdf['ps_car_13'])

sns.distplot((ldf['ps_reg_03']),ax=axarr[0])

sns.distplot((ldf['ps_car_13']),ax=axarr[1])





plt.show()
f,axarr = plt.subplots(3,2,figsize=(15,24))

hist, bin_edges_reg03 = np.histogram(ldf['ps_reg_03'], density=False)

bin_edges_reg03 = [float("{0:.2f}".format(x)) for x in bin_edges_reg03]

sns.barplot(x=bin_edges_reg03[0:len(hist)],y=hist,ax=axarr[0][0])

axarr[0][0].set_title('ps_reg_03 histogram')

ldf['ps_reg_03_cat'] = np.digitize(ldf['ps_reg_03'], bin_edges_reg03)

for tick in axarr[0][0].get_xticklabels():

        tick.set_rotation(90)

        

hist, bin_edges_car_13 = np.histogram(ldf['ps_car_13'], density=False)

bin_edges_car_13 = [float("{0:.2f}".format(x)) for x in bin_edges_car_13]

sns.barplot(x=bin_edges_car_13[0:len(hist)],y=hist,ax=axarr[0][1])

axarr[0][1].set_title('ps_car_13 histogram')

ldf['ps_car_13_cat'] = np.digitize(ldf['ps_car_13'], bin_edges_car_13)

for tick in axarr[0][1].get_xticklabels():

        tick.set_rotation(90)

        

hist, bin_edges_car_14 = np.histogram(rdf['ps_car_14'], density=False)

bin_edges_car_14 = [float("{0:.2f}".format(x)) for x in bin_edges_car_14]

sns.barplot(x=bin_edges_car_14[0:len(hist)],y=hist,ax=axarr[1][1])

axarr[1][1].set_title('ps_car_14 histogram')

ldf['ps_car_14_cat'] = np.digitize(rdf['ps_car_14'], bin_edges_car_14)

for tick in axarr[1][1].get_xticklabels():

        tick.set_rotation(90)



hist, bin_edges_calc_14 = np.histogram(rdf['ps_calc_14'], density=False)

bin_edges_calc_14 = [float("{0:.2f}".format(x)) for x in bin_edges_calc_14]

sns.barplot(x=bin_edges_calc_14[0:len(hist)],y=hist,ax=axarr[1][0])

axarr[1][0].set_title('ps_calc_14 histogram')

ldf['ps_calc_14_cat'] = np.digitize(rdf['ps_calc_14'], bin_edges_calc_14)

for tick in axarr[1][0].get_xticklabels():

        tick.set_rotation(90)



hist, bin_edges_calc_10 = np.histogram(rdf['ps_calc_10'], density=False)

bin_edges_calc_10 = [float("{0:.2f}".format(x)) for x in bin_edges_calc_10]

sns.barplot(x=bin_edges_calc_10[0:len(hist)],y=hist,ax=axarr[2][0])

axarr[2][0].set_title('ps_calc_10 histogram')

ldf['ps_calc_10_cat'] = np.digitize(rdf['ps_calc_10'], bin_edges_calc_10)

for tick in axarr[2][0].get_xticklabels():

        tick.set_rotation(90)

        

hist, bin_edges_calc_11 = np.histogram(rdf['ps_calc_11'], density=False)

bin_edges_calc_11 = [float("{0:.2f}".format(x)) for x in bin_edges_calc_11]

sns.barplot(x=bin_edges_calc_11[0:len(hist)],y=hist,ax=axarr[2][1])

axarr[2][1].set_title('ps_calc_11 histogram')

ldf['ps_calc_11_cat'] = np.digitize(rdf['ps_calc_11'], bin_edges_calc_11)

for tick in axarr[2][1].get_xticklabels():

        tick.set_rotation(90)

        



plt.show()
df.columns[rdf[rdf[car_cat]<0].any()]
nnindcat = df.columns[rdf[rdf[ind_cat]>0].any()]
nncarcat = df.columns[rdf[rdf[car_cat]>0].any()]
pd.get_dummies(rdf['ps_car_09_cat']).sum()
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import svm
tdf = pd.read_csv('../input/test.csv')

dtdf = tdf.copy()

dtdf['ps_car_13']     = np.log(tdf['ps_car_13'])

dtdf['ps_reg_03_cat'] = np.digitize(dtdf['ps_reg_03'], bin_edges_reg03)

dtdf['ps_car_13_cat'] = np.digitize(dtdf['ps_car_13'], bin_edges_car_13)

dtdf['ps_car_14_cat'] = np.digitize(dtdf['ps_car_14'], bin_edges_car_14)

dtdf['ps_calc_14_cat'] = np.digitize(dtdf['ps_calc_14'], bin_edges_calc_14)





prepro = pd.concat([rdf,dtdf])

d1 = pd.get_dummies(prepro['ps_reg_03_cat'],prefix='reg03')

d2 = pd.get_dummies(prepro['ps_car_13_cat'],prefix='car13')

d3 = pd.get_dummies(prepro['ps_car_14_cat'],prefix='car14')

d4 = pd.get_dummies(prepro['ps_calc_14_cat'],prefix='calc14')

d5 = pd.get_dummies(prepro['ps_ind_03'],prefix='ind03')

d6 = pd.get_dummies(prepro['ps_ind_15'],prefix='ind15')

dfeat = pd.concat([d1,d2,d3,d4,d5,d6],axis=1)

for f in ldf[nncarcat]:

    dfeat = pd.concat([dfeat,pd.get_dummies(prepro[f],prefix=f)],axis=1)

for f in ldf[nnindcat]:

    dfeat = pd.concat([dfeat,pd.get_dummies(prepro[f],prefix=f)],axis=1)



train_dummies = dfeat[:len(rdf)]

test_dummies  = dfeat[len(rdf):]
con_feat = ['ps_ind_03','ps_ind_15','ps_reg_03_cat','ps_car_13_cat','ps_car_14_cat','ps_calc_14_cat']



feat = pd.concat([ldf[ind_bin],ldf[nncarcat],ldf[nnindcat],ldf[con_feat]],axis=1)

dfeat =  pd.concat([ldf[ind_bin],train_dummies],axis=1)
clf_con = GradientBoostingClassifier(n_estimators = 100, random_state=0)

#togliere ps_ind_01

scores = cross_val_score(clf_con,feat,ldf['target'], cv=3)

clf_con.fit(feat,ldf['target'])

scores
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dense, Activation

from keras.models import model_from_json

from IPython.display import clear_output

class PlotLosses(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        

        self.fig = plt.figure()

        

        self.logs = []



    def on_epoch_end(self, epoch, logs={}):

        

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.i += 1

        

        clear_output(wait=True)

        plt.plot(self.x, self.losses, label="loss")

        plt.plot(self.x, self.val_losses, label="val_loss")

        plt.legend()

        plt.show();

        

plot_losses = PlotLosses()


try:



    print("trying to load model from disk")

    json_file = open('n1.json', 'r')

    loaded_model_json = json_file.read()

    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model

    print("model loaded")

    print("trying to load weights")

    loaded_model.load_weights("n1.h5")

    model = loaded_model

    print("Loaded model from disk")

except Exception as e:

    print(e)

    print("Creating new model")

    model = Sequential()

    model.add(Dense(256, input_dim=dfeat.shape[1]))

    model.add(Activation('relu'))

    model.add(Dropout(0.3))

    model.add(Dense(128))

    model.add(Activation('relu'))

    model.add(Dropout(0.3))

    model.add(Dense(64))

    model.add(Activation('relu'))

    model.add(Dense(32))

    model.add(Activation('relu'))

    

    model.add(Dense(1))

    model.add(Activation('sigmoid'))



model.compile(optimizer='sgd',

              loss='binary_crossentropy')

model.fit(np.asarray(dfeat), np.asarray(rdf['target']), epochs=30, validation_split=0.3,

          callbacks=[plot_losses], batch_size=32)



## I used about 300 epochs
model_json = model.to_json()

with open("n1.json", "w") as json_file:

    json_file.write(model_json)

model.save_weights("n1.h5")
tfeat = pd.concat([dtdf[ind_bin],dtdf[nncarcat],dtdf[nnindcat],dtdf[con_feat]],axis=1)

dtfeat =  pd.concat([tdf[ind_bin],test_dummies],axis=1)





pred = clf_con.predict_proba(tfeat)
pred1 = [p[1]*0.9 for p in pred]
pred1[0:10]
pred2 = model.predict(np.asarray(dtfeat))
pred2 = [p[0] for p in pred2]
res = pd.DataFrame({'id':tdf['id'],'target':pred1})

res.to_csv('pred1.csv',header=True,index=False)
res = pd.DataFrame({'id':tdf['id'],'target':pred1})

res.to_csv('pred2.csv',header=True,index=False)