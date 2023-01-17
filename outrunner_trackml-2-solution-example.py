import numpy as np

import pandas as pd

import os

from keras.models import Model, Sequential, load_model

from keras.layers import Dense, Input

from keras.optimizers import Adam

from tqdm import tqdm_notebook

print(os.listdir("../input"))

print(os.listdir("../input/trackml/"))

prefix='../input/trackml-particle-identification/'

import zipfile
def init_model(fs = 10):

    model = Sequential()

    model.add(Dense(800, activation='selu', input_shape=(fs,)))

    model.add(Dense(400, activation='selu'))

    model.add(Dense(400, activation='selu'))

    model.add(Dense(400, activation='selu'))

    model.add(Dense(200, activation='selu'))

    model.add(Dense(1, activation='sigmoid'))

    return model



def get_event(event):

    zf = zipfile.ZipFile('../input/trackml-particle-identification/train_1.zip')

    hits= pd.read_csv(zf.open('train_1/%s-hits.csv'%event))

    cells= pd.read_csv(zf.open('train_1/%s-cells.csv'%event))

    truth= pd.read_csv(zf.open('train_1/%s-truth.csv'%event))

    particles= pd.read_csv(zf.open('train_1/%s-particles.csv'%event))

    return hits, cells, truth, particles
# you can jump to step4 for test only.

train = False

if train:

    Train = []

    for i in tqdm_notebook(range(10,20)):

        event = 'event0000010%02d'%i

        hits, cells, truth, particles = get_event(event)

        hit_cells = cells.groupby(['hit_id']).value.count().values

        hit_value = cells.groupby(['hit_id']).value.sum().values

        features = np.hstack((hits[['x','y','z']]/1000, hit_cells.reshape(len(hit_cells),1)/10,hit_value.reshape(len(hit_cells),1)))

        particle_ids = truth.particle_id.unique()

        particle_ids = particle_ids[np.where(particle_ids!=0)[0]]



        pair = []

        for particle_id in particle_ids:

            hit_ids = truth[truth.particle_id == particle_id].hit_id.values-1

            for i in hit_ids:

                for j in hit_ids:

                    if i != j:

                        pair.append([i,j])

        pair = np.array(pair)   

        Train1 = np.hstack((features[pair[:,0]], features[pair[:,1]], np.ones((len(pair),1))))



        if len(Train) == 0:

            Train = Train1

        else:

            Train = np.vstack((Train,Train1))



        n = len(hits)

        size = len(Train1)*3

        p_id = truth.particle_id.values

        i =np.random.randint(n, size=size)

        j =np.random.randint(n, size=size)

        pair = np.hstack((i.reshape(size,1),j.reshape(size,1)))

        pair = pair[((p_id[i]==0) | (p_id[i]!=p_id[j]))]



        Train0 = np.hstack((features[pair[:,0]], features[pair[:,1]], np.zeros((len(pair),1))))



        print(event, Train1.shape)



        Train = np.vstack((Train,Train0))

    del Train0, Train1



    np.random.shuffle(Train)

    print(Train.shape)
if train:

    model = init_model()
if train:

    lr=-5

    model.compile(loss=['binary_crossentropy'], optimizer=Adam(lr=10**(lr)), metrics=['accuracy'])

    History = model.fit(x=Train[:,:-1], y=Train[:,-1], batch_size=8000, epochs=1, verbose=2, validation_split=0.05, shuffle=True)
if train:

    lr=-4

    model.compile(loss=['binary_crossentropy'], optimizer=Adam(lr=10**(lr)), metrics=['accuracy'])

    History = model.fit(x=Train[:,:-1], y=Train[:,-1], batch_size=8000, epochs=20, verbose=2, validation_split=0.05, shuffle=True)
if train:

    lr=-5

    model.compile(loss=['binary_crossentropy'], optimizer=Adam(lr=10**(lr)), metrics=['accuracy'])

    History = model.fit(x=Train[:,:-1], y=Train[:,-1], batch_size=8000, epochs=3, verbose=2, validation_split=0.05, shuffle=True)
# if you skip step2, you still need to run step1 to get training data.

if train:

    try:

        model

    except NameError:

        print('load model')

        model = load_model('../input/trackml/my_model.h5')
if train:

    Train_hard = []



    for i in tqdm_notebook(range(10,20)):



        event = 'event0000010%02d'%i

        hits, cells, truth, particles = get_event(event)

        hit_cells = cells.groupby(['hit_id']).value.count().values

        hit_value = cells.groupby(['hit_id']).value.sum().values

        features = np.hstack((hits[['x','y','z']]/1000, hit_cells.reshape(len(hit_cells),1)/10,hit_value.reshape(len(hit_cells),1)))



        size=30000000

        n = len(truth)

        i =np.random.randint(n, size=size)

        j =np.random.randint(n, size=size)

        p_id = truth.particle_id.values

        pair = np.hstack((i.reshape(size,1),j.reshape(size,1)))

        pair = pair[((p_id[i]==0) | (p_id[i]!=p_id[j]))]



        Train0 = np.hstack((features[pair[:,0]], features[pair[:,1]], np.zeros((len(pair),1))))



        pred = model.predict(Train0[:,:-1], batch_size=20000)

        s = np.where(pred>0.5)[0]



        print(event, len(Train0), len(s))



        if len(Train_hard) == 0:

            Train_hard = Train0[s]

        else:

            Train_hard = np.vstack((Train_hard,Train0[s]))

    del Train0

    print(Train_hard.shape)
if train:

    Train = np.vstack((Train,Train_hard))

    np.random.shuffle(Train)

    print(Train.shape)
if train:

    lr=-4

    model.compile(loss=['binary_crossentropy'], optimizer=Adam(lr=10**(lr)), metrics=['accuracy'])

    History = model.fit(x=Train[:,:-1], y=Train[:,-1], batch_size=8000, epochs=30, verbose=2, validation_split=0.05, shuffle=True)
if train:

    lr=-5

    model.compile(loss=['binary_crossentropy'], optimizer=Adam(lr=10**(lr)), metrics=['accuracy'])

    History = model.fit(x=Train[:,:-1], y=Train[:,-1], batch_size=8000, epochs=10, verbose=2, validation_split=0.05, shuffle=True)
if train:

    lr=-6

    model.compile(loss=['binary_crossentropy'], optimizer=Adam(lr=10**(lr)), metrics=['accuracy'])

    History = model.fit(x=Train[:,:-1], y=Train[:,-1], batch_size=8000, epochs=2, verbose=2, validation_split=0.05, shuffle=True)
try:

    model

except NameError:

    print('load model')

    model = load_model('../input/trackml/my_model_h.h5')
event = 'event000001001'

hits, cells, truth, particles = get_event(event)

hit_cells = cells.groupby(['hit_id']).value.count().values

hit_value = cells.groupby(['hit_id']).value.sum().values

features = np.hstack((hits[['x','y','z']]/1000, hit_cells.reshape(len(hit_cells),1)/10,hit_value.reshape(len(hit_cells),1)))

count = hits.groupby(['volume_id','layer_id','module_id'])['hit_id'].count().values

module_id = np.zeros(len(hits), dtype='int32')



for i in range(len(count)):

    si = np.sum(count[:i])

    module_id[si:si+count[i]] = i
def get_path(hit, mask, thr):

    path = [hit]

    a = 0

    while True:

        c = get_predict(path[-1], thr/2)

        mask = (c > thr)*mask

        mask[path[-1]] = 0

        

        if 1:

            cand = np.where(c>thr)[0]

            if len(cand)>0:

                mask[cand[np.isin(module_id[cand], module_id[path])]]=0

                

        a = (c + a)*mask

        if a.max() < thr*len(path):

            break

        path.append(a.argmax())

    return path



def get_predict(hit, thr=0.5):

    Tx = np.zeros((len(truth),10))

    Tx[:,5:] = features

    Tx[:,:5] = np.tile(features[hit], (len(Tx), 1))

    pred = model.predict(Tx, batch_size=len(Tx))[:,0]

    # TTA

    idx = np.where(pred > thr)[0]

    Tx2 = np.zeros((len(idx),10))

    Tx2[:,5:] = Tx[idx,:5]

    Tx2[:,:5] = Tx[idx,5:]    

    pred1 = model.predict(Tx2, batch_size=len(idx))[:,0]

    pred[idx] = (pred[idx] + pred1)/2

    return pred
# select one hit to construct a track

for hit in range(3):

    path = get_path(hit, np.ones(len(truth)), 0.95)

    gt = np.where(truth.particle_id==truth.particle_id[hit])[0]

    print('hit_id = ', hit+1)

    print('reconstruct :', path)

    print('ground truth:', gt.tolist())
# Predict all pairs for reconstruct by all hits. (takes 2.5hr but can skip)

skip_predict = True



if skip_predict == False:

    TestX = np.zeros((len(features), 10))

    TestX[:,5:] = features



    # for TTA

    TestX1 = np.zeros((len(features), 10))

    TestX1[:,:5] = features



    preds = []



    for i in tqdm_notebook(range(len(features)-1)):

        TestX[i+1:,:5] = np.tile(features[i], (len(TestX)-i-1, 1))



        pred = model.predict(TestX[i+1:], batch_size=20000)[:,0]                

        idx = np.where(pred>0.2)[0]



        if len(idx) > 0:

            TestX1[idx+i+1,5:] = TestX[idx+i+1,:5]

            pred1 = model.predict(TestX1[idx+i+1], batch_size=20000)[:,0]

            pred[idx] = (pred[idx]+pred1)/2



        idx = np.where(pred>0.5)[0]



        preds.append([idx+i+1, pred[idx]])



        #if i==0: print(preds[-1])



    preds.append([np.array([], dtype='int64'), np.array([], dtype='float32')])



    # rebuild to NxN

    for i in range(len(preds)):

        ii = len(preds)-i-1

        for j in range(len(preds[ii][0])):

            jj = preds[ii][0][j]

            preds[jj][0] = np.insert(preds[jj][0], 0 ,ii)

            preds[jj][1] = np.insert(preds[jj][1], 0 ,preds[ii][1][j])



    #np.save('my_%s.npy'%event, preds)

else:

    print('load predicts')

    preds = np.load('../input/trackml/my_%s.npy'%event)
def get_path2(hit, mask, thr):

    path = [hit]

    a = 0

    while True:

        c = get_predict2(path[-1])

        mask = (c > thr)*mask

        mask[path[-1]] = 0

        

        if 1:

            cand = np.where(c>thr)[0]

            if len(cand)>0:

                mask[cand[np.isin(module_id[cand], module_id[path])]]=0

                

        a = (c + a)*mask

        if a.max() < thr*len(path):

            break

        path.append(a.argmax())

    return path



def get_predict2(p):

    c = np.zeros(len(preds))

    c[preds[p, 0]] = preds[p, 1]          

    return c
# reconstruct by all hits. (takes 0.6hr but can skip)

skip_reconstruct = True



if skip_reconstruct == False:

    tracks_all = []

    thr = 0.85

    x4 = True

    for hit in tqdm_notebook(range(len(preds))):

        m = np.ones(len(truth))

        path  = get_path2(hit, m, thr)

        if x4 and len(path) > 1:

            m[path[1]]=0

            path2  = get_path2(hit, m, thr)

            if len(path) < len(path2):

                path = path2

                m[path[1]]=0

                path2  = get_path2(hit, m, thr)

                if len(path) < len(path2):

                    path = path2

            elif len(path2) > 1:

                m[path[1]]=1

                m[path2[1]]=0

                path2  = get_path2(hit, m, thr)

                if len(path) < len(path2):

                    path = path2

        tracks_all.append(path)

    #np.save('my_tracks_all', tracks_all)

else:

    print('load tracks')

    tracks_all = np.load('../input/trackml/my_tracks_all.npy')
def get_track_score(tracks_all, n=4):

    scores = np.zeros(len(tracks_all))

    for i, path in enumerate(tracks_all):

        count = len(path)



        if count > 1:

            tp=0

            fp=0

            for p in path:

                tp = tp + np.sum(np.isin(tracks_all[p], path, assume_unique=True))

                fp = fp + np.sum(np.isin(tracks_all[p], path, assume_unique=True, invert=True))

            scores[i] = (tp-fp*n-count)/count/(count-1)

        else:

            scores[i] = -np.inf

    return scores



def score_event_fast(truth, submission):

    truth = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')

    df = truth.groupby(['track_id', 'particle_id']).hit_id.count().to_frame('count_both').reset_index()

    truth = truth.merge(df, how='left', on=['track_id', 'particle_id'])

    

    df1 = df.groupby(['particle_id']).count_both.sum().to_frame('count_particle').reset_index()

    truth = truth.merge(df1, how='left', on='particle_id')

    df1 = df.groupby(['track_id']).count_both.sum().to_frame('count_track').reset_index()

    truth = truth.merge(df1, how='left', on='track_id')

    truth.count_both *= 2

    score = truth[(truth.count_both > truth.count_particle) & (truth.count_both > truth.count_track)].weight.sum()

    particles = truth[(truth.count_both > truth.count_particle) & (truth.count_both > truth.count_track)].particle_id.unique()



    return score, truth[truth.particle_id.isin(particles)].weight.sum(), 1-truth[truth.track_id>0].weight.sum()



def evaluate_tracks(tracks, truth):

    submission = pd.DataFrame({'hit_id': truth.hit_id, 'track_id': tracks})

    score = score_event_fast(truth, submission)[0]

    track_id = tracks.max()

    print('%.4f %2.2f %4d %5d %.4f %.4f'%(score, np.sum(tracks>0)/track_id, track_id, np.sum(tracks==0), 1-score-np.sum(truth.weight.values[tracks==0]), np.sum(truth.weight.values[tracks==0])))



def extend_path(path, mask, thr, last = False):

    a = 0

    for p in path[:-1]:

        c = get_predict2(p)

        if last == False:

            mask = (c > thr)*mask

        mask[p] = 0

        cand = np.where(c>thr)[0]

        mask[cand[np.isin(module_id[cand], module_id[path])]]=0

        a = (c + a)*mask



    while True:

        c = get_predict2(path[-1])

        if last == False:

            mask = (c > thr)*mask

        mask[path[-1]] = 0

        cand = np.where(c>thr)[0]

        mask[cand[np.isin(module_id[cand], module_id[path])]]=0

        a = (c + a)*mask

            

        if a.max() < thr*len(path):

            break



        path.append(a.argmax())

        if last: break

    

    return path
# calculate track's confidence (about 2 mins)

scores = get_track_score(tracks_all, 8)
# merge tracks by confidence and get score

idx = np.argsort(scores)[::-1]

tracks = np.zeros(len(hits))

track_id = 0



for hit in idx:



    path = np.array(tracks_all[hit])

    path = path[np.where(tracks[path]==0)[0]]



    if len(path)>3:

        track_id = track_id + 1  

        tracks[path] = track_id



evaluate_tracks(tracks, truth)
# multistage

idx = np.argsort(scores)[::-1]

tracks = np.zeros(len(hits))

track_id = 0



for hit in idx:

    path = np.array(tracks_all[hit])

    path = path[np.where(tracks[path]==0)[0]]



    if len(path)>6:

        track_id = track_id + 1  

        tracks[path] = track_id



evaluate_tracks(tracks, truth)



for track_id in range(1, int(tracks.max())+1):

    path = np.where(tracks == track_id)[0]

    path = extend_path(path.tolist(), 1*(tracks==0), 0.6)

    tracks[path] = track_id

        

evaluate_tracks(tracks, truth)

        

for hit in idx:

    path = np.array(tracks_all[hit])

    path = path[np.where(tracks[path]==0)[0]]



    if len(path)>3:

        path = extend_path(path.tolist(), 1*(tracks==0), 0.6)

        track_id = track_id + 1  

        tracks[path] = track_id

        

evaluate_tracks(tracks, truth)



for track_id in range(1, int(tracks.max())+1):

    path = np.where(tracks == track_id)[0]

    path = extend_path(path.tolist(), 1*(tracks==0), 0.5)

    tracks[path] = track_id

        

evaluate_tracks(tracks, truth)



for hit in idx:

    path = np.array(tracks_all[hit])

    path = path[np.where(tracks[path]==0)[0]]



    if len(path)>1:

        path = extend_path(path.tolist(), 1*(tracks==0), 0.5)

    if len(path)>2:

        track_id = track_id + 1

        tracks[path] = track_id

        

evaluate_tracks(tracks, truth)



for track_id in range(1, int(tracks.max())+1):

    path = np.where(tracks== track_id)[0]

    if len(path)%2 == 0:

        path = extend_path(path.tolist(), 1*(tracks==0), 0.5, True)

        tracks[path] = track_id

        

evaluate_tracks(tracks, truth)