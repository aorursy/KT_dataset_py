from multiprocessing import Pool, cpu_count
from IPython.display import YouTubeVideo #YouTubeVideo('-0OWhcdBt0k', 7)
from sklearn import ensemble, metrics, preprocessing
import tensorflow as tf
import pandas as pd
import numpy as np
import glob

sub = pd.read_csv('../input/sample_submission.csv')
lbl = {k:{'label':v, 'count':0} for k,v in pd.read_csv('../input/label_names_2018.csv').values}
train_videos = glob.glob('../input/video/train*')
test_videos = glob.glob('../input/video/test*')
val_videos = glob.glob('../input/video/val*')
frames = glob.glob('../input/frame/*')
print(len(train_videos), len(test_videos), len(val_videos),len(frames))
models = []
train = []
loops_ = 0
for tf_vids in train_videos:
    #f = tf_vids.split('/')[-1].split('.')[0]
    if loops_ % 10 == 0:
        print(loops_)
    for tf_zip in tf.python_io.tf_record_iterator(tf_vids):
        video = tf.train.Example.FromString(tf_zip)
        vid_id = video.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
        mean_rgb = video.features.feature['mean_rgb'].float_list.value
        mean_audio = video.features.feature['mean_audio'].float_list.value
        l = ','.join(map(str, video.features.feature['labels'].int64_list.value))
        train.append([vid_id, l]+ list(mean_rgb) + list(mean_audio))
    if loops_ > 2:
        break
    loops_ += 1

col = ['VideoId','label'] + ['mean_rgb'+str(i) for i in range(1024)] + ['mean_audio'+str(i) for i in range(128)]
train = pd.DataFrame(train, columns=col)
col = [c for c in col if c not in ['VideoId','label']]
y = train['label'].str.get_dummies(sep=',')
model =  ensemble.ExtraTreesClassifier(n_jobs=-1, n_estimators=10, max_depth=20, random_state=99)
sc = preprocessing.StandardScaler()
model.fit(sc.fit_transform(train[col]), y)
models.append([model, y.columns])
val = []
for tf_vids in val_videos:
    for tf_zip in tf.python_io.tf_record_iterator(tf_vids):
        video = tf.train.Example.FromString(tf_zip)
        vid_id = video.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
        mean_rgb = video.features.feature['mean_rgb'].float_list.value
        mean_audio = video.features.feature['mean_audio'].float_list.value
        l = ','.join(map(str, video.features.feature['labels'].int64_list.value))
        val.append([vid_id, l]+ list(mean_rgb) + list(mean_audio))

    col = ['VideoId','label'] + ['mean_rgb'+str(i) for i in range(1024)] + ['mean_audio'+str(i) for i in range(128)]
    val = pd.DataFrame(val, columns=col)
    col = [c for c in col if c not in ['VideoId','label']]
    y = val['label'].str.get_dummies(sep=',')
    for c in models[0][1]:
        if c not in y.columns:
            y[c] = 0
    ycol = [c for c in y.columns if c in models[0][1]]
    y = y[ycol]
    #print(len(y), len(results))
    results = models[0][0].predict_proba(sc.transform(val[col]))
    results = np.array(results).T[1]
    results = pd.DataFrame(results, columns=models[0][1]) 
    print(metrics.average_precision_score(y, results, average='micro'))
    break
def multi_tf_zip(tf_zip):
    video = tf.train.Example.FromString(tf_zip)
    vid_id = video.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
    mean_rgb = video.features.feature['mean_rgb'].float_list.value
    mean_audio = video.features.feature['mean_audio'].float_list.value
    r = [vid_id]+ list(mean_rgb) + list(mean_audio)
    return r

tests = []
loops_ = 0
for tf_vids in test_videos:
    if loops_ % 100 == 0:
        print (loops_)
    test = []
    p = Pool(cpu_count())
    for tf_zip in tf.python_io.tf_record_iterator(tf_vids):
        test.append(tf_zip)
    test = p.map(multi_tf_zip, test)
    p.close(); p.join()
    
    col = ['VideoId'] + ['mean_rgb'+str(i) for i in range(1024)] + ['mean_audio'+str(i) for i in range(128)]
    test = pd.DataFrame(test, columns=col)
    col = [c for c in col if c not in ['VideoId']]
    results = models[0][0].predict_proba(sc.transform(test[col]))
    results = np.array(results).T[1]
    results = pd.DataFrame(results, columns=models[0][1])
    results_ = []
    for i in range(len(results)):
        r = results.iloc[[i]].T.reset_index().sort_values(by=[i], ascending=False)
        r = r[r[i]>0.0][:20]
        results_.append(' '.join([' '.join(map(str, [k, round(v,3)])) for k, v in r.values]))
    test['LabelConfidencePairs'] = results_
    tests.append(test[['VideoId', 'LabelConfidencePairs']])
    loops_ += 1

for tf_vids in train_videos:
    for tf_zip in tf.python_io.tf_record_iterator(tf_vids):
        video = tf.train.Example.FromString(tf_zip)
        for k in list(video.features.feature['labels'].int64_list.value):
            if k in lbl:
                lbl[k]['count']+=1
df = pd.DataFrame.from_dict(lbl, orient='index').sort_values(['count'], ascending=[False])
df['id'] = df.index
df['count'] = df['count'] / sum(df['count'].values)
sub.LabelConfidencePairs = ' '.join([str(int(x))+' '+str(round(y,2)) for x,y in df[['id','count']].values[:20]])
sub.to_csv('abimannan.csv', index=False)
sub.head()
