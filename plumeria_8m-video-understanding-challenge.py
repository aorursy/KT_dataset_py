import numpy as np 
import pandas as pd 
import os
import seaborn as sns
import networkx as nx
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
tf.Session()
from IPython.display import YouTubeVideo
print(os.listdir("../input/"))
print(os.listdir("../input/frame-sample//frame"))
print(os.listdir("../input/video-sample//video"))
video_record00 = "../input/video-sample/video/train00.tfrecord"
frame_record00 = "../input/frame-sample//frame/train00.tfrecord"
label_names=pd.read_csv("../input/label_names_2018.csv")
vocabulary=pd.read_csv("../input/vocabulary.csv")
sample_submission=pd.read_csv("../input/sample_submission.csv")
print(label_names.head(5))
print('The Data in label_names is : {}'.format(label_names.shape))
print(vocabulary.head(5))
print('The data dictionary in vocabulary is : {}'.format(vocabulary.shape))
# Sample Submission File
print(sample_submission.head(5))
#Tensorflow version
print(tf.__version__)
vid_ids = []
labels = []
mean_rgb = []
mean_audio = []

for train00 in tf.python_io.tf_record_iterator(video_record00):
    train_f= tf.train.Example.FromString(train00)
    vid_ids.append(train_f.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
    labels.append(train_f.features.feature['labels'].int64_list.value)
    mean_rgb.append(train_f.features.feature['mean_rgb'].float_list.value)
    mean_audio.append(train_f.features.feature['mean_audio'].float_list.value)
print('Number of videos in train00.tfrecord file  is : ',len(mean_rgb))
#Let us randomly select a video id 18 
print('Select  a youtube video id:',vid_ids[18])
# The list of 20 features of the video d 18 
print('First 20 features of a  selected youtube video is  (',vid_ids[18],'):')
print(mean_rgb[18][:20])
vid_ids = []
labels = []
mean_rgb = []
mean_audio = []

for train00 in tf.python_io.tf_record_iterator(video_record00):
    train_f= tf.train.Example.FromString(train00)
    vid_ids.append(train_f.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
    labels.append(train_f.features.feature['labels'].int64_list.value)
    mean_rgb.append(train_f.features.feature['mean_rgb'].float_list.value)
    mean_audio.append(train_f.features.feature['mean_audio'].float_list.value)
print('Number of videos in train00.tfrecord file  is : ',len(mean_rgb))
#Let us randomly select a video id 18 
print('Select  a youtube video id:',vid_ids[18])
# The list of 20 features of the video d 18 
print('First 20 features of a  selected youtube video is  (',vid_ids[18],'):')
print(mean_rgb[18][:20])
YouTubeVideo('mLEJIW9HeIw')
feat_rgb = []
feat_audio = []
rgb_frame = []
audio_frame = []
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='tensorflow')
for train00 in tf.python_io.tf_record_iterator(frame_record00):        
    train_f = tf.train.SequenceExample.FromString(train00)
    num_frames = len( train_f .feature_lists.feature_list['audio'].feature)
    sess = tf.InteractiveSession()
    # iterate through frames
    for i in range(num_frames):
        rgb_frame.append(tf.cast(tf.decode_raw( train_f .feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0],tf.uint8) ,tf.float32).eval())
        audio_frame.append(tf.cast(tf.decode_raw( train_f .feature_lists.feature_list['audio'].feature[i].bytes_list.value[0],tf.uint8),tf.float32).eval())
    sess.close()
    feat_rgb.append(rgb_frame)
    feat_audio.append(audio_frame)
    break
print('The first video has %d frames' %len(feat_rgb[0]))
sns.lmplot(x='Index', y='TrainVideoCount', data=vocabulary , size=15)
with open('../input/vocabulary.csv', 'r') as f:
  vocabularylist = list(csv.reader(f))
T1=[]
for l in vocabularylist:
    if l[5] != 'NaN' and l[6] !='NaN' and l[5] != '' and l[6] !='' and l[5] !=  l[6] :
        c1 = l[5]
        c2 = l[6]
        tuple = (c1, c2)
    if l[5] != 'NaN' and l[7] !='NaN' and l[5] != '' and l[7] !='' and l[5] !=  l[7] :
        c1 = l[5]
        c2 = l[7]
        tuple = (c1, c2)
    if l[6] != 'NaN' and l[7] !='NaN' and l[6] != '' and l[7] !='' and l[7] !=  l[6] :
        c1 = l[6]
        c2 = l[7]
        tuple = (c1, c2)
    T1.append(tuple)
edges = {k: T1.count(k) for k in set(T1)}
edges
B = nx.DiGraph()
nodecolor=[]
for ed, weight in edges.items():
    if ed[0]!='Vertical2' and ed[0]!='Vertical3' and  ed[1]!='Vertical2' and ed[1]!='Vertical3':
        B.add_edge(ed[0], ed[1], weight=weight)
for k in B.nodes:
    if (k == "Beauty & Fitness"):
        nodecolor.append('blue')
    elif (k == "News"):
        nodecolor.append('Magenta')
    elif (k == "Food & Drink"):
        nodecolor.append('crimson')
    elif (k == "Health"):
        nodecolor.append('green')
    elif (k == "Science"):
        nodecolor.append('yellow')
    elif (k == "Business & Industrial"):
        nodecolor.append('cyan')
    elif (k == "Home & Garden"):
        nodecolor.append('darkorange')
    elif (k == "Travel"):
        nodecolor.append('slategrey')
    elif (k == "Arts & Entertainment"):
        nodecolor.append('red')
    elif (k == "Games"):
        nodecolor.append('grey')
    elif (k == "People & Society"):
        nodecolor.append('lightcoral')
    elif (k == "Shopping"):
        nodecolor.append('maroon')
    elif (k =="Computers & Electronics"):
        nodecolor.append('orangered')
    elif (k == "Hobbies & Leisure"):
        nodecolor.append('saddlebrown')
    elif (k == "Sports"):
        nodecolor.append('lawngreen')
    elif (k == "Real Estate"):
        nodecolor.append('deeppink')
    elif (k == "Finance"):
        nodecolor.append('springgreen')
    elif (k == "Reference"):
        nodecolor.append('royalblue')
    elif (k == "Autos & Vehicles"):
        nodecolor.append('turquoise')
    elif (k == "Internet & Telecom"):
        nodecolor.append('lime')
    elif (k == "Law & Government"):
        nodecolor.append('palegreen')
    elif (k == "Jobs & Education"):
        nodecolor.append('navy')
    elif (k == "Pets & Animals"):
        nodecolor.append('lightpink')
    elif (k == "Books & Literature"):
        nodecolor.append('lightpink')
    
                         
plt.figure(figsize = (15,15))
nx.draw(B, pos=nx.circular_layout(B), node_size=1500, with_labels=True, node_color=nodecolor)
nx.draw_networkx_edge_labels(B, pos=nx.circular_layout(B), edge_labels=nx.get_edge_attributes(B, 'weight'))
plt.title('Weighted graph representing the relationship between the categories', size=20)
plt.show()
