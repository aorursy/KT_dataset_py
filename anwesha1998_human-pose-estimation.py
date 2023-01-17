!pip install -qq git+https://www.github.com/ildoonet/tf-pose-estimation
!pip install -qq pycocotools
%load_ext autoreload
%autoreload 2
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

def load_graph1(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_tensor_from_image_file1(file_name, input_height=299, input_width=299,input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result

def load_labels1(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def classify_scene(image_file):
    
# =============================================================================
#   Note : Provide your own absolute file path for the following
#   You can choose the retrained graph of either as v1.0 or v2.0 
#   Both models are retrained inception models (on my procured dataset)
#   v1.0 was trained for 500 epocs on a preliminary dataset of poses.
#   v2.0 was trained for 4000 epocs on a dataset containing the previous dataset
#   and more.
# =============================================================================
  # Change the path to your convenience
    #file_path = os.path.abspath(os.path.dirname(__file__))
    path = '../input/labels/'
    model_file = path+'output_graph_20000.pb'
    label_file = path+'output_labels.txt'
    input_height = 299
    input_width = 299
    input_mean = 128
    input_std = 128
    input_layer = "Mul"
    output_layer = "final_result"

    graph = load_graph1(model_file)
    t = read_tensor_from_image_file1(image_file,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
        end=time.time()
    results = np.squeeze(results)
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels1(label_file)

    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
    template = "{} (score={:0.5f})"
    for i in top_k:
        print(template.format(labels[i], results[i]))


    return template.format(labels[top_k[0]], results[top_k[0]])


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_tensor_from_image_file(image_file, input_height=299, input_width=299,
				input_mean=0, input_std=255):
    
    float_caster = tf.cast(image_file, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def classify_pose(image_file):
    
# =============================================================================
#   Note : Provide your own absolute file path for the following
#   You can choose the retrained graph of either as v1.0 or v2.0 
#   Both models are retrained inception models (on my procured dataset)
#   v1.0 was trained for 500 epocs on a preliminary dataset of poses.
#   v2.0 was trained for 4000 epocs on a dataset containing the previous dataset
#   and more.
# =============================================================================
  # Change the path to your convenience
    #file_path = os.path.abspath(os.path.dirname(__file__))
    path =  '../input/labels/'#os.path.join(file_path, '../input/labels/')
    model_file = path+'retrained_graph.pb'
    label_file = path+'retrained_labels.txt'
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(image_file,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
        end=time.time()
    results = np.squeeze(results)

    labels = load_labels(label_file)

    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
    template = "{} (score={:0.5f})"
    label = ''
    if results[0] > results[1]:
        label = labels[0]
        result = results[0]
    else:
        label = labels[1]
        result = results[1]

    return template.format(label, result)

%matplotlib inline
import tf_pose
import cv2
from glob import glob
from tqdm import tqdm_notebook
from PIL import Image
import numpy as np
import os

def video_gen(in_path):
    c_cap = cv2.VideoCapture(in_path)
    while c_cap.isOpened():
        ret, frame = c_cap.read()
        if not ret:
            break
        yield c_cap.get(cv2.CAP_PROP_POS_MSEC), frame[:, :, ::-1]
    c_cap.release()

video_paths = glob('../input/videos-pose/*.mp4')
c_video = video_gen(video_paths[0])
for _ in range(200):
    c_ts, c_frame = next(c_video)
plt.imshow(c_frame)
from PIL import Image
#import numpy as np
#img_w, img_h = 200, 200
#data = np.zeros((img_h, img_w, 3), dtype=np.uint8)
#data[100, 100] = [255, 0, 0]
#img = Image.fromarray(c_frame, 'RGB')
#img.save('test.png')
#img.show()

#classify_scene('test.png')
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
tfpe = tf_pose.get_estimator()

humans_vid = tfpe.inference(npimg=c_frame, upsample_size=4.0)
print(humans_vid)
new_vid_img = TfPoseEstimator.draw_humans(c_frame[:, :, ::-1], humans_vid, imgcopy=False)
fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
ax1.imshow(new_vid_img[:, :, ::-1])
body_to_dict = lambda c_fig: {'bp_{}_{}'.format(k, vec_name): vec_val 
                              for k, part_vec in c_fig.body_parts.items() 
                              for vec_name, vec_val in zip(['x', 'y', 'score'],(part_vec.x, 1-part_vec.y, part_vec.score))}
c_fig = humans_vid[0]
body_to_dict(c_fig)
MAX_FRAMES = 200
body_pose_list = []
for vid_path in tqdm_notebook(video_paths, desc='Files'):
    c_video = video_gen(vid_path)
    c_ts, c_frame = next(c_video)
    out_path = '{}_out.avi'.format(os.path.split(vid_path)[0])
    
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('M','J','P','G'),10,(c_frame.shape[1], c_frame.shape[0]))
    
    for (c_ts, c_frame), _ in zip(c_video, tqdm_notebook(range(MAX_FRAMES), desc='Frames')):
        bgr_frame = c_frame[:,:,::-1]
        humans = tfpe.inference(npimg=bgr_frame, upsample_size=4.0)
        for c_body in humans:
            body_pose_list += [dict(video=out_path, time=c_ts, **body_to_dict(c_body))]
        new_image = TfPoseEstimator.draw_humans(bgr_frame, humans, imgcopy=False)
        pose_class = classify_pose(new_image)
    
        cv2.putText(new_image,"Current predicted pose is : %s" %(pose_class),(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
        #cv2.putText(new_image,"Predicted Scene: %s" %(scene_class),(10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        out.write(new_image)
    out.release()

import pandas as pd
body_pose_df = pd.DataFrame(body_pose_list)
body_pose_df#.describe()
fig, m_axs = plt.subplots(1, 2, figsize=(15, 5))
for c_ax, (c_name, c_rows) in zip(m_axs, body_pose_df.groupby('video')):
    for i in range(17):
        c_ax.plot(c_rows['time'], c_rows['bp_{}_y'.format(i)], label='x {}'.format(i))
    c_ax.legend()
    c_ax.set_title(c_name)
#body_pose_df[:400]
fig, m_axs = plt.subplots(1, 2, figsize=(15, 5))
for c_ax, (c_name, n_rows) in zip(m_axs, body_pose_df.groupby('video')):
    for i in range(17):
        c_rows = n_rows.query('bp_{}_score>0.6'.format(i)) # only keep confident results
        c_ax.plot(c_rows['bp_{}_x'.format(i)], c_rows['bp_{}_y'.format(i)], label='BP {}'.format(i))
    c_ax.legend()
    c_ax.set_title(c_name)
body_pose_df.to_csv('body_pose.csv', index=False)
def img_pose(frame):
    humans_img = tfpe.inference(npimg=frame, upsample_size=4.0)
    new_image = TfPoseEstimator.draw_humans(imgrgb[:, :, ::-1], humans_img, imgcopy=False)
    return new_image
img_paths = glob('../input/images/*.jpg')
for path in img_paths:
    frame=cv2.imread(path,cv2.COLOR_BGR2RGB)
    imgrgb = frame[:,:,::-1]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    #plt.imshow(imgrgb)
    ax.imshow(imgrgb)
    
    new_image = img_pose(frame)
    fig, ax2 = plt.subplots(1, 1, figsize=(8, 8))
    ax2.imshow(new_image[:, :, ::-1])
humans_img = tfpe.inference(npimg=frame, upsample_size=4.0)
print(humans_img)
def webcam():
    cap = cv2.VideoCapture(0)

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, image = cap.read()
        if not ret:
            break
        yield cap.get(cv2.CAP_PROP_POS_MSEC), frame[:, :, ::-1]
    cap.release()

        
video_capture = cv2.VideoCapture(0)












