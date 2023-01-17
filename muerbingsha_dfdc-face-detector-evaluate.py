import numpy as np
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import time
import math
from tqdm.notebook import tqdm
import glob
IMG_SIZE = 224
MARGIN = 10
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# MTCNN
!pip install ../input/mtcnn-package/mtcnn-0.1.0-py3-none-any.whl -q
from mtcnn import MTCNN as mMTCNN
mtcnn_detector = mMTCNN()
# facenet_pytorch
from facenet_pytorch import MTCNN as fMTCNN
facenet_detector = fMTCNN(image_size=150, margin=0, keep_all=True, factor=0.5, post_process=False, device=device).eval()
# dlib
!pip install '/kaggle/input/dlibpkg/dlib-19.19.0'
import dlib
dlib_hog_detector = dlib.get_frontal_face_detector()
# insightface
!pip install insightface -q
import insightface
insight_detector = insightface.app.FaceAnalysis()
insight_detector.prepare(ctx_id = -1, nms=0.4)
# mobilenet
import tensorflow as tf
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile('../input/mobilenet-face/frozen_inference_graph_face.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.compat.v1.Session(graph=detection_graph, config=config)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')    
    scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
def mobile_detector(image):
    global boxes,scores,num_detections
    (h, w)=image.shape[:-1]
    imgs=np.array([image])
    (boxes, scores) = sess.run(
        [boxes_tensor, scores_tensor],
        feed_dict={image_tensor: imgs})
    max_=np.where(scores==scores.max())[0][0]
    box=boxes[0][max_]
    ymin, xmin, ymax, xmax = box
    (left, right, top, bottom) = (xmin * w, xmax * w, 
                                  ymin * h, ymax * h)
    left, right, top, bottom = int(left), int(right), int(top), int(bottom)
    return (left, right, top, bottom)
# blazeface
def prepare_blaze():
    blazeface = BlazeFace()
    blazeface.load_weights("/kaggle/input/blazeface-pytorch/blazeface.pth")
    blazeface.load_anchors("/kaggle/input/blazeface-pytorch/anchors.npy")

    # Optionally change the thresholds:
    blazeface.min_score_thresh = 0.95
    blazeface.min_suppression_threshold = 0.3
    return blazeface
blazeface = prepare_blaze()

def get_blaze_boxes(detections):
    result = []
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    # no detect
    img_shape = (128, 128)
    if len(detections) != 0:
        for i in range(detections.shape[0]):
            ymin = detections[i, 0] * img_shape[0]
            xmin = detections[i, 1] * img_shape[1]
            ymax = detections[i, 2] * img_shape[0]
            xmax = detections[i, 3] * img_shape[1]
            result.append((xmin, ymin, xmax, ymax))
    return result


def scale_boxes(boxes, scale_w, scale_h):
    sb = []
    for b in boxes:
        sb.append((b[0] * scale_w, b[1] * scale_h, b[2] * scale_w, b[3] * scale_h))
    return sb


def blaze_detector(img):
    img = cv2.resize(img, (128, 128))
    output = blazeface.predict_on_image(img)
    bbox = scale_boxes(get_blaze_boxes(output), 380/128, 380/128)
    return bbox
    
df = pd.read_csv('../input/deepfake-detection-challenge/sample_submission.csv')
video_names = df['filename']
video_paths = ['../input/deepfake-detection-challenge/test_videos/' + n for n in video_names]
video_paths[:10]
sample_video = video_paths[0]
%%time
before = time.time()

def read_video1(video_path):
    v_cap = cv2.VideoCapture(sample_video)
    v_int = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = []
    for i in range(v_int):
        ret, frame = v_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame)
    return np.array(all_frames)
    
result = read_video1(sample_video)

after = time.time()
print('result {} with time {} seconds'.format(result.shape, round(after-before, 1)))
%%time
before = time.time()

def read_video2(video_path):
    v_cap = cv2.VideoCapture(sample_video)
    v_int = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = []
    for i in range(v_int):
        v_cap.grab()
        ret, frame = v_cap.retrieve()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame)
    return np.array(all_frames)
    
result = read_video2(sample_video)

after = time.time()
print('result {} with time {} seconds'.format(result.shape, round(after-before, 1)))
!pip install ../input/imageio-ffmpeg/imageio_ffmpeg-0.3.0-py3-none-manylinux2010_x86_64.whl
import imageio
%%time
before = time.time()

def read_video3(video_path):
    reader = imageio.get_reader(video_path, 'ffmpeg')
    v_int = reader.count_frames()
    meta = reader.get_meta_data()
    all_frames = []
    for i in range(v_int):
        img = reader.get_data(i)
        all_frames.append(img)
    return np.array(all_frames)
    
result = read_video3(sample_video)

after = time.time()
print('result {} with time {} seconds'.format(result.shape, round(after-before, 1)))
frames_per_video = 72
def read_frames1(frames_per_video=60):
    v_cap = cv2.VideoCapture(sample_video)
    v_int = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = []
    sample_idx = np.linspace(0, v_int-1, frames_per_video, endpoint=True, dtype=np.int)
    for i in range(v_int):
        # speed reading without decode unwanted frame
        ret = v_cap.grab()
        
        if ret is None: 
            print('The {} cannot be read'.format(i))
            continue
            
#         # the frame we want
        if i in sample_idx:
            ret, frame = v_cap.retrieve()
            if ret is None or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame)
            
    return all_frames
%%time
before = time.time()

result = read_frames1(frames_per_video)

after = time.time()
print('result {} with time {} seconds'.format(np.array(result).shape, round(after-before, 1)))
def read_frames2(frames_per_video=60):
    v_cap = cv2.VideoCapture(sample_video)
    v_int = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = []
    sample_idx = np.linspace(0, v_int-1, frames_per_video, endpoint=True, dtype=np.int)
    for i in range(v_int):
        ret, frame = v_cap.read()
        if ret is None or frame is None:
            continue
        
        if i in sample_idx:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame)
        
    return all_frames
%%time
before = time.time()

result = read_frames2(frames_per_video)

after = time.time()
print('result {} with time {} seconds'.format(np.array(result).shape, round(after-before, 1)))
def read_frames3(frames_per_video):
    reader = imageio.get_reader(sample_video, 'ffmpeg')
    v_int = reader.count_frames()
    all_frames = []
    sample_idx = np.linspace(0, v_int-1, frames_per_video, endpoint=True, dtype=np.int)
    for i in sample_idx:
        img = reader.get_data(i)
        all_frames.append(img)
    return all_frames
%%time
before = time.time()

result = read_frames3(frames_per_video)

after = time.time()
print('result {} with time {} seconds'.format(np.array(result).shape, round(after-before, 1)))
def read_frames4(frames_per_video=72):
    v_cap = cv2.VideoCapture(sample_video)
    v_int = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = []
    sample_idx = np.linspace(0, v_int-1, frames_per_video, endpoint=True, dtype=np.int)
    for i in sample_idx:
        v_cap.set(cv2.CAP_PROP_POS_FRAMES, i) 
        ret, frame = v_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame)
    return all_frames
%%time
before = time.time()

result = read_frames4(frames_per_video)

after = time.time()
print('result {} with time {} seconds'.format(np.array(result).shape, round(after-before, 1)))
# unify show faces
show_faces = 9
frames = read_video2(sample_video)
cell = round(math.sqrt(9))
random_idx = np.random.choice(len(frames), show_faces)
from mtcnn import MTCNN
detector = MTCNN()
def detect_face(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    final = []
    detected_faces_raw = detector.detect_faces(img)
    if detected_faces_raw==[]:
        #print('no faces found')
        return []
    confidences=[]
    for n in detected_faces_raw:
        x,y,w,h=n['box']
        final.append([x,y,w,h])
        confidences.append(n['confidence'])
    if max(confidences)<0.9:
        return []
    max_conf_coord=final[confidences.index(max(confidences))]
    #return final
    return max_conf_coord

def crop(img,x,y,w,h, margin, img_shape):
    x-=margin
    y-=margin
    w+=margin*2
    h+=margin*2
    if x<0:
        x=0
    if y<=0:
        y=0
    return cv2.cvtColor(cv2.resize(img[y:y+h,x:x+w],img_shape),cv2.COLOR_BGR2RGB)
def detect_faces_mtcnn(video_path):
    total_faces = []
    
    # full frames
    frames = read_video2(video_path)
    
    for frame in tqdm(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        bounding_box = detect_face(frame)
        if bounding_box == []:
            continue
        else:
            x,y,w,h = bounding_box
            face = crop(frame, x, y, w, h, MARGIN, (IMG_SIZE, IMG_SIZE))
            total_faces.append(face)
            
    return np.array(total_faces)

%%time
before = time.time()

faces = detect_faces_mtcnn(sample_video)

after = time.time()
print('result {} with time {} seconds'.format(faces.shape, round(after-before, 1)))
fig, ax = plt.subplots(cell, cell, figsize=(8, 8))
for i in range(cell):
    for j in range(cell):
        ax[i][j].imshow(faces[random_idx[i*cell+j]])
!pip install ../input/facenet-pytorch-vggface2/facenet_pytorch-2.0.1-py3-none-any.whl
from facenet_pytorch import MTCNN
mtcnn = MTCNN(image_size=IMG_SIZE, margin=MARGIN)
def detect_faces_facenet(video_path):
    total_faces = []
    
    frames = read_video2(video_path)
    
    for frame in tqdm(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = net_facenet(Image.fromarray(frame))
        if faces is None:
            continue
        else:
            total_faces.append(faces.numpy()[0])   
        
    return np.array(total_faces)
%%time
before = time.time()

faces = detect_faces_facenet(sample_video)

after = time.time()
print('result {} with time {} seconds'.format(faces.shape, round(after-before, 1)))
fig, ax = plt.subplots(cell, cell, figsize=(8, 8))
for i in range(cell):
    for j in range(cell):
        ax[i][j].imshow(faces[random_idx[i*cell+j]])
import sys
sys.path.insert(0, "/kaggle/input/blazeface-pytorch")
from blazeface import BlazeFace

net = BlazeFace()
net.load_weights("/kaggle/input/blazeface-pytorch/blazeface.pth")
net.load_anchors("/kaggle/input/blazeface-pytorch/anchors.npy")

# Optionally change the thresholds:
net.min_score_thresh = 0.75
net.min_suppression_threshold = 0.3
def plot_detections(img, detections, with_keypoints=True):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.grid(False)
    ax.imshow(img)
    
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    print("Found %d faces" % detections.shape[0])
        
    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor="r", facecolor="none", 
                                 alpha=detections[i, 16])
        ax.add_patch(rect)

        if with_keypoints:
            for k in range(6):
                kp_x = detections[i, 4 + k*2    ] * img.shape[1]
                kp_y = detections[i, 4 + k*2 + 1] * img.shape[0]
                circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1, 
                                        edgecolor="lightskyblue", facecolor="none", 
                                        alpha=detections[i, 16])
                ax.add_patch(circle)
        
    plt.show()
def tile_frames(frames, target_size):
    num_frames, H, W, _ = frames.shape

    split_size = min(H, W)
    x_step = (W - split_size) // 2
    y_step = (H - split_size) // 2
    num_v = 1
    num_h = 3 if W > H else 1

    splits = np.zeros((num_frames * num_v * num_h, target_size[1], target_size[0], 3), dtype=np.uint8)

    i = 0
    for f in range(num_frames):
        y = 0
        for v in range(num_v):
            x = 0
            for h in range(num_h):
                crop = frames[f, y:y+split_size, x:x+split_size, :]
                splits[i] = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
                x += x_step
                i += 1
            y += y_step

    resize_info = [split_size / target_size[0], split_size / target_size[1]]
    return splits, resize_info
def detect_faces_full_frame(video_path):
    total_detections = []
    frames = read_video2(video_path)
    for frame in tqdm(frames):
        frame = cv2.resize(frame, net.input_size)
        detections = net.predict_on_image(frame)
        
        if len(detections) != 0:
            total_detections.append(detections)
    
    return total_detections
%%time
before = time.time()

detections = detect_faces_full_frame(sample_video)

after = time.time()
print('result {} with time {} seconds'.format(len(detections), round(after-before, 1)))
for frame in frames:
    frame = cv2.resize(frame, net.input_size)

    detections = net.predict_on_image(frame)
    
    # cannot detect
    if len(detections) == 0:
        continue
    else:
        print('Frame idx {}'.format(i))
        plot_detections(frame, detections)
        break
# show the tile effect
first_frames, first_resize_info = tile_frames(frames[:1], (128, 128))

fig = plt.figure(figsize=(14, 14))
columns = 3
rows = 1
for i in range(1, columns*rows +1):
    img = first_frames[i-1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()
def resize_detections(detections, target_size, resize_info):
    '''get detections from [0, 1] to original frame size (1080, 1080)'''
    
    projected = []
    target_w, target_h = target_size  # (128, 128)
    scale_w, scale_h = resize_info #  (1080/128, 1080/128)

    # each frame
    for i in range(len(detections)):
        detection = detections[i].clone()

        # ymin, xmin, ymax, xmax
        for k in range(2): #0, 1
            detection[:, k*2    ] = (detection[:, k*2    ] * target_h) * scale_h
            detection[:, k*2 + 1] = (detection[:, k*2 + 1] * target_w) * scale_w
         
        # keypoints are x,y
        for k in range(2, 8):
            detection[:, k*2    ] = (detection[:, k*2    ] * target_w) * scale_w
            detection[:, k*2 + 1] = (detection[:, k*2 + 1] * target_h) * scale_h

        projected.append(detection)

    return projected 
# show resize_detection effects

# update
target_size = (128, 128)
resize_info = (1080/128, 1080/128)
target_w, target_h = target_size
scale_w, scale_h = resize_info 

first_originals, _ = tile_frames(frames[:1], (1080, 1080))

first_detections = net.predict_on_batch(first_frames, apply_nms=False)
first_detections = resize_detections(first_detections, target_size, resize_info)
    
    
fig = plt.figure(figsize=(14, 14))
for i, detection in enumerate(first_detections):
    
    # show the img
    ax = fig.add_subplot(1, 3, i+1, aspect='equal')
    ax.imshow(first_originals[i])
    

    # each face
    for detect in detection:
#         ymin = (detect[0] * target_h) * scale_h
#         xmin = (detect[1] * target_w) * scale_w
#         ymax = (detect[2] * target_h) * scale_h
#         xmax = (detect[3] * target_w) * scale_w
        
        ymin = detect[0]
        xmin = detect[1]
        ymax = detect[2] 
        xmax = detect[3] 

        ax.add_patch( patches.Rectangle( (xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor="r", alpha=detect[16])) 
        
plt.show()
del first_originals
del fig
def untile_detections(num_frames, frame_size, detections):
    """With N tiles per frame, there also are N times as many detections.
    This function groups together the detections for a given frame; it is
    the complement to tile_frames().
    detections: (192, ?, 17)
    """
    combined_detections = []

    W, H = frame_size
    split_size = min(H, W)
    x_step = (W - split_size) // 2
    y_step = (H - split_size) // 2
    num_v = 1
    num_h = 3 if W > H else 1

    i = 0
    for f in range(num_frames):
        detections_for_frame = []
        y = 0
        for v in range(num_v):
            x = 0
            for h in range(num_h):
                # Adjust the coordinates based on the split positions.
                detection = detections[i].clone()
                if detection.shape[0] > 0:
                    for k in range(2):
                        detection[:, k*2    ] += y
                        detection[:, k*2 + 1] += x
                    for k in range(2, 8):
                        detection[:, k*2    ] += x
                        detection[:, k*2 + 1] += y

                detections_for_frame.append(detection)
                x += x_step
                i += 1
            y += y_step

        combined_detections.append(torch.cat(detections_for_frame))

    return combined_detections
# show untile detections effect
frame_size = (frames.shape[2], frames.shape[1])
detections_one = untile_detections(1, frame_size, first_detections)
detections_one = net.nms(detections_one)[0] # only one frame

fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
ax.imshow(frames[0])

# how many faces
for i in range(len(detections_one)):
    ymin = detections_one[i][0]
    xmin = detections_one[i][1]
    ymax = detections_one[i][2] 
    xmax = detections_one[i][3] 

    ax.add_patch( patches.Rectangle( (xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor="r", alpha=detect[16])) 
        
plt.show()
def add_margin_to_detections(detections, frame_size, margin=0.2):
    """Expands the face bounding box.

    NOTE: The face detections often do not include the forehead, which
    is why we use twice the margin for ymin.

    Arguments:
        detections: a PyTorch tensor of shape (num_detections, 17)
        frame_size: maximum (width, height)
        margin: a percentage of the bounding box's height

    Returns a PyTorch tensor of shape (num_detections, 17).
    """
    offset = torch.round(margin * (detections[:, 2] - detections[:, 0])) # (ymax - ymin)*margin
    detections = detections.clone()
    detections[:, 0] = torch.clamp(detections[:, 0] - offset*2, min=0)            # ymin
    detections[:, 1] = torch.clamp(detections[:, 1] - offset, min=0)              # xmin
    detections[:, 2] = torch.clamp(detections[:, 2] + offset, max=frame_size[1])  # ymax
    detections[:, 3] = torch.clamp(detections[:, 3] + offset, max=frame_size[0])  # xmax
    return detections
# show add margin effect

detections_margin = add_margin_to_detections(detections_one, frame_size=frame_size)


fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
ax.imshow(frames[0])

# how many faces
for i in range(len(detections_margin)):
    ymin = detections_margin[i][0]
    xmin = detections_margin[i][1]
    ymax = detections_margin[i][2] 
    xmax = detections_margin[i][3] 

    ax.add_patch( patches.Rectangle( (xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor="r", alpha=detect[16])) 
        
plt.show()
def crop_faces(frame, detections):
    """Copies the face region(s) from the given frame into a set
    of new NumPy arrays.

    Arguments:
        frame: a NumPy array of shape (H, W, 3)
        detections: a PyTorch tensor of shape (num_detections, 17)

    Returns a list of NumPy arrays, one for each face crop. If there
    are no faces detected for this frame, returns an empty list.
    """
    faces = []
    for i in range(len(detections)):
        ymin, xmin, ymax, xmax = detections[i, :4].cpu().numpy().astype(np.int)
        face = frame[ymin:ymax, xmin:xmax, :]
        faces.append(face)
    return faces
faces_blaze = crop_faces(frames[0], detections_margin)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
ax.imshow(faces_blaze[0])
print(faces_blaze[0].shape)
def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
face_resized = isotropically_resize_image(faces_blaze[0], 224)
face_square = make_square_image(face_resized)
plt.imshow(face_square)
class BlazeFaceExtractor:
    ''' Convenient class'''
    
    
    
    def __init__(self, weights_path=None, anchors_path=None):
        ''' load face extractor ''' 
        
        from blazeface import BlazeFace

        self.net = BlazeFace()
        
        
        if weights_path is None:
            self.net.load_weights("/kaggle/input/blazeface-pytorch/blazeface.pth")
        else:
            self.net.load_weights(weights_path)
            
        if anchors_path is None:
            self.net.load_anchors("/kaggle/input/blazeface-pytorch/anchors.npy")
        else:
            self.net.load_anchors(anchors_path)
            

        # Optionally change the thresholds:
        self.net.min_score_thresh = 0.75
        self.net.min_suppression_threshold = 0.3

        
        
        
    def extract_faces(self, video_path, video_read_fn, output_size=224):
        '''
        Arguments
            video_path: full path for one video
        '''
        self.video_path = video_path
        self.video_read_fn = video_read_fn
        self.output_size = 224
        
        # all or specific number of full frames for one video
        frames = self.video_read_fn(self.video_path)
        
        # 1 - tile
        tiles_frames, tiles_resize_info = tile_frames(frames, self.net.input_size)
        
        # 2 - predict
        detections = net.predict_on_batch(tiles_frames, apply_nms=False)
        
        # 3 - resize detection
        detections_resized = self.resize_detections(detections, self.net.input_size, tiles_resize_info)
        
        # 4 - untile
        frame_size = (frames.shape[2], frames.shape[1])
        detections_one = self.untile_detections(len(frames), frame_size, detections_resized)
        detections_one = self.net.nms(detections_one) # 
        
        # for each frame
        total_faces = []
        for i, detection_single in enumerate(detections_one):
            # no face
            if len(detection_single) == 0:
                continue
                
                
            # 5 - add margin
            detections_margin = self.add_margin_to_detections(detection_single, frame_size=frame_size)

            # 6 - crop
            faces_crop = self.crop_faces(frames[i], detections_margin)
            
            faces_final = []
            for face in faces_crop:
                # 7 - resize
                face_resized = self.isotropically_resize_image(face, self.output_size)
            
                # 8 - fill blank
                face_final = self.make_square_image(face_resized)
                faces_final.append(face_final)
            total_faces.append(faces_final)
        return total_faces
                
    
    
    def tile_frames(self, frames, target_size):
        num_frames, H, W, _ = frames.shape

        split_size = min(H, W)
        x_step = (W - split_size) // 2
        y_step = (H - split_size) // 2
        num_v = 1
        num_h = 3 if W > H else 1

        splits = np.zeros((num_frames * num_v * num_h, target_size[1], target_size[0], 3), dtype=np.uint8)

        i = 0
        for f in range(num_frames):
            y = 0
            for v in range(num_v):
                x = 0
                for h in range(num_h):
                    crop = frames[f, y:y+split_size, x:x+split_size, :]
                    splits[i] = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
                    x += x_step
                    i += 1
                y += y_step

        resize_info = [split_size / target_size[0], split_size / target_size[1]]
        return splits, resize_info
    
    
    
    def resize_detections(self, detections, target_size, resize_info):
        '''get detections from [0, 1] to original frame size (1080, 1080)'''

        projected = []
        target_w, target_h = target_size  # (128, 128)
        scale_w, scale_h = resize_info #  (1080/128, 1080/128)

        # each frame
        for i in range(len(detections)):
            detection = detections[i].clone()

            # ymin, xmin, ymax, xmax
            for k in range(2): #0, 1
                detection[:, k*2    ] = (detection[:, k*2    ] * target_h) * scale_h
                detection[:, k*2 + 1] = (detection[:, k*2 + 1] * target_w) * scale_w

            # keypoints are x,y
            for k in range(2, 8):
                detection[:, k*2    ] = (detection[:, k*2    ] * target_w) * scale_w
                detection[:, k*2 + 1] = (detection[:, k*2 + 1] * target_h) * scale_h

            projected.append(detection)

        return projected 
    
    
    
    def untile_detections(self, num_frames, frame_size, detections):
        """With N tiles per frame, there also are N times as many detections.
        This function groups together the detections for a given frame; it is
        the complement to tile_frames().
        detections: (192, ?, 17)
        """
        combined_detections = []

        W, H = frame_size
        split_size = min(H, W)
        x_step = (W - split_size) // 2
        y_step = (H - split_size) // 2
        num_v = 1
        num_h = 3 if W > H else 1

        i = 0
        for f in range(num_frames):
            detections_for_frame = []
            y = 0
            for v in range(num_v):
                x = 0
                for h in range(num_h):
                    # Adjust the coordinates based on the split positions.
                    detection = detections[i].clone()
                    if detection.shape[0] > 0:
                        for k in range(2):
                            detection[:, k*2    ] += y
                            detection[:, k*2 + 1] += x
                        for k in range(2, 8):
                            detection[:, k*2    ] += x
                            detection[:, k*2 + 1] += y

                    detections_for_frame.append(detection)
                    x += x_step
                    i += 1
                y += y_step

            combined_detections.append(torch.cat(detections_for_frame))

        return combined_detections
        
        
        
    def add_margin_to_detections(self, detections, frame_size, margin=0.2):
        """Expands the face bounding box.

        NOTE: The face detections often do not include the forehead, which
        is why we use twice the margin for ymin.

        Arguments:
            detections: a PyTorch tensor of shape (num_detections, 17)
            frame_size: maximum (width, height)
            margin: a percentage of the bounding box's height

        Returns a PyTorch tensor of shape (num_detections, 17).
        """
        offset = torch.round(margin * (detections[:, 2] - detections[:, 0])) # (ymax - ymin)*margin
        detections = detections.clone()
        detections[:, 0] = torch.clamp(detections[:, 0] - offset*2, min=0)            # ymin
        detections[:, 1] = torch.clamp(detections[:, 1] - offset, min=0)              # xmin
        detections[:, 2] = torch.clamp(detections[:, 2] + offset, max=frame_size[1])  # ymax
        detections[:, 3] = torch.clamp(detections[:, 3] + offset, max=frame_size[0])  # xmax
        return detections


      
    def crop_faces(self, frame, detections):
        """Copies the face region(s) from the given frame into a set
        of new NumPy arrays.

        Arguments:
            frame: a NumPy array of shape (H, W, 3)
            detections: a PyTorch tensor of shape (num_detections, 17)

        Returns a list of NumPy arrays, one for each face crop. If there
        are no faces detected for this frame, returns an empty list.
        """
        faces = []
        for i in range(len(detections)):
            ymin, xmin, ymax, xmax = detections[i, :4].cpu().numpy().astype(np.int)
            face = frame[ymin:ymax, xmin:xmax, :]
            faces.append(face)
        return faces
        
        
        
    def isotropically_resize_image(self, img, size, resample=cv2.INTER_AREA):
        h, w = img.shape[:2]
        if w > h:
            h = h * size // w
            w = size
        else:
            w = w * size // h
            h = size

        resized = cv2.resize(img, (w, h), interpolation=resample)
        return resized
        
        
        
    def make_square_image(self, img):
        h, w = img.shape[:2]
        size = max(h, w)
        t = 0
        b = size - h
        l = 0
        r = size - w
        return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

%%time
before = time.time()

blaze = BlazeFaceExtractor()
total_faces = blaze.extract_faces(sample_video, read_video2, IMG_SIZE)

after = time.time()
print('result {} with time {} seconds'.format(len(total_faces), round(after-before, 1)))
fig, ax = plt.subplots(cell, cell, figsize=(8, 8))
for i in range(cell):
    for j in range(cell):
        ax[i][j].imshow(total_faces[random_idx[i*cell+j]][0])
def load_mobilenetv2_224_075_detector(path):
    input_tensor = Input(shape=(224, 224, 3))
    output_tensor = MobileNetV2(weights=None, include_top=False, input_tensor=input_tensor, alpha=0.75).output
    output_tensor = ZeroPadding2D()(output_tensor)
    output_tensor = Conv2D(kernel_size=(3, 3), filters=5)(output_tensor)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.load_weights(path)
    
    return model
# mobilenetv2 = load_mobilenetv2_224_075_detector("../input/facedetection-mobilenetv2/facedetection-mobilenetv2-size224-alpha0.75.h5")
# mobilenetv2.summary()
imgs_path = glob.glob('../input/dfdcfaceevaluate/*.jpg')
imgs = np.array([cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in imgs_path])
print(imgs.shape)

fig, ax = plt.subplots(4, 5, figsize=(10, 10))
for i in range(4):
    for j in range(5):
        ax[i][j].imshow(cv2.cvtColor(cv2.imread(imgs_path[i*4+j]), cv2.COLOR_BGR2RGB))
fig, ax = plt.subplots(4, 5, figsize=(10, 10))
for i in range(4):
    for j in range(5):
        # show origin face
        ax[i][j].imshow(imgs[i*4+j])
        # show detected
        f = mtcnn_detector.detect_faces(imgs[i*4+j])
        if len(f) <= 0:
            continue
        for n in f:
            x,y,w,h = n['box']
            rect = patches.Rectangle((x,y),w,h, linewidth=1,edgecolor='r',facecolor='none')
            ax[i][j].add_patch(rect)

plt.show()
fig, ax = plt.subplots(4, 5, figsize=(10, 10))
for i in range(4):
    for j in range(5):        
        # show detected
        f = net_facenet(Image.fromarray(imgs[i*4+j]))
        if f is None:
            continue
        else:
            ax[i][j].imshow(imgs[i*4+j])
plt.show()
fig, ax = plt.subplots(4, 5, figsize=(10, 10))
for i in range(4):
    for j in range(5):     
        # show origin face
        ax[i][j].imshow(imgs[i*4+j])
        
        # show detected
        gray = cv2.cvtColor(imgs[i*4+j], cv2.COLOR_BGR2GRAY)
        f = dlib_hog_detector(gray, 1)
        if len(f) > 0:
            f = f[0]
            x = f.left()
            y = f.top()
            w = f.right() - f.left()
            h = f.bottom() - f.top()
           
            rect = patches.Rectangle((x,y),w,h, linewidth=1,edgecolor='r',facecolor='none')
            ax[i][j].add_patch(rect)

plt.show()
fig, ax = plt.subplots(4, 5, figsize=(10, 10))
for i in range(4):
    for j in range(5):     
        # show origin face
        ax[i][j].imshow(imgs[i*4+j])
        
        # show detected
        f = insight_detector.get(imgs[i*4+j])
        if len(f) > 0:
            x = f[0].bbox[0]
            y = f[0].bbox[1]
            w = f[0].bbox[2] - f[0].bbox[0]
            h = f[0].bbox[3] - f[0].bbox[1]
            
            rect = patches.Rectangle((x,y),w,h, linewidth=1,edgecolor='r',facecolor='none')
            ax[i][j].add_patch(rect)
        
plt.show()
fig, ax = plt.subplots(4, 5, figsize=(10, 10))
for i in range(4):
    for j in range(5):     
        # show origin face
        ax[i][j].imshow(imgs[i*4+j])
        
        # show detected
        f = mobile_detector(imgs[4*i+j])
        if len(f) > 0:
            x = f[0] # left
            y = f[2] # top
            w = f[1] - f[0] # right - left
            h = f[3] - f[2] # bottom - top
            
            rect = patches.Rectangle((x,y),w,h, linewidth=1,edgecolor='r',facecolor='none')
            ax[i][j].add_patch(rect)
        
plt.show()
fig, ax = plt.subplots(4, 5, figsize=(10, 10))
for i in range(4):
    for j in range(5):     
        # show origin face
        ax[i][j].imshow(imgs[i*4+j])
        
        # show detected
        f = blaze_detector(imgs[4*i+j])
        if len(f) > 0:
            x = f[0][0] # xmin
            y = f[0][1] # ymin
            w = f[0][2] - f[0][0] # right - left
            h = f[0][3] - f[0][1] # bottom - top
            
            rect = patches.Rectangle((x,y),w,h, linewidth=1,edgecolor='r',facecolor='none')
            ax[i][j].add_patch(rect)
        
plt.show()
import sys
import torch
sys.path.insert(0,"/kaggle/input/retina-face/Pytorch_Retinaface/")
from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace
cfg_re50['image_size'], cfg_mnet['image_size']
def get_model(modelname="mobilenet"):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    torch.set_grad_enabled(False)
    cfg = None
    cfg_mnet['pretrain'] = False
    cfg_re50['pretrain'] = False
    
    if modelname == "mobilenet":
        cfg = cfg_mnet
        pretrained_path = "/kaggle/input/retina-face/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth"
    else:
        cfg = cfg_re50
        pretrained_path = "/kaggle/input/retina-face/Pytorch_Retinaface/weights/Resnet50_Final.pth"
    
    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')))
    net.eval().to(device)
    return net

retina_detector = get_model()