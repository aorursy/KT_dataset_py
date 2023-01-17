import numpy as np 
import pandas as pd 
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
import cv2
from scipy.linalg import inv, block_diag
from numpy import dot
import tensorflow as tf
import tensorflow_hub as hub
import time
import cv2
import glob
import os.path
import json
from annoy import AnnoyIndex
from scipy import spatial

import time
frame_count = 0 
max_age = 10 
min_hits =1 
tracker_list =[]
track_id_list= deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])
debug = True
print("---------------------------------")
print ("Step.1 of 2 - mobilenet_v2_140_224 - Loading Started")
print("---------------------------------")
module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4" 
module = hub.load(module_handle)
ret, frame = cap.read()
img = pipeline(frame, 100)
cap = cv2.VideoCapture('../input/motdataset/4p-c1.avi')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/kaggle/working/tracked2.mp4',fourcc,20, (800, 600))

count = 0
if (cap.isOpened()== False):  
    print("Error opening video  file") 
while(cap.isOpened()): 
    ret, frame = cap.read() 
    if ret == True:
        bbox = get_localization(frame)
        print (bbox)
#         img = pipeline(frame, count)
        count += 1
        out.write(img)
        
    else:
        break
        
cap.release()
out.release()
end  = time.time()        
print(round(end-start, 2), 'Seconds to finish')
def pipeline(img, count):
    global frame_count
    global tracker_list
    global max_age
    global min_hits
    global track_id_list
    global debug
    frame_count+=1
    img_dim = (img.shape[1], img.shape[0])

    z_tmp = get_localization(img)
    z_box = z_tmp[~np.all(z_tmp==0, axis=1)]
    z_box[z_box < 0] = 0
    
    if z_box.shape[0] == 0:
        return img

    x_box =[]
    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)
            
    matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(x_box, z_box, img, iou_thrd = 0.3)
    print ("matched : " +str(matched.size))
    print ('unmatched_dets :'+ str(len(unmatched_dets)) )
    print ("unmatched_trks :"+str(len(unmatched_trks)))
    
    if matched.size >0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk= tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box =xx
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0
            
    if len(unmatched_dets)>0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = Tracker() # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state   # xx = x
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
            tracker_list.append(tmp_trk)
            x_box.append(xx)
            
    if len(unmatched_trks)>0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box =xx
            x_box[trk_idx] = xx
       
            
    good_tracker_list =[]
    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
            good_tracker_list.append(trk)
            x_cv2 = trk.box
            if debug:
                print('updated box: ', x_cv2)
                print('Track id: ',trk.id)
            img= draw_box_label(img, x_cv2, trk.id) # Draw the bounding boxes on the 
                                             # images
    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)  
    
    for trk in deleted_tracks:
        track_id_list.append(trk.id)
    
    tracker_list = [x for x in tracker_list if x.no_losses<=max_age]
    if debug:
        print('Ending tracker_list: ',len(tracker_list))
        print('Ending good tracker_list: ',len(good_tracker_list))
    
    return img
np.random.seed(42)
import cv2
LABELS = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
configPath = '../input/file-cfg/cfg/yolov3.cfg'
weightsPath = '../input/yolov3/yolov3.weights'
net = cv2.dnn.readNetFromDarknet(configPath , weightsPath)

def get_localization(image):
    (H, W) = image.shape[:2]
#     print (H, W)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
    x_up = []
    y_up = []
    x_down = []
    y_down = []
    if len(idxs) > 0:
        detected = np.zeros([len(boxes), 4])
        for i in idxs.flatten():
            if LABELS[classIDs[i]] == 'person':
                detected[i,:] = [boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]]
                
    return detected

class features():
    def __init__(self, total_detection, total_track, img):
        self.detection_feature = [0 for i in range(total_detection)]
        self.track_feature = [0 for i in range(total_track)]
        self.roi_det = [0 for i in range(total_detection)]
        self.roi_trk = [0 for i in range(total_track)]
        self.similar = np.zeros([total_track, total_detection])
        self.image = img
    
    def set_feature(self, det_dims, trk_dims):
        for idx,_ in enumerate(self.roi_det):
            self.roi_det[idx] = load_img(self.image[int(det_dims[idx,1]):int(det_dims[idx,3]), int(det_dims[idx,0]):int(det_dims[idx,2])])
            features = module(self.roi_det[idx])   
            self.detection_feature[idx] = np.squeeze(features)
            
        for idx,_ in enumerate(self.roi_trk):
            self.roi_trk[idx] = load_img(self.image[int(trk_dims[idx][1]):int(trk_dims[idx][3]), int(trk_dims[idx][0]):int(trk_dims[idx][2])])
            features = module(self.roi_trk[idx])   
            self.track_feature[idx] = np.squeeze(features)
        


    def set_similarity(self): # det1 [4] trk1 [4]
        for i in range(0,self.similar.shape[0]):
            for j in range(0,self.similar.shape[1]):
                file_index_to_file_vector = {}
                file_index_to_product_id = {}
                dims = 1792
                n_nearest_neighbors = 1
                trees = 10000
                t = AnnoyIndex(dims, metric='angular')
                t.add_item(j, self.detection_feature[j])
                t.build(trees)
                master_vector = self.track_feature[i]
                master_product_id = i
                nearest_neighbors = t.get_nns_by_item(0, n_nearest_neighbors)
                neighbor_file_vector = self.detection_feature[j]
                similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
                rounded_similarity = int((similarity * 10000)) / 10000.0
                self.similar[i, j] = rounded_similarity
        
        


def load_img(img):
    img = tf.image.resize_with_pad(img, 224, 224)
    img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    return img
bbox_csv = pd.read_csv('../input/pedestrian-dataset/crosswalk.csv')
bbox_csv.head()
bbox_csv.iloc[0,:]

cap = cv2.VideoCapture('../input/pedestrian-dataset/crosswalk.avi')

count = 0
if (cap.isOpened()== False):  
    print("Error opening video  file") 
while(cap.isOpened()): 
    ret, frame = cap.read() 
    if ret == True and count == 1:
        det = np.array([bbox_csv.iloc[1,0], bbox_csv.iloc[1,1], bbox_csv.iloc[1,0]+bbox_csv.iloc[1,2], bbox_csv.iloc[1,1]+bbox_csv.iloc[1,3]])
        trk = np.array([bbox_csv.iloc[0,0], bbox_csv.iloc[0,1], bbox_csv.iloc[0,0]+bbox_csv.iloc[0,2], bbox_csv.iloc[0,1]+bbox_csv.iloc[0,3]])
        print (det, trk)
        value = get_similarity(det, trk, frame)
        print (value)
        count += 1
    elif count < 1:
        count += 1
    else:
        break
        
cap.release()
end  = time.time()        
print(round(end-start, 2), 'Seconds to finish')

def get_similarity(det_box, trk_box, img):
    roi_det = load_img(img[int(det_box[1]):int(det_box[3]), int(det_box[0]):int(det_box[2]), :])
    features_det = module(roi_det)   
    detection_feature = np.squeeze(features_det)
    roi_trk = load_img(img[int(trk_box[1]):int(trk_box[3]), int(trk_box[0]):int(trk_box[2]), :])
    features_trk = module(roi_trk)   
    tracking_feature = np.squeeze(features_trk)
    dims = 1792
    n_nearest_neighbors = 1
    trees = 10000
    t = AnnoyIndex(dims, metric='angular')
    t.add_item(0, detection_feature)
    t.build(trees)
    master_vector = tracking_feature
    master_product_id = 0
    nearest_neighbors = t.get_nns_by_item(0, n_nearest_neighbors)
    neighbor_file_vector = detection_feature
    similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
    rounded_similarity = int((similarity * 10000)) / 10000.0
    return rounded_similarity
        
class Tracker(): # class for Kalman Filter-based tracker
    def __init__(self):
        # Initialize parametes for tracker (history)
        self.id = 0  # tracker's id 
        self.box = [] # list to store the coordinates for a bounding box 
        self.hits = 0 # number of detection matches
        self.no_losses = 0 # number of unmatched tracks (track loss)
        
        # Initialize parameters for Kalman Filtering
        # The state is the (x, y) coordinates of the detection box
        # state: [up, up_dot, left, left_dot, down, down_dot, right, right_dot]
        # or[up, up_dot, left, left_dot, height, height_dot, width, width_dot]
        self.x_state=[] 
        self.dt = 1.   # time interval
        
        # Process matrix, assuming constant velocity model
        self.F = np.array([[1, self.dt, 0,  0,  0,  0,  0, 0],
                           [0, 1,  0,  0,  0,  0,  0, 0],
                           [0, 0,  1,  self.dt, 0,  0,  0, 0],
                           [0, 0,  0,  1,  0,  0,  0, 0],
                           [0, 0,  0,  0,  1,  self.dt, 0, 0],
                           [0, 0,  0,  0,  0,  1,  0, 0],
                           [0, 0,  0,  0,  0,  0,  1, self.dt],
                           [0, 0,  0,  0,  0,  0,  0,  1]])
        
        # Measurement matrix, assuming we can only measure the coordinates
        
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0], 
                           [0, 0, 0, 0, 0, 0, 1, 0]])
        
        
        # Initialize the state covariance
        self.L = 10.0
        self.P = np.diag(self.L*np.ones(8))
        
        
        # Initialize the process covariance
        self.Q_comp_mat = np.array([[self.dt**4/4., self.dt**3/2.],
                                    [self.dt**3/2., self.dt**2]])
        self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat, 
                            self.Q_comp_mat, self.Q_comp_mat)
        
        # Initialize the measurement covariance
        self.R_scaler = 1.0
        self.R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(self.R_diag_array)
        
        
    def update_R(self):   
        R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(R_diag_array)
        
        
        
        
    def kalman_filter(self, z): 
        '''
        Implement the Kalman Filter, including the predict and the update stages,
        with the measurement z
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        #Update
        S = dot(self.H, self.P).dot(self.H.T) + self.R
        K = dot(self.P, self.H.T).dot(inv(S)) # Kalman gain
        y = z - dot(self.H, x) # residual
        x += dot(K, y)
        self.P = self.P - dot(K, self.H).dot(self.P)
        self.x_state = x.astype(int) # convert to integer coordinates 
                                     #(pixel values)
        
    def predict_only(self):  
        '''
        Implment only the predict stage. This is used for unmatched detections and 
        unmatched tracks
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q
        self.x_state = x.astype(int)
      
def box_iou2(a, b):    
    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[2] - b[0])*(b[3] - b[1])
  
    return float(s_intsec)/(s_a + s_b -s_intsec)


def assign_detections_to_trackers(trackers, detections, image, iou_thrd = 0.3):
    
    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        for d,det in enumerate(detections):
            print (trk)
            print (det)
            print (box_iou2(trk,det))
            IOU_mat[t,d] = (box_iou2(trk,det) + get_similarity(det,trk,image))/2
            
    matched_idx = linear_assignment(-IOU_mat)        

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
    
    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)       
def draw_box_label(img, bbox_cv2, id_no, box_color=(0, 255, 255), show_label=True):
    '''
    Helper funciton for drawing the bounding boxes and the labels
    bbox_cv2 = [left, top, right, bottom]
    '''
    #box_color= (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    font_color = (0, 0, 0)
    left, top, right, bottom = bbox_cv2[0], bbox_cv2[1], bbox_cv2[2], bbox_cv2[3]
    
    # Draw the bounding box
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 4)
    
    if show_label:
        # Draw a filled box on top of the bounding box (as the background for the labels)
        cv2.rectangle(img, (left-2, top-45), (right+2, top), box_color, -1, 1)
        
        # Output the labels that show the x and y coordinates of the bounding box center.
#         text_x= 'x='+str((left+right)/2)
        text_x = str(id_no)
        cv2.putText(img,text_x,(left,top-25), font, font_size, font_color, 1, cv2.LINE_AA)
#         text_y= 'y='+str((top+bottom)/2)
#         cv2.putText(img,text_y,(left,top-5), font, font_size, font_color, 1, cv2.LINE_AA)
        plt.imshow(img)
    
    return img    
import tensorflow as tf
import tensorflow_hub as hub
import time
import cv2
import glob
import os.path
def load_img(img):
    img = tf.image.resize_with_pad(img, 224, 224)
    img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    return img


def get_image_feature_vectors():
    i = 0
    start_time = time.time()
    print("---------------------------------")
    print ("Step.1 of 2 - mobilenet_v2_140_224 - Loading Started at %s" %time.ctime())
    print("---------------------------------")
    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4" 
    module = hub.load(module_handle)

    cap = cv2.VideoCapture('../input/pedestrian-dataset/crosswalk.avi')
    count = 0
    if (cap.isOpened()== False):  
        print("Error opening video  file") 
    while(cap.isOpened()): 
        ret, frame = cap.read() 
        if ret == True:
            print (count, end='\r')
            bbox = bbox_csv.iloc[count,:]
            count += 1
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
            if xmin < 0: xmin = 0
            if xmax < 0: xmax = 0
            if ymin < 0: ymin = 0
            if ymax < 0: ymax = 0
            roi = frame[ymin:ymax, xmin:xmax]
            img = load_img(roi)
            features = module(img)   
            feature_set = np.squeeze(features) 
            print(feature_set.shape)
            outfile_name = "frame"+str(count) + ".npz"
            out_path = os.path.join('/kaggle/working/', outfile_name)
            np.savetxt(out_path, feature_set, delimiter=',')

        else:
            break
    cap.release()

get_image_feature_vectors()
import glob
import os.path
import json
from annoy import AnnoyIndex
from scipy import spatial


def get_similarity(det, trk): # det1 [4] trk1 [4]
    file_index_to_file_vector = {}
    file_index_to_product_id = {}
    dims = 1792
    n_nearest_neighbors = 1
    trees = 10000
    t = AnnoyIndex(dims, metric='angular')
    file_index_to_file_vector[file_index] = file_vector
    file_index_to_product_id[file_index] = file_name[5:]
    t.add_item(0, self.track_feature[0])
    t.build(trees)
    master_vector = self.detect_feature[0]
    master_product_id = 0
    nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)
    for j in nearest_neighbors:
        neighbor_file_vector = file_index_to_file_vector[j]
        neighbor_product_id = file_index_to_product_id[j]
        similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
        rounded_similarity = int((similarity * 10000)) / 10000.0
        


def cluster():

    start_time = time.time()
    file_index_to_file_name = {}
    file_index_to_file_vector = {}
    file_index_to_product_id = {}
    dims = 1792
    n_nearest_neighbors = 20
    trees = 10000
    allfiles = glob.glob('/kaggle/working/*.npz')

    t = AnnoyIndex(dims, metric='angular')
    named_nearest_neighbors = []
    
    

    for file_index, i in enumerate(allfiles):
        file_vector = np.loadtxt(i)
        file_name = os.path.basename(i).split('.')[0]
        file_index_to_file_name[file_index] = file_name
        file_index_to_file_vector[file_index] = file_vector
        file_index_to_product_id[file_index] = file_name[5:]
        t.add_item(file_index, file_vector)

        print("---------------------------------")
        print("Annoy index     : %s" %file_index)
        print("Image file name : %s" %file_name)
        print("Product id      : %s" %file_index_to_product_id[file_index])
        print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))

    t.build(trees)

    print ("Step.1 - ANNOY index generation - Finished")
    print ("Step.2 - Similarity score calculation - Started ") 
  
    named_nearest_neighbors = []

    for i in file_index_to_file_name.keys():
        i = 0
        # Assigns master file_name, image feature vectors and product id values
        master_file_name = file_index_to_file_name[i]
        master_vector = file_index_to_file_vector[i]
        master_product_id = file_index_to_product_id[i]

        # Calculates the nearest neighbors of the master item
        nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)

        # Loops through the nearest neighbors of the master item
        for j in nearest_neighbors:
            neighbor_file_name = file_index_to_file_name[j]
            neighbor_file_vector = file_index_to_file_vector[j]
            neighbor_product_id = file_index_to_product_id[j]
      
            similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
            rounded_similarity = int((similarity * 10000)) / 10000.0
            named_nearest_neighbors.append({
                'similarity': rounded_similarity,
                'master_pi': master_product_id,
                'similar_pi': neighbor_product_id})

        print("---------------------------------") 
        print("Similarity index       : %s" %i)
        print("Master Image file name : %s" %file_index_to_file_name[i]) 
        print("Nearest Neighbors.     : %s" %nearest_neighbors) 
        print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))

  
    print ("Step.2 - Similarity score calculation - Finished ") 

    # Writes the 'named_nearest_neighbors' to a json file
    with open('/kaggle/working/nearest_neighbors.json', 'w') as out:
        json.dump(named_nearest_neighbors, out)

    print ("Step.3 - Data stored in 'nearest_neighbors.json' file ") 
    print("--- Prosess completed in %.2f minutes ---------" % ((time.time() - start_time)/60))
cluster()
