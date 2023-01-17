from fastai.vision import *
train_sample_metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').T.reset_index()

train_sample_metadata.columns = ['fname','label','split','original']

train_sample_metadata.head()
fake_sample_df = train_sample_metadata[train_sample_metadata.label == 'FAKE']

real_sample_df = train_sample_metadata[train_sample_metadata.label == 'REAL']
train_dir = Path('/kaggle/input/deepfake-detection-challenge/train_sample_videos/')

test_dir = Path('/kaggle/input/deepfake-detection-challenge/test_videos/')

train_video_files = get_files(train_dir, extensions=['.mp4'])

test_video_files = get_files(test_dir, extensions=['.mp4'])
len(train_video_files), len(test_video_files)
dummy_video_file = train_video_files[0]
!cp /kaggle/input/decord/install.sh . && chmod  +x install.sh && ./install.sh 
sys.path.insert(0,'/kaggle/working/reader/python')



from decord import VideoReader

from decord import cpu

from decord.bridge import set_bridge

set_bridge('torch')

device = torch.device("cuda")
retinaface_stats = tensor([123,117,104]).to(device) # RGB stats for mean



def decord_cpu_video_reader(path, freq=None):

    video = VideoReader(str(path), ctx=cpu())

    len_video = len(video)

    if freq: t = video.get_batch(range(0, len(video), freq)).permute(0,3,1,2)

    else: t = video.get_batch(range(len_video))

    return t, len_video



#export

def get_decord_video_batch_cpu(path, freq=10, sz=640, stats:Tensor=None, device=defaults.device):

    "get resized and mean substracted batch tensor of a sampled video (scale of 255)"

    t_raw, len_video = decord_cpu_video_reader(path, freq)

    H,W = t_raw.size(2), t_raw.size(3) 

    t = F.interpolate(t_raw.to(device).to(torch.float32), (sz,sz))

    if stats is not None: t -= stats[...,None,None]

    return t, t_raw, (H, W)
retinaface_stats = tensor([123,117,104]).to(device) # RGB stats for mean

t, t_raw, (H,W) = get_decord_video_batch_cpu(dummy_video_file, 10, 640, retinaface_stats)
sys.path.insert(0,"/kaggle/input/retina-face-2/Pytorch_Retinaface_2/")
import os

import torch

import torch.backends.cudnn as cudnn

import numpy as np

from data import cfg_mnet, cfg_re50

from layers.functions.prior_box import PriorBox

from utils.nms.py_cpu_nms import py_cpu_nms

import cv2

from models.retinaface import RetinaFace

from utils.box_utils import decode, decode_landm

import time
def check_keys(model, pretrained_state_dict):

    ckpt_keys = set(pretrained_state_dict.keys())

    model_keys = set(model.state_dict().keys())

    used_pretrained_keys = model_keys & ckpt_keys

    unused_pretrained_keys = ckpt_keys - model_keys

    missing_keys = model_keys - ckpt_keys

    print('Missing keys:{}'.format(len(missing_keys)))

    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))

    print('Used keys:{}'.format(len(used_pretrained_keys)))

    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'

    return True





def remove_prefix(state_dict, prefix):

    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''

    print('remove prefix \'{}\''.format(prefix))

    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}





def load_model(model, pretrained_path, load_to_cpu):

    print('Loading pretrained model from {}'.format(pretrained_path))

    if load_to_cpu:

        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)

    else:

        device = torch.cuda.current_device()

        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():

        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')

    else:

        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    check_keys(model, pretrained_dict)

    model.load_state_dict(pretrained_dict, strict=False)

    return model
cfg_re50['image_size'], cfg_mnet['image_size']
cudnn.benchmark = True # keep input size constant for better runtime
def get_model(modelname="mobilenet"):

    torch.set_grad_enabled(False)

    cfg = None

    cfg_mnet['pretrain'] = False

    cfg_re50['pretrain'] = False

    

#     if modelname == "mobilenet": 

#         pretrained_path = "../input/retina-face-2/Pytorch_Retinaface_2/weights/mobilenet0.25_Final.pth"

#         cfg = cfg_mnet

#     else: raise Exception(f"only options are: 'mobilenet' or 'resnet50'")



    if modelname == "mobilenet": 

        pretrained_path = "../input/retina-face-2/Pytorch_Retinaface_2/weights/mobilenet0.25_Final.pth"

        cfg = cfg_mnet

        

    if modelname == "resnet50": 

        pretrained_path = "../input/retina-face/Pytorch_Retinaface/weights/Resnet50_Final.pth"

        cfg = cfg_re50

    

    # net and model

    net = RetinaFace(cfg=cfg, phase='test')

    net = load_model(net, pretrained_path, False)

    net.eval().to(device)

    return net, cfg
def predict(model, t, sz, cfg, 

            confidence_threshold = 0.5, top_k = 5, nms_threshold = 0.5, keep_top_k = 5):

    "get prediction for a batch t by model with image sz"



    resize = 1

    scale_rate = 1



    im_height, im_width = sz, sz 

    scale = torch.Tensor([sz, sz, sz, sz])

    scale = scale.to(device)

    

    

    locs, confs, landmss = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

    locs = locs.to(device)

    confs = confs.to(device)

    landmss = landmss.to(device)

    

    # forward pass

    locs_, confs_, landmss_ = model(t)  

    locs = torch.cat((locs, locs_), 0)

    confs = torch.cat((confs, confs_), 0)

    landmss = torch.cat((landmss, landmss_), 0)

    

    

    bbox_result, landms_result = [], []

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))

    priors = priorbox.forward()

    priors = priors.to(device)

    prior_data = priors.data

    for idx in range(t.size(0)):

        loc = locs[idx]

        conf = confs[idx]

        landms = landmss[idx]



        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])

        boxes = boxes * scale / resize



        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])

        scale1 = torch.Tensor([t.shape[3], t.shape[2], t.shape[3], t.shape[2],

                            t.shape[3], t.shape[2], t.shape[3], t.shape[2],

                            t.shape[3], t.shape[2]])

        scale1 = scale1.to(device)

        landms = landms * scale1 / resize

        landms = landms.cpu().numpy()



        # ignore low scores

        inds = np.where(scores > confidence_threshold)[0]

        boxes = boxes[inds]

        landms = landms[inds]

        scores = scores[inds]



        # keep top-K before NMS

        order = scores.argsort()[::-1][:top_k]

        boxes = boxes[order]

        landms = landms[order]

        scores = scores[order]



        # do NMS

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

        keep = py_cpu_nms(dets, nms_threshold)



        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)

        dets = dets[keep, :]

        landms = landms[keep]

    

        # keep top-K faster NMS

        dets = dets[:keep_top_k, :]

        landms = landms[:keep_top_k, :]



    #     dets = np.concatenate((dets, landms), axis=1)

    #     dets = np.concatenate((dets, landms), axis=1)

        bbox_result.append(dets[:,:-1]) # don't keep confidence score

        landms_result.append(landms)

    

    return  bbox_result, landms_result
%%time

model, cfg = get_model("mobilenet")
def bboxes_to_original_scale(bboxes, H, W, sz):

    """

    convert bbox points to original image scale

    

    bboxes: List of numpy arrays with shape (M, 4) M: # of bbox coordinates

    """

    res = []

    for bb in bboxes:

        h_scale, w_scale = H/sz, W/sz

        orig_bboxes = (bb*array([w_scale, h_scale, w_scale, h_scale])[None, ...]).astype(int)

        res.append(orig_bboxes)

    return res
def resize_bbox_by_scale(bb, bb_scale, H, W):

    """

    resize a bbox with a given scale parameter

    

    bb: a bounding box with (left, top, right, bottom) values

    """

    left, top, right, bottom = bb

    

    cx,cy = (top + bottom)//2, (left + right)//2 

    h,w = (bottom - top), (right - left)

    sh, sw = int(h*bb_scale), int(w*bb_scale)



    stop, sbottom = cx - sh//2, cx + sh//2

    sleft, sright = cy - sw//2, cy + sw//2

    stop, sleft, sbottom, sright = max(0, stop), max(0, sleft), min(H, sbottom), min(W, sright)    

    return (sleft, stop, sright, sbottom)
def landmarks_to_original_scale(landmarks, H, W, sz):

    """

    convert landmarks to original image scale

    

    landmarks: List of numpy arrays with shape (M, 10) M: # landmark coordinates

    """

    res = []

    for landms in landmarks:

        h_scale, w_scale = H/sz, W/sz

        orig_landms = (landms*array([w_scale, h_scale]*5)[None, ...]).astype(int)

        res.append(orig_landms)

    return res
from tqdm import tqdm
# face detection config

freq = 5

model_args = dict(confidence_threshold = 0.5, top_k = 5, nms_threshold = 0.5, keep_top_k = 5)

sz = cfg['image_size']

imgnet_stats = [tensor(o) for o in imagenet_stats]

rescale_param = 1.3
# !pip install -q ../input/pretrainedmodels/pretrainedmodels-0.7.4/

!pip install -q ../input/efficientnetpytorchpip/efficientnet_pytorch-0.6.3/
# from fastai.vision.models.cadene_models import *

from fastai.vision.models.efficientnet import *
# load detection model

class DummyDatabunch:

    c = 2

    path = '.'

    device = defaults.device

    loss_func = None



data = DummyDatabunch()
effnet_model = EfficientNet.from_name("efficientnet-b5", override_params={'num_classes': 2})

learner = Learner(data, effnet_model); learner.model_dir = '.'

learner.load('../input/deepfakerandmergeaugmodels/single_frame_effnetb5_randmerge')

effnetb5_inference_model = learner.model.eval()
effnet_model = EfficientNet.from_name("efficientnet-b7", override_params={'num_classes': 2})

learner = Learner(data, effnet_model); learner.model_dir = '.'

learner.load('../input/deepfakerandmergeaugmodels/single_frame_effnetb7_randmerge_fp16')

effnetb7_inference_model = learner.model.float().eval()
learner = cnn_learner(data, models.resnet34, pretrained=False); learner.model_dir = '.'

learner.load('../input/deepfakerandmergeaugmodels/single_frame_resnet34_randmerge')

resnet_inference_model = learner.model.eval()
predictions = []

video_fnames = []
for fname in tqdm(test_video_files):

    try:

        # append video filename

        video_fnames.append(fname.name)

        

        # read video frames

        t, t_raw, (H,W) = get_decord_video_batch_cpu(fname, freq, sz, retinaface_stats)

        

        # detect bboxes

        bboxes, landmarks = predict(model, t, sz, cfg, **model_args)

        orig_bboxes = bboxes_to_original_scale(bboxes, H, W, sz)

        orig_bboxes = [o.tolist() for o in orig_bboxes]

        

        # create face crop batch

        video_crop_batch, video_crop_batch_tta = [], []

        for frame_no, (_frame, _bb) in enumerate(zip(t_raw, orig_bboxes)):

            # don't try cropping if no detection is available for the frame

            try: _bb[0] 

            except: continue

            # naive: get first bbox, optionally rescale

            left, top, right, bottom  = resize_bbox_by_scale(_bb[0], rescale_param, H, W) 

            

            # crop and save

            face_crop = Image(_frame[:, top:bottom, left:right].to(torch.float32).div(255))

            # resize

            x = face_crop.resize(224).data

            # normalize

            x = normalize(x, *imgnet_stats)

            video_crop_batch.append(x)

            

            # crop and save

            face_crop = Image(_frame[:, top:bottom, left:right].to(torch.float32).div(255)).flip_lr()

            # resize

            x = face_crop.resize(224).data

            # normalize

            x = normalize(x, *imgnet_stats)

            video_crop_batch_tta.append(x)

        

        # batches

        video_crop_batch = torch.stack(video_crop_batch)

        video_crop_batch_tta = torch.stack(video_crop_batch_tta)

        

        

        # do inference

        resnet_out = to_cpu(resnet_inference_model(video_crop_batch.to(device))).softmax(1)

        effnetb5_out = to_cpu(effnetb5_inference_model(video_crop_batch.to(device))).softmax(1)

        effnetb7_out = to_cpu(effnetb7_inference_model(video_crop_batch.to(device))).softmax(1)     

        # tta

        resnet_out_tta = to_cpu(resnet_inference_model(video_crop_batch_tta.to(device))).softmax(1)

        effnetb5_out_tta = to_cpu(effnetb5_inference_model(video_crop_batch_tta.to(device))).softmax(1)

        effnetb7_out_tta = to_cpu(effnetb7_inference_model(video_crop_batch_tta.to(device))).softmax(1)     

        

        

        mean_pred = torch.stack([resnet_out[:,1],

                                 effnetb5_out[:,1],

                                 effnetb7_out[:,1],

                                 resnet_out_tta[:,1],

                                 effnetb5_out_tta[:,1],

                                 effnetb7_out_tta[:,1]], 1).mean(1).mean().item()                

        predictions.append(mean_pred)





    except:

        predictions.append(0.5)
plt.hist(predictions)
fname2pred = dict(zip(video_fnames, predictions))
submission_df = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")
submission_df.label = submission_df.filename.map(fname2pred)
submission_df['label'] = np.clip(submission_df['label'], 0.01, 0.99)
submission_df.to_csv("submission.csv",index=False)