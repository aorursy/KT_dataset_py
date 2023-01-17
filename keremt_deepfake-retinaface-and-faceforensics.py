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
!pip install -q /kaggle/input/pretrainedmodels/pretrainedmodels-0.7.4
sys.path.insert(0,'/kaggle/input/faceforensics-pretrained/FaceForensics/classification/')
xception_model = torch.load("/kaggle/input/faceforensics-pretrained/faceforensics_models_subset/xception/full_raw.p");
!cp /kaggle/input/decord/install.sh . && chmod  +x install.sh && ./install.sh 
sys.path.insert(0,'/kaggle/working/reader/python')



from decord import VideoReader

from decord import cpu, gpu

from decord.bridge import set_bridge

set_bridge('torch')

device = torch.device("cuda")
retinaface_stats = tensor([123,117,104])[...,None,None].to(device) # RGB stats for mean



def get_decord_video_batch_cpu(fname, sz, freq=10, stats=None):

    "get batch tensor for inference, original for cropping and H,W of video"

    video = VideoReader(str(fname), ctx=cpu())

    t = video.get_batch(range(0, len(video), freq))

    H,W = t.shape[2:]

    if sz: t = F.interpolate(t.to(torch.float32), (sz,sz)).to(device)

    if stats is not None: t -= stats

    return (t, (H, W))
sys.path.insert(0,"/kaggle/input/retina-face/Pytorch_Retinaface/")
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

    

    if modelname == "mobilenet":

        cfg = cfg_mnet

        pretrained_path = "/kaggle/input/retina-face/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth"

    else:

        cfg = cfg_re50

        pretrained_path = "/kaggle/input/retina-face/Pytorch_Retinaface/weights/Resnet50_Final.pth"

    

    # net and model

    net = RetinaFace(cfg=cfg, phase='test')

    net = load_model(net, pretrained_path, False)

    net.eval().to(device)

    return net
resize = 1

scale_rate = 1



sz = 640

im_height, im_width = sz, sz 

scale = torch.Tensor([sz, sz, sz, sz])

scale = scale.to(device)





confidence_threshold = 0.5

top_k = 5

nms_threshold = 0.5

keep_top_k = 5









def predict(model:torch.nn.Module, t:tensor, sz:int, cfg):

    "get prediction for a batch t by model with image sz"

    locs, confs, landmss = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

    locs = locs.to(device)

    confs = confs.to(device)

    landmss = landmss.to(device)

    

    # forward pass

    locs_, confs_, landmss_ = model(t)  

    locs = torch.cat((locs, locs_), 0)

    confs = torch.cat((confs, confs_), 0)

    landmss = torch.cat((landmss, landmss_), 0)

    





    result = []

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

        result.append(dets[:,:-1])





    for idx in range(len(result)):

        result[idx][:, :4]=result[idx][:, :4]/scale_rate

#         result[idx][:, 5:]=result[idx][:, 5:]/scale_rate

    

    return result
%%time

model = get_model("mobilenet")
def convert_bboxes(bboxes, H, W, sz):

    "rescale to original image sz"

    res = []

    for bb in bboxes:

        h_scale, w_scale = H/sz, W/sz

        orig_bboxes = (bb*array([w_scale, h_scale, w_scale, h_scale])[None, ...]).astype(int)

        res.append(orig_bboxes)

    return res
def rescale_bbox(bb, bb_scale, H,W):

    "rescale a bbox: (left, top, right, bottom) with a given scale parameter"

    left, top, right, bottom = bb

    

    cx,cy = (top + bottom)//2, (left + right)//2 

    h,w = (bottom - top), (right - left)

    sh, sw = int(h*bb_scale), int(w*bb_scale)



    stop, sbottom = cx - sh//2, cx + sh//2

    sleft, sright = cy - sw//2, cy + sw//2

    stop, sleft, sbottom, sright = max(0, stop), max(0, sleft), min(H, sbottom), min(W, sright)    

    return (sleft, stop, sright, sbottom)
xception_model = xception_model.model
xception_state_dict = torch.load("/kaggle/input/deepfake-trained-models/part-49-xception-single-face.pth")
xception_model.load_state_dict(xception_state_dict['model']);
xception_stats = (tensor([0.5,0.5,0.5], tensor([0.5,0.5,0.5])))
from tqdm import tqdm
retinaface_stats = tensor([123,117,104])[...,None,None].to(device) # RGB stats for mean



def get_decord_video_batch_cpu_inference(fname, sz, freq=10, stats=None):

    "get batch tensor for inference, original for cropping and H,W of video"

    video = VideoReader(str(fname), ctx=cpu())

    t_raw = video.get_batch(range(0, len(video), freq))

    H,W = t_raw.shape[2:]

    if sz: t = F.interpolate(t_raw.to(torch.float32), (sz,sz)).to(device)

    if stats is not None: t -= stats

    return (t, t_raw, (H, W))
predictions = []
%%time

sz = 640

freq = 10

for fname in tqdm(test_video_files):

    try:

        t, t_raw, (H, W) = get_decord_video_batch_cpu_inference(fname, sz, freq, retinaface_stats)

        bboxes = predict(model, t, sz, cfg_mnet)

        orig_bboxes = convert_bboxes(bboxes, H, W, sz)

        del t; gc.collect()



        # collect crops    

        face_crops = []

        for frame_no, (_frame, _bb) in enumerate(zip(t_raw, orig_bboxes)):

            # don't try cropping if no detection is available for the frame

            try: _bb[0] 

            except: continue

            # naive: get first bbox, optionally rescale

            left, top, right, bottom  = rescale_bbox(_bb[0], 1.3, H, W) 

            # crop and save

            face_crop = F.interpolate(_frame[:, top:bottom, left:right][None].float(), (299,299))[0]

            face_crops.append(face_crop)



        # predict

        xb = normalize(torch.stack(face_crops)/255, *xception_stats)

        score = to_cpu(xception_model(xb.cuda()).softmax(1))[:,1].mean()

        predictions.append(score.item())

    except:

        predictions.append(0.5)
plt.hist(predictions)
test_fnames = [o.name for o in test_video_files]
submission_df = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")
submission_df.label = submission_df.filename.map(dict(zip(test_fnames, predictions)))
submission_df.to_csv("submission.csv",index=False)