from fastai.vision import *

import cv2 as cv
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
def frame_img_generator(path, freq=None):

    "frame image generator for a given video file"

    vidcap = cv.VideoCapture(str(path))

    n_frames = 0

    while True:

        success = vidcap.grab()

        if not success: 

            vidcap.release()

            break   

            

        if (freq is None) or (n_frames % freq == 0):

            _, image = vidcap.retrieve()

            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            yield image    

        

        n_frames += 1

        

    vidcap.release()
%%time

frames = list(frame_img_generator(train_video_files[0],10)); len(frames)
def get_video_batch(fname, sz, freq=10):

    "get batch tensor for inference, original for cropping and H,W of video"

    orig_frames = array(frame_img_generator(fname, freq))

    H,W = orig_frames[0].shape[:-1]

    resized_frames = array([cv2.resize(o,(sz,sz)) for o in orig_frames])

    t = (resized_frames - array([123,117,104]))

    t = (torch.from_numpy(t).to(torch.float32)).permute(0,3,1,2).to(device)

    return (t, orig_frames, (H, W))
!pip install ipyexperiments
!cp /kaggle/input/decord/install.sh . && chmod  +x install.sh && ./install.sh 
sys.path.insert(0,'/kaggle/working/reader/python')



from decord import VideoReader

from decord import cpu, gpu

from decord.bridge import set_bridge

set_bridge('torch')
retinaface_stats = tensor([123,117,104])[...,None,None].cuda()
### MEMORY LEAK - LET ME KNOW HOW TO MAKE THIS WORK :)

from torch.utils.dlpack import to_dlpack, from_dlpack



def get_decord_video_batch(fname, sz, freq=10):

    "get batch tensor for inference, original for cropping and H,W of video"

    video = VideoReader(str(fname), ctx=gpu())

#     data = video.get_batch(range(0, len(video), 10))

    data = video.get_batch(range(0, len(video), freq))

    H,W = data.shape[2:]

    data = F.interpolate(data.to(torch.float32), (sz,sz))

    data -= retinaface_stats

    del video; gc.collect()

    return (data, None, (H, W))
from ipyexperiments import IPyExperimentsPytorch

from tqdm import tqdm
# %%time

# sz = 640

# for fname in tqdm(train_video_files[:10]):

#     with IPyExperimentsPytorch() as exp: 

#         t, _, (H, W) = get_decord_video_batch(fname, sz)
from torch.utils.dlpack import to_dlpack, from_dlpack



def get_decord_video_batch_cpu(fname, sz, freq=10):

    "get batch tensor for inference, original for cropping and H,W of video"

    video = VideoReader(str(fname), ctx=cpu())

    data = video.get_batch(range(0, len(video), 10)).cuda()

    H,W = data.shape[2:]

    data = F.interpolate(data.to(torch.float32), (sz,sz))

    data -= retinaface_stats

    return (data, None, (H, W))
# %%time

# data, _, (H, W) = get_decord_video_batch(train_video_files[0], 640)
# del data; gc.collect()
!pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali
from nvidia.dali.pipeline import Pipeline

from nvidia.dali import ops
batch_size=1

sequence_length=30

initial_prefetch_size=16



class VideoPipe(Pipeline):

    "video pipeline for a single video with 30 frames"

    def __init__(self, batch_size, num_threads, device_id, data, shuffle):

        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=16)

        self.input = ops.VideoReader(device="gpu", filenames=data, sequence_length=sequence_length,

                                     shard_id=0, num_shards=1,

                                     random_shuffle=shuffle, initial_fill=initial_prefetch_size)

    def define_graph(self):

        output = self.input(name="Reader")

        return output
fname = train_video_files[0]

pipe = VideoPipe(batch_size=batch_size, num_threads=defaults.cpus, device_id=0, data=[fname], shuffle=False)

pipe.build()

pipe_out = pipe.run()

sequences_out = pipe_out[0].as_cpu().as_array()

data = torch.from_numpy(sequences_out[0])
data.shape[1:3]
def dali_batch(fname, sz=640):

    pipe = VideoPipe(batch_size=batch_size, num_threads=defaults.cpus, device_id=0, data=[fname], shuffle=False)

    pipe.build()

    pipe_out = pipe.run()

    sequences_out = pipe_out[0].as_cpu().as_array()

    data = torch.from_numpy(sequences_out[0])

    H,W = data.shape[1:3]

    data = data.permute(0,3,1,2).cuda()

    data = F.interpolate(data.to(torch.float32), (sz,sz))

    data -= retinaface_stats

    return data, _, (H,W)
%%time

# WARM UP

data = dali_batch(train_video_files[0])
%%time

data = dali_batch(train_video_files[0])
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

device = torch.device("cuda")
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
# %%time 

# sz=640

# (t, _, (H, W)) = get_decord_video_batch(train_video_files[0], sz)
# %%time

# out = predict(model, t, sz, cfg_mnet)
def convert_bboxes(bboxes, H, W, sz):

    "rescale to original image sz"

    res = []

    for bb in bboxes:

        h_scale, w_scale = H/sz, W/sz

        orig_bboxes = (bb*array([w_scale, h_scale, w_scale, h_scale])[None, ...]).astype(int)

        res.append(orig_bboxes)

    return res
# %%time

# _= convert_bboxes(out, H, W, sz)
# del t, out

# gc.collect()
from tqdm import tqdm
%%time

sz = 640

all_orig_bboxes = []

for fname in tqdm(train_video_files):

#     t, _, (H, W) = get_video_batch(fname, sz) # ~30 s

#     t, _, (H, W) = get_decord_video_batch_cpu(fname, sz)

#     t, _, (H, W) = get_decord_video_batch(fname, sz)

    t, _, (H, W) = dali_batch(fname, sz)

    bboxes = predict(model, t, sz, cfg_mnet)

    orig_bboxes = convert_bboxes(bboxes, H, W, sz)

    all_orig_bboxes.append(orig_bboxes)

    del t; gc.collect()
len(all_orig_bboxes)
all_orig_bboxes[0]
i = np.random.choice(400)

orig_frames = list(frame_img_generator(train_video_files[i],10))

orig_bboxes = all_orig_bboxes[i]
train_video_files[i]
orig_frames[0].shape
i,orig_bboxes
train_sample_metadata[train_sample_metadata.fname == train_video_files[i].name]
axes = subplots(5,6, figsize=(3*6,3*5)).flatten()

for idx, (ax, _frame, _bb) in enumerate(zip(axes, orig_frames, orig_bboxes)):

    try:

        left, top, right, bottom = _bb[0] # pick first detection for the given frame

        ax.imshow(_frame[top:bottom, left:right, :])

    except: continue # false negatives
from IPython.display import HTML

from base64 import b64encode



def video_url(fname):

    vid1 = open(fname,'rb').read()

    data_url = "data:video/mp4;base64," + b64encode(vid1).decode()

    return data_url



def play_video(fname1, fname2=None): 

    url1 = video_url(fname1)

    url2 = video_url(fname2) if fname2 else None

    if url1 and url2:

        html = HTML(

            """

        <video width=900 controls>

              <source src="%s" type="video/mp4">

        </video>

        <video width=900 controls>

              <source src="%s" type="video/mp4">

        </video>



        """ % (url1, url2))

    else:

        html = HTML(

            """

        <video width=900 controls>

              <source src="%s" type="video/mp4">

        </video>

        """ % (url1))

    return html
# play_video(train_video_files[i])