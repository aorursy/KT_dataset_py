from __future__ import division



import torch 

import torch.nn as nn

import torch.nn.functional as F 

from torch.autograd import Variable

import numpy as np

import cv2

import tensorflow as tf



def unique(tensor):

    tensor_np = tensor.cpu().numpy()

    unique_np = np.unique(tensor_np)

    unique_tensor = torch.from_numpy(unique_np)

    

    tensor_res = tensor.new(unique_tensor.shape)

    tensor_res.copy_(unique_tensor)

    return tensor_res





def bbox_iou(box1, box2):

    """

    Returns the IoU of two bounding boxes 

    

    

    """

    #Get the coordinates of bounding boxes

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]

    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    

    #get the corrdinates of the intersection rectangle

    inter_rect_x1 =  torch.max(b1_x1, b2_x1)

    inter_rect_y1 =  torch.max(b1_y1, b2_y1)

    inter_rect_x2 =  torch.min(b1_x2, b2_x2)

    inter_rect_y2 =  torch.min(b1_y2, b2_y2)

    

    #Intersection area

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)



    #Union Area

    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)

    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    

    iou = inter_area / (b1_area + b2_area - inter_area)

    

    return iou



def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):



    

    batch_size = prediction.size(0)

    stride =  inp_dim // prediction.size(2)

    grid_size = inp_dim // stride

    bbox_attrs = 5 + num_classes

    num_anchors = len(anchors)

    #print("PRED BEF TR ",prediction[:,:,0],prediction[:,:,1],prediction[:,:,4])

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)

    prediction = prediction.transpose(1,2).contiguous()

    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]



    #Sigmoid the  centre_X, centre_Y. and object confidencce

    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])

    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])

    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    

    #Add the center offsets

    grid = np.arange(grid_size)

    a,b = np.meshgrid(grid, grid)



    x_offset = torch.FloatTensor(a).view(-1,1)

    y_offset = torch.FloatTensor(b).view(-1,1)



    if CUDA:

        x_offset = x_offset.cuda()

        y_offset = y_offset.cuda()



    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    pred_old = prediction.clone() #1

    prediction[:,:,:2] += x_y_offset



    #log space transform height and the width

    anchors = torch.FloatTensor(anchors)



    if CUDA:

        anchors = anchors.cuda()

    #pred_old = prediction.clone() #1

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)

    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    pred_old[:,:,2:4] = torch.exp(pred_old[:,:,2:4])*anchors

    

    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    pred_old[:,:,5: 5 + num_classes] = torch.sigmoid((pred_old[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride

    

    return prediction, pred_old



def write_results(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):

    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)

    prediction = prediction * conf_mask

    

    try:

        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()

    except:

        return 0

    box_corner = prediction.new(prediction.shape)

    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)

    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)

    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 

    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)

    prediction[:,:,:4] = box_corner[:,:,:4]

    

    batch_size = prediction.size(0)

    output = prediction.new(1, prediction.size(2) + 1)

    write = False

    



    for ind in range(batch_size):

        image_pred = prediction[ind]          #image Tensor

       #confidence threshholding 

       #NMS

    

        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)

        max_conf = max_conf.float().unsqueeze(1)

        max_conf_score = max_conf_score.float().unsqueeze(1)

        seq = (image_pred[:,:5], max_conf, max_conf_score)

        image_pred = torch.cat(seq, 1)

        

        non_zero_ind =  (torch.nonzero(image_pred[:,4]))

        image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)

        #Get the various classes detected in the image

        try:

            img_classes = unique(image_pred_[:,-1]) # -1 index holds the class index

        except:

            continue    

        for cls in img_classes:

            #perform NMS

            #get the detections with one particular class

            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)

            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()

            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            

            #sort the detections such that the entry with the maximum objectness

            #confidence is at the top

            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]

            image_pred_class = image_pred_class[conf_sort_index]

            idx = image_pred_class.size(0)   #Number of detections

            if nms:

                for i in range(idx):

                  #Get the IOUs of all boxes that come after the one we are looking at 

                  #in the loop

                    try:

                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])

                    except ValueError:

                        break

                    except IndexError:

                        break

            

                #Zero out all the detections that have IoU > treshhold

                    iou_mask = (ious < nms_conf).float().unsqueeze(1)

                    image_pred_class[i+1:] *= iou_mask       

            

                #Remove the non-zero entries

                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()

                    image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

                

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image

            seq = batch_ind, image_pred_class

            

            if not write:

                output = torch.cat(seq,1)

                write = True

            else:

                out = torch.cat(seq,1)

                output = torch.cat((output,out))

    #batch_id,4 coordinates, max_conf, max_conf_score, object id

    return output

    

def letterbox_image(img, inp_dim):

    '''resize image with unchanged aspect ratio using padding'''

    img_w, img_h = img.shape[1], img.shape[0]

    w, h = inp_dim

    new_w = int(img_w * min(w/img_w, h/img_h))

    new_h = int(img_h * min(w/img_w, h/img_h))

    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)

    

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)



    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

    

    return canvas



def prep_image(img, inp_dim):

    """

    Prepare image for inputting to the neural network. 

    

    Returns a Variable 

    """

    orig_im = cv2.imread(img)

    dim = orig_im.shape[1], orig_im.shape[0]

    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))

    img = img[:,:,::-1].transpose((2,0,1)).copy()

    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

    return img, orig_im, dim



from __future__ import division



import torch 

import torch.nn as nn

import torch.nn.functional as F 

from torch.autograd import Variable

import numpy as np

#from util import * 







def get_test_input(img_path,inp_dim):

    img = cv2.imread(img_path)

    img = cv2.resize(img, (416,416))          #Resize to the input dimension

    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 

    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise

    img_ = torch.from_numpy(img_).float()     #Convert to float

    img_ = Variable(img_)                     # Convert to Variable

    return img_



def parse_cfg(cfgfile):

    """

    Takes a configuration file

    

    Returns a list of blocks. Each blocks describes a block in the neural

    network to be built. Block is represented as a dictionary in the list

    

    """

    

    file = open(cfgfile, 'r')

    lines = file.read().split('\n')                        # store the lines in a list

    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 

    lines = [x for x in lines if x[0] != '#']              # get rid of comments

    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces

    

    block = {}

    blocks = []

    

    for line in lines:

        if line[0] == "[":               # This marks the start of a new block

            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.

                blocks.append(block)     # add it the blocks list

                block = {}               # re-init the block

            block["type"] = line[1:-1].rstrip()     

        else:

            key,value = line.split("=") 

            block[key.rstrip()] = value.lstrip()

    blocks.append(block)

    

    return blocks





class EmptyLayer(nn.Module):

    def __init__(self):

        super(EmptyLayer, self).__init__()

        



class DetectionLayer(nn.Module):

    def __init__(self, anchors):

        super(DetectionLayer, self).__init__()

        self.anchors = anchors







def create_modules(blocks):

    net_info = blocks[0]     #Captures the information about the input and pre-processing    

    module_list = nn.ModuleList()

    prev_filters = 3

    output_filters = []

    

    for index, x in enumerate(blocks[1:]):

        module = nn.Sequential()

    

        #check the type of block

        #create a new module for the block

        #append to module_list

        

        #If it's a convolutional layer

        if (x["type"] == "convolutional"):

            #Get the info about the layer

            activation = x["activation"]

            try:

                batch_normalize = int(x["batch_normalize"])

                bias = False

            except:

                batch_normalize = 0

                bias = True

        

            filters= int(x["filters"])

            padding = int(x["pad"])

            kernel_size = int(x["size"])

            stride = int(x["stride"])

        

            if padding:

                pad = (kernel_size - 1) // 2

            else:

                pad = 0

        

            #Add the convolutional layer

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)

            module.add_module("conv_{0}".format(index), conv)

        

            #Add the Batch Norm Layer

            if batch_normalize:

                bn = nn.BatchNorm2d(filters)

                module.add_module("batch_norm_{0}".format(index), bn)

        

            #Check the activation. 

            #It is either Linear or a Leaky ReLU for YOLO

            if activation == "leaky":

                activn = nn.LeakyReLU(0.1, inplace = True)

                module.add_module("leaky_{0}".format(index), activn)

        

            #If it's an upsampling layer

            #We use Bilinear2dUpsampling

        elif (x["type"] == "upsample"):

            stride = int(x["stride"])

            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")

            module.add_module("upsample_{}".format(index), upsample)

                

        #If it is a route layer

        elif (x["type"] == "route"):

            x["layers"] = x["layers"].split(',')

            #Start  of a route

            start = int(x["layers"][0])

            #end, if there exists one.

            try:

                end = int(x["layers"][1])

            except:

                end = 0

            #Positive anotation

            if start > 0: 

                start = start - index

            if end > 0:

                end = end - index

            route = EmptyLayer()

            module.add_module("route_{0}".format(index), route)

            if end < 0:

                filters = output_filters[index + start] + output_filters[index + end]

            else:

                filters= output_filters[index + start]

    

        #shortcut corresponds to skip connection

        elif x["type"] == "shortcut":

            shortcut = EmptyLayer()

            module.add_module("shortcut_{}".format(index), shortcut)

            

        #Yolo is the detection layer

        elif x["type"] == "yolo":

            mask = x["mask"].split(",")

            mask = [int(x) for x in mask]

    

            anchors = x["anchors"].split(",")

            anchors = [int(a) for a in anchors]

            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]

            anchors = [anchors[i] for i in mask]

    

            detection = DetectionLayer(anchors)

            module.add_module("Detection_{}".format(index), detection)

                              

        module_list.append(module)

        prev_filters = filters

        output_filters.append(filters)

        

    return (net_info, module_list)



class Darknet(nn.Module):

    def __init__(self, cfgfile):

        super(Darknet, self).__init__()

        self.blocks = parse_cfg(cfgfile)

        self.net_info, self.module_list = create_modules(self.blocks)

        

    def forward(self, x, CUDA):

        modules = self.blocks[1:]

        outputs = {}   #We cache the outputs for the route layer

        

        write = 0

        for i, module in enumerate(modules):        

            module_type = (module["type"])

            

            if module_type == "convolutional" or module_type == "upsample":

                x = self.module_list[i](x)

    

            elif module_type == "route":

                layers = module["layers"]

                layers = [int(a) for a in layers]

    

                if (layers[0]) > 0:

                    layers[0] = layers[0] - i

    

                if len(layers) == 1:

                    x = outputs[i + (layers[0])]

    

                else:

                    if (layers[1]) > 0:

                        layers[1] = layers[1] - i

    

                    map1 = outputs[i + layers[0]]

                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

                

    

            elif  module_type == "shortcut":

                from_ = int(module["from"])

                x = outputs[i-1] + outputs[i+from_]

    

            elif module_type == 'yolo':        

                anchors = self.module_list[i][0].anchors

                #Get the input dimensions

                inp_dim = int (self.net_info["height"])

        

                #Get the number of classes

                num_classes = int (module["classes"])

        

                #Transform 

                if CUDA:

                    x = x.data.cuda()

                else:

                    x = x.data

                x, pred_old = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                if not write:              #if no collector has been intialised. 

                    detections = x

                    detections1 = pred_old

                    write = 1

        

                else:       

                    detections = torch.cat((detections, x), 1)

                    detections1 = torch.cat((detections1, pred_old), 1)

        

            outputs[i] = x

        

        return detections, detections1





    def load_weights(self, weightfile):

        #Open the weights file

        fp = open(weightfile, "rb")

    

        #The first 5 values are header information 

        # 1. Major version number

        # 2. Minor Version Number

        # 3. Subversion number 

        # 4,5. Images seen by the network (during training)

        header = np.fromfile(fp, dtype = np.int32, count = 5)

        self.header = torch.from_numpy(header)

        self.seen = self.header[3]   

        

        weights = np.fromfile(fp, dtype = np.float32)

        

        ptr = 0

        for i in range(len(self.module_list)):

            module_type = self.blocks[i + 1]["type"]

    

            #If module_type is convolutional load weights

            #Otherwise ignore.

            

            if module_type == "convolutional":

                model = self.module_list[i]

                try:

                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])

                except:

                    batch_normalize = 0

            

                conv = model[0]

                

                

                if (batch_normalize):

                    bn = model[1]

        

                    #Get the number of weights of Batch Norm Layer

                    num_bn_biases = bn.bias.numel()

        

                    #Load the weights

                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])

                    ptr += num_bn_biases

        

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])

                    ptr  += num_bn_biases

        

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])

                    ptr  += num_bn_biases

        

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])

                    ptr  += num_bn_biases

        

                    #Cast the loaded weights into dims of model weights. 

                    bn_biases = bn_biases.view_as(bn.bias.data)

                    bn_weights = bn_weights.view_as(bn.weight.data)

                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)

                    bn_running_var = bn_running_var.view_as(bn.running_var)

        

                    #Copy the data to model

                    bn.bias.data.copy_(bn_biases)

                    bn.weight.data.copy_(bn_weights)

                    bn.running_mean.copy_(bn_running_mean)

                    bn.running_var.copy_(bn_running_var)

                

                else:

                    #Number of biases

                    num_biases = conv.bias.numel()

                

                    #Load the weights

                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])

                    ptr = ptr + num_biases

                

                    #reshape the loaded weights according to the dims of the model weights

                    conv_biases = conv_biases.view_as(conv.bias.data)

                

                    #Finally copy the data

                    conv.bias.data.copy_(conv_biases)

                    

                #Let us load the weights for the Convolutional layers

                num_weights = conv.weight.numel()

                

                #Do the same as above for weights

                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])

                ptr = ptr + num_weights

                

                conv_weights = conv_weights.view_as(conv.weight.data)

                conv.weight.data.copy_(conv_weights)

                   

f = open("../input/class-labels-500/class-descriptions-500.csv","r")

d_class_label = {}

for rec in f:

    line = rec.split(",")

    d_class_label[line[1][:-1].lower()] = line[0]

d_class_label["remote control"] = "/m/0qjjc"

d_class_label["frisbee"] = "/m/0df_n8"

print(d_class_label)
model = Darknet("../input/yolov3cfg/yolov3.cfg")

inp = get_test_input("../input/google-ai-open-images-object-detection-track/test/challenge2018_test/00001a21632de752.jpg",416)

#print(inp)

pred = model(inp, torch.cuda.is_available())

print ("Pred..",pred)

#f = open("ouput.csv","w")

#print (pred)



model = Darknet("../input/yolov3cfg/yolov3.cfg")

model.load_weights("../input/yolov3-weights/yolov3.weights")
from tensorflow.python.keras import backend as K

sess = K.get_session()
from __future__ import division

import time

import torch 

import torch.nn as nn

from torch.autograd import Variable

import numpy as np

import cv2 

#from util import *

import argparse

import os 

import os.path as osp

#from darknet import Darknet

import pickle as pkl

import pandas as pd

import random

from PIL import Image, ImageDraw, ImageFont #sajin

def arg_parse():

    """

    Parse arguements to the detect module

    

    """

    

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

   

    parser.add_argument("--images", dest = 'images', help = 

                        "Image / Directory containing images to perform detection upon",

                        default = "../input/google-ai-open-images-object-detection-track/test/challenge2018_test/00001a21632de752.jpg", type = str)

    parser.add_argument("--det", dest = 'det', help = 

                        "Image / Directory to store detections to",

                        default = "det", type = str)

    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)

    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)

    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)

    parser.add_argument("--cfg", dest = 'cfgfile', help = 

                        "Config file",

                        default = "../input/yolov3cfg/yolov3.cfg", type = str)

    parser.add_argument("--weights", dest = 'weightsfile', help = 

                        "weightsfile",

                        default = "../input/yolov3-weights/yolov3.weights", type = str)

    parser.add_argument("--reso", dest = 'reso', help = 

                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",

                        default = "416", type = str)

    

    return parser.parse_args()



def get_test_input(image, input_dim):

    img = cv2.imread(image)

    img = cv2.resize(img, (input_dim, input_dim)) 

    img_ =  img[:,:,::-1].transpose((2,0,1))

    img_ = img_[np.newaxis,:,:,:]/255.0

    img_ = torch.from_numpy(img_).float()

    img_ = Variable(img_) 

def getmodel(cfgfile, weightsfile):

        #Set up the neural network

    print("Loading network.....")

    model = Darknet(cfgfile)

    model.load_weights(weightsfile)

    print("Network successfully loaded")

    return model

def load_classes(namesfile):

    fp = open(namesfile, "r")

    names = fp.read().split("\n")[:-1]

    f = []

    #names = [x if x=="tvmonitor": "television" else: x for x in names]

    for x in names:

        if x == "tvmonitor":

            f.append("television")

        elif x == "aeroplane":

            f.append("airplane")

        elif x == "pottedplant":

            f.append("houseplant")

        elif x == "cell phone":

            f.append("mobile phone")

        elif x == "cup":

            f.append("coffee cup")

        elif x == "diningtable":

            f.append("kitchen & dining room table")

        elif x == "sofa":

            f.append("sofa bed")

        elif x == "motorbike":

            f.append("motorcycle")

        elif x == "cow":

            f.append("cattle")

        elif x == "microwave":

            f.append("microwave oven")

        elif x == "remote":

            f.append("remote control")

        elif x == "sports ball":

            f.append("ball")

        else:

            f.append(x)

    names = f

    return names    

def detect(model, images_i, bs_i, confidence_i, nms_thresh_i, class_path, weightsfile, cfgfile, reso):

    #args = arg_parse()



    images = images_i

    batch_size = int(bs_i)

    confidence = float(confidence_i)

    nms_thesh = float(nms_thresh_i)

    #scales = scales_i

    start = 0

    CUDA = torch.cuda.is_available()

    print("CUDA:",str(CUDA))

    num_classes = 80

    classes = load_classes(class_path)

    '''

    #Set up the neural network

    print("Loading network.....")

    model = Darknet(cfgfile)

    model.load_weights(weightsfile)

    print("Network successfully loaded")

    '''

    model.net_info["height"] = reso

    inp_dim = int(model.net_info["height"])

    assert inp_dim % 32 == 0 

    assert inp_dim > 32



    #If there's a GPU availible, put the model on GPU

    if CUDA:

        model.cuda()

    #Set the model in evaluation mode

    model.eval()



    read_dir = time.time()

    #Detection phase

    try:

        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]

    except NotADirectoryError:

        imlist = []

        imlist.append(osp.join(osp.realpath('.'), images))

        #print("Not a directory...")

    except FileNotFoundError:

        print ("No file or directory with the name {}".format(images))

        exit()

    

    #if not os.path.exists(args.det):

        #os.makedirs(args.det)



    load_batch = time.time()

    #loaded_ims = [cv2.imread(x) for x in imlist]



    #im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))

    im_batches = [x[0] for x in batches]

    orig_ims = [x[1] for x in batches]

    im_dim_list = [x[2] for x in batches]

    #im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]

    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

    if CUDA:

        im_dim_list = im_dim_list.cuda()



    leftover = 0

    if (len(im_dim_list) % batch_size):

        leftover = 1



    if batch_size != 1:

        num_batches = len(imlist) // batch_size + leftover            

        im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,

                            len(im_batches))]))  for i in range(num_batches)]  



    i=0

    write = False

    #model(get_test_input(inp_dim, CUDA), CUDA)

    start_det_loop = time.time()

    if CUDA:

        im_dim_list = im_dim_list.cuda()

    

    start_det_loop = time.time()

    objs = {}

    

    for batch in im_batches:

    #load the image 

        start = time.time()

        if CUDA:

            batch = batch.cuda()

        with torch.no_grad():

            prediction, pred_old = model(Variable(batch), CUDA)

        #print("PREDICTION:",prediction)

        #prediction_old = prediction[0]

        #prediction = prediction[1]

        prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)

        #pred_old = write_results(pred_old, confidence, num_classes, nms_conf = nms_thesh)

        #print("PREDICTION: ",prediction)

        #print("PREDICTION_OLD: ",pred_old)

        end = time.time()



        if type(prediction) == int:

            i += 1

            continue

        end = time.time()

        prediction[:,0] += i*batch_size



        if not write:                      #If we have't initialised output

            output = prediction  

            write = 1

        else:

            output = torch.cat((output,prediction))

            

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):

            im_id = i*batch_size + im_num

            #print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))

            print("{0:20s} {1:s}".format("Objects Detected:", ""))

            print("----------------------------------------------------------")

            image_id = image.split("/")[-1].split(".")[0]



        i += 1

        if CUDA:

            torch.cuda.synchronize()       

    try:

        output

    except NameError:

        print ("No detections were made")

        return ['']#exit()



    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())



    scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)





    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2

    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2



    output[:,1:5] /= scaling_factor



    for i in range(output.shape[0]):

        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])

        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])

    

    output_recast = time.time()

    class_load = time.time()

    colors = pkl.load(open("../input/pallete/pallete", "rb"))



    draw = time.time()

    sub_str = " " #initialise to a space value

    def write(x, batches, results):

        c1 = tuple(x[1:3].int())

        c2 = tuple(x[3:5].int())

        img = results[int(x[0])]

        cls = int(x[-1])

        color = random.choice(colors)

        label = "{0}".format(classes[cls])

        cv2.rectangle(img, c1, c2,color, 1)

        #sajin

        #d_class_label[classes[cls]]

        height = img.shape[0]

        width = img.shape[1]

        x_min = np.around(c1[0].data.numpy()/width,decimals = 2)

        y_min = np.around(c1[1].data.numpy()/height,decimals = 2)

        x_max = np.around(c2[0].data.numpy()/width,decimals = 2)

        y_max = np.around(c2[1].data.numpy()/height,decimals = 2)

        confi = np.around(x[-3].data.numpy(), decimals = 2)

        #print("iiii..",t,tf.divide(t ,img.shape[0] ) )

        #print("Box ",image_id,c1, c2, classes[cls],d_class_label[classes[cls].lower()],img.shape)

        #print("Box1 ",image_id,d_class_label[classes[cls].lower()],confi,x_min,y_min,x_max,y_max)

        sub_str = d_class_label[classes[cls].lower()]+' '+str(confi)+' '+str(x_min)+' '+str(y_min)+' '+str(x_max)+' '+str(y_max)

        #font = ImageFont.truetype(font='../input/firamonomedium/FiraMono-Medium.otf',size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))

        #t_size = cv2.getTextSize(label, font, 2 , 1)[0]

        #end

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1 , 1)[0] # 1 to 2 Sajin

        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4

        cv2.rectangle(img, c1, c2,color, -1)

        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, [225,255,255], 1);

        #cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), font, 1, [225,255,255], 1); #sajin

        return sub_str





    sub_str =  list(map(lambda x: write(x, im_batches, orig_ims), output))

    #print("Final...",sub_str)

    #det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))

    #print("Det names: ",det_names)

    det_names = "output_image.png"

    #list(map(cv2.imwrite, det_names, loaded_ims))

    #list(map(cv2.imwrite, "im.png", loaded_ims))

    #list(map(cv2.imshow, det_names, loaded_ims))

    #cv2.waitKey(0)

    #cv2.destroyAllWindows()

    #

    '''

    import matplotlib.pyplot as plt

    for img in orig_ims:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print("Img width:",img.shape[0])

        #plt.xticks(np.arange(img.shape[0], img.shape[0]+1, 100))

        plt.figure(figsize=(12,12))

        plt.imshow(img)

        plt.show()

    

    end = time.time()



    print("SUMMARY")

    print("----------------------------------------------------------")

    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))

    print()

    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))

    print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))

    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))

    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))

    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))

    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))

    print("----------------------------------------------------------")

    '''



    torch.cuda.empty_cache()

    return sub_str
coco_classes = "../input/coco-classesv2/coco_classesv2.txt"

weightsfile = "../input/yolov3-weights/yolov3.weights"

cfgfile = "../input/yolov3cfg/yolov3.cfg"

img_path = "../input/google-ai-open-images-object-detection-track/test/challenge2018_test/00000b4dcff7f799.jpg"

#00001a21632de752.jpg"

'''

batches = 1

model = getmodel(cfgfile, weightsfile)

sub_str1 = detect(model,img_path,batches,0.5,0.4, coco_classes, weightsfile, cfgfile, "416")

print(sub_str1)

'''



#model = Darknet("../input/yolov3cfg/yolov3.cfg")

#inp = get_test_input("../input/google-ai-open-images-object-detection-track/test/challenge2018_test/00001a21632de752.jpg",416)

#print(inp)

#pred = model(inp, torch.cuda.is_available())

#print (pred)

#print (pred.shape)
import torch

print(torch.__version__)

from keras.preprocessing.image import ImageDataGenerator

import math, os

image_path = "../input/google-ai-open-images-object-detection-track/test/"



batch_size = 1

img_generator = ImageDataGenerator().flow_from_directory(image_path, shuffle=False, batch_size = batch_size)

n_rounds = math.ceil(img_generator.samples / img_generator.batch_size)

filenames = img_generator.filenames

print(len(filenames))

#print(filenames)
batches = 1

confidence = 0.5

nms_thresh = 0.4

model = getmodel(cfgfile, weightsfile)

f = open("od_sub1.csv","w")

f.write('ImageId'+','+'PredictionString'+'\n')

det_cnt = 0

for i in range(len(filenames)):

    image = filenames[i]

    print(image)

    sub_str1 = detect(model, image_path + image,batches, confidence, nms_thresh, coco_classes, weightsfile, cfgfile, 416)

    #print(image.split('/')[1]+','+' '.join(sub_str1))

    f.write(image.split('/')[1]+','+' '.join(sub_str1)+'\n')

    if len(sub_str1) > 1:

        det_cnt = det_cnt + 1

    if i == 10:

        print("i is :",i,det_cnt)

        print("% of detection ",det_cnt * 100/i)

        break

print('Closing file....')

f.close()