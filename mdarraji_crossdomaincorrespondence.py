import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable as V

import torchvision.models as models

import torchvision.transforms as transforms

import torchvision.transforms.functional as F

import tensorflow as tf

import numpy as np

from PIL import Image

from matplotlib import pyplot as plt

import math



from torch.autograd import Variable

# Neuron class, takes in row and col coordinates

class Neuron:

    def __init__(self, row, col):

        self.r = row

        self.c = col

    def __repr__(self):

        return "(" + str(self.r) + ", " + str(self.c) + ")"

# returns the index of the max arg of tensor

# function 2

def feat_arg_max(feat):

    f = feat.clone().detach().numpy()

    idx = np.unravel_index(f.argmax(), f.shape)

    return idx



# function to do L2 normalization of a tensor

def L2_norm(A_tensor):

    A = A_tensor.clone().detach().numpy()

    pow_sum = np.power(A, 2).sum()

    A_sqrt = np.power(pow_sum, 0.5)

    return torch.from_numpy(A / A_sqrt)



# input, feature tensors of the new regions from P and Q

# returns 2 lists of touples, one for P and one for Q

# each touple is a neuron p, with is corresponding NN q

# p_list, and q_list is a list of touples where each touple contains touples of coordinates [((x1, y1), (x2, y2))]

def NBB(C_A, C_B, R, neighbor_size):

    

    P_region = R[0]

    Q_region = R[1]



    # iterate through the p's in common_p_q to find its neighbors in Q, (p, q)

    # pass in list of neurons [p1, p2, q1, q2]

    # p1 - bottom left, p2 bottom right, same for q

    qs_for_ps = nearest_neighbor(C_A, C_B, P_region, neighbor_size)

    # iterate through the q's in common_q_p to find its neighbors in P, (q, p)

    ps_for_qs = nearest_neighbor(C_B, C_A, Q_region, neighbor_size)

    # returns in (p, q) format

    # get the candidates that are nearest neighbors to each other

    candidates = get_candidates(qs_for_ps, ps_for_qs)

    # check the activations and find the most meaningful buddies

    # must return in format p[], q[]



    feat_a_norm = normalize_feature_map(C_A)

    feat_b_norm = normalize_feature_map(C_B)



    return meaningful_NBBs(feat_a_norm, feat_b_norm, candidates, .05)



def normalize_feature_map(feat_map):

    """

    Assigns each neuron a value in the range [0, 1] to the

    given feature map

    Args:

        feat_map: feature map tensor

    Returns:

        norm_feat_map: normalized feature map

    """



    feat_min = feat_map.min()

    feat_max = feat_map.max()



    feat_map_norm = (feat_map - feat_min) / (feat_max - feat_min)



    return feat_map_norm



def meaningful_NBBs(C_A, C_B, candidates, act_threshold):

    """

    Use normalized activation maps to seek NNBS which have high activation

    values

    Args:

        feat_a: feature map tensor for image a

        feat_b: feature map tensor for image b

        candiates: list of neural best buddies candiates

        act_threshold: empirically determined activation threshold

    Returns:

        meanigful_buddies: list of neural best buddes with high activation

            values

    """

    feat_a = C_A.clone().squeeze().permute(1, 2, 0)

    feat_b = C_B.clone().squeeze().permute(1, 2, 0)

    print(feat_a.shape)

    print(feat_b.shape)



    num_candidate_pairs = len(candidates)



    meaningful_buddies = []



    for i in range (num_candidate_pairs):



        p_coords = candidates[i][0]

        p_max_activation_indx = feat_arg_max(feat_a[p_coords.r, p_coords.c, :])

        p_max_activation = feat_a[p_coords.r][p_coords.c][p_max_activation_indx]



        

        print("p idx: ", p_max_activation_indx)

        print("p ac: ", p_max_activation)

        

        q_coords = candidates[i][1]

        q_max_activation_indx = feat_arg_max(feat_b[q_coords.r, q_coords.c, :])

        q_max_activation = feat_b[q_coords.r][q_coords.c][q_max_activation_indx]

        

       

        print("q idx: ", q_max_activation_indx)

        print("q ac: ",q_max_activation)



        if (q_max_activation > act_threshold and p_max_activation > act_threshold):

            meaningful_buddies.append(candidates[i])



    return meaningful_buddies





# returns a list of candidates which are nearest neighbors: [(p,q)]

def get_candidates(P_nearest, Q_nearest):

    # to reverse contents in the tuple, facilitates comparison

    # iterates through list of all pairs

    candidates = []

    pq_size = int(math.sqrt(len(P_nearest)))

    for i in range(len(P_nearest)):

        if(i == Q_nearest[P_nearest[i]]):

            p_c = math.floor(1.0 * i / pq_size)

            p_r = i - (p_c * pq_size)

            p = Neuron(p_r, p_c)



            j = P_nearest[i]

            q_c = math.floor(1.0 * j / pq_size)

            q_r = j - (q_c * pq_size)

            q = Neuron(int(q_r), q_c)



            candidates.append([p, q])



    return candidates



# neighborhood function for P

def neighborhood(P_over_L2, p_i, p_j, neigh_size):



    neigh_rad = int((neigh_size - 1) / 2)

    P = P_over_L2.clone().permute(1, 2, 0)



    P_padded = torch.zeros((P.size()[0] + 2 * neigh_rad, P.size()[1] + 2 * neigh_rad, P.size()[2]))



    P_padded[neigh_rad: -neigh_rad, neigh_rad: -neigh_rad] = P

    P_padded = P_padded[p_i: p_i + 2 * neigh_rad + 1, p_j: p_j + 2 * neigh_rad + 1].permute(2, 0, 1)



    return P_padded



# takes in P and Q tensors, and the neighborhood size(5 or 3)

# returns list of nearest neighbors of P

def nearest_neighbor(P_tensor, Q_tensor, P_region, neigh_size):

  

    # info from P tensor

    num_chan = P_tensor.shape[1]

    img_w = P_tensor.shape[2]

    img_h = P_tensor.shape[3]



    # list of nearest neighbors of P

    nearest_buddies = []



    # L2 of P and Q

    P = P_tensor.clone().squeeze()

    P_L2 = P.permute(1, 2, 0).norm(2, 2)



    Q = Q_tensor.clone().squeeze()

    Q_L2 = Q.permute(1, 2 ,0).norm(2, 2)



    # similarity metric

    P_over_L2 = P.div(P_L2)

    Q_over_L2 = Q.div(Q_L2)

    #print( P_over_L2.shape)



    neigh_rad = int((neigh_size - 1) / 2)



    for region in P_region: 

        # region points to calculate 

        top_left_p = P_region[0]

        bottom_right_p = P_region[1]

        

        for p_i in range(top_left_p.r, bottom_right_p.r):

            for p_j in range(top_left_p.c, bottom_right_p.c):

                conv = torch.nn.Conv2d(num_chan, 1, neigh_size, padding=neigh_rad)

                conv.train(False)



                p_neigh = neighborhood(P_over_L2, p_i, p_j, neigh_size)

                #print(" ", p_neigh.shape, neigh_size)

                conv.weight.data.copy_(p_neigh.unsqueeze(0))



                p_cross_corrs = conv(Q_over_L2.unsqueeze(0)).clone().squeeze().view(-1)

                #p_cross_corrs = conv(Q_over_L2.unsqueeze(0)).squeeze().detach().numpy()

                #q_idx = np.unravel_index(p_cross_corrs.argmax(), p_cross_corrs.shape)

                #p = Neuron(p_i, p_j)

                #q = Neuron(q_idx[0], q_idx[1])

                #nearest_buddies.append(q_idx)

                nearest_buddies.append(p_cross_corrs.argmax())

                

    return nearest_buddies



# P and Q should be feature maps for a given layer

# returns the common appearance C(P, Q)

def common_appearance(P, Q, region_p_list, region_q_list):

    # copy of the whole P - going to put common_app in specific region on this later

    p_to_q = P.clone()

    # changed to - [chann, height, width]

    P_copy = P.clone().squeeze()

    Q_copy = Q.clone().squeeze()

    

    for ind in range(len(region_p_list)):

        region_p = region_p_list[ind]

        region_q = region_q_list[ind]



        top_left_p = region_p[0]

        bottom_right_p = region_p[1]

        top_left_q = region_q[0]

        bottom_right_q = region_q[1]



        

        print(P_copy.shape)

        print(Q_copy.shape)

        print("tl p: ", top_left_p)

        print("tl: q: ", top_left_q)

        print("br p: ", bottom_right_p)

        print("br q: ", bottom_right_q)

        # these only represent P, Q in the region (AKA trimmed P, and Q)

        P_copy_reg = P_copy[:, int(top_left_p.r):int(bottom_right_p.r), int(top_left_p.c):int(bottom_right_p.c)]

        Q_copy_reg = Q_copy[:, int(top_left_q.r):int(bottom_right_q.r), int(top_left_q.c):int(bottom_right_q.c)]



        # have to squeeze to remove first dimension: [C, H, W]

        mean_p = P_copy_reg.mean(2).mean(1)

        mean_q = Q_copy_reg.mean(2).mean(1)

        mean_m = (mean_p + mean_q) / 2

        sig_p = P_copy_reg.std(2).std(1)

        sig_q = Q_copy_reg.std(2).std(1)

        sig_m = (sig_p + sig_q) / 2

        # have to permute, in order to be able to subtract the mean correctly

        temp = (P_copy_reg.permute(1,2,0) - mean_p)

        # common_app should be the size of the region we are doing style transfer on

        common_app = (temp/ sig_p * sig_m + mean_m).permute(2,0,1)



        p_to_q[:, :, top_left_p.r:bottom_right_p.r, top_left_p.c:bottom_right_p.c] = common_app

        

    return p_to_q



def refine_search_regions(prev_layer_nbbs, receptive_field_radius, feat_width, feat_height):

    """

    Return refined search regions for every p and q in the previous' layer nbbs

    Args:

        prev_layer_nbbs: Previous' layer (l-1) neural best buddies, represented

            as neurons using the Neuron class

        receptive_field_radius: radius of new search regions

            equal to 4 for l = 2,3 and equal to 6 for l = 4, 5

        feat_width: width of feature map for current layer

        feat_height: height of feature map for current layer

    Returns:

        Ps: List containing new P's

            P = ((r1, c1), (r2, c2))

            where (r1, c1) represent the top left of the search region

            and (r2, c2) represent the bottom right of the search region

        Qs: List containing new Q's

            Q = ((r1, c1), (r2, c2))

            where (r1, c1) represent the top left of the search region

            and (r2, c2) represent the bottom right of the search region

    """



    Ps = []

    Qs = []



    for p, q in prev_layer_nbbs:



        # Top left of search window for P

        P_r1 = max(int(2 * p.r - receptive_field_radius / 2), 0)

        P_c1 = max(int(2 * p.c - receptive_field_radius / 2), 0)

        P_bottom_left = Neuron(P_r1, P_c1)



        # Bottom right of search window for P

        P_r2 = min(int(2 * p.r + receptive_field_radius / 2), feat_width - 1)

        P_c2 = min(int(2 * p.c + receptive_field_radius / 2), feat_height - 1)

        P_top_right = Neuron(P_r2, P_c2)



        # Top left of search window for Q

        Q_r1 = max(int(2 * q.r - receptive_field_radius / 2), 0)

        Q_c1 = max(int(2 * q.c - receptive_field_radius / 2), 0)

        Q_bottom_left = Neuron(Q_c1, Q_r1)



        # Bottom right of search window for Q

        Q_r2 = min(int(2 * q.r + receptive_field_radius / 2), feat_width - 1)

        Q_c2 = min(int(2 * q.c + receptive_field_radius / 2), feat_height - 1)

        Q_top_right = Neuron(Q_c2, Q_r2)



        # Append P and Q to lists

        Ps.append((P_bottom_left, P_top_right))

        Qs.append((Q_bottom_left, Q_top_right))



    return (Ps, Qs)



# Image preprocessing

def img_preprocess_VGG(img):

    # VGGNet was trained on ImageNet where images are normalized by mean=[0.485, 0.456, 0.406]

    # and std=[0.229, 0.224, 0.225].

    # We use the same normalization statistics here.

    transform = transforms.Compose([

        transforms.Resize(224),

        transforms.ToTensor(),

        transforms.Normalize(mean=(0.485, 0.456, 0.406),

                             std=(0.229, 0.224, 0.225))])



    img_preproc = transform(img)

    img_preproc = torch.unsqueeze(img_preproc, 0)

    img_tens = V(img_preproc)



    return img_tens



def image_preprocess_resnet(img):

    # now lets do the same for resnet_18

    to_tensor = transforms.ToTensor()

    scaler = transforms.Scale((224, 224))

    normalize = transforms.Compose([

        transforms.ToPILImage(),

        transforms.Resize(224),

        transforms.ToTensor(),

        transforms.Normalize(mean=(0.485, 0.456, 0.406),

                             std=(0.229, 0.224, 0.225))])

    img_tens = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

    return img_tens



def image_preprocess_alexnet(img):

    to_tensor = transforms.ToTensor()

    scaler = transforms.Scale((299, 299))

    normalize = transforms.Compose([

        transforms.ToPILImage(),

        transforms.Resize(299),

        transforms.ToTensor(),

        transforms.Normalize(mean=(0.485, 0.456, 0.406),

                             std=(0.229, 0.224, 0.225))])

    img_tens = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

    return img_tens



# vgg model

# takes in image a, image b, normalized tensor of image a, normalized tensor of img b

# returns 5 layered feature map of img a and img b

def vgg19_model(img_a, img_b, img_a_tens, img_b_tens):

    print("**********************************V G G 1 9**********************************")

    print("img_a size: ", img_a.size)

    print("img a t-size: ", tf.shape(img_a_tens))

    model = models.vgg19(pretrained=True).eval()

    pyramid_layers = []



    def extract_feature(module, input, output):

        pyramid_layers.append(output)



    relu_idx = [3, 8, 17, 26, 35]

    #relu_idx = [35, 26, 17, 8, 3]

    print("vgg19: ", model.features[0] )

    for j in relu_idx:

        model.features[j].register_forward_hook(extract_feature)



    model(img_a_tens)

    model(img_b_tens)

    for layer in pyramid_layers:

        print("ith layer @ relu: ", layer.size())

        

    pyramid_layers.reverse()

    return pyramid_layers[:5], pyramid_layers[5:]



def resnet_18(img_a, img_b, img_a_tens, img_b_tens):

    print("**********************************R E S N E T 18**********************************")

    print("img_a size: ", img_a.size)

    print("img a t-size: ", tf.shape(img_a_tens))

    model = models.resnet18(pretrained=True).eval()

    # bb = list(model.layer1.children())[1] used to index into the right block of the layer, then add a .relu to get the relu

    layer_list = [list(model.layer1.children())[1].relu, list(model.layer2.children())[1].relu, list(model.layer3.children())[1].relu, list(model.layer4.children())[1].relu]

    pyramid_layers = []

    def extract_feature(module, input, output):

        pyramid_layers.append(output)

    # Attach that function to our selected layers

    for layer in layer_list:

        # print("layer type:", type(layer))

        layer.register_forward_hook(extract_feature)



    # Run the model on our transformed image

    model(img_a_tens)

    model(img_b_tens)

    # remove duplicates..... (not sure why we had them in the first place.)

    pyramid_layers = [ pyramid_layers[idx] for idx in range(0, len(pyramid_layers), 2)]

    # Return the feature vector

    for layer in pyramid_layers: # debug, check layers

        print("ith layer @ relu: ", layer.size())

    # print("first layer:",pyramid_layers[7].size())

    # print("2th layer:",pyramid_layers[4].size())

    return pyramid_layers[:4], pyramid_layers[4:]



def alexnet(img_a, img_b, img_a_tens, img_b_tens):

    print("**********************************A L E X N E T**********************************")

    print(type(img_a_tens) )

    model = models.alexnet(pretrained=True).eval()

    print(model)

    pyramid_layers = []

    layer_list = [model.features[1], model.features[4], model.features[7], model.features[9], model.features[11]]

    def extract_feature(module, input, output):

        pyramid_layers.append(output)



    for layer in layer_list:

        # print("layer type:", type(layer))

        layer.register_forward_hook(extract_feature)



    model(img_a_tens)

    model(img_b_tens)



    for layer in pyramid_layers: # debug, check layers

        print("ith layer @ relu: ", layer.size())

    return pyramid_layers[:5], pyramid_layers[5:]



def plot_neurons(n_list, i, img):

    # Mock data

    # n1 = Neuron(1, 2)

    # n2 = Neuron(150, 150)

    # n3 = Neuron(10, -2)

    # n4 = Neuron(-1, -3)

    # n_list = [[n1, n2], [n3, n4]]

    img_plot = plt.imshow(img)

    

    for pair in n_list:

        neuron = pair[i]

        print("plotting", neuron.r, neuron.c)

        # figure(1)

        # plt.scatter(neuron.r, neuron.c)

        # figure(2)

        plt.scatter(neuron.r, neuron.c)



def main():

    img_a = Image.open("../input/dog1.jpg")

    img_b = Image.open("../input/dog2.jpg")

    img_a_tens = img_preprocess_VGG(img_a)

    img_b_tens = img_preprocess_VGG(img_b)



    feat_a_19, feat_b_19 = vgg19_model(img_a, img_b, img_a_tens, img_b_tens)



    # print("vgg 19 types:", type(feat_a_19))



    # img_a_tens = image_preprocess_resnet(img_a)

    # img_b_tens = image_preprocess_resnet(img_b)

    # feat_a_18, feat_b_18 = resnet_18(img_a, img_b, img_a_tens, img_b_tens)

    # print(feat_a_18[0] )

    # img_a_tens = image_preprocess_alexnet(img_a)

    # img_b_tens = image_preprocess_alexnet(img_b)

    # feat_a_v3, feat_b_v3 = alexnet(img_a, img_b, img_a_tens, img_b_tens)

    

    receptive_field_rs = [4, 4, 6, 6]

    neigh_sizes = [5, 5, 5, 3, 3]

    C_A = feat_a_19[4]

    C_B = feat_b_19[4]

    

    top_left_p = Neuron(0, 0)

    bottom_right_p = Neuron(C_A.shape[2] - 1, C_A.shape[2] - 1)

    

    top_left_q = Neuron(0, 0)

    bottom_right_q = Neuron(C_B.shape[2] - 1, C_B.shape[2] - 1)

    

    R = [(top_left_p, bottom_right_p), (top_left_q, bottom_right_q)]

    nbbs = None

   

    for l in range (4, 0, -1):

        

        print ("------ Layer ", l + 1, " ------")



        feat_a = feat_a_19[l]

        feat_b = feat_b_19[l]



        nbbs = NBB(C_A, C_B, R, neigh_sizes[l])

        print(nbbs)



        if l > 1:



            feat_width = feat_a.shape[2]

            feat_height = feat_b.shape[3]

            R = refine_search_regions(nbbs, receptive_field_rs[l - 1], feat_width, feat_height)



            C_A = common_appearance(feat_a, feat_b, R[0], R[1])

            C_B = common_appearance(feat_b, feat_a, R[1], R[0])

    

    plt.figure(1)

    plot_neurons(nbbs, 0, img_a)

    plt.figure(2)

    plot_neurons(nbbs, 1, img_b)

    plt.show()

main()