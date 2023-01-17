# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable as V

from torchvision.models import vgg19

import torchvision.transforms as transforms

import torchvision.transforms.functional as F

import numpy as np

from PIL import Image
torch.cuda.set_device(0)



vgg = vgg19(True).eval().cuda()


preprocess = transforms.Compose([transforms.Resize(224),

                                         transforms.ToTensor(),

                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                                              std=[0.229, 0.224, 0.225])])



im_a = Image.open("../input/dog1.jpg")

im_a_tens = V(preprocess(im_a).unsqueeze(0)).cuda()

out = vgg(im_a_tens)

print('Prediction for cat.jpg is: %s'%(np.argmax(out.cpu().detach().numpy())))



im_b = Image.open("../input/dog2.jpg")

im_b_tens = V(preprocess(im_b).unsqueeze(0)).cuda()

out = vgg(im_b_tens)

print('Prediction for dog.jpg is: %s'%(np.argmax(out.cpu().detach().numpy())))
print(vgg)
feature_pyramid = []

def extract_feature(self, input, output):

    feature_pyramid.append(output)



layer_inds = [3, 8, 17, 26, 35]



for layer_ind in layer_inds:

    vgg.features[layer_ind].register_forward_hook(extract_feature)
feature_pyramid = []

vgg(im_a_tens)

vgg(im_b_tens)

feat_a = feature_pyramid[:5]

feat_b = feature_pyramid[5:]
# TODO: need mean and standard deviations across each channel instead of everything



class CommonAppearance(torch.nn.Module):

    def __init__(self):

        super(CommonAppearance, self).__init__()

        self.train(False)

        

    def forward(self, P, Q):

        P = P.squeeze()

        Q = Q.squeeze()

        mu_a = P.mean(2).mean(1)

        mu_b = Q.mean(2).mean(1)

        mu_m = (mu_a + mu_b) / 2

        sig_a = P.std(2).std(1)

        sig_b = Q.std(2).std(1)

        sig_m = (sig_a + sig_b) / 2

        P_common = ((P.permute(1,2,0) - mu_a) / sig_a * sig_m + mu_m).permute(2,0,1)

        Q_common = ((Q.permute(1,2,0) - mu_b) / sig_b * sig_m + mu_m).permute(2,0,1)        

        return P_common, Q_common



commonAppearance = CommonAppearance()
im_a_common, im_b_common = commonAppearance(im_a_tens, im_b_tens)



F.to_pil_image(im_a_common.cpu())
F.to_pil_image(im_b_common.cpu())
class NeuralBestBuddies(torch.nn.Module):

    def __init__(self):

        super(NeuralBestBuddies, self).__init__()

        self.train(False)

        

    def _get_neighborhood(self, P, i, j, neigh_rad):

        # 2

        P = P.permute(1, 2, 0)

        P_padded = torch.zeros((P.size()[0] + 2 * neigh_rad, P.size()[1] + 2 * neigh_rad, P.size()[2]))

        P_padded[neigh_rad: -neigh_rad, neigh_rad: -neigh_rad] = P

        return P_padded[i: i + 2 * neigh_rad + 1, j: j + 2 * neigh_rad + 1].permute(2, 0, 1)

    

    def forward(self, Ps, Qs, neigh_rad, gamma=0.05):

        """

        args:

            P: 4D tensor of features in NCHW format

            Q: 4D tensor of features in NCHW format

            neigh_rad: int representing amount of surrounding neighbors to include in cross correlation.

                       so neigh_rad of 1 takes cross correlation of 3x3 patches of neurons

            gamma: (optional) activation threshold

        output:

            NBB pairs

        """

        

        height = Ps.size()[2]

        width = Ps.size()[3]

        n_channels = Ps.size()[1]

        

        best_buddies = []

        

        for P, Q in zip(Ps, Qs):

            #2

            P_L2 = P.clone().permute(1,2,0).norm(2, 2)

            Q_L2 = Q.clone().permute(1,2,0).norm(2, 2)

            

            P_over_L2 = P.div(P_L2)

            Q_over_L2 = Q.div(Q_L2)

                                    

            P_nearest = []

            Q_nearest = []

            for i in range(0, height):

                for j in range(0, width):

                    p_neigh = self._get_neighborhood(P_over_L2, i, j, neigh_rad)

                    # 1

                    conv = torch.nn.Conv2d(n_channels, 1, neigh_rad * 2 + 1, padding=neigh_rad).cuda()

                    conv.train(False)

                    conv.weight.data.copy_(p_neigh.unsqueeze(0))

                    p_cross_corrs = conv(Q_over_L2.unsqueeze(0)).squeeze().view(-1)

                    # 4

                    P_nearest.append(p_cross_corrs.argmax())

                    

                    q_neigh = self._get_neighborhood(Q_over_L2, i, j, neigh_rad)

                    conv = torch.nn.Conv2d(n_channels, 1, neigh_rad * 2 + 1, padding=neigh_rad).cuda()

                    conv.train(False)

                    conv.weight.data.copy_(q_neigh.unsqueeze(0))

                    q_cross_corrs = conv(P_over_L2.unsqueeze(0)).squeeze().view(-1)

                    Q_nearest.append(q_cross_corrs.argmax())

            

            # 5

            P_L2_min = P_L2.min()

            P_L2_max = P_L2.max()

            P_normalized = (P_L2.view(-1) - P_L2_min) / (P_L2_max - P_L2_min)

            

            Q_L2_min = Q_L2.min()

            Q_L2_max = Q_L2.max()

            Q_normalized = (Q_L2.view(-1) - Q_L2_min) / (Q_L2_max - Q_L2_min)

            

            for i in range(len(P_nearest)):

                if(i == Q_nearest[P_nearest[i]] and P_normalized[i] > gamma and Q_normalized[P_nearest[i]] > gamma):

                    best_buddies.append((i, int(P_nearest[i])))

                    

        return best_buddies
nbb = NeuralBestBuddies()



lambda_5 = nbb(feat_a[4], feat_b[4], 1)

print(*lambda_5)
import matplotlib.pyplot as plt

import matplotlib.ticker as plticker

from matplotlib import colors as mcolors

from random import shuffle



colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

shuffle(colors)



def plot_with_grid(subplt, img, n_cells, neigh_rad, nbbs, a_or_b, my_dpi=60):

    ax=plt.subplot(*subplt)

    grid_width = im_a.size[0] / n_cells

    loc = plticker.MultipleLocator(base=grid_width)

    ax.xaxis.set_major_locator(loc)

    ax.yaxis.set_major_locator(loc)

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.grid(which='major', axis='both', linestyle='-', color='lightgrey')

    ax.imshow(img)

    nbb_index = 0 if a_or_b == 'a' else 1

    for index, coords in enumerate(nbbs):

        j=grid_width/2+(coords[nbb_index]//(n_cells))*grid_width

        i=grid_width/2+(coords[nbb_index]%(n_cells))*grid_width

        ax.add_artist(plt.Circle((i, j), 3, color=colors[index], alpha=0.9))

    

plot_with_grid((1,2,1), im_a, feat_a[4].size()[2], 1, lambda_5, 'a')

plot_with_grid((1,2,2), im_b, feat_a[4].size()[2], 1, lambda_5, 'b')
class RefineSearchWindow(torch.nn.Module):

    def __init__(self):

        super(RefineSearchWindow, self).__init__()

        self.train(False)

        

    def forward(self, P, Q, prev_nbbs, row_size, rec_rad):

        P = P[0].permute(1,2,0)

        Q = Q[0].permute(1,2,0)



        new_Ps = []

        new_Qs = []

        P_coords = []

        Q_coords = []

        for p_coord, q_coord in prev_nbbs:

            p_center = (p_coord // row_size, p_coord % row_size)

            q_center = (q_coord // row_size, q_coord % row_size)

            p_x1 = max(int(2 * p_center[0] - rec_rad / 2), 0)

            p_x2 = min(int(2 * p_center[0] + rec_rad / 2), P.size()[1])

            p_y1 = max(int(2 * p_center[1] - rec_rad / 2), 0)

            p_y2 = min(int(2 * p_center[1] + rec_rad / 2), P.size()[0])

            q_x1 = max(int(2 * q_center[0] - rec_rad / 2), 0)

            q_x2 = min(int(2 * q_center[0] + rec_rad / 2), Q.size()[1])

            q_y1 = max(int(2 * q_center[1] - rec_rad / 2), 0)

            q_y2 = min(int(2 * q_center[1] + rec_rad / 2), Q.size()[0])

            new_Ps.append(P[p_x1: p_x2, p_y1: p_y2])

            new_Qs.append(Q[q_x1: q_x2, q_y1: q_y2])

            P_coords.append((p_x1, p_y1))

            Q_coords.append((q_x1, q_y1))

        return (new_Ps, new_Qs, P_coords, Q_coords)
refineSearchWindow = RefineSearchWindow()

Ps_4, Qs_4, P_coords, Q_coords = refineSearchWindow(feat_a[3], feat_b[3], lambda_5, feat_a[4].size()[3], 6)