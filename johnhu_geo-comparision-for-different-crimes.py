# -*- coding: utf-8 -*-
import os

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import statsmodels.nonparametric.api as smnp
import seaborn as sns


%matplotlib inline  
matplotlib.rcParams['figure.figsize'] = (15, 25)

sns.despine(fig=None, left=False, right=False, top=False, bottom=False, trim=True)
lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]

def parse_timeInfo(pd_input, colname="Dates"):
    l_y = []
    l_m = []
    l_d = []
    l_h = []
    for i in range(pd_input.shape[0]):
        dt = pd_input[colname][i]
        l_y.append(pd.Timestamp(dt).year)
        l_m.append(pd.Timestamp(dt).month)
        l_d.append(pd.Timestamp(dt).day)
        l_h.append(pd.Timestamp(dt).hour)
    
    pd_input['Year'] = l_y
    pd_input['Month'] = l_m
    pd_input['Day'] = l_d
    pd_input['Hour'] = l_h


def num_l_word(l_word, M_wordDict=None):
    if M_wordDict is None:
        l_word_unique = sorted(list(set(l_word)))
        M_wordDict = dict(zip(l_word_unique, range(len(l_word_unique))))
    l_word_idx = [ M_wordDict[w] for w in l_word ]
    return l_word_idx, M_wordDict

def Df_wordParseToNum(pd_input, colname, M_wordDict=None):
    l_word = list(pd_input[colname])
    l_word_idx, M_wordDict = num_l_word(l_word, M_wordDict)
    pd_input[colname] = l_word_idx
    return M_wordDict

def kde_support(data, bw, gridsize, cut, clip):
    support_min = max(data.min() - bw * cut, clip[0])
    support_max = min(data.max() + bw * cut, clip[1])
    return np.linspace(support_min, support_max, gridsize)

def smnp_kde(pd_input, cut, gridsize, clipsize, bw="scott"):
    bw_func = getattr(smnp.bandwidths, "bw_" + bw)
    x_bw = bw_func(pd_input["X"].values)
    y_bw = bw_func(pd_input["Y"].values)
    bw = [x_bw, y_bw]
    kde = smnp.KDEMultivariate( pd_input.T.values, "cc", bw)
    x_support = kde_support(pd_input['X'].values, x_bw, gridsize, cut, clipsize[0])
    y_support = kde_support(pd_input['Y'].values, y_bw, gridsize, cut, clipsize[1])
    
    xx, yy = np.meshgrid(x_support, y_support)
    Z = kde.pdf([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, Z


def get_heatmap(pd_train, clipsize, category):
    l_colNameUsed = [ 'X', 'Y']
    pd_train_used = pd_train[pd_train['Category']==category][ l_colNameUsed ]
    cut = 10
    gridsize = 100
    xx, yy, Z = smnp_kde(pd_train_used, cut=cut, gridsize=gridsize, clipsize=clipsize)
    return xx, yy, Z


def remove_axis(ax):
    ax.get_xaxis().set_ticks( [] )
    ax.get_xaxis().set_ticklabels( [] )
    ax.get_yaxis().set_ticks( [] )
    ax.get_yaxis().set_ticklabels( [] )

    
def plot_one_heatmap(xx, yy, Z, clipsize, png_name):
    mapdata = np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
    up_max = np.percentile(Z, 99)
    Z[Z > up_max] = up_max
    cut = 10
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(1,1,1)
    ax1.imshow(mapdata,extent=lon_lat_box, cmap=plt.get_cmap('gray'))
    ax1.contourf(xx, yy, Z, cut, cmap="jet", shade=True, alpha=0.5).collections[0].set_alpha(0)
    remove_axis(ax1)
    name = os.path.basename(png_name).split(".")[0]
    ax1.text(xx[0,:][50], yy[:,0][95], name, horizontalalignment='center', 
             verticalalignment='top', color="white", fontsize=25)
    fig.savefig(png_name)
    fig.show()


def plot_all_single_heatmap(pd_train, M_categoryDict, clipsize, png_name):
    mapdata = np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
    fig = plt.figure(figsize=(35,35))
    col = 4
    row = 4
    row_ext = 0.9/float(row)
    col_ext = 0.9/float(col)
    l_row_beg = [ 0.05+idx*row_ext for idx in range(row)]
    l_col_beg = [ 0.95-idx*col_ext for idx in range(col)]
    
    M_sampleInfo = {}
    for i,category in enumerate(sorted(M_categoryDict.keys())): 
        print(category)
        xx, yy, Z = get_heatmap(pd_train, clipsize, category)
        # avoid too long time
        if i>=16:
            continue
        ax = fig.add_subplot(col, row, i+1)
#        ax = fig.add_axes([l_row_beg[(i%row)], l_col_beg[int(i/row)], row_ext, col_ext])
        M_sampleInfo[category] = Z.ravel()
        up_max = np.percentile(Z, 99)
        Z[Z > up_max] = up_max
        cut = 10
        ax.imshow(mapdata,extent=lon_lat_box, cmap=plt.get_cmap('gray'))
        ax.contourf(xx, yy, Z, cut, cmap="jet", shade=True, alpha=0.5).collections[0].set_alpha(0)
        ax.text(xx[0,:][50], yy[:,0][95], category, horizontalalignment='center', 
             verticalalignment='top', color="white", fontsize=15)
        remove_axis(ax)

    fig.savefig(png_name)
    fig.show()
    pd_sampDensMatMelt = pd.DataFrame(M_sampleInfo)
    return pd_sampDensMatMelt

def plot_cmp_heatmap(xx, yy, Z1, Z2, clipsize, png_name):
    mapdata = np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
    up_max1 = np.percentile(Z1, 99)
    Z1[Z1 > up_max1] = up_max1
    up_max2 = np.percentile(Z2, 99)
    Z2[Z2 > up_max2] = up_max2
    
    cut = 10
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
    ax1.imshow(mapdata,extent=lon_lat_box, cmap=plt.get_cmap('gray'))
    ax2.imshow(mapdata,extent=lon_lat_box, cmap=plt.get_cmap('gray'))
    delta_Z_h = Z1-Z2
    delta_Z_l = Z2-Z1
    delta_Z_h[delta_Z_h < 0] = 0
    delta_Z_l[delta_Z_l < 0] = 0
    ax1.contourf(xx, yy, delta_Z_h, cut, cmap="jet", shade=True, 
                 alpha=0.5).collections[0].set_alpha(0)
    ax2.contourf(xx, yy, delta_Z_l, cut, cmap="jet", shade=True, 
                 alpha=0.5).collections[0].set_alpha(0)
    
    name1 = os.path.basename(png_name).split(".")[0].split("__")[0]
    name2 = os.path.basename(png_name).split(".")[0].split("__")[1]
    ax1.text(xx[0,:][50], yy[:,0][95], "%s higher than %s" % (name1, name2), 
             horizontalalignment='center', verticalalignment='top', color="white", fontsize=25)
    ax2.text(xx[0,:][50], yy[:,0][95], "%s lower than %s" % (name1, name2), 
             horizontalalignment='center', verticalalignment='top', color="white", fontsize=25)

    remove_axis(ax1)
    remove_axis(ax2)
    fig.savefig(png_name)
    fig.show()
    
infile_train = "../input/train.csv"
pd_train = pd.read_csv(infile_train)
parse_timeInfo(pd_train)
l_colNameUsed = [ 'X', 'Y']
y, M_categoryDict = num_l_word(list(pd_train['Category']) ) 
CateGoryDict = {"Category":sorted(M_categoryDict.keys())}
xx, yy, Z = get_heatmap(pd_train, clipsize, "ARSON")
plot_one_heatmap(xx, yy, Z, clipsize, "ARSON.png")
#Too long time
#pd_sampDensMatMelt = plot_all_single_heatmap(pd_train, M_categoryDict, clipsize, "all.png")
c1 = "ARSON"
c2 = "ASSAULT"
c_input = "%s__%s.png" % (c1, c2)
xx, yy, Z1 = get_heatmap(pd_train, clipsize, c1)
xx, yy, Z2 = get_heatmap(pd_train, clipsize, c2)
plot_cmp_heatmap(xx, yy, Z1, Z2, clipsize, c_input)

