from skimage import morphology,io

from skimage.color import rgb2gray

import matplotlib.pyplot as plt

import numpy as np

import os



#load images to python dictionary data structure

#args -> fraction = percentage of images to load

def load_image_dict(fraction=1):

    if(fraction>1 or fraction<0):

        fraction = 1

        

    inf_list = []

    uninf_list = []

    

    inf_path = "../input/cell-images-for-detecting-malaria/cell_images/Parasitized/"

    uninf_path = "../input/cell-images-for-detecting-malaria/cell_images/Uninfected/"

    inf_images = os.listdir(inf_path)

    uninf_images = os.listdir(uninf_path)

    for idx in range(0,int(len(inf_images)*fraction)):

        inf_img = inf_path + inf_images[idx]

        inf_img = darken(io.imread(inf_img))

        #inf_img = filters.threshold_multiotsu(inf_img, classes=3, nbins=256)

        inf_img = darken(ToGray(inf_img))

        inf_list.append(inf_img)

    

    for idx in range(0,int(len(uninf_images)*fraction)):   

        uninf_img = uninf_path + uninf_images[idx]

        uninf_img = darken(io.imread(uninf_img))

        #uninf_img = filters.threshold_multiotsu(uninf_img, classes=3, nbins=256)

        uninf_img = darken(ToGray(uninf_img))

        uninf_list.append(uninf_img)

    

    return {'uninfected':uninf_list, 'infected':inf_list}

from skimage import filters



def sobel(dataset,type):

    imgs = []

    for x in range(0,3):

        img = filters.sobel(dataset[type][x])

        imgs.append(img)

    return imgs



#convert image to grayscale

def ToGray(img):

    black = img[:,:,0] == 0

    img[black] = [255, 255, 255]



    img = rgb2gray(img)

    #img = rgb2gray(img)

    return img



#it darken the image

def darken(img):

    img_darked = img

    #light_spots = np.array((img > img.max())

    dark_spots = img <= (img.min()+0.2)

    #img[light_spots] = 1

    img_darked[dark_spots] = img.min()#+5*(img.min())

    return img_darked

#returns 3images from dataset

#will be used to view those 3images

def create_list(dataset,type,idx):

    temp = []

    for x in range(idx,idx+3):

        img = dataset[type][x]

        temp.append(img)

    return temp





#display first 3 image from the given list

def ShowAsRow(img_list,title_list):

    img1 = img_list[0]

    img2 = img_list[1]

    img3 = img_list[2]

    from skimage import data

    from skimage import exposure



    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),

                                        sharex=True, sharey=True)

    #for aa in (ax1, ax2, ax3):

        #aa.set_axis_off()



    ax1.imshow(img1)

    ax1.set_title(title_list[0])

    ax2.imshow(img2)

    ax2.set_title(title_list[1])

    ax3.imshow(img3)

    ax3.set_title(title_list[2])



    plt.tight_layout()

    plt.show()
import plotly.graph_objects as go

import numpy as np



# Helix equation

def Surface_data(img):

    x = []

    y = []

    z = []

    for idx in range(0,img.shape[0]):

        temp_z = []

        for idy in range(0,img.shape[1]):

            if(idy==idx):

                x.append(idx)

                y.append(idy)

            temp_z.append(img[idx][idy])

        z.append(temp_z)

    

    # tight layout

    #fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    return {'x':np.array(x),'y':np.array(y),'z':np.array(z)}









def plot_surface(dataset=0,type="",idx=0,img=0):

    data = []

    if(type!=""):

         data = Surface_data(dataset[type][0])

    else:

        data = Surface_data(img)

        

    fig = go.Figure(go.Surface(

        contours = {

            "x": {"show": True, "start": 1.5, "end": 2, "size": 0.04, "color":"white"},

            "z": {"show": True, "start": 0.5, "end": 0.8, "size": 0.05}

        },

        x = data['x'],

        y = data['y'],

        z =data['z']

        )

    )

    fig.update_layout(

            scene = {

                "xaxis": {"nticks": 20},

                "zaxis": {"nticks": 4},

                'camera_eye': {"x": 0, "y": -1, "z": 0.5},

                "aspectratio": {"x": 1, "y": 1, "z": 0.2}

            })

    fig.show()
#work with what type

type = 'infected'

#load dataset

dataset = load_image_dict(0.1)

#ShowAsRow(create_list(dataset,type))
#This graphs helps by giving the idea on how to approach the problem

plot_surface(dataset,"infected")
from skimage import data, color

from skimage.feature import canny



# Load picture and detect edges

def FindEdges(img_list):

    edges = []

    for x in range(0,3):

        img = img_list[x]

        edges.append(canny(img,1))

    return edges
from scipy import ndimage as ndi

def Malaria_cells(edges_list):

    malaria = []

    for x in range(0,3):

        edges = edges_list[x]

        fill_edges = ndi.binary_fill_holes(edges)

        filled = np.bitwise_xor(fill_edges,edges)

        malaria.append(filled)

    return malaria
#+dataset[type][0]#filters.sobel_h(dataset[type][0])+filters.sobel_v(dataset[type][0])+filters.sobel(dataset[type][0])+dataset[type][0]

#img = filters.threshold_local(img,block_size=3)#threshold_local

#plot_surface(img=img)

def MalariaRefined(img_list,malaria_list):

    results = []

    for x in range(0,3):

        malaria = malaria_list[x]

        #img_sobel = filters.sobel(darken(img_list[x]))

        result = malaria#*(img_sobel+1)

        results.append(result)

    return results
def Area(img):

    surface = img.shape[0]*img.shape[1]

    cell = np.where(img > 0, 1, 0).sum()

    return cell/surface



def Malaria(cell_list,malaria_refined):

    results = []

    for x in range(0,3):

        percent = Area(malaria_refined[x])*Area(cell_list[x])*100

        results.append(percent)

        msg = "This cell has {} % malaria\n".format(percent)

        print(msg)

    return results

    
def ReportStatus(dataset,cell_type="infected",idx=0):

    cell_list = create_list(dataset,cell_type,idx)

    titles = ["{} cell index = {}".format(cell_type,idx),"{} cell index = {}".format(cell_type,idx+1),"{} cell index = {}".format(cell_type,idx+2)]

    print("plot of THREE 3 Cells after preprocessing<darken then converted to gray scale>\n")

    ShowAsRow(cell_list,titles)

    print("\n_____________________________________________________________________________\n")

    print("Plot of image edges respectively")

    edges = FindEdges(cell_list)

    ShowAsRow(edges,titles)

    print("\n_____________________________________________________________________________\n")

    print("Plot of MALARIA from cell respectively")

    malaria_list = Malaria_cells(edges)

    ShowAsRow(malaria_list,titles)

    print("\n_____________________________________________________________________________\n")

    print("Plot of refined MALARIA from cell respectively")

    malaria_refined = MalariaRefined(cell_list,malaria_list)

    ShowAsRow(malaria_refined,titles)

    print("\n_____________________________________________________________________________\n")

    print("Plot of MALARIA from cell respectively")

    infection_score = Malaria(cell_list,malaria_refined)

    for score in infection_score:

        print("Cell on index number {} is infected by {} %".format(idx,score))
ReportStatus(dataset,cell_type="infected",idx=3)