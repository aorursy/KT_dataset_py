# Listage des répertoires de l'environnement de travail.

import os



print("Liste des fichiers dans le répertoire données en entrée :")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print("    "  + os.path.join(dirname, filename))
# Les packages

import sys



import numpy as np

import scipy as sp

import matplotlib.pyplot as plt



from skimage import io

from skimage.data import astronaut

from skimage.util import crop

from skimage.segmentation import slic

from skimage.segmentation import mark_boundaries

from skimage import color



# help(cc)

# Plus de détails sur les Colormap disponibles : https://colorcet.holoviz.org/user_guide/index.html

import colorcet as cc

# Chargement d'une image de Weizmann

name = "img_3083_modif"

img = io.imread("/kaggle/input/img_3083_modif/src_color/img_3083_modif.png")

img_shape = (*img.shape, img.shape[0]*img.shape[1])

print( "Image '{}' h:{}, v:{}, nb bandes: {}, nb pixels: {}".format( name, *img_shape ))

io.imsave(name + ".png", img)
# Chargement de la vérité terrain

name_gt = "Img_3083_modif_24"

img_gt = io.imread("/kaggle/input/img_3083_modif/human_seg/img_3083_modif_24.png")

img_gt_shape = (*img_gt.shape, img_gt.shape[0]*img_gt.shape[1])

print("Image gt '{}' h:{}, v:{}, nb bandes: {}, nb pixels: {}".format( name_gt + ".png" , *img_gt_shape) )

io.imsave(name_gt + "_gt.png", img_gt)



# Retaillage

ret_h = img_gt_shape[0]-img_shape[0]

ret_l = img_gt_shape[1]-img_shape[1]

ret_h, ret_l

if ret_h != 0 or ret_l != 0 :

    sys.stderr.write(">>>  Attention l'image et la vérité terrain diffèrent en taille  <<<")

    img_rt = crop(img_gt, ( (0,ret_h), (0,ret_l), (0,0) ), copy=False )

else :

    img_rt = img_gt



img_rt_shape = (*img_rt.shape, img_rt.shape[0]*img_rt.shape[1])

print("Image rt '{}' h:{}, v:{}, nb bandes: {}, nb pixels: {}".format( name_gt + ".png" , *img_rt_shape) )

io.imsave(name_gt + "_gt_rt.png", img_rt)
# Traitement

segments_1 = slic(img, n_segments=100, compactness=10)

segments_2 = slic(img, n_segments=200, compactness=10)

segments_5 = slic(img, n_segments=500, compactness=10)

segments_10 = slic(img, n_segments=1000, compactness=10)

segments_1p = slic(img, n_segments=img_shape[3]//100, compactness=10)

segments_10p = slic(img, n_segments=img_shape[3]//10, compactness=10)



img_seg_1 = mark_boundaries(img, segments_1)

io.imsave(name + "_100s.png", img_seg_1)



img_seg_2 = mark_boundaries(img, segments_2)

io.imsave(name + "_200s.png", img_seg_2)



img_seg_5 = mark_boundaries(img, segments_5)

io.imsave(name + "_500s.png", img_seg_5)



img_seg_10 = mark_boundaries(img, segments_10)

io.imsave(name + "_1000s.png", img_seg_10)



img_seg_1p = mark_boundaries(img, segments_1p)

io.imsave(name + "_1ps.png", img_seg_1p)



img_seg_10p = mark_boundaries(img, segments_10p)

io.imsave(name + "_10ps.png", img_seg_10p)





fig, ax = plt.subplots(7, 2, figsize=(20, 60), sharex=True, sharey=True)



ax[0][0].imshow(img)

ax[0][0].set_title("Original")

ax[0][1].imshow(img_rt)

ax[0][1].set_title("GT")



ax[1][0].imshow(img_seg_1)

ax[1][0].set_title('SLIC 100 segments')

ax[1][1].imshow(mark_boundaries(img_rt, segments_1))

ax[1][1].set_title('SLIC GT 100 segments')



ax[2][0].imshow(img_seg_2)

ax[2][0].set_title('SLIC 200 segments')

ax[2][1].imshow(mark_boundaries(img_rt, segments_2))

ax[2][1].set_title('SLIC GT 200 segments')



ax[3][0].imshow(img_seg_5)

ax[3][0].set_title('SLIC 500 segments')

ax[3][1].imshow(mark_boundaries(img_rt, segments_5))

ax[3][1].set_title('SLIC GT 500 segments')



ax[4][0].imshow(img_seg_10)

ax[4][0].set_title('SLIC 1000 segments')

ax[4][1].imshow(mark_boundaries(img_rt, segments_10))

ax[4][1].set_title('SLIC GT 1000 segments')



ax[5][0].imshow(img_seg_1p)

ax[5][0].set_title('SLIC 1% segments')

ax[5][1].imshow(mark_boundaries(img_rt, segments_1p))

ax[5][1].set_title('SLIC GT 1% segments')



ax[6][0].imshow(img_seg_10p)

ax[5][0].set_title('SLIC 10% segments')

ax[6][1].imshow(mark_boundaries(img_rt, segments_10p))

ax[6][1].set_title('SLIC GT 10% segments')



plt.show()
# Calcul

nb_segments = img_shape[3] // 100   # soit 1% 

compacite = 10.0



segments_0 = slic(img, n_segments=nb_segments, compactness=100)



print("Dimension de l'image          : %d, %d, %d" % img.shape )

print("Dimension de l'image-segments : %d, %d" % segments_0.shape )

print("Nb segments                   : %d demandés, %d obtenus" % (nb_segments, len(np.unique(segments_0))))
# Affichage



fig, ax = plt.subplots(3, 2, figsize=(20, 20), sharex=True, sharey=True)



img_seg_0 = color.label2rgb(segments_0, img, colors=cc.glasbey_bw_minc_20, alpha=0.50, kind='overlay')



ax[0,0].imshow(img)

ax[0,0].set_title("Original")

ax[0,1].imshow(img_seg_0)

ax[0,1].set_title("Segments (nb=%d superpixels demandés, %d obtenus)" % (nb_segments, len(np.unique(segments_0))))



ax[1,0].imshow(mark_boundaries(img, segments_0))

ax[1,0].set_title("Orig + Segments")

ax[1,1].imshow(mark_boundaries(img_rt, segments_0))

ax[1,1].set_title("GT + segments")



ax[2,0].imshow(color.label2rgb(segments_0, img, colors=cc.glasbey_bw_minc_20, alpha=0.35, kind='overlay'))

ax[2,0].set_title("alpha=%.2f" % (0.35))

ax[2,1].imshow(color.label2rgb(segments_0, img, colors=cc.glasbey_bw_minc_20, alpha=0.90, kind='overlay'))

ax[2,1].set_title("alpha=%.2f" % (0.90))



plt.show()
nb_segments_desire = img_shape[3] // 100   # soit 1% 

compacites_liste = [ 0.01 , 0.1 , 1 , 10 , 100 ]



fig, ax = plt.subplots(len(compacites_liste)//2+1, 2, figsize=(20, 20), sharex=True, sharey=True)

ax = ax.flatten()



# L'original

ax[0].imshow(img)

ax[0].set_title("Original")



# Les différentes compacités

for i, comp in enumerate(compacites_liste):

    segments_X = slic(img, n_segments=nb_segments_desire, compactness=comp)

    img_seg_X = color.label2rgb(segments_X, img, colors=cc.glasbey_bw_minc_20, alpha=0.8)

    ax[i+1].imshow(img_seg_X)

    ax[i+1].set_title("Segments (nb=%d désirés, %d obtenus) (compacité=%.2f)" % (nb_segments_desire, len(np.unique(segments_X)), comp))





plt.show()
nb_segments_desire = img_shape[3] // 100   # soit 1% 

compacites_liste = [ 0.01 , 0.1 , 1 , 10 , 100 ]



fig, ax = plt.subplots(len(compacites_liste)//2+1, 2, figsize=(20, 20), sharex=True, sharey=True)

ax = ax.flatten()



# L'original

ax[0].imshow(img)

ax[0].set_title("Original")



# Les différentes compacités

for i, comp in enumerate(compacites_liste):

    segments_X = slic(img, n_segments=nb_segments_desire, compactness=comp)

    img_seg_X = color.label2rgb(segments_X, img, kind="avg")

    ax[i+1].imshow(img_seg_X)

    ax[i+1].set_title("Segments (nb=%d désirés, %d obtenus) (compacité=%.2f)" % (nb_segments_desire, len(np.unique(segments_X)), comp))





plt.show()
nb_segments_desire = img_shape[3] // 100   # soit 1% 

compacites_liste = [ 0.01 , 0.1 , 1 , 10 , 100 ]



fig, ax = plt.subplots(len(compacites_liste)//2+1, 2, figsize=(20, 20), sharex=True, sharey=True)

ax = ax.flatten()



# L'original

ax[0].imshow(img)

ax[0].set_title("Original")



# Les différentes compacités

for i, comp in enumerate(compacites_liste):

    segments_X = slic(img, n_segments=nb_segments_desire, compactness=comp)

    img_seg_X = color.label2rgb(segments_X, img, colors=cc.glasbey_bw_minc_20, alpha=0.8)

    img_con_X = mark_boundaries(img_seg_X, segments_X)

    ax[i+1].imshow(img_con_X)

    ax[i+1].set_title("Segments (nb=%d désirés, %d obtenus) (compacité=%.2f)" % (nb_segments_desire, len(np.unique(segments_X)), comp))





plt.show()
nb_segments_desire = img_shape[3] // 100   # soit 1% 

compacites_liste = [ 0.01 , 0.1 , 1 , 10 , 100 ]



fig, ax = plt.subplots(len(compacites_liste)//2+1, 2, figsize=(20, 20), sharex=True, sharey=True)

ax = ax.flatten()



# L'original

ax[0].imshow(img)

ax[0].set_title("Original")



# Les différentes compacités

for i, comp in enumerate(compacites_liste):

    segments_X = slic(img, n_segments=nb_segments_desire, compactness=comp)

    img_seg_X = color.label2rgb(segments_X, img, kind="avg")

    img_con_X = mark_boundaries(img_seg_X, segments_X)

    ax[i+1].imshow(img_con_X)

    ax[i+1].set_title("Segments (nb=%d désirés, %d obtenus) (compacité=%.2f)" % (nb_segments_desire, len(np.unique(segments_X)), comp))





plt.show()
# La grille d'expérimentation

compac_liste = [ 0.01 , 0.1 , 1 , 10 , 100 ]

nb_seg_liste = [ 100, 200, 500, 1000, 5000 ]



fig, ax = plt.subplots(len(nb_seg_liste), len(compac_liste), figsize=(20, 20), sharex=True, sharey=True)



# Les différentes compacités

for lig, all_nb_seg in enumerate(nb_seg_liste):

    for col, all_comp in enumerate(compac_liste):

        all_seg = slic(img, n_segments = all_nb_seg, compactness = all_comp)

        all_img_seg = color.label2rgb(all_seg, img, kind = "avg")

        ax[lig, col].imshow(all_img_seg)

        ax[lig, col].set_title("nb=%d d / %d o\ncompacité=%.2f" % (all_nb_seg, len(np.unique(all_seg.max())), all_comp))

        all_name = "%s_all_%ds_%.2fc.png" % (name, all_nb_seg, all_comp)

        io.imsave(all_name, all_img_seg)



plt.show()

print("%d fichiers sauvegardés" % ((col+1) * (lig+1)))
# Chargement d'une image de Weizmann

name = "windowCN_0078"

img = io.imread("/kaggle/input/windowcn_0078/src_color/windowcn_0078.png")

img_shape = (*img.shape, img.shape[0]*img.shape[1])

print( "Image '{}' h:{}, v:{}, nb bandes: {}, nb pixels: {}".format( name, *img_shape ))

io.imsave(name + ".png", img)



# Chargement de la vérité terrain

name_gt = "windowCN_0078_40"

img_gt = io.imread("/kaggle/input/windowcn_0078/human_seg/windowcn_0078_40.png")

img_gt_shape = (*img_gt.shape, img_gt.shape[0]*img_gt.shape[1])

print("Image gt '{}' h:{}, v:{}, nb bandes: {}, nb pixels: {}".format( name_gt + ".png" , *img_gt_shape) )

io.imsave(name_gt + "_gt.png", img_gt)



# Retaillage

ret_h = img_gt_shape[0]-img_shape[0]

ret_l = img_gt_shape[1]-img_shape[1]

ret_h, ret_l

if ret_h != 0 or ret_l != 0 :

    sys.stderr.write(">>>  Attention l'image et la vérité terrain diffèrent en taille  <<<")

    img_rt = crop(img_gt, ( (0,ret_h), (0,ret_l), (0,0) ), copy=False )

else :

    img_rt = img_gt



img_rt_shape = (*img_rt.shape, img_rt.shape[0]*img_rt.shape[1])

print("Image rt '{}' h:{}, v:{}, nb bandes: {}, nb pixels: {}".format( name_gt + ".png" , *img_rt_shape) )

io.imsave(name_gt + "_gt_rt.png", img_rt)
# La grille d'expérimentation

compac_liste = [ 0.01 , 0.1 , 1 , 10 , 100 ]

nb_seg_liste = [ 100, 200, 500, 1000, 5000 ]



fig, ax = plt.subplots(len(nb_seg_liste), len(compac_liste), figsize=(20, 20), sharex=True, sharey=True)



# Les différentes compacités

for lig, all_nb_seg in enumerate(nb_seg_liste):

    for col, all_comp in enumerate(compac_liste):

        all_seg = slic(img, n_segments = all_nb_seg, compactness = all_comp)

        all_img_seg = color.label2rgb(all_seg, img, kind = "avg")

        ax[lig, col].imshow(all_img_seg)

        ax[lig, col].set_title("nb=%d d / %d o\ncompacité=%.2f" % (all_nb_seg, len(np.unique(all_seg)), all_comp))

        all_name = "%s_all_%ds_%.2fc.png" % (name, all_nb_seg, all_comp)

        io.imsave(all_name, all_img_seg)



plt.show()

print("%d fichiers sauvegardés" % ((col+1) * (lig+1)))