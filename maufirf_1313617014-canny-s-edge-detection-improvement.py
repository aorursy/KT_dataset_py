import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from numpy.lib.stride_tricks import as_strided
def get_all_window(M, w):

    M = np.pad(M, w//2, 'symmetric')

    sub_shape = (w, w)

    view_shape = tuple(np.subtract(M.shape, sub_shape) + 1) + sub_shape

    arr_view = as_strided(M, view_shape, M.strides * 2)

    arr_view = arr_view.reshape((-1,) + sub_shape)

    return arr_view



def fastest_convolution_2d(im, K):

    w, _ = K.shape

    m,n = im.shape

    im_all_subw = get_all_window(im, w)

    X = np.sum(np.sum(im_all_subw * K, 1), 1)

    return X.reshape(m,n)
grad_operators = {

    'prewitt':(

        np.array([[-1,0,1],

                  [-1,0,2],

                  [-1,0,1]]),

        np.array([[1,1,1],

                  [0,0,0],

                  [-1,-1,-1]])

    ),

    'robertcross':(

        np.array([[1,0],

                  [0,1]]),

        np.array([[0,-1],

                  [-1,0]]) 

    ),

    'sobelfeldman':(

        np.array([[-3,0,3],

                  [-10,0,10],

                  [-3,0,3]]),

        np.array([[3,10,3],

                  [0,0,0],

                  [-3,-10,-3]])

    ),

    'scharr':(

        np.array([[47,0,-47],

                  [162,0,-162],

                  [47,0,-47]]),

        np.array([[47,162,47],

                  [0,0,0],

                  [-47,-162,-47]])

    ),

    'sobel':(

        np.array([[-1,0,1],

                  [-2,0,2],

                  [-1,0,1]]),

        np.array([[1,2,1],

                  [0,0,0],

                  [-1,-2,-1]])

    )

}
def join_gradient_euclidean(Gx, Gy):

    return np.sqrt((Gx**2)+(Gy**2))



def join_gradient_manhattan(Gx, Gy):

    return np.abs(Gx)+np.abs(Gy)
path = '../input/meanshiftimgs/mean_shift_image.png'

if path[-4:]=='.png':

    im = np.round(mpimg.imread(path)*255).astype(int)

elif path[-4:]=='.jpg':

    im = mpimg.imread(path)

w = 11

sigma = 1.4

imgrey = np.round(0.3 * im[:,:,0] + 0.59 * im[:,:,1] + 0.11 * im[:,:,2]).astype(int)



f = plt.figure(figsize=(18,15))

f.add_subplot(121).imshow(im)

f.add_subplot(122).imshow(imgrey, cmap='gray')

def gaussian_kernel2d(w, sigma):

    w = w + (w % 2 == 0)

    F = np.zeros([w,w])

    mid = w//2 

    k = np.arange(w) - mid

    denom = 2*np.pi*sigma**2

    for i in k:

        for j in k:

            par = (i**2 + j**2)/(2*sigma**2)

            F[i + mid,j + mid] = np.exp(-par)/denom

    return F
gaussk = gaussian_kernel2d(w, sigma)

im_gauss = fastest_convolution_2d(imgrey, gaussk)



f = plt.figure(figsize=(18,15))

f.add_subplot(221, title='Original').imshow(imgrey, cmap='gray')

f.add_subplot(222, title='Result').imshow(im_gauss, cmap='gray')

f.add_subplot(153, title='Kernel').imshow(gaussk, cmap='hot')
def adaptive_weight(Gx, Gy, h):

    return np.exp(np.sqrt(join_gradient_euclidean(Gx, Gy) / (2 * (h ** 2))))



def adaptive_convolution_2d(im, weight):

    w = 3

    m,n = im.shape

    im_all_subw = get_all_window(im, w)

    weight_all_subw = get_all_window(weight, w)

    X = np.sum(np.sum(im_all_subw * weight_all_subw, 1), 1) / np.sum(np.sum(weight_all_subw, 1), 1)

    return X.reshape(m,n)



def adaptive_filter(im, n=5, h=1.5, operator='sobel'):

    # 1. K = 1, set the iteration n and the coefficient of the amplitude of the edge h.

    #K = 1

    #h = 1.5

    weights = []

    iteration_result = [im]

    

    # Iterative section

    for i in range(n):

        # 2. Calculate the gradient value Gx and Gy

        op = grad_operators[operator]

        Gx = fastest_convolution_2d(im,op[0])

        Gy = fastest_convolution_2d(im,op[1])



        # 3. Calculate the weight

        weight = adaptive_weight(Gx, Gy, h)

        weights.append(weight)



        # 4. Convolve

        im = adaptive_convolution_2d(im, weight)

        iteration_result.append(im)

        

    return im, weights, iteration_result
n_adaptive = 5

im_adaptive, adaptive_weights, adaptive_iteration_result = adaptive_filter(imgrey, n_adaptive)



f = plt.figure(figsize=(18,15))

f.add_subplot(221,title='Iteration 0').imshow(imgrey, cmap='gray')

f.add_subplot(222,title='Final').imshow(im_adaptive, cmap='gray')

for i in range(1, n_adaptive-1):

    f.add_subplot(4,n_adaptive-2,i+(n_adaptive-2)*2,title='Iteration ' + str(i)).imshow(adaptive_iteration_result[i], cmap='gray')
def find_gradient(im, operator='sobel'):

    Kx, Ky = grad_operators[operator]

    m,n = im.shape

    w = 3

    

    im_all_subw = get_all_window(im, w)

    Gx = np.sum(np.sum(im_all_subw * Kx, 1), 1).reshape(m,n)

    Gy = np.sum(np.sum(im_all_subw * Ky, 1), 1).reshape(m,n)

    theta = np.arctan2(np.abs(Gy), Gx)

    theta = theta*180/np.pi

    return Gx, Gy, theta
Gx, Gy, theta = find_gradient(im_gauss)

G = join_gradient_euclidean(Gx, Gy) # Euclidean because kaggle has free RAM and GPU yay



f = plt.figure(figsize=(18,15))

f.add_subplot(221,title='Gx').imshow(Gx, cmap='gray')

f.add_subplot(222,title='Gy').imshow(Gy, cmap='gray')

f.add_subplot(223,title='G').imshow(G, cmap='gray')

f.add_subplot(224,title='Theta').imshow(theta, cmap='viridis')
f = plt.figure(figsize=(18,15))



Gxt, Gyt, _ = find_gradient(im_gauss, 'prewitt')

f.add_subplot(221,title='Prewitt operator').imshow(join_gradient_euclidean(Gxt, Gyt), cmap='gray')



Gxt, Gyt, _ = find_gradient(im_gauss, 'sobelfeldman')

f.add_subplot(222,title='Sobel-Feldman operator').imshow(join_gradient_euclidean(Gxt, Gyt), cmap='gray')



Gxt, Gyt, _ = find_gradient(im_gauss, 'scharr')

f.add_subplot(212,title='Scharr operator').imshow(join_gradient_euclidean(Gxt, Gyt), cmap='gray')



f.show()
def round_to_closest_n(x, n, clock=0):

    out = np.round(x / n) * n

    if clock:

        out %= clock

    return out



def non_maximum_suppression(im,theta):

    ntheta = round_to_closest_n(theta, 45, 180)

    thetafilters = np.array([

        [[0,0,0],[-1,2,-1],[0,0,0]],

        [[-1,0,0],[0,2,0],[0,0,-1]],

        [[0,-1,0],[0,2,0],[0,-1,0]],

        [[0,0,-1],[0,2,0],[-1,0,0]]])

    

    per_angle_res = [fastest_convolution_2d(im, thetafilters[i]) * (ntheta==(45*i)) for i in range(4)]

    return np.sum(per_angle_res,0), per_angle_res, ntheta
im_nms, in_nms_perkernel, ntheta = non_maximum_suppression(G,theta)

im_nms_pos = im_nms * (im_nms >= 0)



f = plt.figure(figsize=(18,24))

f.add_subplot(321,title='Degree 0').imshow(in_nms_perkernel[0], cmap='gray')

f.add_subplot(322,title='Degree 45').imshow(in_nms_perkernel[1], cmap='gray')

f.add_subplot(323,title='Degree 90').imshow(in_nms_perkernel[2], cmap='gray')

f.add_subplot(324,title='Degree 135').imshow(in_nms_perkernel[3], cmap='gray')

f.add_subplot(325,title='Result image').imshow(im_nms_pos, cmap='gray')

f.add_subplot(326,title='Quantified Theta').imshow(ntheta, cmap='viridis')
im_nms_pos_classes = np.unique(im_nms_pos)

print(f'{im_nms_pos_classes[0]}..{im_nms_pos_classes[-1]}')
def double_tresholding(im, hi=80, lo=20):

    strong = im > hi

    weak = (im >= hi) == (im <= lo)

    return strong, weak
maxcap = np.max(im_nms_pos)

strong_a, weak_a = double_tresholding(im_nms_pos,(60*maxcap)/255,(10*maxcap)/255)



m,n = imgrey.shape

ta = np.zeros((m,n,3))

ta[:,:,0] = strong_a

ta[:,:,1] = (strong_a + weak_a) > 0

ta[:,:,2] = weak_a

plt.figure(figsize=(18,15)).add_subplot(111, title='Yellows are strong edges').imshow(ta,cmap='gray')
def histogram(img):

    img = img.astype(np.float)

    row, col = img.shape

    maxstr = np.int(np.floor(np.max(img))+1)

    y = np.array([np.sum(np.bitwise_and(img>=pixstr-0.5,img<pixstr+0.5)) for pixstr in range(maxstr+1)])

    return y



def get_tresholds(hist, separate=False):

    pixels = np.sum(hist)

    mx = hist.size

    tres = []

    

    for i in range(1, mx):

        le = hist[0:i]

        ri = hist[i:mx]

        

        vb = np.var(le)

        wb = np.sum(le) / pixels

        mb = np.mean(le)

        

        vf= np.var(ri)

        wf = np.sum(ri) / pixels

        mf = np.mean(ri)

        

        V2w = wb * (vb) + wf * (vf)

        V2b = wb * wf * (mb - mf)**2

        

        if not np.isnan(V2w): tres.append((i,V2w))

            

    if separate:

        return list(zip(*tres))

    return tres
hist = histogram(im_nms_pos)

pixstr, pixtres = get_tresholds(hist, True)

maxstr = np.int(np.floor(maxcap)+1)



up_tres = pixstr[pixtres.index(max(pixtres))]

lo_tres = up_tres * 0.75



print(f'Upper treshold = {up_tres} , Lower treshold = {lo_tres}')



f = plt.figure(figsize=(18,6))

f.add_subplot(121, title='Histogram').bar(np.arange(0,maxstr+1), hist, color='b', width=5, align='center', alpha=0.25)

f.add_subplot(122, title='Inter-classgroup deviation').plot(pixstr,pixtres,up_tres,max(pixtres),'ro')

f.show()
strong_b, weak_b = double_tresholding(im_nms_pos,up_tres,lo_tres)



tb = np.zeros((m,n,3))

tb[:,:,0] = strong_b

tb[:,:,1] = (strong_b + weak_b) > 0

tb[:,:,2] = weak_b

plt.figure(figsize=(18,15)).add_subplot(111, title='Yellows are strong edges').imshow(tb,cmap='gray')
def hysteresis(strong, weak):

    # basically we're just repeating 8-neighbor convolutions

    K = np.array([

            [1,1,1],

            [1,0,1],

            [1,1,1]

        ])

    union = strong + weak

    blob = strong

    blobbefore = np.ones(union.shape)-blob

    #diff = np.ones(union.shape)

    while not np.all(blob==blobbefore):

        blobbefore = blob

        blob = np.bitwise_and((fastest_convolution_2d(blob,K)+strong)>0,union)

    return blob
blob_b = hysteresis(strong_b, weak_b)



edctb = np.zeros((m,n,3))

edctb[:,:,0] = blob_b

edctb[:,:,1] = strong_b

edctb[:,:,2] = ((strong_b.astype('u8') + weak_b.astype('u8')) - blob_b.astype('u8')) + strong_b.astype('u8')

plt.figure(figsize=(18,15)).add_subplot(111, title='Reds are grabbed as strong edges, blue are defective weak edges.').imshow(edctb,cmap='gray')
plt.figure(figsize=(18,15)).add_subplot(111, title='Final result').imshow(blob_b,cmap='gray')
blob_a = hysteresis(strong_a, weak_a)



edcta = np.zeros((m,n,3))

edcta[:,:,0] = blob_a

edcta[:,:,1] = strong_a

edcta[:,:,2] = ((strong_a.astype('u8') + weak_a.astype('u8')) - blob_a.astype('u8')) + strong_a.astype('u8')

plt.figure(figsize=(18,15)).add_subplot(111, title='Reds are grabbed as strong edges, blue are defective weak edges.').imshow(edcta,cmap='gray')
plt.figure(figsize=(18,15)).add_subplot(111, title='Final result').imshow(blob_a,cmap='gray')
def hysteresis_fast(strong, weak):

    union = (strong + weak) > 0

    K = np.array([

            [1,1,1],

            [1,-4,1],

            [1,1,1]

        ])

    return np.bitwise_and(fastest_convolution_2d(union,K)>=0,union)
fin_a = hysteresis_fast(strong_a, weak_a)



edctfa = np.zeros((m,n,3))

edctfa[:,:,0] = fin_a

edctfa[:,:,1] = strong_a

edctfa[:,:,2] = ((strong_a.astype('u8') + weak_a.astype('u8')) - fin_a.astype('u8')) + strong_a.astype('u8')

plt.figure(figsize=(18,15)).add_subplot(111, title='Reds are grabbed as strong edges, blue are defective weak edges.').imshow(edctfa,cmap='gray')
plt.figure(figsize=(18,15)).add_subplot(111, title='Final result').imshow(fin_a,cmap='gray')