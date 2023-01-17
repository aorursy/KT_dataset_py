import numpy as np

import math

from sklearn.preprocessing import LabelBinarizer

import time

import csv

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

%matplotlib inline
#Prepare function

def gaborc(H,k): #H is size of filter, l is direction, k is coeff for sigma

    buf = np.zeros([H, H])

    l = 2*math.pi*math.cos(k)/H

    m = 2*math.pi*math.sin(k)/H

    n = 1/math.sqrt(2*math.pi*H*H)

    for i in range(H):

        for j in range(H):

            buf[i,j] = math.cos((l*(i-H/2)+m*(H/2-j))) * n * math.exp(-((i-H/2)**2+(H/2-j)**2)/(2*H*H))

    return buf



def gabors(H,k): #H is size of filter, l is direction, k is coeff for sigma

    buf = np.zeros([H, H])

    l = 2*math.pi*math.cos(k)/H

    m = 2*math.pi*math.sin(k)/H

    n = 1/math.sqrt(2*math.pi*H)

    for i in range(H):

        for j in range(H):

            buf[i,j] = math.sin((l*(i-H/2)+m*(H/2-j))) * n * math.exp(-((i-H/2)**2+(H/2-j)**2)/(2*H*H))

    return buf



def zpadding(x, n, m, p): # x is (nxm)x1 array, p is addtional zeros

    y = x.reshape([n,m])    

    ab = np.zeros([p,m])

    side = np.zeros([n+2*p, p])

    y = np.r_[ab,y,ab]

    y = np.c_[side,y,side]

    return y.reshape([1,(n+2*p)*(m+2*p)])

    

def Wconv(h, n, m): # h is filter matrix. nxm is conv-ed matrix

    nh = len(h)    

    nc = n-nh+1

    mc = m-nh+1

    buf = np.zeros([nc*mc,n*m])

    for i in range(mc):

        for j in range(nc):

            for p in range(nh):

                for q in range(nh): 

                    buf[i+j*mc,i+j*m+p+q*m] = h[p,q]

    return buf

    

def pooling(x,n,m,p,s): # p is size of pooling, s is stride, maximum pooling

    nc = int((n-p)/s)+1

    mc = int((m-p)/s)+1

    buf = np.zeros([nc,mc])

    x = x.reshape([n,m]) 

    w = np.zeros([nc*mc,n*m])

    for i in range(nc):

        for j in range(mc):

            maxr = 0

            maxq = 0

            for r in range(p):

                for q in range(p):

                    if (buf[i,j] < x[i*s+r,j*s+q]):

                        maxr = r #update 

                        maxq = q #update                 

            buf[i,j] = x[i*s+maxr,j*s+maxq]

            w[i*mc+j,(i*s+maxr)*m+j*s+maxq] = 1 # w is pick up matrix

    buf = buf.reshape([nc*mc])

    return (buf, w) 

    

def relu(x): # x is 1-dim array

    n = x.size

    buf = np.zeros([n])    

    pos = np.zeros([n])

    for i in range(n):

        if (x[0,i]>0):

            buf[i] = x[0,i]

            pos[i] = 1 # pos is activated position

    return (buf, pos) 



def connorm(x):

    c = 1e-3

    return (x - np.mean(x))/math.sqrt((c**2+ np.std(x**2)))



def softmax(y):

    buf = np.exp(y)

    return buf/np.sum(buf) 
#It takes long time to learn with full data set

#Now, 'train_cnt' is only 100, and 'test_cnt' is only 10

def main ():

# load mnist data from kaggle dataset

    data = np.loadtxt("../input/train.csv",delimiter=",", skiprows=1)

    T_n, X_train = np.split(data, [1], axis=1) #data: [label, px1, px2,,,]

    X_train /= X_train.max() #adjust maximum as 1

    T_train = LabelBinarizer().fit_transform(T_n) # transoform to 1-of-k form   

    N = len(T_train)

    D = len(X_train[0,:]) # dimension of image



# parameters

    cl = 10 # num of class

    d = int(math.sqrt(D))

    pd = 4 # zero padding size

    st = 2 # stride

    pl = 3 # pooling size

    eta = 0.01 # learning coeff

    train_cnt = 100#N # train data (by one image)



# first convolution filter [dh1 x dh1]matrix x nh1

    dh1 = 5 # dim of filter

    nh1 = 6 # number of filter

    H1 = np.zeros([nh1,dh1,dh1])

    for i in range(int(nh1/2)):

        H1[i,:,:] = gaborc(dh1,i*math.pi/nh1*2)

        H1[i+int(nh1/2),:,:] = gabors(dh1,i*math.pi/nh1*2)

        

# second convolution filter [dh2 x dh2]matrix x nh2

    dh2 = 3 # dim of filter

    nh2 = 4 # number of filter

    H2 = np.zeros([nh2,dh2,dh2])

    for i in range(int(nh2/2)):

        H2[i,:,:] = gaborc(dh2,i*math.pi/nh2*2)

        H2[i+int(nh2/2),:,:] = gabors(dh2,i*math.pi/nh2*2)

        

# dimensions for each propagation

    d_pd = d+2*pd # 1st padding 28 + 2 * 4

    d_cv = d_pd - dh1 + 1 # 1st convolution 36 - 11 +1

    d_pl = int((d_cv - pl)/st) + 1 # 1st pooling

    

    d_pd2 = d_pl+2*pd # 2nd padding 28 + 2 * 4

    d_cv2 = d_pd2 - dh2 + 1 # 2nd convolution 36 - 11 +1

    d_pl2 = int((d_cv2 - pl)/st) + 1 # 2nd pooling

                                

# learning

# in/out of layer is 1-dim array

    w2 = 0.01 * np.random.random(nh2*nh1*d_pl2*d_pl2*cl).reshape([cl,nh1*nh2*d_pl2*d_pl2]) # all connection 10 <- 12x12x12  

    bo = 0.01 * np.random.random(cl) #bias for all connection

    bc1 = 0.01 * np.random.random(nh1) # bias for convolution

    bc2 = 0.01 * np.random.random(nh2) # bias for convolution

    correct = 0



# prepare matrix - 1st convolution layer                                

    wc1 = np.zeros([nh1, d_cv*d_cv, d_pd*d_pd]) # convolution matrix 36x36 -> 26x26

    conv1 = np.zeros([nh1,d_cv*d_cv])

    pos1 = np.zeros([nh1,d_cv*d_cv])

    B1 = np.zeros([nh1,d_cv*d_cv])

    pool1 = np.zeros([nh1,d_pl*d_pl])

    wp1 = np.zeros([nh1,d_pl*d_pl,d_cv*d_cv]) #pooling matrix

    conm1 = np.zeros([nh1, d_pl*d_pl])

                   

# second convolution layer

    zpd2 = np.zeros([nh1,d_pd2*d_pd2])

    wc2 = np.zeros([nh2, nh1, d_cv2*d_cv2, d_pd2*d_pd2]) # convolution matrix 36x36 -> 26x26

    conv2 = np.zeros([nh2, nh1, d_cv2*d_cv2])

    pos2 = np.zeros([nh2, nh1, d_cv2*d_cv2])

    B2 = np.zeros([nh2, d_cv2*d_cv2])

    pool2 = np.zeros([nh2, nh1, d_pl2*d_pl2])

    wp2 = np.zeros([nh2, nh1, d_pl2*d_pl2, d_cv2*d_cv2]) #pooling matrix

    conm2 = np.zeros([nh2, nh1, d_pl2*d_pl2])        



# start learning -- learn one by one (not batch learning)

    start = time.time()

    Y_train = []



# forward propergation  

    for a in range(train_cnt):

        m = int(a) #pick up parameter

        x = X_train[m]

        zpd1 = zpadding(x,d,d,pd)

        for i in range(nh1):

            for j in range(d_cv*d_cv):

                B1[i,j] = bc1[i]

        for i in range(nh2):

            for j in range(d_cv2*d_cv2):

                B2[i,j] = bc2[i]

        for k in range(nh1): # 1st convolution layers

            wc1[k,:,:] = Wconv(H1[k,:,:],d_pd,d_pd)

            conv_dum1 = wc1[k,:,:].dot(zpd1[0,:]).reshape([1,d_cv*d_cv]) + B1[k,:]

            conv1[k,:], pos1[k,:] = relu(conv_dum1)  # 1st convoluition

            pool1[k,:], wp1[k,:,:] = pooling(conv1[k,:],d_cv,d_cv, pl, st) # 1st pooling 

            conm1[k,:]= connorm(pool1[k,:])

            zpd2[k,:] = zpadding(conm1[k,:],d_pl,d_pl,pd)

            for l in range(nh2): # 2nd convolution layers

                wc2[l,k,:,:] = Wconv(H2[l,:,:],d_pd2,d_pd2)

                conv_dum2 = wc2[l,k,:,:].dot(zpd2[0,:]).reshape([1,d_cv2*d_cv2]) + B2[l,:] #2nd Relu

                conv2[l,k,:], pos2[l,k,:] = relu(conv_dum2)  # 2nd convoluition

                pool2[l,k,:], wp2[l,k,:,:] = pooling(conv2[l,k,:],d_cv2,d_cv2, pl, st) # 2nd pooling       

                conm2[l,k,:]= connorm(pool2[l,k,:])

                

        conm_rs = conm2.reshape([nh2*nh1*d_pl2*d_pl2])

        nw = conm_rs.size

        out_dum = w2.dot(conm_rs[:])+bo[:]

        y = softmax(out_dum) # 10-dim

        

#pickup simulated label

        maxc = 0

        buf = 0

        for i in range(cl):

            if y[i] > buf:

                buf = y[i]

                maxc = i

        Y_train.append(maxc)    

        if maxc == T_n[a]:

            correct += 1

            

# back propagation

# back propagation for 2nd layer

        d1 = y-T_train[m,:] #10 dim, diff of output

        d2 = w2.T.dot(d1).reshape([nh2,nh1,d_pl2*d_pl2]) #diff of 2nd pooling  

        d3 = np.zeros([nh2,nh1,d_cv2*d_cv2]) # diff of 2nd Relu

        for k in range(nh1):

            for l in range(nh2):

                d3[l,k,:] = wp2[l,k,:,:].T.dot(d2[l,k,:]) 

        d4 = np.zeros([nh2,nh1,d_cv2*d_cv2]) #diff of 2nd convolution

        for k in range(nh1):

            for l in range(nh2):

                d4[l,k,:] = pos2[l,k,:]*d3[l,k,:] 

        d4_rs = d4.reshape([nh2,nh1,d_cv2,d_cv2])   

        d5 = np.zeros([nh2,dh2,dh2]) #diff of 2nd filters



        dw2 = np.zeros([cl,nw])

        for i in range(cl):

            bo[i] -= eta*d1[i]

            for j in range(nw):

                dw2[i,j] = d1[i]*conm_rs[j]

        w2 -= eta*dw2

        

        zpd2_rs = zpd2.reshape([nh1,d_pd2,d_pd2])

        for k in range(nh1):

            for l in range(nh2):

                for i in range(d_cv2):

                    for j in range(d_cv2):

                        bc2[l] -= eta * d4_rs[l,k,i,j]

                        for s in range(dh2):

                            for t in range(dh2):

                                d5[l,s,t] += d4_rs[l,k,i,j]*zpd2_rs[k,i+s,j+t]

        H2 -= eta*d5

        

#back propagation of 1st layer

        d6_dum = np.zeros([nh1,d_pd2,d_pd2]) # diff of 2nd padding

        d6 = np.zeros([nh1,d_pl,d_pl]) #diff of 1st pooling

        for k in range(nh1):

            for l in range(nh2):

                d6_dum[k,:,:] += wc2[l,k,:,:].T.dot(d4[l,k,:]).reshape([d_pd2,d_pd2])

            for i in range(d_pl):

                for j in range(d_pl):

                    d6[k,i,j] = d6_dum[k,i+pd,j+pd]

        d6_rs = d6.reshape([nh1,d_pl*d_pl])

        d7 = np.zeros([nh1,d_cv*d_cv]) # diff of 1st Relu

        for k in range(nh1):

            d7[k,:] = wp1[k,:,:].T.dot(d6_rs[k,:]) 

        d8 = np.zeros([nh1, d_cv*d_cv]) # diff of 1st convolution

        for k in range(nh1):        

            d8[k,:] = pos1[k,:]*d7[k,:] 

        d8_rs = d8.reshape([nh1,d_cv,d_cv]) # 26x26 dim, d of Convolution   

        d9 = np.zeros([nh1,dh1,dh1]) # diff of 1st filters



        zpd1_rs = zpd1.reshape([d_pd,d_pd])

        for k in range(nh1):

            for i in range(d_cv):

                for j in range(d_cv):

                    bc1[k] -= eta * d8_rs[k,i,j]

                    for s in range(dh1):

                        for t in range(dh1):

                            d9[k,s,t] +=  d8_rs[k,i,j]*zpd1_rs[i+s,j+t]

        H1 -= eta*d9

 

    elapsed_time = time.time() - start

    print("time", elapsed_time)

    

    print('Y_train',Y_train[(train_cnt-10):train_cnt])

    print('T_train',T_n[(train_cnt-10):train_cnt].T)

    print('correct',correct/train_cnt)



#check with test data

    X_test = np.loadtxt("../input/test.csv",delimiter=",", skiprows=1)

    n_test = len(X_test[:,0]) # the amount of Y_test

    X_test /= X_test.max() #adjust maximum as 1 

    test_cnt =  10#n_test



    Y_test = np.zeros(test_cnt)

    for a in range(test_cnt):

        m = int(a)

        x = X_test[m]

        zpd1 = zpadding(x,d,d,pd)   # zero padding 28x28 -> 36x36  

        for k in range(nh1): # 1st convolution layers

            wc1[k,:,:] = Wconv(H1[k,:,:],d_pd,d_pd)

            conv_dum1 = wc1[k,:,:].dot(zpd1[0,:]).reshape([1,d_cv*d_cv]) + B1[k,:]

            conv1[k,:], pos1[k,:] = relu(conv_dum1)  # 1st convoluition

            pool1[k,:], wp1[k,:,:] = pooling(conv1[k,:],d_cv,d_cv, pl, st) # 1st pooling 

            conm1[k,:]= connorm(pool1[k,:])

            zpd2[k,:] = zpadding(conm1[k,:],d_pl,d_pl,pd)

            for l in range(nh2): # 2nd convolution layers

                wc2[l,k,:,:] = Wconv(H2[l,:,:],d_pd2,d_pd2)

                conv_dum2 = wc2[l,k,:,:].dot(zpd2[0,:]).reshape([1,d_cv2*d_cv2]) + B2[l,:] #2nd Relu

                conv2[l,k,:], pos2[l,k,:] = relu(conv_dum2)  # 2nd convoluition

                pool2[l,k,:], wp2[l,k,:,:] = pooling(conv2[l,k,:],d_cv2,d_cv2, pl, st) # 2nd pooling       

                conm2[l,k,:]= connorm(pool2[l,k,:])

                

        conm_rs = conm2.reshape([nh2*nh1*d_pl2*d_pl2])

        nw = conm_rs.size

        out_dum = w2.dot(conm_rs[:])+bo[:]

        y = softmax(out_dum) # 10-dim

        

        buf = 0

        for i in range(cl):

            if y[i] > buf:

                buf = y[i]

                maxc = i

        Y_test[a] = maxc

        

    print('out put label for test data, last 10')

    print(Y_test[(test_cnt-10):(test_cnt)])

    

    f = open('out.csv', 'w')

    writer = csv.writer(f)

    writer.writerows([['ImageId','Label']])

    for i in range(test_cnt):

        writer.writerows([[i+1,int(Y_test[i])]])

    f.close()



    print('last one of X_test is ',Y_test[test_cnt-1], '. Is it correct??')

    sns.heatmap(X_test[test_cnt-1,:].reshape(28,28))
if __name__ == '__main__':

    main()