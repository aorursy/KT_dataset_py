import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%pylab inline
def plot_examples(d, x1,x2):
    fig, ax = subplots(1,1,figsize=(17,4))
    plt.subplot(141)
    pylab.imshow(d.values[4,:].reshape(x1,x2));
    plt.subplot(142)
    pylab.imshow(d.values[6,:].reshape(x1,x2));
    plt.subplot(143)
    pylab.imshow(d.values[7,:].reshape(x1,x2));
    plt.subplot(144)
    pylab.imshow(d.values[8,:].reshape(x1,x2));
# Rescalig image to new dimentions x_dim and y_dim
# Linear approximation of pixels brightness is used

def img_deform(im, y_dim, x_dim):  
    #print('im init shape', im.shape)
    
    # Scale image along width
    im_new = np.zeros((y_dim, im.shape[1]))
    for x in range(0, im.shape[1]):
        im_new[:,x] = pylab.interp(np.arange(0, im.shape[0], im.shape[0]*1./y_dim), range(im.shape[0]), im[:,x])
    
    # Scale image along height   
    im_new_2 = np.zeros((y_dim, x_dim))
    for y in range(0, y_dim):
        im_new_2[y,:] = pylab.interp(np.arange(0, im.shape[1], im.shape[1]*1./x_dim), range(im_new.shape[1]), im_new[y,:])        
    
    #print('im new shape', im_new_2.shape)
    return np.round(im_new_2,0)
# Cutting empty space from image: from bottom, top, left and right 
# delete rows and columns respectively while sum of pixels brightness is equal to zero

def im_cut(im):
    x0 = 0
    x1 = im.shape[1]+1
    y0 = 0
    y1 = im.shape[0]+1    
    s = 0
    for i in range(5):
        si = im[:,i].sum()
        if si==0:
            x0 += 1
        else:
            break
    s = 0
    for i in range(5):
        si = im[:,-i].sum()
        if si==0:
            x1 -= 1
        else:
            break
    s = 0
    for i in range(5):
        si = im[i,:].sum()
        if si==0:
            y0 += 1
        else:
            break
    s = 0
    for i in range(5):
        si = im[-i,:].sum()
        if si==0:
            y1 -= 1
        else:
            break
    im_new = im[y0:y1,x0:x1]
    #print('img_cut: '+str(im_new.shape))
    
    return im_new  
# Transform images with cutting and rescaling in cycle

def transform_imgs(d0, x0, x1=np.NaN, x2=np.NaN):
    d1=[]
    for n, img in enumerate(d0.values[:,:]):
        #print(n)
        img_new = img.reshape(x0,x0)
        if x1 is not np.NaN and x2 is not np.NaN:
            img_new = (im_cut(img_new))
            img_new = (img_deform(img_new, x1, x2))
        d1.append(list(ravel(img_new)))
    d1 = pd.DataFrame(data=d1)
    #print(d1.shape)
    return d1
# import DATA

d0 = pd.read_csv('../input/train.csv')
labs = d0.label
d0.drop('label', axis=1, inplace=True)
print(d0.shape)
d0.head()
d1 = transform_imgs(d0, 28)
d2 = transform_imgs(d0, 28, 14, 14)
plot_examples(d1, 28, 28)
plot_examples(d2, 14, 14)
from sklearn import model_selection as ms
x_train1, x_test1, y_train1, y_test1 = ms.train_test_split(d1.values, labs, random_state=0)
x_train2, x_test2, y_train2, y_test2 = ms.train_test_split(d2.values, labs, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=4, weights='distance')
%%time
from sklearn import metrics

clf.fit(x_train1, y_train1)
y_pred1 = clf.predict(x_test1)
print(metrics.classification_report(y_test1, y_pred1, digits=3))
%%time

clf.fit(x_train2, y_train2)
y_pred2 = clf.predict(x_test2)
print(metrics.classification_report(y_test2, y_pred2, digits=3))
## Take a glance at wrong classified images 

y = y_pred2
pyplot.figure(figsize(10, 6))
n = 0
for i in range(len(y)):
    if y[i]!=y_test2.values[i]:
        pyplot.subplot(2, 4, n+1)
        plt.title(str(y[i])+' ('+str(y_test2.values[i])+')')
        pylab.imshow(d2.values[y_test2.index[i],:].reshape(14,14))
        n += 1
    if n==8:
        break
d_submit = pd.read_csv('../input/test.csv')
d_submit.head()
d_submit = transform_imgs(d_submit, 28, 14, 14)
#y_submit = clf.predict(d_submit)
#submit_df = pd.DataFrame({'ImageId':d_submit.index, 'Label':y_submit})
#submit_df.to_csv('submit_data.csv',index=False)
