import pathlib
import imageio
import numpy as np
import pandas as pd

# Glob the training data and load a single image path
training_paths = pathlib.Path('../input/stage1_train').glob('*/images/*.png')

training_sorted = sorted([x for x in training_paths])
im_path = training_sorted[45]
im = imageio.imread(str(im_path))
print("Total training images:%i"%len(training_sorted))
print('im :'+str(format(im.shape)))
#Función para obtener la mascara en una sola imagen a partir del directorio de la imagen
def getMask(im_path):
    yy=(im_path.parents[1]/'masks').glob('*.png')
    yy=[x for x in yy]
    im_mask=imageio.imread(str(yy[0]))
    for dirIm in yy:
        im_t=imageio.imread(str(dirIm))
        im_mask=im_t | im_mask
    return im_mask
print('Done')
import matplotlib.pyplot as plt
im_path = training_sorted[0]
im = imageio.imread(str(im_path))
im_mask= getMask(im_path)
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.imshow(im[:,:,1],cmap='gray')
plt.axis('off')
plt.title('Original Image')

plt.subplot(1,3,2)
plt.imshow(im_mask)
plt.axis('off')
plt.title('Mask')

# fondo
fondo= im[:,:,1] & (~ im_mask )
plt.subplot(1,3,3)
plt.imshow(fondo)
plt.axis('off')
plt.title('fondo')

plt.tight_layout()
plt.show()
# Lectura de las imagenes y las máscaras
tipos={1:'gray',3:'rgb',4:'rgba'}
tIm=[]
imagenes=[]
mascaras=[]
for dfile in training_sorted:
    im=imageio.imread(str(dfile))
    imagenes.append(im)
    mascaras.append(getMask(dfile))
    height, width, levels = im.shape
    m=[int(im[:,:,c].mean()) for c in range(levels)]
    typeIm='';
    if m[0]== m[1] and m[1] == m[2]:
        typeIm=tipos[1]
    else:
        typeIm=tipos[3]
    tIm.append((width,height,typeIm))
print('Done')
dfTypes=pd.DataFrame(tIm,columns=['width','height','type'])
agrupaciones=dfTypes.groupby(['width','height'])
dimensiones=agrupaciones.groups.keys()
cantidades=agrupaciones.size()
for i, t in enumerate(list(dimensiones)):
    print('%i-width x height: %s   total:%i'%(i+1,str(t),cantidades[i]))
print('gray :%i'%len(dfTypes[dfTypes.type=='gray']))
print('rgb :%i'%len(dfTypes[dfTypes.type=='rgb']))

dfTypes.head(4)
imRgb=[imagenes[i] for i in dfTypes.index[dfTypes.type=='rgb'].tolist()]
imRgbMask=[mascaras[i] for i in dfTypes.index[dfTypes.type=='rgb'].tolist()]
print('Total imágenes RGB: %i'%len(imRgb))
imRgb[0].shape
imRgbMask[0].shape
for i,img in enumerate( imRgb):
    v=[ img[:,:,c].mean() for c in range(3)]
    v=sum(v)/len(v)
    if v>128:
        imRgb[i]=255-img
ch=imRgb[0][:,:,1]
media= ch.mean()
vmin=np.amin(ch)
vmax=np.amax(ch)
hist, bins= np.histogram(ch,256)
hmaxA=hist[0:128].argmax()
hmaxB=hist[128:].argmax()+128
# Grafico del histograma
plt.bar(bins[:-1],hist, width=1)
plt.show()
print('media: %.2f   vmin: %i    vmax: %i   hmaxA: %i   hmaxB: %i'%(media, vmin, vmax, hmaxA,hmaxB))
def descriptoresCh(ch):
    hist, bins= np.histogram(ch,256)
    hmaxA=hist[0:128].argmax()
    hmaxB=hist[128:].argmax()+128
    vmax=np.amax(ch)
    vmin=np.amin(ch)
    media=ch.mean()
    return [media, vmin, vmax, hmaxA,hmaxB]
descriptores=[]
for i,img in enumerate( imRgb):
    di=[]
    di.extend(descriptoresCh(img[:,:,0]))
    di.extend(descriptoresCh(img[:,:,1]))
    di.extend(descriptoresCh(img[:,:,2]))
    descriptores.append(di)
col=[]
for ch in ('r','g', 'b'):
    for d in ['media', 'vmin', 'vmax', 'hmaxA', 'hmaxB']:
        col.append('%s_%s'%(ch,d))
descriptores= pd.DataFrame(descriptores, columns=col)
descriptores.head(3)
from skimage.filters import threshold_otsu
imc=imRgb[0]
msk=imRgbMask[0]
nmsk=np.sum(msk==255)
print(np.amax(msk))
print(imc.shape)
ttd=[]
nwmsk=[]
dfmsk=[]
for c in range(3):
    imch=imc[:,:,c]
    thresh_val = threshold_otsu(imch)
    maskCh = np.where(imch > thresh_val, 255, 0)
    dif= maskCh ^ msk
    v= np.sum(dif==255)/nmsk
    nwmsk.append(maskCh)
    dfmsk.append(dif)
    ttd.append(v)
print(ttd)
plt.figure(figsize=(14,12))
for i, chName in enumerate(['R','G', 'B']):
    plt.subplot(3,3,i+1)
    plt.imshow(imc[:,:,i],cmap='gray')
    plt.axis('off')
    plt.title('Canal %s'%chName)
for i, chName in enumerate(['R','G', 'B']):
    plt.subplot(3,3,i+4)
    plt.imshow(nwmsk[i],cmap='gray')
    plt.axis('off')
    plt.title('Canal %s'%chName)
for i, chName in enumerate(['R','G', 'B']):
    plt.subplot(3,3,i+7)
    plt.imshow(dfmsk[i],cmap='gray')
    plt.axis('off')
    plt.title('Canal %s'%chName)
plt.show()
plt.figure(figsize=(4,4))
plt.imshow(msk,cmap='gray')
plt.axis('off')
plt.title('Mascara')
plt.show()
# Se escogerá el mejor canal en cada imagen a color
# Se indica el canal con un indice 0 - R, 1 - G, 2 - B.
mejorCH=[]
for i,img in enumerate( imRgb):
    imc=img
    msk=imRgbMask[i]
    nmsk=np.sum(msk==255)
    ttd=[] #Donde se almacena el porcentaje de no aciertos, que idealmente debe ser cero
    for c in range(3):
        imch=imc[:,:,c]
        thresh_val = threshold_otsu(imch)
        maskCh = np.where(imch > thresh_val, 255, 0)
        dif= maskCh ^ msk
        v= np.sum(dif==255)/nmsk
        ttd.append(v)
    mejorCH.append(ttd.index(min(ttd)))

fig = plt.figure(1,figsize=(7,4))
plt.plot(mejorCH)
plt.show()
np.histogram(mejorCH,[-1,0.5,1.5,2.5])
#Dado que el canal B no es muy común, y que solo hay dos casos, se descartará y se dejarán solo dos canales, se asume que para estos dos casos
#El mejor canal es el verde G.
mejorCH[mejorCH.index(2)]=1
mejorCH[mejorCH.index(2)]=1
print(mejorCH)
print(type(mejorCH[0]))
dtf=descriptores
dtf['canal']=pd.Series(mejorCH)
dtf.head(5)
from sklearn.model_selection import train_test_split
train, test = train_test_split(dtf, test_size = 0.2,random_state=3)
print('(# Datos, # descriptores)\n Entranimento: %s\n Test: %s'%(str(train.shape),str(test.shape)))
descriptores=[]
for ch in ('r_','g_','b_'):
    descriptores.append(ch+'media')
    descriptores.append(ch+'vmin')
    descriptores.append(ch+'vmax')
    descriptores.append(ch+'hmaxA')
    descriptores.append(ch+'hmaxB')
train_X = train[descriptores]
train_y=train.canal
test_X= test[descriptores] 
test_y =test.canal  
print(len(test_y))
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics #for checking the model accuracy
modeloClasificador = GaussianNB()
modeloClasificador.fit(train_X, train_y)
prediccion=modeloClasificador.predict(test_X) 
print('The accuracy of the GaussianNB is:',metrics.accuracy_score(prediccion,test_y))
a=test_X.iloc[[3]]
b= a.values.tolist()
print(b)
#print(test_y.iloc[:])

x=modeloClasificador.predict(b)
print(x[0])
# Print the image dimensions
print('Original image shape: {}'.format(im.shape))

# Coerce the image into grayscale format (if not already)
from skimage.color import rgb2gray
im_gray = rgb2gray(im)
print('New image shape: {}'.format(im_gray.shape))
# Now, let's plot the data
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(im)
plt.axis('off')
plt.title('Original Image')

plt.subplot(1,2,2)
plt.imshow(im_gray, cmap='gray')
plt.axis('off')
plt.title('Grayscale Image')

plt.tight_layout()
plt.show()
from skimage.filters import threshold_otsu
thresh_val = threshold_otsu(im_gray)
mask = np.where(im_gray > thresh_val, 1, 0)

# Make sure the larger portion of the mask is considered background
if np.sum(mask==0) < np.sum(mask==1):
    mask = np.where(mask, 0, 1)
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
im_pixels = im_gray.flatten()
plt.hist(im_pixels,bins=50)
plt.vlines(thresh_val, 0, 100000, linestyle='--')
plt.ylim([0,50000])
plt.title('Grayscale Histogram')

plt.subplot(1,2,2)
mask_for_display = np.where(mask, mask, np.nan)
plt.imshow(im_gray, cmap='gray')
plt.imshow(mask_for_display, cmap='rainbow', alpha=0.5)
plt.axis('off')
plt.title('Image w/ Mask')

plt.show()
from scipy import ndimage
labels, nlabels = ndimage.label(mask)

label_arrays = []
for label_num in range(1, nlabels+1):
    label_mask = np.where(labels == label_num, 1, 0)
    label_arrays.append(label_mask)

print('There are {} separate components / objects detected.'.format(nlabels))
# Create a random colormap
from matplotlib.colors import ListedColormap
rand_cmap = ListedColormap(np.random.rand(256,3))

labels_for_display = np.where(labels > 0, labels, np.nan)
plt.imshow(im_gray, cmap='gray')
plt.imshow(labels_for_display, cmap=rand_cmap)
plt.axis('off')
plt.title('Labeled Cells ({} Nuclei)'.format(nlabels))
plt.show()
for label_ind, label_coords in enumerate(ndimage.find_objects(labels)):
    cell = im_gray[label_coords]
    
    # Check if the label size is too small
    if np.product(cell.shape) < 10: 
        print('Label {} is too small! Setting to 0.'.format(label_ind))
        mask = np.where(labels==label_ind+1, 0, mask)

# Regenerate the labels
labels, nlabels = ndimage.label(mask)
print('There are now {} separate components / objects detected.'.format(nlabels))
fig, axes = plt.subplots(1,6, figsize=(10,6))

for ii, obj_indices in enumerate(ndimage.find_objects(labels)[0:6]):
    cell = im_gray[obj_indices]
    axes[ii].imshow(cell, cmap='gray')
    axes[ii].axis('off')
    axes[ii].set_title('Label #{}\nSize: {}'.format(ii+1, cell.shape))

plt.tight_layout()
plt.show()
# Get the object indices, and perform a binary opening procedure
two_cell_indices = ndimage.find_objects(labels)[1]
cell_mask = mask[two_cell_indices]
cell_mask_opened = ndimage.binary_opening(cell_mask, iterations=8)
fig, axes = plt.subplots(1,4, figsize=(12,4))

axes[0].imshow(im_gray[two_cell_indices], cmap='gray')
axes[0].set_title('Original object')
axes[1].imshow(mask[two_cell_indices], cmap='gray')
axes[1].set_title('Original mask')
axes[2].imshow(cell_mask_opened, cmap='gray')
axes[2].set_title('Opened mask')
axes[3].imshow(im_gray[two_cell_indices]*cell_mask_opened, cmap='gray')
axes[3].set_title('Opened object')


for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])

print('RLE Encoding for the current mask is: {}'.format(rle_encoding(label_mask)))
## Funciones para convertir adecuadamente una imagen a escala de grises
def descriptoresCh(ch):
    hist, bins= np.histogram(ch,256)
    hmaxA=hist[0:128].argmax()
    hmaxB=hist[128:].argmax()+128
    vmax=np.amax(ch)
    vmin=np.amin(ch)
    media=ch.mean()
    return [media, vmin, vmax, hmaxA,hmaxB]

#Funcion para convertir a escala de grises, usando el clasificador
def rgbToGray(im):
    height, width, levels = im.shape
    m=[int(im[:,:,c].mean()) for c in range(levels)]
    typeIm='';
    if m[0]== m[1] and m[1] == m[2]:
        #La imagen esta en escala de grises
        imG=im[:,:,0] #Toma un solo canal
    else:
        #En caso contrario usa el clasificador
        di=[]#Descriptores de los diferentes canales
        di.extend(descriptoresCh(img[:,:,0]))
        di.extend(descriptoresCh(img[:,:,1]))
        di.extend(descriptoresCh(img[:,:,2]))
        x=modeloClasificador.predict([di])
        imG=im[:,:,x[0]] #Toma un solo canal
    return imG
            
import pandas as pd

def analyze_image(im_path):
    '''
    Take an image_path (pathlib.Path object), preprocess and label it, extract the RLE strings 
    and dump it into a Pandas DataFrame.
    '''
    # Read in data and convert to grayscale
    im_id = im_path.parts[-3]
    im = imageio.imread(str(im_path))
    #im_gray = rgb2gray(im)
    im_gray = rgbToGray(im)#Cambio

    # Mask out background and extract connected objects
    thresh_val = threshold_otsu(im_gray)
    mask = np.where(im_gray > thresh_val, 1, 0)
    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)    
        labels, nlabels = ndimage.label(mask)
    labels, nlabels = ndimage.label(mask)
    
    # Loop through labels and add each to a DataFrame
    im_df = pd.DataFrame()
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        if label_mask.flatten().sum() > 10:
            rle = rle_encoding(label_mask)
            s = pd.Series({'ImageId': im_id, 'EncodedPixels': rle})
            im_df = im_df.append(s, ignore_index=True)
    
    return im_df


def analyze_list_of_images(im_path_list):
    '''
    Takes a list of image paths (pathlib.Path objects), analyzes each,
    and returns a submission-ready DataFrame.'''
    all_df = pd.DataFrame()
    for im_path in im_path_list:
        im_df = analyze_image(im_path)
        all_df = all_df.append(im_df, ignore_index=True)
    
    return all_df
testing = pathlib.Path('../input/stage1_test/').glob('*/images/*.png')
df = analyze_list_of_images(list(testing))
df.to_csv('submission.csv', index=None)
