import pathlib
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Glob the training data and load a single image path
training_paths = pathlib.Path('../input/stage1_train').glob('*/images/*.png')

training_sorted = sorted([x for x in training_paths])
im_path = training_sorted[45]
im = imageio.imread(str(im_path))
print("Total training images:%i"%len(training_sorted))
print('im :'+str(format(im.shape)))
from matplotlib.colors import ListedColormap
np.random.seed(1)
lc=(np.random.rand(256,3)+0.4)/1.4
lc[0]=[0,0,0]
rand_cmap = ListedColormap(lc)

# Función para obtener la región donde hay células
# en una  máscara a partir del directorio de la imagen
def getMask(im_path):
    yy=(im_path.parents[1]/'masks').glob('*.png')
    yy=[x for x in yy]
    im_mask=imageio.imread(str(yy[0]))
    for dirIm in yy:
        im_t=imageio.imread(str(dirIm))
        im_mask=im_t | im_mask
    return im_mask
# Función para obtener las regiones donde hay células
# en una sola imagen a partir del directorio de la imagen
def getRegion(im_path):
    yy=(im_path.parents[1]/'masks').glob('*.png')
    yy=[x for x in yy]
    im_labels=imageio.imread(str(yy[0]))
    im_labels= im_labels//255
    for i,dirIm in enumerate(yy[1:]):
        im_t=imageio.imread(str(dirIm))
        im_labels=(im_t//255)*(i+1) + im_labels
    return (im_labels, len(yy))

#Función para graficar una célula dada la dirección de la imagen
# la dirección es un path (pathlib)
def plotImg(im_path, figsize=(12,9),cmaps=('gnuplot', 'tab20', 'brg')):
    im = imageio.imread(str(im_path))
    im_mask= getMask(im_path)
    im_regiones, nlabels = getRegion(im_path)
    fondo= im[:,:,1] & (~ im_mask )
    plt.figure(figsize=figsize)
    plt.subplot(2,2,1)
    plt.imshow(im[:,:,1],cmap=cmaps[0],norm=None,vmin=0,vmax=255)
    plt.axis('off')
    plt.title('Original Image')
    plt.subplot(2,2,2)
    plt.imshow(fondo, cmap=cmaps[0],norm=None, vmin=0,vmax=255)
    plt.axis('off')
    plt.title('Background')
    plt.subplot(2,2,3)
    plt.imshow(im_regiones, cmap=rand_cmap)
    plt.axis('off')
    plt.title('Regions (%s Nuclei)'%nlabels)
    plt.subplot(2,2,4)
    plt.imshow(im_mask, cmap=rand_cmap)
    plt.axis('off')
    plt.title('Mask')
print('Done')
im_path = training_sorted[0]
plotImg(im_path)
from skimage.color import rgb2gray
from scipy import signal
from skimage.filters import threshold_otsu

# Obtiene la imagen normalizada en escala de grises
def getGrayImg(im_path):
    im = imageio.imread(str(im_path))
    im_gray = rgb2gray(im)
    if im_gray.mean() > 0.5:
        return 1-im_gray
    return im_gray
#Descripción de un array
def describe(array, name='array', show=True, otsu=False):
    datos=(1,1,1)
    if otsu:
        datos=(threshold_otsu(array), array.min(),array.max())
    else:
        datos=(array.mean(), array.min(),array.max())
    if show:
        print(name, '\n  mean:', datos[0],'\n  min:', datos[1],'\n  max:', datos[2])
    return datos
#Obtiene un kernel ponderando por distancias
def getDistKernel(radio=1):
    row=[i for i in range(-radio,radio+1)]
    a=np.array([row for i in range(radio*2+1)])
    b=a.transpose()
    d= np.multiply(a,a)+np.multiply(b,b)
    vmax=d.max()
    d=((d/vmax)-1)*(-1)    
    return d
# binarización de imágenes con una imágen de referencia
def binarizaMedias(im_original, im_umbral,factor=0.5,otsu=False):
    meanA,maxA,minA=describe(im_original,show=False,otsu=otsu)
    meanB,maxB,minB=describe(im_umbral,show=False,otsu=otsu)
    imA= ((im_original-minA)/(maxA-minA))-(meanA-minA)/(maxA-minA)
    imB=((im_umbral-minB)/(maxB-minB))-(meanB-minB)/(maxB-minB)
    #describe(imA,'Original')
    #describe(imB,'Umbral')
    return np.where(imA>imB*factor,0,255)
# Mide la exactitud de coincidencia entre dos imagenes binarizadas
def mideExactitud(im_ideal, im_calculada, imagen=False, mensaje=False):
    dif=np.zeros(im_ideal.shape,dtype=np.uint)
    dif=im_ideal^im_calculada
    AreaMascara=np.sum(im_calculada==255)
    v= np.sum(dif==255)/AreaMascara
    if mensaje:
        print('Exactitud', 1-v)
    if imagen:
        return(v, dif)
    return v

print("Done")
# Realiza analisis en multiples parametros para binarizar una imagen
# se da una tupla con los pares de factor, radio del kernel
def analizaParametros(im_path,factores=(0.4,0.5), radios=(1,2,3,4)):
    im_gray= getGrayImg(im_path)
    im_mask= getMask(im_path)
    resultados=[]
    for radio in radios:
        kernel=getDistKernel(radio)
        normal_result = signal.convolve2d(im_gray,kernel, boundary='symm', mode='same')
        for factor in factores:
            img_bin=binarizaMedias(im_gray,normal_result,factor=factor,otsu=False)
            img_binOtsu=binarizaMedias(im_gray,normal_result,factor=factor,otsu=True)
            #print('Shape img_bin:',img_bin.shape, 'Shape im_mask:', im_mask.shape)
            exact=mideExactitud(im_mask,img_bin)
            exactOtsu=mideExactitud(im_mask,img_binOtsu)
            resultados.append([factor,radio,exact,exactOtsu]) 
    return resultados
print("Done")
def getMejor(resultados, mensaje=True):
    rMean=[ex[2]for ex in resultados]
    rOtsu=[ex[3]for ex in resultados]
    iMean=rMean.index(min(rMean))
    iOtsu=rOtsu.index(min(rOtsu))
    
    mejorMean=(resultados[iMean][1],resultados[iMean][0] , 1-resultados[iMean][2])
    mejorOtsu=(resultados[iOtsu][1],resultados[iOtsu][0] , 1-resultados[iOtsu][3])
    if mensaje:
        print('Mejor exactitud (radio,factor):\n','\tMean: (%i,%f) = %f \n'%mejorMean,'\tOtsu: (%i,%f) = %f '%mejorOtsu, )
    return (mejorMean,mejorOtsu)
print('Done')
im_path = training_sorted[5]
factores=np.linspace(0.0,0.9,20)
radios=[1,2,3,4,5,6,7,8,9,10]
#factores=[0,0.1,0.2,0.3,0.4]
exact= analizaParametros(im_path,factores=factores ,radios=radios)
exactitud=[(ex[-2],ex[-1])for ex in exact]
getMejor(exact)
if len(radios)>1:
    im_exact=np.zeros((len(factores),len(radios)))
    im_exactOtsu=np.zeros((len(factores),len(radios)))
    for f,r,em,eo in exact:
        ix=np.where(factores==f)
        iy=radios.index(r)
        im_exact[ix,iy]=1-em
        im_exactOtsu[ix,iy]=1-eo
    valoresEjes=[min(radios),max(radios),max(factores),min(factores)]
    fig = plt.figure(1,figsize=(10,5))
    ax = fig.add_subplot(1,2,1)#Impares
    fig.colorbar(ax.imshow(im_exact,cmap='inferno',extent=valoresEjes,aspect=15))
    ax.set_adjustable('box-forced')
    ax.set_title('Exactitud por media')
    plt.ylabel('factor')
    plt.xlabel('radio')
    ax = fig.add_subplot(1,2,2)#Impares
    fig.colorbar(ax.imshow(im_exactOtsu,cmap='inferno',extent=valoresEjes,aspect=15))
    #ax[1].imshow(, cmap=plt.cm.gray)
    ax.set_title('Exactitud por Otsu')
    ax.set_adjustable('box-forced')
    plt.ylabel('Factor')
    plt.xlabel('Fadio')
    
else:
    #print(pares)
    plt.figure(figsize=(10,5))
    plt.plot(factores,exactitud, label=['a','b'])
    plt.grid()
    plt.show()

im_path = training_sorted[5]
cmaps=('gnuplot','gray')
figsize=(10,10)
imgray= getGrayImg(im_path)
im_mask= getMask(im_path)

radio=1
factor=0.7
otsu=True
kernel=getDistKernel(radio)
normal_result = signal.convolve2d(imgray,kernel, boundary='symm', mode='same')

img_bin=binarizaMedias(imgray,normal_result,factor=factor,otsu=otsu)
ex,img_dif=mideExactitud(im_mask,img_bin,imagen=True)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize,
                         sharex=True, sharey=True)
ax = axes.ravel()
titles = ['Original', 'Ideal', 'Diferencia','Binarizado']
imgs = [imgray, im_mask, img_dif, img_bin  ]
for n in range(0, len(imgs)):
    ax[n].imshow(imgs[n], cmap=plt.cm.gray)
    ax[n].set_title(titles[n])
    ax[n].set_adjustable('box-forced')
    #ax[n].axis('off')

plt.tight_layout()
print(imgray.shape, normal_result.shape)
print('  Factor:', factor, '  radio:', radio, '  Otsu:',otsu , ' Exactitud:',1-ex)

plt.show()

factores=np.linspace(0.0,0.9,20)
radios=[1,2,3,4,5,6,7,8,9,10]
columns=['Otsu','Radio','Factor','Exactitud']
df = pd.DataFrame(columns=columns)
for i,im_path in enumerate(training_sorted[:20]):
    #factores=[0,0.1,0.2,0.3,0.4]
    exact= analizaParametros(im_path,factores=factores ,radios=radios)
    mejorMean,mejorOtsu=getMejor(exact, mensaje=False)
    if mejorMean[-1]>mejorOtsu[-1]:
        tipoRef=(0,mejorMean[0],mejorMean[1],mejorMean[2])
    else:
        tipoRef=(1,mejorOtsu[0],mejorOtsu[1],mejorOtsu[2])
    df.loc[i]= tipoRef

path=pathlib.Path('featuresA.csv')
df.to_csv(path)
df.head(5)