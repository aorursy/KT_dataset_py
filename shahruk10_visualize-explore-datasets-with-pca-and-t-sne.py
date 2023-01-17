# basic libraries required for i/o and processing
import glob
import cv2
import numpy as np

# importing plotly and setting up offline notebook
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

# importing libraries from sklearn to apply PCA and TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
def loadLabels(databases, dataFolder = '../input'):
    
    '''
     Helper function to facilitate easy loading of data labels
     given the name of the training set. You can append data labels
     from several csv files easily by passing all of them as a list
     
     e.g : data = loadLabels(['training-a','training-b'])
    '''
    
    # using a dictionary to store filepaths and labels
    data = {}
    
    for db in databases:
        
        # data directory followed by labels file name
        labelsPath = dataFolder + '/%s.csv' % db
        labels = np.genfromtxt(labelsPath, delimiter=',', dtype=str)

        # field names in first row (row 0)
        fields = labels[0].tolist()

        # getting all rows from row 1 onwards for filename and digit column 
        fileNames = labels[1:, fields.index('filename')]
        digits = labels[1:, fields.index('digit')]

        # creating a dictionary - will come in handy later to pick and choose digits !        
        for (fname, dgt) in zip(fileNames, digits):
            data[fname] = {}
            data[fname]['path'] = dataFolder + '/%s/%s' % (db,fname)
            data[fname]['label'] = int(dgt)
    
    return data
# data sets to analyze
db = ['training-a']

# loading data from csv file
print("Loading Labels for %s" % " ".join(db))
dataLabels = loadLabels(db)

# randomly selecting n samples from dataset; WARNING : large no. of samples may take a while to process
n = 6000
samples =  np.random.choice( list(dataLabels.keys()), n if n<len(dataLabels) else len(dataLabels), replace=False)

# loading selected images (as grayscale) along with labels
print("Loading Images")
images = [cv2.imread(dataLabels[fname]['path'],0) for fname in samples]
labels = [dataLabels[fname]['label'] for fname in samples]

# annotations are filenames; used to label points on scatter plot below
annots = [fname for fname in samples]
def process(img, crop=True, m = 5):
    '''
    This function takes care of all pre-processing, from denoising to 
    asserting black backgrounds. If the crop flag is passed, the 
    image is cropped to its approximate bounding box (bbox). 
    The m parameter (pixels) adds a some extra witdh/height to the bbox 
    to ensure full digit is contained within it.
    '''
    # blurring - removes high freq noise
    img2 = cv2.medianBlur(img,3)

    # threshold image
    thresh, img3 = cv2.threshold(img2, 128, 255, cv2.THRESH_OTSU)
    
    # ensure backroung is black
    # if no. of white pixels is majority => bg = white, need to invert
    if len(img3[img3==255]) > len(img3[img3==0]):
        img3 = cv2.bitwise_not(img3)

    if crop:      
        # getting contours on the image  
        _,contours,_ = cv2.findContours(img3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # get largest contour by area
        cMax = max(contours, key=cv2.contourArea) 

        # get box that encloses the largest contour -> bounding box
        x,y,w,h = cv2.boundingRect(cMax)

        # adding margins
        x1 = (x-m) if (x-m)>0 else x
        y1 = (y-m) if (y-m)>0 else y
        
        x2 = (x+w+m) if (x+w+m)<img.shape[1] else (x+w)
        y2 = (y+h+m) if (y+h+m)<img.shape[0] else (y+h)

        # cropping
        img3 = img3[ y1:y2, x1:x2 ]

    return img3
# processing images - cleaning, thresholding, cropping
print("Processing Images")
processed = [process(img) for img in images]

# resizing down to model input size : (32,32) for conveinice, images processed quickly 
print("Resizing")
imagesR = [ cv2.resize(img, (32,32)) for img in images ]
processedR = [ cv2.resize(img, (32,32)) for img in processed ]

# flattening all images (2D) to 1D array; i.e simply taking each
# row of each image and stacking next to each other
print("Flattening")
imagesF = [ img.flatten() for img in imagesR ]
processedF = [ img.flatten() for img in processedR ]

# normalizing pixel values
print("Normalizing")
imagesN = StandardScaler().fit_transform(imagesF)
processedN = StandardScaler().fit_transform(processedF)

# Running PCA on scaled orginal images and processed; generating 3 components;
print("Performing PCA for 3 components")
pca = PCA(n_components=3)
pca0 = pca.fit_transform(imagesN)
pca0 = StandardScaler().fit_transform(pca0)

pca = PCA(n_components=3)
pca1 = pca.fit_transform(processedN)
pca1 = StandardScaler().fit_transform(pca1)
def plotly3D(data, labels, annotations= None, title = 't-SNE Plot'):
    '''
    This function takes in 3 dimensional data points 
    and plots them in a 3D scatter plot, color coded
    by their labels
    '''
    # getting unique classes from labels
    # in this case: 0-9
    nClasses = len(np.unique(labels))
    
    # we will plot points for each digit seperately 
    # and color coded; they be stored here
    points = []
    
    # going over each digit
    for label in np.unique(labels):
        
        # getting data points for that digit; coods must be column vectors
        x = data[np.where(labels == label), 0].reshape(-1,1)
        y = data[np.where(labels == label), 1].reshape(-1,1)
        z = data[np.where(labels == label), 2].reshape(-1,1)
        
        # adding file name to each point
        if annotations is not None:
            annotations = np.array(annotations)
            ptLabels = annotations[np.where(labels == label)]
        else:
            ptLabels = None
            
        # creating points in 3d space
        trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', text = ptLabels,
                             marker=dict(size=8, color=label, colorscale='Viridis', opacity=0.6))
        
        # adding plot to list of plots
        points.append(trace)
    
    # plotting all of the datapoints
    layout = dict(title = title,showlegend=True) 
    fig = go.Figure(data=points, layout=layout)
    py.iplot(fig)
# plotting t-SNE before and after processing
plotly3D(pca0, labels, annotations = annots, title = 'PCA before Processing')
plotly3D(pca1, labels, annotations = annots, title = 'PCA after Processing')
# Running PCA on scaled orginal images and processed; generating 20 components;
print("Performing PCA for 20 components")
pca = PCA(n_components=20)
pca0 = pca.fit_transform(imagesN)
pca0 = StandardScaler().fit_transform(pca0)

pca = PCA(n_components=20)
pca1 = pca.fit_transform(processedN)
pca1 = StandardScaler().fit_transform(pca1)
# Running t-SNE on PCA outputs; generating 3 dimensional data points
print("Calculating TSNE")
tsne = TSNE(n_components=3, perplexity=40, verbose=2, n_iter=500,early_exaggeration=1)
tsne0 = tsne.fit_transform(pca0)
tsne0 = StandardScaler().fit_transform(tsne0)

tsne = TSNE(n_components=3, perplexity=40, verbose=2, n_iter=500,early_exaggeration=1)
tsne1 = tsne.fit_transform(pca1)
tsne1 = StandardScaler().fit_transform(tsne1)
# plotting t-SNE before and after processing
plotly3D(tsne0, labels, annotations=annots, title = 't-SNE before Processing')
plotly3D(tsne1, labels, annotations=annots, title = 't-SNE after Processing')
