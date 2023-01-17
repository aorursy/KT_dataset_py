RUNALL = False
SKIP_SVM = False #SVM takes a HUUGE amount of time
SKIP_TSNE = False
SKIP_GRIDS = False

import numpy as np
import cv2
import time
import random
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import glob
import os
import sys
from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
#MulticoreTSNE is faster because ... it uses multicore
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.cluster import KMeans

import numpy
from IPython.display import HTML
from PIL import Image
from io import BytesIO
import base64

PCA_n_components = 28
RF_n_estimators = 20
width = 34
height = 34
mean = 0.068000
var = 0.048000

def load_images(width,height,mean,var):
    start = time.time()
    fruit_images = []
    fruit_images_noisy = []
    labels = []
    labels_noisy = []
    imgcount = 0
    for fruit_dir_path in glob.glob("../input/*/fruits-360/Training/*"):
        fruit_label = fruit_dir_path.split("/")[-1]
        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            image = cv2.resize(image, (width, height))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            fruit_images.append(image)
            labels.append(fruit_label)

            #get more training data to increase accuracy
            row,col,ch = image.shape
            sigma = var**0.5
            gauss = np.array(image.shape)
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            img2 = cv2.flip( image, 1 )
            img2 = img2 + img2 * gauss
            img2 = img2.astype('uint8')
            fruit_images_noisy.append(img2)
            labels_noisy.append(fruit_label)
            imgcount += 1
            #for fast testing noise, outcomment this
            #if imgcount>100:
            #    break
        #for fast testing noise, outcomment this
        #if imgcount>100:
            #break
        
    fruit_images = np.array(fruit_images)
    labels = np.array(labels)
    end = time.time()
    print("Loading %d images took %d s" % (imgcount,end-start))
    return (fruit_images,fruit_images_noisy,labels,labels_noisy)
RUNALL = True
(fruit_images,fruit_images_noisy,labels,labels_noisy) = load_images(width,height,mean,var)
label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
id_to_label_dict
label_ids = np.array([label_to_id_dict[x] for x in labels])
label_ids_combined = [*label_ids,*label_ids]
def plot_image_grid(images, nb_rows, nb_cols, figsize=(5, 5)):
    assert len(images) == nb_rows*nb_cols, "Number of images should be the same as (nb_rows*nb_cols)"
    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=figsize)
    
    n = 0
    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            # axs[i, j].xaxis.set_ticklabels([])
            # axs[i, j].yaxis.set_ticklabels([])
            axs[i, j].axis('off')
            axs[i, j].imshow(images[n])
            n += 1        
if not SKIP_GRIDS:
    plot_image_grid(fruit_images[0:100], 10, 10)
    plot_image_grid(fruit_images_noisy[0:100], 10, 10)
if not SKIP_GRIDS:
    plot_image_grid(fruit_images[2000:2400], 20, 20, figsize=(10,10))
    plot_image_grid(fruit_images_noisy[2000:2400], 20, 20, figsize=(10,10))
scaler = StandardScaler()
scaler_combined = StandardScaler()
start = time.time()

images_scaled = scaler.fit_transform([i.flatten() for i in fruit_images])

images_scaled_combined = scaler_combined.fit_transform([i.flatten() for i in [*fruit_images,*fruit_images_noisy]])

end = time.time()
print("StandardScaler took %d s" % (end-start))
start = time.time()

pca = PCA(n_components=PCA_n_components)
pca_result = pca.fit_transform(images_scaled)

pca_combined = PCA(n_components=PCA_n_components)
pca_result_combined = pca_combined.fit_transform(images_scaled_combined)

end = time.time()
print("PCA took %d s" % (end-start))
if not SKIP_TSNE:
    start = time.time()
    tsne = TSNE(n_jobs=4,n_components=2, perplexity=40.0)
    tsne_result = tsne.fit_transform(pca_result)
    tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
    end = time.time()
    print("TSNE took %d s" % (end-start))
def visualize_scatter(data_2d, label_ids, id_to_label_dict=None, figsize=(20,20)):
    if not id_to_label_dict:
        id_to_label_dict = {v:i for i,v in enumerate(np.unique(label_ids))}
    
    plt.figure(figsize=figsize)
    plt.grid()
    
    nb_classes = len(np.unique(label_ids))
    
    cmap = plt.cm.get_cmap("jet", nb_classes)
    
    for i, label_id in enumerate(np.unique(label_ids)):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    c= cmap(i),
                    linewidth='5',
                    alpha=0.8,
                    label=id_to_label_dict[label_id])
    #plt.legend(loc='best')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
          fancybox=True, shadow=True, ncol=1, fontsize=figsize[0])
def visualize_scatter_with_images(data_2d, images, figsize=(45,45), image_zoom=1):
    fig, ax = plt.subplots(figsize=figsize)
    plt.grid()
    artists = []
    for xy, i in zip(data_2d, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(data_2d)
    ax.autoscale()
    plt.show()
if not SKIP_TSNE:
    start = time.time()
    visualize_scatter(tsne_result_scaled, label_ids, id_to_label_dict, figsize=(25, 25))
    end = time.time()
    print("visualize_scatter took %d s" % (end-start))
if not SKIP_TSNE:
    start = time.time()
    visualize_scatter_with_images(tsne_result_scaled, fruit_images, image_zoom=0.4, figsize=(25, 25))
    end = time.time()
    print("visualize_scatter_with_images took %d s" % (end - start))
start = time.time()

X_train, X_test, y_train, y_test = train_test_split(pca_result, label_ids,stratify = label_ids, test_size=0.25, random_state=42)

X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(pca_result_combined, label_ids_combined,stratify = label_ids_combined, test_size=0.25, random_state=42)

end = time.time()
print("train_test_split took %d s" % (end - start))
start = time.time()

forest = RandomForestClassifier(n_estimators=RF_n_estimators,n_jobs=4,random_state=42)
forest = forest.fit(X_train, y_train)

forest_combined = RandomForestClassifier(n_estimators=RF_n_estimators,n_jobs=4,random_state=42)
forest_combined = forest_combined.fit(X_train_combined, y_train_combined)

end = time.time()
print("RandomForestClassifier took %d s" % (end - start))
start = time.time()

test_predictions = forest.predict(X_test)

test_predictions_combined = forest_combined.predict(X_test_combined)

end = time.time()
print("test_predictions took %d s" % (end - start))
start = time.time()

precision = accuracy_score(test_predictions, y_test) * 100

precision_combined = accuracy_score(test_predictions_combined, y_test_combined) * 100

end = time.time()
print("RandomForest accuracy_score took %d s" % (end - start))
print("Accuracy with RandomForest: {0:.6f}".format(precision))
print("Accuracy with RandomForest and noisy images: {0:.6f}".format(precision_combined))
if not SKIP_SVM:
    start = time.time()
    svm_clf = svm.SVC(gamma='auto')
    svm_clf = svm_clf.fit(X_train, y_train) 
    svm_clf_combined = svm.SVC(gamma='auto')
    svm_clf_combined = svm_clf_combined.fit(X_train_combined, y_train_combined) 
    end = time.time()
    print("SVM took %d s" % (end - start))
if not SKIP_SVM:
    start = time.time()
    test_predictions = svm_clf.predict(X_test)
    test_predictions_combined = svm_clf_combined.predict(X_test_combined)
    end = time.time()
    print("svm_clf.predict took %d s" % (end - start))
if not SKIP_SVM:
    start = time.time()
    precision = accuracy_score(test_predictions, y_test) * 100
    precision_combined = accuracy_score(test_predictions_combined, y_test_combined) * 100
    end = time.time()
    print("SVM accuracy_score took %d s" % (end - start))
    print("Accuracy with SVM: {0:.6f}".format(precision))
    print("Accuracy with SVM and noisy images: {0:.6f}".format(precision_combined))
def load_validation_images(label_to_id_dict,width,height):
    start = time.time()
    validation_fruit_images = []
    validation_image_paths = {}
    validation_labels = [] 
    for fruit_dir_path in glob.glob("../input/*/fruits-360/Test/*"):
        fruit_label = fruit_dir_path.split("/")[-1]
        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            image = cv2.resize(image, (width, height))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            validation_fruit_images.append(image)
            validation_labels.append(fruit_label)
            validation_image_paths[fruit_label] = image_path
    validation_fruit_images = np.array(validation_fruit_images)
    validation_labels = np.array(validation_labels)
    validation_label_ids = np.array([label_to_id_dict[x] for x in validation_labels])
    flattened_images = [i.flatten() for i in validation_fruit_images]
    print("validation_flattened_images len: %d" % len(flattened_images))
    return (flattened_images,validation_label_ids,validation_image_paths)
(validation_images,validation_label_ids,validation_image_paths) = load_validation_images(label_to_id_dict,width,height)

validation_images_scaled = scaler.transform(validation_images)
validation_pca_result = pca.transform(validation_images_scaled)

validation_images_scaled_combined = scaler_combined.transform(validation_images)
validation_pca_combined_result = pca_combined.transform(validation_images_scaled_combined)

end = time.time()
print("Validation preperation took %d s" % (end - start))

start = time.time()
test_predictions = forest.predict(validation_pca_result)
precision = accuracy_score(test_predictions, validation_label_ids) * 100

test_predictions_combined = forest_combined.predict(validation_pca_combined_result)
precision_combined = accuracy_score(test_predictions_combined, validation_label_ids) * 100
print("Validation Accuracy with Random Forest: {0:.6f}".format(precision))
print("Validation Accuracy with Random Forest and noisy images: {0:.6f}".format(precision_combined))
end = time.time()
print("Random Forest prediction took %d s" % (end - start))
def get_thumbnail(path):
    i = Image.open(path)
    i.thumbnail((150, 150), Image.LANCZOS)
    return i

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(path):
    return f'<img src="data:image/jpeg;base64,{image_base64(get_thumbnail(path))}">'

data = []
processed = []
unique, counts = numpy.unique(y_train, return_counts=True)
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
for t in range(0,len(test_predictions)):
    id = validation_label_ids[t]
    if not id in processed:
        test_predictions_sub =  []
        test_predictions_sub_combined =  []
        id_list = []
        id_list_combined = []
        for i in range(0,len(test_predictions)):
            if validation_label_ids[i] == id:
                test_predictions_sub.append(test_predictions[i])
                id_list.append(id)
        for i in range(0,len(test_predictions_combined)):
            if validation_label_ids[i] == id:
                test_predictions_sub_combined.append(test_predictions_combined[i])
                id_list_combined.append(id)
        accuracy = accuracy_score(test_predictions_sub,id_list)*100
        accuracy_WNI = accuracy_score(test_predictions_sub_combined,id_list_combined)*100
        data.append((id_to_label_dict[id],accuracy,accuracy_WNI,counts[id],validation_image_paths[id_to_label_dict[id]]))
        processed.append(id)
if not id in processed:
    accuracy = accuracy_score(test_predictions_sub,id_list)*100
    accuracy_WNI = accuracy_score(test_predictions_sub_combined,id_list_combined)*100
    data.append((id_to_label_dict[id],accuracy,accuracy_WNI,counts[id],validation_image_paths[id_to_label_dict[id]]))
data.sort(key=lambda x:x[2])
data.reverse()
pd.set_option('display.max_colwidth', -1)
df=pd.DataFrame(data, columns=["Label","Accuracy","Accuracy WNI","Count in y_train", "Image"])
HTML(df.to_html(escape=False ,formatters=dict(Image=image_formatter)))

if not SKIP_SVM:
    start = time.time()
    test_predictions = svm_clf.predict(validation_pca_result)
    test_predictions_combined = svm_clf_combined.predict(validation_pca_combined_result)
    precision = accuracy_score(test_predictions, validation_label_ids) * 100
    precision_combined = accuracy_score(test_predictions_combined, validation_label_ids) * 100
    end = time.time()
    print("Validation Accuracy with SVM: {0:.6f}".format(precision))
    print("Validation Accuracy with SVM and noisy images: {0:.6f}".format(precision_combined))
    print("SVM prediction took %d s" % (end - start))
if not RUNALL:
    scaler = StandardScaler()
    validation_images = None
    validation_images_scaled = None

    #these were the default values
    width = 45
    height = 45
    mean = 0.03
    var = 0.03
    PCA_n_components = 50
    RF_n_estimators = 10
    min_samples_leaf = 1
    
    totalstart = time.time()
    data = []
    for step in range(0,50):
        print("########%d#######" % step)
        x = random.randint(10,48)
        # i bet the optimal values can also be found by machine learning, but for now we just use random numbers and decide manually
        PCA_n_components = random.randint(20,200)
        RF_n_estimators = random.randint(5,20)
        width = x
        height = x
        mean = random.randint(10,600) / 1000.0
        var = random.randint(10,600) / 1000.0
        min_samples_leaf = random.randint(1,10)

        print("Params PCA_n_components: %d RF_n_estimators: %d width: %d height: %d mean: %f var: %f min_samples_leaf: %d" %
             (
                 PCA_n_components,
                 RF_n_estimators,
                 width,
                 height,
                 mean,
                 var,
                 min_samples_leaf
             )
             )

        start = time.time()
        (fruit_images,fruit_images_noisy,labels,labels_noisy) = load_images(width,height,mean,var)
        images_scaled = scaler.fit_transform([i.flatten() for i in fruit_images])
        label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
        id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
        label_ids = np.array([label_to_id_dict[x] for x in labels])

        print("running PCA")
        pca = PCA(n_components=PCA_n_components)
        pca_result = pca.fit_transform(images_scaled)

        print("loading Validationdata")
        (validation_images,validation_label_ids,validation_image_paths) = load_validation_images(label_to_id_dict,width,height)
        validation_images_scaled = scaler.transform(validation_images)
        validation_pca_result = pca.transform(validation_images_scaled)

        X_train, X_test, y_train, y_test = train_test_split(pca_result, label_ids,stratify = label_ids, test_size=0.25, random_state=42)
        forest = RandomForestClassifier(n_estimators=RF_n_estimators,min_samples_leaf=min_samples_leaf,n_jobs=4,random_state=42)
        forest = forest.fit(X_train, y_train)
        test_predictions = forest.predict(X_test)
        accuracy1 = accuracy_score(test_predictions, y_test) * 100
        print("accuracy against training data: %f" % accuracy1)
        test_predictions = forest.predict(validation_pca_result)
        accuracy2 = accuracy_score(test_predictions, validation_label_ids) * 100
        print("accuracy against validation data: %f" % accuracy2)
        elapsed_time = (time.time()-start)
        print("elapsed_time %d s" % elapsed_time)
        data.append((
            PCA_n_components,
            RF_n_estimators,
            width,
            height,
            mean,
            var,
            min_samples_leaf,
            accuracy2,
            elapsed_time
        ))
        if (time.time()-totalstart) > 40*60:
            break
    pd.DataFrame(data, columns=[
        "PCA_n_components",
        "RF_n_estimators",
        "width",
        "height",
        "mean",
        "var",
        "min_samples_leaf",
        "Accuracy",
        "Time"])