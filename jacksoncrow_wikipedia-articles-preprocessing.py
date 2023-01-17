import pandas as pd

import numpy as np

import string

import json

import shutil



from os import listdir, mkdir

from os.path import isfile, isdir, join, exists

from keras.preprocessing import image

from keras.applications.resnet import ResNet152, preprocess_input
def _globalMaxPool1D(tensor):

    _,_,_,size = tensor.shape

    return [tensor[:,:,:,i].max() for i in range(size)]



def _getImageFeatures(model, img_path):

    img = image.load_img(img_path, target_size=None)



    img_data = image.img_to_array(img)

    img_data = np.expand_dims(img_data, axis=0)

    img_data = preprocess_input(img_data)



    feature_tensor = model.predict(img_data)

    get_img_id = lambda p: p.split('/')[-1].split('.')[0]

    return {

        "id": get_img_id(img_path),

        "features": _globalMaxPool1D(feature_tensor),

    }



def _getTextFeatures(text_path):

    with open(text_path) as json_file:

        data = json.loads(json.load(json_file))

        text = data['text'].replace("\n", " ")

        return {

            'id': data['id'],

            'text': text.translate(str.maketrans('', '', string.punctuation)),

        }

    

def _getValidImagePaths(article_path):

    img_path = join(article_path, 'img/')

    return [join(img_path, f) for f in listdir(img_path) if isfile(join(img_path, f)) and f[-4:].lower() == ".jpg"]



def GetArticleData(model, article_path):

    article_data = _getTextFeatures(join(article_path, 'text.json'))

    article_data["img"] = []

    for img_path in _getValidImagePaths(article_path):

        img_features = _getImageFeatures(model, img_path)

        article_data["img"].append(img_features)

        

    return article_data



def PreprocessArticles(data_path, limit=None):

    article_paths = [join(data_path, f) for f in listdir(data_path) if isdir(join(data_path, f))]

    limit = limit if limit else len(article_paths) + 1

    model = ResNet152(weights='imagenet', include_top=False) 

    

    articles = []

    for path in article_paths:

        article_data = GetArticleData(model, path)

        articles.append(article_data)

        if len(articles) >= limit: break

            

    return articles
%%time

# Remember to enable Internet access for kernel to avoid errors during downloading of pretrained model

data_path = '/kaggle/input/wiki-articles-multimodal/data/'

articles = PreprocessArticles(data_path, limit=3)
print('ID of the first article = ', articles[0]['id'])
print('First 1000 characters from article\'s text:\n')

print(articles[0]['text'][:1000])
print('In img field we have a list of all images related to the article')

images = articles[0]['img']

print(type(images))

print('Image Count = ', len(images))



print("\nEach image represented as a list of its visual features")

first_image = images[0]['features']

print(type(first_image))

print('Size of visual features vector = ', len(first_image))

print('First 10 value of feature vector = ', first_image[:10])