guessed = 0 # this is how many pets you saw

correct = 0 # this is how many times you (human) predicted correctly
# DATA PREPARATION (JUST RUN FIRST TIME)

import pandas as pd

import numpy as np

import json

train = pd.read_csv("../input/train/train.csv")

breed=pd.read_csv('../input/breed_labels.csv')

breedmap = breed.set_index("BreedID").drop("Type", axis=1).to_dict()['BreedName']

train['Breed1'] = train['Breed1'].map(breedmap)

train['Breed2'] = train['Breed2'].map(breedmap)

color=pd.read_csv('../input/color_labels.csv')

colormap = color.to_dict()['ColorName']

for c in ['Color1', 'Color2', 'Color3']:

    train[c] = train[c].map(colormap)

state=pd.read_csv('../input/state_labels.csv')

statemap = state.set_index("StateID").to_dict()['StateName']

c='State'

train[c] = train[c].map(statemap)

train['Type'] = train['Type'].map({2:'Cat',1:'Dog'})

mmap = {1:'Yes',2:'No',3:'not sure'}

for c in ['Vaccinated', 'Dewormed', 'Sterilized']:

    train[c] = train[c].map(mmap)

train['Health'] = train['Health'].map({0:"NA",1:"Healthy",2:"Minor Injury",3:"Serious Injury"})
pets=train.iterrows()
# RUN THIS TO LOAD NEXT PET EVERY TIME

import os, re

def purge(dir, pattern):

    for f in os.listdir(dir):

        if re.search(pattern, f):

            os.remove(os.path.join(dir, f))

purge('.', 'image*') # delete previous images

from PIL import Image, ImageDraw

pet = next(pets)[1]

ID,index = pet['PetID'],str(int(pet['PhotoAmt']))

images = []

for i in range(int(pet['PhotoAmt'])):

    annotations = json.load(open("../input/train_metadata/"+ID+"-"+index+".json"))

    vects = annotations['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices']

    for vect in vects:

        for k in ['x', 'y']:

            if vect.get(k) is None:

                vect[k] = 0

    im = Image.open("../input/train_images/"+ID+"-"+str(i + 1)+".jpg")

    draw = ImageDraw.Draw(im)

    draw.polygon([

    vects[0]['x'], vects[0]['y'],

    vects[1]['x'], vects[1]['y'],

    vects[2]['x'], vects[2]['y'],

    vects[3]['x'], vects[3]['y']], None, 'red')

    im.save('image'+str(i)+'.jpg', 'JPEG')    

    images.append(im)
# RUN THIS TO SHOW IMAGE

from matplotlib import pyplot as plt

plt.figure(figsize=(30,10))

for i, im in enumerate(images):

    if i > 8: break

    plt.subplot('1'+str(min(9, len(images)))+str(i + 1))

    plt.imshow(np.asarray(im))
pd.DataFrame(pet).transpose().drop('AdoptionSpeed', axis=1)
pet['Description']
guess = 1 # GUESS ADOPTION SPEED HERE

real = int(pet['AdoptionSpeed'])

print(real)

guessed += 1

if guess == real: correct += 1
print("HUMAN ACCURACY SCORE: "+str(correct/guessed))