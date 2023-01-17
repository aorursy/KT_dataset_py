!ls
!ls ..
!ls ../..
def list_images(path):

    #  Use sorted for more predicative behaviour

    return sorted([f for f in os.listdir(path) if not f.startswith(".") and f.endswith(".jpg")])
import pandas as pd

def load_labels():

    label_df = pd.read_csv("../input/labels.csv")

    dogs = list_images(TRAIN_PATH)

    dogs.sort()

    for dog in dogs:

        try:

            dog_id = dog.split(".")[0]

            label = label_df[label_df.id == dog_id].iloc[0, 1]

            labels[dog] = label

        except IndexError:

            print("%s not found" % dog_id)

            labels[dog] = None

            pass

all_labels = """affenpinscher

                afghan_hound

                african_hunting_dog

                airedale

                american_staffordshire_terrier

                appenzeller

                australian_terrier

                basenji

                basset

                beagle

                bedlington_terrier

                bernese_mountain_dog

                black-and-tan_coonhound

                blenheim_spaniel

                bloodhound

                bluetick

                border_collie

                border_terrier

                borzoi

                boston_bull

                bouvier_des_flandres

                boxer

                brabancon_griffon

                briard

                brittany_spaniel

                bull_mastiff

                cairn

                cardigan

                chesapeake_bay_retriever

                chihuahua

                chow

                clumber

                cocker_spaniel

                collie

                curly-coated_retriever

                dandie_dinmont

                dhole

                dingo

                doberman

                english_foxhound

                english_setter

                english_springer

                entlebucher

                eskimo_dog

                flat-coated_retriever

                french_bulldog

                german_shepherd

                german_short-haired_pointer

                giant_schnauzer

                golden_retriever

                gordon_setter

                great_dane

                great_pyrenees

                greater_swiss_mountain_dog

                groenendael

                ibizan_hound

                irish_setter

                irish_terrier

                irish_water_spaniel

                irish_wolfhound

                italian_greyhound

                japanese_spaniel

                keeshond

                kelpie

                kerry_blue_terrier

                komondor

                kuvasz

                labrador_retriever

                lakeland_terrier

                leonberg

                lhasa

                malamute

                malinois

                maltese_dog

                mexican_hairless

                miniature_pinscher

                miniature_poodle

                miniature_schnauzer

                newfoundland

                norfolk_terrier

                norwegian_elkhound

                norwich_terrier

                old_english_sheepdog

                otterhound

                papillon

                pekinese

                pembroke

                pomeranian

                pug

                redbone

                rhodesian_ridgeback

                rottweiler

                saint_bernard

                saluki

                samoyed

                schipperke

                scotch_terrier

                scottish_deerhound

                sealyham_terrier

                shetland_sheepdog

                shih-tzu

                siberian_husky

                silky_terrier

                soft-coated_wheaten_terrier

                staffordshire_bullterrier

                standard_poodle

                standard_schnauzer

                sussex_spaniel

                tibetan_mastiff

                tibetan_terrier

                toy_poodle

                toy_terrier

                vizsla

                walker_hound

                weimaraner

                welsh_springer_spaniel

                west_highland_white_terrier

                whippet

                wire-haired_fox_terrier

                yorkshire_terrier""".split("\n")

all_labels = [i.strip() for i in all_labels]
import os

OUTPUT_PATH = os.path.join(".")

TRAIN_PATH = os.path.join("..", "input", "train")

TEST_PATH = os.path.join("..", "input", "test")
labels = dict()

load_labels()
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.applications.vgg16 import VGG16

import keras.preprocessing.image

import numpy as np
width = 224

height = 224

def create_model(input_shape=(3, width, height)):

    vgg_model = VGG16(weights='imagenet')

    flatten = vgg_model.get_layer("flatten")



    model = Sequential()

    model.add(flatten)

    model.add(Dense(10000))

    model.add(Dense(120, activation="softmax", input_shape=input_shape))



    model.compile(loss='categorical_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])



    return model
def train():

    model = create_model()

    dogs = list_images(TRAIN_PATH)[:10]

    images = []

    ys = []

    for dog in dogs:

        image = keras.preprocessing.image.load_img(os.path.join(TRAIN_PATH, dog), target_size=(width, height))

        image = keras.preprocessing.image.img_to_array(image)

        images.append(image)

        label = labels[dog]



        if label is None:

            print("Label not found for train dog %s" % dog)



        y = np.zeros(120)

        y[all_labels.index(label)] = 1

        ys.append(y)

    images = np.array(images)

    ys = np.array(ys)

    model.fit(images, ys, verbose=2, epochs=1)

    model.save(os.path.join(OUTPUT_PATH, "vgg.h5"))
def predict():

    dogs = list_images(TEST_PATH)[:10]  # TODO

    images = []

    for dog in dogs:

        image = keras.preprocessing.image.load_img(os.path.join(TEST_PATH, dog), target_size=(width, height))

        image = keras.preprocessing.image.img_to_array(image)

        images.append(image)

    images = np.array(images)



    model = keras.models.load_model(os.path.join(OUTPUT_PATH, "vgg.h5"))

    ys = model.predict(images)



    with open(os.path.join(OUTPUT_PATH, "predict_vgg.csv"), "w") as out:

        out.write("id,")

        for label in all_labels[:-1]:

            out.write(label + ",")

        out.write(all_labels[-1])

        out.write("\n")



        for i in range(len(ys)):

            out.write("%s," % dogs[i].split(".")[0])

            for prob in ys[i][:-1]:

                out.write("%f," % prob)

            out.write("%f" % ys[i][-1])

            out.write("\n")
train()