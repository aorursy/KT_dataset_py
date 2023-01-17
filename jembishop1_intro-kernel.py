!pip install pygame pypianoroll

import pickle

import random

with open("../input/music/music.pk", "rb") as f:

    music = pickle.load(f)

music = random.sample(music, 200)
music[0]
from collections import Counter

import matplotlib.pyplot as plt

c = Counter([i["composer"] for i in music]).items()

fig, ax = plt.subplots(figsize=(40, 20))

ax.bar(*zip(*c))
from pypianoroll import Multitrack, Track

import pygame

def write_midi(arr, filename):

    Multitrack(tracks=[Track(arr*127)]).write(filename)

def play(filename):

    pygame.init()

    pygame.mixer.music.load(filename)

    pygame.mixer.music.play()

    
import itertools

from scipy.sparse import vstack

import tqdm

num_composers = 40

chunk_size = 1024

groups = itertools.groupby(sorted(music, key=lambda x: x["composer"]), lambda x: x["composer"])

segments = []

for composer, pieces in tqdm.tqdm_notebook(groups, total=num_composers):

    pieces_list = list(i["piece"].tocsr() for i in pieces)

    n = sum([i.shape[0] for i in pieces_list])//chunk_size

    if n!=0:

        trimmed_concat  = vstack(pieces_list)[:chunk_size*n]

        composer_segs = [(trimmed_concat[i:i+chunk_size], composer) for i in range(0,n*chunk_size,chunk_size)]

        segments.extend(composer_segs)

random.shuffle(segments)
c = Counter(seg[1] for seg in segments).items()

fig, ax = plt.subplots(figsize=(40, 20))

ax.bar(*zip(*c))
def test(num):

    answers = []

    for seg, comp in segments[:num]:

        write_midi(seg.toarray(), "temp.mid")

        play("temp.mid")

        inp = input("Who was it?")

        if inp=="quit":

            break

        if len(inp)>=3 and inp.lower() in comp.lower():

            print(f"Correct the composer was {comp}")

            answers.append((comp, True))

        else:

            print(f"Incorrect the composer was {comp}")

            answers.append((comp, False))

    return answers

        

    
#test(10) will work locally