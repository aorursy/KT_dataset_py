import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
%matplotlib inline
project_name = 'Dog Breed Classification'
print(os.listdir('../input/stanford-dogs-dataset'))
annotation_list = []

for dpath,dnames,files in os.walk('../input/stanford-dogs-dataset/annotations/Annotation'):
    if files:
        for file in files:
            annotation_list.append(file)
        
len(annotation_list)
annotation_list = pd.DataFrame({'Images':annotation_list})
annotation_list.head()
labels = {
    0 : 'Chihuahua',
    1 : 'Japanese_spaniel',
    2 : 'Maltese_dog',
    3 : 'Pekinese',
    4 : 'Shih-Tzu',
    5 : 'Blenheim_spaniel',
    6 : 'papillon',
    7 : 'toy_terrier',
    8 : 'Rhodesian_ridgeback',
    9 : 'Afghan_hound',
    10 : 'basset',
    11 : 'beagle',
    12 : 'bloodhound',
    13 : 'bluetick',
    14 : 'black-and-tan_coonhound',
    15 : 'Walker_hound',
    16 : 'English_foxhound',
    17 : 'redbone',
    18 : 'borzoi',
    19 : 'Irish_wolfhound',
    20 : 'Italian_greyhound',
    21 : 'whippet',
    22 : 'Ibizan_hound',
    23 : 'Norwegian_elkhound',
    24 : 'otterhound',
    25 : 'Saluki',
    26 : 'Scottish_deerhound',
    27 : 'Weimaraner',
    28 : 'Staffordshire_bullterrier',
    29 : 'American_Staffordshire_terrier',
    30 : 'Bedlington_terrier',
    31 : 'Border_terrier',
    32 : 'Kerry_blue_terrier',
    33 : 'Irish_terrier',
    34 : 'Norfolk_terrier',
    35 : 'Norwich_terrier',
    36 : 'Yorkshire_terrier',
    37 : 'wire-haired_fox_terrier',
    38 : 'Lakeland_terrier',
    39 : 'Sealyham_terrier',
    40 : 'Airedale',
    41 : 'cairn',
    42 : 'Australian_terrier',
    43 : 'Dandie_Dinmont',
    44 : 'Boston_bull',
    45 : 'miniature_schnauzer',
    46 : 'giant_schnauzer',
    47 : 'standard_schnauzer',
    48 : 'Scotch_terrier',
    49 : 'Tibetan_terrier',
    50 : 'silky_terrier',
    51 : 'soft-coated_wheaten_terrier',
    52 : 'West_Highland_white_terrier',
    53 : 'Lhasa',
    54 : 'flat-coated_retriever',
    55 : 'curly-coated_retriever',
    56 : 'golden_retriever',
    57 : 'Labrador_retriever',
    58 : 'Chesapeake_Bay_retriever',
    59 : 'German_short-haired_pointer',
    60 : 'vizsla',
    61 : 'English_setter',
    62 : 'Irish_setter',
    63 : 'Gordon_setter',
    64 : 'Brittany_spaniel',
    65 : 'clumber',
    66 : 'English_springer',
    67 : 'Welsh_springer_spaniel',
    68 : 'cocker_spaniel',
    69 : 'Sussex_spaniel',
    70 : 'Irish_water_spaniel',
    71 : 'kuvasz',
    72 : 'schipperke',
    73 : 'groenendael',
    74 : 'malinois',
    75 : 'briard',
    76 : 'kelpie',
    77 : 'komondor',
    78 : 'Old_English_sheepdog',
    79 : 'Shetland_sheepdog',
    80 : 'collie',
    81 : 'Border_collie',
    82 : 'Bouvier_des_Flandres',
    83 : 'Rottweiler',
    84 : 'German_shepherd',
    85 : 'Doberman',
    86 : 'miniature_pinscher',
    87 : 'Greater_Swiss_Mountain_dog',
    88 : 'Bernese_mountain_dog',
    89 : 'Appenzeller',
    90 : 'EntleBucher',
    91 : 'boxer',
    92 : 'bull_mastiff',
    93 : 'Tibetan_mastiff',
    94 : 'French_bulldog',
    95 : 'Great_Dane',
    96 : 'Saint_Bernard',
    97 : 'Eskimo_dog',
    98 : 'malamute',
    99 : 'Siberian_husky',
    100 : 'affenpinscher',
    101 : 'basenji',
    102 : 'pug',
    103 : 'Leonberg',
    104 : 'Newfoundland',
    105 : 'Great_Pyrenees',
    106 : 'Samoyed',
    107 : 'Pomeranian',
    108 : 'chow',
    109 : 'keeshond',
    110 : 'Brabancon_griffon',
    111 : 'Pembroke',
    112 : 'Cardigan',
    113 : 'toy_poodle',
    114 : 'miniature_poodle',
    115 : 'standard_poodle',
    116 : 'Mexican_hairless',
    117 : 'dingo',
    118 : 'dhole',
    119 : 'African_hunting_dog'
}
annotation_list.Images[0]
annotation_list['Labels'] = 0
for img in annotation_list.Images:
    if img[:9] == 'n02085620':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 0
    elif img[:9] == 'n02085782':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 1
    elif img[:9] == 'n02085936':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 2
    elif img[:9] == 'n02086079':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 3
    elif img[:9] == 'n02086240':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 4
    elif img[:9] == 'n02086646':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 5
    elif img[:9] == 'n02086910':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 6
    elif img[:9] == 'n02087046':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 7
    elif img[:9] == 'n02087394':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 8
    elif img[:9] == 'n02088094':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 9
    elif img[:9] == 'n02088238':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 10
    elif img[:9] == 'n02088364':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 11
    elif img[:9] == 'n02088466':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 12
    elif img[:9] == 'n02088632':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 13
    elif img[:9] == 'n02089078':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 14
    elif img[:9] == 'n02089867':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 15
    elif img[:9] == 'n02089973':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 16
    elif img[:9] == 'n02090379':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 17
    elif img[:9] == 'n02090622':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 18
    elif img[:9] == 'n02090721':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 19
    elif img[:9] == 'n02091032':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 20
    elif img[:9] == 'n02091134':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 21
    elif img[:9] == 'n02091244':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 22
    elif img[:9] == 'n02091467':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 23
    elif img[:9] == 'n02091635':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 24
    elif img[:9] == 'n02091831':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 25
    elif img[:9] == 'n02092002':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 26
    elif img[:9] == 'n02092339':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 27
    elif img[:9] == 'n02093256':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 28
    elif img[:9] == 'n02093428':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 29
    elif img[:9] == 'n02093647':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 30
    elif img[:9] == 'n02093754':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 31
    elif img[:9] == 'n02093859':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 32
    elif img[:9] == 'n02093991':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 33
    elif img[:9] == 'n02094114':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 34
    elif img[:9] == 'n02094258':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 35
    elif img[:9] == 'n02094433':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 36
    elif img[:9] == 'n02095314':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 37
    elif img[:9] == 'n02095570':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 38
    elif img[:9] == 'n02095889':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 39
    elif img[:9] == 'n02096051':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 40
    elif img[:9] == 'n02096177':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 41
    elif img[:9] == 'n02096294':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 42
    elif img[:9] == 'n02096437':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 43
    elif img[:9] == 'n02096585':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 44
    elif img[:9] == 'n02097047':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 45
    elif img[:9] == 'n02097130':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 46
    elif img[:9] == 'n02097209':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 47
    elif img[:9] == 'n02097298':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 48
    elif img[:9] == 'n02097474':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 49
    elif img[:9] == 'n02097658':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 50
    elif img[:9] == 'n02098105':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 51
    elif img[:9] == 'n02098286':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 52
    elif img[:9] == 'n02098413':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 53
    elif img[:9] == 'n02099267':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 54
    elif img[:9] == 'n02099429':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 55
    elif img[:9] == 'n02099601':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 56
    elif img[:9] == 'n02099712':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 57
    elif img[:9] == 'n02099849':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 58
    elif img[:9] == 'n02100236':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 59
    elif img[:9] == 'n02100583':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 60
    elif img[:9] == 'n02100735':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 61
    elif img[:9] == 'n02100877':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 62
    elif img[:9] == 'n02101006':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 63
    elif img[:9] == 'n02101388':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 64
    elif img[:9] == 'n02101556':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 65
    elif img[:9] == 'n02102040':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 66
    elif img[:9] == 'n02102177':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 67
    elif img[:9] == 'n02102318':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 68
    elif img[:9] == 'n02102480':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 69
    elif img[:9] == 'n02102973':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 70
    elif img[:9] == 'n02104029':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 71
    elif img[:9] == 'n02104365':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 72
    elif img[:9] == 'n02105056':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 73
    elif img[:9] == 'n02105162':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 74
    elif img[:9] == 'n02105251':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 75
    elif img[:9] == 'n02105412':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 76
    elif img[:9] == 'n02105505':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 77
    elif img[:9] == 'n02105641':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 78
    elif img[:9] == 'n02105855':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 79
    elif img[:9] == 'n02106030':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 80
    elif img[:9] == 'n02106166':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 81
    elif img[:9] == 'n02106382':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 82
    elif img[:9] == 'n02106550':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 83
    elif img[:9] == 'n02106662':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 84
    elif img[:9] == 'n02107142':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 85
    elif img[:9] == 'n02107312':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 86
    elif img[:9] == 'n02107574':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 87
    elif img[:9] == 'n02107683':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 88
    elif img[:9] == 'n02107908':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 89
    elif img[:9] == 'n02108000':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 90
    elif img[:9] == 'n02108089':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 91
    elif img[:9] == 'n02108422':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 92
    elif img[:9] == 'n02108551':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 93
    elif img[:9] == 'n02108915':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 94
    elif img[:9] == 'n02109047':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 95
    elif img[:9] == 'n02109525':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 96
    elif img[:9] == 'n02109961':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 97
    elif img[:9] == 'n02110063':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 98
    elif img[:9] == 'n02110185':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 99
    elif img[:9] == 'n02110627':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 100
    elif img[:9] == 'n02110806':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 101
    elif img[:9] == 'n02110958':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 102
    elif img[:9] == 'n02111129':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 103
    elif img[:9] == 'n02111277':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 104
    elif img[:9] == 'n02111500':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 105
    elif img[:9] == 'n02111889':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 106
    elif img[:9] == 'n02112018':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 107
    elif img[:9] == 'n02112137':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 108
    elif img[:9] == 'n02112350':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 109
    elif img[:9] == 'n02112706':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 110
    elif img[:9] == 'n02113023':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 111
    elif img[:9] == 'n02113186':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 112
    elif img[:9] == 'n02113624':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 113
    elif img[:9] == 'n02113712':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 114
    elif img[:9] == 'n02113799':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 115
    elif img[:9] == 'n02113978':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 116
    elif img[:9] == 'n02115641':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 117
    elif img[:9] == 'n02115913':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 118
    elif img[:9] == 'n02116738':
        idx = annotation_list[annotation_list['Images'] == img].index.values
        idx = int(idx)
        annotation_list.at[idx,'Labels'] = 119
annotation_list
img_dir = '../input/stanford-dogs-dataset/images/Images'
labels[0]
class StanfordDogs(Dataset):
    def __init__(self,data,root_dir,transform=None):
        self.df = data
        self.transform = transform
        self.root_dir = root_dir
    def __len__(self):
        return len(self.df) 
    def __getitem__(self,idx):
        row = self.df.loc[idx]
        img_id, img_label = row['Images'], row['Labels']
        img_fname = self.root_dir + '/' + str(img_id[:9]) + '-' + labels[img_label] + "/" + str(img_id) + ".jpg"
        img = Image.open(img_fname).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(img_label)
stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
dataset = StanfordDogs(annotation_list, img_dir, transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats)]))
len(dataset)
def show_sample(img, target):
    plt.imshow(img.permute(1, 2, 0))
    print('Labels:', target, 'Shape:', img.shape)
show_sample(*dataset[100])
torch.manual_seed(11)
val_pct = 0.1
test_pct = 0.1
val_size = int(val_pct * len(dataset))
test_size = int(test_pct * len(dataset))
train_size = len(dataset) - val_size - test_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size,val_size,test_size])
len(train_ds), len(val_ds), len(test_ds)
batch_size = 128
train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 2, pin_memory = True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers = 2, pin_memory = True)
def show_batch(dl):
    for images,labels in dl:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xticks([]); ax.set_yticks([])
        data = images
        ax.imshow(make_grid(data * stats[1][0] + stats[0][0], nrow=16).permute(1, 2, 0))
        break
show_batch(train_dl)
def F_score(output, label, threshold=0.5, beta=1):
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self,batch):
        images, labels = batch
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    
        loss = F.cross_entropy(out, labels)   
        acc = accuracy(out, labels)           
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
class DogsCnn(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Flatten(), 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 120),
            nn.ReLU(),
        )
        
    def forward(self, xb):
        return self.network(xb)
class DogsResnet(DogsCnn):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 120)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)
device = get_default_device()
device
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            sched.step()
        
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history
model = to_device(DogsResnet(), device)
history = [evaluate(model, val_dl)]
history
epochs = 2
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
%%time
history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, weight_decay=weight_decay, grad_clip=grad_clip, opt_func=opt_func)

