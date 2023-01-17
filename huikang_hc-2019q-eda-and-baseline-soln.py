%reset -sf

import random

import collections

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator



random.seed(42)



!ls /kaggle/input/hashcode-photo-slideshow/
filepath = "/kaggle/input/hashcode-photo-slideshow/d_pet_pictures.txt"

with open(filepath) as f:

    pictures = [row.strip().split() for row in f.readlines()][1:]
!head -3 "/kaggle/input/hashcode-photo-slideshow/d_pet_pictures.txt"
pic_tags = {}  # maps idx to tags

horizontal_photos = []  # horizontal photos only

vertical_photos = []  # vertical photos only

for i,picture in enumerate(pictures):

    pic_tags[i] = set(picture[2:])

    if picture[0] == "H":

        horizontal_photos.append(i)

    elif picture[0] == "V":

        vertical_photos.append(i)

print(len(vertical_photos), len(horizontal_photos))
def calc_tags_pair_score(tags1, tags2):

    # given two sets of tags, calculate the score

    return min(len(tags1 & tags2), len(tags1 - tags2), len(tags2 - tags1))



def calc_idxs_pair_score(idxs1, idxs2):

    # given two tuples of indices, calculate the score

    return calc_tags_pair_score(

        set.union(*[pic_tags[idx] for idx in idxs1]),

        set.union(*[pic_tags[idx] for idx in idxs2]))



def calc_idxs_pair_score_max(idxs1, idxs2):

    # given two tuples of indices, calculate the maximum possible score by tag length

    return min(len(set.union(*[pic_tags[idx] for idx in idxs1])),

               len(set.union(*[pic_tags[idx] for idx in idxs2])))//2



def calc_sequence(idxs_lst):

    # given the sequence of indices, calculate the score

    check_validity(idxs_lst)

    score = 0

    for before, after in zip(idxs_lst[:-1], idxs_lst[1:]):

        score += calc_idxs_pair_score(before, after)            

    return score



def calc_sequence_max(idxs_lst):

    # given the sequence of indices, calculate the score

    check_validity(idxs_lst)

    score = 0

    for before, after in zip(idxs_lst[:-1], idxs_lst[1:]):

        score += calc_idxs_pair_score_max(before, after)            

    return score



def check_validity(idxs_lst):

    all_pics = [idx for idxs in idxs_lst for idx in idxs]

    if len(all_pics) != len(set(all_pics)):

        print("Duplicates found")

    all_verts = [idx for idxs in idxs_lst for idx in idxs if len(idxs) == 2]

    if (set(all_verts) - set(vertical_photos)):

        print("Horizontal photos found in vertical combinations")

    all_horis = [idx for idxs in idxs_lst for idx in idxs if len(idxs) == 1]

    if (set(all_horis) - set(horizontal_photos)):

        print("Vertical photos found in horizontal arrangement")
idxs_list = [(a,b) for a,b in zip(vertical_photos[0::2], vertical_photos[1::2])] 

idxs_list.extend([(a,) for a in horizontal_photos])

calc_sequence(idxs_list), calc_sequence_max(idxs_list), len(idxs_list)
random.shuffle(vertical_photos)

vertical_photos.sort(key=lambda idx: len(pic_tags[idx]))

idxs_list_combined = [(a,b) for a,b in zip(vertical_photos[0::2], vertical_photos[1::2])]

idxs_list = idxs_list_combined + [(a,) for a in horizontal_photos]

calc_sequence(idxs_list), calc_sequence_max(idxs_list), len(idxs_list)
random.shuffle(idxs_list)

idxs_list.sort(key = lambda idxs: sum([len(pic_tags[idx])//2 for idx in idxs]))

calc_sequence(idxs_list), calc_sequence_max(idxs_list), len(idxs_list)
tags_set = sorted(set(tag for idx,tags in pic_tags.items() for tag in tags))

tags_counter_all = collections.OrderedDict((tag,0) for tag in tags_set)

tags_counter_horizontal = collections.OrderedDict((tag,0) for tag in tags_set)

tags_counter_vertical = collections.OrderedDict((tag,0) for tag in tags_set)

for idx in horizontal_photos:

    for tag in pic_tags[idx]:

        tags_counter_horizontal[tag] += 1

        tags_counter_all[tag] += 1

for idx in vertical_photos:

    for tag in pic_tags[idx]:

        tags_counter_vertical[tag] += 1

        tags_counter_all[tag] += 1
plt.figure(figsize=(14,4))

plt.scatter([v for k,v in tags_counter_vertical.items()],

            [v for k,v in tags_counter_horizontal.items()], label="one tag")

plt.xlabel("freqency of tags in vertical photos")

plt.ylabel("freqency of tags in horizontal photos")

plt.title("total number of tags: " + str(len(tags_counter_all)))

plt.legend()

plt.show()
plt.figure(figsize=(14,4))

bins = np.arange(0,6000,50)

plt.hist([[count for tag,count in tags_counter_horizontal.items()],

          [count/2 for tag,count in tags_counter_vertical.items()]],

         bins=bins, stacked=True,

         label=["number of tags on horizontal photos in the frequency bucket",

                "number of tags on vertical photos in the frequency bucket"])

plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend()

plt.show()
tags_freq_mean = np.mean(list(tags_counter_all.values()))

more_frequent_tags = set([tag for tag,count in tags_counter_all.items() 

                          if count > tags_freq_mean])

less_frequent_tags = set([tag for tag,count in tags_counter_all.items() 

                          if count <= tags_freq_mean])

int(tags_freq_mean), len(more_frequent_tags), len(less_frequent_tags)
plt.figure(figsize=(14,4))

plt.scatter([len([tag for tag in pic_tags[idx] if tag in more_frequent_tags]) 

             + np.random.uniform() for idx in vertical_photos],

            [len([tag for tag in pic_tags[idx] if tag in less_frequent_tags]) 

             + np.random.uniform() for idx in vertical_photos], 

            s = 1, alpha=0.2, label = "one vertical photo")

plt.xlabel("number of more common tags")

plt.ylabel("number of less common tags")

plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend()

plt.show()
plt.figure(figsize=(14,4))

plt.hist([len(pic_tags[idx]) for idx in horizontal_photos], bins=range(20), alpha=0.5,

         label="distribution of number of tags of horizontal photos")

plt.hist([len(pic_tags[idx]) for idx in vertical_photos], bins=range(20), alpha=0.2,

         label="distribution of number of tags of vertical photos")

for rect in plt.gca().patches:

    height = rect.get_height()

    plt.gca().annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 

                       xytext=(0, 0), textcoords='offset points', 

                       ha='center', va='bottom', fontsize=8)

plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend()

plt.show()
plt.figure(figsize=(14,4))

plt.plot([calc_idxs_pair_score(idxs1,idxs2) for idxs1,idxs2 in 

          zip(idxs_list[:-1], idxs_list[1:])], alpha=0.5, 

         label="score of neighbours in the sequence")

plt.plot([calc_idxs_pair_score_max(idxs1,idxs2) for idxs1,idxs2 in 

          zip(idxs_list[:-1], idxs_list[1:])], alpha=0.5, 

         label="maximum possible score of neighbours in the sequence")

plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend()

plt.show()
plt.figure(figsize=(14,4))

plt.hist([len(pic_tags[idxs[0]] | pic_tags[idxs[1]])

          for idxs in zip(vertical_photos[::2], vertical_photos[1::2])], 

         bins=range(36), alpha=0.5,

         label="distribution of number of tags of combined photos")

for rect in plt.gca().patches:

    height = rect.get_height()

    plt.gca().annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 

                       xytext=(0, 0), textcoords='offset points', 

                       ha='center', va='bottom', fontsize=7)

plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend()

plt.show()
plt.figure(figsize=(14,4))

plt.hist([len(pic_tags[idx1] & pic_tags[idx2]) for idx1,idx2 in 

          zip(vertical_photos[::2], vertical_photos[1::2])], bins=range(20), alpha=0.5,

         label="distribution of number of overlapping tags " + 

               "between neighbours of vertical photos sorted by number of tags")

for rect in plt.gca().patches:

    height = rect.get_height()

    plt.gca().annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 

                       xytext=(0, 0), textcoords='offset points', 

                       ha='center', va='bottom', fontsize=8)

plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend()

plt.show()
plt.figure(figsize=(14,4))

plt.hist([[calc_idxs_pair_score(idxs1,idxs2) for idxs1,idxs2 in 

          zip(idxs_list[::2], idxs_list[1::2]) if len(idxs1) + len(idxs2) == 4],

          [calc_idxs_pair_score(idxs1,idxs2) for idxs1,idxs2 in 

          zip(idxs_list[::2], idxs_list[1::2]) if len(idxs1) + len(idxs2) == 3],

          [calc_idxs_pair_score(idxs1,idxs2) for idxs1,idxs2 in 

          zip(idxs_list[::2], idxs_list[1::2]) if len(idxs1) + len(idxs2) == 2]],

         bins=range(20), alpha=0.5,

         label=[

        "distribution of scores for vertical-vertical neighbours",

        "distribution of scores for horizontal-vertical neighbours",

        "distribution of scores for horizontal-horizontal neighbours"], 

         stacked=True)

plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend()

plt.show()
plt.figure(figsize=(14,4))

plt.hist([[calc_idxs_pair_score_max(idxs1,idxs2) for idxs1,idxs2 in 

          zip(idxs_list[::2], idxs_list[1::2]) if len(idxs1) + len(idxs2) == 4],

          [calc_idxs_pair_score_max(idxs1,idxs2) for idxs1,idxs2 in 

          zip(idxs_list[::2], idxs_list[1::2]) if len(idxs1) + len(idxs2) == 3],

          [calc_idxs_pair_score_max(idxs1,idxs2) for idxs1,idxs2 in 

          zip(idxs_list[::2], idxs_list[1::2]) if len(idxs1) + len(idxs2) == 2]],

         bins=range(20), alpha=0.5,

         label=[

        "maximum possible scores of the transition between vertical-vertical pair",

        "maximum possible scores of the transition between vertical-horizontal pair",

        "maximum possible scores of the transition between horizontal-horizontal pair"], 

         stacked=True)

plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend()

plt.show()
(sum(len(pic_tags[idx]) for idx in vertical_photos)//2 + \

 sum(len(pic_tags[idx])//2 for idx in horizontal_photos))
submission_lines = []

submission_lines.append(str(len(idxs_list)))

for idxs in idxs_list:

    submission_lines.append(" ".join([str(idx) for idx in idxs]))
with open("submission.txt", "w") as f:

    f.writelines("\n".join(submission_lines))
!head -3 submission.txt
calc_sequence(idxs_list), calc_sequence_max(idxs_list), len(idxs_list)