from collections import OrderedDict

import numpy as np

from tqdm import tqdm



def parse_input(file):

	with open(file) as f:

		lines = f.readlines()[1:]



	pictures = OrderedDict()

	for i in range(len(lines)):

		line = lines[i]

		orientation, _, *tags = line.split()

		pictures[i] = {

			"orientation": orientation,

			"tags": list(tags),

            "amount": len(list(tags))

		}

	return pictures
def get_average_amount_of_tags():

    photos = parse_input('../input/hashcode-photo-slideshow/d_pet_pictures.txt')

    amount = []

    for i in photos:

        amount.append(photos[i]['amount'])

    return sum(amount) / len(amount)

    

amount_of_tags = get_average_amount_of_tags()

print('Average amount of tags: ', amount_of_tags)
def list_of_slides():

    photos = parse_input('../input/hashcode-photo-slideshow/d_pet_pictures.txt')

    ids = photos.keys()

    vert = [i for i in ids if photos[i]['orientation'] == 'V']

    all_ids = []

    for i in range(int(len(vert)/2)):

        A = set(photos[vert[i*2]]['tags'])

        B = set(photos[vert[i*2]]['tags'])

        all_ids.append((vert[i*2], vert[i*2+1], A.union(B)))

    for i in ids:

        if photos[i]['orientation'] == 'H':

            all_ids.append((i, set(photos[i]['tags'])))

    return all_ids



all_slides = list_of_slides()



def create_tag_dict(array):

    photos = parse_input('../input/hashcode-photo-slideshow/d_pet_pictures.txt')

    tags = {}

    for i in array:

        for tag in i[-1]:

            if tag not in tags:

                tags[tag]=0

                tags[tag] += 1

            else:

                tags[tag] += 1

    return tags



tag_map = create_tag_dict(all_slides)



amounts = []

for i in tag_map:

    amounts.append(tag_map[i])

    

average = sum(amounts) / len(amounts)

print('# of tags: ', len(amounts))

print('Average amount of ids per tag: ', average)
print('Estimated optimal amount of candidates: ', amount_of_tags * (average - 1))