import itertools

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from enum import Enum

from tqdm import tqdm

from typing import Union, List, Callable

from functools import lru_cache

from collections import defaultdict

from dataclasses import dataclass



np.random.seed(12)
# define some models and functions



class Orientation(Enum):

    Horizontal = 0

    Vertical = 1

    Combined = 2



    def __repr__(self):

        return self.name



    def __str__(self):

        return self.__repr__()





@dataclass

class Photo:

    id: Union[int, tuple]

    tags: set

    orientation: Orientation



    @classmethod

    def from_string(cls, id: int, line: str) -> "Photo":

        orient, _, *tags = line.strip().split()



        if orient == "V":

            orient = Orientation.Vertical

        elif orient == "H":

            orient = Orientation.Horizontal

        else:

            raise ValueError("Unknown orientation: '{}'.".format(orient))



        return Photo(id=id, tags=set(tags), orientation=orient)



    def __len__(self):

        return len(self.tags)



    def __and__(self, other):

        return len(self.tags & other.tags)



    def __sub__(self, other):

        return len(self.tags - other.tags)



    def __or__(self, other):

        assert self.orientation == Orientation.Vertical and other.orientation == Orientation.Vertical



        return Photo(

            id=(self.id, other.id),

            tags=self.tags | other.tags,

            orientation=Orientation.Combined,

        )



    def __hash__(self):

        return hash(self.id)

    

    def max_score(self) -> int:

        return len(self) // 2





def calc_score(p1: Photo, p2: Photo) -> int:

    return min(p1 & p2, p1 - p2, p2 - p1)





@lru_cache(maxsize=2**20)

def lazy_calc_score(p1: Photo, p2: Photo) -> int:

    return calc_score(p1, p2)





def calc_max_score(p1: Photo, p2: Photo) -> int:

    return min(len(p1), len(p2)) // 2





def calc_lost_score(p1: Photo, p2: Photo) -> int:

    return p1.max_score() + p2.max_score() - 2 * calc_score(p1, p2)





def _apply(sequence: List[Photo], function: Callable[[Photo, Photo], int]) -> int:

    if len(sequence) <= 1:

        return 0

    return sum(function(sequence[i], sequence[i - 1]) for i in range(1, len(sequence)))





def sequence_score(sequence: List[Photo]) -> int:

    return _apply(sequence, calc_score)



    

def sequence_max_score(sequence: List[Photo]) -> int:

    return _apply(sequence, calc_max_score)



 

def sequence_lost_score(sequence: List[Photo]) -> int:

    return _apply(sequence, calc_lost_score)





def read_file(path: str) -> List[Photo]:

    data = []

    with open(path, "r") as file:

        num_photo = int(file.readline())

        for i, line in enumerate(file):

            data.append(Photo.from_string(i, line))

    return data





def check_sequence(sequence: List[Photo]):

    all_id = set()

    for photo in sequence:

        photo_id = photo.id

        assert isinstance(photo_id, (int, tuple)), f"Wrong id format: {photo_id}"



        if isinstance(photo_id, tuple):

            assert len(photo.id) == 2, f"Wrong id format: {photo_id}"

            assert photo_id[0] != photo_id[1], f"Wrong id format: {photo_id}"

        else:

            photo_id = (photo_id,)



        for x in photo_id:

            assert x not in all_id, f"id {x} not unique"

            all_id.add(x)



            

def create_submission(submission: List[Photo], path="submission.txt"):

    check_sequence(submission)

    with open(path, "w+") as f:

        f.write("{}\n".format(len(submission)))

        for photo in submission:

            photo_id = photo.id

            if not isinstance(photo_id, tuple):

                photo_id = (photo_id,)

            f.write("{}\n".format(" ".join(map(str, photo_id))))

            



def show(submission: List[Photo]):

    total_score, total_max_score = [], []

    p1 = submission[0]

    for p2 in submission[1:]:

        total_score.append(calc_score(p1, p2))

        total_max_score.append(calc_max_score(p1, p2))

        p1 = p2

        

    print("Total score: {}/{}".format(sum(total_score), sum(total_max_score)))



    # slide size distribution

    horizontal_hist, vertical_hist = defaultdict(int), defaultdict(int)

    for photo in submission:

        is_vertical = isinstance(photo.id, tuple)

        hist = vertical_hist if is_vertical else horizontal_hist

        hist[len(photo)] += 1

    fig = plt.figure(figsize=(14,4))

    plt.bar(horizontal_hist.keys(), horizontal_hist.values(), label="horizontal", alpha=0.5)

    plt.bar(vertical_hist.keys(), vertical_hist.values(), label="vertical", alpha=0.5)

    plt.xlabel("number of tags"); plt.ylabel("number of slides"); plt.legend(); plt.show()

    

    # score

    fig = plt.figure(figsize=(14,4))

    plt.plot(total_score, label="score", alpha=0.5)

    plt.plot(total_max_score, label="max score", alpha=0.5)

    plt.xlabel("slide"); plt.ylabel("score"); plt.legend(); plt.show()



    # number of slides

    nb_horizontal, nb_vertical = [0], [0]

    for photo in submission:

        is_vertical = isinstance(photo.id, tuple)

        nb_horizontal.append(nb_horizontal[-1] + (not is_vertical))

        nb_vertical.append(nb_vertical[-1] + is_vertical)

    fig = plt.figure(figsize=(14,4))

    plt.plot(nb_horizontal, label="horizontal", alpha=0.5)

    plt.plot(nb_vertical, label="vertical", alpha=0.5)

    plt.xlabel("slide"); plt.ylabel("number of slides"); plt.legend(); plt.show()



    # loss

    horizontal_loss, vertical_loss = [0], [0]

    for score, max_score, photo in zip(total_score, total_max_score, submission):

        is_vertical = isinstance(photo.id, tuple)

        loss = max_score - score

        horizontal_loss.append(horizontal_loss[-1] + loss * (not is_vertical))

        vertical_loss.append(vertical_loss[-1] + loss * is_vertical)

    fig = plt.figure(figsize=(14,4))

    plt.plot(horizontal_loss, label="horizontal", alpha=0.5)

    plt.plot(vertical_loss, label="vertical", alpha=0.5)

    plt.plot([sum(x) for x in zip(horizontal_loss, vertical_loss)], label="total", alpha=0.5)

    plt.xlabel("slide"); plt.ylabel("loss"); plt.legend(); plt.show()
data = read_file(path="../input/hashcode-photo-slideshow/d_pet_pictures.txt")

print(f"Score = {sequence_score(data)} / {sequence_max_score(data)}")
def stitch(sequences, th=1):

    """ trying to connect two different sequences """

    if len(sequences) <= 1:

        return sequences

    

    if th == 0:

        return [sum(sequences, [])]

    

    for i, j in itertools.combinations(range(len(sequences)), r=2):

        s1, s2 = sequences[i], sequences[j]



        if not s1 or not s2:

            continue



        if lazy_calc_score(s1[-1], s2[0]) >= th:

            sequences[i], sequences[j] = [], s1 + s2

            continue



        if lazy_calc_score(s1[-1], s2[-1]) >= th:

            sequences[i], sequences[j] = [], s1 + s2[::-1]

            continue



        if lazy_calc_score(s1[0], s2[0]) >= th:

            sequences[i], sequences[j] = [], s1[::-1] + s2

            continue



        if lazy_calc_score(s1[0], s2[-1]) >= th:

            sequences[i], sequences[j] = [], s1[::-1] + s2[::-1]

            continue

                

    return [s for s in sequences if s]

def _do_insert(s1, s2, th):

    """ trying to insert sequence 1 into sequence 2 """

    if not s1 or len(s2) <= 1:

        return False, s2

    

    for i, p2 in enumerate(s2[1:], start=1):

        p1 = s2[i - 1]

        

        if lazy_calc_score(p1, s1[0]) + lazy_calc_score(s1[-1], p2) >= 2 * th:

            return True, s2[:i] + s1 + s2[i:]



        if lazy_calc_score(p1, s1[-1]) + lazy_calc_score(s1[0], p2) >= 2 * th:

            return True, s2[:i] + s1[::-1] + s2[i:]

        

    return False, s2





def insert(sequences, th):

    if len(sequences) <= 1:

        return sequences

    

    for i, j in itertools.product(range(len(sequences)), repeat=2):

        if i != j:

            status, combined_sequence = _do_insert(sequences[i], sequences[j], th=th)

            if status:

                sequences[i], sequences[j] = [], combined_sequence

    

    return [s for s in sequences if s]





def _do_shuffle(s1, s2, th=1, p=1):

    """ trying to swap some subsequence from sequence 1 and sequence 2 """

    if not s1 or len(s2) <= 1:

        return s1, s2

    

    for i, p2 in enumerate(s2[1:], start=1):

        p1 = s2[i - 1]



        if lazy_calc_score(p1, s1[0]) >= th:

            if np.random.random_sample() < p:

                return s2[:i] + s1, s2[i:]



        if lazy_calc_score(p1, s1[-1]) >= th:

            if np.random.random_sample() < p:

                return s2[:i] + s1[::-1], s2[i:]

    

    return s1, s2







def shuffle(sequences, th=1, p=1):

    if len(sequences) <= 1 or p == 0:

        return sequences

    

    for i, j in itertools.product(range(len(sequences)), repeat=2):

        if i != j:

            sequences[i], sequences[j] = _do_shuffle(sequences[i], sequences[j], th=th, p=p)



    return [s for s in sequences if s]





        

def create_sub_sequences(sequence, slide_score=1):

    """ create list of perfect subsequence """

    out = []

    if not sequence:

        return out

    

    sub_sequence = [sequence[0]]

    sequence = sequence[1:]

    while sequence:

        p1 = sub_sequence[-1]

        

        _next = None

        for i, p2 in enumerate(sequence):

            if p2 & p1 == slide_score:

                _next = i

                break

        

        if _next is not None:

            p2 = sequence[i]

            sub_sequence.append(p2)

            sequence = sequence[:i] + sequence[i + 1:]

        else:

            out.append(sub_sequence)

            sub_sequence = [sequence[0]]

            sequence = sequence[1:]

    out.append(sub_sequence)

    

    assert all(sequence_lost_score(s) == 0 for s in out)

    

    return out





def create_photo_sequences(photos):

    horisontal_sequences = []

    for size in sorted({len(x) // 2 * 2 for x in photos}):

        sizes = (size, size + 1)

        print(">>> Processing {}...".format(sizes))

        slide_score = size // 2

        sequence = [x for x in photos if len(x) in sizes]

        if not sequence:

            continue



        sequences = create_sub_sequences(sequence, slide_score=slide_score)



        nb_attempts = 0

        previous_total_score = 0

        while True:

            # subsequence post processing

            # trying to reduce number of subsequences, all subsequences must remain perfect

            sequences = stitch(sequences, th=slide_score)

            sequences = insert(sequences, th=slide_score)

            sequences = shuffle(sequences, th=slide_score, p=0.2)



            total_score = sum(sequence_score(x) for x in sequences)

            if total_score <= previous_total_score:

                nb_attempts += 1

            else:

                nb_attempts = 0

            previous_total_score = total_score



            if len(sequences) == 1 or nb_attempts >= 10:

                break



        assert all(sequence_lost_score(s) == 0 for s in sequences)



        sequence = sum(sequences, [])

        print("Nb sub sequences", len(sequences), ", Nb photos", len(sequence))

        print(f"Score = {sequence_score(sequence)} / {sequence_max_score(sequence)}")



        horisontal_sequences += sequences

        size += 2



    return sum(horisontal_sequences, [])
# Match vertical photos

# Please see https://www.kaggle.com/huikang/441k-in-11-mins for more details

MERGE_WINDOW = 10000

REARRANGE_FOR_MERGE = True



def match_vertical_photos(vertical_photos):

    vertical_photos = sorted(vertical_photos, key=len)



    vertical_tmp = vertical_photos[::-1]  # start from photo with most tags



    if REARRANGE_FOR_MERGE:  

        # so we can easily match photos with more tags with photos with less tags

        vertical_photos[0::2] = vertical_tmp[:len(vertical_photos) // 2]

        vertical_photos[1::2] = vertical_tmp[len(vertical_photos) // 2:][::-1]



    vertical_photos, vertical_tmp = [vertical_photos[0]], vertical_photos[1:]



    for i in tqdm(range(len(vertical_tmp))):

        p1 = vertical_photos[-1]

        best = -9999

        best_next_ptr = 0

        cnt = 0

        for j, p2 in enumerate(vertical_tmp):

            if len(vertical_photos)%2 == 0:  # we do not need to consider between pairs

                break

            if best == 0:

                # we have found an optimal match

                break

            if cnt > MERGE_WINDOW:

                # early stopping in the search for a paired photo

                break

            sc = - (p1 & p2)

            num_tags_if_paired = len(p1 | p2)

            if num_tags_if_paired % 2 == 1:  

                # penalise if the total number of tags is odd

                sc = min(sc, -0.9)

            if num_tags_if_paired > 22 and REARRANGE_FOR_MERGE:  

                # to encourage the total number of tags around 22

                sc = min(sc, -0.02 * num_tags_if_paired)

            if sc > best:

                best = sc

                best_next_ptr = j

            cnt += 1

        vertical_photos.append(vertical_tmp[best_next_ptr])

        vertical_tmp = vertical_tmp[:best_next_ptr] + vertical_tmp[best_next_ptr+1:]



    combined_photo = [a | b for a,b in zip(vertical_photos[0::2], vertical_photos[1::2])]

    

    return combined_photo
vertical_photos = [x for x in data if x.orientation == Orientation.Vertical]

combined_photos = match_vertical_photos(vertical_photos)

all_photos = combined_photos + [x for x in data if x.orientation == Orientation.Horizontal]

submission = create_photo_sequences(all_photos)
show(submission)
def _reverse(sequence, start, end):

    return sequence[:start] + sequence[start:end][::-1] + sequence[end:]



    

def _improve(sequence, i, greedy=False):

    l1, l2 = sequence[i - 1], sequence[i]

    l12, max_l12 = lazy_calc_score(l1, l2), calc_max_score(l1, l2)

    for j in range(i + 1, len(sequence)):

        r1, r2 = sequence[j - 1], sequence[j]

        max_r12 = calc_max_score(r1, r2)

        current_max_score = max_l12 + max_r12

        

        max_lr1 = calc_max_score(l1, r1)

        max_lr2 = calc_max_score(l2, r2)

        new_max_score = max_lr1 + max_lr2

        

        if not greedy and new_max_score < current_max_score:

            continue

        

        r12 = lazy_calc_score(r1, r2)

        current_score = l12 + r12

        

        lr1 = calc_score(l1, r1)

        lr2 = calc_score(l2, r2)

        new_score = lr1 + lr2

        

        if new_score > current_score:

            sequence = _reverse(sequence, i, j)

            break

    

    return sequence

    

    

def post_process(submission, greedy=False):  

    p1 = submission[0]

    for i in range(1, len(submission)):

        p2 = submission[i]

        if lazy_calc_score(p1, p2) < calc_max_score(p1, p2):

            submission = _improve(submission, i, greedy=greedy)

        p1 = p2

    return submission
nb_attempts = 0

previous_score = 0

greedy = False

while True:

    print(f"Score = {sequence_score(submission)} / {sequence_max_score(submission)}")

    current_score = sequence_score(submission)

    

    if current_score <= previous_score:

        nb_attempts += 1

    else:

        nb_attempts = 0

    if nb_attempts >= 2:

        if not greedy:

            greedy = True

        else:

            break

    previous_score = current_score

    

    submission = post_process(submission[::-1], greedy=greedy)
show(submission)
create_submission(submission)