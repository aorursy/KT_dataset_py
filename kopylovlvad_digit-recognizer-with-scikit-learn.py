from typing import List, Dict, Tuple
import csv
import os
from sklearn.tree import *
from sklearn.ensemble import *
print('Available files:')
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def collect_data(
        filereader: List[List[str]], 
        overview: bool = False,
        main_ration: float = 0.8
    ) -> dict:
    train_data: List[List[int]] = []
    test_data: List[List[int]] = []
    train_data_answers: List[str] = []
    test_data_answers: List[str] = []
    all_data_dict: Dict[str, List[List[int]]] = {}
    rows_len: int = len(filereader) - 1
    print('file size: %d' % rows_len)
    print('collect data:')
    i = 0
    for row in filereader:
        i += 1
        if i % (rows_len/10) == 0:
            print(str((i / rows_len) * 100)+'%')
        if i == 1:
            continue
        tmp_row = []
        for j in range(len(row)):
            # without label
            if j == 0:
                continue
            value: int = int(row[j])
            tmp_row.append(value)

        if not row[0] in all_data_dict.keys():
            all_data_dict[row[0]] = []
        all_data_dict[row[0]].append(tmp_row)

    del filereader
    l: int
    all_data_size: int = 0
    if overview == True:
        for k in all_data_dict.keys():
            l = len(all_data_dict[k])
            print('%d items with "%s" label' % (l, k))
            all_data_size += l
        print('Data size: %d' % all_data_size)

    # 80-30 %% by default
    for k in all_data_dict.keys():
        l = len(all_data_dict[k])
        for i in range(l):
            d_row: List[int] = all_data_dict[k][i]
            if i < l * main_ration:
                train_data.append(d_row)
                train_data_answers.append(k)
            else:
                test_data.append(d_row)
                test_data_answers.append(k)

    del all_data_size
    return {
        'train_data': train_data,
        'train_data_answers': train_data_answers,
        'test_data': test_data,
        'test_data_answers': test_data_answers
    }
csvfile = open('../input/train.csv', newline='')
filereader = list(csv.reader(csvfile, delimiter=',', quotechar='"'))
collect_dict = collect_data(filereader, overview=True)
train_data = collect_dict['train_data']
train_data_answers = collect_dict['train_data_answers']
test_data = collect_dict['test_data']
test_data_answers = collect_dict['test_data_answers']

print('We have %s train samples' % str(len(train_data)))
print('We have %s test samples' % str(len(test_data)))

print('end')
# the function for experiment
def perform(classifier,
            train_data,
            train_data_answers,
            test_data,
            test_data_answers) -> None:
    string = ''
    string += classifier.__class__.__name__

    # train
    classifier.fit(train_data, train_data_answers)

    # score
    score: float = classifier.score(test_data, test_data_answers)
    score = round(score * 100, 1)
    string += ' has score: ' + str(score) + '%'
    print(string)
    return None
print('Results of RandomForestClassifier():')
perform(
    RandomForestClassifier(),
    train_data,
    train_data_answers,
    test_data,
    test_data_answers
)
print('Results of BaggingClassifier():')
perform(
    BaggingClassifier(),
    train_data,
    train_data_answers,
    test_data,
    test_data_answers
)
print('Results of ExtraTreesClassifier():')
perform(
    ExtraTreesClassifier(),
    train_data,
    train_data_answers,
    test_data,
    test_data_answers
)
import matplotlib.pyplot as plt

def row_to_pixmap(
    row: List[Tuple[str, int]],
)-> List[List[int]]:
    px_arr: List[List[int]] = []
    tmp_px_arr: List[int] = []
    for i in range(len(row)):
        value: int = int(row[i])
        tmp_px_arr.append(value)
        if (i + 1) % 28 == 0:
            px_arr.append(tmp_px_arr)
            tmp_px_arr = []
    return px_arr

def pixmap_to_array(
    pix_map: List[List[int]]
) -> List[int]:
    pix_arr: List[int] = []
    for pix_row in pix_map:
        for cell in pix_row:
            pix_arr.append(cell)
    return pix_arr

def show_image_from_pmap(pix_map: List[List[int]]) -> None:
    fig = plt.figure(figsize=(2, 2))
    plt.axis('off')
    plt.imshow(pix_map)
    plt.show()
    return None
k: int = 22  # Sample by index
print("Label Prediction: %s" % filereader[k][0])

show_image_from_pmap(row_to_pixmap(filereader[k][1:]))
def increase_contrast(
        pix_map: List[List[int]],
        extra_value: int = 10,
        min_value: int = 100,
        max_value: int = 250) -> List[List[int]]:
    new_pix_map: List[List[int]] = []

    for pix_y in range(len(pix_map)):
        tmp_arr: List[int] = []
        pix_row = pix_map[pix_y]
        for pix_x in range(len(pix_row)):
            cell = pix_row[pix_x]

            # logic
            if cell > min_value:
                cell += extra_value
            if cell > max_value:
                cell = max_value

            tmp_arr.append(cell)
        new_pix_map.append(tmp_arr)

    return new_pix_map


def bold_image_logic(
    pix_x: int,
    pix_y: int,
    pix_map: List[List[int]],
    add_value: int,
    max_coef: int
) -> int:
    around_pixels: List[int] = []
    bottom_limit = len(pix_map) - 1
    right_limit = len(pix_map[0]) - 1

    # top
    if pix_y-1 > 0:
        pix = pix_map[pix_y-1][pix_x]
        around_pixels.append(pix)

        # top-right
        if pix_x+1 < right_limit:
            pix = pix_map[pix_y-1][pix_x+1]
            around_pixels.append(pix)

        # top-left
        if pix_x-1 > 0:
            pix = pix_map[pix_y-1][pix_x-1]
            around_pixels.append(pix)

    # bottom
    if pix_y+1 < bottom_limit:
        pix = pix_map[pix_y+1][pix_x]
        around_pixels.append(pix)

        # bottom-right
        if pix_x+1 < right_limit:
            pix = pix_map[pix_y+1][pix_x+1]
            around_pixels.append(pix)

        # bottom-left
        if pix_x-1 > 0:
            pix = pix_map[pix_y+1][pix_x-1]
            around_pixels.append(pix)

    # right
    if pix_x+1 < right_limit:
        pix = pix_map[pix_y][pix_x+1]
        around_pixels.append(pix)

    # left
    if pix_x-1 > 0:
        pix = pix_map[pix_y][pix_x-1]
        around_pixels.append(pix)

    cell = pix_map[pix_y][pix_x]
    strong_pixel_size: int = list(
        filter(lambda x: x > max_coef, around_pixels))
    if len(strong_pixel_size) > 0:
        cell += add_value
    return cell


def bold_image(
    pix_map: List[List[int]],
    add_value: int = 25,
    max_coef: int = 200
) -> List[List[int]]:
    new_pix_map: List[List[int]] = []

    for pix_y in range(len(pix_map)):
        tmp_arr: List[int] = []
        pix_row = pix_map[pix_y]
        for pix_x in range(len(pix_row)):
            tmp_arr.append(
                bold_image_logic(pix_x, pix_y, pix_map, add_value, max_coef)
            )

        new_pix_map.append(tmp_arr)
    return new_pix_map

k: int = 22  # Sample by index
print("Label Prediction: %s" % filereader[k][0])

sample_pix_map = row_to_pixmap(filereader[k][1:])
sample_pix_map = increase_contrast(sample_pix_map, extra_value=75)
sample_pix_map = bold_image(sample_pix_map, add_value=150, max_coef=150)

show_image_from_pmap(sample_pix_map)
def make_font_bolder(
    array: List[List[int]]
) -> List[List[int]]:
    result: List[List[int]] = []
    for sample in array:
        pix_map: List[List[int]] = row_to_pixmap(sample)

        # action
        pix_map = increase_contrast(pix_map, extra_value=75)
        pix_map = bold_image(pix_map, add_value=150, max_coef=150)
        result.append(pixmap_to_array(pix_map))
    return result


print('process train_data')
bold_train_data: List[List[int]] = make_font_bolder(train_data)
print('process test_data')
bold_test_data: List[List[int]] = make_font_bolder(test_data)


perform(
    ExtraTreesClassifier(),
    bold_train_data,
    train_data_answers,
    bold_test_data,
    test_data_answers
)