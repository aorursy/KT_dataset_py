from typing import List, Dict, Tuple
import clusters
import csv_helper as csv_h
import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-l", '--limit', help='Csv row limit', type=int)
arguments = parser.parse_args()
csv_row_limit: int = arguments.limit or 100  # 8120

print('csv_limit is: %d' % csv_row_limit)

dirname: str = os.path.dirname(os.path.abspath(__file__))
item_names, props, data = csv_h.csv_to_vector(dirname + '/mushrooms.csv')

data = data[:csv_row_limit]
item_names = item_names[:csv_row_limit]
print('We have %d train items' % len(data))

f = open(dirname + '/tmp_data/mushrooms_data_vector1.pickle', 'wb')
pickle.dump((item_names, props, data), f)
f.close()

from typing import List, Dict
import clusters
import os
import csv
import argparse
import csv_helper as csv_h
import pickle
import sys

#
# prepare data
#

dirname: str = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("-l", '--limit', help='Csv row limit', type=int)
parser.add_argument("-o", '--offset', help='Offset limit', type=int)
arguments = parser.parse_args()
csv_row_limit: int = arguments.limit or 100  # 8120
offset: int = arguments.offset or 100  # 8120
print('csv_limit is: %d' % csv_row_limit)
print('offset is: %d' % offset)


#
# prepare csv
#

print('Opening csv ... ', end='')
file_path: str = dirname + '/mushrooms.csv'
test_item_names: List[str]
_props: List[str]
test_data: List[List[int]]
test_item_names, _props, test_data = csv_h.csv_to_vector(file_path)
del _props

test_item_names = test_item_names[offset:][:csv_row_limit]
test_data = test_data[offset:][:csv_row_limit]
print('end')

print('We have %d test_items' % len(test_data))

#
# pickling
#

print('Pickling ... ', end='')
f = open(dirname + '/tmp_data/mushrooms_data_vector1.pickle', 'rb')
know_item_names, know_props, know_data = pickle.load(f)
f.close()

print('Train data items: %d' % len(know_data))


def p_e_verict(three_item: List[str]) -> str:
    p_size: int = 0
    e_size: int = 0
    for name in three_item:
        word: str = name[len(name)-1]
        if word == 'p':
            p_size += 1
        elif word == 'e':
            e_size += 1
        else:
            raise BaseException('word is not into [p,e]')

    if p_size > e_size:
        return 'p'
    else:
        return 'e'


#
# processing
#
cassify_data: List[str] = []
for i in range(len(test_data)):
    test_name: str = test_item_names[i]
    test_row = test_data[i]
    three_closest_name: List[str] = []  # list with names
    three_closest_name = clusters.get_three_closest_names(
        test_row,
        know_item_names,
        know_data,
        distance=clusters.tanimoto_coeff
    )
    cassify_data.append(p_e_verict(three_closest_name))


#
# checking
#
stat: Dict[str, int] = {
    'equal': 0,
    'not_equal': 0
}
print('Checking ... ', end='')
for i in range(len(test_data)):
    test_full_name: str = test_item_names[i]
    test_name = test_full_name[len(test_full_name) - 1]
    predict_name: str = cassify_data[i]

    if predict_name == test_name:
        stat['equal'] = stat['equal'] + 1
    else:
        stat['not_equal'] = stat['not_equal'] + 1
print('')

print('Equal is: %d' % stat['equal'])
print('Not equal is: %d' % stat['not_equal'])
divisor: float = (stat['equal'] + stat['not_equal']) / 100

if divisor == 0:
    print('Accuracy is 0')
else:
    print('Accuracy is %f' % (stat['equal'] / divisor))
