from typing import List, Dict, Tuple, Union, Set
import csv
import re
import random
import csv
import decision_tree_kv
import os

print('Aviables files')
print(os.listdir("../input"))
all_data: List[List[str]] = list(csv.reader(
    open('../input/mushrooms.csv', newline='')))

columns_titles_list: List[str] = []
uniq_columns_values: Dict[str, Dict[str, int]] = {}
i: int = 0
for row in all_data:
    i += 1
    if i == 1:
        # head row
        for cell in row:
            columns_titles_list.append(cell)
            uniq_columns_values[cell] = {}
        continue
    
    for j in range(len(row)):
        column_title = columns_titles_list[j]
        cell = row[j]
        if cell in uniq_columns_values[column_title].keys():
            uniq_columns_values[column_title][cell] += 1
        else:
            uniq_columns_values[column_title].setdefault(cell, 1)
    
print('Data overview:')
print('Rows count: %d' % len(all_data)) #with head row
print('Columns count: %d' % len(all_data[0]))
print('Columns: ' + str(columns_titles_list))
print('')
print('Columns uniq values:')

uniq_values_str: str
for column_name in uniq_columns_values.keys():
    uniq_values_str = ''
    for k_value in uniq_columns_values[column_name].keys():
        all_count: int = sum([v for v in uniq_columns_values[column_name].values()])
        v: float = uniq_columns_values[column_name][k_value] / (all_count/100)
        uniq_values_str += k_value+"("+str(round(v, 1))+"%); "
    print("Column '%s' has unique values: \n %s" % (column_name, uniq_values_str))
accuracy_list: List[float] = []
i: int
train_data: List[List[str]]
test_data: List[List[str]]
data_p: List[List[str]]
data_e: List[List[str]]
stat: Dict[str, int]
    
def transform_row(row_data: List[str]) -> List[str]:
    # replace 'class' column to end of the row
    mushroom_class = row_data[0]
    tmp_row: List[str] = []
    for j in range(len(row_data)):
        if not j == 0:
            # without mushroom class
            tmp_row.append(row_data[j])
    tmp_row.append(mushroom_class)
    if not len(row_data) == len(tmp_row):
        raise BaseException('lens are not equal')
    if len(tmp_row) == 0:
        raise BaseException('tmp_row len equals 0')
    return tmp_row
    
for ex_number in range(4):
    print('Running experiment #%d ...' % (ex_number + 1))
    all_data: List[List[str]] = list(csv.reader(open('../input/mushrooms.csv', newline='')))
    random.shuffle(all_data)
    train_data = []
    test_data = []
    data_p = []
    data_e = []
    i = 0
    
    for row in all_data:
        i += 1
        if row[0] == 'class':
            # ignore head row
            continue
        
        transformed_row: List[str] = []
        transformed_row = transform_row(row)
        m_class = transformed_row[len(transformed_row) - 1]
        if m_class == 'e':
            data_e.append(transformed_row)
        elif m_class == 'p':
            data_p.append(transformed_row)
        else:
            raise BaseException('undefined mushroom class')
    
    i = 0
    for row in data_e:
        i += 1
        if i < len(data_e) * 0.5:
            train_data.append(row)
        else:
            test_data.append(row)
    i = 0
    for row in data_p:
        i += 1
        if i < len(data_p) * 0.5:
            train_data.append(row)
        else:
            test_data.append(row)

    print('Edible samples count: %d' % len(data_e))
    print('Poisonous samples count: %d' % len(data_p))
    print('Train subset size: %d samples' % len(train_data))
    print('Test subset size: %d samples' % len(test_data))
    print('All data size: %d samples' % len(all_data))

    # train decision tree
    tree = decision_tree_kv.buildtree(train_data, scoref=decision_tree_kv.giniimpurity)
    
    # check accuracy
    stat = {'equal': 0, 'not_equal': 0, 'not_equal_e': 0, 'not_equal_p': 0}
    for test_row in test_data:
        right_class = test_row[len(test_row) - 1]
        row_without_class = test_row[0:len(test_row)-1]

        r: dict = decision_tree_kv.classify(row_without_class, tree)
        if len(r.keys()) > 1:
            raise BaseException('so many classes from tree')

        if list(r.keys())[0] == right_class:
            stat['equal'] += 1
        else:
            stat['not_equal'] += 1
            if right_class == 'e':
                stat['not_equal_e'] += 1
            elif right_class == 'p':
                stat['not_equal_p'] += 1
            else:
                raise BaseException('undefined right class')
        
    divider: float = (stat['equal'] + stat['not_equal'])/100
    accuracy: float = 0.0
    if not divider == 0:
        accuracy = stat['equal']/divider
    accuracy_list.append(accuracy)
    
    print('')
    print('Result of experiment:')
    print('Equal test samples count: %d' % stat['equal'])
    print('Not equal test samples count: %d' % stat['not_equal'])
    print('Amount of mistakes with "e" class: %d' % stat['not_equal_e'])
    print('Amount of mistakes with "p" class: %d' % stat['not_equal_p'])
    print('Accuracy '+str(round(accuracy, 2))+'%')
    print('')
    print('')
print('Final result:')
average_accuracy: float = sum(accuracy_list) / float(len(accuracy_list))
print('Accuracy '+str(round(average_accuracy, 1))+'%')


column_list = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 
             'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 
             'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 
             'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 
             'spore-print-color', 'population', 'habitat', 'class']
def get_uniq_col_indexes(tree_string: str) -> List[int]:
    # function for print columns indexes
    reg_res: List = re.compile(r'column\[(\d+)\]').findall(tree_string)
    reg_res = list(set(reg_res))
    reg_res = [int(i) for i in reg_res]
    reg_res.sort()
    return reg_res

all_data: List[List[str]] = list(csv.reader(open('../input/mushrooms.csv', newline='')))
train_data, test_data, data_p, data_e = [], [], [], []
i = 0
random.shuffle(all_data)
for row in all_data:
    i += 1
    if row[0] == 'class':
        # ignore head row
        continue

    transformed_row: List[str] = []
    transformed_row = transform_row(row)
    m_class = transformed_row[len(transformed_row) - 1]
    if m_class == 'e':
        data_e.append(transformed_row)
    elif m_class == 'p':
        data_p.append(transformed_row)
    else:
        raise BaseException('undefined mushroom class')

i = 0
for row in data_e:
    i += 1
    if i < len(data_e) * 0.5:
        train_data.append(row)
    else:
        test_data.append(row)
i = 0
for row in data_p:
    i += 1
    if i < len(data_p) * 0.5:
        train_data.append(row)
    else:
        test_data.append(row)

# train decision tree
tree = decision_tree_kv.buildtree(train_data, scoref=decision_tree_kv.giniimpurity)

# check accuracy
stat = {'equal': 0, 'not_equal': 0}
for test_row in test_data:
    right_class = test_row[len(test_row) - 1]
    row_without_class = test_row[0:len(test_row)-1]
    r: dict = decision_tree_kv.classify(row_without_class, tree)
    if list(r.keys())[0] == right_class:
        stat['equal'] += 1
    else:
        stat['not_equal'] += 1

divider: float = (stat['equal'] + stat['not_equal'])/100
accuracy: float = 0.0
if not divider == 0:
    accuracy = stat['equal']/divider
print('Accuracy '+str(round(accuracy, 2))+'%')

print('')
print('')

# print flowchart
tree_str: str = decision_tree_kv.tree_to_str(tree)
print('Flowchart:')
print(tree_str)
print('')
print('List of used columns:')
used_columns = get_uniq_col_indexes(tree_str)
for label in used_columns:
    print('Columns "%s"(%d)' % (column_list[label], label))