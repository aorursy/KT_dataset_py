import os
# print(os.listdir("../input"))
from typing import List, Dict, Tuple, Union, Set
import csv

uniq_columns_data: Dict[str, Set[str]] = {}
columns_list: List[str] = []
filereader = csv.reader(open('../input/data.csv', newline=''))
lines: int = 0
b_rows: int = 0
m_rows: int = 0
for row in filereader:
    if row[0] == 'id':
        # head row
        for cell in row:
            if len(cell) > 0:
                uniq_columns_data[cell] = set()
                columns_list.append(cell)
    else:
        # row with data
        for i in range(len(row)):
            cell = row[i]
            column_name = columns_list[i]
            uniq_columns_data[column_name].add(cell)
        
        if row[1] == 'M':
            m_rows += 1
        elif row[1] == 'B':
            b_rows += 1
        else:
            raise BaseException('undefined diagnosis')
    lines += 1

print('File has %d columns:' % len(columns_list))
print('Columns list:')
for i in range(len(columns_list)):
    print('column[%d]: %s' % (i, columns_list[i]))

print('')
for k in uniq_columns_data.keys():
    print('Column "%s" has %d uniq values' % (k, len(uniq_columns_data[k])))
print('')
print('Rows amount: %d' % lines)
print('Rows with "malignant" diagnosis amount: %d' % m_rows)
print('Rows with "benign" diagnosis amount: %d' % b_rows)
        
        
import os
# print(os.listdir("../input"))
from typing import List, Dict, Tuple, Union, Set
import csv
# source https://github.com/kopylovvlad/decision_tree_kv
import decision_tree_kv
import random

all_data:  List[List[str]]
train_data: List[List[Union[float, str]]] = []
test_data: List[List[Union[float, str]]] = []
data_m: List[List[Union[float, str]]] = []
data_b: List[List[Union[float, str]]] = []    
accuracy_list: List[float] = []

    
for experiment_number in range(4):
    print('Running experiment %d' % (experiment_number + 1))
    all_data = list(csv.reader(open('../input/data.csv', newline='')))
    train_data = []
    test_data = []
    data_m = []
    data_b = []    

    # shuffle data
    random.shuffle(all_data)

    # collect data to data_m and data_b sebsets
    for row in all_data:
        i += 1
        if row[0] == 'id':
            # ignore head-row
            continue

        tmp_row: List[Union[str, float]] = []
        diagnosis: str = row[1]

        for j in range(len(row)):
            # without id and diagnosis
            if j == 0 or j == 1:
                continue
            tmp_row.append(float(row[j]))

        # set diagnosis as last column
        tmp_row.append(str(diagnosis))

        if diagnosis == 'M':
            data_m.append(tmp_row)
        elif diagnosis == 'B':
            data_b.append(tmp_row)
        else:
            raise BaseException('Undefined diagnosis')


    i = 0
    # filling train data set
    for row in data_m:
        i += 1
        if i < len(data_m) * 0.8:
            train_data.append(row)
        else:
            test_data.append(row)

    i = 0
    # filling test data set
    for row in data_b:
        i += 1
        if i < len(data_b) * 0.8:
            train_data.append(row)
        else:
            test_data.append(row)
    
    print('Data overview:')
    print('Columns amount: %d' % len(train_data[0]))
    print('Data M amount: %d' % len(data_m))
    print('Data B amount: %d' % len(data_b))
    print('Train data amount: %d' % len(train_data))
    print('Test data amount: %d' % len(test_data))


    # train decision tree
    tree = decision_tree_kv.buildtree(train_data, scoref=decision_tree_kv.giniimpurity)

    # check accuracy
    stat: Dict[str, int]
    stat = {'equal': 0, 'not_equal': 0, 'not_equal_m': 0, 'not_equal_b': 0}
    for test_row in test_data:
        rigth_diagnosis = test_row[len(test_row) - 1]
        row_without_result = test_row[0:len(test_row)-1]

        r: dict = decision_tree_kv.classify(row_without_result, tree)
        if len(r.keys()) > 1:
            raise BaseException('so many keys in predict data')
        predict_diagnosis: str = list(r.keys())[0]
        if predict_diagnosis == rigth_diagnosis:
            stat['equal'] += 1
        else:
            stat['not_equal'] += 1
            if rigth_diagnosis == "M":
                stat['not_equal_m'] += 1
            elif rigth_diagnosis == "B":
                stat['not_equal_b'] += 1
            else:
                raise BaseException('Undefined rigth_diagnosis')

    print('Result of experiments #%d :' % (experiment_number + 1))
    print('Equal diagnosis amount: %d' % stat['equal'])
    print('Not equal diagnosis amount: %d' % stat['not_equal'])
    print('Amount of mistakes with "M" diagnosis: %d' % stat['not_equal_m'])
    print('Amount of mistakes with "B" diagnosis: %d' % stat['not_equal_b'])

    divider: float = (stat['equal'] + stat['not_equal'])/100
    accuracy: float = 0.0
    if not divider == 0:
        accuracy = stat['equal']/divider
    print('Accuracy '+str(accuracy)+'%')
    accuracy_list.append(accuracy)
    print('')
    print('')

final_accuracy = float(sum(accuracy_list)) / float(len(accuracy_list))
print('Final accuracy: %f' % final_accuracy)
import os
# print(os.listdir("../input"))
from typing import List, Dict, Tuple, Union, Set
import csv
# source https://github.com/kopylovvlad/decision_tree_kv
import decision_tree_kv
import random
import re

all_data:  List[List[str]]
train_data: List[List[Union[float, str]]] = []
test_data: List[List[Union[float, str]]] = []
data_m: List[List[Union[float, str]]] = []
data_b: List[List[Union[float, str]]] = []    

def get_uniq_col_indexes(tree_string: str) -> None:
    reg_res: List = re.compile(r'column\[(\d+)\]').findall(tree_string)
    reg_res = list(set(reg_res))
    reg_res = [int(i) for i in reg_res]
    reg_res.sort()
    print(reg_res)
    return None

    
all_data = list(csv.reader(open('../input/data.csv', newline='')))
train_data = []
test_data = []
data_m = []
data_b = []    

i: int = 0
# collect data to data_m and data_b sebsets
for row in all_data:
    i += 1
    if row[0] == 'id':
        # ignore head-row
        continue

    tmp_row: List[Union[str, float]] = []
    diagnosis: str = row[1]

    for j in range(len(row)):
        # without id and diagnosis
        if j == 0 or j == 1:
            continue
        tmp_row.append(float(row[j]))

    # set diagnosis as last column
    tmp_row.append(str(diagnosis))

    if diagnosis == 'M':
        data_m.append(tmp_row)
    elif diagnosis == 'B':
        data_b.append(tmp_row)
    else:
        raise BaseException('Undefined diagnosis')


i = 0
# filling train data set
for row in data_m:
    i += 1
    if i < len(data_m) * 0.8:
        train_data.append(row)
    else:
        test_data.append(row)

i = 0
# filling test data set
for row in data_b:
    i += 1
    if i < len(data_b) * 0.8:
        train_data.append(row)
    else:
        test_data.append(row)

# train decision tree
tree = decision_tree_kv.buildtree(train_data, scoref=decision_tree_kv.giniimpurity)

print('Decision tree flowchart:')
tree_str: str = decision_tree_kv.tree_to_str(tree)
print(tree_str)
print('')

print('List of uniq columns in flowchart:')
get_uniq_col_indexes(tree_str)
import os
# print(os.listdir("../input"))
from typing import List, Dict, Tuple, Union, Set
import csv
# source https://github.com/kopylovvlad/decision_tree_kv
import decision_tree_kv
import random

all_data:  List[List[str]]
train_data: List[List[Union[float, str]]] = []
test_data: List[List[Union[float, str]]] = []
data_m: List[List[Union[float, str]]] = []
data_b: List[List[Union[float, str]]] = []    
accuracy_list: List[float] = []

def get_uniq_col_indexes(tree_string: str) -> None:
    reg_res: List = re.compile(r'column\[(\d+)\]').findall(tree_string)
    reg_res = list(set(reg_res))
    reg_res = [int(i) for i in reg_res]
    reg_res.sort()
    print(reg_res)
    return None

    
for experiment_number in range(4):
    print('Running experiment %d' % (experiment_number + 1))
    all_data = list(csv.reader(open('../input/data.csv', newline='')))
    train_data = []
    test_data = []
    data_m = []
    data_b = []    

    # shuffle data
    random.shuffle(all_data)

    # collect data to data_m and data_b sebsets
    for row in all_data:
        i += 1
        if row[0] == 'id':
            # ignore head-row
            continue
        
        tmp_row2: List[Union[str, float]] = []
        tmp_row: List[Union[str, float]] = []
        diagnosis: str = row[1]

        for j in range(len(row)):
            # without id and diagnosis
            if j == 0 or j == 1:
                continue
            tmp_row.append(float(row[j]))
        
        # limit for features amount
        for jj in range(len(tmp_row)):
          if jj in [1, 4, 5, 13, 21, 22, 24, 27, 28]:
            tmp_row2.append(tmp_row[jj])
        
        tmp_row = tmp_row2
        # set diagnosis as last column
        tmp_row.append(str(diagnosis))

        if diagnosis == 'M':
            data_m.append(tmp_row)
        elif diagnosis == 'B':
            data_b.append(tmp_row)
        else:
            raise BaseException('Undefined diagnosis')


    i = 0
    # filling train data set
    for row in data_m:
        i += 1
        if i < len(data_m) * 0.8:
            train_data.append(row)
        else:
            test_data.append(row)

    i = 0
    # filling test data set
    for row in data_b:
        i += 1
        if i < len(data_b) * 0.8:
            train_data.append(row)
        else:
            test_data.append(row)
    
    print('Data overview:')
    print('Columns amount: %d' % len(train_data[0]))
    print('Data M amount: %d' % len(data_m))
    print('Data B amount: %d' % len(data_b))
    print('Train data amount: %d' % len(train_data))
    print('Test data amount: %d' % len(test_data))


    # train decision tree
    tree = decision_tree_kv.buildtree(train_data, scoref=decision_tree_kv.giniimpurity)

    # check accuracy
    stat: Dict[str, int]
    stat = {'equal': 0, 'not_equal': 0, 'not_equal_m': 0, 'not_equal_b': 0}
    for test_row in test_data:
        rigth_diagnosis = test_row[len(test_row) - 1]
        row_without_result = test_row[0:len(test_row)-1]

        r: dict = decision_tree_kv.classify(row_without_result, tree)
        if len(r.keys()) > 1:
            raise BaseException('so many keys in predict data')
        predict_diagnosis: str = list(r.keys())[0]
        if predict_diagnosis == rigth_diagnosis:
            stat['equal'] += 1
        else:
            stat['not_equal'] += 1
            if rigth_diagnosis == "M":
                stat['not_equal_m'] += 1
            elif rigth_diagnosis == "B":
                stat['not_equal_b'] += 1
            else:
                raise BaseException('Undefined rigth_diagnosis')

    print('Result of experiments #%d :' % (experiment_number + 1))
    print('Equal diagnosis amount: %d' % stat['equal'])
    print('Not equal diagnosis amount: %d' % stat['not_equal'])
    print('Amount of mistakes with "M" diagnosis: %d' % stat['not_equal_m'])
    print('Amount of mistakes with "B" diagnosis: %d' % stat['not_equal_b'])

    divider: float = (stat['equal'] + stat['not_equal'])/100
    accuracy: float = 0.0
    if not divider == 0:
        accuracy = stat['equal']/divider
    print('Accuracy '+str(accuracy)+'%')
    accuracy_list.append(accuracy)
    print('')
    print('')

final_accuracy = float(sum(accuracy_list)) / float(len(accuracy_list))
print('Final accuracy: %f' % final_accuracy)


print('Decision tree flowchart:')
tree_str: str = decision_tree_kv.tree_to_str(tree)
print(tree_str)
print('')

print('List of uniq columns in flowchart:')
get_uniq_col_indexes(tree_str)
