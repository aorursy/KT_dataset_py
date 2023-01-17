from typing import List, Tuple, Dict, Set, Callable
import knn_kv as knn
import os
import random
import csv
print(os.listdir("../input"))

head_row = ["", "carat", "cut", "color", "clarity",
            "depth", "table", "price", "x", "y", "z"]
cut_set: Set[str] = set()
color_set: Set[str] = set()
clarity_set: Set[str] = set()

csvfile = open('../input/diamonds.csv', newline='')
filereader = csv.reader(csvfile, delimiter=',', quotechar='"')
    
i = 0
for row in filereader:
    i += 1
    if i == 1:
        continue
    for j in range(len(head_row)):
        if j == 0:
            continue
        if head_row[j] == 'cut':
            cut_set.add(row[j])
        elif head_row[j] == 'color':
            color_set.add(row[j])
        elif head_row[j] == 'clarity':
            clarity_set.add(row[j])


cut_list: List[str] = list(cut_set)
color_list: List[str] = list(color_set)
clarity_list: List[str] = list(clarity_set)

print('We have %d values in "cut"' % len(cut_list))
print('We have %d values in "color"' % len(color_list))
print('We have %d values in "clarity"' % len(clarity_list))
train_samples: List[dict] = []
test_samples:List[dict] = []
i: int = 0
x_index = head_row.index('x')
y_index = head_row.index('y')
limit: float = 0.8
csvfile = open('../input/diamonds.csv', newline='')
filereader = list(csv.reader(csvfile, delimiter=',', quotechar='"'))
random.shuffle(filereader)

filereader_len = len(filereader)
for row in filereader:
    i += 1
    if row[1] == 'carat':
        continue
    price = 0.0
    features = []
    for j in range(len(head_row)):
        if j == 0:
            continue

        if head_row[j] == 'price':
            price = float(row[j])
        elif head_row[j] == 'cut':
            features.append(
                float(cut_list.index(row[j]))
            )
        elif head_row[j] == 'color':
            features.append(
                float(color_list.index(row[j]))
            )
        elif head_row[j] == 'clarity':
            features.append(
                float(clarity_list.index(row[j]))
            )
        else:
            features.append(
                float(row[j])
            )

    # add diamond ration
    if float(row[y_index]) > 0.0:
        features.append(
            float(row[x_index]) / float(row[y_index]))
    else:
        features.append(0)

    if i < filereader_len * limit:
        train_samples.append({
            'features': features,
            'result': price
        })
    else:
        test_samples.append({
            'features': features,
            'result': price
        })

print('We have %d train samples' % len(train_samples))
print('We have %d test samples' % len(test_samples))

def main_test(train_samples: List[dict], test_samples: List[dict], func: Callable, delta: int = 100) -> None:
    stat: Dict[str, int] = { 'equal': 0, 'not_equal': 0 }
    for test_sample in test_samples:
        is_equal: bool = func(train_samples, test_sample, delta)
        if is_equal == True:
            stat['equal'] = stat['equal'] + 1
        else:
            stat['not_equal'] = stat['not_equal'] + 1
    accuracy: float = 0.0
    all: int = stat['equal'] + stat['not_equal']
    if all / 100 > 0:
        accuracy = stat['equal'] / (all / 100)
    print('Equal prices: %d samples' % stat['equal'])
    print('Not equal prices: %d samples' % stat['not_equal'])
    print('Acuracy: %s percents' % str(round(accuracy, 1)))
    return None
print('ready')
def knn_factory(neighbours_number: int = 6):
    def knn_function(train_samples: List[dict], test_sample: dict, delta: float) -> bool:
        predict_price: float = knn.knnestimate(train_samples, test_sample['features'], neighbours_number)
        real_price: float = test_sample['result']
        return real_price - delta <= predict_price <= real_price + delta 
    return knn_function

knn6_function = knn_factory(6)
knn5_function = knn_factory(5)
knn4_function = knn_factory(4)
knn3_function = knn_factory(3)

print('Test for standart k-nn algorithm with 6 neighbours:')
main_test(
    train_samples,
    test_samples[:200],
    knn6_function
)
print('')

print('Test for standart k-nn algorithm with 5 neighbours:')
main_test(
    train_samples,
    test_samples[:200],
    knn5_function
)
print('')

print('Test for standart k-nn algorithm with 4 neighbours:')
main_test(
    train_samples,
    test_samples[:200],
    knn4_function
)
print('')

print('Test for standart k-nn algorithm with 3 neighbours:')
main_test(
    train_samples,
    test_samples[:200],
    knn3_function
)
print('')

def weightedknn_factory(neighbours_number: int = 6, weightf=knn.gaussian):
    def knn_function(train_samples: List[dict], test_sample: dict, delta: float) -> bool:
        predict_price: float = knn.weightedknn(
            train_samples, 
            test_sample['features'], 
            neighbours_number,
            weightf=weightf)
        real_price: float = test_sample['result']
        return real_price - delta <= predict_price <= real_price + delta 
    return knn_function

print('gaussian function:')
print('Test for weighted k-nn algorithm with 3 neighbours:')
main_test(
    train_samples,
    test_samples[:400],
    weightedknn_factory(3)
)
print('')
print('')


print('inverseweight function:')
print('Test for weighted k-nn algorithm with 3 neighbours:')
main_test(
    train_samples,
    test_samples[:400],
    weightedknn_factory(3, weightf=knn.inverseweight)
)
print('')


print('subtractweight function:')
print('Test for weighted k-nn algorithm with 3 neighbours:')
main_test(
    train_samples,
    test_samples[:400],
    weightedknn_factory(3, weightf=knn.subtractweight)
)
print('')
print('end')
weighted3knn_function = weightedknn_factory(3, weightf=knn.inverseweight)

new_head_row = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z", "ratio"]

def delete_columns(columns_index: List[int], samples: List[dict]) -> List[dict]:
    new_samples: List[dict] = []
    for sample in samples:
        new_features: List[float] = []
        for i in range(len(sample['features'])):
            if i in columns_index:
                continue
            new_features.append(sample['features'][i])
        new_samples.append({
            'features': new_features,
            'result': sample['result']
        })
    return new_samples

print('default:')
main_test(
    delete_columns([],train_samples),
    delete_columns([],test_samples[:400]),
    weighted3knn_function
)
print('')

for i in range(len(new_head_row)):
    head_row = new_head_row[i]
    print('Accuracy without "%s" column' % head_row)
    main_test(
        delete_columns([i],train_samples),
        delete_columns([i],test_samples[:400]),
        weighted3knn_function
    )
    print('')

print('end')
table_index = 5
print('default:')
main_test(
    delete_columns([table_index], train_samples),
    delete_columns([table_index], test_samples[:800]),
    weighted3knn_function
)
print('')

for i in range(len(new_head_row)):
    if i == 5:
        continue

    head_row = new_head_row[i]
    without_columns: List[int] = [ new_head_row[table_index], new_head_row[i] ]
    print('Accuracy without ', end='')
    print(', '.join(without_columns), end='')
    print(' columns')
    without_columns_indexes = [5, i]
    
    main_test(
        delete_columns(without_columns_indexes, train_samples),
        delete_columns(without_columns_indexes, test_samples[:800]),
        weighted3knn_function
    )
    print('')

# table, depth
table_index = 5
depth_index = 4

print('Run experiment without coulms: "table", "depth"')
main_test(
    delete_columns([table_index, depth_index], train_samples),
    delete_columns([table_index, depth_index], test_samples),
    weighted3knn_function
)
print('end')
