import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import csv
import random

def list_take(list_elements, list_drop):
    list_new = list()
    for i, ele in enumerate(list_elements):
        if i == 1:
            survived = ele
            continue
        if i not in list_drop:
            list_new.append(ele)
    list_new.append(survived)
    return list_new


def list_take(list_elements, list_drop, label_modify=False):
    list_new = list()
    for i, ele in enumerate(list_elements):
        if label_modify and i == 1:
            survived = ele
            continue
        if i not in list_drop:
            list_new.append(ele)
    if label_modify:
        list_new.append(survived)
    return list_new


def load_data(file_path="train.csv", encoding="utf-8"):
    """load data from file_path encode as encoding
    split datas and labels into train and test"""
    data_set = []
    with open(file_path, encoding=encoding) as fp:
        csv_reader = csv.reader(fp)
        for ri, row in enumerate(csv_reader):
            if ri == 0:
                feat_list = list_take(row, [0, 3, 8, 10], True)
            else:
                data_set.append(list_take(row, [0, 3, 8, 10], True))
    data_set = np.array(data_set)
    age_avg = int(np.mean(data_set[data_set[:, 2] != '', 2].astype(np.float)))
    data_set[data_set[:, 2] == '', 2] = str(age_avg)
    embark_list = list(set(data_set[:, -2].tolist()) - {''})
    null_num =len(np.nonzero(data_set[:, -2] == ''))
    mul = int(null_num / len (embark_list)) + 1
    data_set[data_set[:, -2] == '', -2] = np.array(random.sample(embark_list * mul , null_num))
    return data_set[:, :-1], data_set[:, -1], feat_list[:-1]


def load_xdata(file_path="train.csv", encoding="utf-8"):
    """load data from file_path encode as encoding
    split datas and labels into train and test"""
    data_set = []
    with open(file_path, encoding=encoding) as fp:
        csv_reader = csv.reader(fp)
        for ri, row in enumerate(csv_reader):
            if ri == 0:
                continue
            else:
                data_set.append(list_take(row, [0, 2, 7, 9], False))
    data_set = np.array(data_set)
    age_avg = int(np.mean(data_set[data_set[:, 3] != '', 3].astype(np.float)))
    data_set[data_set[:, 3] == '', 3] = str(age_avg)
    embark_list = list(set(data_set[:, -1].tolist()) - {''})
    null_num =len(np.nonzero(data_set[:, -1] == ''))
    mul = int(null_num / len (embark_list)) + 1
    data_set[data_set[:, -1] == '', -1] = np.array(random.sample(embark_list * mul , null_num))
    return data_set


def load_ydata(file_path="train.csv", encoding="utf-8"):
    """load data from file_path encode as encoding
    split datas and labels into train and test"""
    data_set = []
    with open(file_path, encoding=encoding) as fp:
        csv_reader = csv.reader(fp)
        for ri, row in enumerate(csv_reader):
            if ri == 0:
                continue
            else:
                data_set.append(row[-1])
    return np.array(data_set)



def map_str_label(data_mat):
    feat_nums = data_mat.shape[1]
    feat_map = []
    for i in range(feat_nums):
        value_set = set(data_mat[:, i].tolist())
        j = 0
        feat_dict = {}
        for value in value_set:
            feat_dict[j] = value
            data_mat[data_mat[:, i] == value, i] = j
            j = j + 1
        feat_map.append(feat_dict)
    return data_mat, feat_map


def test_accuracy(x, y, xtest_input=None, ytest_input=None):

    ''''' 拆分训练数据与测试数据 '''
    x, x_label = map_str_label(x)
    y, y_label = map_str_label(y)
    y = y.reshape(y.shape[0])
    if xtest_input is None or ytest_input is None:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    else:
        x_test, _ = map_str_label(xtest_input)
        y_test, _ = map_str_label(ytest_input)
        x_train, y_train = x, y 

    ''''' 使用信息熵作为划分标准，进行随机森林训练 '''
    clf = RandomForestClassifier(n_estimators=10, criterion="entropy", max_features="auto", min_samples_split=4)
    # print(clf)
    clf.fit(x_train, y_train)

    # class_names = []
    # for class_label in y_label[0].values():
    #     class_names.append(class_label)
    ''''' 把决策树结构写入文件 '''
    # with open("car_data.dot", 'w+') as f:
    #     f = tree.export_graphviz(clf, feature_names=feat_list, class_names=class_names, rounded=True, out_file=f)

    ''''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
    # print(clf.feature_importances_)
    '''''测试结果的打印'''
    answer = clf.predict(x_test)
    
    # print(x_test)
    # print(answer)
    # print(y_test)
    print(np.mean(answer == y_test))
    return np.mean(answer == y_test)
    # print(np.mean(answer == y_test))
accuracy = 0.0
time = 1000
data, labels, feat_list = load_data("../input/train.csv")
x = np.array(data)
xtest = load_xdata("../input/test.csv")
ytest = load_ydata("../input/gender_submission.csv")
labels = np.array(labels)
y = labels.reshape((x.shape[0], 1))
while time:
    time = time - 1
    accuracy += test_accuracy(x, y, xtest, ytest)
accuracy /= 1000
print("accuracy:\t", accuracy)