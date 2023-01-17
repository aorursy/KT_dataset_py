import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns



# Загрузка датасета

from google.colab import files

import io
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['train.csv']))
df.info()
df.describe()
TARGET_NAME = 'mean_exam_points'

FEATURE_NAMES = ['age', 'years_of_experience', 'lesson_price', 'qualification',

       'physics', 'chemistry', 'biology', 'english', 'geography', 'history']
# Теперь приведём наши данные к формату np.array

X = df[FEATURE_NAMES].values

y = df[TARGET_NAME].values



print(X.shape, y.shape)
# Реализуем класс узла



class Node:

    

    def __init__(self, index, t, true_branch, false_branch):

        self.index = index  # индекс признака, по которому ведется сравнение с порогом в этом узле

        self.t = t  # значение порога

        self.true_branch = true_branch  # поддерево, удовлетворяющее условию в узле

        self.false_branch = false_branch  # поддерево, не удовлетворяющее условию в узле
# И класс терминального узла (листа)



class Leaf:

    

    def __init__(self, data, labels):

        self.data = data

        self.labels = labels

        self.prediction = self.predict()

        

    def predict(self):

        #  найдем значение как среднее по выборке   

        prediction = np.mean(self.labels)

        return prediction
# И класс дерева

class Tree:



  def __init__(self, max_depth=50):

    self.max_depth = max_depth

    self.tree = None



  # Расчёт дисперсии значений

  def dispersion(self, labels):

    return np.std(labels)



  # Расчет качества



  def quality(self, left_labels, right_labels, current_dispersion):



    # доля выбоки, ушедшая в левое поддерево

    p = float(left_labels.shape[0]) / (left_labels.shape[0] + right_labels.shape[0])

    

    return current_dispersion - p * self.dispersion(left_labels) - (1 - p) * self.dispersion(right_labels)



    # Разбиение датасета в узле



  def split(self, data, labels, index, t):

    

    left = np.where(data[:, index] <= t)

    right = np.where(data[:, index] > t)

        

    true_data = data[left]

    false_data = data[right]

    true_labels = labels[left]

    false_labels = labels[right]

        

    return true_data, false_data, true_labels, false_labels



    # Нахождение наилучшего разбиения



  def find_best_split(self, data, labels):

    

    #  обозначим минимальное количество объектов в узле

    min_leaf = 5



    current_dispersion = self.dispersion(labels)



    best_quality = 0

    best_t = None

    best_index = None

    

    n_features = data.shape[1]

    

    for index in range(n_features):

      # будем проверять только уникальные значения признака, исключая повторения

      t_values = np.unique([row[index] for row in data])

      

      for t in t_values:

        true_data, false_data, true_labels, false_labels = self.split(data, labels, index, t)

        #  пропускаем разбиения, в которых в узле остается менее 5 объектов

        if len(true_data) < min_leaf or len(false_data) < min_leaf:

          continue

        

        current_quality = self.quality(true_labels, false_labels, current_dispersion)

        

        #  выбираем порог, на котором получается максимальный прирост качества

        if current_quality > best_quality:

          best_quality, best_t, best_index = current_quality, t, index



    return best_quality, best_t, best_index



    # Построение дерева с помощью рекурсивной функции



  def build_tree(self, data, labels, tree_depth, max_depth):



    quality, t, index = self.find_best_split(data, labels)



    #  Базовый случай - прекращаем рекурсию, когда нет прироста в качества

    if quality == 0:

      return Leaf(data, labels)



    # Базовый случай (2) - прекращаем рекурсию, когда достигнута максимальная глубина дерева

    if tree_depth >= max_depth:

      return Leaf(data, labels)



    # Увеличиваем глубину дерева на 1

    tree_depth += 1



    true_data, false_data, true_labels, false_labels = self.split(data, labels, index, t)



    # Рекурсивно строим два поддерева

    true_branch = self.build_tree(true_data, true_labels, tree_depth, max_depth)

    false_branch = self.build_tree(false_data, false_labels, tree_depth, max_depth)



    # Возвращаем класс узла со всеми поддеревьями, то есть целого дерева

    return Node(index, t, true_branch, false_branch)



  def predict_object(self, obj, node):



    #  Останавливаем рекурсию, если достигли листа

    if isinstance(node, Leaf):

      answer = node.prediction

      return answer



    if obj[node.index] <= node.t:

      return self.predict_object(obj, node.true_branch)

    else:

      return self.predict_object(obj, node.false_branch)



  def predict(self, data):

    

    val = []

    for obj in data:

      prediction = self.predict_object(obj, self.tree)

      val.append(prediction)

    return val



  def fit(self, data, labels):

    self.tree = self.build_tree(data, labels, 0, self.max_depth)

    return self
class GradientBoosting:

  

  def __init__(self, n_trees, max_depth, coefs, eta):

    self.n_trees = n_trees

    self.max_depth = max_depth

    self.coefs = coefs

    self.eta = eta

    self.trees = []



  def bias(self, y, z):

    return (y - z)



  def fit(self, X_train, y_train):

    

    # Деревья будем записывать в список

    trees = []

    

    for i in range(self.n_trees):

        tree = Tree(max_depth=self.max_depth)



        # инициализируем бустинг начальным алгоритмом, возвращающим ноль, 

        # поэтому первый алгоритм просто обучаем на выборке и добавляем в список

        if len(self.trees) == 0:

            # обучаем первое дерево на обучающей выборке

            tree.fit(X_train, y_train)

        else:

            # Получим ответы на текущей композиции

            target = self.predict(X_train)

            

            # алгоритмы начиная со второго обучаем на сдвиг

            bias = self.bias(y_train, target)

            tree.fit(X_train, bias)



        self.trees.append(tree)

        

    return self



  def predict(self, X):

    # Реализуемый алгоритм градиентного бустинга будет инициализироваться нулевыми значениями,

    # поэтому все деревья из списка trees_list уже являются дополнительными и при предсказании прибавляются с шагом eta

    return np.array([sum([self.eta* coef * alg.predict([x])[0] for alg, coef in zip(self.trees, self.coefs)]) for x in X])
def r_2(y_pred, y_true):

  numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)

  denominator = ((y_true - np.average(y_true)) ** 2).sum(axis=0,

                                                          dtype=np.float64)

  return 1 - (numerator / denominator)
train_data, test_data, train_labels, test_labels = train_test_split(X, y, 

                                                                    test_size = 0.3,

                                                                    random_state = 1)
# Число деревьев в ансамбле

n_trees = 10



# для простоты примем коэффициенты равными 1

coefs = [1] * n_trees



# Максимальная глубина деревьев

max_depth = 5



# Шаг

eta = 1
gb = GradientBoosting(n_trees, max_depth, coefs, eta)

gb.fit(train_data, train_labels)

train_answers = gb.predict(train_data)

test_answers = gb.predict(test_data)
r_2(test_answers, test_labels)
r_2(train_answers, train_labels)
gb_final = GradientBoosting(n_trees, max_depth, coefs, eta)

gb_final.fit(X, y)
uploaded = files.upload()
df_test = pd.read_csv(io.BytesIO(uploaded['test.csv']))
df_test.info()
# Теперь приведём наши данные к формату np.array

X_test = df_test[FEATURE_NAMES].values



print(X_test.shape)
test_pred = gb_final.predict(X_test)
submissions = pd.concat([df_test['Id'], pd.Series(test_pred)], axis=1)

submissions = submissions.rename(columns={0: 'mean_exam_points'})
submissions.to_csv('ASirotkin_predictions_1.csv',index=None)
files.download("ASirotkin_predictions_1.csv")