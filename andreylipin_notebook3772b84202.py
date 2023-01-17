import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train2.csv')
df.head()
df['choose'][(df['mean_exam_points']<50)&(df['choose']==0)]=1
df['choose'][(df['mean_exam_points']<50)&(df['choose']==1)]=0
sns.boxenplot(df['choose'],df['mean_exam_points']);
dups_shape = df.pivot_table(index=['mean_exam_points'], aggfunc='size')
dups_shape
df.describe().T
sns.distplot(df['age'],bins=15)
sns.scatterplot(df['lesson_price'],df['mean_exam_points']);
sns.boxplot(df['qualification'],df['mean_exam_points'],orient='v');
df['age_group']=pd.cut(df['age'],[0,40,46,51,100],labels=[1,2,3,4]).astype('float')
df['price_group']=pd.cut(df['lesson_price'],[0,1300,1550,2150,10000],labels=[1,2,3,4]).astype('float')
df.loc[df['years_of_experience']>0,'year_age_group']=df.loc[df['years_of_experience']>0,'age']/df.loc[df['years_of_experience']>0,'years_of_experience']
df.loc[df['years_of_experience']==0,'year_age_group']=df.loc[df['years_of_experience']==0,'age']
quartiles = [df['year_age_group'].quantile(0),
            df['year_age_group'].quantile(0.25),
            df['year_age_group'].quantile(0.5),
            df['year_age_group'].quantile(0.75),
            df['year_age_group'].quantile(1)]
df['year_age_group'] = pd.cut(df['year_age_group'],quartiles,labels=[1,2,3,4]).astype('float')
quartiles_score = [df['mean_exam_points'].quantile(0),
#             df['mean_exam_points'].quantile(0.25),
#             df['mean_exam_points'].quantile(0.5),
            df['mean_exam_points'].quantile(0.75),
            df['mean_exam_points'].quantile(1)]
df['mean_exam_points_group']=pd.cut(df['mean_exam_points'],quartiles_score,labels=[1,2]).astype('float')
df.head(10)
fig,axis = plt.subplots(1,4,figsize=(15,10))
for i in df['price_group'].unique():
    sns.boxplot(df[df['price_group']==i]['age_group'],df[df['price_group']==i]['mean_exam_points'],orient='v',ax=axis[int(i)-1])
    plt.tight_layout()
df.describe().T
sns.boxplot(df['price_group'],df['mean_exam_points'],orient='v');
df['qualification_height']=pd.cut(df['qualification'],[-1,3,5],labels=[0,1]).astype('float')

df.head()
data, labels = df.iloc[:,1:].drop('choose',axis=1).values,df.iloc[:,12].values
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
        self.data = data # значения признаков
        self.labels = labels  # y_true
        self.prediction = self.predict()  # y_pred
        
    def predict(self):
        # подсчет количества объектов разных классов
        classes = {}  # сформируем словарь "класс: количество объектов"
        for label in self.labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1
        #  найдем класс, количество объектов которого будет максимальным в этом листе и вернем его   
        prediction = max(classes, key=classes.get)
        return prediction
def gini(labels):
    #  подсчет количества объектов разных классов
    classes = {}
    for label in labels:
        if label not in classes:
            classes[label] = 0
        classes[label] += 1
    
    #  расчет критерия
    impurity = 1
    for label in classes:
        p = classes[label] / len(labels)
        impurity -= p ** 2
        
    return impurity
# Расчет качества

def quality(left_labels, right_labels, current_gini):

    # доля выборки, ушедшей в левое поддерево
    p = float(left_labels.shape[0]) / (left_labels.shape[0] + right_labels.shape[0]) # для правого (1-p)
    
    return current_gini - p * gini(left_labels) - (1 - p) * gini(right_labels) # Функционал качества
# Разбиение датасета в узле

def split(data, labels, index, t):
    
    left = np.where(data[:, index] <= t)
    right = np.where(data[:, index] > t)
        
    true_data = data[left]
    false_data = data[right]
    true_labels = labels[left]
    false_labels = labels[right]
        
    return true_data, false_data, true_labels, false_labels
# Нахождение наилучшего разбиения

def find_best_split(data, labels):
    
    #  обозначим минимальное количество объектов в узле
    min_leaf = 5

    current_gini = gini(labels) 

    best_quality = 0
    best_t = None # лучший порог разбиения
    best_index = None # лучший индекс разбиения
    
    n_features = data.shape[1] # кол-во признаков
    
    for index in range(n_features): # проход по всем признакам
        t_values = [row[index] for row in data] # берем столбец/признак с соотв. индексом
        
        for t in t_values: # проход по признаку
            true_data, false_data, true_labels, false_labels = split(data, labels, index, t) # делаем разбиение
            #  пропускаем разбиения, в которых в узле остается менее 5 объектов
            if len(true_data) < min_leaf or len(false_data) < min_leaf:
                continue # начинаем следующий проход цикла, минуя оставшееся тело цикла
            
            # расчет качества текущего разбиения
            current_quality = quality(true_labels, false_labels, current_gini)
            
            #  выбираем порог, на котором получается максимальный прирост качества
            if current_quality > best_quality:
                best_quality, best_t, best_index = current_quality, t, index

    return best_quality, best_t, best_index
# Построение дерева с помощью рекурсивной функции

def build_tree(data, labels):

    quality, t, index = find_best_split(data, labels) # ищем лучшее разбиение
    #  Базовый случай - прекращаем рекурсию, когда нет прироста в качества
    # неопределенность после разбиения осталась такой же как до
    if quality == 0: # критерий останова
#         print('leaf')
        return Leaf(data, labels) # считаем прогноз для листьев

    # если качество улучшилось, то делим дерево по лучшему разбиению
    true_data, false_data, true_labels, false_labels = split(data, labels, index, t)

    # Рекурсивно строим два поддерева
    true_branch = build_tree(true_data, true_labels)
    false_branch = build_tree(false_data, false_labels)

    # Возвращаем класс узла со всеми поддеревьями, то есть целого дерева
    return Node(index, t, true_branch, false_branch)
# Проход объекта по дереву для его классификации

def classify_object(obj, node):

    #  Останавливаем рекурсию, если достигли листа
    if isinstance(node, Leaf): # проверка текущий узел это лист?
        answer = node.prediction # считаем прогноз для листа
        return answer

    if obj[node.index] <= node.t: # если значение признака меньше порога t
        return classify_object(obj, node.true_branch) # рекурсия: отправляем объект в true-ветку
    else:
        return classify_object(obj, node.false_branch) # рекурсия: отправляем объект в false-ветку
# Предсказание деревом для всего датасета

def predict(data, tree):
    
    classes = []
    for obj in data:
        prediction = classify_object(obj, tree) # определяем ветки для объектов
        classes.append(prediction)
    return classes
# Построим дерево по обучающей выборке
my_tree = build_tree(data, labels)
# Введем функцию подсчета точности как доли правильных ответов
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
# Точность на обучающей выборке
answers = predict(data, my_tree)
train_accuracy = accuracy_metric(labels, answers)
train_accuracy
test_df = pd.read_csv('test2.csv',index_col=0)
test_df['age_group']=pd.cut(test_df['age'],[0,40,46,61,100],labels=[1,2,3,4]).astype('float')
test_df['price_group']=pd.cut(test_df['lesson_price'],[0,1300,1500,2150,10000],labels=[1,2,3,4]).astype('float')
test_df['qualification_height']=pd.cut(test_df['qualification'],[0,3,5],labels=[0,1]).astype('float')
test_df['year_age_group']=''
test_df.loc[test_df['years_of_experience']>0,'year_age_group']=test_df.loc[test_df['years_of_experience']>0,'age']/test_df.loc[test_df['years_of_experience']>0,'years_of_experience']
test_df.loc[test_df['years_of_experience']==0,'year_age_group']=test_df.loc[test_df['years_of_experience']==0,'age']
test_df['year_age_group'] = pd.cut(test_df['year_age_group'],quartiles,labels=[1,2,3,4]).astype('float')
test_df['mean_exam_points_group']=pd.cut(test_df['mean_exam_points'],quartiles_score,labels=[1,2]).astype('float')
a = test_df['qualification'].copy()
b = test_df['price_group'].copy()

for i in test_df['qualification'].unique():
    for k in test_df.columns:
        test_df.loc[test_df['qualification']==i,k] = (test_df.loc[test_df['qualification']==i,k].mean() - test_df.loc[test_df['qualification']==i,k]) / test_df.loc[test_df['qualification']==i,k].mean()
        
test_df['qualification'] = a
test_df['price_group'] = b
# Просмотр результата
test_df.head()


answers = predict(test_df.values, my_tree)

test_df['choose'] = answers
test_df['choose'].to_csv('andrey_lipin_solution2.csv')
test_df['choose'].value_counts()