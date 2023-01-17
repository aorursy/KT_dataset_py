# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import time
import random
X = pd.read_csv('D:/data/x.csv')
Y = pd.read_csv('D:/data/y.csv')
input_data = pd.concat([X,Y], axis = 1)
"""class tag:
    def __init(self, Id, Value, Operator, fitness = 0):
        self.id = Id
        self.value = Value
        self.operator = Operator"""

# Operator는 get_fitness단계에서 바꿔가면서 해보자
# 왜냐하면 이 데이터는 한두개의 조합으로 끝나버리니깐.
# 3개 이상 조합이 안나오는 것을 확인하면 됨.
class tag:
    def __init__(self, Id, Value, fitness = 0):
        self.id = Id
        self.value = Value
        
class firefly:
    def __init__(self, tag_class_list, fitness):
        self.tag_class_list = tag_class_list
        self.fitness = fitness
def get_tag_list(input_data, target_N, flies_cnt, n = 1):
    # n = 1 일 때는 정말 다해본다.
    """if n == 1:
        
    else:"""
        
    if n >1 :
    # n > 1 일 때는 조합별로 한 번 50개 뿌려보고 별로면 pass
    # 별로의 기준 : 전부 target_N도 작고 성능도 작은경우. 이 때는 나아질 기미가 없기 때문
        firefly_list = []
        fitness_list = [0]
        firefly_list_add = []
        for i in range(2, n+1):
            iters = 1
            while max(fitness_list) <= 0.1 or iters < 10 :
                fitness_list = []
                firefly_list_add = []
                iters += 1
                tag_ids = random.sample(list(input_data.columns[:50]), k = i)
                for j in range(flies_cnt):
                    tag_class_list = []
                    # 각 태그의 median 근방에서 시작하도록 수정
                    tag_values = random.choices([2,3,4,5,6,7,8], weights = [0.1,0.1,0.3,0.3,0.3,0.1,0.1],k=2)
                    for tag_id, tag_value in zip(tag_ids, tag_values):
                        tag_class_list.append(tag(tag_id, tag_value))
                    fitness = get_fitness(input_data, target_N, tag_class_list)
                    fitness_list.append(fitness)
                    firefly_list_add.append(firefly(tag_class_list, fitness))
                    print('fitness_', max(fitness_list), 'tag_ids', tag_ids, 'tag_value', tag_values)
            firefly_list += firefly_list_add
            print(firefly_list)
        return firefly_list
def get_fitness(data, target_N, tag_class_list):
    out_data = data.copy(deep = True)
    operator_set = ['+', '+']
    # operator 조합을 여기서 여러개 해볼까???
    # operator를 유전 알고리즘으로? 아니면 Tabu search? or greedy algoritm
    """for tag in tag_class_list:
        if tag.operator == '+':
            out_data = out_data[out_data[tag.id] >= tag.value]
        elif tag.operator == '-':
            out_data = out_data[out_data[tag.id] >= tag.value]
    """    
    for tag, op in zip(tag_class_list, operator_set):
        if op == '+':
            out_data = out_data[out_data[tag.id] >= tag.value]
        elif op == '-':
            out_data = out_data[out_data[tag.id] >= tag.value]
        
        cnt = out_data.shape[0]
        response_rate = np.mean(out_data['y'])
        
        if cnt > target_N * 2 or cnt < target_N*0.5:
            return response_rate - 0.1 # 최종으로 뽑힌 인원이 target 근방이 아닌 경우 penaly
        else:
            return response_rate
def move(firefly_i, firefly_j):
    # 거리
    r_list = []
    for k, (i, j) in enumerate( zip(firefly_i.tag_class_list, firefly_i.tag_class_list) ):
        r_list.append((i.value -j.value)**2)
    r = np.sqrt(sum(r_list))
    for k, (i, j) in enumerate( zip(firefly_i.tag_class_list, firefly_i.tag_class_list) ):
        gamma = 0.001; beta0 = 1; alpha = 2.5
        attrac = beta0*(1/(1+gamma*r**2))
        # 이동 얼마나 하는지 확인
        if firefly_i.tag_class_list[k].value <= 9 and firefly_i.tag_class_list[k].value > 1:
            firefly_i.tag_class_list[k].value += attrac*(i.value -j.value) + alpha*random.uniform(-1/2,1/2)
    return firefly_i
def random_move(firefly_i):
    for i, j in enumerate(firefly_i.tag_class_list):
        if firefly_i.tag_class_list[i].value <= 9 and firefly_i.tag_class_list[i].value > 1:
            firefly_i.tag_class_list[i].value += 2*random.uniform(-1/2, -1/2)
    return firefly_i
def Firefly_Algorithm(input_data, target_N = 10, flies_cnt = 50, iters = 100):
    # initial population
    firefly_list = get_tag_list(input_data,target_N,flies_cnt, 2)
    
    best_list = []
    for t in range(5):
        for i in range(flies_cnt):
            for j in range(i):
                if firefly_list[j].fitness > firefly_list[i].fitness:
                    # i -> j 이동
                    firefly_list[i] = move(firefly_list[i], firefly_list[j])
                    firefly_list[i].fitness = get_fitness(input_data, target_N, firefly_list[i].tag_class_list)
                """else:
                    firefly_list[i] = random_move(firefly_list[i])"""
                # fitness 업데이트
                #print(firefly_list[i].tag_class_list)
                
            # 업데이트 된 fireflies 중에 가장 좋은거 => best 리스트에 저장
            fitness_list = [i.fitness for i in firefly_list]
            f_best = sorted([fitness_list])[-1]
            best_list.append(f_best)
        print("{}번째 싸이클 | Running Time : {} | 성능 : {}".format(t, 1,max(best_list)))
    return best_list
    
    # tag_id 뽑기 1개부터 점점 늘려나감.
    # 1개일 때 좋았던 것들을 고정. => 2개로 늘렸을 때 사용
    
a = Firefly_Algorithm(data)





