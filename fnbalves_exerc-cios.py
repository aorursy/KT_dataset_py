#Crie classificadores para o MNIST dataset
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
data, target = load_digits(return_X_y=True)

label_set = list(set(target))
print('Num examples', len(data))
print('Shape', np.shape(data[0]))
print('Label set', label_set)

#Desenvolva seus estudos a partir daqui
#Crie um classificador para diferenciar pessoas fisicas ou juridicas
import pickle

corpus = pickle.load(open('../input/name_company_corpus.pickle', 'rb'))

names = [x[0] for x in corpus if x[1] == 'NAME']
companies = [x[0] for x in corpus if x[1] == 'COMPANY']

all_texts = names + companies
labels = [0]*(len(names)) + [1]*(len(companies))
print('NUM PEOPLE', len(names))
print('NUM COMPANIES', len(companies))

#Desenvolva seus estudos a partir daqui


