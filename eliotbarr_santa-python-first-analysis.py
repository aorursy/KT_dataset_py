import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
santa = pd.read_csv('../input/gifts.csv')
santa.head()
amt = 100000

class Horse:

    def __init__(self,id):

        self.weight = max(0, np.random.normal(5,2,1)[0])

        self.name = 'horse_' + str(id)



class Ball:

    def __init__(self,id):

        self.weight = max(0, 1 + np.random.normal(1,0.3,1)[0])

        self.name = 'ball_' + str(id)



class Bike:

    def __init__(self,id):

        self.weight = max(0, np.random.normal(20,10,1)[0])

        self.name = 'bike_' + str(id)



class Train:

    def __init__(self,id):

        self.weight = max(0, np.random.normal(10,5,1)[0])

        self.name = 'train_' + str(id)

        

class Coal:

    def __init__(self,id):

        self.weight = 47 * np.random.beta(0.5,0.5,1)[0]

        self.name = 'coal_' + str(id)

        

class Book:

    def __init__(self,id):

        self.weight = np.random.chisquare(2,1)[0]

        self.name = "book_" + str(id)

        

class Doll:

    def __init__(self,id):

        self.weight = np.random.gamma(5,1,1)[0]

        self.name = "doll_" + str(id)



class Block:

    def __init__(self,id):

        self.weight = np.random.triangular(5,10,20,1)[0]

        self.name = "blocks_" + str(id)

        

class Gloves:

    def __init__(self,id):

        self.weight = 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]

        self.name = "gloves_" + str(id)



books = [Book(x) for x in range(amt)]

horses = [Horse(x) for x in range(amt)]

bikes = [Bike(x) for x in range(amt)]

trains = [Train(x) for x in range(amt)]

coals = [Coal(x) for x in range(amt)]

dolls = [Doll(x) for x in range(amt)]

balls = [Ball(x) for x in range(amt)]

blocks = [Block(x) for x in range(amt)]

gloves = [Gloves(x) for x in range(amt)]
sns.distplot([horse.weight for horse in horses],bins=100)

plt.title('horses distribution')
pd.DataFrame([horse.weight for horse in horses]).describe()
sns.distplot([book.weight for book in books],bins=100)

plt.title('books distribution')
pd.DataFrame([book.weight for book in books]).describe()
sns.distplot([bike.weight for bike in bikes],bins=100)

plt.title('bikes distribution')
pd.DataFrame([bike.weight for bike in bikes]).describe()
sns.distplot([train.weight for train in trains],bins=100)

plt.title('trains distribution')
pd.DataFrame([train.weight for train in trains]).describe()
sns.distplot([coal.weight for coal in coals],bins=100)

plt.title('coals distribution')
pd.DataFrame([coal.weight for coal in coals]).describe()
sns.distplot([doll.weight for doll in dolls],bins=100)

plt.title('dolls distribution')
pd.DataFrame([doll.weight for doll in dolls]).describe()
sns.distplot([ball.weight for ball in balls],bins=100)

plt.title('balls distribution')
pd.DataFrame([ball.weight for ball in balls]).describe()
sns.distplot([block.weight for block in blocks],bins=100)

plt.title('blocks distribution')
pd.DataFrame([block.weight for block in blocks]).describe()
sns.distplot([glove.weight for glove in gloves],bins=100)

plt.title('gloves distribution')
pd.DataFrame([glove.weight for glove in gloves]).describe()
santa_list = [(gift.split('_')[0], gift.split('_')[1]) for gift in santa['GiftId']]

santa_element = pd.DataFrame(santa_list,columns=['Gift','Id'])
santa_element.head()
sns.countplot(x = 'Gift', data=santa_element)

plt.title('Gift Count')
santa_dummy = pd.get_dummies(santa_element['Gift'])
santa_full = pd.concat([santa_dummy, santa_element['Id']], axis=1)
santa_full.head()