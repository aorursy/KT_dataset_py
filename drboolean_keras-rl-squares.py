# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
%matplotlib inline
with open('../input/game.json', 'r') as contents:
    data = json.load(contents)

def boundingRect(a):
    return {'width': int(a['width']), 'height': int(a['height']), 'x': int(a['x']), 'y': int(a['y'])}
canvas_padding = 0

def getCanvas(group):
    minX = min([el['x'] for el in group])
    minY = min([el['y'] for el in group])
    maxX = max([el['x'] + el['width'] for el in group])
    maxY = max([el['y'] + el['height'] for el in group])
    
    canvasWidth = int(maxX - minX) + canvas_padding
    canvasHeight = int(maxY - minY) + canvas_padding
    canvas = np.zeros((canvasHeight, canvasWidth))
#     print("Canvas: %d, %d" % (canvasHeight, canvasWidth))
    
    # For each element in the group, convert to a pixel value
    for el in group:
        # Top border
        topRow = el['y'] - minY
        canvas[topRow, (el['x'] - minX):(el['x'] + el['width'] - minX)] = 1.0
        
        # Bottom border
        bottomRow = el['y'] - minY + el['height'] - 1
        canvas[bottomRow, (el['x'] - minX):(el['x'] + el['width'] - minX)] = 1.0
        
        # Left border
        leftBorder = el['x'] - minX
        canvas[(el['y'] - minY):(el['y'] - minY + el['height']), leftBorder] = 1.0
        
        # Right border
        rightBorder = el['x'] - minX + el['width'] - 1
        canvas[(el['y'] - minY):(el['y'] - minY + el['height']), rightBorder] = 1.0

    return (canvas, (minX, minY, maxX, maxY))

def render(guesses, canvas):
    plt.imshow(canvas, cmap='Greys', interpolation='none', origin='lower')
    for rect in guesses:
        plt.gca().add_patch(matplotlib.patches.Rectangle((rect['x'], rect['y']), rect['width'], rect['height'], ec='r', fc='none'))

def flatMapAttribute(key, uiType, groups):
    return [a[key] for g in groups for a in g['atomics'] if a['uiType'] == uiType]

def median(xs):
    xs = sorted(xs)
    half = len(xs) / 2
    if(len(xs) % 2 == 0):
        return average([xs[int(half-1)], xs[int(half)]])
    return xs[int(half)]

def average(xs):
    return sum(xs) / len(xs)

def samplePoints(ps):
    minX, minY, maxX, maxY = ps
    return (random.sample(range(minX, maxX), 1)[0], random.sample(range(minY, maxY), 1)[0])

def makeDefaultRect(uiType, groups, points):
#    x, y = samplePoints(points)
    return {
        'x': 0,
        'y': 0,
        'width': average(flatMapAttribute('width', uiType, groups)),
        'height': average(flatMapAttribute('height', uiType, groups))
    }

def initialRects(uiTypes, groups, space):
    return [makeDefaultRect(t, groups, space) for t in uiTypes]
def distanceOfNearestAtomic(cfg, candidate):
    toBeChecked = [c for c in cfg if c['uiType'] == candidate['uiType']]

def distance(rect, atomic):
    wd = abs(rect['width'] - atomic['width'])
    hd = abs(rect['height'] - atomic['height'])
    xd = abs(rect['x'] - atomic['x'])
    yd = abs(rect['y'] - atomic['y'])
    return wd + hd + xd + yd

def groupDistance(cfg, group):
    distances = []
    for atomic in group:
        bestMatchByDistance = sorted([{'rect': r, 'atomic': atomic, 'distance': distance(r, atomic)} for r in cfg], key=lambda x: x['distance'])[0]
        distances.append(bestMatchByDistance)

    return {'group': group, 'distances': distances, 'score': average([d['distance'] for d in distances])}

def score(config, groups):
    return sorted([groupDistance(config, g) for g in groups], key=lambda x: x['score'])


def getCursor(state):
    return state['guesses'][state['cursor']]

def move(state, args):
    direction, points = args
    minX, minY, maxX, maxY = points
    current_rect = getCursor(state)
    n = 3
    if(direction == 'down'):
        current_rect['y'] = current_rect['y'] - n# if current_rect['y'] - n >= minY else current_rect['y']
    if(direction == 'left'):
        current_rect['x'] = current_rect['x'] - n# if current_rect['x'] - n >= minX else current_rect['x']
    if(direction == 'up'):
        current_rect['y'] = current_rect['y'] + n #if (current_rect['y'] + n + current_rect['height']) <= maxY else current_rect['y']
    if(direction == 'right'):
        current_rect['x'] = current_rect['x'] + n #if current_rect['x'] + n + current_rect['width'] <= maxX else current_rect['x']
    return state

def changeCursor(state, _):
    state['cursor'] = (state['cursor'] + 1) % len(state['guesses'])
    return state

def resize(state, dimensions):
    current_rect = getCursor(state)
    current_rect['width'] = dimensions[0]
    current_rect['height'] = dimensions[1]
    return state

actions = {
    'move': move,
    'resize': resize,
    'changeCursor':changeCursor
}

def act(game_state, payload):
    action, arg = payload
    return actions[action](game_state, arg)

def rectsFromGroup(group):
    return [boundingRect(a) for a in group['atomics']]

class Game:
    def __init__(self, data):
        self.data = data
        self.state = self.initial_state()
        
    def initial_state(self):
        self.tries = 400
        uiTypes = self.data['uiTypes'][0] #[[]]
        groups = self.data['groups']
        group = groups[2] # one for now
        rects = rectsFromGroup(group)
        canvas, points = getCanvas(rects)
        self.canvas = canvas
        self.guesses = [makeDefaultRect(t, groups, points) for t in uiTypes]
        self.points = points
        return {'cursor': 0, 'rects': rects, 'guesses': self.guesses}

    def step(self, payload):
        payload = (payload[0], (payload[1], self.points))
        self.tries = self.tries - 1
        done = self.tries < 0 or self.score() > 975
        newState = act(self.state, payload)
        return (newState, self.score(), done, {})

    def reset(self):
        self.state = self.initial_state()
        return self.state
    
    def render(self):
        return render(self.state['guesses'], self.canvas)
    
    def score(self):
        s = score(self.state['guesses'], [self.state['rects']])[0]['score'] #returns all groups sorted
        return 1000 - s # cheating
g = Game(data)
g.render()
g.score()
# for r in range(21):
#     g.step(('move', 'right'))

g = Game(data)
g.render()
g.score()

# winning game

for r in range(12):
    g.step(('move', 'up'))
    
for r in range(105):
    g.step(('move', 'right'))
    
g.step(('changeCursor', None))

for r in range(5):
    g.step(('move', 'up'))
    
for r in range(105):
    g.step(('move', 'right'))

g.step(('changeCursor', None))

for r in range(5):
    g.step(('move', 'up'))
    
for r in range(10):
    g.step(('move', 'right'))
    
# skip resize for the big one
    
g.render()
g.score()
import random

def coords(points):
    minX, minY, maxX, maxY = points
    xRange = range(minX, maxX)
    yRange = range(minY, maxY)
    return [xRange, yRange, xRange, yRange] # x,y,w,h

def sampleRect(rect):
    rs = []
    for rng in rect:
        print('range', rng)
        rs.append(random.sample(rng, 1)[0])
        
    return rs #[random.sample(rng, 1) for rng in rect] 

class ObsSpace:
    def __init__(self, guesses, points):
        self.points = points
        self.cursors = range(len(guesses)) #prob 1 hot?
        self.guesses = [coords(points) for _ in guesses]
        self.n = 1 + len(guesses) # [cursors, rects]
        
    def sample(self):
        return [sampleRect(s) for s in self.guesses]

class ActSpace:
    def __init__(self, guesses, points):
        self.actions = [0,1,2,3,4] # left right up down, changeCursor
        self.n = len(self.actions)
        
    def sample(self):
        return random.sample(self.actions, 1)[0]
from sklearn.preprocessing import MinMaxScaler

def flatten(xs):
    return [y for x in xs for y in x]

def obsRectToRect(obs_rect):
    x,y,width,height = obs_rect
    return {'x': x, 'y': y, 'width': width, 'height': height}

def rectToObs(rect):
    return [rect['x'], rect['y'], rect['width'], rect['height']]

# def obsToState(obs):
#     return {'cursor': obs[0][0], 'rects': [obsRectToRect(r) for r in obs[1:]]}

def stateToObs(state):
    return [state['cursor']] + flatten([rectToObs(r) for r in state['guesses']])


def numToDirection(x):
    return ['down', 'left', 'up', 'right'][x]

def actToStep(num):
    if num == 4:
        return ('changeCursor', None)
    return ('move', numToDirection(num))

class Model:
    def __init__(self, game):
        self.game = game
        self.scaler = MinMaxScaler()
        minX, minY, maxX, maxY = game.points
        self.scaler.fit([[minX, minY, maxX, maxY]])
        self.observation_space = ObsSpace(game.guesses, game.points)
        self.action_space = ActSpace(game.guesses, game.points)
        
    def state(self, state):
        return stateToObs(state)
#        return np.reshape(s, 4, -1)
        
    def step(self, payload):
        newState, score, done, _ = self.game.step(actToStep(payload))
        return (self.state(newState), score, done, {})

    def reset(self):
        return self.state(self.game.reset())
    
    def render(self, mode="human"):
        return self.game.render()
    
    def score(self):
        return self.game.score()
        

# s = obsToState(g.observation_space.sample())
# rects = s['rects']
# cv = getCanvas(rects)[0]
# g.action_space.sample()

# render([], getCanvas()[0])
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

import random
import numpy as np

env = Model(Game(data))
random.seed(123)
np.random.seed(123)

nb_steps = 80000
nb_actions = 5
obs_dim = 17

model = Sequential()
model.add(Flatten(input_shape=(1,17)))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=80000, window_length=1)
policy = EpsGreedyQPolicy(0.05)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=2000, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mse'])

dqn.fit(env, nb_steps=nb_steps, visualize=False, verbose=1)
def test(m):
    g = Model(Game(data))
    for r in range(400):
        print('current score', g.score())
        state = np.reshape(g.state(g.game.state), [1,1,17])
        pred = m.predict(state)
        action = actToStep(np.argmax(pred[0])) # [40, 20, 49, 10]
        print('predict', action)
        (_, _, done, _) = g.game.step(action)
        if done:
            return "WIN!"
        

    g.render()
    
test(dqn.model)
