from learntools.core import binder; binder.bind(globals())
from learntools.python.ex3 import *
print('Setup complete.')
# Your code goes here. Define a function called 'sign'
def sign(num):
    if num < 0:
        return -1
    elif num == 0:
        return 0
    else:
        return 1

# Check your answer
q1.check()
#q1.solution()
def to_smash(total_candies):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between 3 friends.
    
    >>> to_smash(91)
    1
    """
    print("Splitting", total_candies, "candies" if total_candies > 1 else "candy")
    return total_candies % 3

to_smash(91)
to_smash(1)
def to_smash(total_candies):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between 3 friends.
    
    >>> to_smash(91)
    1
    """
    print("Splitting", total_candies, "candies" if total_candies > 1 else "candy")
    return total_candies % 3

to_smash(91)
to_smash(1)
# Check your answer (Run this code cell to receive credit!)
q2.solution()
def prepared_for_weather(have_umbrella, rain_level, have_hood, is_workday):
    # Don't change this code. Our goal is just to find the bug, not fix it!
    return have_umbrella or rain_level < 5 and have_hood or not rain_level > 0 and is_workday

# Change the values of these inputs so they represent a case where prepared_for_weather
# returns the wrong answer.
have_umbrella = False
rain_level = 0.0
have_hood = False
is_workday = False

# Check what the function returns given the current values of the variables above
actual = prepared_for_weather(have_umbrella, rain_level, have_hood, is_workday)
print(actual)

# Check your answer
q3.check()
#q3.hint()
#q3.solution()
def is_negative(number):
    if number < 0:
        return True
    else:
        return False

def concise_is_negative(number):
    return True if number < 0 else False

# Check your answer
q4.check()
#q4.hint()
#q4.solution()
def onionless(ketchup, mustard, onion):
    """Return whether the customer doesn't want onions.
    """
    return not onion
def wants_all_toppings(ketchup, mustard, onion):
    """Return whether the customer wants "the works" (all 3 toppings)
    """
    return ketchup and mustard and onion

# Check your answer
q5.a.check()
#q5.a.hint()
#q5.a.solution()
def wants_plain_hotdog(ketchup, mustard, onion):
    """Return whether the customer wants a plain hot dog with no toppings.
    """
    return not (ketchup or mustard or onion)

# Check your answer
q5.b.check()
#q5.b.hint()
#q5.b.solution()
def exactly_one_sauce(ketchup, mustard, onion):
    """Return whether the customer wants either ketchup or mustard, but not both.
    (You may be familiar with this operation under the name "exclusive or")
    """
    return (int(ketchup) + int(mustard)) == 1

# Check your answer
q5.c.check()
#q5.c.hint()
#q5.c.solution()
def exactly_one_topping(ketchup, mustard, onion):
    """Return whether the customer wants exactly one of the three available toppings
    on their hot dog.
    """
    return (int(ketchup) + int(mustard) + int(onion)) == 1

# Check your answer
q6.check()
#q6.hint()
#q6.solution()
# см. ниже
import sys, time
import numpy as np

class SeventhAssignmentHandler:
    def __init__(self, filename = 'buffer.txt'):
        print('We are starting now.')
        self.filename = filename
        self.inputs = []
        self.outputs = []
        self.defaultStdOut = sys.stdout
        
    def redirectToFile(self):
        sys.stdout = open(self.filename, 'a')
        
    def redirectBack(self):
        f = sys.stdout
        sys.stdout = self.defaultStdOut
        f.close()
        
    def whoWon(self):
        f = open(self.filename, 'r')
        line = f.readlines()[-1]
        return True if 'Player wins' in line else False
    
    # первая функция - для сбора данных
    # чтобы использовать, необходимо глобально переопределить "should_hit()":
    # should_hit = <object_name>.should_hit
    def should_hit(self, dt, pt, pla, pha):
        self.currDT = float(dt)
        self.currPT = float(pt)
        self.currPLA = float(pla)
        self.currPHA = float(pha)
        self.currSH = float(1)
        self.updateData()
        return bool(self.currSH)
    
    def updateData(self):
        self.currGame.append([[self.currDT, self.currPT, self.currPLA, self.currPHA], [self.currSH]])
    
    def uploadData(self):
        for decision in self.currGame:
            self.inputs.append(decision[0])
            self.outputs.append(decision[1])
    
    def collectDataSet(self, iters = 5000):
        for iter in range(iters):
            self.currGame = []
            self.redirectToFile()
            q7.simulate_one_game()
            self.redirectBack()
            whoWon = self.whoWon()
            if not whoWon and len(self.currGame):
                # print('WhoWon result: from %d' % self.currGame[-1][1][0], end = ' ')
                self.currGame[-1][1][0] = 1 - self.currGame[-1][1][0]
                # print('to %d' % self.currGame[-1][1][0])
            self.uploadData()
            
    def prepareData(self, rearrange = True, viewDataSet = False):
        cInputs = []
        cOutputs = []
        for i in range(len(self.inputs)):
            cell = self.inputs[i]
            fl = True
            for j in range(len(cInputs)):
                if cell[0] == cInputs[j][0] and cell[1] == cInputs[j][1] and cell[2] == cInputs[j][2] and cell[3] == cInputs[j][3]:
                    fl = False
            if not fl:
                continue
            cout = self.outputs[i]
            cntr = 1
            for j in range(len(self.inputs)):
                if cell[0] == self.inputs[j][0] and cell[1] == self.inputs[j][1] and cell[2] == self.inputs[j][2] and cell[3] == self.inputs[j][3] and i != j:
                    cntr += 1
                    cout[0] += self.outputs[j][0]
            cout[0] /= cntr
            cInputs.append(cell)
            cOutputs.append(cout)
        if viewDataSet:
            for k in range(len(cInputs)):
                print('Очки: %2d против %2d. Итог: %.3f' % (cInputs[k][0], cInputs[k][1], cOutputs[k][0]))
        self.inputs = np.array((self.rearrangeInputs(cInputs) if rearrange else cInputs))
        self.outputs = cOutputs
    
    def rearrange(array, old_min, old_max, new_min = 0, new_max = 1):
        for index in range(len(array)): 
            array[index] = float(((array[index] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min)
        return array
        
    def rearrangeInputs(self, inputs):
        rearranged_inputs = []
        inputs = np.array(inputs)
        ra = [[0, 21], [0, 21], [0, 5], [0, 5]]
        for column in range(len(inputs[0])):
            old_min, old_max = ra[column][0], ra[column][1]
            rearranged_inputs.append(SeventhAssignmentHandler.rearrange(inputs.T[column], old_min, old_max))
        return np.array(rearranged_inputs).T
        
    def train(self, epochs = 10000, logEvery = -1, printDataSet = False):
        if printDataSet:
            print(self.inputs)
            print(self.outputs)
        np.random.seed(int(time.time()))
        self.layer1 = 2 * np.random.random((len(self.inputs[0]), 10)) - 1
        self.layer2 = 2 * np.random.random((10, len(self.outputs[0]))) - 1
        for epoch in range(epochs):
            l1 = 1 / (1 + np.exp(-(np.dot(self.inputs, self.layer1))))
            l2 = 1 / (1 + np.exp(-(np.dot(l1, self.layer2))))
            l2_delta = (self.outputs - l2) * (l2 * (1 - l2))
            l1_delta = l2_delta.dot(self.layer2.T) * (l1 * (1 - l1))
            self.layer2 += l1.T.dot(l2_delta)
            self.layer1 += self.inputs.T.dot(l1_delta)
            if (epoch + 1) % logEvery == 0 and logEvery != -1:
                print(f'Эпоха {epoch}')
                print('l1: ')
                print(l1)
                print()
                print('l2: ')
                print(l2)
                print()
                print('l1_delta: ')
                print(l1_delta)
                print()
                print('l2_delta: ')
                print(l2_delta)
                print()
                print('layer1: ')
                print(self.layer1)
                print()
                print('layer2: ')
                print(self.layer2)
    
    # вторая функция - для ответов
    # чтобы использовать, необходимо глобально переопределить "should_hit()":
    # should_hit = <object_name>.should_hit_nn
    def should_hit_nn(self, dt, pt, pla, pha):
        inputs = self.rearrangeInputs([[float(dt), float(pt), float(pla), float(pha)]])
        result = (1 / (1 + np.exp(-(np.dot(1 / (1 + np.exp(-(np.dot(inputs, self.layer1)))), self.layer2)))))
        # if out:
        #     print(result)
        return bool(result >= 0.5)
    
    # третья функция - лучший результат без нейросетей
    # чтобы использовать, необходимо глобально переопределить "should_hit()":
    # should_hit = <object_name>.should_hit_best_without_nn
    def should_hit_best_without_nn(self, dt, pt, pla, pha):
        if pt > 13:
            return False
        else:
            return True
A = SeventhAssignmentHandler()
should_hit = A.should_hit
A.collectDataSet(100)
A.prepareData()
print(A.currGame)
print(A.inputs)
print(A.outputs)
A.train()
A.should_hit_nn(17, 21, 0, 0)
# def should_hit(dealer_total, player_total, player_low_aces, player_high_aces):
#     """Return True if the player should hit (request another card) given the current game
#     state, or False if the player should stay.
#     When calculating a hand's total value, we count aces as "high" (with value 11) if doing so
#     doesn't bring the total above 21, otherwise we count them as low (with value 1). 
#     For example, if the player's hand is {A, A, A, 7}, we will count it as 11 + 1 + 1 + 7,
#     and therefore set player_total=20, player_low_aces=2, player_high_aces=1.
#     """
#     if player_total > 12 and dealer_total < 14:
#         return False
#     elif player_total == 12 and dealer_total in (15, 16):
#         return False
#     elif player_total in (19, 20, 21):
#         return False
#     else:
#         return True


should_hit = A.should_hit_nn

q7.simulate_one_game()
should_hit = A.should_hit_nn
q7.simulate(n_games = 50000)
should_hit = A.should_hit_best_without_nn
q7.simulate(n_games=50000)