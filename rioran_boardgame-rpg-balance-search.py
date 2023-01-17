# Libraries

import random as rnd
# Classes for random dices

# Single dice class

class Dice: # 6 sided dice as default

    def __init__(self, sides = 6):

        self.sides = sides

    def roll(self):

        return rnd.randint(1, self.sides) # any value from 1 to 6

# 2 dices to roll at the same time

class Dices(Dice): # Inherit single dice properties

    def __init__(self, dice=None):

        if dice is None: dice = Dice()

        self.dice = dice

    def roll(self):

        return self.dice.roll() + self.dice.roll()

# public variable for dices to roll

dices = Dices()
# Units classes

# Main unit class

class Unit:

    def __init__(self, is_alive=None, hp=None, dmg=None, initiative=None, evasion=None, deathblow=None, accuracy=None):

        # Default values for a random unit

        if is_alive is None:    is_alive = True  # False for dead units

        if hp is None:          hp = 40          # Health points to stay alive

        if dmg is None:         dmg = 10         # Damage for enemy HP to be reduced

        if initiative is None:  initiative = 0   # Bonus to a dice roll for first turn

        if evasion is None:     evasion = 2      # 13 - evasion = point, starting from which enemy strike was eluded

        if deathblow is None:   deathblow = 1    # 13 - death blow = point of single shot kill

        if accuracy is None:    accuracy = 8     # 13 - accuracy = point when strike can hit the target

        # Assignment

        self.is_alive = is_alive

        self.hp = hp

        self.dmg = dmg

        self.initiative = initiative

        self.evasion = evasion

        self.deathblow = deathblow

        self.accuracy = accuracy

    def hit(self, enemy, text_mode=False):

        hit_rate = dices.roll()

        if hit_rate >= 13 - self.accuracy: # Strike reached enemy

            evasion_rate = dices.roll()

            if evasion_rate < 13 - enemy.evasion: # Enemy didn't evade

                if hit_rate >= 13 - self.deathblow: # If one shot kill

                    enemy.is_alive, enemy.hp = 0, 0

                    if text_mode: print(self.name + ' rolled ' + str(hit_rate) + ' (' + str(13-self.deathblow) + ' to kill), ' + 

                                        enemy.name + ' rolled ' + str(evasion_rate) + ' (' + str(13-enemy.evasion) + ' to evade) - death with a single blow!')

                else: # If it's a normal strike

                    enemy.hp = max(enemy.hp - self.dmg, 0)

                    if enemy.hp == 0:

                        enemy.is_alive = 0

                        if text_mode: print(self.name + ' rolled ' + str(hit_rate) + ' (' + str(13-self.deathblow) + ' to kill), ' + 

                                        enemy.name + ' rolled ' + str(evasion_rate) + ' (' + str(13-enemy.evasion) + ' to evade) - a finishing blow landed!')

                    else:

                        if text_mode: print(self.name + ' rolled ' + str(hit_rate) + ' (' + str(13-self.deathblow) + ' to kill), ' + 

                                        enemy.name + ' rolled ' + str(evasion_rate) + ' (' + str(13-enemy.evasion) + ' to evade) - and took ' + str(self.dmg) + ' damage! ' +

                                           str(enemy.hp) + ' HP remains.')

            else:

                if text_mode: print(self.name + ' rolled ' + str(hit_rate) + ' (' + str(13 - self.deathblow) + ' to kill), ' + 

                                    enemy.name + ' rolled ' + str(evasion_rate) + ' (' + str(13-enemy.evasion) + ' to evade) - attack evaded!')

        else:

            if text_mode: print(self.name + ' rolled ' + str(hit_rate) + ' (' + str(13-self.accuracy) + ' to hit) and missed!')
# Class for unit specific form

class LongSword(Unit):

    def __init__(self, name=None, is_alive=None, hp=None, dmg=None, initiative=None, evasion=None, deathblow=None, accuracy=None):

        super(LongSword, self).__init__(is_alive, hp, evasion)

        self.name = name

        self.dmg = 30 # 2H sword is devastating, 2 hits to kill 40 HP

        self.initiative = 1 # 2H sword has higher radius of reach = advantage to hit first

        self.deathblow = 3 # 2H sword will one shot more frequently

        self.accuracy = 6 # Lesser accuracy due to lower weapon control

# Class for unit specific form

class SwordShield(Unit):

    def __init__(self, name=None, is_alive=None, hp=None, dmg=None, initiative=None, evasion=None, deathblow=None, accuracy=None):

        super(SwordShield, self).__init__(is_alive, hp, initiative, deathblow, accuracy)

        self.name = name

        self.evasion = 4 # Easier to evade using shield

        self.accuracy = 9 # Easier to place a hit while you are defended

        self.dmg = 15 # Damage is lower than for 2H weapon
def duel(hero1, hero2, text_mode=False):

    if text_mode: print('======= ' + hero1.name + ' VS ' + hero2.name + ' =======')

    while hero1.is_alive * hero2.is_alive > 0: # Fight unless someone is dead

        hero1_move_rate = dices.roll() + hero1.initiative

        hero2_move_rate = dices.roll() + hero2.initiative

        if text_mode: print('Turn: ' + hero1.name + ' rolled ' + str(hero1_move_rate) + ' (+' + str(hero1.initiative) + ' bonus), ' + 

                            hero2.name + ' rolled ' + str(hero2_move_rate) + ' (+' + str(hero2.initiative) + ' bonus)')

        while hero1_move_rate == hero2_move_rate: # Roll dices for first turn unless values are not equal

            hero1_move_rate = dices.roll() + hero1.initiative

            hero2_move_rate = dices.roll() + hero2.initiative

            if text_mode: print('Turn: ' + hero1.name + ' rolled ' + str(hero1_move_rate) + ' (+' + str(hero1.initiative) + ' bonus), ' + 

                            hero2.name + ' rolled ' + str(hero2_move_rate) + ' (+' + str(hero2.initiative) + ' bonus)')

        if hero1_move_rate > hero2_move_rate: # Better initiative = more chances to hit first

            hero1.hit(hero2, text_mode)

            if hero2.is_alive: hero2.hit(hero1, text_mode) # Corpses doesn't fight

        else:

            hero2.hit(hero1, text_mode)

            if hero1.is_alive: hero1.hit(hero2, text_mode) # Corpses doesn't fight

    if text_mode:

        if hero1.is_alive:

            victor = hero1.name

        else:

            victor = hero2.name

        print('All hail ' + victor + ' for a glorious victory!')

        print('======= ' + victor + ' WON!!! =======')

    return hero1.is_alive + 0, hero2.is_alive + 0
results = [0, 0]

for i in range(1, 100001):

    hero1 = LongSword(name='Rioran')

    hero2 = SwordShield(name='Knight')

    duel(hero1, hero2)

    results[0] += hero1.is_alive

    results[1] += hero2.is_alive

print(hero1.name + ' won ' + str(results[0]) + ' times')

print(hero2.name + ' won ' + str(results[1]) + ' times')
hero1 = LongSword(name='Rioran')

hero2 = SwordShield(name='Knight')

duel_result = duel(hero1, hero2, text_mode=True)