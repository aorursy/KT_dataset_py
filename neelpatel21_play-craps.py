import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import random as rn
class Craps:
    PASS_LINE = [7,11]
    PASS_LINE_A = [2,3,12]
    
    def __init__(self):
        pass
    
    def rollDice():
        d1 = rn.randint(1,6)
        d2 = rn.randint(1,6)
        return d1+d2
    
    def bet(isPassLine=True, amount=10):
        rolls = []
        r = Craps.rollDice()
        rolls.append(r)
        if r in Craps.PASS_LINE:
            if isPassLine:
                return amount*2, rolls
            else:
                return 0, rolls
        elif r in Craps.PASS_LINE_A:
            if not isPassLine:
                return amount*2, rolls
            else:
                return 0, rolls
        else:
            pointNumber = r
            r = Craps.rollDice()
            rolls.append(r)
            while r!=pointNumber and r!=7:
                r = Craps.rollDice()
                rolls.append(r)
            
            if r==7:
                if not isPassLine:
                    return amount*2, rolls
                else:
                    return 0, rolls
            else:
                if isPassLine:
                    return amount*2, rolls
                else:
                    return 0, rolls

def testRollDice():
    for i in range(10000):
        r = Craps.rollDice()
        if type(r) is not int:
            raise Exception(f'Invalid type of roll dice {r}.')
        if r<2 or r>12:
            raise Exception(f'Invalid roll dice number {r}.')

def testBet():
    for i in range(10000):
        r, dh = Craps.bet()
        if type(r) is not int or type(dh) is not list:
            raise Exception(f'Invalid return type of bet {r}, {dh}.')
        if dh[0] in Craps.PASS_LINE+Craps.PASS_LINE_A and len(dh)>1:
            raise Exception(f'Invalid game status {r}, {dh}.') 
            
        if dh[0] in Craps.PASS_LINE and r==0:
            raise Exception(f'Invalid game status {r}, {dh}.')    
        if dh[0] in Craps.PASS_LINE_A and r!=0:
            raise Exception(f'Invalid game status {r}, {dh}.')    
        if dh[0] not in Craps.PASS_LINE + Craps.PASS_LINE_A and dh[-1] != 7 and r==0:
            raise Exception(f'Invalid game status {r}, {dh}.')    
        if dh[0] not in Craps.PASS_LINE + Craps.PASS_LINE_A and dh[-1] == 7 and r!=0:
            raise Exception(f'Invalid game status {r}, {dh}.')    
        
        
testRollDice()
testBet()
n = 100
l = []
for i in range(n):
    l.append(Craps.bet())
    

print(f'total wins {len([x[0] for x in l if x[0]>0])} out of {n}')
l