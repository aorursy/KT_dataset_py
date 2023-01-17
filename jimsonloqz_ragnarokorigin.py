import numpy as np
import pandas as pd
import seaborn as sns
import itertools as it

from scipy import stats
from collections import Counter
from matplotlib import pyplot as plt

sns.set_style('darkgrid')
def ComputeAvgExp(*args, names=None, mobExp=1, showCombinations=False): 
    """wd
    Computes the average exp gain per second individually when party grinding, 
    taking into account the exp bonus that depends on party size.
    
    Parameters
    ----------
    args : float
        The average mob kill per second. Each arg corresponds to one party member.
    names: list of str, optional
        The names of each member in the party in the same order as args. 
    mobExp : float, optional 
        The exp gained corresponding to one mob kill.q
        
    Returns
    -------
    float :
        The average exp per second gained by each member of the party.
    """
    n = len(args)
    
    exp = mobExp * sum(args) / n 
    
    if n > 1:
        exp *= 1 + n/10
    
    if names is not None:
        maxLen = max(map(len,names))
        print('Exp/s gain if not in a party:')
        for x, name in zip(args, names):
            print(f'{name:<{maxLen+1}}: {x*mobExp:.2f}')
            
    if showCombinations:
        print('----------------------------')
        print('Showing all player combinations:') 
        nameExp = list(zip(names, args))
        for i in range(2, n):
            for comb in it.combinations(nameExp, i):
                N, E = zip(*comb)
                print(f"{' + '.join(N):<{maxLen*i+1}}: {ComputeAvgExp(*E,mobExp=mobExp):.2f}")
    
    return exp
# ComputeAvgExp(13/60, 15/60, 18/60, 21/60, 10/60, 
#               names=['jopepay','AntonioGlide','Balmung','DarkNorth','Stikket'], 
#               mobExp=800, showCombinations=True)
# Card drop probability for a single mob kill.
p = 0.0001

fails = np.random.negative_binomial(1, p)

print(f'Total kills on first card drop:\n{fails+1}')
# Card drop probability for a single mob kill.
p = 0.00001
# No. of card drops.
cards = 0
# No. of kills.
kills = 35000

proba = stats.binom.pmf(k=cards, n=kills, p=p)

print(f'Probability of {cards:,} card drop(s) in {kills:,} kills:\n{proba*100:.4f}%')
class RefineSystem:
    def __init__(self):
        # Index corresponds to the probability of success when hammering the current refine level of the gear.
        self.Q = [1,1,1,0.55,0.5,0.45,0.4,0.3,0.25,0.1]
        # Index corresponds to the number of ores needed to hammer the current refine level of the gear.
        self.R = [1,2,2,4,4,6,12,20,30,40]
        # Index corresponds to the amount of zenny needed to hammer the current refine level of the gear.
        self.Z = [5000,10000,15000,20000,32000,40000,50000,60000,80000,120000]
        self.damagedProba = 0.05
        self.refineDownProba = 0.1
        
        # Counts for each ore consumed, and number of repairs done.
        self.zenny = 0
        self.ores = 0
        self.highOres = 0
        self.repairs = 0
        
        # Current refine level.
        self.level = 0
        
    def _GetHammerResult(self, proba):
        result = np.random.choice(['success','fail'], p=[proba,1-proba])
        
        if result == 'fail':
            result = np.random.choice(['unchanged','down','damaged'], 
                                      p=[1-self.damagedProba-self.refineDownProba,
                                         self.refineDownProba,
                                         self.damagedProba])
        
        return result

    def _Hammer(self):
        if self.level > 14:
            raise ValueError('Maximum refine level reached!')
        
        if self.level < 10:
            self.ores += self.R[self.level]
            self.zenny += self.Z[self.level]
            result = self._GetHammerResult(self.Q[self.level])
            if result == 'success':
                self.level += 1
            elif result == 'down':
                self.level -= 1
            elif result == 'damaged':
                self.repairs += 1
        else:
            self.highOres += self.R[self.level]
            self.zenny += self.Z[self.level]
            result = self._GetHammerResult(self.Q[self.level])
    
    def SimulateRefine(self, target, start=0):
        self.level = start
        while self.level < target:
            self._Hammer()
        
        print(
            f'+{start} to +{target} refine level reached. Total materials consumed:\n'
            f'Ores = {self.ores}\n'
            f'Zenny = {self.zenny:,}\n'
            f'Repairs = {self.repairs}'
        )
Hollgrehenn = RefineSystem()
Hollgrehenn.SimulateRefine(start=3, target=10)
baseCritDmg = 300000
doubleAtkChance = 0.25
doubleAtkMultiplier = 2
giantGrowthChance = 0.3
giantGrowthMultiplier = 2.5
stormBlastChance = 0.15
stormBlastMultiplier = 25/2

averageDmgPerHit = (baseCritDmg 
                    + (baseCritDmg * giantGrowthMultiplier * giantGrowthChance) 
                    + (baseCritDmg * stormBlastMultiplier * stormBlastChance))
DPS = averageDmgPerHit * 7

print(f'Average DPS = {DPS:,}')
# For physical melee damager.
# Still doesn't factor in Refine Atk and Ignore Def.
def ComputeDPS(atk, aspd=1, crit=0, critDmg=1.5, physDmg=1, shortRangeDmg=1, skillMod=1, sizeMod=1, raceMod=1, elementMod=1, converterMod=1):
    pass