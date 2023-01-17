import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import sys
import re
from datetime import datetime
import math

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
MAX_WEIGHT = 50.0

toys = {
    "horse":  { "sample": lambda: max(0, np.random.normal(5,2,1)[0]), "sample_type": "normal(5,2)" },
    "ball":   { "sample": lambda: max(0, 1 + np.random.normal(1,0.3,1)[0]), "sample_type": "normal(1,0.3)" },
    "bike":   { "sample": lambda: max(0, np.random.normal(20,10,1)[0]), "sample_type": "normal(20,10)" },
    "train":  { "sample": lambda: max(0, np.random.normal(10,5,1)[0]), "sample_type": "normal(10,5)" },
    "coal":   { "sample": lambda: 47 * np.random.beta(0.5,0.5,1)[0], "sample_type": "47*beta(0.5,0.5)" },
    "book":   { "sample": lambda: np.random.chisquare(2,1)[0], "sample_type": "chi(2)" },
    "doll":   { "sample": lambda: np.random.gamma(5,1,1)[0], "sample_type": "gamma(5,1)" },
    "block":  { "sample": lambda: np.random.triangular(5,10,20,1)[0], "sample_type": "triagl(5,10,20)" },
    "gloves": { "sample": lambda: 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0], "sample_type": "0.3:3+rand(1), 0.7:rand(1)" },
}
toy_names = list(toys)

gifts_df = pd.read_csv("../input/gifts.csv", sep=",")
gifts = gifts_df["GiftId"].values
print("{} gifts".format(len(gifts)))

for t in toys:
    # get ranges
    samples = [toys[t]["sample"]() for _ in range(1000)]
    toys[t]["max"] = max(samples)
    toys[t]["min"] = min(samples)
    
    # get gift counts
    ids = [g for g in gifts if t in g.split("_")[0]]
    toys[t]["ids"] = ids
    toys[t]["count"] = len(ids)
    
    # print toy type stats
    print("{:4}\tdist: {:26}\trange:{:5.2f} - {:5.2f}\tcount:{:6,}".format(t, toys[t]["sample_type"], toys[t]["min"], toys[t]["max"], toys[t]["count"]))
# Visualize distributions

plt.figure(figsize=(10,10))
for i,t in enumerate(toys):
    plt.subplot(3,3,i+1)
    samples = [toys[t]["sample"]() for _ in range(10000)]
    plt.hist(samples, bins=np.linspace(0,47,80), normed=True)
    plt.title(t)
# Gets a single bag worth of toys that passes the accept threshold
def get_one_bag_of_toys(toys_dups, max_toys_to_consider=12, num_simulations=40, min_toys_per_bag=3, accept_threshold=0.80, max_weight=50.0):

    items_list = random.sample(toys_dups, min(max_toys_to_consider, len(toys_dups)))
    res = []
    res_weights = []
    res_items = []

    for num_toys in range(1,max_toys_to_consider):
        items = items_list[:num_toys]

        # run simulation
        weights = [sum([toys[t]["sample"]() for t in items]) for _ in range(num_simulations)]
        percent_accepted = float(len([w for w in weights if w <= max_weight]))/float(num_simulations)

        if percent_accepted < accept_threshold:
            break
            
        res.append(percent_accepted)
        res_weights.append(np.mean(weights))
        res_items = items
           
    if min_toys_per_bag > num_toys:
        return [], (res[-1] if len(res) > 0 else 0.0), (res_weights[-1] if len(res_weights) > 0 else 0.0)
    else:
        return res_items, res[-1], res_weights[-1]
   
num_options_to_consider = 5

toys_dups = [t for t in toys for _ in range(toys[t]["count"])]
random.shuffle(toys_dups)

scores = []
bags = []
weights = []
start = datetime.now()
    
for bag_i in range(1000):
    if bag_i > 0 and bag_i % 100 == 0:
        print("{}\t{}/1000\tw: {:.2f}\tw/bag: {:.1f}".format(str(datetime.now() - start), bag_i, sum(weights), sum(weights)/float(bag_i)))
        
    options = [get_one_bag_of_toys(toys_dups) for _ in range(num_options_to_consider)]
    options_sorted = sorted([(res, ar, res_w) for ar,res,res_w in options if len(ar) > 2 and res > 0.90], key=lambda x: -len(x[1]))

    if len(options_sorted) > 0:
        best = options_sorted[0]
        best_score, best_toys, best_avg_weight = best

        scores.append(best_score)
        bags.append(best_toys)
        weights.append(best_avg_weight)

        for toy in best_toys:
            toys_dups.remove(toy)
    else:
        pass
    
print("\ntotal weight: {:,.0f}, bags used: {}, toys left: {}\n".format(sum(weights), len([b for b in bags if len(b) > 0]), len(toys_dups)))
    
# Add any toys that were not added
if len(bags) < 1000:
    bags_left = [[] for x in range(1000 - len(bags))]
    print("adding {} toys to the remaining {} empty bags".format(len(toys_dups), len(bags_left)))
    
    bag_i = 0
    for t in toys_dups:
        if bag_i >= len(bags_left):
            bag_i = 0
        bags_left[bag_i].append(t)
        if len(bags_left[bag_i]) > 2:
            bag_i += 1
    bags = bags + bags_left
GOOD = 0
ACCEPT_BAD = 1
LEN_NORM = 2
REJECT = 3


class BagSaModel:
    
    def __init__(self, S, temp=10.0, temp_step=0.00001, num_simulations=50, max_weight=50.0, max_items_per_bag=12):
        self.S = [s[:] for s in S]
        self.temp = temp
        self.temp_step = temp_step
        self.max_items_per_bag = max_items_per_bag
        
        self.num_simulations = num_simulations
        self.max_weight = max_weight
        self.scores = [self.get_score(s) for i,s in enumerate(self.S)]
        
        self.best_score = sum(self.scores)
        self.best_S = [s[:] for s in self.S]
        self.score_all_steps = []
        self.steps_type = []
        self.state_snapshots = []
        self.score_snapshots = []
        
        self.iteration = 0
        
    def get_samples(self, s, num_iters=100):
        weights = [sum([toys[t]["sample"]() for t in s]) for _ in range(num_iters)]
        return weights
        
    def get_score(self, s, include_weight=True): 
        if len(s) == 0:
            return 1.0
        
        weights = [sum([toys[t]["sample"]() for t in s]) for _ in range(self.num_simulations)]
        percent_accepted = float(len([w for w in weights if w <= self.max_weight]))/float(self.num_simulations)
        
        # This is the expected weight that will be banked based on simulation
        if include_weight:
            return np.mean([w if w <= 50.0 else 0.0 for w in weights])
        return percent_accepted
    
    # Performs a "swap" of a random item between two bags
    #   The "swap" is accepted if the score improves or otherwise it will
    #   accept worse scores with prob exp(delta/temp)
    def swap_sa_step(self):    
        self.temp = self.temp * self.temp_step
        
        s_i, s_j = [], []
        while len(s_i) <= 3 or len(s_j) >= self.max_items_per_bag:
            i, j = random.randint(0,999), random.randint(0,999)
            while i == j:
                j = random.randint(0,999)

            s_i, s_j = self.S[i][:], self.S[j][:]
            len_i, len_j = len(s_i), len(s_j)
            
        prev_score = sum(self.scores)
        score_i, score_j = self.get_score(s_i), self.get_score(s_j)
        
        # swap an item
        if len(s_i) > 0:
            x = random.sample(s_i, 1)[0]
            s_i.remove(x)
            s_j.append(x)
            
        # evaluate scores after
        score_i_after, score_j_after = self.get_score(s_i), self.get_score(s_j) 
        new_score = prev_score - score_i - score_j + score_i_after + score_j_after
        
        delta_score = 0.5*( (score_i_after + score_j_after) - (score_i + score_j) )
        self.score_all_steps.append(new_score)
        
        accept_good = delta_score > 0.0
        r = random.random()
        
        accept_bad = (math.exp(delta_score/self.temp) > r) if not accept_good and delta_score < -0.0001 and abs(delta_score/self.temp) < 10 else False
        accept_len_norm = (abs(delta_score) <= 0.001) and (abs(len_i-len_j) > abs(len(s_i)-len(s_j)))
        
        if accept_good or accept_bad or accept_len_norm:
            self.S[i], self.scores[i] = s_i, score_i_after
            self.S[j], self.scores[j] = s_j, score_j_after
            
            if new_score > self.best_score:
                self.best_score, self.best_S = new_score, [s[:] for s in self.S]
                
        if accept_good:
            self.steps_type.append(GOOD)
        elif accept_bad:
            self.steps_type.append(ACCEPT_BAD)
        elif accept_len_norm:
            self.steps_type.append(LEN_NORM)
        else:
            self.steps_type.append(REJECT)
            
        self.iteration += 1
        
        if self.iteration % 1000 == 0:
            self.state_snapshots.append([s[:] for s in self.S])
            self.score_snapshots.append(new_score)
            
INIT_TEMP = 4.0 # The bigger this number, the longer it will accept "random" swaps
TEMP_STEP = 0.99992 # This bigger this number, the slower the cooling schedule
NUM_SIMULATIONS_PER_STEP = 40 # The larger this number, the more acccurate (but each swap becomes slower)
MAX_WEIGHT = 47.0 # Use less than 50 to get a margin for error 
MAX_ITEMS_PER_BAG = 10000 # Upper bound for number of items per bag
NUM_ITERATIONS = 200000 # Number of iterations to run algorithm for

saModel = BagSaModel([bags[i][:] if i < len(bags) else [] for i in range(1000)], temp=INIT_TEMP, temp_step=TEMP_STEP, num_simulations=NUM_SIMULATIONS_PER_STEP, max_weight=MAX_WEIGHT, max_items_per_bag=MAX_ITEMS_PER_BAG)

start = datetime.now()
num_iters = NUM_ITERATIONS
for k in range(num_iters):
    if k > 0 and k % 10000 == 0:
        score = np.mean(saModel.score_all_steps[-100:])
        var = np.var([saModel.score_all_steps[-100:]])
        temp = saModel.temp
        accept_good = len([x for x in saModel.steps_type[-1000:] if x == GOOD])
        accept_bad = len([x for x in saModel.steps_type[-1000:] if x == ACCEPT_BAD])
        accept_len = len([x for x in saModel.steps_type[-1000:] if x == LEN_NORM])
        reject = len([x for x in saModel.steps_type[-1000:] if x == REJECT])
        print("{:7,}/{:7,} bags:{:5,}, score:{:6,.4f}, var:{:6.1f}, temp:{:.4f}, good/bad/len/rej:{:3}/{:3}/{:3}/{:3}".format(k,num_iters, len([b for b in saModel.S if len(b) >= 3]), score, var, temp, accept_good, accept_bad, accept_len, reject))
    
    saModel.swap_sa_step()
plt.plot(saModel.score_all_steps)
plt.title("score over iterations while training");
bags = [b[:] for b in saModel.best_S]
bag_weights = [saModel.get_samples(b, 1000) for b in bags]

plt.figure(figsize=(9,6))
plt.hist([w for ws in bag_weights for w in ws if w < 100], bins=np.linspace(0,100,100));
plt.title("bag samples weight distribution for all bags")

item_cnt = sum([len(b) for b in bags])
score = np.mean([np.mean([1*(w < 50.0) for w in ws]) for ws in bag_weights])
print("E[Pr accepting bag]: {:.2f}, cnt:{}/{}".format(score, item_cnt, len(gifts)));
print("plots of weight distribution for random bags")

num_samples = 14
plt.figure(figsize=(7,3*num_samples/2))
for i,k in enumerate(random.sample(range(1000), num_samples)):
    bag = saModel.best_S[k]
    name = " ".join(bag)
    weights = bag_weights[k]
    
    percent_accepted = float(len([w for w in weights if w <= 50.0]))/float(len(weights))
    
    plt.subplot(num_samples/2, 2,i+1)
    plt.hist(weights, bins=np.linspace(0,80,80), normed=True, histtype='bar')
    plt.xlim(0, 80)
    plt.title("Pr:{:4.2f} - {}".format(percent_accepted, name));
# Trim all bags that have less than THRESHOLD% chance of being accepted
THRESHOLD = 0.80

bag_probs = [float(len([w for w in weights if w <= 50.0]))/float(len(weights)) for weights in bag_weights]
new_probs = bag_probs[:]

bad_bag_idxs = [(i,p) for i,p in enumerate(bag_probs) if p < THRESHOLD]
print("{} bags below threshold".format(len(bad_bag_idxs)))

bag_drop_counts = []
for bag_i,bag_prob in bad_bag_idxs:
    bag = bags[bag_i][:]
    prob = bag_prob
    dropped = 0
    while len(bag) > 3 and prob < THRESHOLD:
        x = random.sample(bag, 1)[0]
        bag.remove(x)
        dropped += 1
        
        weights = saModel.get_samples(bag, 500)
        new_prob = float(len([w for w in weights if w <= 50.0]))/float(len(weights))
        
        if new_prob >= THRESHOLD:
            bag_drop_counts.append(dropped)
            
            bags[bag_i] = bag
            new_probs[bag_i] = new_prob
            break
            
plt.hist(bag_drop_counts);
plt.title("hist of # bags that dropped x items");
trimmed_bag_weights = [saModel.get_samples(b, 500) for b in bags]

bag_scores = [sum([w for w in weights if w <= 50.0])/len(weights) for weights in bag_weights]
trimmed_bag_scores = [sum([w for w in weights if w <= 50.0])/len(weights) for weights in trimmed_bag_weights]

print("Expected score: before: {:.2f}, after trimming: {:.2f}".format(sum(bag_scores), sum(trimmed_bag_scores)))
print("E[Pr accepting bag] before: {:.2f}, after trimming: {:.2f}".format(np.mean(bag_probs), np.mean(new_probs)))
print("num toys used: {}/{}".format(len([t for b in bags for t in b]), len(gifts)))
toy_ids = {t:sorted([x for x in toys[t]["ids"]], reverse=True) for t in toy_names}

bags_ids = []
for b in bags:
    bag_toy_ids = []
    for toy in b:
        if len(toy_ids[toy]) == 0:
            raise Exception("toy count error!")
            
        bag_toy_ids.append(toy_ids[toy].pop())
    bags_ids.append(bag_toy_ids)
            
submit_df = pd.DataFrame({"Gifts": [" ".join(b) for b in bags_ids]})
submit_df.to_csv("./SA_solution.csv", sep=",", index=False)