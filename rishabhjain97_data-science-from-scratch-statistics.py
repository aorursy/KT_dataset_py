import random
from typing import List
num_friends : List[int] = [random.randrange(100) for i in range(204)]

from collections import Counter
import matplotlib.pyplot as plt 
friends_count = Counter(num_friends)
#print(friends_count)
xs = range(101)
ys = [friends_count[i] for i in xs]
#print(ys)
plt.bar(xs,ys)
plt.axis([0, 101, 0, 7])
plt.title("Histogram of Friend Counts")
plt.xlabel("# of friends")
plt.ylabel("# of people")
plt.show()

num_points: int = len(num_friends)
largest_value = max(num_friends)
smallest_value = min(num_friends)
print(num_points, largest_value, smallest_value)
sorted_values = sorted(num_friends)
smallest_value = sorted_values[0]
second_largest_values = sorted_values[-2]
print(second_largest_values)
def mean(xs:List[float]) -> float:
    return sum(xs)/len(xs)
print(mean(num_friends))
def _median_odd(xs: List[float]) -> float:
    return sorted(xs)[len(xs)//2]
def _median_even(xs:List[float])-> float:
    hi_point = len(xs)//2
    low_point = hi_point - 1
    return (sorted(xs)[hi_point]+ sorted(xs)[low_point])/2

def median(xs: List[float]) -> float:
    return _median_even(xs) if len(xs)%2==0 else _median_odd(xs)

assert median([1, 10, 2, 9, 5]) == 5
assert median([1, 9, 2, 10]) == (2 + 9) / 2

print(median(num_friends))
def quantile(xs:List[float], p: float) -> float:
    p_index: int = int(p * len(xs))
    return sorted(xs)[p_index]
print(quantile(num_friends, 0.99))
def mode(xs: List[float]) -> float:
    counts = Counter(xs)
    #max_count = counts.most_common(3)
    max_count = max(counts.values())
    #print(sorted(counts.values()))
    #print(max_count)
    #assert max_count == max_count2
    return [i for i, count in counts.items() if count == max_count]
print(mode(num_friends))

def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)
print(data_range(num_friends))
    
def dot(x: List[float],y: List[float]) -> float:
    assert len(x)==len(y)
    return sum(x_i*y_i for x_i,y_i in zip(x,y))
def sum_of_squares(xs: List[float]) -> float:
    return (dot(xs,xs))
assert dot([1, 2, 3], [4, 5, 6]) == 32  # 1 * 4 + 2 * 5 + 3 * 6
assert sum_of_squares([1, 2, 3]) == 14
import math
def de_mean(xs: List[float]) -> List[float]:
    return [xs_i - mean(xs) for xs_i in xs]
#print(de_mean(num_friends))
def variance(xs: List[float]) -> float:
    assert len(xs)>=2
    n  = len(xs)
    deviations = de_mean(xs)
    return(sum_of_squares(deviations))/(n-1)
print(variance(num_friends))

def standard_deviation(xs:List[float]) -> float:
    return math.sqrt(variance(xs))

print (standard_deviation(num_friends))
    
def interquantile_range(xs:List[float])->float:
    return quantile(xs,0.75) - quantile(xs,0.25)
print(interquantile_range(num_friends))
def covariance(xs:List[float],ys:List[float]) -> float:
    assert len(xs) == len(ys)
    return dot(de_mean(xs),de_mean(ys))/(len(xs)-1)
def correlation(xs: List[float], ys: List[float]):
    stdev_xs = standard_deviation(xs)
    stdev_ys = standard_deviation(ys)
    if stdev_xs > 0 and stdev_ys > 0:
        return covariance(xs,ys)/stdev_xs/stdev_ys
    else:
        return 0
outlier = num_friends.index(100)
num_friends_good = [x for i, x in emurate(num_friends) if i!=outlier]
daily_minutes_good = [x for i,x in enumerate(num_friends) if i!=outlier]
assert 0.57 < correlation(num_friends_good, daily_minutes_good) < 0.58