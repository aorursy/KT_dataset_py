# From previous walkthrough, create avg and std variables

nums = [22,21,12,49,13,63,59]

avg = 34.14

std = 20.46
## Add avg, std of nums to a dictionary

stats = dict(std=std,avg=avg)

# Alternative -> stats = {'std':std, 'avg': avg, 'median': 21}

print(stats)
## Add median and the actual numbers to the stats dictionary

stats['median'] = 22

stats['nums'] = nums

print(stats)



## Access the 4th index in the nums data structure within the dictionary structure (compound structure)

print(stats['nums'][4])



## Remove median from the stats dictionary

stats.pop('median')

print(stats)
## Print out the keys and values individually using a loop

for key, value in stats.items():

    print(key,value)



## Print out only the keys in the dictionary

for key in stats.keys():

    print(key)



## Print out only the values in the dictionary

for value in stats.values():

    print(value)