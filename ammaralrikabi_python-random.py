import random
colors_list = ['red', 'green', 'yellow', 'blue', 'red', 'black']
random.shuffle(colors_list)
colors_list
# you can use sample or shuffle to cut the list

random.sample(colors_list, len(colors_list))
# return list 
random.choices(colors_list)
# return string
random.choice(colors_list)
colors_set = {'red', 'green', 'yellow', 'blue', 'red', 'black'}
# you can use sample or shuffle to cut the set

random.sample(colors_set, len(colors_set))
random.sample(colors_set, 2)
colors_tuple = ('red', 'green', 'yellow', 'blue', 'red', 'black')
random.sample(colors_tuple, len(colors_tuple))
random.choices(colors_tuple)
random.choice(colors_tuple)
colors_str = 'red green black yellow black blue'
# return sttring
random.choice(colors_str)
# return list
random.choices(colors_str)
random.choice(random.choice([range(10),range(10,20)]))
random.choice(list([1,2,3,4]))
sum(random.sample(range(9),9))