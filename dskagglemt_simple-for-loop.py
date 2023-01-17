import numpy as np
l = [x for x in range(0,150)]

l
c = 20

my_list = []



for i in range(0, len(l), 20):

    my_list.append(l[i:c])

    print("Range is from {} to {}".format(i, c))

    c += 20
my_list