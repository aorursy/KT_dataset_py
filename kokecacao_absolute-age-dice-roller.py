import numpy as np

import random



def f(atoms, dice_side):

    out = []



    while atoms != 0:

        for i in range(atoms):

            rad = random.randint(1, dice_side)

            if rad == 1:

                atoms-=1

        out.append(atoms)

    return out

print("D4")

print(f(atoms=50, dice_side=4))



print("D6")

print(f(atoms=50, dice_side=6))



print("D8")

print(f(atoms=50, dice_side=8))



print("D20")

print(f(atoms=50, dice_side=20))