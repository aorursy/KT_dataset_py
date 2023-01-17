pi = '3141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609173637178721468440901224953430146549585371050792279689258923542019956112129021960864034418159813629774771309960518707211349999998372978049951059731732816096318595024459455346908302642522308253344685035261931188171010003137838752886587533208381420617177669147303598253490428755468731159562863882353787593751957781857780532171226806613001927876611195909216420198'
dict = {'1': [1, 5, 8], '2': [3, 6, 10], '3': [5, 8, 12], '4': [6, 10, 13], '5': [8, 12, 15], '6': [10, 13, 17], '7': [12, 15, 18], '8': [1, 5, 8], '9': [3, 6, 10], '0': [5, 8, 12]}
up_by = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
chromatic = {'1': 262, '2': 277, '3': 294, '4': 311, '5': 330, '6': 349, '7': 370, '8': 392, '9': 415, '10': 440, '11': 466, '12': 494, '13': 523, '14': 554, '15': 587, '16': 622, '17': 659, '18': 698, '19': 740, '20': 784, '21': 831, '22': 880, '23': 932, '24': 988, '25': 1047, '26': 1109, '27': 1175, '28': 1245, '29': 1319, '30': 1397, '31': 1480, '32': 1568, '33': 1661, '34': 1760, '35': 1865, '36': 1976}
import numpy as np

import matplotlib.pyplot as plt

from IPython.display import (

    Audio, display, clear_output)

from ipywidgets import widgets

from functools import partial

import time

import random

%matplotlib inline
rate = 16000.
def synth(f, t):

    x = np.sin(f * 2. * np.pi * t)

    display(Audio(x, rate=rate, autoplay=True))
duration = 1

t = np.linspace(

    0., duration, int(rate * duration))
def run_pi(key):

    for digit in pi:

        duration = random.uniform(1, 3)

        t = np.linspace(0., duration, int(rate * duration))

        lst = dict[digit]

        synth(chromatic[str(lst[0]+ key)], t)

        synth(chromatic[str(lst[1] + key)], t)

        synth(chromatic[str(lst[2] + key)], t)

        time.sleep(duration)
def run_example(key):

    for i in range(1, 9):

        duration = 2

        t = np.linspace(0., duration, int(rate * duration))

        lst = dict[str(i)]

        synth(chromatic[str(lst[0] + key)], t)

        synth(chromatic[str(lst[1] + key)], t)

        synth(chromatic[str(lst[2] + key)], t)

        time.sleep(duration)
run_example(0)
in_key = str(input("Hear the chords in a different key! Enter C,C#,D,D#,E,F,F#,G,G#,A,A#,or B: "))

print(in_key, "Major")

run_example(up_by[in_key])
in_put = input("Time for the grand finale! Which key would you like to hear all the digits of pi in? ")

print(in_put, "Major")

run_pi(up_by[in_put])