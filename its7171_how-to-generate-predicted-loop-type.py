!yes '' | cpan -i Graph

!git clone https://github.com/hendrixlab/bpRNA
import pandas as pd

import numpy as np

import json
def get_predicted_loop_type(sequence, structure, debug=False):

    !echo $sequence > a.dbn

    !echo "$structure" >> a.dbn

    !export PERL5LIB=/root/perl5/lib/perl5 && perl bpRNA/bpRNA.pl a.dbn

    result = [l.strip('\n') for l in open('a.st')]

    if debug:

        print(sequence)

        print(structure)

        print(result[5])

    return result
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

sequence = train.sequence.values[0]

structure = train.structure.values[0]

predicted_loop_type = train.predicted_loop_type.values[0]



result = get_predicted_loop_type(sequence, structure, debug=True)

print(predicted_loop_type)
for i in range(20):

    result = get_predicted_loop_type(sequence, structure)

    if predicted_loop_type == result[5]:

        print(f'{i} ok')

        print(result[5])

        print(predicted_loop_type)

    elif predicted_loop_type.replace('X','M') == result[5].replace('X','M'):

        print(f'{i} ok except X and M difference')

        print(result[5])

        print(predicted_loop_type)

    else:

        print(f'{i} ng')

        print(result[5])

        print(predicted_loop_type)
for i, arr in enumerate(train[['sequence', 'structure', 'predicted_loop_type', 'id']].values):

    if i >= 100:

        break

    result = get_predicted_loop_type(arr[0], arr[1])

    

    if result[5] == arr[2]:

        print(f'{i} ok for {arr[3]}')

    elif result[5].replace('X','M') == arr[2].replace('X','M'):

        print(f'{i} ok for {arr[3]} except X and M difference')

    else:

        print(f'{i} predicted_loop_type is not same for {arr[3]}')

        print(result[5])

        print(arr[2])

        break