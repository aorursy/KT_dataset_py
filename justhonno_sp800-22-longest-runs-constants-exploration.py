from dataclasses import dataclass

from typing import Any



@dataclass

class Run:

    value: Any

    length: int = 1



def asruns(seq):

    firstval = seq[0]

    pointer = Run(firstval, length=0)

    for value in seq:

        if value == pointer.value:

            pointer.length += 1

        else:

            yield pointer

            pointer = Run(value)

    else:

        yield pointer
from itertools import product



maxlengths = []

for seq in product([0, 1], repeat=8):

    runs = asruns(seq)

    

    one_runs = (run for run in asruns(seq) if run.value == 1)



    maxlen = 0

    for run in one_runs:

        if run.length > maxlen:

            maxlen = run.length

                

    maxlengths.append(maxlen)
from collections import Counter



counts = Counter(maxlengths)

nsequences = sum(counts.values())



for maxlen, count in counts.items():

    prop = count / nsequences

    propf = round(prop, 2)

    print(f"maxlen={maxlen}, count={count}, prop={propf}")
import altair as alt

import pandas as pd



sortedcounts = sorted(counts.items(), key=lambda item: item[0])



data = pd.DataFrame(sortedcounts, columns=["maxlen", "occurences"])



alt.Chart(data).mark_bar().encode(

    x="maxlen",

    y="occurences"

)