import pandas as pd

import numpy as np

import seaborn as sns

from pathlib import Path

from matplotlib import pyplot as plt

from ast import literal_eval

from IPython.display import display, Markdown
data_dir = Path("/kaggle/input/data-science-bowl-2019/")

sample_submission=pd.read_csv(data_dir / "sample_submission.csv")

specs=pd.read_csv(data_dir / "specs.csv")
sample_submission.head()
def print_specs(specs):

    old = pd.options.display.max_colwidth 

    pd.options.display.max_colwidth = 999

    try:

        for i, event_id, info, args in specs.itertuples():

            display(Markdown(f"""__{event_id}__





{info}"""))

            display(pd.DataFrame(data=literal_eval(args)))

    finally:

        pd.options.display.max_colwidth = old

print_specs(specs)